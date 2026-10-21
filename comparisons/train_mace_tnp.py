"""
Train the Joint MACE-TNP model on AAAI interventional data.

Uses a custom PyTorch training loop (no CausalInferenceTrainer) to avoid
mismatch between per-node and joint training paradigms.

Default hyperparameters are tuned for parameter-count matching with CVAE
(d_model=88, num_mixture_components=20 → ~746k params, matching CVAE).

Usage:
    python train_mace_tnp.py \
        --data-dir data/mace_tnp/8var_linadd_beta \
        --save-path models/mace_tnp_8var_linadd_beta.pt \
        --epochs 500 --batch-size 64 --n-vars 8 --track

Prerequisites:
    conda activate mace_tnp
    # CITNP installed as editable package
    # HDF5 data created via convert_data_to_hdf5.py
"""

import argparse
import glob
import io
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torch.utils.data import DataLoader
from scipy.special import logsumexp

# CITNP imports
from CITNP.models.causalinferencemodel import CausalInfModelConfig
from CITNP.utils.datautils import MultipleFileDataset

from mace_tnp_model import JointMoGCausalInferenceModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(split_dir: str) -> MultipleFileDataset:
    """Load all HDF5 chunk files from a split directory."""
    hdf5_files = sorted(glob.glob(os.path.join(split_dir, '*.hdf5')))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {split_dir}")
    return MultipleFileDataset(hdf5_files)


def collate_fn(batch):
    """
    Materialise h5py slices (lazy arrays) and stack into tensors.

    MultipleFileDataset.__getitem__ returns 6-tuples:
        (obs_data, int_data, causal_graph, intvn_indices, variable_counts, masks)
    where masks may be None.

    Applies per-sample z-score normalization using obs_data statistics,
    matching the normalization done by CITNP's transformer_inference_split_withpadding.
    """
    obs_list, int_list, graph_list, intvn_list, count_list, _mask_list = zip(*batch)

    obs_np  = np.stack([np.array(x) for x in obs_list]).astype(np.float32)   # (B, n_obs, n_vars)
    int_np  = np.stack([np.array(x) for x in int_list]).astype(np.float32)   # (B, n_int, n_vars)

    # Z-score normalize using obs statistics (axis=1 = sample axis)
    mean = obs_np.mean(axis=1, keepdims=True)   # (B, 1, n_vars)
    std  = obs_np.std(axis=1, keepdims=True) + 1e-8
    obs_np = (obs_np - mean) / std
    int_np = (int_np - mean) / std

    graphs     = np.stack([np.array(x) for x in graph_list]).astype(np.float32)
    intvn_idx  = np.array([int(x) for x in intvn_list], dtype=np.int64)
    var_counts = np.array([int(x) for x in count_list], dtype=np.int64)

    obs_data   = torch.from_numpy(obs_np)
    int_data   = torch.from_numpy(int_np)
    graph_t    = torch.from_numpy(graphs)
    intvn_t    = torch.from_numpy(intvn_idx)
    count_t    = torch.from_numpy(var_counts)
    return obs_data, int_data, graph_t, intvn_t, count_t


def train_epoch(model, loader, optimizer, device, n_vars: int, clip_val: float = 1.0) -> float:
    """Run one training epoch; return mean NLL."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for obs_data, int_data, graphs, intvn_idx, var_counts in loader:
        obs_data  = obs_data.to(device, non_blocking=True)
        int_data  = int_data.to(device, non_blocking=True)
        intvn_idx = intvn_idx.to(device, non_blocking=True)

        # Use a single randomly-sampled target token as the encoder query,
        # matching the single-token path used in predict() at test time.
        # The NLL loss is still evaluated against all int_data samples.
        B, n_tgt, D = int_data.shape
        rand_idx = torch.randint(0, n_tgt, (B,), device=device)
        query_target = int_data[torch.arange(B, device=device), rand_idx].unsqueeze(1)  # (B, 1, D)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out  = model(obs_data, query_target, intvn_idx)
            loss = model.calculate_loss(out, int_data)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def val_epoch(model, loader, device, n_vars: int) -> float:
    """Run one validation epoch; return mean NLL."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for obs_data, int_data, graphs, intvn_idx, var_counts in loader:
            obs_data  = obs_data.to(device, non_blocking=True)
            int_data  = int_data.to(device, non_blocking=True)
            intvn_idx = intvn_idx.to(device, non_blocking=True)

            B, n_tgt, D = int_data.shape
            rand_idx = torch.randint(0, n_tgt, (B,), device=device)
            query_target = int_data[torch.arange(B, device=device), rand_idx].unsqueeze(1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out  = model(obs_data, query_target, intvn_idx)
                loss = model.calculate_loss(out, int_data)

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    def rbf(A, B):
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-sq / (2 * sigma ** 2))
    Kxx, Kyy, Kxy = rbf(X, X), rbf(Y, Y), rbf(X, Y)
    n, m = len(X), len(Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (n*(n-1))
                 + (Kyy.sum() - np.trace(Kyy)) / (m*(m-1))
                 - 2 * Kxy.mean())


def compute_wasserstein(X: np.ndarray, Y: np.ndarray,
                        epsilon: float = 1e-2, num_iters: int = 100) -> float:
    """Sinkhorn-regularized joint Wasserstein distance (mirrors utils.py)."""
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    X_norm = np.sum(X ** 2, axis=1, keepdims=True)   # (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T  # (1, m)
    M = np.sqrt(np.maximum(X_norm + Y_norm - 2 * X @ Y.T, 0) + 1e-8)

    K = -M / epsilon
    K -= K.max()

    log_u = np.zeros(n)
    log_v = np.zeros(m)
    for _ in range(num_iters):
        log_u = np.log(a) - logsumexp(K + log_v[None, :], axis=1)
        log_v = np.log(b) - logsumexp(K.T + log_u[None, :], axis=1)

    transport_plan = np.exp(K + log_u[:, None] + log_v[None, :])
    return float(np.sum(transport_plan * M))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    def mean_dist(A, B):
        diff = A[:, None] - B[None, :]
        return np.mean(np.sqrt(np.sum(diff**2, axis=-1)))
    return float(2*mean_dist(X, Y) - mean_dist(X, X) - mean_dist(Y, Y))


def collate_fn_raw(batch):
    """Like collate_fn but returns un-normalized data (for metric computation)."""
    obs_list, int_list, graph_list, intvn_list, count_list, _mask_list = zip(*batch)
    obs_np   = np.stack([np.array(x) for x in obs_list]).astype(np.float32)
    int_np   = np.stack([np.array(x) for x in int_list]).astype(np.float32)
    intvn_np = np.array([int(x) for x in intvn_list], dtype=np.int64)
    return obs_np, int_np, intvn_np


@torch.no_grad()
def compute_val_metrics(
    model: JointMoGCausalInferenceModel,
    val_dataset,
    device: torch.device,
    intervention_value: float = 5.0,
    batch_size: int = 32,
    n_samples: int = 50,
    max_batches: int = 10,
) -> dict:
    """
    Compute distribution metrics on a subset of the validation set.

    Uses un-normalized data so that model.predict() (which normalizes internally)
    and the true samples are compared in the original data space.
    Returns mean MMD, Wasserstein, and energy distance.
    """
    model.eval()
    raw_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_raw, num_workers=0)

    mmd_vals, wsd_vals, energy_vals = [], [], []

    for batch_idx, (obs_np, int_np, intvn_np) in enumerate(raw_loader):
        if batch_idx >= max_batches:
            break

        B = obs_np.shape[0]
        obs_tensor   = torch.tensor(obs_np,  dtype=torch.float32, device=device)
        intvn_tensor = torch.tensor(intvn_np, dtype=torch.long,   device=device)
        intvn_vals   = torch.full((B,), intervention_value,
                                  dtype=torch.float32, device=device)

        # predict() handles z-score normalization internally; returns (B, n_samples, n_vars)
        pred = model.predict(obs_tensor, intvn_tensor,
                             intervention_value=intvn_vals,
                             n_samples=n_samples).cpu().numpy()

        for i in range(B):
            true_i = int_np[i]   # (n_int, n_vars)
            pred_i = pred[i]     # (n_samples, n_vars)
            mmd_vals.append(compute_mmd(true_i, pred_i))
            wsd_vals.append(compute_wasserstein(true_i, pred_i))
            energy_vals.append(energy_distance(true_i, pred_i))

    return {
        'val/mmd':    float(np.mean(mmd_vals)),
        'val/wsd':    float(np.mean(wsd_vals)),
        'val/energy': float(np.mean(energy_vals)),
    }


@torch.no_grad()
def visualize_distributions(
    model,
    val_dataset,
    device,
    n_vars: int,
    epoch: int,
    intervention_value: float = 5.0,
    batch_size: int = 32,
    n_examples: int = 8,
    n_pred_samples: int = 500,
    save_path: str | None = None,
):
    """
    Sample from the predicted GMM (test-time: predict() path) and plot vs real
    interventional samples, projected onto (var0, var1).

    Uses raw un-normalized data so that predict() handles normalization internally,
    matching the setup used in compute_val_metrics().

    Returns a wandb.Image (if wandb is importable) or None.
    Caller is responsible for including it in wandb.log().
    """
    model.eval()

    raw_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn_raw, num_workers=0)
    obs_np, int_np, intvn_np = next(iter(raw_loader))

    n_examples = min(n_examples, obs_np.shape[0])
    obs_np    = obs_np[:n_examples]
    int_np    = int_np[:n_examples]
    intvn_np  = intvn_np[:n_examples]

    obs_tensor   = torch.tensor(obs_np,  dtype=torch.float32, device=device)
    intvn_tensor = torch.tensor(intvn_np, dtype=torch.long,   device=device)
    intvn_vals   = torch.full((n_examples,), intervention_value,
                              dtype=torch.float32, device=device)

    # predict() uses test-time inference (dummy single target), matching metric evaluation
    pred_samples = model.predict(obs_tensor, intvn_tensor,
                                 intervention_value=intvn_vals,
                                 n_samples=n_pred_samples).cpu().numpy()  # (n_examples, n_pred_samples, D)
    real_samples = int_np  # (n_examples, n_tgt, D) in original data space

    ncols = n_examples
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6))
    if ncols == 1:
        axes = axes[:, None]

    for i in range(n_examples):
        j = int(intvn_np[i])

        real_x, real_y = real_samples[i, :, 0], real_samples[i, :, 1]
        pred_x, pred_y = pred_samples[i, :, 0], pred_samples[i, :, 1]

        xlim = (min(real_x.min(), pred_x.min()) - 0.5,
                max(real_x.max(), pred_x.max()) + 0.5)
        ylim = (min(real_y.min(), pred_y.min()) - 0.5,
                max(real_y.max(), pred_y.max()) + 0.5)

        axes[0, i].scatter(real_x, real_y, s=4, alpha=0.4, color='steelblue', rasterized=True)
        axes[0, i].set(xlim=xlim, ylim=ylim, title=f'Input {i+1}: Real do(X{j})',
                       xlabel='var 0', ylabel='var 1')

        axes[1, i].scatter(pred_x, pred_y, s=4, alpha=0.4, color='darkorange', rasterized=True)
        axes[1, i].set(xlim=xlim, ylim=ylim, title=f'Input {i+1}: Pred do(X{j})',
                       xlabel='var 0', ylabel='var 1')

    fig.suptitle(f'Epoch {epoch}: Real vs Predicted (var0 vs var1, original space, do(X_j={intervention_value}))',
                 fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')

    wandb_image = None
    try:
        import wandb
        from PIL import Image as PILImage
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        pil_img = PILImage.open(buf).copy()
        buf.close()
        wandb_image = wandb.Image(pil_img, caption=f'Epoch {epoch}')
    except ImportError:
        pass

    plt.close(fig)
    return wandb_image


def save_checkpoint(model, optimizer, epoch: int, val_loss: float, path: str, config_dict: dict):
    """Save model checkpoint with config for later reconstruction."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss':             val_loss,
        'config':               config_dict,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train Joint MACE-TNP model')
    parser.add_argument('--data-dir',    required=True, help='Root data dir with train/val/test sub-dirs')
    parser.add_argument('--save-path',   required=True, help='Path to save best model checkpoint (.pt)')
    parser.add_argument('--epochs',      type=int,   default=1000)
    parser.add_argument('--batch-size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--d-model',     type=int,   default=88)    # matched to CVAE with K=20; dim_feedforward = 352
    parser.add_argument('--num-heads',   type=int,   default=4)
    parser.add_argument('--num-encoder-layers', type=int, default=4)   # matches CVAE encoder_attention_layers
    parser.add_argument('--num-mixture-components', type=int, default=20)  # matched to CVAE prior_mixture_components
    parser.add_argument('--n-vars',      type=int,   default=8)
    parser.add_argument('--intervention-value', type=float, default=5.0,
                        help='Fixed value for the intervened variable when computing val metrics.')
    parser.add_argument('--track',       action='store_true', help='Enable wandb logging')
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # wandb setup
    if args.track:
        import wandb
        run_name = os.path.splitext(os.path.basename(args.save_path))[0]
        wandb.init(project='mace-tnp', name=run_name, config=vars(args))

    # Build datasets and loaders
    print("Loading datasets...")
    train_path = os.path.join(args.data_dir, 'train')
    val_path   = os.path.join(args.data_dir, 'val')
    print(f"  Train dir: {train_path}")
    print(f"  Val dir:   {val_path}")
    train_ds = build_dataset(train_path)
    val_ds   = build_dataset(val_path)
    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # MultipleFileDataset with num_workers > 0 causes index out of range errors.
    # Use single-process loading instead.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Build model config
    # Map argparse names → CausalInfModelConfig fields
    config = CausalInfModelConfig(
        d_model=args.d_model,
        emb_depth=2,
        dim_feedforward=args.d_model * 4,
        nhead=args.num_heads,
        dropout=0.0,
        num_layers_encoder=args.num_encoder_layers,
        num_nodes=args.n_vars,
        num_mixture_components=args.num_mixture_components,
        sample_attn_mode='MHCA',
        linear_attention=False,
        mean_loss_across_samples=True,
        device=str(device),
        dtype='float32',
    )

    # Save config dict for checkpoint reconstruction
    config_dict = {
        'd_model':                args.d_model,
        'num_heads':              args.num_heads,
        'num_encoder_layers':     args.num_encoder_layers,
        'num_mixture_components': args.num_mixture_components,
        'n_vars':                 args.n_vars,
    }

    model = JointMoGCausalInferenceModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    if device.type == 'cuda':
        print("Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_path = args.save_path.replace('.pt', '_best.pt')

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.n_vars)
        val_loss   = val_epoch(model, val_loader, device, args.n_vars)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train_nll={train_loss:.4f}  val_nll={val_loss:.4f}")

        log_dict = {'train/nll': train_loss, 'val/nll': val_loss,
                    'lr': scheduler.get_last_lr()[0]}

        if epoch % 50 == 0 or epoch == 1:
            dist_metrics = compute_val_metrics(model, val_ds, device,
                                               intervention_value=args.intervention_value,
                                               batch_size=args.batch_size)
            log_dict.update(dist_metrics)
            print(f"            mmd={dist_metrics['val/mmd']:.4f}  "
                  f"wsd={dist_metrics['val/wsd']:.4f}  "
                  f"energy={dist_metrics['val/energy']:.4f}")

            # Distribution visualisation — generate image and add to the same log_dict
            vis_png = args.save_path.replace('.pt', '_dist.png')
            wandb_image = visualize_distributions(
                model, val_ds, device, args.n_vars, epoch,
                intervention_value=args.intervention_value,
                batch_size=args.batch_size,
                n_examples=min(8, args.batch_size),
                save_path=vis_png,
            )
            if wandb_image is not None:
                log_dict['val/distribution_2d'] = wandb_image

        if args.track:
            import wandb
            wandb.log(log_dict, step=epoch)

        # Save best and latest checkpoints
        latest_path = args.save_path.replace('.pt', '_latest.pt')
        save_checkpoint(model, optimizer, epoch, val_loss, latest_path, config_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, config_dict)

    # Save final checkpoint
    save_checkpoint(model, optimizer, args.epochs, val_loss, args.save_path, config_dict)

    print(f"\nTraining complete.")
    print(f"  Best val NLL: {best_val_loss:.4f} → {best_path}")
    print(f"  Final checkpoint: {args.save_path}")

    if args.track:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
