# DEPRECATED: All functionality has been merged into evaluate_causal.py.
# Use evaluate_causal.py instead.
"""
Evaluate a trained Joint MACE-TNP model using the same metrics as evaluate_model.py.

Reports MMD, Wasserstein distance, energy distance, p-value (distribution metrics)
and ATE MAE/RMSE (causal quantity metrics), matching the CVAE evaluation exactly.

Usage:
    python evaluate_mace_tnp.py \
        --data-path data/AAAI/8var_linadd_beta_test.npz \
        --model-path models/mace_tnp_8var_linadd_beta_best.pt

Prerequisites:
    conda activate mace_tnp   # torch + CITNP; no JAX needed
"""

import argparse
import os
import numpy as np
import torch
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily
from scipy.stats import wasserstein_distance as scipy_wasserstein
from scipy.stats import multivariate_normal as scipy_mvn
from scipy.special import logsumexp as scipy_logsumexp

from CITNP.models.causalinferencemodel import CausalInfModelConfig
from mace_tnp_model import JointMoGCausalInferenceModel


# ---------------------------------------------------------------------------
# Pure NumPy/SciPy metric implementations (no JAX)
# ---------------------------------------------------------------------------

def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """MMD with RBF kernel between two sample sets (n_samples, n_vars)."""
    def rbf_kernel(A, B, s):
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-sq / (2 * s ** 2))

    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    n, m = len(X), len(Y)
    mmd = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
        - 2 * Kxy.mean()
    return float(mmd)


def compute_wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean per-variable 1-D Wasserstein distance."""
    n_vars = X.shape[1]
    return float(np.mean([scipy_wasserstein(X[:, d], Y[:, d]) for d in range(n_vars)]))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two sample sets."""
    def mean_dist(A, B):
        diff = A[:, None] - B[None, :]  # (n, m, d)
        return np.mean(np.sqrt(np.sum(diff ** 2, axis=-1)))

    return float(2 * mean_dist(X, Y) - mean_dist(X, X) - mean_dist(Y, Y))


def energy_test(X: np.ndarray, Y: np.ndarray, n_permutations: int = 199) -> tuple:
    """Permutation test for energy statistic. Returns (stat, p-value)."""
    stat = energy_distance(X, Y)
    combined = np.concatenate([X, Y], axis=0)
    n = len(X)
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(combined))
        Xp = combined[perm[:n]]
        Yp = combined[perm[n:]]
        if energy_distance(Xp, Yp) >= stat:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return stat, pval


# ---------------------------------------------------------------------------
# Causal helpers (inlined from evaluate_causal.py, no JAX)
# ---------------------------------------------------------------------------

def get_descendants(adj_matrix: np.ndarray, node: int) -> set:
    """BFS descendants of `node` in adjacency matrix (edge i→j means adj[i,j]=1)."""
    n = adj_matrix.shape[0]
    descendants = set()
    queue = [node]
    visited = {node}
    while queue:
        current = queue.pop(0)
        if current != node:
            descendants.add(current)
        for child in range(n):
            if adj_matrix[current, child] == 1 and child not in visited:
                visited.add(child)
                queue.append(child)
    return descendants


def compute_ate_metrics(ate_gt, ate_pred, desc_mask, valid_mask):
    """MAE/RMSE metrics on all/descendant/non-descendant subsets."""
    error    = np.abs(ate_pred - ate_gt)
    error_sq = (ate_pred - ate_gt) ** 2

    metrics = {}
    metrics['MAE_all']  = float(np.mean(error[valid_mask]))
    metrics['RMSE_all'] = float(np.sqrt(np.mean(error_sq[valid_mask])))

    desc_and_valid = desc_mask & valid_mask
    if desc_and_valid.sum() > 0:
        metrics['MAE_descendants']  = float(np.mean(error[desc_and_valid]))
        metrics['RMSE_descendants'] = float(np.sqrt(np.mean(error_sq[desc_and_valid])))
    else:
        metrics['MAE_descendants']  = float('nan')
        metrics['RMSE_descendants'] = float('nan')

    non_desc_and_valid = (~desc_mask) & valid_mask
    if non_desc_and_valid.sum() > 0:
        # Non-descendant ATE should be zero; measure predicted magnitude
        metrics['MAE_non_descendants']    = float(np.mean(np.abs(ate_pred[non_desc_and_valid])))
        metrics['MAE_non_desc_noise_floor'] = float(np.mean(np.abs(ate_gt[non_desc_and_valid])))
    else:
        metrics['MAE_non_descendants']    = float('nan')
        metrics['MAE_non_desc_noise_floor'] = float('nan')

    return metrics


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device) -> JointMoGCausalInferenceModel:
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint['config']
    d_model = cfg['d_model']
    config = CausalInfModelConfig(
        d_model=d_model,
        emb_depth=2,
        dim_feedforward=d_model * 4,
        nhead=cfg['num_heads'],
        dropout=0.0,
        num_layers_encoder=cfg['num_encoder_layers'],
        num_nodes=cfg['n_vars'],
        num_mixture_components=cfg['num_mixture_components'],
        sample_attn_mode='MHCA',
        linear_attention=False,
        mean_loss_across_samples=True,
        device=str(device),
        dtype='float32',
    )
    model = JointMoGCausalInferenceModel(config)
    state_dict = checkpoint['model_state_dict']
    # Strip _orig_mod. prefix added by torch.compile
    state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def get_samples_for_example(
    model: JointMoGCausalInferenceModel,
    obs_data: np.ndarray,
    intvn_index: int,
    intvn_value: float,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    """Draw samples for a single (SCM, intervention) pair. Returns (n_samples, n_vars)."""
    obs_tensor   = torch.tensor(obs_data, dtype=torch.float32, device=device).unsqueeze(0)
    intvn_tensor = torch.tensor([intvn_index], dtype=torch.long, device=device)
    val_tensor   = torch.tensor([intvn_value], dtype=torch.float32, device=device)
    samples = model.predict(obs_tensor, intvn_tensor, intervention_value=val_tensor,
                            n_samples=n_samples)
    return samples.squeeze(0).cpu().numpy()


def _marg_gmm_nll_np(weights, means, covs, true_samples, dims):
    """
    NLL of true_samples under the marginal of a GMM restricted to `dims`.
    weights: (K,), means: (K, D), covs: (K, D, D) — all numpy.
    true_samples: (n_samples, D) — numpy.
    dims: list of int indices.
    """
    w = np.asarray(weights)
    m = np.asarray(means)[:, dims]
    c = np.asarray(covs)[:, dims, :][:, :, dims]  # (K, |dims|, |dims|)
    x = np.asarray(true_samples)[:, dims]
    K = len(w)
    lp = np.full((len(x), K), -np.inf)
    for ki in range(K):
        if w[ki] > 1e-15:
            try:
                lp[:, ki] = scipy_mvn.logpdf(x, m[ki], c[ki])
            except Exception:
                pass
    log_mix = scipy_logsumexp(lp + np.log(np.maximum(w, 1e-15))[None], axis=1)
    return float(-np.mean(log_mix))


@torch.no_grad()
def get_log_prob_for_example(
    model: JointMoGCausalInferenceModel,
    obs_data: np.ndarray,
    true_target: np.ndarray,
    intvn_index: int,
    intvn_value: float,
    device: torch.device,
) -> float:
    """
    Mean NLL of true_target under the predicted GMM, in original data space.

    The model predicts in z-score normalized space, so a Jacobian correction
    log p_orig(x) = log p_norm((x-mu)/sigma) - sum(log(sigma)) is applied
    to bring NLL back to the original scale for fair comparison with CVAE.
    """
    obs_tensor = torch.tensor(obs_data,    dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_obs, D)
    tgt_tensor = torch.tensor(true_target, dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_tgt, D)
    intvn_tensor = torch.tensor([intvn_index], dtype=torch.long,    device=device)
    val_tensor   = torch.tensor([intvn_value], dtype=torch.float32, device=device)

    # Normalize context (identical to predict())
    mean = obs_tensor.mean(dim=1, keepdim=True)   # (1, 1, D)
    std  = obs_tensor.std(dim=1,  keepdim=True) + 1e-8

    context_norm = (obs_tensor - mean) / std
    tgt_norm     = (tgt_tensor - mean) / std       # (1, n_tgt, D)

    # Build dummy target token (identical to predict())
    D = obs_data.shape[-1]
    dummy_target = torch.zeros(1, 1, D, device=device, dtype=obs_tensor.dtype)
    var_mean = mean[0, 0, intvn_index]
    var_std  = std[0,  0, intvn_index]
    dummy_target[0, 0, intvn_index] = (val_tensor[0] - var_mean) / var_std

    outcome_index = torch.where(
        intvn_tensor == 0,
        torch.ones(1, dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    )

    out = model.forward(context_norm, dummy_target, intvn_tensor, outcome_index)

    mix = Categorical(probs=out.weights)
    comp = MultivariateNormal(loc=out.means, scale_tril=out.L)
    gmm  = MixtureSameFamily(mix, comp)

    # log_prob in normalized space: (1, n_tgt)
    log_prob_norm = gmm.log_prob(tgt_norm)

    # Jacobian correction: log p_orig = log p_norm - sum(log sigma)
    log_jacobian  = torch.log(std).sum(dim=-1)     # (1, 1)
    log_prob_orig = log_prob_norm - log_jacobian   # (1, n_tgt)

    return float(-log_prob_orig.mean().cpu())


@torch.no_grad()
def get_dist_info_for_example(
    model: JointMoGCausalInferenceModel,
    obs_data: np.ndarray,
    true_target: np.ndarray,
    intvn_index: int,
    intvn_value: float,
    device: torch.device,
) -> tuple:
    """
    Single forward pass returning (nll, weights, means_orig, covs_orig).

    Runs the same forward pass as get_log_prob_for_example but additionally
    returns the GMM parameters transformed back to original (unnormalized) data
    space so that marginals over arbitrary variable subsets can be computed.

    Returns:
        nll          : float, negative log-likelihood in original space
        weights_np   : (K,) numpy — mixture weights
        means_orig   : (K, D) numpy — component means in original space
        covs_orig    : (K, D, D) numpy — full component covariance matrices
    """
    obs_tensor   = torch.tensor(obs_data,    dtype=torch.float32, device=device).unsqueeze(0)
    tgt_tensor   = torch.tensor(true_target, dtype=torch.float32, device=device).unsqueeze(0)
    intvn_tensor = torch.tensor([intvn_index], dtype=torch.long,    device=device)
    val_tensor   = torch.tensor([intvn_value], dtype=torch.float32, device=device)

    mean = obs_tensor.mean(dim=1, keepdim=True)
    std  = obs_tensor.std(dim=1,  keepdim=True) + 1e-8

    context_norm = (obs_tensor - mean) / std
    tgt_norm     = (tgt_tensor - mean) / std

    D = obs_data.shape[-1]
    dummy_target = torch.zeros(1, 1, D, device=device, dtype=obs_tensor.dtype)
    dummy_target[0, 0, intvn_index] = (val_tensor[0] - mean[0, 0, intvn_index]) / std[0, 0, intvn_index]

    outcome_index = torch.where(
        intvn_tensor == 0,
        torch.ones(1,  dtype=torch.long, device=device),
        torch.zeros(1, dtype=torch.long, device=device),
    )

    out = model.forward(context_norm, dummy_target, intvn_tensor, outcome_index)

    mix  = Categorical(probs=out.weights)
    comp = MultivariateNormal(loc=out.means, scale_tril=out.L)
    gmm  = MixtureSameFamily(mix, comp)

    log_prob_norm = gmm.log_prob(tgt_norm)
    log_jacobian  = torch.log(std).sum(dim=-1)
    log_prob_orig = log_prob_norm - log_jacobian
    nll = float(-log_prob_orig.mean().cpu())

    # Transform GMM parameters to original data space.
    # If x_norm ~ N(m_norm, L @ L^T) then x_orig = x_norm * std + mean,
    # so x_orig ~ N(m_norm * std + mean, diag(std) @ L @ L^T @ diag(std)).
    # Cholesky of the original covariance: L_orig = diag(std) @ L.
    std_np    = std[0, 0].cpu().numpy()    # (D,)
    mean_np   = mean[0, 0].cpu().numpy()   # (D,)
    w_np      = out.weights[0].cpu().numpy()    # (K,)
    m_norm_np = out.means[0].cpu().numpy()      # (K, D)
    L_norm_np = out.L[0].cpu().numpy()          # (K, D, D)

    means_orig = m_norm_np * std_np[None] + mean_np[None]  # (K, D)
    L_orig     = L_norm_np * std_np[:, None]               # (K, D, D): scale rows by std
    covs_orig  = L_orig @ L_orig.transpose(0, 2, 1)        # (K, D, D)

    return nll, w_np, means_orig, covs_orig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate Joint MACE-TNP model')
    parser.add_argument('--data-path',    required=True,  help='Path to test NPZ file')
    parser.add_argument('--model-path',   required=True,  help='Path to trained .pt checkpoint')
    parser.add_argument('--n-samples',    type=int, default=100,
                        help='Number of model samples per (SCM, intervention)')
    parser.add_argument('--n-eval-scms',  type=int, default=1000,
                        help='Number of SCMs to evaluate (capped at available)')
    parser.add_argument('--intervention-value', type=float, default=5.0,
                        help='Fixed value to use for the intervened variable at query time. '
                             'Default 5.0. Using the true data mean would leak test information.')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------
    # Load data directly from NPZ
    # -------------------------------------------------------------------
    print(f"\nLoading data from {args.data_path}...")
    with np.load(args.data_path) as f:
        input_data  = f['input'].astype(np.float32)       # (n_scms, n_obs, n_vars)
        target_data = f['target_data'].astype(np.float32) # (n_intvn, n_scms, n_tgts, n_int, n_vars)
        graphs      = f['graphs'].astype(np.float32)      # (n_scms, n_vars, n_vars)
        targets     = f['targets'].astype(np.float32)     # (n_scms, n_intvn, n_tgts, n_vars)

    n_scms, n_obs, n_vars = input_data.shape
    n_interventions = target_data.shape[0]
    n_targets       = target_data.shape[2]
    n_int_samples   = target_data.shape[3]
    print(f"Data: {n_scms} SCMs, {n_obs} obs, {n_vars} vars, "
          f"{n_interventions} interventions, {n_targets} targets")

    # Subsample SCMs for speed
    n_eval = min(args.n_eval_scms, n_scms)
    eval_idx     = np.random.choice(n_scms, n_eval, replace=False)
    input_eval   = input_data[eval_idx]    # (n_eval, n_obs, n_vars)
    graphs_eval  = graphs[eval_idx]
    targets_eval = targets[eval_idx]       # (n_eval, n_intvn, n_tgts, n_vars)

    # target_data_eval: (n_eval, n_intvn, n_tgts, n_int, n_vars)
    target_data_eval = target_data[:, eval_idx].transpose(1, 0, 2, 3, 4)

    print(f"Evaluating on {n_eval} SCMs")

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # -------------------------------------------------------------------
    # Generate model samples
    # -------------------------------------------------------------------
    print(f"\nGenerating model samples (intervention_value={args.intervention_value})...")
    # predicted_samples[(i, j)][k]:    (n_int_samples, n_vars)
    # predicted_nll[(i, j)][k]:        scalar NLL in original space
    # predicted_gmm_params[(i, j)][k]: (weights, means_orig, covs_orig) for marginals
    predicted_samples   = {}
    predicted_nll       = {}
    predicted_gmm_params = {}
    for j in range(n_targets):
        for i in range(n_interventions):
            per_pair        = []
            per_pair_nll    = []
            per_pair_params = []
            for k in range(n_eval):
                intvn_idx = int(np.argmax(targets_eval[k, i, j]))
                samp = get_samples_for_example(
                    model, input_eval[k], intvn_idx, args.intervention_value,
                    n_int_samples, device)
                per_pair.append(samp)
                nll, w, m, c = get_dist_info_for_example(
                    model, input_eval[k],
                    target_data_eval[k, i, j],
                    intvn_idx, args.intervention_value, device)
                per_pair_nll.append(nll)
                per_pair_params.append((w, m, c))
            predicted_samples[(i, j)]    = per_pair
            predicted_nll[(i, j)]        = per_pair_nll
            predicted_gmm_params[(i, j)] = per_pair_params

    # Flatten to (N, n_int_samples, n_vars) for distribution metrics
    true_flat = []
    pred_flat = []
    for j in range(n_targets):
        for i in range(n_interventions):
            for k in range(n_eval):
                true_flat.append(target_data_eval[k, i, j])       # (n_int_samples, n_vars)
                pred_flat.append(predicted_samples[(i, j)][k])

    true_flat = np.stack(true_flat, axis=0)  # (N, n_int_samples, n_vars)
    pred_flat = np.stack(pred_flat, axis=0)

    N = true_flat.shape[0]
    print(f"Evaluation pairs: {N}")

    # -------------------------------------------------------------------
    # Distribution metrics
    # -------------------------------------------------------------------
    print("\nComputing distribution metrics...")
    mmd_vals      = [compute_mmd(true_flat[i], pred_flat[i])               for i in range(N)]
    wsd_vals      = [compute_wasserstein_distance(true_flat[i], pred_flat[i]) for i in range(N)]
    energy_vals   = [energy_distance(true_flat[i], pred_flat[i])           for i in range(N)]
    pval_vals     = [energy_test(true_flat[i], pred_flat[i])[1]            for i in range(N)]

    mmd_mean    = float(np.mean(mmd_vals))
    wsd_mean    = float(np.mean(wsd_vals))
    energy_mean = float(np.mean(energy_vals))
    p_mean      = float(np.mean(pval_vals))
    nll_mean    = float(np.mean([nll for nlls in predicted_nll.values() for nll in nlls]))

    print(f"  int_mmd:    {mmd_mean:.6f}")
    print(f"  int_wsd:    {wsd_mean:.6f}")
    print(f"  int_energy: {energy_mean:.6f}")
    print(f"  int_p:      {p_mean:.6f}")
    print(f"  int_nll:    {nll_mean:.6f}")

    # -------------------------------------------------------------------
    # Variance diagnostics (detect distribution collapse)
    # -------------------------------------------------------------------
    # For each evaluation pair, compute per-variable std in true vs predicted.
    # std_ratio < 1 → model underdispersed (variance collapse).
    # mean_err small with high energy → confirmed variance collapse.
    true_std  = true_flat.std(axis=1)   # (N, n_vars)
    pred_std  = pred_flat.std(axis=1)   # (N, n_vars)
    true_mean = true_flat.mean(axis=1)  # (N, n_vars)
    pred_mean = pred_flat.mean(axis=1)  # (N, n_vars)

    std_ratio  = float(np.mean(pred_std / (true_std + 1e-8)))
    mean_mae   = float(np.mean(np.abs(pred_mean - true_mean)))
    std_mae    = float(np.mean(np.abs(pred_std  - true_std)))

    print(f"\nVariance diagnostics (collapse detection):")
    print(f"  mean std_ratio (pred/true): {std_ratio:.4f}  "
          f"(<1 = underdispersed, >1 = overdispersed)")
    print(f"  mean MAE of means:          {mean_mae:.4f}")
    print(f"  mean MAE of std devs:       {std_mae:.4f}")

    # Energy decomposition: 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    # If E[||Y-Y'||] ≈ 0, collapse is driving the energy.
    def _mean_pw_dist(A):
        diff = A[:, None] - A[None, :]
        return np.mean(np.sqrt(np.sum(diff**2, axis=-1) + 1e-12))

    cross_terms = [2 * np.mean(np.sqrt(np.sum((true_flat[n][:, None] - pred_flat[n][None, :])**2, axis=-1)))
                   for n in range(N)]
    self_true   = [_mean_pw_dist(true_flat[n]) for n in range(N)]
    self_pred   = [_mean_pw_dist(pred_flat[n]) for n in range(N)]

    print(f"\nEnergy decomposition (mean over pairs):")
    print(f"  2·E[||X-Y||]:   {np.mean(cross_terms):.4f}")
    print(f"    E[||X-X'||]:  {np.mean(self_true):.4f}  (true self-distance)")
    print(f"    E[||Y-Y'||]:  {np.mean(self_pred):.4f}  (pred self-distance; ~0 = collapsed)")

    # -------------------------------------------------------------------
    # Marginal metrics: descendants and non-descendants
    # -------------------------------------------------------------------
    print("\nComputing marginal metrics for descendants / non-descendants...")
    nll_desc_list, nll_nondesc_list = [], []
    energy_desc_list, energy_nondesc_list = [], []

    for j in range(n_targets):
        for i in range(n_interventions):
            for k in range(n_eval):
                intvn_idx = int(np.argmax(targets_eval[k, i, j]))
                adj = graphs_eval[k]
                desc     = get_descendants(adj, intvn_idx)
                non_desc = set(range(n_vars)) - desc - {intvn_idx}
                desc_dims     = sorted(desc)
                non_desc_dims = sorted(non_desc)

                true_k = target_data_eval[k, i, j]          # (n_int_samples, n_vars)
                pred_k = predicted_samples[(i, j)][k]        # (n_int_samples, n_vars)
                w_k, m_k, c_k = predicted_gmm_params[(i, j)][k]

                if desc_dims:
                    energy_desc_list.append(
                        energy_distance(true_k[:, desc_dims], pred_k[:, desc_dims]))
                    nll_desc_list.append(
                        _marg_gmm_nll_np(w_k, m_k, c_k, true_k, desc_dims))
                if non_desc_dims:
                    energy_nondesc_list.append(
                        energy_distance(true_k[:, non_desc_dims], pred_k[:, non_desc_dims]))
                    nll_nondesc_list.append(
                        _marg_gmm_nll_np(w_k, m_k, c_k, true_k, non_desc_dims))

    desc_energy_mean    = float(np.mean(energy_desc_list))    if energy_desc_list    else float('nan')
    desc_nll_mean       = float(np.mean(nll_desc_list))       if nll_desc_list       else float('nan')
    nondesc_energy_mean = float(np.mean(energy_nondesc_list)) if energy_nondesc_list else float('nan')
    nondesc_nll_mean    = float(np.mean(nll_nondesc_list))    if nll_nondesc_list    else float('nan')

    # -------------------------------------------------------------------
    # ATE metrics
    # -------------------------------------------------------------------
    print("\nComputing ATE metrics...")
    obs_mean = input_eval.mean(axis=1)  # (n_eval, n_vars)

    ate_gt   = np.zeros((n_eval, n_interventions, n_vars))
    ate_pred = np.zeros((n_eval, n_interventions, n_vars))
    for k in range(n_eval):
        for i in range(n_interventions):
            ate_gt[k, i]   = target_data_eval[k, i, 0].mean(axis=0) - obs_mean[k]
            ate_pred[k, i] = predicted_samples[(i, 0)][k].mean(axis=0) - obs_mean[k]

    desc_mask  = np.zeros((n_eval, n_interventions, n_vars), dtype=bool)
    valid_mask = np.ones((n_eval, n_interventions, n_vars), dtype=bool)
    for k in range(n_eval):
        adj = graphs_eval[k]
        for i in range(n_interventions):
            intervened_vars = np.flatnonzero(targets_eval[k, i, 0] > 0.5)
            for v in intervened_vars:
                for y in get_descendants(adj, v):
                    desc_mask[k, i, y] = True
                valid_mask[k, i, v] = False

    metrics = compute_ate_metrics(ate_gt, ate_pred, desc_mask, valid_mask)

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MACE-TNP Evaluation Results")
    print("=" * 70)
    print(f"  int_mmd:    {mmd_mean:.6f}")
    print(f"  int_wsd:    {wsd_mean:.6f}")
    print(f"  int_energy: {energy_mean:.6f}")
    print(f"  int_p:      {p_mean:.6f}")
    print(f"  int_nll:    {nll_mean:.6f}")
    print()
    print("Marginal metrics (descendants):")
    print(f"  desc_energy: {desc_energy_mean:.6f}")
    print(f"  desc_nll:    {desc_nll_mean:.6f}")
    print("Marginal metrics (non-descendants):")
    print(f"  nondesc_energy: {nondesc_energy_mean:.6f}")
    print(f"  nondesc_nll:    {nondesc_nll_mean:.6f}")
    print()
    print("Variance diagnostics:")
    print(f"  std_ratio (pred/true): {std_ratio:.4f}")
    print(f"  MAE of means:          {mean_mae:.4f}")
    print(f"  MAE of std devs:       {std_mae:.4f}")
    print(f"  pred self-dist E[||Y-Y'||]: {np.mean(self_pred):.4f}")
    print()
    print("ATE Metrics:")
    for key, val in metrics.items():
        if np.isnan(val):
            print(f"  {key}: N/A")
        else:
            print(f"  {key}: {val:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
