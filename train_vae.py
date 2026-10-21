import jax.random
import optax
from tqdm import tqdm
from dataclasses import dataclass
import tyro
from models import ConditionalVAE
from utils import *
from flax.training import train_state
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import jax_dataloader as jdl
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
# jax.config.update("jax_debug_nans", True)  # Very slow; only enable for debugging NaNs

"""
Training entrypoint for the Conditional VAE models.

This script exposes a small command-line interface (via `tyro`) that
configures model/optimization/hyperparameters, loads SCM datasets saved as
NumPy `.npz` files, initializes the Flax `ConditionalVAE` model, and runs a
training loop that periodically saves checkpoints (dill format) and
diagnostic plots. The top-level `Args` dataclass documents the CLI options.

The module changes are documentation-only (no behavior changes) made to make
the CLI and dataset expectations explicit.
"""


# Loss and training functions
def interventional_vae_loss(state, params, batch, intervention_indices, rng, n_kl_samples=1, kl_beta=0.5):
    """
    Computes the loss for the Conditional VAE with interventional data.

    This function:
    1. Extracts data from the batch
    2. Performs forward passes through the VAE for all intervention/target pairs
    3. Computes reconstruction loss using log probability
    4. Computes KL divergence between posterior and prior distributions
    5. Returns the weighted sum of reconstruction loss and KL divergence
    """
    # Step 1: Extract data from the batch
    dm, *target_data, graphs, targets = batch
    targets = targets.swapaxes(0, 1)  # Adjust dimensions for processing
    n_targets = targets.shape[-2]  # Number of target variables
    n_vars = targets.shape[-1]  # Total number of variables
    rng, vaekey, dropoutkey = jax.random.split(rng, 3)  # Split RNG for different uses

    # Step 2: Single vectorized forward pass computing posterior + prior together.
    # forward_with_prior() calls obs_data_network once and feeds z_obs to both
    # the posterior encoder and the prior GMM head, halving the encoder compute.
    def apply_fn_combined(x_ij, intervention_index_i, intervention_targets_ij):
        return state.apply_fn(
            {'params': params},
            x=x_ij,
            dm=dm,
            intervention_index=intervention_index_i,
            intervention_targets=intervention_targets_ij,
            is_training=True,
            rng=vaekey,
            rngs={"dropout": dropoutkey},
            method="forward_with_prior",
        )

    apply_combined_over_targets = jax.vmap(apply_fn_combined, in_axes=(0, None, 0))
    apply_combined_over_interventions = jax.vmap(apply_combined_over_targets, in_axes=(0, 0, 0))

    x_data = jnp.array([[target_data[i][:, j, ...] for j in range(n_targets)]
                        for i in range(len(intervention_indices))])
    intervention_targets_data = jnp.array([
        [targets[i, :, j] for j in range(n_targets)]
        for i in range(len(intervention_indices))
    ])

    dec_weights, dec_means, dec_covs, latent_means, latent_logvars, prior_weights, prior_means, prior_covs = \
        apply_combined_over_interventions(x_data, intervention_indices, intervention_targets_data)

    # Step 3: Compute reconstruction loss
    distr = state.apply_fn({"params": params}, method="get_decoded_distr",
                           weights=dec_weights, means=dec_means, covs=dec_covs)
    log_probs = jax.vmap(lambda sample: distr.log_prob(sample), out_axes=3, in_axes=3)(x_data)
    reconstruction_loss = -jnp.mean(log_probs)

    # Step 4: Compute KL divergence
    q_z_given_x_u = state.apply_fn({"params": params}, method="get_encoded_distribution",
                                   mean=latent_means, logvar=latent_logvars)
    priors = state.apply_fn({"params": params},
                            weights=prior_weights, means=prior_means, covs=prior_covs,
                            method="get_conditional_prior")

    # Calculate KL divergence using log probabilities
    # Sample z ~ q (reparameterized) and evaluate both log-probs at the same sample,
    # giving an unbiased MC estimate of E_{z~q}[log q(z) - log p(z)].
    rng, kl_rng = jax.random.split(rng)
    z_samples = q_z_given_x_u.sample(seed=kl_rng, sample_shape=(n_kl_samples,))
    log_probs_prior = priors.log_prob(z_samples)
    log_probs_q = q_z_given_x_u.log_prob(z_samples)
    kl = jnp.mean(log_probs_q - log_probs_prior)

    # Step 5: Return total loss (reconstruction loss + weighted KL divergence)
    return reconstruction_loss + kl_beta * kl


def interventional_train_step(state, batch, intervention_indicies, rng, n_kl_samples=1, kl_beta=0.5):
    """
    Performs a single training step for the Conditional VAE with interventional data.

    Args:
        state: The current training state containing parameters and optimizer state.
        batch: A batch of data containing observational data, interventional data, graphs, and targets.
        intervention_indicies: Indices of interventions to model.
        rng: Random number generator key.
        n_kl_samples: Number of Monte Carlo samples for the KL estimate.
        kl_beta: Weight on the KL divergence term.

    Returns:
        A tuple (updated_state, loss).
    """
    # Step 1: Create a loss function that takes parameters as input
    loss_fn = lambda params: interventional_vae_loss(state, params, batch, intervention_indicies, rng, n_kl_samples, kl_beta)
    
    # Step 2: Compute loss value and gradients
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Step 3: Update parameters using gradients
    state = state.apply_gradients(grads=grads)
    
    return state, loss


@dataclass
class Args:
    seed: int = 42
    """Seed for the random number generator."""
    track: bool = True
    """Toggle logging on wandb"""
    logging_freq: int = 100
    """Logging frequency in epochs."""
    batch_size: int = 256
    """Minibatch size"""
    epochs: int = 5000
    """Number of training epochs"""
    hidden_dim: int = 256
    """Number of hidden nodes in the FFNN of the transformer blocks."""
    embedding_dim: int = 64
    """Embedding dimension of each column feature."""
    encoder_attention_layers: int = 4
    """Number of attention layers in the encoder transformer. Must be a multiple of 2."""
    decoder_attention_layers: int = 4
    """Number of attention layers in the decoder transformer."""
    attention_heads: int = 4
    """Number of attention heads in the transformers"""
    dropout: float = 0.0
    """Dropout rate for the VAE."""
    decoder_mixture_components: int = 4
    """Number of Gaussian mixture components for the decoder distribution."""
    prior_mixture_components: int = 20
    """Number of Gaussian mixture components for the prior distribution."""
    n_kl_samples: int = 4
    """Number of latent samples for the KL Monte Carlo estimate during training."""
    n_prior_samples: int = 5
    """Number of prior samples used to build the two-level mixture during evaluation."""
    kl_beta: float = 0.5
    """Weight on the KL divergence term in the ELBO loss."""
    learning_rate: float = 5e-4
    """Learning rate for the optimizer."""
    data_mode: str = 'interventional'
    """Whether observational data or interventional data is used"""
    model: str = 'CVAE'
    """Which kind of VAE model to use. Standard VAE (VAE), conditional VAE (CVAE), or conditional structure VAE (CSVAE)"""
    data_path: str = 'data/TMLR/8var_linadd_normal'  # base path; script appends _train.npz and _val.npz
    """Filepath to the data."""
    save_path: str = 'models/TMLR/8var_linadd_normal'
    """Directory to save the checkpoint."""
    comp_dtype = jnp.bfloat16


def main():
    """Main training function for the Conditional VAE model.

    Notes / conventions:
    - The ``--data-path`` argument is expected to be the base path for the dataset
        (without the trailing ``_train.npz`` / ``_val.npz``). The script appends
        ``"_train.npz"`` for the training set and ``"_val.npz"`` for validation.
    - For interventional data the expected saved arrays (in the .npz files) are:
        * ``input`` : shape (n_scms, n_samples, n_vars)
        * ``target_data`` : shape (n_interventions, n_scms, n_targets, n_samples, n_vars)
        * ``graphs`` : shape (n_scms, n_vars, n_vars)
        * ``targets`` : shape (n_scms, n_interventions, n_targets, n_vars)
        These names and shapes match the output produced by ``generate_data.py`` in
        interventional mode.
    - Checkpoints are written via :func:`save_model_state(save_path, model, state)` and
        use the project's dill-based format (the saved file contains the model
        class, serialized train state, params and optimizer state). Use
        :func:`load_model_state(path)` to reconstruct the model + TrainState.

    What this function does:
    1. Parses command line arguments
    2. Sets up distributed training if available and optionally initializes
            wandb for tracking (only on the main process)
    3. Loads and prepares data using :func:`load_scm_data`
    4. Initializes the Flax :class:`ConditionalVAE`, optimizer and ``TrainState``
    5. Runs the training loop, logging and periodically saving checkpoints and
            diagnostic plots. The training step is jitted for performance.
    """
    # Parse command line arguments
    args = tyro.cli(Args)

    # Set up distributed training
    process_id = jax.process_index()
    process_count = jax.process_count()
    is_main_process = process_id == 0

    # Only track metrics on the main process
    if not is_main_process:
        args.track = False
    print(f"Process {process_id} of {process_count-1} initialized")

    # Initialize wandb for experiment tracking
    if args.track and is_main_process:
        import wandb
        wandb.init(
            project="identifiable_ACTIVA",
            config=vars(args),
            save_code=True,
        )

    rng = jax.random.PRNGKey(args.seed)

    # Create a device mesh for parallelization
    devices = jax.devices()
    print(f"Available Devices: {devices}")

    # Create a 1D mesh for data parallelism
    mesh = jax.make_mesh((len(devices),), ("batch",))

    # Load data and prepare data loader
    data_train = load_scm_data(scm_file=args.data_path+"_train.npz", mode=args.data_mode, dtype=args.comp_dtype)
    data_val = load_scm_data(scm_file=args.data_path + "_val.npz", mode=args.data_mode, dtype=args.comp_dtype)
    n_samples = data_train.arrays[0].shape[1]
    n_interventions = len(data_train[1])-3
    n_targets = data_train.arrays[-1].shape[-2]  # Note that the target setup in this code works only if every intervention has the same number of targets
    batch_size = min(args.batch_size, len(data_train))
    data_loader = jdl.DataLoader(data_train, backend="jax", batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    flattened_interv_data_train = jnp.concat(
        [data_train.arrays[i + 1][:, j] for j in range(n_targets) for i in range(n_interventions)], axis=0)
    flattened_interv_data_val = jnp.concat(
        [data_val.arrays[i + 1][:, j] for j in range(n_targets) for i in range(n_interventions)], axis=0)

    # Save original data distribution plot
    rng, sample_rng, sample_rng_val = jax.random.split(rng, 3)
    plotting_samples = jnp.array(jax.random.choice(a=jnp.arange(len(flattened_interv_data_train)), key=sample_rng, shape=(10,)))
    plotting_samples_val = jnp.array(jax.random.choice(a=jnp.arange(len(flattened_interv_data_val)), key=sample_rng_val, shape=(10,)))
    save_distribution_plot(flattened_interv_data_train[plotting_samples], "data/train/Original Data Distribution Samples", "original_distribution_samples_train.png",
                           colored=True, tracking=args.track, has_batch_dim=True)
    save_distribution_plot(flattened_interv_data_val[plotting_samples_val], "data/val/Original Data Distribution Samples", "original_distribution_samples_val.png",
                           colored=True, tracking=args.track, has_batch_dim=True)

    # Set up the VAE
    input_data_dim = output_data_dim = data_train.arrays[0].shape[-1]

    vae_config = {
        'data_dim': output_data_dim,
        'hidden_dim': args.hidden_dim,
        'embedding_dim': args.embedding_dim,
        'dropout': args.dropout,
        'attention_heads': args.attention_heads,
        'encoder_attention_layers': args.encoder_attention_layers,
        'decoder_attention_layers': args.decoder_attention_layers,
        'decoder_mixture_components': args.decoder_mixture_components,
        'prior_mixture_components': args.prior_mixture_components,
    }

    model = ConditionalVAE(**vae_config, n_interventions=n_interventions, n_samples=len(data_train.arrays[0][1]))

    # Initialize model
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng,
                        jnp.zeros((batch_size, 1, input_data_dim), dtype=args.comp_dtype),
                        jnp.ones((batch_size, data_train.arrays[0].shape[1], input_data_dim), dtype=args.comp_dtype),
                        jnp.zeros((batch_size, ), dtype=args.comp_dtype),
                        jnp.zeros((batch_size, output_data_dim), dtype=args.comp_dtype),
                        True,  init_rng,
                        method=model.init_networks)['params']

    # Set up optimizer
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.apply_if_finite(optax.adamw(learning_rate=args.learning_rate), 10))

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # JIT-compiled helpers that batch encode/prior calls across all intervention×target combos,
    # replacing Python loops with a single vmapped XLA kernel call per logging step.
    @jax.jit
    def _batch_encode(params, x_all, dm, idx_all, targets_all, dropout_key):
        def encode_one(x, idx, targets):
            return model.apply({'params': params}, x=x, dm=dm, intervention_index=idx,
                               intervention_targets=targets, rngs={'dropout': dropout_key},
                               method=model.encode, is_training=False)[0]
        return jax.vmap(encode_one)(x_all, idx_all, targets_all)

    @jax.jit
    def _batch_prior(params, dm, idx_all, targets_all, dropout_key):
        def prior_one(idx, targets):
            return model.apply({'params': params}, dm=dm, intervention_index=idx,
                               intervention_targets=targets, rngs={'dropout': dropout_key},
                               method=model.encode_prior, is_training=False)
        return jax.vmap(prior_one)(idx_all, targets_all)

    intervention_indices = jnp.repeat(jnp.arange(0, n_interventions, 1, dtype=args.comp_dtype)[jnp.newaxis, :], batch_size, axis=0).T

    rng, train_rng = jax.random.split(rng)
    jited_train = jax.jit(
        lambda state, batch, intervention_indicies, rng: interventional_train_step(
            state, batch, intervention_indicies, rng, args.n_kl_samples, args.kl_beta),
        donate_argnums=(0,))

    best_val_metric = float('inf')

    # Precompute logging constants (fixed across all epochs)
    n_latents = min(500, len(data_train))
    n_latents_val = min(500, len(data_val))
    intervention_indicies_train = jnp.repeat(jnp.arange(0, model.n_interventions, 1)[jnp.newaxis, :], n_latents, axis=0).T
    intervention_indicies_val = jnp.repeat(jnp.arange(0, model.n_interventions, 1)[jnp.newaxis, :], n_latents_val, axis=0).T
    log_flat_train = jnp.concat([data_train.arrays[i + 1][:n_latents, j] for j in range(n_targets) for i in range(n_interventions)], axis=0)
    log_flat_val = jnp.concat([data_val.arrays[i + 1][:n_latents_val, j] for j in range(n_targets) for i in range(n_interventions)], axis=0)
    log_ps_train = plotting_samples
    log_ps_val = plotting_samples_val
    # Stacked inputs for batched encode/prior (K = n_interventions * n_targets iterations → 1 vmapped call)
    dm_log_train = data_train.arrays[0][:n_latents]
    dm_log_val = data_val.arrays[0][:n_latents_val]
    x_log_train = jnp.stack([data_train.arrays[1 + i][:n_latents, j] for j in range(n_targets) for i in range(n_interventions)])
    x_log_val = jnp.stack([data_val.arrays[1 + i][:n_latents_val, j] for j in range(n_targets) for i in range(n_interventions)])
    idx_log_train = jnp.stack([intervention_indicies_train[i] for j in range(n_targets) for i in range(n_interventions)])
    idx_log_val = jnp.stack([intervention_indicies_val[i] for j in range(n_targets) for i in range(n_interventions)])
    targets_log_train = jnp.stack([data_train.arrays[-1][:n_latents, i, j, :] for j in range(n_targets) for i in range(n_interventions)])
    targets_log_val = jnp.stack([data_val.arrays[-1][:n_latents_val, i, j, :] for j in range(n_targets) for i in range(n_interventions)])

    # main training loop
    with mesh:
        batch_sharding = NamedSharding(mesh, P('batch'))
        interv_sharding = NamedSharding(mesh, P(None, 'batch'))
        for epoch in tqdm(range(args.epochs)):
            for batch in data_loader:
                sharded_batch = jax.device_put(batch, batch_sharding)
                sharded_intervention_indices = jax.device_put(intervention_indices, interv_sharding)

                rng, step_rng = jax.random.split(rng)
                state, loss = jited_train(state, sharded_batch, sharded_intervention_indices, step_rng)

            # LOGGING
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
            if args.track and is_main_process:
                wandb.log({"train/loss": loss, "train/epoch": epoch}, commit=True)
            if epoch % args.logging_freq == 0 or epoch == 0:
                # Save checkpoint
                save_model_state(args.save_path, model, state)

                # Save encoded data distribution plot
                rng, dropoutkey = jax.random.split(rng)

                _enc_tr = _batch_encode(state.params, x_log_train, dm_log_train, idx_log_train, targets_log_train, dropoutkey)
                encoded_mean_train = _enc_tr.reshape(-1, _enc_tr.shape[-1])
                save_distribution_plot(encoded_mean_train, "encodings/train/distribution", "encoded_distribution_train.png", tracking=args.track)
                save_distribution_plot(encoded_mean_train[log_ps_train], "encodings/train/distribution_samples", "encoded_distribution_samples_train.png", colored=True, tracking=args.track)

                _enc_val = _batch_encode(state.params, x_log_val, dm_log_val, idx_log_val, targets_log_val, dropoutkey)
                encoded_mean_val = _enc_val.reshape(-1, _enc_val.shape[-1])
                save_distribution_plot(encoded_mean_val, "encodings/val/distribution", "encoded_distribution_val.png", tracking=args.track)
                save_distribution_plot(encoded_mean_val[log_ps_val], "encodings/val/distribution_samples", "encoded_distribution_samples_val.png", colored=True, tracking=args.track)

                # --- Prior-based reconstruction (following inference procedure) ---
                # Compute prior distribution parameters for train and val in single vmapped calls
                pri_w_tr, pri_m_tr, pri_c_tr = _batch_prior(state.params, dm_log_train, idx_log_train, targets_log_train, dropoutkey)
                K_tr = pri_w_tr.shape[0] * pri_w_tr.shape[1]
                pri_w_tr = pri_w_tr.reshape(K_tr, pri_w_tr.shape[-1])
                pri_m_tr = pri_m_tr.reshape(K_tr, pri_m_tr.shape[-2], pri_m_tr.shape[-1])
                pri_c_tr = pri_c_tr.reshape(K_tr, pri_c_tr.shape[-3], pri_c_tr.shape[-2], pri_c_tr.shape[-1])
                prior_train_distribution = model.apply(
                    {'params': state.params}, weights=pri_w_tr, means=pri_m_tr, covs=pri_c_tr, rngs={'dropout': dropoutkey}, method=model.get_conditional_prior
                )

                pri_weights, pri_means, pri_covs = _batch_prior(state.params, dm_log_val, idx_log_val, targets_log_val, dropoutkey)
                K_val = pri_weights.shape[0] * pri_weights.shape[1]
                pri_weights = pri_weights.reshape(K_val, pri_weights.shape[-1])
                pri_means = pri_means.reshape(K_val, pri_means.shape[-2], pri_means.shape[-1])
                pri_covs = pri_covs.reshape(K_val, pri_covs.shape[-3], pri_covs.shape[-2], pri_covs.shape[-1])
                prior_val_distribution = model.apply(
                    {'params': state.params}, weights=pri_weights, means=pri_means, covs=pri_covs, rngs={'dropout': dropoutkey}, method=model.get_conditional_prior
                )

                # Sample n_prior_samples latents from each prior and build two-level mixture
                # (following the same procedure as model.inference())
                rng, samplekey, dropoutkey = jax.random.split(rng, 3)

                decoded_dist_train = model.apply({'params': state.params}, encoded_mean_train, is_training=False, method=model.decode, rngs={"dropout": dropoutkey})
                decoded_dist_train = model.apply({'params': state.params}, weights=decoded_dist_train[0], means=decoded_dist_train[1], covs=decoded_dist_train[2], method=model.get_decoded_distr)
                decoded_dist_val = model.apply({'params': state.params}, encoded_mean_val, is_training=False, method=model.decode, rngs={"dropout": dropoutkey})
                decoded_dist_val = model.apply({'params': state.params}, weights=decoded_dist_val[0], means=decoded_dist_val[1], covs=decoded_dist_val[2], method=model.get_decoded_distr)

                rng, prior_samplekey = jax.random.split(rng)
                reconstructed_data_prior_val = None
                decoded_dist_prior_val = None
                for prior_distribution, prior_split_key, suffix in [
                    (prior_train_distribution, prior_samplekey, "train"),
                    (prior_val_distribution, prior_samplekey, "val"),
                ]:
                    z_prior = prior_distribution.sample(seed=prior_split_key, sample_shape=(args.n_prior_samples,))
                    # z_prior: (n_prior_samples, total_batch, n_vars, latent_dim)
                    dec_w, dec_m, dec_c = jax.vmap(
                        lambda z_i: model.apply({'params': state.params}, z_i, is_training=False,
                                                method=model.decode, rngs={"dropout": dropoutkey})
                    )(z_prior)
                    _b, _s = z_prior.shape[1], args.n_prior_samples
                    decoded_dist_prior = distrax.MixtureSameFamily(
                        distrax.Categorical(probs=jnp.ones((_b, _s)) / _s),
                        distrax.MixtureSameFamily(
                            distrax.Categorical(probs=dec_w.swapaxes(0, 1)),
                            distrax.MultivariateNormalFullCovariance(
                                loc=dec_m.swapaxes(0, 1),
                                covariance_matrix=dec_c.swapaxes(0, 1)
                            )
                        )
                    )
                    rng, samplekey = jax.random.split(rng)
                    reconstructed_data_prior = decoded_dist_prior.sample(seed=samplekey, sample_shape=(n_samples,)).swapaxes(0, 1)
                    save_distribution_plot(reconstructed_data_prior, f"reconstruction/{suffix}/Reconstructed Data Distribution Prior", f"reconstructed_distribution_prior_{suffix}.png", tracking=args.track, has_batch_dim=True)
                    _ps = log_ps_val if suffix == "val" else log_ps_train
                    save_distribution_plot(reconstructed_data_prior[_ps], f"reconstruction/{suffix}/Reconstructed Data Distribution Prior Samples", f"reconstructed_distribution_prior_samples_{suffix}.png", colored=True, tracking=args.track, has_batch_dim=True)
                    if suffix == "val":
                        reconstructed_data_prior_val = reconstructed_data_prior
                        decoded_dist_prior_val = decoded_dist_prior

                # plot samples from the decoded distribution based on samples from the encoder
                rng, samplekey = jax.random.split(rng)
                reconstructed_data_train = decoded_dist_train.sample(seed=samplekey, sample_shape=(n_samples,)).swapaxes(0, 1)
                save_distribution_plot(reconstructed_data_train, "reconstruction/train/Reconstructed Data Distribution Encodings", "reconstructed_distribution_encodings_train.png", tracking=args.track, has_batch_dim=True)
                save_distribution_plot(reconstructed_data_train[log_ps_train], "reconstruction/train/Reconstructed Data Distribution Encodings Samples", "reconstructed_distribution_encodings_samples_train.png", colored=True, tracking=args.track, has_batch_dim=True)

                reconstructed_data_val = decoded_dist_val.sample(seed=samplekey, sample_shape=(n_samples,)).swapaxes(0, 1)
                save_distribution_plot(reconstructed_data_val, "reconstruction/val/Reconstructed Data Distribution Encodings", "reconstructed_distribution_encodings_val.png", tracking=args.track, has_batch_dim=True)
                save_distribution_plot(reconstructed_data_val[log_ps_val], "reconstruction/val/Reconstructed Data Distribution Encodings Samples", "reconstructed_distribution_encodings_samples_val.png", colored=True, tracking=args.track, has_batch_dim=True)

                # Compute NLL on val set under the prior-based inference distribution.
                # log_flat_val: (n_latents_val * n_interventions * n_targets, n_samples, n_vars); vmap over n_samples.
                val_nll = float(-jnp.mean(
                    jax.vmap(decoded_dist_prior_val.log_prob)(log_flat_val.swapaxes(0, 1))
                ))
                print(f"  Val NLL (prior): {val_nll:.4f}")

                if is_main_process and val_nll < best_val_metric:
                    best_val_metric = val_nll
                    save_model_state(args.save_path + "_best", model, state)
                    print(f"  New best model saved to {args.save_path}_best")

                if args.track and is_main_process:
                    wandb.log({"val/nll_prior": val_nll}, commit=False)
                    var_val = jnp.var(encoded_mean_val.T, axis=1)
                    wandb.log({"encodings/val/mean variance": var_val.mean()}, commit=False)
                    wandb.log({"encodings/val/variance dim" + str(i): var_val[i] for i in range(len(var_val))}, commit=False)
                    var_train = jnp.var(encoded_mean_train.T, axis=1)
                    wandb.log({"encodings/train/mean variance": var_train.mean()}, commit=False)
                    wandb.log({"encodings/train/variance dim" + str(i): var_train[i] for i in range(len(var_train))}, commit=False)

                    # Evaluate the reconstructed data
                    # compute metrics for each scm
                    wandb.log({
                        "train/mean_interv_energy": jnp.mean(jax.vmap(lambda i: energy_distance(log_flat_train[i], reconstructed_data_train[i]))(jnp.arange(0, len(log_flat_train), 1))),
                        "val/mean_interv_energy": jnp.mean(jax.vmap(lambda i: energy_distance(log_flat_val[i], reconstructed_data_val[i]))(jnp.arange(0, len(log_flat_val), 1))),
                        "val/mean_interv_energy_prior": jnp.mean(jax.vmap(lambda i: energy_distance(log_flat_val[i], reconstructed_data_prior_val[i]))(jnp.arange(0, len(log_flat_val), 1))),
                    }, commit=False)

if __name__ == "__main__":
    main()

