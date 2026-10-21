"""
Evaluate a trained model against held-out interventional datasets.

This script loads a model checkpoint and dataset, runs the model to obtain
predicted interventional distributions, and computes distribution-level
statistical metrics (MMD, energy distance, Wasserstein, NLL, and descendant
vs. non-descendant marginal breakdowns).

For causal quantity evaluation (ATE, MAE), use evaluate_causal.py instead.
"""

from utils import *

import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np
from scipy.stats import multivariate_normal as scipy_mvn
from scipy.special import logsumexp as scipy_logsumexp

from dataclasses import dataclass
import tyro
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)


def _get_descendants(adj, node):
    """BFS descendants of node in adjacency matrix (adj[i,j]=1 means i→j)."""
    n = adj.shape[0]
    desc, visited, queue = set(), {node}, [node]
    while queue:
        cur = queue.pop(0)
        if cur != node:
            desc.add(cur)
        for child in range(n):
            if adj[cur, child] == 1 and child not in visited:
                visited.add(child)
                queue.append(child)
    return desc


def _energy_dist_np(X, Y):
    """NumPy energy distance between two sample sets."""
    def _mpw(A, B):
        d = A[:, None] - B[None, :]
        return np.mean(np.sqrt(np.sum(d ** 2, axis=-1) + 1e-12))
    return float(2 * _mpw(X, Y) - _mpw(X, X) - _mpw(Y, Y))


def _marg_gmm_nll_np(weights, means, covs, true_samples, dims):
    """
    NLL of true_samples under the marginal of a GMM restricted to `dims`.
    weights: (K,), means: (K, D), covs: (K, D, D) — all numpy.
    true_samples: (n_samples, D) — numpy.
    dims: list of int indices.
    """
    w = np.array(weights)
    m = np.array(means)[:, dims]              # (K, |dims|)
    c = np.array(covs)[:, dims, :][:, :, dims]  # (K, |dims|, |dims|)
    x = np.array(true_samples)[:, dims]       # (n_samples, |dims|)
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


def mean_energy_pvals(data, reconstructed_data):
    """
    Compute the mean p-value from the energy test across multiple SCMs.

    Args:
        data: A sequence/array where each entry is an (n_samples, n_vars) array
              representing the true interventional samples for a single SCM.
        reconstructed_data: A sequence/array with the same layout representing
                            samples drawn from the model for each SCM.

    Returns:
        The mean p-value (scalar) produced by `energy_test` averaged across the
        supplied SCMs.

        Notes:
                - This helper uses `jax.lax.fori_loop` so it remains compatible with JAX
                    transformations if needed.
                - Both `data` and `reconstructed_data` are sequences indexed by SCM and
                    are expected to have identical ordering when computing per-SCM tests.
    """

    def body_fn(i, carry):
        """ Accumulate p-values from energy_test. """
        _, pval = energy_test(
            data[i],
            reconstructed_data[i]
        )
        return carry + pval

    n = len(data)
    sum_pvals = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fn,
        init_val=0.0  # initial sum
    )
    return sum_pvals / n


@dataclass
class Args:
    seed: int = 42
    """Seed for the random number generator."""
    data_mode: str = 'interventional'
    """Whether observational data or interventional data is used"""
    data_path: str = 'data/TMLR/8var_linadd_normal_test.npz'
    """Filepath to the data."""
    model_path: str = 'models/TMLR/8var_linadd_normal_peach_sun_best'
    """Directory to save the checkpoint."""


if __name__ == '__main__':
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)
    print("Data path:", args.data_path)
    print("Model path:", args.model_path)

    # Load data
    data = load_scm_data(scm_file=args.data_path, mode=args.data_mode)
    samples_to_consider = min(len(data), 1000)
    rng, choice_key = jax.random.split(rng)
    data_sample_indicies = jax.random.choice(choice_key, len(data.arrays[0]), shape=(samples_to_consider,), replace=False)
    n_samples = data.arrays[0].shape[1]
    n_targets = data.arrays[1].shape[1]
    n_interventions = len(data.arrays)-3

    # Load model
    model, state = load_model_state(args.model_path)

    intervention_indicies = jnp.repeat(jnp.arange(0, model.n_interventions, 1)[jnp.newaxis, :],
                                             samples_to_consider,
                                             axis=0).T
    # predict distributions
    rng, dropoutkey, sample_key = jax.random.split(rng, 3)
    decoded_dist = [model.apply({'params': state.params},
                                dm=data.arrays[0][data_sample_indicies],
                                intervention_index=intervention_indicies[i],
                                intervention_targets=data.arrays[-1][data_sample_indicies, i, :, j],
                                is_training=False,
                                samples_from_prior=10,
                                rng=sample_key,
                                rngs={"dropout": dropoutkey},
                                method=model.inference)
                           for j in range(n_targets) for i in range(len(intervention_indicies))]

    # Sample distributions
    rng, samplekey = jax.random.split(rng)
    reconstructed_data = [decoded_dist[i].sample(seed=samplekey,
                                                 sample_shape=(n_samples,)).swapaxes(0, 1) for i
                                in range(len(intervention_indicies) * n_targets)]

    # Flatting for further processing
    flattened_reconstructed_interv_data = jnp.concat(reconstructed_data, axis=0)
    input_data = jnp.array(data.arrays[0])
    flattened_interv_data = jnp.concat(
        [data.arrays[i + 1][data_sample_indicies, j] for j in range(n_targets) for i in range(n_interventions)], axis=0)

    # compute and print metrics
    print('int_mmd', jnp.mean(jax.vmap(
            lambda scm_index: compute_mmd(flattened_interv_data[scm_index],
                                          flattened_reconstructed_interv_data[scm_index]))(
            jnp.arange(0, len(flattened_interv_data), 1))))
    print('int_wsd', jnp.mean(jax.vmap(
            lambda scm_index: compute_wasserstein_distance(flattened_interv_data[scm_index],
                                          flattened_reconstructed_interv_data[scm_index]))(
            jnp.arange(0, len(flattened_interv_data), 1))))
    print('int_energy', jnp.mean(jax.vmap(
            lambda scm_index: energy_distance(flattened_interv_data[scm_index],
                                          flattened_reconstructed_interv_data[scm_index]))(
            jnp.arange(0, len(flattened_interv_data), 1))))
    print('int_p', mean_energy_pvals(flattened_interv_data, flattened_reconstructed_interv_data))
    print('int_covs', jnp.array([decoded_dist[i][0].variance() for i in range(len(decoded_dist))]).mean())

    # Compute NLL: negative log-likelihood of true target samples under predicted distribution
    true_data_list = [data.arrays[i + 1][data_sample_indicies, j]
                      for j in range(n_targets) for i in range(n_interventions)]
    nll_per_dist = []
    for dist, true_samples in zip(decoded_dist, true_data_list):
        # true_samples: (samples_to_consider, n_samples, n_vars)
        # dist has batch_shape (samples_to_consider,), event_shape (n_vars,)
        true_samples_t = true_samples.swapaxes(0, 1)  # (n_samples, samples_to_consider, n_vars)
        log_probs = dist.log_prob(true_samples_t)  # (n_samples, samples_to_consider)
        nll_per_dist.append(-log_probs.mean())
    print('int_nll', jnp.mean(jnp.array(nll_per_dist)))

    # -----------------------------------------------------------------------
    # Marginal metrics: descendants and non-descendants
    # -----------------------------------------------------------------------
    # Graphs: data.arrays[-2] (n_scms, n_vars, n_vars)
    # Targets: data.arrays[-1] (n_scms, n_interventions, n_targets, n_vars)
    graphs_arr = np.array(data.arrays[-2])   # (n_scms, n_vars, n_vars)
    targets_arr = np.array(data.arrays[-1])  # (n_scms, n_interventions, n_targets, n_vars)
    n_vars = graphs_arr.shape[-1]

    nll_desc_list, nll_nondesc_list = [], []
    energy_desc_list, energy_nondesc_list = [], []

    pair_idx = 0
    for j in range(n_targets):
        for i in range(n_interventions):
            dist = decoded_dist[pair_idx]
            # reconstructed_data[pair_idx]: (samples_to_consider, n_samples, n_vars)
            rec_s = np.array(reconstructed_data[pair_idx])
            # true samples: (samples_to_consider, n_samples, n_vars)
            true_s = np.array(data.arrays[i + 1][data_sample_indicies, j])

            # Extract nested GMM parameters from the distrax distribution.
            # Outer mixture selects a prior sample (S components); inner is the
            # decoder GMM (K components). We flatten to S*K components.
            outer_probs = np.array(dist.mixture_distribution.probs)  # (batch, S)
            inner = dist.components_distribution
            inner_probs = np.array(inner.mixture_distribution.probs)           # (batch, S, K)
            inner_means = np.array(inner.components_distribution.loc)          # (batch, S, K, D)
            inner_covs  = np.array(inner.components_distribution.covariance_matrix)  # (batch, S, K, D, D)

            batch, S, K, D = inner_means.shape
            # flat_w[b] sums to 1 since outer_probs[b] and inner_probs[b,s] each sum to 1
            flat_w     = (outer_probs[:, :, None] * inner_probs).reshape(batch, S * K)  # (batch, S*K)
            flat_means = inner_means.reshape(batch, S * K, D)                            # (batch, S*K, D)
            flat_covs  = inner_covs.reshape(batch, S * K, D, D)                          # (batch, S*K, D, D)

            for k in range(samples_to_consider):
                scm_idx = int(data_sample_indicies[k])
                int_var = int(np.argmax(targets_arr[scm_idx, i, j]))
                adj = graphs_arr[scm_idx]
                desc     = _get_descendants(adj, int_var)
                non_desc = set(range(n_vars)) - desc - {int_var}
                desc_dims     = sorted(desc)
                non_desc_dims = sorted(non_desc)

                true_k = true_s[k]  # (n_samples, n_vars)
                rec_k  = rec_s[k]   # (n_samples, n_vars)
                w_k = flat_w[k]; m_k = flat_means[k]; c_k = flat_covs[k]

                if desc_dims:
                    energy_desc_list.append(
                        _energy_dist_np(true_k[:, desc_dims], rec_k[:, desc_dims]))
                    nll_desc_list.append(
                        _marg_gmm_nll_np(w_k, m_k, c_k, true_k, desc_dims))
                if non_desc_dims:
                    energy_nondesc_list.append(
                        _energy_dist_np(true_k[:, non_desc_dims], rec_k[:, non_desc_dims]))
                    nll_nondesc_list.append(
                        _marg_gmm_nll_np(w_k, m_k, c_k, true_k, non_desc_dims))

            pair_idx += 1

    print('desc_energy',    np.mean(energy_desc_list)    if energy_desc_list    else float('nan'))
    print('desc_nll',       np.mean(nll_desc_list)       if nll_desc_list       else float('nan'))
    print('nondesc_energy', np.mean(energy_nondesc_list) if energy_nondesc_list else float('nan'))
    print('nondesc_nll',    np.mean(nll_nondesc_list)    if nll_nondesc_list    else float('nan'))
