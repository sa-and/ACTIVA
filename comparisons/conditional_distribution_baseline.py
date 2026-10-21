# DEPRECATED: All functionality has been merged into evaluate_causal.py.
# Use evaluate_causal.py instead.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import numpy.linalg

from utils import load_scm_data, compute_wasserstein_distance, compute_mmd
import networkx as nx
from utils import *
import jax.numpy as jnp
from scipy.stats import multivariate_normal
from scipy.stats import multivariate_normal as scipy_mvn
from distrax import Normal, MultivariateNormalFullCovariance
from jax import random


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


def _mvn_nll_np(x, mean, cov):
    """NLL of samples x under N(mean, cov). Returns nan on numerical failure."""
    try:
        return float(-scipy_mvn.logpdf(x, mean, cov).mean())
    except Exception:
        return float('nan')

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)


if __name__ == '__main__':
    data_path = "./data/TMLR/8var_sergio_nodrop_test.npz"
    data_test = load_scm_data(scm_file=data_path, mode='interventional')
    interv_value = 0.0
    print(data_path)
    print("intervention value", interv_value)
    input_data = data_test.arrays[0]
    # graphs = [nx.relabel_nodes(nx.DiGraph(adj), {i: 'X'+str(i) for i in range(input_data.shape[-1])}) for adj in data_test.arrays[-1]]
    # columns = ['X'+str(i) for i in range(input_data.shape[-1])]
    key = random.PRNGKey(42)
    # distr = multivariate_normal.fit()
    mmds = {str(i)+'_'+str(j): [] for i in range(len(data_test.arrays[1:-2])) for j in range(data_test.arrays[1].shape[1])}
    mmds['obs'] = []
    wsds = {str(i)+'_'+str(j): [] for i in range(len(data_test.arrays[1:-2])) for j in range(data_test.arrays[1].shape[1])}
    wsds['obs'] = []
    energies = { str(i)+'_'+str(j): [] for i in range(len(data_test.arrays[1:-2])) for j in range(data_test.arrays[1].shape[1])}
    energies['obs'] = []
    ps = {str(i)+'_'+str(j): [] for i in range(len(data_test.arrays[1:-2])) for j in range(data_test.arrays[1].shape[1])}
    ps['obs'] = []
    nlogprobs = {str(i)+'_'+str(j): [] for i in range(len(data_test.arrays[1:-2])) for j in range(data_test.arrays[1].shape[1])}
    nlogprobs['obs'] = []
    covss = {str(i) + '_' + str(j): [] for i in range(len(data_test.arrays[1:-2])) for j in
          range(data_test.arrays[1].shape[1])}
    covss['obs'] = []
    n_vars = input_data.shape[-1]
    graphs_arr = np.array(data_test.arrays[-2])  # (n_scms, n_vars, n_vars)

    desc_energies, desc_nlls       = [], []
    nondesc_energies, nondesc_nlls = [], []

    not_converged = 0
    for test_index in range(min(1000, len(data_test))):  # do with 1000 iter
        try:
            means, covs = fit_multivariate_gaussian(input_data[test_index])
            key, subkey = random.split(key)
            samples = sample_gaussian(means, covs, input_data.shape[1])
        except numpy.linalg.LinAlgError as e:
            print(test_index, e)
            not_converged += 1
            mmds['obs'].append(jnp.nan)
            wsds['obs'].append(jnp.nan)
            nlogprobs['obs'].append(jnp.nan)
            energies['obs'].append(jnp.nan)
            ps['obs'].append(jnp.nan)
            covss['obs'].append(jnp.nan)
            continue
        # Observational case
        mmds['obs'].append(compute_mmd(samples, data_test.arrays[0][test_index]))
        wsds['obs'].append(compute_wasserstein_distance(samples, data_test.arrays[0][test_index]))
        nlogprobs['obs'].append(negative_log_likelihood_mvg(samples, means, covs))
        energy, p = energy_test(samples, data_test.arrays[0][test_index])
        energies['obs'].append(energy)
        ps['obs'].append(p)
        covss['obs'].append(jnp.diagonal(covs).mean())
        for i in range(len(data_test.arrays[1:-2])):
            for j in range(data_test.arrays[i+1].shape[1]):
                cond_means, cond_covs = conditional_gaussian_params(means, covs, j, interv_value)
                # cond_means = jnp.concat([cond_means[:j], jnp.array([interv_value]), cond_means[j:]])
                # cond_covs = jnp.concat([cond_covs[:, :j], jnp.zeros((cond_covs.shape[0], 1)), cond_covs[:, j:]], axis=-1)
                # cond_covs = jnp.concat([cond_covs[:j, :], jnp.zeros((1, cond_covs.shape[1])), cond_covs[j:, :]],
                #                        axis=0)
                try:
                    samples = sample_gaussian(cond_means, cond_covs, input_data.shape[1])
                except numpy.linalg.LinAlgError as e:
                    print(test_index, i,j,e)
                    not_converged += 1
                    mmds[str(i) + '_' + str(j)].append(jnp.nan)
                    wsds[str(i) + '_' + str(j)].append(jnp.nan)
                    energies[str(i) + '_' + str(j)].append(jnp.nan)
                    ps[str(i) + '_' + str(j)].append(jnp.nan)
                    covss[str(i) + '_' + str(j)].append(jnp.nan)
                    nlogprobs[str(i) + '_' + str(j)].append(jnp.nan)
                    continue
                samples = jnp.concat([samples[:, :j], jnp.ones((samples.shape[0], 1))*interv_value, samples[:, j:]], axis=-1)
                mmds[str(i)+'_'+str(j)].append(compute_mmd(samples, data_test.arrays[i+1][test_index, j]))
                wsds[str(i)+'_'+str(j)].append(compute_wasserstein_distance(samples, data_test.arrays[i+1][test_index, j]))
                energy, p = energy_test(samples, data_test.arrays[i+1][test_index, j])
                energies[str(i)+'_'+str(j)].append(energy)
                ps[str(i)+'_'+str(j)].append(p)
                covss[str(i) + '_' + str(j)].append(jnp.diagonal(cond_covs).mean())
                true_interv_samples = data_test.arrays[i+1][test_index, j]  # (n_samples, n_vars)
                true_interv_others = jnp.concat([true_interv_samples[:, :j], true_interv_samples[:, j+1:]], axis=-1)
                # Full joint NLL = conditional NLL (n_vars-1) + NLL of X_j.
                # X_j is deterministically set to interv_value with probability 1,
                # so log p(x_j = interv_value) = log(1) = 0, contributing 0 to NLL.
                nll_conditional = negative_log_likelihood_mvg(true_interv_others, cond_means, cond_covs)
                nll_x_j = 0.0  # deterministic intervention: prob=1 at interv_value
                nlogprobs[str(i)+'_'+str(j)].append(nll_conditional + nll_x_j)

                # --- Marginal metrics for descendants and non-descendants ---
                # The baseline conditions on variable j (the intervention target).
                # cond_means / cond_covs represent the (n_vars-1)-dim distribution
                # over all other variables given X_j = interv_value.
                adj = graphs_arr[test_index]
                # Use j as the intervention variable (consistent with the baseline's
                # conditioning choice above).
                int_var = j
                desc     = _get_descendants(adj, int_var)
                non_desc = set(range(n_vars)) - desc - {int_var}
                desc_dims     = sorted(desc)
                non_desc_dims = sorted(non_desc)

                # In the conditional distribution (n_vars-1 dims), variable indices
                # are shifted by -1 for any variable with original index > j.
                def _adj(dim_list):
                    return [d if d < j else d - 1 for d in dim_list]

                true_samples_np = np.array(true_interv_samples)  # (n_samples, n_vars)
                pred_samples_np = np.array(samples)               # (n_samples, n_vars)
                cond_means_np   = np.array(cond_means)
                cond_covs_np    = np.array(cond_covs)

                if desc_dims:
                    adj_desc = _adj(desc_dims)
                    desc_energies.append(
                        _energy_dist_np(true_samples_np[:, desc_dims],
                                        pred_samples_np[:, desc_dims]))
                    m_d = cond_means_np[adj_desc]
                    c_d = cond_covs_np[np.ix_(adj_desc, adj_desc)]
                    desc_nlls.append(
                        _mvn_nll_np(true_samples_np[:, desc_dims], m_d, c_d))

                if non_desc_dims:
                    adj_nondesc = _adj(non_desc_dims)
                    nondesc_energies.append(
                        _energy_dist_np(true_samples_np[:, non_desc_dims],
                                        pred_samples_np[:, non_desc_dims]))
                    m_nd = cond_means_np[adj_nondesc]
                    c_nd = cond_covs_np[np.ix_(adj_nondesc, adj_nondesc)]
                    nondesc_nlls.append(
                        _mvn_nll_np(true_samples_np[:, non_desc_dims], m_nd, c_nd))

    print('obs_mmd', jnp.nanmean(jnp.array(mmds['obs'])))
    print('int_mmd', jnp.nanmean(jnp.array([item[1] for item in mmds.items()][:-1])))
    print('obs_wsd', jnp.nanmean(jnp.array(wsds['obs'])))
    print('int_wsd', jnp.nanmean(jnp.array([item[1] for item in wsds.items()][:-1])))
    print('obs_energy', jnp.nanmean(jnp.array(energies['obs'])))
    print('int_energy', jnp.nanmean(jnp.array([item[1] for item in energies.items()][:-1])))
    print('obs_p', jnp.nanmean(jnp.array(ps['obs'])))
    print('int_p', jnp.nanmean(jnp.array([item[1] for item in ps.items()][:-1])))
    print('obs_covs', jnp.nanmean(jnp.array(covss['obs'])))
    print('int_covs', jnp.nanmean(jnp.array([item[1] for item in covss.items()][:-1])))
    print('obs_nlogprob', jnp.nanmean(jnp.array(nlogprobs['obs'])))
    print('int_nll', jnp.nanmean(jnp.array([item[1] for item in nlogprobs.items()][:-1])))
    print('not converged', not_converged)
    print('desc_energy',    np.nanmean(desc_energies)    if desc_energies    else float('nan'))
    print('desc_nll',       np.nanmean(desc_nlls)        if desc_nlls        else float('nan'))
    print('nondesc_energy', np.nanmean(nondesc_energies) if nondesc_energies else float('nan'))
    print('nondesc_nll',    np.nanmean(nondesc_nlls)     if nondesc_nlls     else float('nan'))