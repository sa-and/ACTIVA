"""
Evaluate a trained CVAE model on causal quantity estimation.

Computes ground-truth causal quantities from held-out interventional data and
compares multiple methods: baselines (zero, Gaussian conditional, linear causal)
and the CVAE model. Outputs formatted tables of metrics for:
  - ATE: Average Treatment Effect = E[Y | do(X_j=v)] - E[Y | obs]
  - Interventional Mean: E[Y | do(X_j=v)]
  - Interventional Variance: Var[Y | do(X_j=v)]
  - CATE: Conditional ATE given first PC of all valid pre-treatment covariates (lo/hi strata)
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import tyro
import dill
import optax
from flax.training import train_state
from utils import load_scm_data, fit_multivariate_gaussian, conditional_gaussian_params, sample_gaussian
import warnings
from scipy.stats import multivariate_normal as scipy_mvn
from scipy.special import logsumexp as scipy_logsumexp


@dataclass
class Args:
    data_path: str = 'data/AAAI/2var_linadd_beta_test.npz'
    """Path to the test dataset (.npz file)."""
    model_path: str = 'models/2var_linadd_beta_legendary_sun'
    """Path to the trained CVAE model checkpoint."""
    mace_tnp_model_path: str = 'models/mace_tnp_2var_linadd_beta_best.pt'
    """Path to a trained MACE-TNP checkpoint (.pt). Leave empty to skip MACE-TNP evaluation."""
    seed: int = 42
    """Random seed for reproducibility."""
    n_eval_scms: int = 2000
    """Number of SCMs to evaluate (for speed; use all if > available)."""
    n_model_samples: int = 200
    """Number of samples drawn from CVAE per SCM×target."""
    samples_from_prior: int = 20
    """Number of prior samples used in CVAE inference."""
    n_bootstrap: int = 1000
    """Number of bootstrap resamples for 95% confidence intervals."""
    intervention_value: float = 5.0
    """Fixed value used for all do(X_j = v) interventions. Must match data generation."""
    track: bool = False
    """Whether to log results to wandb."""
    results_path: str = ''
    """Path to save results as JSON. If empty, auto-generated from model_path."""
    n_plot_scms: int = 4
    """Number of SCMs to include in the distribution comparison plot (0 to skip)."""
    plot_path: str = ''
    """Path to save the distribution comparison plot (.pdf). Auto-generated if empty."""
    top_k_components: int = 1
    """Evaluate an additional variant using only the top-k highest-weight mixture components."""


def kahn_topological_sort(adj_matrix):
    """
    Topological sort using Kahn's algorithm.

    Args:
        adj_matrix: numpy array of shape (n_vars, n_vars) where adj_matrix[i, j] = 1
                   means edge i -> j (i is parent of j).

    Returns:
        List of node indices in topological order (parents before children).
    """
    n = adj_matrix.shape[0]
    adj = np.array(adj_matrix, dtype=int)

    in_degree = adj.sum(axis=0)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []

    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for child in range(n):
            if adj[node, child] == 1:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    if len(topo_order) != n:
        raise ValueError("Graph contains a cycle; topological sort not possible.")

    return topo_order


def get_descendants(adj_matrix, node):
    """
    Get all descendants of a given node using BFS.

    Args:
        adj_matrix: numpy array of shape (n_vars, n_vars) where adj_matrix[i, j] = 1
                   means edge i -> j.
        node: The source node.

    Returns:
        Set of all descendant node indices (excluding the source node itself).
    """
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


def _pc1_direction(obs_k, valid_c):
    """
    Return the first principal component direction of obs_k[:, valid_c].

    This is used to project all valid pre-treatment covariates onto a single
    scalar score Z = X_{valid_c} @ v, enabling joint multivariate conditioning
    rather than looping over each covariate individually.

    Returns:
        v: (len(valid_c),) unit vector (first PC of obs covariate block).
           When len(valid_c) == 1 returns np.ones(1) (trivial).
    """
    if len(valid_c) == 1:
        return np.ones(1)
    X_sub = obs_k[:, valid_c].astype(float)
    X_sub -= X_sub.mean(axis=0)
    if X_sub.std() < 1e-12:          # degenerate: all constant
        return np.ones(len(valid_c)) / np.sqrt(len(valid_c))
    _, _, Vt = np.linalg.svd(X_sub, full_matrices=False)
    return Vt[0]                     # (len(valid_c),)


QTE_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def _gmm_marginal_quantile_1d(weights, means_1d, vars_1d, tau):
    """
    Compute the τ-th quantile of a 1D Gaussian mixture by root-finding.

    Args:
        weights: (C,) mixture weights (sum to 1)
        means_1d: (C,) component means
        vars_1d: (C,) component variances
        tau: scalar in (0, 1)

    Returns:
        scalar: x such that CDF_gmm(x) = tau
    """
    from scipy.stats import norm as sp_norm
    from scipy.optimize import brentq

    stds = np.sqrt(np.maximum(vars_1d, 1e-12))
    lo = float(np.min(means_1d - 10.0 * stds))
    hi = float(np.max(means_1d + 10.0 * stds))

    def cdf_minus_tau(x):
        return float(np.sum(weights * sp_norm.cdf((x - means_1d) / stds))) - tau

    # Handle degenerate brackets (shouldn't happen with 10-sigma range)
    if cdf_minus_tau(lo) > 0:
        return lo
    if cdf_minus_tau(hi) < 0:
        return hi
    return brentq(cdf_minus_tau, lo, hi, xtol=1e-6)


def _oas_cov(X):
    """
    Oracle Approximating Shrinkage (OAS) covariance estimator (Chen et al. 2010).

    A closed-form shrinkage estimator that is robust with few samples relative
    to the number of variables. Shrinkage target is (tr(S)/p) * I.
    Works reliably with as few as 80 samples for ~8 variables.

    Args:
        X: (n_samples, n_vars) array of observations

    Returns:
        Sigma: (n_vars, n_vars) shrinkage covariance estimate
    """
    n, p = X.shape
    if n <= 1:
        return np.eye(p)
    X_c = X - X.mean(axis=0)
    S = X_c.T @ X_c / n  # biased MLE covariance
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    mu = tr_S / p  # shrinkage target scale
    denom = (n + 1 - 2.0 / p) * (tr_S2 - tr_S ** 2 / p)
    if abs(denom) < 1e-12:
        return S + 1e-6 * np.eye(p)
    rho = min(1.0, max(0.0, ((1 - 2.0 / p) * tr_S2 + tr_S ** 2) / denom))
    return (1.0 - rho) * S + rho * mu * np.eye(p)



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


def _gmm_nll_np(weights, means, covs, true_samples):
    """NLL of true_samples under a full-dimensional GMM."""
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)
    c = np.asarray(covs, dtype=np.float64)
    x = np.asarray(true_samples, dtype=np.float64)
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


def _marg_gmm_nll_np(weights, means, covs, true_samples, dims):
    """NLL of true_samples under the marginal of a GMM restricted to `dims`."""
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(means, dtype=np.float64)[:, dims]
    c_full = np.asarray(covs, dtype=np.float64)
    c = c_full[:, :, :][:, dims, :][:, :, dims]  # (K, |dims|, |dims|)
    x = np.asarray(true_samples, dtype=np.float64)[:, dims]
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


def _compute_mmd_np(X, Y, sigma=1.0):
    """MMD with RBF kernel between two sample sets (NumPy)."""
    def rbf_kernel(A, B, s):
        sq = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-sq / (2 * s ** 2))
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    n, m = len(X), len(Y)
    mmd = ((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
           + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
           - 2 * Kxy.mean())
    return float(mmd)


def _compute_wsd_np(X, Y):
    """Mean per-variable 1D Wasserstein distance (NumPy)."""
    from scipy.stats import wasserstein_distance as scipy_wasserstein
    n_vars = X.shape[1]
    return float(np.mean([scipy_wasserstein(X[:, d], Y[:, d]) for d in range(n_vars)]))


class GaussianConditionalBaseline:
    """Baseline: Fit observational Gaussian, use conditional distribution for ATE."""

    def __init__(self):
        self.mus = None
        self.sigmas = None
        self.v_j = None

    def fit(self, data, target_data, graphs, intervention_value: float = 5.0):
        """
        Fit multivariate Gaussians to observational data.

        Args:
            data: (n_scms, n_samples, n_vars) observational samples
            target_data: (n_scms, n_targets, n_samples, n_vars)
            graphs: (n_scms, n_vars, n_vars) adjacency matrices (unused here)
            intervention_value: fixed do(X_j = v) value
        """
        n_scms, n_samples, n_vars = data.shape
        n_targets = target_data.shape[1]
        self.mus = np.zeros((n_scms, n_vars))
        self.sigmas = np.zeros((n_scms, n_vars, n_vars))
        self.v_j = np.full((n_scms, n_targets, n_vars), intervention_value)

        for k in range(n_scms):
            mu, sigma = fit_multivariate_gaussian(data[k])
            self.mus[k] = np.array(mu)
            self.sigmas[k] = np.array(sigma)

    def _conditional_params(self, k, j):
        """Return (cond_mu, cond_Sigma, j_var) after conditioning X_{j_var} = v_j."""
        j_var = j  # variable index == target index for single-target interventions
        v_j = float(self.v_j[k, j, j_var])
        cond_mu, cond_Sigma = conditional_gaussian_params(self.mus[k], self.sigmas[k], j_var, v_j)
        return np.array(cond_mu), np.array(cond_Sigma), j_var

    def predict(self, n_scms, n_targets, n_vars):
        """Predict ATE using conditional Gaussian."""
        ate_pred = np.zeros((n_scms, n_targets, n_vars))

        for k in range(n_scms):
            obs_mean = self.mus[k]
            for j in range(n_targets):
                cond_mu, _, j_var = self._conditional_params(k, j)
                v_j = float(self.v_j[k, j, j_var])
                y_idx = 0
                for y in range(n_vars):
                    if y == j_var:
                        ate_pred[k, j, y] = v_j - obs_mean[y]
                    else:
                        ate_pred[k, j, y] = cond_mu[y_idx] - obs_mean[y]
                        y_idx += 1

        return ate_pred

    def predict_intvar(self, n_scms, n_targets, n_vars):
        """Predict interventional variance using diagonal of conditional Gaussian covariance."""
        intvar_pred = np.zeros((n_scms, n_targets, n_vars))

        for k in range(n_scms):
            for j in range(n_targets):
                _, cond_Sigma, j_var = self._conditional_params(k, j)
                y_idx = 0
                for y in range(n_vars):
                    if y == j_var:
                        intvar_pred[k, j, y] = 0.0  # intervened variable has zero variance
                    else:
                        intvar_pred[k, j, y] = float(cond_Sigma[y_idx, y_idx])
                        y_idx += 1

        return intvar_pred

    def predict_cate(self, obs_data, n_scms, n_targets, n_vars, graphs, targets_eval):
        """
        Predict CATE analytically using truncated Gaussian conditional expectations.

        Conditioning is on the first principal component Z = X_{valid_c} @ v of all
        valid pre-treatment covariates jointly (Wager & Athey 2018 style).

        For each (k, j, y):
          1. Condition on X_j = v_j (exact MVN step) → (n_vars-1)-dim Gaussian N(μ₁, Σ₁).
          2. Project valid pre-treatment covariates onto first PC direction v.
             Z ~ N(μ_z, σ_z²): μ_z = μ₁_{valid_c} @ v, σ_z² = v^T Σ₁_{cc} v.
             Cov(Y_y, Z) = Σ₁_{valid_c, y} @ v = σ_zy.
          3. Apply truncated normal formula with α = (m − μ_z)/σ_z, m = median of obs scores:
               E[Y_y | Z ≤ m] = μ_y − (σ_zy/σ_z) · φ(α)/Φ(α)
               E[Y_y | Z > m] = μ_y + (σ_zy/σ_z) · φ(α)/(1−Φ(α))
          4. CATE = E[Y_y | do(X_j=v), Z ∈ bin] − E[Y_y | obs].
        """
        from scipy.stats import norm

        cate_pred = np.zeros((n_scms, n_targets, n_vars, 2))

        for k in range(n_scms):
            adj = graphs[k]
            obs_k = obs_data[k]  # (n_samples, n_vars)

            for j in range(n_targets):
                intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
                if len(intervened_vars) == 0:
                    continue
                j_var = int(intervened_vars[0])
                v_j = float(self.v_j[k, j, j_var])
                desc_j = get_descendants(adj, j_var)

                # Condition on X_j = v_j once per (k, j) — exact MVN step
                mu1, Sigma1 = conditional_gaussian_params(self.mus[k], self.sigmas[k], j_var, v_j)
                mu1, Sigma1 = np.array(mu1), np.array(Sigma1)
                remaining = [i for i in range(n_vars) if i != j_var]  # original → reduced index

                for y in range(n_vars):
                    if y == j_var:
                        continue

                    valid_c = [c for c in range(n_vars)
                               if c != j_var and c not in desc_j and c != y]
                    if not valid_c:
                        continue

                    # First PC direction over valid pre-treatment covariates
                    v = _pc1_direction(obs_k, valid_c)
                    valid_c_r = [remaining.index(c) for c in valid_c]
                    y_r = remaining.index(y)

                    # Project onto PC1: Z ~ N(μ_z, σ_z²)
                    mu_z = float(mu1[valid_c_r] @ v)
                    Sigma_cc = Sigma1[np.ix_(valid_c_r, valid_c_r)]
                    sigma_z = float(np.sqrt(max(float(v @ Sigma_cc @ v), 1e-12)))
                    sigma_zy = float(Sigma1[valid_c_r, y_r] @ v)

                    obs_scores = obs_k[:, valid_c] @ v
                    median_z = float(np.median(obs_scores))

                    mu_y = float(mu1[y_r])
                    obs_mean_y = float(self.mus[k][y])

                    alpha = (median_z - mu_z) / sigma_z
                    phi_a = float(norm.pdf(alpha))
                    Phi_a = float(norm.cdf(alpha))
                    eps = 1e-12
                    coeff = sigma_zy / sigma_z

                    cate_pred[k, j, y, 0] = (mu_y - coeff * phi_a / (Phi_a + eps)) - obs_mean_y
                    cate_pred[k, j, y, 1] = (mu_y + coeff * phi_a / (1.0 - Phi_a + eps)) - obs_mean_y

        return cate_pred

    def predict_qte(self, obs_quantiles, quantiles, n_scms, n_targets, n_vars):
        """
        Predict QTE using the marginal Gaussian from the conditional distribution.

        Q_τ[Y_y | do(X_j=v)] = μ_cond_y + sqrt(σ²_cond_y) * Φ⁻¹(τ)
        QTE(τ) = Q_τ[predicted] - obs_quantiles[k, q, y]

        Returns:
            qte_pred: (n_scms, n_targets, n_quantiles, n_vars)
        """
        from scipy.stats import norm as sp_norm
        n_quantiles = len(quantiles)
        ppf_vals = np.array([sp_norm.ppf(tau) for tau in quantiles])  # (n_quantiles,)
        qte_pred = np.zeros((n_scms, n_targets, n_quantiles, n_vars))

        for k in range(n_scms):
            for j in range(n_targets):
                cond_mu, cond_Sigma, j_var = self._conditional_params(k, j)
                cond_std = np.sqrt(np.maximum(np.diag(np.array(cond_Sigma)), 0.0))

                # Build interventional quantiles for all vars
                intv_quants = np.zeros((n_quantiles, n_vars))
                y_idx = 0
                for y in range(n_vars):
                    if y == j_var:
                        # Intervened variable is fixed to v_j: all quantiles equal v_j
                        intv_quants[:, y] = float(self.v_j[k, j, j_var])
                    else:
                        intv_quants[:, y] = float(cond_mu[y_idx]) + cond_std[y_idx] * ppf_vals
                        y_idx += 1

                qte_pred[k, j] = intv_quants - obs_quantiles[k]  # (n_quantiles, n_vars)

        return qte_pred

    def predict_intcov(self, n_scms, n_targets, n_vars):
        """
        Predict full interventional covariance matrix from conditional Gaussian parameters.

        Under do(X_j=v), the joint interventional covariance is the conditional Gaussian
        covariance of X_{-j} | X_j=v, with the j-th row and column set to zero (intervened
        variable has zero variance).
        """
        intcov_pred = np.zeros((n_scms, n_targets, n_vars, n_vars))
        for k in range(n_scms):
            for j in range(n_targets):
                _, cond_Sigma, j_var = self._conditional_params(k, j)
                cond_Sigma = np.array(cond_Sigma)  # (n_vars-1, n_vars-1)
                remaining = [i for i in range(n_vars) if i != j_var]
                for r1, y1 in enumerate(remaining):
                    for r2, y2 in enumerate(remaining):
                        intcov_pred[k, j, y1, y2] = float(cond_Sigma[r1, r2])
        return intcov_pred

    def get_dist_params(self, k, j):
        """Return (cond_mu, cond_Sigma, j_var) for distribution metrics."""
        return self._conditional_params(k, j)

    def predict_dist_nll(self, k, j, true_samples, dims=None):
        """
        NLL of true_samples under the predicted conditional Gaussian, excluding the
        intervened variable j in all conditions (all, desc, non_desc).
        If dims is provided, only those dims (minus j) are evaluated.
        """
        cond_mu, cond_Sigma, j_var = self._conditional_params(k, j)
        n_vars = self.mus[k].shape[0]
        remaining = [d for d in range(n_vars) if d != j_var]
        if dims is None:
            dims_to_use = remaining
        else:
            dims_to_use = [d for d in dims if d in remaining]
        if not dims_to_use:
            return float('nan')
        r_idx = [remaining.index(d) for d in dims_to_use]
        mu_sub = np.array(cond_mu)[r_idx]
        cov_sub = np.array(cond_Sigma)[np.ix_(r_idx, r_idx)] + 1e-6 * np.eye(len(r_idx))
        return _mvn_nll_np(np.array(true_samples)[:, dims_to_use], mu_sub, cov_sub)


class LinearCausalBaseline:
    """Linear SCM baseline: fits linear mechanisms from observational data using the oracle graph."""

    def __init__(self):
        self.coeffs = None
        self.intercepts = None
        self.obs_means = None
        self.v_j = None
        self.graphs = None
        self.residual_vars = None  # list of dicts: residual_var[y] per SCM
        self.obs_data = None       # stored for CATE median computation

    def fit(self, data, target_data, graphs, intervention_value: float = 5.0):
        """
        Fit linear SCM: Y_y = W[y] @ X_parents(y) + b[y]

        Args:
            data: (n_scms, n_samples, n_vars)
            target_data: (n_scms, n_targets, n_samples, n_vars)
            graphs: (n_scms, n_vars, n_vars) adjacency matrices
            intervention_value: fixed do(X_j = v) value
        """
        n_scms, n_samples, n_vars = data.shape
        n_targets = target_data.shape[1]
        self.graphs = graphs
        self.obs_means = data.mean(axis=1)
        self.obs_data = [np.array(data[k]) for k in range(n_scms)]
        self.v_j = np.full((n_scms, n_targets, n_vars), intervention_value)
        self.coeffs = []
        self.intercepts = []
        self.residual_vars = []

        from scipy import linalg

        for k in range(n_scms):

            adj = np.array(graphs[k], dtype=int)
            topo_order = kahn_topological_sort(adj)

            coeff_k = {}
            intercept_k = {}
            residual_var_k = {}

            for y in topo_order:
                parents = [i for i in range(n_vars) if adj[i, y] == 1]

                if len(parents) == 0:
                    coeff_k[y] = np.zeros(0)
                    intercept_k[y] = float(self.obs_means[k, y])
                    residual_var_k[y] = float(np.var(data[k, :, y]))
                else:
                    data_k = np.array(data[k])
                    X_parents = data_k[:, parents]
                    if X_parents.ndim == 1:
                        X_parents = X_parents.reshape(-1, 1)
                    Y_y = data_k[:, y]
                    X_aug = np.column_stack([X_parents, np.ones(n_samples)])
                    try:
                        coef_aug = linalg.lstsq(X_aug, Y_y)[0]
                        coeff_k[y] = coef_aug[:-1]
                        intercept_k[y] = float(coef_aug[-1])
                        residuals = Y_y - X_aug @ coef_aug
                        residual_var_k[y] = float(np.var(residuals))
                    except Exception:
                        coeff_k[y] = np.zeros(len(parents))
                        intercept_k[y] = float(self.obs_means[k, y])
                        residual_var_k[y] = float(np.var(data[k, :, y]))

            self.coeffs.append((coeff_k, topo_order))
            self.intercepts.append(intercept_k)
            self.residual_vars.append(residual_var_k)

    def predict(self, n_scms, n_targets, n_vars):
        """Predict ATE by propagating intervention through linear SCM."""
        ate_pred = np.zeros((n_scms, n_targets, n_vars))

        for k in range(n_scms):
            coeff_k, topo_order = self.coeffs[k]
            intercept_k = self.intercepts[k]
            obs_mean = self.obs_means[k]

            for j in range(n_targets):
                v_j = float(self.v_j[k, j, j])

                E_do = np.copy(obs_mean)
                E_do[j] = v_j

                for y in topo_order:
                    if y == j:
                        E_do[y] = v_j
                    else:
                        parents = [i for i in range(n_vars) if self.graphs[k][i, y] == 1]
                        if len(parents) > 0:
                            E_do[y] = coeff_k[y] @ E_do[parents] + intercept_k[y]
                        else:
                            E_do[y] = intercept_k[y]

                ate_pred[k, j, :] = E_do - obs_mean

        return ate_pred

    def _build_W_matrix(self, k, n_vars):
        """
        Build full coefficient matrix W (n_vars, n_vars) where W[y, p] is the
        fitted OLS coefficient of parent p in the structural equation for y.
        """
        W = np.zeros((n_vars, n_vars))
        coeff_k, _ = self.coeffs[k]
        for y in range(n_vars):
            parents = [i for i in range(n_vars) if self.graphs[k][i, y] == 1]
            if parents and y in coeff_k and len(coeff_k[y]) == len(parents):
                for idx, p in enumerate(parents):
                    W[y, p] = coeff_k[y][idx]
        return W

    def _compute_interventional_covariance(self, k, j, n_vars):
        """
        Compute the full interventional covariance matrix under do(X_j = v).

        Uses the exact linear-SCM formula on the mutilated graph:
            Σ_do = (I - W_do)^{-1} D_do (I - W_do)^{-T}
        where W_do has row j zeroed (j has no parents in the mutilated graph)
        and D_do = diag(residual_var) with D_do[j,j] = 0 (j is fixed to v).
        """
        W = self._build_W_matrix(k, n_vars)
        W_do = W.copy()
        W_do[j, :] = 0.0  # cut all incoming edges to j

        D_do = np.array([
            self.residual_vars[k].get(i, 0.0) if i != j else 0.0
            for i in range(n_vars)
        ])

        I_minus_W = np.eye(n_vars) - W_do
        try:
            inv = np.linalg.inv(I_minus_W)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(I_minus_W)

        Sigma_do = inv @ np.diag(D_do) @ inv.T
        return Sigma_do

    def _compute_interventional_mean(self, k, j, n_vars):
        """Propagate do(X_j = v) through the fitted linear SCM to get E[X | do(X_j=v)]."""
        coeff_k, topo_order = self.coeffs[k]
        intercept_k = self.intercepts[k]
        v_j = float(self.v_j[k, j, j])

        E_do = np.copy(self.obs_means[k])
        for y in topo_order:
            if y == j:
                E_do[y] = v_j
            else:
                parents = [i for i in range(n_vars) if self.graphs[k][i, y] == 1]
                if parents:
                    E_do[y] = coeff_k[y] @ E_do[parents] + intercept_k[y]
                else:
                    E_do[y] = intercept_k[y]
        return E_do

    def predict_intvar(self, n_scms, n_targets, n_vars):
        """
        Predict interventional variance using the exact linear-SCM covariance formula.

        Computes Σ_do = (I - W_do)^{-1} D_do (I - W_do)^{-T} on the mutilated graph
        and returns diag(Σ_do). This correctly accounts for cross-covariances between
        parents that share common ancestors, which the naive independent-variance
        propagation misses.
        """
        intvar_pred = np.zeros((n_scms, n_targets, n_vars))
        for k in range(n_scms):
            for j in range(n_targets):
                Sigma_do = self._compute_interventional_covariance(k, j, n_vars)
                intvar_pred[k, j, :] = np.diag(Sigma_do)
        return intvar_pred

    def predict_cate(self, n_scms, n_targets, n_vars):
        """
        Predict CATE using truncated Gaussian conditionals from the interventional distribution.

        Under do(X_j=v), X ~ N(μ_do, Σ_do). Conditioning is on the first PC of all valid
        pre-treatment covariates jointly: Z = X_{valid_c} @ v.

        Z ~ N(μ_z, σ_z²): μ_z = μ_do_{valid_c} @ v, σ_z² = v^T Σ_do_{cc} v.
        Cov(Y_y, Z) = Σ_do_{valid_c, y} @ v = σ_zy.

            E[Y_y | Z ≤ m] = μ_do_y − (σ_zy/σ_z) · φ(α)/Φ(α)
            E[Y_y | Z > m] = μ_do_y + (σ_zy/σ_z) · φ(α)/(1−Φ(α))

        where α = (m − μ_z)/σ_z and m = median of Z scores on obs data.
        """
        from scipy.stats import norm

        cate_pred = np.zeros((n_scms, n_targets, n_vars, 2))

        for k in range(n_scms):
            adj = self.graphs[k]
            obs_mean = self.obs_means[k]
            obs_k = self.obs_data[k]

            for j in range(n_targets):
                j_var = j
                desc_j = get_descendants(adj, j_var)
                mu_do = self._compute_interventional_mean(k, j, n_vars)
                Sigma_do = self._compute_interventional_covariance(k, j, n_vars)

                for y in range(n_vars):
                    if y == j_var:
                        continue
                    valid_c = [c for c in range(n_vars)
                               if c != j_var and c not in desc_j and c != y]
                    if not valid_c:
                        continue

                    v = _pc1_direction(obs_k, valid_c)

                    mu_z = float(mu_do[valid_c] @ v)
                    Sigma_cc = Sigma_do[np.ix_(valid_c, valid_c)]
                    sigma_z = float(np.sqrt(max(float(v @ Sigma_cc @ v), 1e-12)))
                    sigma_zy = float(Sigma_do[valid_c, y] @ v)

                    obs_scores = obs_k[:, valid_c] @ v
                    median_z = float(np.median(obs_scores))

                    alpha = (median_z - mu_z) / sigma_z
                    phi_a = float(norm.pdf(alpha))
                    Phi_a = float(norm.cdf(alpha))
                    eps = 1e-12
                    coeff = sigma_zy / sigma_z

                    cate_pred[k, j, y, 0] = (float(mu_do[y]) - coeff * phi_a / (Phi_a + eps)) - float(obs_mean[y])
                    cate_pred[k, j, y, 1] = (float(mu_do[y]) + coeff * phi_a / (1.0 - Phi_a + eps)) - float(obs_mean[y])

        return cate_pred

    def predict_qte(self, obs_quantiles, quantiles, n_scms, n_targets, n_vars):
        """
        Predict QTE using the interventional Gaussian computed by the linear SCM.

        Under do(X_j=v), X ~ N(μ_do, Σ_do) (diagonal variance suffices for marginals).
        Q_τ[Y_y | do(X_j=v)] = μ_do_y + sqrt(Σ_do_yy) * Φ⁻¹(τ)
        QTE(τ) = Q_τ[predicted] - obs_quantiles[k, q, y]

        Returns:
            qte_pred: (n_scms, n_targets, n_quantiles, n_vars)
        """
        from scipy.stats import norm as sp_norm
        n_quantiles = len(quantiles)
        ppf_vals = np.array([sp_norm.ppf(tau) for tau in quantiles])  # (n_quantiles,)
        qte_pred = np.zeros((n_scms, n_targets, n_quantiles, n_vars))

        for k in range(n_scms):
            for j in range(n_targets):
                mu_do = self._compute_interventional_mean(k, j, n_vars)
                Sigma_do = self._compute_interventional_covariance(k, j, n_vars)
                std_do = np.sqrt(np.maximum(np.diag(Sigma_do), 0.0))  # (n_vars,)
                # intv_quants[q, y] = mu_do[y] + std_do[y] * ppf(tau_q)
                intv_quants = mu_do[None, :] + std_do[None, :] * ppf_vals[:, None]
                qte_pred[k, j] = intv_quants - obs_quantiles[k]  # (n_quantiles, n_vars)

        return qte_pred

    def predict_intcov(self, n_scms, n_targets, n_vars):
        """
        Predict full interventional covariance matrix using the exact linear-SCM formula.

        Delegates to the existing _compute_interventional_covariance which computes
        Σ_do = (I - W_do)^{-1} D_do (I - W_do)^{-T} on the mutilated graph.
        """
        intcov_pred = np.zeros((n_scms, n_targets, n_vars, n_vars))
        for k in range(n_scms):
            for j in range(n_targets):
                intcov_pred[k, j] = self._compute_interventional_covariance(k, j, n_vars)
        return intcov_pred

    def predict_dist_nll(self, k, j, true_samples, dims=None):
        """
        NLL of true_samples under N(mu_do, Sigma_do), excluding the intervened variable j
        in all conditions (all, desc, non_desc).
        If dims is provided, only those dims (minus j) are evaluated.
        """
        n_vars = len(self.obs_means[k])
        mu_do = self._compute_interventional_mean(k, j, n_vars)
        Sigma_do = self._compute_interventional_covariance(k, j, n_vars)
        j_var = j
        if dims is None:
            dims_to_use = [d for d in range(n_vars) if d != j_var]
        else:
            dims_to_use = [d for d in dims if d != j_var]
        if not dims_to_use:
            return float('nan')
        mu_sub = mu_do[dims_to_use]
        cov_sub = Sigma_do[np.ix_(dims_to_use, dims_to_use)] + 1e-6 * np.eye(len(dims_to_use))
        return _mvn_nll_np(np.array(true_samples)[:, dims_to_use], mu_sub, cov_sub)


class CVAEMethod:
    """CVAE inference method."""

    def __init__(self, model, state, n_model_samples, samples_from_prior):
        self.model = model
        self.state = state
        self.n_model_samples = n_model_samples
        self.samples_from_prior = samples_from_prior

    def _predict_raw(self, data, targets_data, n_eval_scms, n_targets, rng):
        """
        Run CVAE inference fully analytically; no Monte Carlo sampling.

        The model produces a two-level GMM for p(Y | do(X_j=v)):
          - Outer level: s = samples_from_prior latent samples z ~ p(z|intervention),
            each weighted uniformly (1/s).
          - Inner level: each z is decoded to a GMM with K components.
          - Combined: s*K components with weights flat_w[i,k] = (1/s) · dec_weights[i,k].

        IntMean and IntVar are computed exactly:
            E[Y]     = Σᵢₖ flat_w · μᵢₖ
            Var[Y_d] = Σᵢₖ flat_w · (σ²ᵢₖ_d + μ²ᵢₖ_d)  −  E[Y_d]²

        The flattened GMM parameters (flat_w, means, covs) are stored per (batch, j)
        for use in analytical CATE computation.

        Returns:
            intmean:    (n_eval_scms, n_targets, n_vars)
            intvar:     (n_eval_scms, n_targets, n_vars)
            gmm_params: dict[(batch_start, j)] →
                          (flat_w, means, covs) with shapes
                          (batch_size, s*K), (batch_size, s*K, n_vars),
                          (batch_size, s*K, n_vars, n_vars)
        """
        n_vars = data.shape[2]
        intmean = np.zeros((n_eval_scms, n_targets, n_vars))
        intvar = np.zeros((n_eval_scms, n_targets, n_vars))
        gmm_params = {}

        batch_size = min(32, n_eval_scms)

        for batch_start in range(0, n_eval_scms, batch_size):
            batch_end = min(batch_start + batch_size, n_eval_scms)
            batch_size_actual = batch_end - batch_start

            dm_batch = jnp.array(data[batch_start:batch_end])

            for j in range(n_targets):
                targets_batch = jnp.array(targets_data[batch_start:batch_end, 0, j, :])
                intervention_index = jnp.zeros(batch_size_actual, dtype=jnp.int32)

                rng, dk1, dk2, dk3 = jax.random.split(rng, 4)

                pri_weights, pri_means, pri_covs = self.model.apply(
                    {'params': self.state.params},
                    dm=dm_batch,
                    intervention_index=intervention_index,
                    intervention_targets=targets_batch,
                    is_training=False,
                    rngs={"dropout": dk1},
                    method=self.model.encode_prior,
                )

                prior_dist = self.model.apply(
                    {'params': self.state.params},
                    weights=pri_weights, means=pri_means, covs=pri_covs,
                    rngs={"dropout": dk2},
                    method=self.model.get_conditional_prior,
                )

                rng, sample_key = jax.random.split(rng)
                prior_samples = prior_dist.sample(
                    seed=sample_key, sample_shape=(self.samples_from_prior,)
                )  # (s, b, latent_dim)

                def decode_z(z):
                    return self.model.apply(
                        {'params': self.state.params},
                        z,
                        is_training=False,
                        rngs={"dropout": dk3},
                        method=self.model.decode,
                    )

                dec_weights, dec_means, dec_covs = jax.vmap(decode_z)(prior_samples)
                # dec_weights: (s, b, K)
                # dec_means:   (s, b, K, n_vars)
                # dec_covs:    (s, b, K, n_vars, n_vars)

                s = prior_samples.shape[0]

                # Rearrange to (b, s, K, ...)
                w = dec_weights.swapaxes(0, 1)    # (b, s, K)
                m = dec_means.swapaxes(0, 1)      # (b, s, K, n_vars)
                cov = dec_covs.swapaxes(0, 1)     # (b, s, K, n_vars, n_vars)
                cov_diag = jnp.diagonal(cov, axis1=-2, axis2=-1)  # (b, s, K, n_vars)

                # Combined weight: (1/s) * dec_weights  (outer is uniform over s)
                flat_w = w / s  # (b, s, K)

                # Analytical IntMean: E[Y] = Σᵢₖ flat_w · μᵢₖ
                E_Y = (flat_w[..., None] * m).sum(axis=(1, 2))  # (b, n_vars)

                # Analytical IntVar via law of total variance:
                # E[Y²_d] = Σᵢₖ flat_w · (σ²ᵢₖ_d + μ²ᵢₖ_d)
                E_Y2 = (flat_w[..., None] * (cov_diag + m ** 2)).sum(axis=(1, 2))
                Var_Y = E_Y2 - E_Y ** 2  # (b, n_vars)

                intmean[batch_start:batch_end, j, :] = np.array(E_Y)
                intvar[batch_start:batch_end, j, :] = np.array(Var_Y)

                # Store flattened GMM params for analytical CATE
                b = batch_size_actual
                flat_w_np = np.array(flat_w).reshape(b, -1)          # (b, s*K)
                m_np = np.array(m).reshape(b, -1, n_vars)             # (b, s*K, n_vars)
                cov_np = np.array(cov).reshape(b, -1, n_vars, n_vars) # (b, s*K, n_vars, n_vars)
                gmm_params[(batch_start, j)] = (flat_w_np, m_np, cov_np)

        return intmean, intvar, gmm_params

    def predict(self, data, targets_data, n_eval_scms, n_targets, rng):
        """Predict ATE = intmean - obs_mean."""
        intmean, _, _ = self._predict_raw(data, targets_data, n_eval_scms, n_targets, rng)
        obs_mean = data.mean(axis=1)
        return intmean - obs_mean[:, None, :]

    def predict_cate(self, data, targets_data, n_eval_scms, n_targets, n_vars,
                     graphs, targets_eval, gmm_params):
        """
        Predict CATE analytically from the decoded GMM using truncated Gaussian theory.

        Conditioning is on the first PC of all valid pre-treatment covariates jointly:
        Z = X_{valid_c} @ v.  For each GMM component i, Z ~ N(μ_z_i, σ_z_i²) where:
            μ_z_i  = μ_i[valid_c] @ v
            σ_z_i² = v^T Σ_i[valid_c, valid_c] v
            σ_zy_i = Σ_i[valid_c, y] @ v

        Mixture conditional expectation:
            E[Y_y | Z ≤ m] = Σ_i (w_i · Φ(αᵢ) · E[Y_y|comp=i, Z≤m])
                             ─────────────────────────────────────────
                                    Σ_i (w_i · Φ(αᵢ))

        CATE = E[Y_y | do(X_j=v), Z ∈ bin] − E[Y_y | obs].
        """
        from scipy.stats import norm

        batch_size = min(32, n_eval_scms)
        cate_pred = np.zeros((n_eval_scms, n_targets, n_vars, 2))
        obs_means = data.mean(axis=1)  # (n_eval_scms, n_vars)

        for k in range(n_eval_scms):
            adj = graphs[k]
            obs_k = data[k]

            batch_start = (k // batch_size) * batch_size
            k_in_batch = k - batch_start

            for j in range(n_targets):
                intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
                if len(intervened_vars) == 0:
                    continue
                j_var = int(intervened_vars[0])
                desc_j = get_descendants(adj, j_var)

                params = gmm_params.get((batch_start, j))
                if params is None:
                    continue
                flat_w_b, m_b, cov_b = params
                w   = flat_w_b[k_in_batch]  # (n_components,)
                mu  = m_b[k_in_batch]       # (n_components, n_vars)
                cov = cov_b[k_in_batch]     # (n_components, n_vars, n_vars)

                for y in range(n_vars):
                    if y == j_var:
                        continue
                    valid_c = [c for c in range(n_vars)
                               if c != j_var and c not in desc_j and c != y]
                    if not valid_c:
                        continue

                    # First PC direction over valid pre-treatment covariates (from obs data)
                    v = _pc1_direction(obs_k, valid_c)        # (len(valid_c),)

                    # Per-component projections onto PC1
                    mu_z   = mu[:, valid_c] @ v               # (K,)
                    mu_y   = mu[:, y]                         # (K,)
                    cov_cc = cov[:, valid_c, :][:, :, valid_c]  # (K, |c|, |c|)
                    sigma_z  = np.sqrt(np.maximum(
                        np.einsum('i,kij,j->k', v, cov_cc, v), 1e-12))              # (K,)
                    sigma_zy = cov[:, valid_c, y] @ v                               # (K,)

                    obs_scores = obs_k[:, valid_c] @ v
                    median_z = float(np.median(obs_scores))

                    alpha = (median_z - mu_z) / sigma_z       # (K,)
                    phi_a = norm.pdf(alpha)
                    Phi_a = norm.cdf(alpha)

                    eps = 1e-12
                    coeff = sigma_zy / sigma_z                 # per-component regression coeff

                    e_y_lo_comp = mu_y - coeff * phi_a / (Phi_a + eps)
                    e_y_hi_comp = mu_y + coeff * phi_a / (1.0 - Phi_a + eps)

                    w_lo = w * Phi_a
                    w_hi = w * (1.0 - Phi_a)
                    obs_mean_y = float(obs_means[k, y])

                    cate_pred[k, j, y, 0] = float((w_lo * e_y_lo_comp).sum() / (w_lo.sum() + eps)) - obs_mean_y
                    cate_pred[k, j, y, 1] = float((w_hi * e_y_hi_comp).sum() / (w_hi.sum() + eps)) - obs_mean_y

        return cate_pred

    def predict_intcov(self, gmm_params, n_eval_scms, n_targets, n_vars, batch_size=32):
        """
        Predict full interventional covariance matrix analytically from the decoded GMM.

        Uses the law of total covariance over the two-level mixture:
            Cov[X] = E[Cov[X|Z]] + Cov[E[X|Z]]
                   = Σ_i w_i Σ_i  +  Σ_i w_i μ_i μ_i^T  −  (Σ_i w_i μ_i)(Σ_i w_i μ_i)^T

        Args:
            gmm_params: dict[(batch_start, j)] → (flat_w, m, cov)
                        shapes (b, n_comp), (b, n_comp, n_vars), (b, n_comp, n_vars, n_vars)

        Returns:
            intcov_pred: (n_eval_scms, n_targets, n_vars, n_vars)
        """
        intcov_pred = np.zeros((n_eval_scms, n_targets, n_vars, n_vars))
        for batch_start in range(0, n_eval_scms, batch_size):
            batch_end = min(batch_start + batch_size, n_eval_scms)
            for j in range(n_targets):
                params = gmm_params.get((batch_start, j))
                if params is None:
                    continue
                flat_w, m, cov = params  # (b, n_comp), (b, n_comp, n_vars), (b, n_comp, n_vars, n_vars)
                flat_w = flat_w.astype(np.float32)
                m = m.astype(np.float32)
                cov = cov.astype(np.float32)
                E_Y = (flat_w[:, :, None] * m).sum(axis=1)               # (b, n_vars)
                E_cov = (flat_w[:, :, None, None] * cov).sum(axis=1)     # (b, n_vars, n_vars)
                mu_outer = np.einsum('bc,bci,bcj->bij', flat_w, m, m)    # (b, n_vars, n_vars)
                EY_outer = np.einsum('bi,bj->bij', E_Y, E_Y)             # (b, n_vars, n_vars)
                intcov_pred[batch_start:batch_end, j] = E_cov + mu_outer - EY_outer
        return intcov_pred


class MACETNPMethod:
    """
    MACE-TNP inference method (PyTorch-based), fully analytical.

    Loads a JointMoGCausalInferenceModel checkpoint from comparisons/ and calls
    forward() to obtain the K-component GMM (means, Cholesky L, weights) in
    normalized space, then denormalizes to original data space.

    IntMean and IntVar are computed exactly from the GMM parameters via the law
    of total expectation/variance — no Monte Carlo needed.  CATE uses the same
    per-component truncated Gaussian formula as CVAEMethod.predict_cate().
    """

    def __init__(self, model_path: str, intervention_value: float, device=None):
        import sys
        import os
        import torch

        comparisons_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparisons')
        if comparisons_dir not in sys.path:
            sys.path.insert(0, comparisons_dir)

        from CITNP.models.causalinferencemodel import CausalInfModelConfig
        from mace_tnp_model import JointMoGCausalInferenceModel

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.intervention_value = intervention_value
        self._torch = torch

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
        self.model = JointMoGCausalInferenceModel(config)
        state_dict = checkpoint['model_state_dict']
        # Strip _orig_mod. prefix added by torch.compile when saving
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def _get_gmm_params(self, obs_data_k, j_var):
        """
        Run forward() for one (SCM, intervention) pair and return GMM parameters
        in original (denormalized) data space.

        Args:
            obs_data_k: (n_obs, n_vars) numpy array — observational samples for SCM k
            j_var:      int — index of the intervened variable

        Returns:
            weights: (K,)           mixture weights
            means:   (K, n_vars)    component means in original space
            covs:    (K, n_vars, n_vars) full covariance matrices in original space
        """
        torch = self._torch
        obs_np = obs_data_k.astype(np.float32)
        n_obs, n_vars = obs_np.shape

        # Per-sample z-score normalization (matches training collate_fn)
        mean_np = obs_np.mean(axis=0)  # (n_vars,)
        std_np  = obs_np.std(axis=0) + 1e-8

        obs_norm = (obs_np - mean_np) / std_np  # (n_obs, n_vars)

        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, n_obs, n_vars)

        # Build dummy target: single token with only the (normalized) intervention value
        dummy_target = torch.zeros(1, 1, n_vars, dtype=torch.float32, device=self.device)
        v_norm = (self.intervention_value - float(mean_np[j_var])) / float(std_np[j_var])
        dummy_target[0, 0, j_var] = v_norm

        intvn_tensor = torch.tensor([j_var], dtype=torch.long, device=self.device)
        # outcome_index: pick smallest index ≠ j_var (API compatibility)
        out_idx = 1 if j_var == 0 else 0
        outcome_tensor = torch.tensor([out_idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(obs_tensor, dummy_target, intvn_tensor, outcome_tensor)

        # GMM parameters in normalized space
        weights_np = out.weights.squeeze(0).cpu().numpy()       # (K,)
        means_norm  = out.means.squeeze(0).cpu().numpy()        # (K, n_vars)
        L_norm      = out.L.squeeze(0).cpu().numpy()            # (K, n_vars, n_vars)

        # Denormalize: if Y_norm ~ N(μ_norm, Σ_norm)
        #   then Y_orig = Y_norm * std + mean ~ N(μ_norm * std + mean, diag(std) @ Σ_norm @ diag(std))
        means_orig = means_norm * std_np[None, :] + mean_np[None, :]  # (K, n_vars)

        # Σ_norm = L_norm @ L_norm^T;  Σ_orig = diag(std) @ Σ_norm @ diag(std)
        Sigma_norm = np.einsum('kij,klj->kil', L_norm, L_norm)  # (K, n_vars, n_vars)
        Sigma_orig = std_np[None, :, None] * Sigma_norm * std_np[None, None, :]  # (K, n_vars, n_vars)

        return weights_np, means_orig, Sigma_orig

    def _predict_raw(self, data, targets_eval, n_eval_scms, n_targets, n_vars):
        """
        Compute GMM-based IntMean and IntVar analytically for all (SCM, target) pairs.

        E[Y]   = Σ_k w_k · μ_k
        Var[Y_d] = Σ_k w_k · (Σ_kk_d + μ²_k_d) − E[Y_d]²   (law of total variance)

        Also stores GMM parameters per (k, j) for later use in predict_cate().

        Returns:
            intmean:    (n_eval_scms, n_targets, n_vars)
            intvar:     (n_eval_scms, n_targets, n_vars)
            gmm_params: dict[(k, j)] → (weights, means, covs)
                          shapes (K,), (K, n_vars), (K, n_vars, n_vars)
        """
        intmean = np.zeros((n_eval_scms, n_targets, n_vars))
        intvar  = np.zeros((n_eval_scms, n_targets, n_vars))
        gmm_params = {}

        for k in range(n_eval_scms):
            if k % 20 == 0:
                print(f"\n    SCM {k}/{n_eval_scms}", end="", flush=True)

            for j in range(n_targets):
                intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
                if len(intervened_vars) == 0:
                    continue
                j_var = int(intervened_vars[0])

                weights, means, covs = self._get_gmm_params(data[k], j_var)
                # means: (K, n_vars), covs: (K, n_vars, n_vars), weights: (K,)

                # Analytical IntMean
                E_Y = (weights[:, None] * means).sum(axis=0)  # (n_vars,)

                # Analytical IntVar via law of total variance
                cov_diag = np.diagonal(covs, axis1=1, axis2=2)  # (K, n_vars)
                E_Y2 = (weights[:, None] * (cov_diag + means ** 2)).sum(axis=0)
                Var_Y = E_Y2 - E_Y ** 2

                intmean[k, j] = E_Y
                intvar[k, j]  = Var_Y
                gmm_params[(k, j)] = (weights, means, covs)

        return intmean, intvar, gmm_params

    def predict_cate(self, data, gmm_params, n_eval_scms, n_targets, n_vars,
                     graphs, targets_eval):
        """
        Predict CATE analytically from the GMM using truncated Gaussian theory.

        Conditioning is on the first PC of all valid pre-treatment covariates jointly:
        Z = X_{valid_c} @ v.  Identical PC-projection approach to CVAEMethod.predict_cate().

        Returns:
            cate_pred: (n_eval_scms, n_targets, n_vars, 2)
        """
        from scipy.stats import norm

        cate_pred = np.zeros((n_eval_scms, n_targets, n_vars, 2))
        obs_means = data.mean(axis=1)  # (n_eval_scms, n_vars)

        for k in range(n_eval_scms):
            adj = graphs[k]
            obs_k = data[k]

            for j in range(n_targets):
                intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
                if len(intervened_vars) == 0:
                    continue
                j_var = int(intervened_vars[0])
                desc_j = get_descendants(adj, j_var)

                params = gmm_params.get((k, j))
                if params is None:
                    continue
                w, mu, cov = params  # (K,), (K, n_vars), (K, n_vars, n_vars)

                for y in range(n_vars):
                    if y == j_var:
                        continue
                    valid_c = [c for c in range(n_vars)
                               if c != j_var and c not in desc_j and c != y]
                    if not valid_c:
                        continue

                    v = _pc1_direction(obs_k, valid_c)        # (len(valid_c),)

                    mu_z   = mu[:, valid_c] @ v               # (K,)
                    mu_y   = mu[:, y]                         # (K,)
                    cov_cc = cov[:, valid_c, :][:, :, valid_c]  # (K, |c|, |c|)
                    sigma_z  = np.sqrt(np.maximum(
                        np.einsum('i,kij,j->k', v, cov_cc, v), 1e-12))              # (K,)
                    sigma_zy = cov[:, valid_c, y] @ v                               # (K,)

                    obs_scores = obs_k[:, valid_c] @ v
                    median_z = float(np.median(obs_scores))

                    alpha = (median_z - mu_z) / sigma_z
                    phi_a = norm.pdf(alpha)
                    Phi_a = norm.cdf(alpha)

                    eps = 1e-12
                    coeff = sigma_zy / sigma_z

                    e_y_lo_comp = mu_y - coeff * phi_a / (Phi_a + eps)
                    e_y_hi_comp = mu_y + coeff * phi_a / (1.0 - Phi_a + eps)

                    w_lo = w * Phi_a
                    w_hi = w * (1.0 - Phi_a)
                    obs_mean_y = float(obs_means[k, y])

                    cate_pred[k, j, y, 0] = float((w_lo * e_y_lo_comp).sum() / (w_lo.sum() + eps)) - obs_mean_y
                    cate_pred[k, j, y, 1] = float((w_hi * e_y_hi_comp).sum() / (w_hi.sum() + eps)) - obs_mean_y

        return cate_pred

    def predict_intcov(self, gmm_params, n_eval_scms, n_targets, n_vars):
        """
        Predict full interventional covariance matrix analytically from the MACE-TNP GMM.

        Law of total covariance:
            Cov[X] = Σ_i w_i Σ_i  +  Σ_i w_i μ_i μ_i^T  −  E[X] E[X]^T

        Args:
            gmm_params: dict[(k, j)] → (weights, means, covs)
                        shapes (K,), (K, n_vars), (K, n_vars, n_vars)

        Returns:
            intcov_pred: (n_eval_scms, n_targets, n_vars, n_vars)
        """
        intcov_pred = np.zeros((n_eval_scms, n_targets, n_vars, n_vars))
        for k in range(n_eval_scms):
            for j in range(n_targets):
                params = gmm_params.get((k, j))
                if params is None:
                    continue
                weights, means, covs = params  # (K,), (K, n_vars), (K, n_vars, n_vars)
                E_Y = (weights[:, None] * means).sum(axis=0)            # (n_vars,)
                E_cov = (weights[:, None, None] * covs).sum(axis=0)     # (n_vars, n_vars)
                mu_outer = np.einsum('k,ki,kj->ij', weights, means, means)  # (n_vars, n_vars)
                EY_outer = np.outer(E_Y, E_Y)                           # (n_vars, n_vars)
                intcov_pred[k, j] = E_cov + mu_outer - EY_outer
        return intcov_pred


def _sample_from_gmm(weights, means, covs, n_samples, rng):
    """
    Draw samples from a Gaussian mixture model.

    Args:
        weights: (K,) mixture weights (sum to 1)
        means:   (K, n_vars) component means
        covs:    (K, n_vars, n_vars) component covariances
        n_samples: number of samples to draw
        rng: numpy default_rng instance

    Returns:
        (n_samples, n_vars) array of samples
    """
    K, n_vars = means.shape
    w = np.array(weights, dtype=np.float64)
    w = np.clip(w, 0, None)
    w /= w.sum()
    comp_ids = rng.choice(K, size=n_samples, p=w)
    samples = np.zeros((n_samples, n_vars))
    for k_idx in range(K):
        mask = comp_ids == k_idx
        n_k = mask.sum()
        if n_k > 0:
            cov_k = covs[k_idx]
            # Ensure positive-definite by adding a small jitter
            cov_k = cov_k + 1e-6 * np.eye(n_vars)
            samples[mask] = rng.multivariate_normal(means[k_idx], cov_k, size=n_k)
    return samples


def _sample_gauss_conditional(gauss_baseline, k, j, n_samples, rng):
    """
    Sample from the Gaussian conditional distribution for SCM k, intervention j.

    Returns (n_samples, n_vars) in original data space.
    """
    n_vars = gauss_baseline.mus[k].shape[0]
    j_var = j
    v_j = float(gauss_baseline.v_j[k, j, j_var])
    cond_mu, cond_Sigma = conditional_gaussian_params(
        gauss_baseline.mus[k], gauss_baseline.sigmas[k], j_var, v_j
    )
    cond_mu = np.array(cond_mu)
    cond_Sigma = np.array(cond_Sigma) + 1e-6 * np.eye(n_vars - 1)
    samples_reduced = rng.multivariate_normal(cond_mu, cond_Sigma, size=n_samples)
    samples = np.zeros((n_samples, n_vars))
    idx = 0
    for d in range(n_vars):
        if d == j_var:
            samples[:, d] = v_j
        else:
            samples[:, d] = samples_reduced[:, idx]
            idx += 1
    return samples


def _sample_from_linear_baseline(linear_baseline, k, j, n_samples, rng):
    """
    Sample from the interventional distribution implied by the fitted linear SCM.

    Under do(X_j = v), X ~ N(mu_do, Sigma_do) with the j-th variable fixed to v_j.
    Returns (n_samples, n_vars) array.
    """
    n_vars = len(linear_baseline.obs_means[k])
    mu_do = linear_baseline._compute_interventional_mean(k, j, n_vars)
    Sigma_do = linear_baseline._compute_interventional_covariance(k, j, n_vars)
    Sigma_do = Sigma_do + 1e-6 * np.eye(n_vars)
    samples = rng.multivariate_normal(mu_do, Sigma_do, size=n_samples)
    v_j = float(linear_baseline.v_j[k, j, j])
    samples[:, j] = v_j
    return samples


def _cvae_top_component_predictions(cvae_gmm_params, n_eval_scms, n_targets, n_vars, batch_size,
                                     top_k=1):
    """
    Extract IntMean, IntVar, and reduced GMM params from the CVAE mixture using only the
    top-k highest-weight components per (SCM, target) pair (weights renormalized).

    Returns:
        intmean:        (n_eval_scms, n_targets, n_vars)
        intvar:         (n_eval_scms, n_targets, n_vars)
        top_gmm_params: same key structure as cvae_gmm_params but each entry has k components
                        shapes (b, k), (b, k, n_vars), (b, k, n_vars, n_vars)
    """
    intmean = np.zeros((n_eval_scms, n_targets, n_vars))
    intvar = np.zeros((n_eval_scms, n_targets, n_vars))
    top_gmm_params = {}

    for batch_start in range(0, n_eval_scms, batch_size):
        batch_end = min(batch_start + batch_size, n_eval_scms)
        b = batch_end - batch_start

        for j in range(n_targets):
            params = cvae_gmm_params.get((batch_start, j))
            if params is None:
                continue
            flat_w, m, cov = params  # (b, s*K), (b, s*K, n_vars), (b, s*K, n_vars, n_vars)

            n_comp = flat_w.shape[1]
            k = min(top_k, n_comp)
            # top-k indices per row (descending order of weight)
            top_idx = np.argsort(flat_w, axis=1)[:, -k:]  # (b, k)

            rows = np.arange(b)[:, None]  # (b, 1) for fancy indexing
            sel_w = flat_w[rows, top_idx]                        # (b, k)
            sel_m = m[rows, top_idx, :]                          # (b, k, n_vars)
            sel_cov = cov[rows, top_idx, :, :]                   # (b, k, n_vars, n_vars)

            # Renormalize weights
            sel_w = sel_w / (sel_w.sum(axis=1, keepdims=True) + 1e-12)

            # IntMean and IntVar from the reduced mixture
            E_Y = (sel_w[:, :, None] * sel_m).sum(axis=1)                          # (b, n_vars)
            cov_diag = np.diagonal(sel_cov, axis1=-2, axis2=-1)                    # (b, k, n_vars)
            E_Y2 = (sel_w[:, :, None] * (cov_diag + sel_m ** 2)).sum(axis=1)
            Var_Y = E_Y2 - E_Y ** 2

            intmean[batch_start:batch_end, j, :] = E_Y
            intvar[batch_start:batch_end, j, :] = Var_Y

            top_gmm_params[(batch_start, j)] = (sel_w, sel_m, sel_cov)

    return intmean, intvar, top_gmm_params


def _mace_top_component_predictions(mace_gmm_params, n_eval_scms, n_targets, n_vars, top_k=1):
    """
    Extract IntMean, IntVar, and reduced GMM params from the MACE-TNP mixture using only
    the top-k highest-weight components per (SCM, target) pair (weights renormalized).

    Returns:
        intmean:        (n_eval_scms, n_targets, n_vars)
        intvar:         (n_eval_scms, n_targets, n_vars)
        top_gmm_params: same key structure as mace_gmm_params but with k components
                        shapes (k,), (k, n_vars), (k, n_vars, n_vars)
    """
    intmean = np.zeros((n_eval_scms, n_targets, n_vars))
    intvar = np.zeros((n_eval_scms, n_targets, n_vars))
    top_gmm_params = {}

    for k in range(n_eval_scms):
        for j in range(n_targets):
            params = mace_gmm_params.get((k, j))
            if params is None:
                continue
            weights, means, covs = params  # (K,), (K, n_vars), (K, n_vars, n_vars)

            n_comp = len(weights)
            kk = min(top_k, n_comp)
            top_idx = np.argsort(weights)[-kk:]  # (kk,) ascending → last kk are highest

            sel_w = weights[top_idx]
            sel_w = sel_w / (sel_w.sum() + 1e-12)
            sel_m = means[top_idx, :]        # (kk, n_vars)
            sel_cov = covs[top_idx, :, :]    # (kk, n_vars, n_vars)

            # IntMean and IntVar from the reduced mixture
            E_Y = (sel_w[:, None] * sel_m).sum(axis=0)
            cov_diag = np.diagonal(sel_cov, axis1=-2, axis2=-1)  # (kk, n_vars)
            E_Y2 = (sel_w[:, None] * (cov_diag + sel_m ** 2)).sum(axis=0)
            Var_Y = E_Y2 - E_Y ** 2

            intmean[k, j] = E_Y
            intvar[k, j] = Var_Y

            top_gmm_params[(k, j)] = (sel_w, sel_m, sel_cov)

    return intmean, intvar, top_gmm_params


def _cvae_best_component_ate(cvae_gmm_params, intmean_gt, obs_mean,
                              n_eval_scms, n_targets, n_vars, batch_size=32):
    """
    Oracle ATE variant: for each (k, j), select the GMM component whose mean is
    closest (L2 distance) to the true interventional mean, then compute ATE
    purely from that single component's mean.

    This is an oracle metric — it uses ground-truth information to pick the
    best component — and measures the ceiling performance achievable by mode selection.

    Args:
        cvae_gmm_params: dict[(batch_start, j)] → (flat_w, means, covs)
        intmean_gt: (n_eval_scms, n_targets, n_vars) true interventional means
        obs_mean:   (n_eval_scms, n_vars) observational means

    Returns:
        ate_pred: (n_eval_scms, n_targets, n_vars)
    """
    ate_pred = np.zeros((n_eval_scms, n_targets, n_vars))

    for batch_start in range(0, n_eval_scms, batch_size):
        batch_end = min(batch_start + batch_size, n_eval_scms)

        for j in range(n_targets):
            params = cvae_gmm_params.get((batch_start, j))
            if params is None:
                continue
            _, m, _ = params  # (b, n_comp, n_vars)

            for bi, k in enumerate(range(batch_start, batch_end)):
                mu = m[bi]                         # (n_comp, n_vars)
                true_mean = intmean_gt[k, j]       # (n_vars,)
                dists = np.sum((mu - true_mean[None, :]) ** 2, axis=1)  # (n_comp,)
                best_idx = int(np.argmin(dists))
                ate_pred[k, j] = mu[best_idx] - obs_mean[k]

    return ate_pred


def _mace_best_component_ate(mace_gmm_params, intmean_gt, obs_mean,
                              n_eval_scms, n_targets, n_vars):
    """
    Oracle ATE variant: for each (k, j), select the GMM component whose mean is
    closest (L2 distance) to the true interventional mean, then compute ATE
    purely from that single component's mean.

    Args:
        mace_gmm_params: dict[(k, j)] → (weights, means, covs)
        intmean_gt: (n_eval_scms, n_targets, n_vars) true interventional means
        obs_mean:   (n_eval_scms, n_vars) observational means

    Returns:
        ate_pred: (n_eval_scms, n_targets, n_vars)
    """
    ate_pred = np.zeros((n_eval_scms, n_targets, n_vars))

    for k in range(n_eval_scms):
        for j in range(n_targets):
            params = mace_gmm_params.get((k, j))
            if params is None:
                continue
            _, means, _ = params               # (K, n_vars)
            true_mean = intmean_gt[k, j]       # (n_vars,)
            dists = np.sum((means - true_mean[None, :]) ** 2, axis=1)  # (K,)
            best_idx = int(np.argmin(dists))
            ate_pred[k, j] = means[best_idx] - obs_mean[k]

    return ate_pred


def predict_qte_from_cvae_gmm(cvae_gmm_params, obs_quantiles, quantiles,
                               n_eval_scms, n_targets, n_vars, batch_size=32):
    """
    Predict QTE(τ) for each (k, j, y) by computing marginal quantiles of the CVAE GMM.

    CVAE gmm_params key: (batch_start, j) → (flat_w, means, covs)
        flat_w: (batch_size, C)
        means:  (batch_size, C, n_vars)
        covs:   (batch_size, C, n_vars, n_vars)

    Returns:
        qte_pred: (n_eval_scms, n_targets, n_quantiles, n_vars)
    """
    n_quantiles = len(quantiles)
    qte_pred = np.zeros((n_eval_scms, n_targets, n_quantiles, n_vars))

    for batch_start in range(0, n_eval_scms, batch_size):
        batch_end = min(batch_start + batch_size, n_eval_scms)
        for j in range(n_targets):
            params = cvae_gmm_params.get((batch_start, j))
            if params is None:
                continue
            flat_w, m_batch, cov_batch = params
            # flat_w: (b, C), m_batch: (b, C, n_vars), cov_batch: (b, C, n_vars, n_vars)
            for bi, k in enumerate(range(batch_start, batch_end)):
                weights = flat_w[bi]
                means = m_batch[bi]   # (C, n_vars)
                cov_diag = np.diagonal(cov_batch[bi], axis1=-2, axis2=-1)  # (C, n_vars)
                for y in range(n_vars):
                    for qi, tau in enumerate(quantiles):
                        q_val = _gmm_marginal_quantile_1d(
                            weights, means[:, y], cov_diag[:, y], tau
                        )
                        qte_pred[k, j, qi, y] = q_val - obs_quantiles[k, qi, y]

    return qte_pred


def predict_qte_from_mace_gmm(mace_gmm_params, obs_quantiles, quantiles,
                               n_eval_scms, n_targets, n_vars):
    """
    Predict QTE(τ) for each (k, j, y) by computing marginal quantiles of the MACE-TNP GMM.

    MACE gmm_params key: (k, j) → (weights, means, covs)
        weights: (K,)
        means:   (K, n_vars)
        covs:    (K, n_vars, n_vars)

    Returns:
        qte_pred: (n_eval_scms, n_targets, n_quantiles, n_vars)
    """
    n_quantiles = len(quantiles)
    qte_pred = np.zeros((n_eval_scms, n_targets, n_quantiles, n_vars))

    for k in range(n_eval_scms):
        for j in range(n_targets):
            params = mace_gmm_params.get((k, j))
            if params is None:
                continue
            weights, means, covs = params
            cov_diag = np.diagonal(covs, axis1=-2, axis2=-1)  # (K, n_vars)
            for y in range(n_vars):
                for qi, tau in enumerate(quantiles):
                    q_val = _gmm_marginal_quantile_1d(
                        weights, means[:, y], cov_diag[:, y], tau
                    )
                    qte_pred[k, j, qi, y] = q_val - obs_quantiles[k, qi, y]

    return qte_pred


def plot_distribution_comparison(
    input_eval,
    target_eval,
    targets_eval,
    gauss_baseline,
    linear_baseline,
    mace_method,
    mace_gmm_params,
    cvae_gmm_params,
    mace_top_gmm_params,
    cvae_top_gmm_params,
    n_vars,
    n_eval_scms,
    intervention_value=5.0,
    n_plot_scms=4,
    n_samples=500,
    seed=0,
    save_path=None,
    top_k_label='top-1',
):
    """
    Plot a grid comparing interventional distributions across methods.

    Layout: rows = SCMs, columns = [Real, Linear Causal, Gaussian Conditional, MACE-TNP*, ACTIVA (ours)]
    (* MACE-TNP columns omitted if mace_method is None)

    Each cell shows a (var0, var1) scatter with up to 3 overlaid distributions:
      - Observational samples (gray, only in Real row)
      - do(X0 = v) samples (blue)
      - do(X1 = v) samples (orange)

    Returns the matplotlib Figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)

    n_plot = min(n_plot_scms, n_eval_scms)
    plot_indices = rng.choice(n_eval_scms, size=n_plot, replace=False)

    methods = ['Real', 'Linear Causal', 'Gaussian Cond.']
    if mace_method is not None:
        methods.append('MACE-TNP')
        if top_k_label is not None:
            methods.append(f'MACE-TNP {top_k_label}')
    methods.append('ACTIVA (ours)')
    if top_k_label is not None:
        methods.append(f'ACTIVA {top_k_label}')
    n_methods = len(methods)
    n_targets = target_eval.shape[1]

    # Dynamic color palette: one color per intervention target.
    # Skip tab10 index 7 (gray) to avoid clashing with the observational samples color.
    _cmap = plt.cm.get_cmap('tab10', 10)
    _non_gray_indices = [i for i in range(10) if i != 7]
    INT_COLORS = [(_cmap(_non_gray_indices[j % len(_non_gray_indices)]),
                   f'$\\mathrm{{do}}(V_{{{j}}})$') for j in range(n_targets)]

    # Layout: rows = SCMs, columns = methods
    fig, axes = plt.subplots(n_plot, n_methods, figsize=(3.2 * n_methods, 3.0 * n_plot),
                             squeeze=False)

    cvae_batch_size = min(32, n_eval_scms)

    for row, k in enumerate(plot_indices):
        obs_samples = np.array(input_eval[k])  # (n_obs, n_vars)
        real_ints = [np.array(target_eval[k, j]) for j in range(n_targets)]

        # ---------- collect samples per method ----------
        method_samples = {}

        # Real
        method_samples['Real'] = {'obs': obs_samples,
                                  **{f'int_{j}': real_ints[j] for j in range(n_targets)}}

        # Linear Causal (oracle)
        lc_ints = [_sample_from_linear_baseline(linear_baseline, k, j, n_samples, rng)
                   for j in range(n_targets)]
        method_samples['Linear Causal'] = {'obs': obs_samples,
                                           **{f'int_{j}': lc_ints[j] for j in range(n_targets)}}

        # Gaussian Conditional
        gc_ints = [_sample_gauss_conditional(gauss_baseline, k, j, n_samples, rng)
                   for j in range(n_targets)]
        method_samples['Gaussian Cond.'] = {'obs': obs_samples,
                                            **{f'int_{j}': gc_ints[j] for j in range(n_targets)}}

        # MACE-TNP
        if mace_method is not None:
            mace_ints = []
            for j in range(n_targets):
                mp = mace_gmm_params.get((k, j))
                mace_ints.append(_sample_from_gmm(*mp, n_samples, rng) if mp is not None else None)
            method_samples['MACE-TNP'] = {'obs': obs_samples,
                                          **{f'int_{j}': mace_ints[j] for j in range(n_targets)}}

            if top_k_label is not None:
                mace_top_ints = []
                for j in range(n_targets):
                    mtp = mace_top_gmm_params.get((k, j)) if mace_top_gmm_params else None
                    mace_top_ints.append(_sample_from_gmm(*mtp, n_samples, rng) if mtp is not None else None)
                method_samples[f'MACE-TNP {top_k_label}'] = {
                    'obs': obs_samples,
                    **{f'int_{j}': mace_top_ints[j] for j in range(n_targets)}}

        # ACTIVA
        batch_start = (k // cvae_batch_size) * cvae_batch_size
        k_in_batch  = k - batch_start

        def _cvae_sample(params, k_in_b):
            if params is None:
                return None
            flat_w, m, cov = params
            w  = flat_w[k_in_b]         # (s*K,)
            mu = m[k_in_b]              # (s*K, n_vars)
            cv = cov[k_in_b]            # (s*K, n_vars, n_vars)
            return _sample_from_gmm(w, mu, cv, n_samples, rng)

        cvae_ints = []
        for j in range(n_targets):
            cp = cvae_gmm_params.get((batch_start, j))
            cvae_ints.append(_cvae_sample(cp, k_in_batch))
        method_samples['ACTIVA (ours)'] = {'obs': obs_samples,
                                           **{f'int_{j}': cvae_ints[j] for j in range(n_targets)}}

        if top_k_label is not None:
            cvae_top_ints = []
            for j in range(n_targets):
                ctp = cvae_top_gmm_params.get((batch_start, j)) if cvae_top_gmm_params else None
                cvae_top_ints.append(_cvae_sample(ctp, k_in_batch))
            method_samples[f'ACTIVA {top_k_label}'] = {
                'obs': obs_samples,
                **{f'int_{j}': cvae_top_ints[j] for j in range(n_targets)}}

        # ---------- determine common axis limits across all methods for this SCM ----------
        all_x, all_y = [], []
        for method_label in methods:
            for key, pts in method_samples[method_label].items():
                if pts is not None and len(pts) > 0:
                    all_x.append(pts[:, 0])
                    all_y.append(pts[:, 1])
        if all_x:
            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            x_lo, x_hi = np.percentile(x_all, [1, 99])
            y_lo, y_hi = np.percentile(y_all, [1, 99])
            x_margin = (x_hi - x_lo) * 0.1 + 0.3
            y_margin = (y_hi - y_lo) * 0.1 + 0.3
            xlim = (x_lo - x_margin, x_hi + x_margin)
            ylim = (y_lo - y_margin, y_hi + y_margin)
        else:
            xlim = ylim = (-5, 5)

        # ---------- draw each method column ----------
        for col_idx, method_label in enumerate(methods):
            ax = axes[row, col_idx]
            sdata = method_samples[method_label]

            # Observational samples (gray)
            pts = sdata.get('obs')
            if pts is not None and len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=4, alpha=0.35, color='gray', rasterized=True, label='Obs.')

            # All intervention targets with distinct colors
            for j, (color, label) in enumerate(INT_COLORS):
                pts = sdata.get(f'int_{j}')
                if pts is not None and len(pts) > 0:
                    ax.scatter(pts[:, 0], pts[:, 1],
                               s=4, alpha=0.35, color=color, rasterized=True, label=label)

            ax.set(xlim=xlim, ylim=ylim)
            ax.set_xlabel('$V_0$', fontsize=10)
            ax.set_ylabel('$V_1$', fontsize=10)
            ax.tick_params(labelsize=8)

            if row == 0:
                ax.set_title(method_label, fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'SCM {k}\n$V_1$', fontsize=10)

    # Add legend to first cell
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='Obs.')
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=6, label=lbl)
        for c, lbl in INT_COLORS
    ]
    axes[0, 0].legend(handles=handles, fontsize=6, loc='upper left')

    fig.suptitle(
        f'Interventional Distribution Comparison  ($V_0$ vs $V_1$ projection)\n'
        f'{n_plot} randomly sampled SCMs',
        fontsize=12,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"\nDistribution comparison plot saved to: {save_path}")

    return fig


def compute_pointwise_metrics(gt, pred, desc_mask, valid_mask, valid_mask_full=None,
                              non_desc_vs_zero=False):
    """
    Compute pointwise metrics (MAE, RMSE) on different subsets.

    Args:
        gt: (n_scms, n_targets, n_vars) ground truth values
        pred: (n_scms, n_targets, n_vars) predicted values
        desc_mask: (n_scms, n_targets, n_vars) 1 if y is descendant of intervened var
        valid_mask: (n_scms, n_targets, n_vars) 0 for pairs to exclude (excludes V_i)
        valid_mask_full: (n_scms, n_targets, n_vars) mask including V_i; defaults to all-True
        non_desc_vs_zero: if True, non-descendant MAE is computed against zero (ATE case)

    Returns:
        dict with metrics. ``_all`` includes the intervened variable; ``_all_vi`` excludes it.
    """
    if valid_mask_full is None:
        valid_mask_full = np.ones_like(valid_mask, dtype=bool)
    error = np.abs(pred - gt)
    error_sq = (pred - gt) ** 2

    metrics = {}

    # All variables including the intervened variable (V_i)
    metrics['MAE_all'] = np.mean(error[valid_mask_full])
    metrics['RMSE_all'] = np.sqrt(np.mean(error_sq[valid_mask_full]))

    # All variables excluding the intervened variable (All/V_i)
    metrics['MAE_all_vi'] = np.mean(error[valid_mask])
    metrics['RMSE_all_vi'] = np.sqrt(np.mean(error_sq[valid_mask]))

    desc_and_valid = desc_mask & valid_mask
    if desc_and_valid.sum() > 0:
        metrics['MAE_descendants'] = np.mean(error[desc_and_valid])
        metrics['RMSE_descendants'] = np.sqrt(np.mean(error_sq[desc_and_valid]))
    else:
        metrics['MAE_descendants'] = np.nan
        metrics['RMSE_descendants'] = np.nan

    non_desc_and_valid = (~desc_mask) & valid_mask
    if non_desc_and_valid.sum() > 0:
        if non_desc_vs_zero:
            # Evaluate predicted magnitude (true value is 0 for ATE of non-descendants).
            metrics['MAE_non_descendants'] = np.mean(np.abs(pred[non_desc_and_valid]))
            metrics['MAE_non_desc_noise_floor'] = np.mean(np.abs(gt[non_desc_and_valid]))
        else:
            metrics['MAE_non_descendants'] = np.mean(error[non_desc_and_valid])
            metrics['RMSE_non_descendants'] = np.sqrt(np.mean(error_sq[non_desc_and_valid]))
    else:
        metrics['MAE_non_descendants'] = np.nan
        if non_desc_vs_zero:
            metrics['MAE_non_desc_noise_floor'] = np.nan
        else:
            metrics['RMSE_non_descendants'] = np.nan

    return metrics


def compute_metrics_with_ci(gt, pred, desc_mask, valid_mask, n_bootstrap=1000, seed=0,
                             non_desc_vs_zero=False, valid_mask_full=None):
    """Compute pointwise metrics with 95% bootstrap confidence intervals."""
    metrics = compute_pointwise_metrics(gt, pred, desc_mask, valid_mask, valid_mask_full,
                                        non_desc_vs_zero)

    n_scms = gt.shape[0]
    rng = np.random.default_rng(seed)

    bootstrap_samples = {k: [] for k in metrics}
    for _ in range(n_bootstrap):
        idx = rng.choice(n_scms, size=n_scms, replace=True)
        vmf_idx = valid_mask_full[idx] if valid_mask_full is not None else None
        boot = compute_pointwise_metrics(gt[idx], pred[idx], desc_mask[idx], valid_mask[idx],
                                         vmf_idx, non_desc_vs_zero)
        for k, v in boot.items():
            bootstrap_samples[k].append(v)

    ci_metrics = {}
    for k, vals in bootstrap_samples.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            ci_metrics[f'{k}_ci_lower'] = np.percentile(finite, 2.5)
            ci_metrics[f'{k}_ci_upper'] = np.percentile(finite, 97.5)
        else:
            ci_metrics[f'{k}_ci_lower'] = np.nan
            ci_metrics[f'{k}_ci_upper'] = np.nan

    return {**metrics, **ci_metrics}


def compute_cate_ground_truth(obs_data, target_eval, graphs, targets_eval,
                               n_eval_scms, n_targets, n_vars):
    """
    Compute CATE ground truth from held-out interventional samples.

    For each (k, j, y) pair: find all valid pre-treatment covariates (not j_var,
    not descendants of j_var, not y), compute first PC direction v from obs data,
    project interventional samples onto Z = X_{valid_c} @ v, median-split on Z,
    and compute E[Y_y | Z ∈ bin] − obs_mean[y] in each stratum.

    This multivariate conditioning (via PC projection) is consistent with the
    standard CATE literature (Wager & Athey 2018).

    Returns:
        cate_gt: (n_eval_scms, n_targets, n_vars, 2)  last dim = [lo, hi] strata
    """
    cate_gt = np.zeros((n_eval_scms, n_targets, n_vars, 2))

    for k in range(n_eval_scms):
        adj = graphs[k]
        obs_k = obs_data[k]  # (n_obs_samples, n_vars)
        obs_mean_k = obs_k.mean(axis=0)

        for j in range(n_targets):
            intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
            if len(intervened_vars) == 0:
                continue
            j_var = int(intervened_vars[0])
            desc_j = get_descendants(adj, j_var)

            intv_samples = target_eval[k, j, :, :]  # (n_intv_samples, n_vars)

            for y in range(n_vars):
                if y == j_var:
                    continue

                valid_c = [c for c in range(n_vars)
                           if c != j_var and c not in desc_j and c != y]
                if not valid_c:
                    continue

                # Project all valid pre-treatment covariates onto first PC (from obs data)
                v = _pc1_direction(obs_k, valid_c)
                obs_scores = obs_k[:, valid_c] @ v
                median_z = float(np.median(obs_scores))

                intv_scores = intv_samples[:, valid_c] @ v
                lo_mask = intv_scores <= median_z
                hi_mask = intv_scores > median_z

                if lo_mask.sum() > 0:
                    cate_gt[k, j, y, 0] = float(intv_samples[lo_mask, y].mean()) - float(obs_mean_k[y])
                if hi_mask.sum() > 0:
                    cate_gt[k, j, y, 1] = float(intv_samples[hi_mask, y].mean()) - float(obs_mean_k[y])

    return cate_gt


def compute_cate_metrics(cate_gt, cate_pred, desc_mask, valid_mask):
    """
    Compute CATE MAE metrics.

    Args:
        cate_gt, cate_pred: (n_scms, n_targets, n_vars, 2)
        desc_mask, valid_mask: (n_scms, n_targets, n_vars) broadcast over strata
    """
    desc_4d = np.stack([desc_mask, desc_mask], axis=-1)
    valid_4d = np.stack([valid_mask, valid_mask], axis=-1)

    error = np.abs(cate_pred - cate_gt)
    metrics = {}

    metrics['CATE_MAE_all'] = np.mean(error[valid_4d])

    desc_and_valid_4d = desc_4d & valid_4d
    if desc_and_valid_4d.sum() > 0:
        metrics['CATE_MAE_descendants'] = np.mean(error[desc_and_valid_4d])
    else:
        metrics['CATE_MAE_descendants'] = np.nan

    non_desc_and_valid_4d = (~desc_4d) & valid_4d
    if non_desc_and_valid_4d.sum() > 0:
        # True CATE for non-descendants is 0 (intervention has no causal effect).
        metrics['CATE_MAE_non_descendants'] = np.mean(np.abs(cate_pred[non_desc_and_valid_4d]))
    else:
        metrics['CATE_MAE_non_descendants'] = np.nan

    return metrics


def compute_cate_metrics_with_ci(cate_gt, cate_pred, desc_mask, valid_mask,
                                  n_bootstrap=1000, seed=0):
    """Compute CATE metrics with 95% bootstrap CIs (resampling over SCMs)."""
    metrics = compute_cate_metrics(cate_gt, cate_pred, desc_mask, valid_mask)

    n_scms = cate_gt.shape[0]
    rng = np.random.default_rng(seed)

    bootstrap_samples = {k: [] for k in metrics}
    for _ in range(n_bootstrap):
        idx = rng.choice(n_scms, size=n_scms, replace=True)
        boot = compute_cate_metrics(cate_gt[idx], cate_pred[idx],
                                     desc_mask[idx], valid_mask[idx])
        for k, v in boot.items():
            bootstrap_samples[k].append(v)

    ci_metrics = {}
    for k, vals in bootstrap_samples.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            ci_metrics[f'{k}_ci_lower'] = np.percentile(finite, 2.5)
            ci_metrics[f'{k}_ci_upper'] = np.percentile(finite, 97.5)
        else:
            ci_metrics[f'{k}_ci_lower'] = np.nan
            ci_metrics[f'{k}_ci_upper'] = np.nan

    return {**metrics, **ci_metrics}


def compute_qte_ground_truth(input_eval, target_eval, quantiles, n_eval_scms, n_targets, n_vars):
    """
    Compute ground-truth QTE from held-out samples.

    QTE(τ, k, j, y) = Q_τ[Y_y | do(X_j=v), SCM k] − Q_τ[Y_y | obs, SCM k]

    Returns:
        obs_quantiles: (n_eval_scms, n_quantiles, n_vars)  empirical obs quantiles
        qte_gt:        (n_eval_scms, n_targets, n_quantiles, n_vars)
    """
    n_quantiles = len(quantiles)
    obs_quantiles = np.zeros((n_eval_scms, n_quantiles, n_vars))
    qte_gt = np.zeros((n_eval_scms, n_targets, n_quantiles, n_vars))

    for k in range(n_eval_scms):
        for qi, tau in enumerate(quantiles):
            obs_quantiles[k, qi, :] = np.quantile(input_eval[k], tau, axis=0)

        for j in range(n_targets):
            for qi, tau in enumerate(quantiles):
                intv_q = np.quantile(target_eval[k, j], tau, axis=0)  # (n_vars,)
                qte_gt[k, j, qi, :] = intv_q - obs_quantiles[k, qi, :]

    return obs_quantiles, qte_gt


def compute_qte_metrics(qte_gt, qte_pred, desc_mask, valid_mask):
    """
    Compute QTE metrics averaged over quantiles {0.1,0.25,0.5,0.75,0.9}.

    For each (k, j, y): avg_qte_mae = mean_τ |qte_pred(τ) - qte_gt(τ)|

    This per-pair scalar is then averaged over all/desc/non-desc valid pairs.
    Also computes standardized variants (*_norm) divided by std(qte_gt) over the same subset.

    Args:
        qte_gt:   (n_scms, n_targets, n_quantiles, n_vars) ground truth
        qte_pred: (n_scms, n_targets, n_quantiles, n_vars) predicted
        desc_mask:  (n_scms, n_targets, n_vars) bool — y is descendant of j
        valid_mask: (n_scms, n_targets, n_vars) bool — pair to include

    Returns:
        dict with QTE_MAE_all, QTE_MAE_descendants, QTE_MAE_non_descendants and _norm variants
    """
    # Average absolute error over quantiles → (n_scms, n_targets, n_vars)
    per_pair_mae = np.mean(np.abs(qte_pred - qte_gt), axis=2)
    # Std of gt averaged over quantiles per pair → used for normalization
    gt_mean_over_q = qte_gt.mean(axis=2)  # (n_scms, n_targets, n_vars)

    metrics = {}

    def _compute_subset(mask_3d, suffix):
        if mask_3d.sum() == 0:
            metrics[f'QTE_MAE{suffix}'] = np.nan
            metrics[f'QTE_MAE{suffix}_norm'] = np.nan
            return
        gt_std = max(float(np.std(gt_mean_over_q[mask_3d])), 1e-12)
        metrics[f'QTE_MAE{suffix}'] = float(np.mean(per_pair_mae[mask_3d]))
        metrics[f'QTE_MAE{suffix}_norm'] = metrics[f'QTE_MAE{suffix}'] / gt_std

    _compute_subset(valid_mask, '_all')
    _compute_subset(desc_mask & valid_mask, '_descendants')
    _compute_subset((~desc_mask) & valid_mask, '_non_descendants')

    return metrics


def compute_qte_metrics_with_ci(qte_gt, qte_pred, desc_mask, valid_mask,
                                 n_bootstrap=1000, seed=0):
    """Compute QTE metrics with 95% bootstrap CIs (resampling over SCMs)."""
    metrics = compute_qte_metrics(qte_gt, qte_pred, desc_mask, valid_mask)

    n_scms = qte_gt.shape[0]
    rng = np.random.default_rng(seed)

    bootstrap_samples = {k: [] for k in metrics}
    for _ in range(n_bootstrap):
        idx = rng.choice(n_scms, size=n_scms, replace=True)
        boot = compute_qte_metrics(qte_gt[idx], qte_pred[idx],
                                   desc_mask[idx], valid_mask[idx])
        for k, v in boot.items():
            bootstrap_samples[k].append(v)

    ci_metrics = {}
    for k, vals in bootstrap_samples.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            ci_metrics[f'{k}_ci_lower'] = np.percentile(finite, 2.5)
            ci_metrics[f'{k}_ci_upper'] = np.percentile(finite, 97.5)
        else:
            ci_metrics[f'{k}_ci_lower'] = np.nan
            ci_metrics[f'{k}_ci_upper'] = np.nan

    return {**metrics, **ci_metrics}


def compute_intcov_ground_truth(target_eval, n_eval_scms, n_targets, n_vars):
    """
    Estimate interventional covariance matrices from held-out samples.

    Uses the Oracle Approximating Shrinkage (OAS) estimator (Chen et al. 2010)
    which is reliable with as few as 80 samples for typical n_vars (2–16).
    OAS shrinks the sample covariance towards a scaled identity matrix using
    a closed-form optimal shrinkage coefficient.

    Returns:
        intcov_gt: (n_eval_scms, n_targets, n_vars, n_vars)
    """
    intcov_gt = np.zeros((n_eval_scms, n_targets, n_vars, n_vars))
    for k in range(n_eval_scms):
        for j in range(n_targets):
            intcov_gt[k, j] = _oas_cov(target_eval[k, j].astype(float))
    return intcov_gt


def compute_intcov_metrics(intcov_gt, intcov_pred, desc_mask, valid_mask):
    """
    Compute full covariance matrix error metrics (MAE, RMSE) using Frobenius norm.

    Full covariance matrix (diagonal + off-diagonal), using the upper triangle to avoid
    double-counting symmetric entries. Each unique entry (y1, y2) with y1 <= y2 is included.

    Pairs are split by descendant status:
      - all:          all valid upper-triangle entries
      - both_desc:    both y1 and y2 are descendants of the intervened variable
      - neither_desc: neither y1 nor y2 is a descendant

    Also computes standardized variants (*_norm) divided by std(gt) over the same subset.

    Args:
        intcov_gt:   (n_scms, n_targets, n_vars, n_vars) ground truth covariances
        intcov_pred: (n_scms, n_targets, n_vars, n_vars) predicted covariances
        desc_mask:   (n_scms, n_targets, n_vars) True if variable is a descendant of j
        valid_mask:  (n_scms, n_targets, n_vars) True if variable is not the intervened var

    Returns:
        dict with CovMAE/CovRMSE for _all, _both_desc, _neither_desc
    """
    n_scms, n_targets, n_vars = desc_mask.shape

    # Upper-triangle mask including diagonal (avoids double-counting symmetric entries)
    triu_4d = np.triu(np.ones((n_vars, n_vars), dtype=bool))[None, None, :, :]

    # Pair-level masks: (n_scms, n_targets, n_vars, n_vars)
    valid_pair = valid_mask[:, :, :, None] & valid_mask[:, :, None, :]
    desc_pair = desc_mask[:, :, :, None] & desc_mask[:, :, None, :]
    neither_desc_pair = (~desc_mask[:, :, :, None]) & (~desc_mask[:, :, None, :]) & valid_pair

    abs_err = np.abs(intcov_pred - intcov_gt)
    sq_err = (intcov_pred - intcov_gt) ** 2

    metrics = {}

    def _subset(mask_4d, suffix):
        m = mask_4d & triu_4d
        if m.sum() == 0:
            for key in [f'CovMAE{suffix}', f'CovRMSE{suffix}']:
                metrics[key] = np.nan
            return
        metrics[f'CovMAE{suffix}'] = float(np.mean(abs_err[m]))
        metrics[f'CovRMSE{suffix}'] = float(np.sqrt(np.mean(sq_err[m])))

    _subset(valid_pair, '_all')
    _subset(desc_pair & valid_pair, '_both_desc')
    _subset(neither_desc_pair, '_neither_desc')

    return metrics


def compute_intcov_metrics_with_ci(intcov_gt, intcov_pred, desc_mask, valid_mask,
                                    n_bootstrap=1000, seed=0):
    """Compute covariance metrics with 95% bootstrap CIs (resampling over SCMs)."""
    metrics = compute_intcov_metrics(intcov_gt, intcov_pred, desc_mask, valid_mask)

    n_scms = intcov_gt.shape[0]
    rng = np.random.default_rng(seed)

    bootstrap_samples = {k: [] for k in metrics}
    for _ in range(n_bootstrap):
        idx = rng.choice(n_scms, size=n_scms, replace=True)
        boot = compute_intcov_metrics(intcov_gt[idx], intcov_pred[idx],
                                      desc_mask[idx], valid_mask[idx])
        for k, v in boot.items():
            bootstrap_samples[k].append(v)

    ci_metrics = {}
    for k, vals in bootstrap_samples.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            ci_metrics[f'{k}_ci_lower'] = np.percentile(finite, 2.5)
            ci_metrics[f'{k}_ci_upper'] = np.percentile(finite, 97.5)
        else:
            ci_metrics[f'{k}_ci_lower'] = np.nan
            ci_metrics[f'{k}_ci_upper'] = np.nan

    return {**metrics, **ci_metrics}


def compute_dist_metrics_with_ci(vals_all, vals_desc, vals_nondesc,
                                  metric_name, n_bootstrap=1000, seed=0,
                                  vals_all_full=None):
    """
    Compute distribution metric mean with 95% bootstrap CIs.

    Args:
        vals_all:      list of per-(k,j) scalars excluding the intervened variable (All/V_i)
        vals_desc:     list of per-(k,j) scalars for descendant subset
        vals_nondesc:  list of per-(k,j) scalars for non-descendant subset
        metric_name:   string prefix for metric keys (e.g. 'Energy', 'NLL')
        n_bootstrap:   number of bootstrap resamples
        seed:          random seed
        vals_all_full: list of per-(k,j) scalars including the intervened variable (All)

    Returns:
        dict with keys like f'{metric_name}_all', f'{metric_name}_all_vi',
        f'{metric_name}_descendants', f'{metric_name}_non_descendants' and CI bounds.
    """
    rng = np.random.default_rng(seed)
    metrics = {}

    def _ci(vals, key):
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            metrics[key] = float('nan')
            metrics[f'{key}_ci_lower'] = float('nan')
            metrics[f'{key}_ci_upper'] = float('nan')
            return
        metrics[key] = float(np.mean(arr))
        boot = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
        metrics[f'{key}_ci_lower'] = float(np.percentile(boot, 2.5))
        metrics[f'{key}_ci_upper'] = float(np.percentile(boot, 97.5))

    if vals_all_full is not None:
        _ci(vals_all_full, f'{metric_name}_all')
    _ci(vals_all,     f'{metric_name}_all_vi')
    _ci(vals_desc,    f'{metric_name}_descendants')
    _ci(vals_nondesc, f'{metric_name}_non_descendants')

    return metrics


def print_results_table(title, results, metrics_to_show, n_bootstrap=None, top_k_label='top-1'):
    """Print a formatted results table with optional bootstrap CI label."""
    COL_W = 27
    METHOD_W = max(25, max((len(k) for k in results), default=0) + 2)
    total_w = METHOD_W + 4 + len(metrics_to_show) * (COL_W + 3)

    print("\n" + "=" * total_w)
    suffix = f"  (95% bootstrap CI, n_bootstrap={n_bootstrap})" if n_bootstrap else ""
    print(f"{title}{suffix}")
    print("=" * total_w)

    header = "Method".ljust(METHOD_W) + " | "
    for metric in metrics_to_show:
        header += metric.ljust(COL_W) + " | "
    print(header)
    print("-" * total_w)

    def _fmt(res, key):
        val = res.get(key, np.nan)
        lo = res.get(f'{key}_ci_lower', np.nan)
        hi = res.get(f'{key}_ci_upper', np.nan)
        if np.isnan(val):
            return 'N/A'.ljust(COL_W)
        cell = f"{val:.4f} [{lo:.4f}, {hi:.4f}]"
        return cell.ljust(COL_W)

    METHOD_ORDER = [
        "Linear Causal",
        "ACTIVA (ours)",
        f"ACTIVA {top_k_label}",
        "ACTIVA best-comp",
        "MACE-TNP",
        f"MACE-TNP {top_k_label}",
        "MACE-TNP best-comp",
        "Gaussian Conditional",
    ]
    ordered = [m for m in METHOD_ORDER if m in results]
    # Append any unexpected methods not in METHOD_ORDER
    ordered += [m for m in results if m not in METHOD_ORDER]
    for method_name in ordered:
        row = method_name.ljust(METHOD_W) + " | "
        for metric in metrics_to_show:
            row += _fmt(results[method_name], metric) + " | "
        print(row)

    print("=" * total_w)


def main():
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("Causal Quantity Evaluation: ATE, IntMean, IntVar, CATE")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data_loader = load_scm_data(args.data_path, mode='interventional')

    input_data = np.array(data_loader.arrays[0])  # (n_scms, n_samples, n_vars)
    target_data = np.array(data_loader.arrays[1])  # (n_scms, n_targets, n_samples, n_vars)
    graphs = np.array(data_loader.arrays[2])       # (n_scms, n_vars, n_vars)
    targets = np.array(data_loader.arrays[3])      # (n_scms, n_interventions, n_targets, n_vars)

    n_scms, n_samples, n_vars = input_data.shape
    n_targets = target_data.shape[1]
    print(f"Data shape: {n_scms} SCMs, {n_samples} samples, {n_vars} variables, {n_targets} targets")

    # Subsample SCMs for evaluation
    n_eval_scms = min(args.n_eval_scms, n_scms)
    eval_indices = np.random.choice(n_scms, n_eval_scms, replace=False)

    # Fingerprint of the evaluated SCM subset. evaluate_gcm_baseline.py prints the
    # same line; the two must match for the baseline comparison to be paired.
    import hashlib as _hashlib
    _idx_fingerprint = _hashlib.md5(
        np.ascontiguousarray(eval_indices).tobytes()).hexdigest()[:12]
    print(f"eval_indices fingerprint: {_idx_fingerprint}  "
          f"(first 8: {eval_indices[:8].tolist()})")

    input_eval = input_data[eval_indices]
    target_eval = target_data[eval_indices]
    graphs_eval = graphs[eval_indices]
    targets_eval = targets[eval_indices]

    print(f"Evaluating on {n_eval_scms} SCMs")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    with open(args.model_path, 'rb') as f:
        load_dict = dill.load(f)

    params_dict = load_dict['params']
    model = load_dict['model_state']

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params_dict,
        tx=optax.identity()
    )
    state = state.replace(step=load_dict.get('step', 0))

    # Compute ground truth quantities
    print("\nComputing ground truth quantities...")
    obs_mean = input_eval.mean(axis=1)   # (n_eval_scms, n_vars)

    ate_gt = np.zeros((n_eval_scms, n_targets, n_vars))
    intvar_gt = np.zeros((n_eval_scms, n_targets, n_vars))

    for k in range(n_eval_scms):
        for j in range(n_targets):
            target_mean = target_eval[k, j, :, :].mean(axis=0)
            target_var = target_eval[k, j, :, :].var(axis=0)
            intvar_gt[k, j, :] = target_var
            ate_gt[k, j, :] = target_mean - obs_mean[k, :]

    print(f"ATE range: [{ate_gt.min():.4f}, {ate_gt.max():.4f}]")
    print(f"IntVar range: [{intvar_gt.min():.4f}, {intvar_gt.max():.4f}]")

    print("  Computing interventional covariance ground truth (OAS estimator)...", end="", flush=True)
    intcov_gt = compute_intcov_ground_truth(target_eval, n_eval_scms, n_targets, n_vars)
    offdiag_mask = ~np.eye(n_vars, dtype=bool)
    offdiag_vals = intcov_gt[:, :, offdiag_mask]
    print(f" done  (off-diag range: [{offdiag_vals.min():.4f}, {offdiag_vals.max():.4f}])")

    if n_eval_scms > 0:
        print(f"\nDebug: First SCM (k=0):")
        print(f"  Graph sparsity: {(graphs_eval[0] == 0).sum()} zeros out of {graphs_eval[0].size} entries")
        print(f"  obs_mean[0, :3]: {obs_mean[0, :3]}")

    # Compute descendant masks
    print("\nComputing descendant masks...")
    desc_mask = np.zeros((n_eval_scms, n_targets, n_vars), dtype=bool)
    for k in range(n_eval_scms):
        adj = graphs_eval[k]
        for j in range(n_targets):
            intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
            for v in intervened_vars:
                for y in get_descendants(adj, v):
                    desc_mask[k, j, y] = True

    valid_mask = np.ones((n_eval_scms, n_targets, n_vars), dtype=bool)
    for k in range(n_eval_scms):
        for j in range(n_targets):
            intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
            for v in intervened_vars:
                valid_mask[k, j, v] = False

    # All-variable mask including the intervened variable (no exclusions)
    valid_mask_full = np.ones((n_eval_scms, n_targets, n_vars), dtype=bool)

    # Initialize methods
    print("\nInitializing baselines and methods...")

    print(f"  Intervention value: do(X_j = {args.intervention_value})")

    gauss_baseline = GaussianConditionalBaseline()
    gauss_baseline.fit(input_eval, target_eval, graphs_eval, args.intervention_value)

    print("  Fitting Linear Causal baseline...", end="", flush=True)
    linear_baseline = LinearCausalBaseline()
    linear_baseline.fit(input_eval, target_eval, graphs_eval, args.intervention_value)
    print(" done")

    cvae_method = CVAEMethod(model, state, args.n_model_samples, args.samples_from_prior)

    # Run CVAE inference once — reuse for all metrics (fully analytical, no sampling)
    print("\n  CVAE inference (analytical, runs once for all metrics)...", end="", flush=True)
    cvae_intmean, cvae_intvar, cvae_gmm_params = cvae_method._predict_raw(
        input_eval, targets_eval, n_eval_scms, n_targets, rng
    )
    print(" done")

    # Skip top-k calculations if disabled
    cvae_top_intmean = None
    cvae_top_intvar = None
    cvae_top_gmm_params = None
    top_k_label = None
    if args.top_k_components and args.top_k_components > 0:
        cvae_batch_size = min(32, n_eval_scms)
        cvae_top_intmean, cvae_top_intvar, cvae_top_gmm_params = _cvae_top_component_predictions(
            cvae_gmm_params, n_eval_scms, n_targets, n_vars, cvae_batch_size, top_k=args.top_k_components
        )
        top_k_label = f'top-{args.top_k_components}'

    # Load and run MACE-TNP if requested (fully analytical — uses GMM parameters directly)
    mace_method = None
    mace_gmm_params = None
    mace_intmean = None
    mace_intvar = None
    mace_top_intmean = None
    mace_top_intvar = None
    mace_top_gmm_params = None
    if args.mace_tnp_model_path:
        print(f"\n  Loading MACE-TNP model from {args.mace_tnp_model_path}...", end="", flush=True)
        try:
            mace_method = MACETNPMethod(args.mace_tnp_model_path, args.intervention_value)
            print(" done")
            print("  MACE-TNP inference (analytical GMM, runs once for all metrics)...",
                  end="", flush=True)
            mace_intmean, mace_intvar, mace_gmm_params = mace_method._predict_raw(
                input_eval, targets_eval, n_eval_scms, n_targets, n_vars
            )
            print(" done")
            if top_k_label is not None:  # Only compute top-k if enabled
                mace_top_intmean, mace_top_intvar, mace_top_gmm_params = _mace_top_component_predictions(
                    mace_gmm_params, n_eval_scms, n_targets, n_vars, top_k=args.top_k_components
                )
        except Exception as e:
            print(f"\n  [WARNING] MACE-TNP loading/inference failed: {e}")
            mace_method = None

    # Best-component ATE (oracle: pick the component closest to true interventional mean)
    intmean_gt = ate_gt + obs_mean[:, None, :]  # (n_eval_scms, n_targets, n_vars)
    cvae_batch_size_bc = min(32, n_eval_scms)
    cvae_best_comp_ate = _cvae_best_component_ate(
        cvae_gmm_params, intmean_gt, obs_mean, n_eval_scms, n_targets, n_vars,
        batch_size=cvae_batch_size_bc
    )
    mace_best_comp_ate = None
    if mace_method is not None and mace_gmm_params is not None:
        mace_best_comp_ate = _mace_best_component_ate(
            mace_gmm_params, intmean_gt, obs_mean, n_eval_scms, n_targets, n_vars
        )

    # ATE predictions
    ate_preds = {
        'Gaussian Conditional': gauss_baseline.predict(n_eval_scms, n_targets, n_vars),
        'Linear Causal': linear_baseline.predict(n_eval_scms, n_targets, n_vars),
        'ACTIVA (ours)': cvae_intmean - obs_mean[:, None, :],
        'ACTIVA best-comp': cvae_best_comp_ate,
    }
    if top_k_label is not None:
        ate_preds[f'ACTIVA {top_k_label}'] = cvae_top_intmean - obs_mean[:, None, :]
    if mace_method is not None:
        ate_preds['MACE-TNP'] = mace_intmean - obs_mean[:, None, :]
        ate_preds['MACE-TNP best-comp'] = mace_best_comp_ate
        if top_k_label is not None:
            ate_preds[f'MACE-TNP {top_k_label}'] = mace_top_intmean - obs_mean[:, None, :]

    # IntVar predictions
    intvar_preds = {
        'Gaussian Conditional': gauss_baseline.predict_intvar(n_eval_scms, n_targets, n_vars),
        'Linear Causal': linear_baseline.predict_intvar(n_eval_scms, n_targets, n_vars),
        'ACTIVA (ours)': cvae_intvar,
    }
    if top_k_label is not None:
        intvar_preds[f'ACTIVA {top_k_label}'] = cvae_top_intvar
    if mace_method is not None:
        intvar_preds['MACE-TNP'] = mace_intvar
        if top_k_label is not None:
            intvar_preds[f'MACE-TNP {top_k_label}'] = mace_top_intvar

    # IntCov predictions (interventional covariance)
    batch_size_intcov = min(32, n_eval_scms)
    intcov_preds = {
        'Gaussian Conditional': gauss_baseline.predict_intcov(n_eval_scms, n_targets, n_vars),
        'Linear Causal': linear_baseline.predict_intcov(n_eval_scms, n_targets, n_vars),
        'ACTIVA (ours)': cvae_method.predict_intcov(
            cvae_gmm_params, n_eval_scms, n_targets, n_vars, batch_size=batch_size_intcov),
    }
    if top_k_label is not None:
        intcov_preds[f'ACTIVA {top_k_label}'] = cvae_method.predict_intcov(
            cvae_top_gmm_params, n_eval_scms, n_targets, n_vars, batch_size=batch_size_intcov)
    if mace_method is not None:
        intcov_preds['MACE-TNP'] = mace_method.predict_intcov(
            mace_gmm_params, n_eval_scms, n_targets, n_vars)
        if top_k_label is not None:
            intcov_preds[f'MACE-TNP {top_k_label}'] = mace_method.predict_intcov(
                mace_top_gmm_params, n_eval_scms, n_targets, n_vars)

    # CATE predictions
    print("\nComputing CATE ground truth...", end="", flush=True)
    cate_gt = compute_cate_ground_truth(
        input_eval, target_eval, graphs_eval, targets_eval, n_eval_scms, n_targets, n_vars
    )
    print(" done")

    print("  CATE: Gaussian Conditional...", end="", flush=True)
    cate_gauss = gauss_baseline.predict_cate(
        input_eval, n_eval_scms, n_targets, n_vars, graphs_eval, targets_eval
    )
    print(" done")

    print("  CATE: Linear Causal...", end="", flush=True)
    cate_linear = linear_baseline.predict_cate(n_eval_scms, n_targets, n_vars)
    print(" done")

    print("  CATE: ACTIVA (ours)...", end="", flush=True)
    cate_cvae = cvae_method.predict_cate(
        input_eval, targets_eval, n_eval_scms, n_targets, n_vars,
        graphs_eval, targets_eval, cvae_gmm_params
    )
    print(" done")

    cate_cvae_top1 = None
    if top_k_label is not None:
        print(f"  CATE: ACTIVA {top_k_label}...", end="", flush=True)
        cate_cvae_top1 = cvae_method.predict_cate(
            input_eval, targets_eval, n_eval_scms, n_targets, n_vars,
            graphs_eval, targets_eval, cvae_top_gmm_params
        )
        print(" done")

    cate_preds = {
        'Gaussian Conditional': cate_gauss,
        'Linear Causal': cate_linear,
        'ACTIVA (ours)': cate_cvae,
    }
    if top_k_label is not None:
        cate_preds[f'ACTIVA {top_k_label}'] = cate_cvae_top1
    if mace_method is not None:
        print("  CATE: MACE-TNP...", end="", flush=True)
        cate_preds['MACE-TNP'] = mace_method.predict_cate(
            input_eval, mace_gmm_params, n_eval_scms, n_targets, n_vars, graphs_eval, targets_eval
        )
        print(" done")

        if top_k_label is not None:
            print(f"  CATE: MACE-TNP {top_k_label}...", end="", flush=True)
            cate_preds[f'MACE-TNP {top_k_label}'] = mace_method.predict_cate(
                input_eval, mace_top_gmm_params, n_eval_scms, n_targets, n_vars, graphs_eval, targets_eval
            )
            print(" done")

    # # QTE predictions
    # print("\nComputing QTE ground truth and predictions...")
    # print(f"  Quantiles: {QTE_QUANTILES}")
    # obs_quantiles, qte_gt = compute_qte_ground_truth(
    #     input_eval, target_eval, QTE_QUANTILES, n_eval_scms, n_targets, n_vars
    # )
    # print(f"  QTE range: [{qte_gt.min():.4f}, {qte_gt.max():.4f}]")
    #
    # print("  QTE: Gaussian Conditional...", end="", flush=True)
    # qte_preds = {
    #     'Gaussian Conditional': gauss_baseline.predict_qte(
    #         obs_quantiles, QTE_QUANTILES, n_eval_scms, n_targets, n_vars),
    # }
    # print(" done")
    #
    # print("  QTE: Linear Causal...", end="", flush=True)
    # qte_preds['Linear Causal'] = linear_baseline.predict_qte(
    #     obs_quantiles, QTE_QUANTILES, n_eval_scms, n_targets, n_vars
    # )
    # print(" done")
    #
    # print("  QTE: ACTIVA (ours)...", end="", flush=True)
    # cvae_batch_size_qte = min(32, n_eval_scms)
    # qte_preds['ACTIVA (ours)'] = predict_qte_from_cvae_gmm(
    #     cvae_gmm_params, obs_quantiles, QTE_QUANTILES, n_eval_scms, n_targets, n_vars,
    #     batch_size=cvae_batch_size_qte
    # )
    # print(" done")
    #
    # if top_k_label is not None:
    #     print(f"  QTE: ACTIVA {top_k_label}...", end="", flush=True)
    #     qte_preds[f'ACTIVA {top_k_label}'] = predict_qte_from_cvae_gmm(
    #         cvae_top_gmm_params, obs_quantiles, QTE_QUANTILES, n_eval_scms, n_targets, n_vars,
    #         batch_size=cvae_batch_size_qte
    #     )
    #     print(" done")
    #
    # if mace_method is not None:
    #     print("  QTE: MACE-TNP...", end="", flush=True)
    #     qte_preds['MACE-TNP'] = predict_qte_from_mace_gmm(
    #         mace_gmm_params, obs_quantiles, QTE_QUANTILES, n_eval_scms, n_targets, n_vars
    #     )
    #     print(" done")
    #
    #     if top_k_label is not None:
    #         print(f"  QTE: MACE-TNP {top_k_label}...", end="", flush=True)
    #         qte_preds[f'MACE-TNP {top_k_label}'] = predict_qte_from_mace_gmm(
    #             mace_top_gmm_params, obs_quantiles, QTE_QUANTILES, n_eval_scms, n_targets, n_vars
    #         )
    #         print(" done")

    # Compute metrics with CI
    print("\nComputing metrics with bootstrap CIs...")
    best_comp_names = {'ACTIVA best-comp', 'MACE-TNP best-comp'}
    method_names = [n for n in ate_preds if n not in best_comp_names]

    ate_results, intvar_results, intcov_results, cate_results = {}, {}, {}, {}

    for name in method_names:
        print(f"  {name}...", end="", flush=True)
        ate_results[name] = compute_metrics_with_ci(
            ate_gt, ate_preds[name], desc_mask, valid_mask,
            n_bootstrap=args.n_bootstrap, seed=args.seed, non_desc_vs_zero=True,
            valid_mask_full=valid_mask_full
        )
        intvar_results[name] = compute_metrics_with_ci(
            intvar_gt, intvar_preds[name], desc_mask, valid_mask,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
            valid_mask_full=valid_mask_full
        )
        if name in intcov_preds:
            intcov_results[name] = compute_intcov_metrics_with_ci(
                intcov_gt, intcov_preds[name], desc_mask, valid_mask,
                n_bootstrap=args.n_bootstrap, seed=args.seed
            )
        cate_results[name] = compute_cate_metrics_with_ci(
            cate_gt, cate_preds[name], desc_mask, valid_mask,
            n_bootstrap=args.n_bootstrap, seed=args.seed
        )
        # if name in qte_preds:
        #     qte_results[name] = compute_qte_metrics_with_ci(
        #         qte_gt, qte_preds[name], desc_mask, valid_mask,
        #         n_bootstrap=args.n_bootstrap, seed=args.seed
        #     )
        print(" done")

    # Best-component ATE metrics (oracle upper bound for mode selection)
    for name in best_comp_names:
        if ate_preds.get(name) is None:
            continue
        print(f"  {name}...", end="", flush=True)
        ate_results[name] = compute_metrics_with_ci(
            ate_gt, ate_preds[name], desc_mask, valid_mask,
            n_bootstrap=args.n_bootstrap, seed=args.seed, non_desc_vs_zero=True,
            valid_mask_full=valid_mask_full
        )
        print(" done")

    # Print tables
    print_results_table(
        "ATE Estimation Results",
        ate_results,
        ['MAE_all', 'MAE_all_vi', 'RMSE_all', 'MAE_descendants', 'MAE_non_descendants', 'MAE_non_desc_noise_floor'],
        n_bootstrap=args.n_bootstrap,
        top_k_label=top_k_label,
    )

    print_results_table(
        "Interventional Variance Results",
        intvar_results,
        ['MAE_all', 'MAE_all_vi', 'RMSE_all', 'MAE_descendants', 'RMSE_descendants', 'MAE_non_descendants'],
        n_bootstrap=args.n_bootstrap,
        top_k_label=top_k_label,
    )

    if intcov_results:
        print_results_table(
            "Interventional Covariance Results — full matrix MAE/RMSE",
            intcov_results,
            ['CovMAE_all', 'CovRMSE_all', 'CovMAE_both_desc', 'CovMAE_neither_desc'],
            n_bootstrap=args.n_bootstrap,
            top_k_label=top_k_label,
        )

    print_results_table(
        "CATE Results",
        cate_results,
        ['CATE_MAE_all', 'CATE_MAE_descendants', 'CATE_MAE_non_descendants'],
        n_bootstrap=args.n_bootstrap,
        top_k_label=top_k_label,
    )

    # -----------------------------------------------------------------------
    # Distribution metrics: Energy Distance and NLL
    # -----------------------------------------------------------------------
    print("\nComputing distribution metrics (Energy Distance, NLL)...")
    n_dist_samples = n_samples  # match number of data samples

    dist_method_names = ['Gaussian Conditional', 'Linear Causal', 'ACTIVA (ours)']
    if top_k_label is not None:
        dist_method_names.append(f'ACTIVA {top_k_label}')
    if mace_method is not None:
        dist_method_names.append('MACE-TNP')
        if top_k_label is not None:
            dist_method_names.append(f'MACE-TNP {top_k_label}')

    # Per-(k,j) scalar lists for each subset
    energy_all_full_vals = {name: [] for name in dist_method_names}  # All (includes V_i)
    energy_all_vals      = {name: [] for name in dist_method_names}  # All/V_i (excludes V_i)
    energy_desc_vals     = {name: [] for name in dist_method_names}
    energy_nondesc_vals  = {name: [] for name in dist_method_names}
    nll_all_full_vals    = {name: [] for name in dist_method_names}  # All (includes V_i where possible)
    nll_all_vals         = {name: [] for name in dist_method_names}  # All/V_i (excludes V_i)
    nll_desc_vals        = {name: [] for name in dist_method_names}
    nll_nondesc_vals     = {name: [] for name in dist_method_names}

    rng_dist = np.random.default_rng(args.seed)
    cvae_batch_size_dist = min(32, n_eval_scms)

    for k in range(n_eval_scms):
        for j in range(n_targets):
            true_samples_kj = np.array(target_eval[k, j])  # (n_samples, n_vars)

            intervened_vars = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
            j_var = int(intervened_vars[0]) if len(intervened_vars) > 0 else j
            adj = graphs_eval[k]
            desc_set = get_descendants(adj, j_var)
            non_desc_set = set(range(n_vars)) - desc_set - {j_var}
            desc_dims     = sorted(desc_set)
            non_desc_dims = sorted(non_desc_set)
            all_dims_full = list(range(n_vars))                    # All (includes V_i)
            all_dims_vi   = [d for d in range(n_vars) if d != j_var]  # All/V_i (excludes V_i)

            # --- Gaussian Conditional ---
            pred_gauss = _sample_gauss_conditional(gauss_baseline, k, j, n_dist_samples, rng_dist)
            energy_all_full_vals['Gaussian Conditional'].append(
                _energy_dist_np(true_samples_kj[:, all_dims_full], pred_gauss[:, all_dims_full]))
            energy_all_vals['Gaussian Conditional'].append(
                _energy_dist_np(true_samples_kj[:, all_dims_vi], pred_gauss[:, all_dims_vi]))
            # NLL: Gaussian conditional can't model V_i (it was conditioned out); all_full = all_vi
            nll_all_full_vals['Gaussian Conditional'].append(
                gauss_baseline.predict_dist_nll(k, j, true_samples_kj))
            nll_all_vals['Gaussian Conditional'].append(
                gauss_baseline.predict_dist_nll(k, j, true_samples_kj))
            if desc_dims:
                energy_desc_vals['Gaussian Conditional'].append(
                    _energy_dist_np(true_samples_kj[:, desc_dims], pred_gauss[:, desc_dims]))
                nll_desc_vals['Gaussian Conditional'].append(
                    gauss_baseline.predict_dist_nll(k, j, true_samples_kj, dims=desc_dims))
            if non_desc_dims:
                energy_nondesc_vals['Gaussian Conditional'].append(
                    _energy_dist_np(true_samples_kj[:, non_desc_dims], pred_gauss[:, non_desc_dims]))
                nll_nondesc_vals['Gaussian Conditional'].append(
                    gauss_baseline.predict_dist_nll(k, j, true_samples_kj, dims=non_desc_dims))

            # --- Linear Causal ---
            pred_linear = _sample_from_linear_baseline(linear_baseline, k, j, n_dist_samples, rng_dist)
            energy_all_full_vals['Linear Causal'].append(
                _energy_dist_np(true_samples_kj[:, all_dims_full], pred_linear[:, all_dims_full]))
            energy_all_vals['Linear Causal'].append(
                _energy_dist_np(true_samples_kj[:, all_dims_vi], pred_linear[:, all_dims_vi]))
            # NLL: linear baseline can't model V_i (it was fixed by intervention); all_full = all_vi
            nll_all_full_vals['Linear Causal'].append(
                linear_baseline.predict_dist_nll(k, j, true_samples_kj))
            nll_all_vals['Linear Causal'].append(
                linear_baseline.predict_dist_nll(k, j, true_samples_kj))
            if desc_dims:
                energy_desc_vals['Linear Causal'].append(
                    _energy_dist_np(true_samples_kj[:, desc_dims], pred_linear[:, desc_dims]))
                nll_desc_vals['Linear Causal'].append(
                    linear_baseline.predict_dist_nll(k, j, true_samples_kj, dims=desc_dims))
            if non_desc_dims:
                energy_nondesc_vals['Linear Causal'].append(
                    _energy_dist_np(true_samples_kj[:, non_desc_dims], pred_linear[:, non_desc_dims]))
                nll_nondesc_vals['Linear Causal'].append(
                    linear_baseline.predict_dist_nll(k, j, true_samples_kj, dims=non_desc_dims))

            # --- ACTIVA ---
            batch_start_k = (k // cvae_batch_size_dist) * cvae_batch_size_dist
            k_in_batch_k  = k - batch_start_k
            cvae_params_kj = cvae_gmm_params.get((batch_start_k, j))
            if cvae_params_kj is not None:
                fw, mm, cv = cvae_params_kj
                w_k  = fw[k_in_batch_k]
                mu_k = mm[k_in_batch_k]
                cv_k = cv[k_in_batch_k]
                pred_cvae = _sample_from_gmm(w_k, mu_k, cv_k, n_dist_samples, rng_dist)
                energy_all_full_vals['ACTIVA (ours)'].append(
                    _energy_dist_np(true_samples_kj[:, all_dims_full], pred_cvae[:, all_dims_full]))
                energy_all_vals['ACTIVA (ours)'].append(
                    _energy_dist_np(true_samples_kj[:, all_dims_vi], pred_cvae[:, all_dims_vi]))
                nll_all_full_vals['ACTIVA (ours)'].append(
                    _marg_gmm_nll_np(w_k, mu_k, cv_k, true_samples_kj, all_dims_full))
                nll_all_vals['ACTIVA (ours)'].append(
                    _marg_gmm_nll_np(w_k, mu_k, cv_k, true_samples_kj, all_dims_vi))
                if desc_dims:
                    energy_desc_vals['ACTIVA (ours)'].append(
                        _energy_dist_np(true_samples_kj[:, desc_dims], pred_cvae[:, desc_dims]))
                    nll_desc_vals['ACTIVA (ours)'].append(
                        _marg_gmm_nll_np(w_k, mu_k, cv_k, true_samples_kj, desc_dims))
                if non_desc_dims:
                    energy_nondesc_vals['ACTIVA (ours)'].append(
                        _energy_dist_np(true_samples_kj[:, non_desc_dims], pred_cvae[:, non_desc_dims]))
                    nll_nondesc_vals['ACTIVA (ours)'].append(
                        _marg_gmm_nll_np(w_k, mu_k, cv_k, true_samples_kj, non_desc_dims))

            # --- ACTIVA top-k ---
            if top_k_label is not None and cvae_top_gmm_params is not None:
                top_params_kj = cvae_top_gmm_params.get((batch_start_k, j))
                if top_params_kj is not None:
                    fw_t, mm_t, cv_t = top_params_kj
                    w_kt  = fw_t[k_in_batch_k]
                    mu_kt = mm_t[k_in_batch_k]
                    cv_kt = cv_t[k_in_batch_k]
                    pred_cvae_top = _sample_from_gmm(w_kt, mu_kt, cv_kt, n_dist_samples, rng_dist)
                    name_top = f'ACTIVA {top_k_label}'
                    energy_all_full_vals[name_top].append(
                        _energy_dist_np(true_samples_kj[:, all_dims_full], pred_cvae_top[:, all_dims_full]))
                    energy_all_vals[name_top].append(
                        _energy_dist_np(true_samples_kj[:, all_dims_vi], pred_cvae_top[:, all_dims_vi]))
                    nll_all_full_vals[name_top].append(
                        _marg_gmm_nll_np(w_kt, mu_kt, cv_kt, true_samples_kj, all_dims_full))
                    nll_all_vals[name_top].append(
                        _marg_gmm_nll_np(w_kt, mu_kt, cv_kt, true_samples_kj, all_dims_vi))
                    if desc_dims:
                        energy_desc_vals[name_top].append(
                            _energy_dist_np(true_samples_kj[:, desc_dims], pred_cvae_top[:, desc_dims]))
                        nll_desc_vals[name_top].append(
                            _marg_gmm_nll_np(w_kt, mu_kt, cv_kt, true_samples_kj, desc_dims))
                    if non_desc_dims:
                        energy_nondesc_vals[name_top].append(
                            _energy_dist_np(true_samples_kj[:, non_desc_dims], pred_cvae_top[:, non_desc_dims]))
                        nll_nondesc_vals[name_top].append(
                            _marg_gmm_nll_np(w_kt, mu_kt, cv_kt, true_samples_kj, non_desc_dims))

            # --- MACE-TNP ---
            if mace_method is not None:
                mace_params_kj = mace_gmm_params.get((k, j))
                if mace_params_kj is not None:
                    w_m, mu_m, cv_m = mace_params_kj
                    pred_mace = _sample_from_gmm(w_m, mu_m, cv_m, n_dist_samples, rng_dist)
                    energy_all_full_vals['MACE-TNP'].append(
                        _energy_dist_np(true_samples_kj[:, all_dims_full], pred_mace[:, all_dims_full]))
                    energy_all_vals['MACE-TNP'].append(
                        _energy_dist_np(true_samples_kj[:, all_dims_vi], pred_mace[:, all_dims_vi]))
                    nll_all_full_vals['MACE-TNP'].append(
                        _marg_gmm_nll_np(w_m, mu_m, cv_m, true_samples_kj, all_dims_full))
                    nll_all_vals['MACE-TNP'].append(
                        _marg_gmm_nll_np(w_m, mu_m, cv_m, true_samples_kj, all_dims_vi))
                    if desc_dims:
                        energy_desc_vals['MACE-TNP'].append(
                            _energy_dist_np(true_samples_kj[:, desc_dims], pred_mace[:, desc_dims]))
                        nll_desc_vals['MACE-TNP'].append(
                            _marg_gmm_nll_np(w_m, mu_m, cv_m, true_samples_kj, desc_dims))
                    if non_desc_dims:
                        energy_nondesc_vals['MACE-TNP'].append(
                            _energy_dist_np(true_samples_kj[:, non_desc_dims], pred_mace[:, non_desc_dims]))
                        nll_nondesc_vals['MACE-TNP'].append(
                            _marg_gmm_nll_np(w_m, mu_m, cv_m, true_samples_kj, non_desc_dims))

                # --- MACE-TNP top-k ---
                if top_k_label is not None and mace_top_gmm_params is not None:
                    mace_top_params_kj = mace_top_gmm_params.get((k, j))
                    if mace_top_params_kj is not None:
                        w_mt, mu_mt, cv_mt = mace_top_params_kj
                        pred_mace_top = _sample_from_gmm(w_mt, mu_mt, cv_mt, n_dist_samples, rng_dist)
                        name_mt = f'MACE-TNP {top_k_label}'
                        energy_all_full_vals[name_mt].append(
                            _energy_dist_np(true_samples_kj[:, all_dims_full], pred_mace_top[:, all_dims_full]))
                        energy_all_vals[name_mt].append(
                            _energy_dist_np(true_samples_kj[:, all_dims_vi], pred_mace_top[:, all_dims_vi]))
                        nll_all_full_vals[name_mt].append(
                            _marg_gmm_nll_np(w_mt, mu_mt, cv_mt, true_samples_kj, all_dims_full))
                        nll_all_vals[name_mt].append(
                            _marg_gmm_nll_np(w_mt, mu_mt, cv_mt, true_samples_kj, all_dims_vi))
                        if desc_dims:
                            energy_desc_vals[name_mt].append(
                                _energy_dist_np(true_samples_kj[:, desc_dims], pred_mace_top[:, desc_dims]))
                            nll_desc_vals[name_mt].append(
                                _marg_gmm_nll_np(w_mt, mu_mt, cv_mt, true_samples_kj, desc_dims))
                        if non_desc_dims:
                            energy_nondesc_vals[name_mt].append(
                                _energy_dist_np(true_samples_kj[:, non_desc_dims], pred_mace_top[:, non_desc_dims]))
                            nll_nondesc_vals[name_mt].append(
                                _marg_gmm_nll_np(w_mt, mu_mt, cv_mt, true_samples_kj, non_desc_dims))

    print(" done")
    print("\nComputing distribution metric CIs...")
    energy_results = {}
    nll_results = {}

    for name in dist_method_names:
        energy_results[name] = compute_dist_metrics_with_ci(
            energy_all_vals[name], energy_desc_vals[name], energy_nondesc_vals[name],
            'Energy', n_bootstrap=args.n_bootstrap, seed=args.seed,
            vals_all_full=energy_all_full_vals[name]
        )
        nll_results[name] = compute_dist_metrics_with_ci(
            nll_all_vals[name], nll_desc_vals[name], nll_nondesc_vals[name],
            'NLL', n_bootstrap=args.n_bootstrap, seed=args.seed,
            vals_all_full=nll_all_full_vals[name]
        )

    print_results_table(
        "Energy Distance Results",
        energy_results,
        ['Energy_all', 'Energy_all_vi', 'Energy_descendants', 'Energy_non_descendants'],
        n_bootstrap=args.n_bootstrap,
        top_k_label=top_k_label,
    )

    print_results_table(
        "NLL Results",
        nll_results,
        ['NLL_all', 'NLL_all_vi', 'NLL_descendants', 'NLL_non_descendants'],
        n_bootstrap=args.n_bootstrap,
        top_k_label=top_k_label,
    )

    print("\nMetrics format: mean [95% CI lower, upper]")
    print("  ATE:     E[V | do(V_j=v_j)] - E[V | do(∅)]")
    print("  IntVar:  Var[V | do(V_j=v_j)]")
    print("  IntCov:  full Cov[V | do(V_j=v_j)] (Frobenius norm, upper triangle), GT estimated via OAS shrinkage")
    print("  CATE:    E[V | do(V_j=v_j), V_c in bin] - E[V | do(∅)], averaged over valid c")
    print("  Energy:  energy distance between predicted and true interventional distributions")
    print("  NLL:     negative log-likelihood of true samples under predicted distribution")
    print()

    all_results = {
        'ate': ate_results,
        'intvar': intvar_results,
        'intcov': intcov_results,
        'cate': cate_results,
        'energy': energy_results,
        'nll': nll_results,
        # 'mmd': mmd_results,   # commented out
        # 'wsd': wsd_results,   # commented out
    }

    # Save results to JSON for downstream plotting
    import json

    if args.results_path:
        out_path = args.results_path
    else:
        model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
        data_stem = os.path.splitext(os.path.basename(args.data_path))[0]
        out_dir = os.path.dirname(args.model_path) or '.'
        out_path = os.path.join(out_dir, f'results_{model_stem}_{data_stem}.json')

    def _to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (v != v) else v  # NaN → null
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    serializable = _to_serializable(all_results)
    # Add metadata
    serializable['_meta'] = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'n_eval_scms': n_eval_scms,
        'n_bootstrap': args.n_bootstrap,
        'intervention_value': args.intervention_value,
    }
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Distribution comparison plot (after JSON save so a plot crash doesn't lose results)
    if args.n_plot_scms > 0 and n_targets >= 2:
        plot_path = args.plot_path
        if not plot_path:
            model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
            out_dir = os.path.dirname(args.model_path) or '.'
            plot_path = os.path.join(out_dir, f'dist_comparison_{model_stem}.pdf')
        print(f"\nGenerating distribution comparison plot ({args.n_plot_scms} SCMs)...")
        import matplotlib
        matplotlib.use('Agg')
        fig = plot_distribution_comparison(
            input_eval=input_eval,
            target_eval=target_eval,
            targets_eval=targets_eval,
            gauss_baseline=gauss_baseline,
            linear_baseline=linear_baseline,
            mace_method=mace_method,
            mace_gmm_params=mace_gmm_params,
            cvae_gmm_params=cvae_gmm_params,
            mace_top_gmm_params=mace_top_gmm_params,
            cvae_top_gmm_params=cvae_top_gmm_params,
            n_vars=n_vars,
            n_eval_scms=n_eval_scms,
            intervention_value=args.intervention_value,
            n_plot_scms=args.n_plot_scms,
            n_samples=500,
            seed=args.seed,
            save_path=plot_path,
            top_k_label=top_k_label,
        )
        if args.track:
            import wandb
            import io
            from PIL import Image as PILImage
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            wandb.log({'dist_comparison': wandb.Image(PILImage.open(buf).copy())})
            buf.close()
        import matplotlib.pyplot as plt
        plt.close(fig)

    return all_results


if __name__ == '__main__':
    results = main()
