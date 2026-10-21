import jax.numpy as jnp
import jax_dataloader as jdl
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
import numpy.linalg

import wandb
import dill
from flax.training import train_state
from flax import serialization
import jax
import distrax
from functools import partial
from numpy.random import default_rng
import  numpy as np



def safe_log_prob(distribution, x):
    """Evaluate log_prob and replace any NaN/Inf with -1e6 for numerical stability."""
    log_prob = distribution.log_prob(x)
    log_prob = jnp.where(jnp.isinf(log_prob), -1e6, log_prob)
    log_prob = jnp.where(jnp.isnan(log_prob), -1e6, log_prob)
    return log_prob


"""
Utility helpers used across training, evaluation and plotting scripts.

This module provides:
 - Serialization helpers for saving/loading Flax models and TrainState
 - Data loading wrapper for the SCM datasets used in experiments
 - Plotting helpers that integrate with W&B (optional)
 - A small collection of statistical metrics (MMD, Wasserstein, Energy)
"""

def save_model_state(path, model, state):
    """
    Save a Flax model and associated TrainState to disk.

    The file written at `path` is a dill archive containing the following keys:
      - model_state: the model object (nn.Module instance)
      - params: the TrainState.params dict (model parameters)
      - tx: the optimizer used (TrainState.tx)
      - step: training step integer

    This format is intended for later round-trip via `load_model_state`.

    Args:
        path: destination path for the dill file (string or Path-like).
        model: an initialized Flax nn.Module instance.
        state: a flax.training.TrainState instance containing params and tx.
    """
    save_dict = {
        'model_state': model,
        'params': state.params,
        'tx': state.tx,
        'step': state.step,
    }
    with open(path, 'wb') as f:
        dill.dump(save_dict, f)


def load_model_state(path):
    """
    Load a Flax model and TrainState from a dill checkpoint written by `save_model_state`.

    Args:
        path: path to the dill file (string or Path-like).

    Returns:
        (model, state) where `model` is the nn.Module instance and `state` is a
        reconstructed flax.training.TrainState with params, tx, and step restored.
    """
    with open(path, 'rb') as f:
        load_dict = dill.load(f)

    model = load_dict['model_state']

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=load_dict['params'],
        tx=load_dict['tx']
    )
    state = state.replace(step=load_dict.get('step', 0))
    return model, state



def load_scm_data(scm_file: str, mode: str ='interventional', dtype=jnp.float32) -> jdl.ArrayDataset:
    """
    Load SCM data saved as a NumPy .npz file and wrap it into a jax_dataloader ArrayDataset.

    The function expects the following layout for the interventional mode (the format
    produced by `generate_data.py`): keys `input`, `target_data`, `graphs`, `targets`.

    In observational mode it returns an ArrayDataset containing a single array stored
    as `arr_0` in the .npz (the legacy format produced by `generate_basic_scms.py`).

    Args:
        scm_file: Path to the .npz file.
        mode: Either `'observational'` or `'interventional'`.
        dtype: jax dtype to cast numeric arrays to (default jnp.float32).

    Returns:
        jax_dataloader.ArrayDataset containing the loaded arrays in the order expected
        by the training and evaluation scripts.

        Ordering and shapes (interventional mode):
            - arrays[0] : observational input, shape (n_scms, n_samples, n_vars)
            - arrays[1..K]: target interventional datasets per intervention -> each has shape (n_scms, n_targets, n_samples, n_vars)
            - arrays[-2]: graphs, shape (n_scms, n_vars, n_vars)
            - arrays[-1]: targets, shape (n_scms, n_interventions, n_targets, n_vars)
    """
    with jnp.load(scm_file) as data:
        if mode == 'observational':
            return jdl.ArrayDataset(data['arr_0'])
        elif mode == 'interventional':
            input_data = jnp.astype(data['input'], dtype)
            target_data = jnp.astype(data['target_data'], dtype)
            graphs = jnp.astype(data['graphs'], dtype)
            targets = jnp.astype(data['targets'], dtype)
            return jdl.ArrayDataset(*[input_data] + [td for td in target_data] + [graphs] + [targets])


def determine_plot_axis(data):
    """
    Compute a square plotting axis for 2-D datasets stored in a jax_dataloader.ArrayDataset.

    The function inspects the first stored array in `data` and computes a bounding box
    with a small margin so scatter/histogram visualizations have breathing room.

    Args:
        data: a jax_dataloader.ArrayDataset where the first array has shape
              (n_scms, n_samples, 2) or similar. The function reads min/max over the
              first two dimensions and produces (xmin, xmax, ymin, ymax).

    Returns:
        A 4-tuple (xmin, xmax, ymin, ymax) suitable for matplotlib axis limits.
    """
    # Determine axis of the plots
    plot_axis = (data.arrays[0][:, :, 0].min(), data.arrays[0][:, :, 0].max(), data.arrays[0][:, :, 1].min(),
                 data.arrays[0][:, :, 1].max())
    margin = max(plot_axis) * 0.1
    return (plot_axis[0] - margin, plot_axis[1] + margin, plot_axis[2] - margin, plot_axis[3] + margin)


@jax.jit
def jax_hist_1d(x, num_bins=7):
    """
    Computes a 1D histogram of `x` with `num_bins` bins using JAX.
    If x_min == x_max, artificially expand so there's at least 1 bin width.
    """
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    # If all data is identical, expand artificially so the histogram is visible
    x_max = jnp.where(x_max == x_min, x_min + 1.0, x_max)

    bin_edges = jnp.linspace(x_min, x_max, num_bins + 1)
    indices = jnp.searchsorted(bin_edges, x, side='right') - 1
    indices = jnp.clip(indices, 0, num_bins - 1)
    counts = jnp.bincount(indices, length=num_bins)
    return counts, bin_edges


def save_distribution_plot(data, title, filename, tracking, colored=False, axis=None, has_batch_dim=False):
    """
    Quick scatter plot writer used by training/evaluation scripts.

    This lightweight helper writes a 2-D scatter plot to `filename`. When
    `has_batch_dim=True` the input is expected to have shape (B, S, 2) and the
    sample dimension is flattened for plotting. When `colored=True`, distinct
    colors are used for subsets.

    Args:
        data: array-like holding 2-D points or a batch of such arrays. Expected
              shapes include (N, 2) or (B, S, 2) when `has_batch_dim=True`.
        title: title string (also used as W&B key if `tracking` is True).
        filename: path where the PNG/PDF is written.
        tracking: if True, the plot is logged to W&B (wandb must be initialized).
        colored: whether to color subsets independently.
        axis: optional axis tuple (xmin, xmax, ymin, ymax).
        has_batch_dim: whether `data` includes a batch dimension (B, S, 2).
    """
    if colored:
        data = data[:10]
    colors = ['#1f77b4', '#8c564b', '#17becf',  '#7f7f7f', '#2ca02c',  '#bcbd22',  '#ff7f0e', '#e377c2',  '#9467bd', '#d62728'][:len(data)]
    if has_batch_dim:
        colors = [ele for ele in colors for _ in range(data.shape[1])]
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))  # Flatten sample dimension
    plt.figure(figsize=(8, 8))
    if colored:
        plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.9)
    else:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.1)
    # plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if axis:
        plt.axis(axis)
    else:
        plt.axis('equal')
    if tracking:
        wandb.log({str(title): wandb.Image(plt)}, commit=False)
    plt.savefig(filename)
    plt.close()


def save_distribution_plot_colored(
        data,
        title,
        filename,
        tracking,
        colored=False,
        axis=None,
        has_batch_dim=False
):
    """
    Draws a scatter plot of `data` with optional marginal histograms.

    - If colored=True, each subset gets a distinct color + marginal histograms.
    - If colored=False, a single scatter is drawn (no histograms).
    - The main scatter is strictly square by computing and setting a square bounding box.
    - The histograms are physically smaller (via GridSpec) and are aligned exactly over
      the same x-range (top) or y-range (right) as the scatter.
    - If data is degenerate (all x or all y are the same), the histogram artificially
      expands that dimension so at least one visible bin is drawn.

        data can be:
            - A single array of shape (N, 2) if has_batch_dim=False,
            - A list of arrays [d0, d1, ... dB], each shape (Ni, 2),
            - Or an array of shape (B, S, 2) if has_batch_dim=True.
    """
    base_colors = [
        '#1f77b4', '#8c564b', '#17becf', '#7f7f7f',
        '#2ca02c', '#bcbd22', '#ff7f0e', '#e377c2',
        '#9467bd', '#d62728'
    ]

    # -----------------------------
    # 1) Gather all data subsets
    # -----------------------------
    if has_batch_dim:
        B = data.shape[0]
        subsets = [np.asarray(data[i]) for i in range(B)]
        colors = base_colors[:B]
    else:
        if isinstance(data, list):
            subsets = data
            B = len(subsets)
            colors = base_colors[:B]
        else:
            subsets = [np.asarray(data)]
            B = 1
            colors = [base_colors[0]]

    # -----------------------------
    # 2) Compute overall bounding box of *all* data => make it square
    # -----------------------------
    all_x = np.concatenate([s[:, 0] for s in subsets]) if len(subsets) > 0 else np.array([])
    all_y = np.concatenate([s[:, 1] for s in subsets]) if len(subsets) > 0 else np.array([])

    if all_x.size > 0 and all_y.size > 0:
        min_x, max_x = float(all_x.min()), float(all_x.max())
        min_y, max_y = float(all_y.min()), float(all_y.max())
    else:
        # In case there's no data, default a range
        min_x, max_x = 0.0, 1.0
        min_y, max_y = 0.0, 1.0

    # If user gave an axis=[xmin,xmax,ymin,ymax], we can override or combine:
    # (But let's first compute a square bounding box from the actual data.)
    range_x = max_x - min_x
    range_y = max_y - min_y

    # If there's no variation in x, artificially expand
    if range_x == 0.0:
        range_x = 1.0  # e.g. add 1.0 for a visible domain
        min_x -= 0.5
        max_x += 0.5

    # If there's no variation in y, artificially expand
    if range_y == 0.0:
        range_y = 1.0
        min_y -= 0.5
        max_y += 0.5

    # Ensure a square domain
    range_ = max(range_x, range_y)

    # Center the domain around the data midpoint, but expand so it's square
    mid_x = 0.5 * (max_x + min_x)
    mid_y = 0.5 * (max_y + min_y)

    min_x = mid_x - 0.5 * range_
    max_x = mid_x + 0.5 * range_
    min_y = mid_y - 0.5 * range_
    max_y = mid_y + 0.5 * range_

    padding_fraction = 0.05  # 5% padding
    padding = padding_fraction * range_
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    # If the user explicitly gave axis=[xmin, xmax, ymin, ymax], we can
    # combine them by overriding or ignoring the above.
    # But typically, you'd prefer the code to do the square bounding box itself.
    # If you do want to force the user axis, uncomment:
    # if axis is not None:
    #     min_x, max_x, min_y, max_y = axis

    # -----------------------------
    # 3) Create figure layout
    #    2x2 GridSpec with smaller top row + smaller right column
    # -----------------------------
    fig = plt.figure(figsize=(8, 8))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 2,
        width_ratios=[4, 1],  # left is 4 units wide, right is 1 => smaller right
        height_ratios=[1, 4],  # top is 1 unit tall, bottom is 4 => smaller top
        wspace=0.05, hspace=0.05
    )

    ax_scatter = fig.add_subplot(gs[1, 0])  # bottom-left
    if colored:
        ax_histx = fig.add_subplot(gs[0, 0])  # top-left
        ax_histy = fig.add_subplot(gs[1, 1])  # bottom-right
        # Hide tick labels for the marginal axes
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)
        max_count_x = 0
        max_count_y = 0
    else:
        ax_histx = None
        ax_histy = None

    # -----------------------------
    # 4) Plot data
    # -----------------------------
    for i, subset in enumerate(subsets):
        x_vals = subset[:, 0]
        y_vals = subset[:, 1]

        if colored:
            ax_scatter.scatter(x_vals, y_vals, c=colors[i], alpha=0.9, rasterized=True)

            # Hist (with JAX)
            counts_x, bin_edges_x = jax_hist_1d(jnp.array(x_vals))
            counts_y, bin_edges_y = jax_hist_1d(jnp.array(y_vals))
            counts_x = np.asarray(counts_x)
            bin_edges_x = np.asarray(bin_edges_x)
            counts_y = np.asarray(counts_y)
            bin_edges_y = np.asarray(bin_edges_y)

            # Plot top histogram
            width_x = bin_edges_x[1:] - bin_edges_x[:-1]
            ax_histx.bar(bin_edges_x[:-1], counts_x,
                         width=width_x, color=colors[i], alpha=0.6)
            max_count_x = max(max_count_x, counts_x.max())

            # Plot right histogram
            width_y = bin_edges_y[1:] - bin_edges_y[:-1]
            ax_histy.barh(bin_edges_y[:-1], counts_y,
                          height=width_y, color=colors[i], alpha=0.6)
            max_count_y = max(max_count_y, counts_y.max())
        else:
            # Single scatter
            ax_scatter.scatter(x_vals, y_vals, alpha=0.1, rasterized=True)

    # -----------------------------
    # 5) Adjust main scatter axis
    #    -> strictly square domain
    # -----------------------------
    ax_scatter.set_xlim([min_x, max_x])
    ax_scatter.set_ylim([min_y, max_y])
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("x")
    ax_scatter.set_ylabel("y")

    # -----------------------------
    # 6) If colored, align hist axes
    # -----------------------------
    if colored:
        # The top histogram should cover the same x-range
        ax_histx.set_xlim(ax_scatter.get_xlim())
        # Then set its y-limit so the bars just fit
        ax_histx.set_ylim(0, max_count_x * 1.1 if max_count_x > 0 else 1)

        # The right histogram should cover the same y-range
        ax_histy.set_ylim(ax_scatter.get_ylim())
        # Then set its x-limit so the bars just fit
        ax_histy.set_xlim(0, max_count_y * 1.1 if max_count_y > 0 else 1)

    # -----------------------------
    # 7) (Optional) W&B logging + save
    # -----------------------------
    if tracking:
        wandb.log({str(title): wandb.Image(plt)}, commit=False)

    plt.savefig(filename)
    plt.close()


def compute_kernel_matrix(X, Y, bandwidth):
    """
    Computes the Gaussian kernel matrix between samples X and Y.

    Args:
        X: array of shape (n_samples_x, n_features)
        Y: array of shape (n_samples_y, n_features)
        bandwidth: float, bandwidth parameter for the Gaussian kernel

    Returns:
        Kernel matrix of shape (n_samples_x, n_samples_y)
    """
    XX = jnp.sum(X ** 2, axis=1).reshape(-1, 1)
    YY = jnp.sum(Y ** 2, axis=1).reshape(1, -1)
    distances = XX + YY - 2 * jnp.dot(X, Y.T)
    K = jnp.exp(-distances / (2 * bandwidth ** 2))
    return K


@jax.jit
def compute_mmd(X, Y):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples using a Gaussian kernel.

    Args:
        X: array of shape (n_samples_x, n_features)
        Y: array of shape (n_samples_y, n_features)
        bandwidth: float or None, bandwidth for the Gaussian kernel. If None, uses median heuristic.
        biased: bool, whether to use the biased estimator.

    Returns:
        MMD value (float)
    """
    # determine bandwidth fromd data
    Z = jnp.concatenate([X, Y], axis=0)
    distances = jnp.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    distances = jnp.sqrt(distances + 1e-8)  # Add epsilon to prevent sqrt(0)
    distances = distances[jnp.triu_indices(Z.shape[0], k=1)]
    median_bandwidth = jnp.median(distances)

    K_xx = compute_kernel_matrix(X, X, median_bandwidth)
    K_yy = compute_kernel_matrix(Y, Y, median_bandwidth)
    K_xy = compute_kernel_matrix(X, Y, median_bandwidth)

    n = X.shape[0]
    m = Y.shape[0]

    sum_K_xx = (jnp.sum(K_xx) - jnp.sum(jnp.diag(K_xx))) / (n * (n - 1))
    sum_K_yy = (jnp.sum(K_yy) - jnp.sum(jnp.diag(K_yy))) / (m * (m - 1))
    sum_K_xy = jnp.sum(K_xy) / (n * m)
    mmd_squared = sum_K_xx + sum_K_yy - 2 * sum_K_xy

    return jnp.sqrt(jnp.maximum(mmd_squared, 0))


@jax.jit
def compute_wasserstein_distance(X, Y, epsilon=1e-2, num_iters=100):
    """
    Computes the approximate Wasserstein distance between two empirical distributions
    using the Sinkhorn algorithm.

    Args:
        X: array of shape (n_samples_x, n_features)
        Y: array of shape (n_samples_y, n_features)
        epsilon: float, the entropic regularization parameter
        num_iters: int, number of Sinkhorn iterations

    Returns:
        Wasserstein distance (float)
    """
    n = X.shape[0]
    m = Y.shape[0]
    a = jnp.ones(n) / n  # Uniform weights for X
    b = jnp.ones(m) / m  # Uniform weights for Y

    # compute pairwise distance
    X_norm = jnp.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = jnp.sum(Y ** 2, axis=1).reshape(1, -1)
    distances = X_norm + Y_norm - 2 * jnp.dot(X, Y.T)
    M = jnp.maximum(distances, 0.0)  # Ensure non-negative distances
    M = jnp.sqrt(M + 1e-8)  # Compute the ground cost (Euclidean distance)

    # compute sinkhorn_distance
    n, m = M.shape
    # Initialize log(u) and log(v)
    log_u = jnp.zeros(n)
    log_v = jnp.zeros(m)

    # Precompute kernel in log-space
    K = -M / epsilon  # Note: K = exp(-M / epsilon)

    # Stabilize K by subtracting its maximum value
    K_max = jnp.max(K)
    K -= K_max

    for _ in range(num_iters):
        # Update log_u and log_v using log-sum-exp for numerical stability
        log_u = jnp.log(a) - jax.scipy.special.logsumexp(K + log_v[None, :], axis=1)
        log_v = jnp.log(b) - jax.scipy.special.logsumexp(K.T + log_u[None, :], axis=1)

    # Compute the transportation plan in log-space
    log_transport_plan = K + log_u[:, None] + log_v[None, :]
    transport_plan = jnp.exp(log_transport_plan)

    distance = jnp.sum(transport_plan * M)
    return distance


@jax.jit
def pairwise_distances(X, Y):
    """
    Compute pairwise Euclidean distances between two sets of points X and Y.
    X shape: (n, d)
    Y shape: (m, d)
    Returns a matrix of shape (n, m), where entry (i, j) is ||X[i] - Y[j]||.
    """
    # Expand X and Y to allow broadcasting:
    # X[:, None, :] -> shape (n, 1, d)
    # Y[None, :, :] -> shape (1, m, d)
    # Subtract and square, then sum over the last axis (d), take sqrt.
    diff = X[:, None, :] - Y[None, :, :]
    dist_matrix = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    return dist_matrix


@jax.jit
def energy_distance(X, Y):
    """
    Compute the sample Energy Distance between two datasets X and Y.
    X shape: (n, d)
    Y shape: (m, d)

    Energy Distance formula (for Euclidean distances):
        E(X, Y) = 2/(n*m) * sum(||X_i - Y_j||)
                  - 1/(n^2)   * sum(||X_i - X_k||)
                  - 1/(m^2)   * sum(||Y_j - Y_l||)
    """
    n, _ = X.shape
    m, _ = Y.shape

    # Pairwise distances
    XY = pairwise_distances(X, Y)
    XX = pairwise_distances(X, X)
    YY = pairwise_distances(Y, Y)

    # Compute sums
    sum_XY = jnp.sum(XY)
    sum_XX = jnp.sum(XX)
    sum_YY = jnp.sum(YY)

    # Energy distance statistic
    ed = (2.0 / (n * m)) * sum_XY \
         - (1.0 / (n ** 2)) * sum_XX \
         - (1.0 / (m ** 2)) * sum_YY
    return ed


@jax.jit
def energy_test(X, Y, num_permutations=100, seed=0):
    """
    Perform a permutation-based Energy Test to determine whether two
    samples X and Y come from the same distribution.

    Parameters
    ----------
    X : jnp.ndarray, shape (n, d)
        First dataset.
    Y : jnp.ndarray, shape (m, d)
        Second dataset.
    num_permutations : int, optional
        Number of permutations for the permutation test.
    seed : int, optional
        Seed for random number generation.

    Returns
    -------
    energy_stat : float
        Observed energy distance statistic between X and Y.
    p_value : float
        Approximate p-value from the permutation test.
    """
    # Convert to JAX arrays if not already
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    # Observed statistic
    observed_stat = energy_distance(X, Y)

    # Concatenate data
    Z = jnp.concatenate([X, Y], axis=0)
    n = X.shape[0]
    m = Y.shape[0]

    # Random keys
    key = jax.random.PRNGKey(seed)

    # Function to compute a permuted statistic
    def permuted_stat(k):
        # Permute along the 0th axis
        perm = jax.random.permutation(k, Z, independent=True)
        Xp = perm[:n, :]
        Yp = perm[n:, :]
        return energy_distance(Xp, Yp)

    # Generate a batch of random keys for vmap
    keys = jax.random.split(key, num_permutations)

    # Compute distribution of permuted statistics
    perm_stats = jax.vmap(permuted_stat)(keys)

    # Compute p-value: fraction of permuted stats >= observed
    p_value = jnp.mean(perm_stats >= observed_stat)

    return observed_stat, p_value


def fit_multivariate_gaussian(X):
    """
    Fit a single multivariate normal distribution to the data X.
    Returns the mean vector and covariance matrix.

    Parameters
    ----------
    X : jnp.ndarray of shape (n, d)
        The dataset (n samples, d features).

    Returns
    -------
    mu : jnp.ndarray of shape (d,)
        Mean of the Gaussian.
    Sigma : jnp.ndarray of shape (d, d)
        Covariance of the Gaussian.
    """
    mu = jnp.mean(X, axis=0)
    Sigma = jnp.cov(X, rowvar=False)
    return mu, Sigma


def conditional_gaussian_params(mu, Sigma, i, value):
    """
    Compute parameters of p(X_{-i} | X_i = value).

    This returns the mean and covariance of the (d-1)-dimensional
    Gaussian for X_{-i} given X_i = value.

    Parameters
    ----------
    mu : np.ndarray of shape (d,)
        Mean of the full Gaussian.
    Sigma : np.ndarray of shape (d, d)
        Covariance of the full Gaussian.
    i : int
        The index of the dimension to condition on.
    value : float
        The value we set for X_i.

    Returns
    -------
    cond_mu : np.ndarray of shape (d-1,)
        Conditional mean of X_{-i} given X_i = value.
    cond_Sigma : np.ndarray of shape (d-1, d-1)
        Conditional covariance of X_{-i} given X_i = value.
    """
    d = len(mu)
    # Indices of the "other" dimensions
    other_idx = jnp.array([k for k in range(d) if k != i])

    # Partition mean
    mu_i = mu[i]  # scalar
    mu_others = mu[other_idx]  # shape (d-1,)

    # Partition covariance
    # Sigma_{ii} is scalar if we condition on a single variable
    Sigma_ii = Sigma[i, i]  # shape ()
    Sigma_i_others = Sigma[i, other_idx]  # shape (d-1,)
    Sigma_others_i = Sigma[other_idx, i]  # shape (d-1,)
    Sigma_others_others = Sigma[other_idx][:, other_idx]  # shape (d-1, d-1)

    # Compute conditional mean
    # cond_mu = mu_others + Sigma_{-i,i} * Sigma_{ii}^{-1} * (value - mu_i)
    cond_mu = mu_others + Sigma_others_i * (1.0 / Sigma_ii) * (value - mu_i)

    # Compute conditional covariance
    # cond_Sigma = Sigma_{-i,-i} - Sigma_{-i,i} * Sigma_{ii}^{-1} * Sigma_{i,-i}
    cond_Sigma = Sigma_others_others - jnp.outer(Sigma_others_i, Sigma_i_others) * (1.0 / Sigma_ii)

    return cond_mu, cond_Sigma


def sample_gaussian(mu, Sigma, n_samples=1, random_state=None):
    """
    Sample from a multivariate normal with given mean and covariance.

    Parameters
    ----------
    mu : np.ndarray of shape (d,)
    Sigma : np.ndarray of shape (d, d)
    n_samples : int
    random_state : int or None

    Returns
    -------
    samples : np.ndarray of shape (n_samples, d)
    """
    rng = default_rng(random_state)
    return rng.multivariate_normal(mu, Sigma, size=n_samples)


def negative_log_likelihood_mvg(X, mu, Sigma):
    """
    Compute the negative log-likelihood of one or more samples X
    under the multivariate Gaussian N(mu, Sigma).

    Parameters
    ----------
    X : np.ndarray
        Either shape (d,) for a single sample or (n, d) for multiple samples.
    mu : np.ndarray of shape (d,)
        Mean of the Gaussian.
    Sigma : np.ndarray of shape (d, d)
        Covariance of the Gaussian.

    Returns
    -------
    nll : float or np.ndarray
        - If X is (d,), returns a scalar (the NLL of one sample).
        - If X is (n, d), returns shape (n,) (the NLL for each sample).

    Formula (for each sample x):
        NLL(x; mu, Sigma) = 1/2 * [ (x - mu)^T Σ^-1 (x - mu)
                                    + log(det(Σ))
                                    + d * log(2π) ]
    """
    # Ensure X is 2D: (n, d)
    X = jnp.atleast_2d(X)  # if X was (d,), it becomes (1, d)

    n, d = X.shape
    if mu.shape[0] != d or Sigma.shape != (d, d):
        raise ValueError("Shapes of X, mu, and Sigma do not match.")

    # Compute the difference (X - mu) for all samples
    diff = X - mu[jnp.newaxis, :]  # shape (n, d)

    # Invert the covariance
    inv_Sigma = jnp.linalg.inv(Sigma)

    # Compute the quadratic term for each sample:
    # (x - mu).T @ inv_Sigma @ (x - mu)
    # Using einsum, shape( n ):
    quad_term = jnp.einsum('ni,ij,nj->n', diff, inv_Sigma, diff)

    # Log of the determinant of Sigma (scalar)
    logdet = jnp.log(jnp.linalg.det(Sigma))

    # The negative log-likelihood for each sample
    nll = 0.5 * (quad_term + logdet + d * jnp.log(2.0 * jnp.pi))

    # If the original X was a single sample (d,), return a float instead of array
    if n == 1:
        return nll[0]
    else:
        return nll


def plot_distrax_contours(
    dists,
    grid_points: int = 200,
    levels=15,
    colors=None,
    labels=None,
    ax=None,
):
    """
    Overlay contour lines of multiple 2‑D distrax distributions.

    Parameters
    ----------
    dists      : list[distrax.Distribution]
                 Each must have event_shape == (2,).
    grid_points: int   – number of evaluation points per axis (default 200).
    levels     : int or 1‑D sequence – passed to `ax.contour`.
    colors     : list[str] – optional colour cycle.
    labels     : list[str] – legend labels, same length as `dists` (optional).
    ax         : matplotlib.axes.Axes – draw on this axis if provided.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Global plotting window:  ±4σ around all means (heuristic)
    locs = jnp.stack([d.mean() for d in dists])
    scales = jnp.stack([jnp.sqrt(d.variance()) for d in dists])
    lo = jnp.min(locs - 5 * scales, axis=0)
    hi = jnp.max(locs + 5 * scales, axis=0)

    xs = jnp.linspace(lo[0], hi[0], grid_points)
    ys = jnp.linspace(lo[1], hi[1], grid_points)
    X, Y = jnp.meshgrid(xs, ys)
    XY = jnp.stack([X, Y], axis=-1)      # (gp, gp, 2)

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Draw one contour set per distribution
    for i, dist in enumerate(dists):
        Z = dist.prob(XY)                # PDF evaluated on grid
        ax.contour(
            X,
            Y,
            Z,
            levels=levels,
            colors=[colors[i % len(colors)]],
            linewidths=1.2,
        )

    # Optional legend
    if labels:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], color=colors[i % len(colors)], lw=2)
            for i in range(len(dists))
        ]
        ax.legend(handles, labels, loc="best")

    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.set_aspect("equal")
    ax.relim()  # recompute data limits
    ax.autoscale_view()
    return ax

if __name__ == "__main__":
    print()
