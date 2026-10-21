"""
Evaluate ACTIVA on the non-identifiability dataset produced by `gen_ambiguous_scms.py`.

Every row of that dataset carries the same observational context and the same
intervention mask; rows differ only in which SCM (A: X->Y, or B: Y->X) generated their
interventional data. Two paths through the model behave very differently under do(X = v):

  posterior path  q(z | x, dm) -> decode.  The encoder sees the interventional data x,
                  so it *can* tell A from B. Reconstructions should track the branch
                  that produced them. This is the control that shows the decoder is
                  capable of representing both branches.

  prior path      p(z | dm, I) -> decode.  Sees only the shared context and mask, which
                  are byte-identical for A and B rows. It cannot identify the SCM, so
                  the Bayes-optimal predictive is the *mixture* of the two branches.
                  This is the path used at inference time by `model.inference`.

The same comparison is run twice: once in data space (the decoded interventional
distribution) and once in latent space. The latent-space version is the direct test of
the theoretical claim that the conditional prior matches the aggregate posterior: with
two branches present at rate p and 1-p, the aggregate posterior is a two-cluster
distribution, and the prior GMM should reproduce both clusters and their weights.

Both the prior and the posterior use a fixed 0.1*I covariance in latent space
(see `ConditionalVAE.encode_prior` / `get_encoded_distribution`), so their components
are directly comparable.

Outputs (one figure per file, as <prefix>_<name>.png and .pdf):
  <prefix>_data_posterior   decoded distribution, posterior path, vs ground truth
  <prefix>_data_prior       decoded distribution, prior path, vs weighted ground truth
  <prefix>_latent_z<var>    latent marginal per dimension: both posteriors vs the prior

Example (the configuration of Sec. 7.4 of the paper):
    python evaluate_ambiguous.py --data-path data/ambiguous_2var_75 \
        --model-path models/TMLR/ambiguous_2var_75_best --output-prefix ambiguous_75 \
        --samples-from-prior 50000
"""

import argparse
import json

import distrax
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import load_model_state

X, Y = 0, 1
SCM_A, SCM_B = 0, 1
VAR_NAMES = ('X', 'Y')

# Colour roles are fixed across every figure: the two SCMs and the mixture.
# Taken from METHOD_COLORS in plot_causal_results.py (the paper's main-figure palette)
# so these figures sit alongside the rest of the paper without a palette clash.
COL_A = '#DD8452'      # orange  - SCM A (X->Y)
COL_B = '#8172B3'      # purple  - SCM B (Y->X)
COL_MIX = '#55A868'    # green   - the mixture (prior path / conditional prior)

# Latent dimensions whose two branch means sit closer than this many component standard
# deviations carry no information about which SCM produced the data, so plotting them
# says nothing about the question; they are reported numerically instead.
SEPARATION_THRESHOLD = 1.0

LATENT_COV = 0.1       # fixed latent covariance shared by prior and posterior


# --------------------------------------------------------------------------------------
# component bookkeeping
# --------------------------------------------------------------------------------------

def flatten_prior_mixture(dec_weights, dec_means, dec_covs, dim):
    """Flatten the two-level inference mixture into a flat 1-D GMM over `dim`.

    `model.inference` builds a MixtureSameFamily whose outer categorical is uniform
    over the `s` prior samples and whose inner categorical holds the decoder weights.
    The marginal over one coordinate is therefore a flat GMM with s * n_components
    Gaussians and weights (1/s) * w_c. Marginalising a Gaussian mixture onto one
    coordinate just reads off that coordinate's mean and diagonal variance entry.

    Args:
        dec_weights: (s, batch, n_components)
        dec_means:   (s, batch, n_components, n_vars)
        dec_covs:    (s, batch, n_components, n_vars, n_vars)

    Returns:
        (weights, means, variances), each (batch, s * n_components), weights sum to 1.
    """
    s, b, k = dec_weights.shape
    weights = (dec_weights / s).transpose(1, 0, 2).reshape(b, s * k)
    means = dec_means[..., dim].transpose(1, 0, 2).reshape(b, s * k)
    variances = dec_covs[..., dim, dim].transpose(1, 0, 2).reshape(b, s * k)
    # Cast out of the model's bfloat16 compute dtype: it has ~3 decimal digits of
    # precision and its scalars do not support the format codes used when reporting.
    return tuple(np.asarray(a, dtype=np.float64) for a in (weights, means, variances))


def summarise_modes(weights, means, variances, assignment):
    """Summarise a 1-D GMM per mode, given a component->mode assignment.

    Per-mode variance uses the law of total variance over the components assigned to
    that mode, so it accounts for both component widths and their spread:
        Var = sum_i w_i (s_i^2 + m_i^2) / pi  -  mu^2
    """
    weights = np.asarray(weights).ravel()
    means = np.asarray(means).ravel()
    variances = np.asarray(variances).ravel()
    assignment = np.asarray(assignment).ravel()
    weights = weights / weights.sum()

    out = []
    for m_idx in range(2):
        sel = assignment == m_idx
        mass = weights[sel].sum()
        if mass <= 0:
            out.append({'mass': 0.0, 'mean': np.nan, 'var': np.nan, 'n_components': 0})
            continue
        w = weights[sel] / mass
        mu = float((w * means[sel]).sum())
        second = float((w * (variances[sel] + means[sel] ** 2)).sum())
        out.append({'mass': float(mass), 'mean': mu, 'var': second - mu ** 2,
                    'n_components': int(sel.sum())})
    return out


def assign_1d(means, centres):
    """Assign each component to the nearest of two 1-D centres."""
    return np.argmin(np.abs(np.asarray(means).ravel()[:, None] - np.asarray(centres)[None, :]),
                     axis=1)


def assign_nd(means, centres):
    """Assign each component to the nearest of two centres in full latent space."""
    means = np.asarray(means)
    centres = np.asarray(centres)
    d = np.linalg.norm(means[:, None, :] - centres[None, :, :], axis=-1)
    return np.argmin(d, axis=1)


def per_prior_sample_mass(dec_weights, dec_means, dim, mode_centres):
    """Mass each *individual* decoded GMM puts on the first mode.

    The inference mixture is two-level: an outer uniform mixture over prior latents,
    each decoding to its own GMM. A 50/50 aggregate can arise two ways, and they are
    mechanistically different:

      - every decoded GMM is itself bimodal (~0.5 each) -> the decoder hedges;
      - each decoded GMM commits to one branch (~0 or ~1) and the prior spreads latents
        across both regions -> the prior hedges.

    `prior_dist.sample` draws an independent latent per batch element, so the decoded
    GMMs number s * batch, not s. All batch rows carry identical inputs here, so they
    are s * batch genuine draws from the same prior and are pooled.
    """
    s, b, k = dec_weights.shape
    w = np.asarray(dec_weights, dtype=np.float64).reshape(s * b, k)
    m = np.asarray(dec_means[..., dim], dtype=np.float64).reshape(s * b, k)
    w = w / w.sum(axis=1, keepdims=True)
    assignment = np.argmin(np.abs(m[..., None] - np.asarray(mode_centres)[None, None, :]), axis=-1)
    return (w * (assignment == 0)).sum(axis=1)


def top_components(weights, means, variances, n=8):
    """Return the n heaviest individual components as (weight, mean, var) tuples."""
    w, m, v = np.asarray(weights).ravel(), np.asarray(means).ravel(), np.asarray(variances).ravel()
    w = w / w.sum()
    order = np.argsort(w)[::-1][:n]
    return [(float(w[i]), float(m[i]), float(v[i])) for i in order]


# --------------------------------------------------------------------------------------
# model paths
# --------------------------------------------------------------------------------------

def posterior_path(model, state, dm, x, intervention_index, intervention_targets, rng, n_samples):
    """Run q(z | x, dm), returning latent means, latent samples and decoded samples."""
    rng, callkey, dropoutkey, zkey, samplekey = jax.random.split(rng, 5)

    z_mean, z_logvar = model.apply(
        {'params': state.params},
        x=x, dm=dm,
        intervention_index=intervention_index,
        intervention_targets=intervention_targets,
        is_training=False,
        rngs={'dropout': dropoutkey},
        method=model.encode,
    )
    z_mean = np.asarray(z_mean, dtype=np.float64)          # (n_rows, n_vars)
    z_samples = (z_mean[None] + np.sqrt(LATENT_COV)
                 * np.asarray(jax.random.normal(zkey, (n_samples,) + z_mean.shape)))

    dec_w, dec_m, dec_c, _, _ = model.apply(
        {'params': state.params},
        x=x, dm=dm,
        intervention_index=intervention_index,
        intervention_targets=intervention_targets,
        is_training=False,
        rng=callkey,
        rngs={'dropout': dropoutkey},
    )
    dist = distrax.MixtureSameFamily(
        distrax.Categorical(probs=dec_w),
        distrax.MultivariateNormalFullCovariance(loc=dec_m, covariance_matrix=dec_c))
    decoded = np.asarray(dist.sample(seed=samplekey, sample_shape=(n_samples,)))
    return z_mean, z_samples.reshape(-1, z_mean.shape[-1]), decoded


def prior_path(model, state, dm, intervention_index, intervention_targets, rng,
               samples_from_prior, n_samples):
    """Run p(z | dm, I) -> decode, returning samples and raw component parameters.

    This mirrors `model.inference` step by step rather than calling it, because the
    quantitative half of this script needs the individual decoded components, which
    the returned distrax object does not expose conveniently.
    """
    rng, dropoutkey, latentkey, zkey, samplekey = jax.random.split(rng, 5)
    n_rows = dm.shape[0]

    # Prior parameters on the full batch, so the determinism check downstream can verify
    # that every row really does produce identical parameters.
    pri_w, pri_m, pri_c = model.apply(
        {'params': state.params},
        dm=dm,
        intervention_index=intervention_index,
        intervention_targets=intervention_targets,
        is_training=False,
        rngs={'dropout': dropoutkey},
        method=model.encode_prior,
    )
    # Sample from a single representative row. Every row carries byte-identical inputs, so
    # they all define the same prior; using one row makes `samples_from_prior` the literal
    # number of prior latents rather than that number per row, which keeps the Monte-Carlo
    # sample size explicit and the decode cheap enough to push it high.
    prior_dist = model.apply(
        {'params': state.params}, weights=pri_w[:1], means=pri_m[:1], covs=pri_c[:1],
        method=model.get_conditional_prior)

    # Latent samples for the latent-space comparison, kept separate from the latents
    # used to decode so the two analyses do not share sampling noise. The total is held at
    # n_samples * n_rows so histogram resolution does not depend on the batch layout.
    z_prior = np.asarray(prior_dist.sample(seed=zkey, sample_shape=(n_samples * n_rows,)))
    z_prior = z_prior.reshape(-1, z_prior.shape[-1])

    prior_samples = prior_dist.sample(seed=latentkey, sample_shape=(samples_from_prior,))

    def decode_one(z):
        return model.apply({'params': state.params}, z=z, is_training=False,
                           rngs={'dropout': dropoutkey}, method=model.decode)

    dec_w, dec_m, dec_c = jax.vmap(decode_one)(prior_samples)

    decoded = np.asarray(sample_two_level(dec_w, dec_m, dec_c, n_samples * n_rows, samplekey))
    prior_gmm = (np.asarray(pri_w, dtype=np.float64), np.asarray(pri_m, dtype=np.float64))
    return decoded, z_prior, (dec_w, dec_m, dec_c), prior_gmm


def sample_two_level(dec_w, dec_m, dec_c, n_total, rng):
    """Sample the two-level inference mixture in O(n_total) rather than O(n_total * s).

    `model.inference` assembles a distrax MixtureSameFamily whose outer categorical is
    uniform over the s prior latents. Sampling that object materialises every component
    before gathering, which costs O(n_total * s) and becomes intractable once s is large
    enough to make the Monte-Carlo error negligible. Drawing the latent index first and
    only then the decoder component is equivalent and costs O(n_total).
    """
    w = jnp.asarray(dec_w[:, 0], dtype=jnp.float32)      # (s, k)
    m = jnp.asarray(dec_m[:, 0], dtype=jnp.float32)      # (s, k, d)
    c = jnp.asarray(dec_c[:, 0], dtype=jnp.float32)      # (s, k, d, d)
    k_latent, k_comp, k_noise = jax.random.split(rng, 3)

    idx_s = jax.random.randint(k_latent, (n_total,), 0, w.shape[0])   # uniform outer mixture
    idx_k = jax.random.categorical(k_comp, jnp.log(w[idx_s] + 1e-30))  # decoder component
    mu = m[idx_s, idx_k]
    cov = c[idx_s, idx_k]
    chol = jnp.linalg.cholesky(cov)
    eps = jax.random.normal(k_noise, mu.shape)
    return mu + jnp.einsum('nij,nj->ni', chol, eps)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path', type=str, default='data/ambiguous_2var',
                        help='Base path; _test.npz and _meta.json are appended.')
    parser.add_argument('--model-path', type=str, default='models/TMLR/ambiguous_2var')
    parser.add_argument('--output-prefix', type=str, default='ambiguous',
                        help='Figures are written as <prefix>_<name>.png and .pdf.')
    parser.add_argument('--target-var', type=int, default=X,
                        help='Intervened variable index (0 = X, 1 = Y). Default: X.')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Samples drawn per row from each predicted distribution.')
    parser.add_argument('--samples-from-prior', type=int, default=10,
                        help='Number of prior latents decoded to form the inference mixture. '
                             'This is the Monte-Carlo sample size behind the data-space '
                             'mixture-weight estimate, whose sd scales as 1/sqrt(this).')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    tv = args.target_var
    outcome = Y if tv == X else X

    with np.load(args.data_path + '_test.npz') as d:
        input_data = jnp.asarray(d['input'], dtype=jnp.float32)
        target_data = jnp.asarray(d['target_data'], dtype=jnp.float32)
        targets = jnp.asarray(d['targets'], dtype=jnp.float32)
        scm_ids = np.asarray(d['scm_id'])

    with open(args.data_path + '_meta.json') as f:
        meta = json.load(f)
    v = meta['args']['intervention_value']
    gt = meta['interventional'][f'do({VAR_NAMES[tv]})']
    gt_a = tuple(gt['A'][VAR_NAMES[outcome]])
    gt_b = tuple(gt['B'][VAR_NAMES[outcome]])
    p_a = meta['mixture_weights']['A']

    model, state = load_model_state(args.model_path)
    rng = jax.random.PRNGKey(args.seed)

    rows = {SCM_A: np.flatnonzero(scm_ids == SCM_A), SCM_B: np.flatnonzero(scm_ids == SCM_B)}
    print(f"Test rows: SCM A = {len(rows[SCM_A])}, SCM B = {len(rows[SCM_B])}")

    assert bool(jnp.all(input_data == input_data[0])), "obs contexts differ across test rows"
    assert bool(jnp.all(targets == targets[0])), "intervention masks differ across test rows"
    print("Verified: observational context and intervention mask identical across all rows.\n")

    # One shared key for the prior path across both branches. The prior sees identical
    # inputs for A-rows and B-rows, so with the same key its output must be bit-identical;
    # a per-branch key would make them differ through sampling noise alone and destroy
    # the determinism check below.
    rng, k_prior = jax.random.split(rng)

    results = {}
    for sid in (SCM_A, SCM_B):
        idx = rows[sid]
        dm = input_data[idx]
        x = target_data[0, idx, tv]                      # (n_rows, n_samples, n_vars)
        int_targets = targets[idx, 0, tv]                # (n_rows, n_vars)
        int_index = jnp.zeros(len(idx), dtype=jnp.float32)

        rng, k_post = jax.random.split(rng)
        z_mean, z_post, post = posterior_path(
            model, state, dm, x, int_index, int_targets, k_post, args.n_samples)
        pri, z_prior, dec_params, prior_gmm = prior_path(
            model, state, dm, int_index, int_targets, k_prior,
            args.samples_from_prior, args.n_samples)

        results[sid] = {
            'ground_truth': np.asarray(x)[..., outcome].ravel(),
            'posterior': post[..., outcome].ravel(),
            'prior': pri[..., outcome].ravel(),
            'z_mean': z_mean,
            'z_posterior': z_post,
            'z_prior': z_prior,
            'components': flatten_prior_mixture(*dec_params, dim=outcome),
            'raw_components': dec_params,
            'prior_gmm': prior_gmm,
        }

    _report_data(results, gt_a, gt_b, p_a, v, tv, outcome)
    separating_dims = _report_latent(results, p_a)

    _plot_data_posterior(results, v, tv, outcome, args.output_prefix)
    _plot_data_prior(results, p_a, v, tv, outcome, args.output_prefix)
    _plot_latent(results, p_a, separating_dims, v, tv, args.output_prefix)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

BAR = '=' * 96


def _report_data(results, gt_a, gt_b, p_a, v, tv, outcome):
    ivar, ovar = VAR_NAMES[tv], VAR_NAMES[outcome]

    print(BAR)
    print(f"DATA SPACE — empirical moments of {ovar} under do({ivar} = {v})")
    print(BAR)
    print(f"{'source':<34} {'mean':>10} {'var':>10}")
    print('-' * 96)
    for sid, name in ((SCM_A, 'A'), (SCM_B, 'B')):
        r = results[sid]
        for label, key in (('ground truth', 'ground_truth'), ('posterior path', 'posterior'),
                           ('prior path', 'prior')):
            a = r[key]
            print(f"  SCM {name} rows / {label:<20} {a.mean():>10.3f} {a.var():>10.3f}")
    print()
    print(f"  analytic  branch A                 {gt_a[0]:>10.3f} {gt_a[1]:>10.3f}")
    print(f"  analytic  branch B                 {gt_b[0]:>10.3f} {gt_b[1]:>10.3f}")
    mix_mean = p_a * gt_a[0] + (1 - p_a) * gt_b[0]
    mix_var = (p_a * (gt_a[1] + gt_a[0] ** 2) + (1 - p_a) * (gt_b[1] + gt_b[0] ** 2)) - mix_mean ** 2
    print(f"  analytic  mixture                  {mix_mean:>10.3f} {mix_var:>10.3f}")

    wa, ma, va = results[SCM_A]['components']
    wb, mb, vb = results[SCM_B]['components']
    pw_a, pm_a = results[SCM_A]['prior_gmm']
    pw_b, pm_b = results[SCM_B]['prior_gmm']

    # The two branches need not contain the same number of rows (they do not at an
    # asymmetric mixture weight), so compare row 0 of each rather than elementwise.
    # Every row within a branch carries identical inputs, which is checked separately.
    d_within = max(float(np.abs(pw_a - pw_a[0]).max()), float(np.abs(pm_a - pm_a[0]).max()),
                   float(np.abs(pw_b - pw_b[0]).max()), float(np.abs(pm_b - pm_b[0]).max()))
    d_prior = max(float(np.abs(pw_a[0] - pw_b[0]).max()), float(np.abs(pm_a[0] - pm_b[0]).max()))
    print("\nPrior determinism (must be 0 — inputs are byte-identical across every row):")
    print(f"  max |row - row 0| within each branch:    {d_within:.3e}")
    print(f"  max |A - B| over prior GMM params:       {d_prior:.3e}")
    print("  (decoded components are not compared across branches: the branches hold")
    print("   different row counts, so they draw different numbers of prior latents.)")

    centres = [gt_a[0], gt_b[0]]
    modes = summarise_modes(wa, ma, va, assign_1d(ma, centres))
    print()
    print(BAR)
    print(f"DATA SPACE — prior-path mixture vs. ground-truth modes")
    print(BAR)
    print(f"{'mode':<26} {'mass':>10} {'mean':>10} {'var':>10}   {'#comp':>6}")
    print('-' * 96)
    for m, (label, truth, true_mass) in zip(
            modes,
            ((f'branch A (near {gt_a[0]:+.2f})', gt_a, p_a),
             (f'branch B (near {gt_b[0]:+.2f})', gt_b, 1 - p_a))):
        print(f"  {label:<24} {m['mass']:>10.3f} {m['mean']:>10.3f} {m['var']:>10.3f}   "
              f"{m['n_components']:>6d}")
        print(f"  {'  ground truth':<24} {true_mass:>10.3f} {truth[0]:>10.3f} {truth[1]:>10.3f}")
        print(f"  {'  error':<24} {m['mass'] - true_mass:>+10.3f} {m['mean'] - truth[0]:>+10.3f} "
              f"{m['var'] - truth[1]:>+10.3f}")

    print()
    print(BAR)
    print("DATA SPACE — mixture weights vs. the simulator")
    print(BAR)
    samples = results[SCM_A]['prior']
    emp_a = float(np.mean(np.abs(samples - centres[0]) < np.abs(samples - centres[1])))
    print(f"{'estimator':<46} {'branch A':>10} {'branch B':>10} {'|err|':>8}")
    print('-' * 96)
    print(f"  {'simulator (ground truth)':<44} {p_a:>10.3f} {1 - p_a:>10.3f} {'-':>8}")
    print(f"  {'summed component weights':<44} {modes[0]['mass']:>10.3f} {modes[1]['mass']:>10.3f} "
          f"{abs(modes[0]['mass'] - p_a):>8.3f}")
    print(f"  {'sample fraction nearest each mode':<44} {emp_a:>10.3f} {1 - emp_a:>10.3f} "
          f"{abs(emp_a - p_a):>8.3f}")
    print("\n  The summed component weights are the exact quantity to compare against the")
    print("  simulator. The sample fraction is a coarser check: it is biased by mode overlap,")
    print("  since branch B has variance 1.0 and so genuinely puts some mass nearer A's centre.")

    print()
    print(BAR)
    print("DATA SPACE — where does the hedging live?")
    print(BAR)
    dec_w, dec_m, _ = results[SCM_A]['raw_components']
    per_s = per_prior_sample_mass(dec_w, dec_m, outcome, centres)
    committed_a = float(np.mean(per_s > 0.95))
    committed_b = float(np.mean(per_s < 0.05))
    print(f"Branch-A mass of each individually decoded GMM ({per_s.size} prior latents):")
    print(f"  mean {per_s.mean():.3f}   std {per_s.std():.3f}   "
          f"min {per_s.min():.3f}   max {per_s.max():.3f}")
    print(f"  committed to A (mass > 0.95): {committed_a:>6.1%}")
    print(f"  committed to B (mass < 0.05): {committed_b:>6.1%}")
    print(f"  genuinely bimodal (in between): {1 - committed_a - committed_b:>5.1%}")
    print(f"\n  (mean must equal the summed component weight for branch A above: "
          f"{modes[0]['mass']:.3f})")
    if per_s.min() > 0.15 and per_s.max() < 0.85:
        print("  -> every decoded GMM is itself bimodal: the DECODER carries the hedging.")
    elif per_s.std() > 0.35:
        print("  -> individual GMMs commit to one branch; the mixture comes from the PRIOR")
        print("     spreading latents across both regions.")
    else:
        print("  -> mixed: hedging is shared between the prior and the decoder.")

    print()
    print(BAR)
    print("DATA SPACE — heaviest individual components of the prior-path mixture")
    print(BAR)
    print(f"{'weight':>10} {'mean':>10} {'var':>10}")
    print('-' * 96)
    for w, m, s2 in top_components(wa, ma, va, n=10):
        print(f"{w:>10.2e} {m:>10.3f} {s2:>10.3f}")
    print()


def _report_latent(results, p_a):
    """Compare the conditional prior against the aggregate posterior in latent space."""
    z_a = results[SCM_A]['z_mean']          # (n_rows, n_vars)
    z_b = results[SCM_B]['z_mean']
    n_vars = z_a.shape[-1]
    centres = np.stack([z_a.mean(axis=0), z_b.mean(axis=0)])   # (2, n_vars)
    pri_w, pri_m = results[SCM_A]['prior_gmm']                 # (b, K), (b, K, n_vars)

    print(BAR)
    print("LATENT SPACE — posterior cluster centres (mean of q(z|x,dm) means per branch)")
    print(BAR)
    print(f"{'branch':<20} " + " ".join(f"{'z[' + VAR_NAMES[d] + ']':>12}" for d in range(n_vars))
          + f" {'within-branch sd':>18}")
    print('-' * 96)
    for name, z in (('A (X->Y)', z_a), ('B (Y->X)', z_b)):
        print(f"  {name:<18} " + " ".join(f"{z.mean(axis=0)[d]:>12.3f}" for d in range(n_vars))
              + f" {float(z.std(axis=0).mean()):>18.4f}")
    sep = float(np.linalg.norm(centres[0] - centres[1]))
    print(f"\n  cluster separation ||z_A - z_B|| = {sep:.3f}   "
          f"(latent component sd = {np.sqrt(LATENT_COV):.3f})")
    print(f"  separation in units of component sd: {sep / np.sqrt(LATENT_COV):.1f}")

    # Per-dimension separation. Where the branch means coincide, that coordinate carries
    # no information about which SCM generated the data — the ambiguity is confined to
    # the coordinates listed as separating.
    per_dim = np.abs(centres[0] - centres[1]) / np.sqrt(LATENT_COV)
    separating = [d for d in range(n_vars) if per_dim[d] >= SEPARATION_THRESHOLD]
    overlapping = [d for d in range(n_vars) if d not in separating]
    print(f"\n{'dimension':<14} {'mean A':>10} {'mean B':>10} {'|diff|':>10} "
          f"{'in comp. sd':>13}   verdict")
    print('-' * 96)
    for d in range(n_vars):
        verdict = 'separates branches' if d in separating else 'means overlap'
        print(f"  {'z[' + VAR_NAMES[d] + ']':<12} {centres[0][d]:>10.3f} {centres[1][d]:>10.3f} "
              f"{abs(centres[0][d] - centres[1][d]):>10.3f} {per_dim[d]:>13.1f}   {verdict}")
    if overlapping:
        print(f"\n  The branch means coincide in "
              f"{', '.join('z[' + VAR_NAMES[d] + ']' for d in overlapping)}: the model places")
        print("  both SCMs at the same point there, so SCM identity is encoded entirely in "
              f"{', '.join('z[' + VAR_NAMES[d] + ']' for d in separating)}.")

    # All batch rows carry identical inputs, so the prior GMM is the same for every row;
    # take row 0 and treat its K components as the conditional prior.
    w = pri_w[0]
    m = pri_m[0]
    w = w / w.sum()
    assignment = assign_nd(m, centres)

    print()
    print(BAR)
    print("LATENT SPACE — conditional prior p(z|dm,I) vs. aggregate posterior")
    print(BAR)
    print(f"The prior has {len(w)} components. Assigning each to its nearest posterior")
    print("cluster centre, per-cluster mass should match the branch rate in the data.\n")
    print(f"{'cluster':<22} {'prior mass':>12} {'simulator':>12} {'|err|':>8}   {'#comp':>6}")
    print('-' * 96)
    for m_idx, (name, true_mass) in enumerate((('A (X->Y)', p_a), ('B (Y->X)', 1 - p_a))):
        mass = float(w[assignment == m_idx].sum())
        print(f"  {name:<20} {mass:>12.3f} {true_mass:>12.3f} {abs(mass - true_mass):>8.3f}   "
              f"{int((assignment == m_idx).sum()):>6d}")

    # With more prior components than equivalent SCMs, per-cluster mass alone cannot show
    # that the prior recovered the right structure: it assigns every component to a nearest
    # centre regardless of how far away it sits. These diagnostics test whether the prior
    # actually concentrates onto the posterior clusters rather than smearing across latent
    # space, which is the substantive question once K exceeds the number of branches.
    dist = np.linalg.norm(m - centres[assignment], axis=-1) / np.sqrt(LATENT_COV)
    nonzero = w > 1e-4
    entropy = -np.sum(w[nonzero] * np.log(w[nonzero]))
    stray = float(w[dist > 3.0].sum())
    print()
    print(BAR)
    print("LATENT SPACE — does the prior concentrate onto the posterior clusters?")
    print(BAR)
    print(f"  prior components K:                              {len(w)}")
    print(f"  effective number of components exp(H(w)):        {np.exp(entropy):.2f}")
    print(f"  components carrying weight > 0.01:               {int((w > 0.01).sum())}")
    print(f"  max distance to assigned cluster centre:         {dist.max():.2f} component sd")
    print(f"  weight-averaged distance:                        {float((w * dist).sum()):.2f} component sd")
    print(f"  stray mass (> 3 sd from both cluster centres):   {stray:.4f}")
    print(f"\n{'weight':>10} {'cluster':>10} {'dist (comp sd)':>16}")
    print('-' * 96)
    for i in np.argsort(w)[::-1]:
        if w[i] < 1e-4:
            continue
        print(f"{w[i]:>10.4f} {('A' if assignment[i] == 0 else 'B'):>10} {dist[i]:>16.2f}")
    n_negligible = int((w < 1e-4).sum())
    if n_negligible:
        print(f"  ({n_negligible} further components carry weight < 1e-4 and are omitted)")

    print()
    print(f"{'dimension':<14} {'source':<26} {'mean':>10} {'var':>10}")
    print('-' * 96)
    for d in range(n_vars):
        dim_name = f'z[{VAR_NAMES[d]}]'
        # Aggregate posterior: each row contributes N(z_mean, 0.1 I) at equal weight.
        agg_means = np.concatenate([z_a[:, d], z_b[:, d]])
        agg_w = np.concatenate([np.full(len(z_a), p_a / len(z_a)),
                                np.full(len(z_b), (1 - p_a) / len(z_b))])
        agg_mu = float((agg_w * agg_means).sum())
        agg_var = float((agg_w * (LATENT_COV + agg_means ** 2)).sum()) - agg_mu ** 2

        pri_mu = float((w * m[:, d]).sum())
        pri_var = float((w * (LATENT_COV + m[:, d] ** 2)).sum()) - pri_mu ** 2

        rows_out = (
            ('posterior, branch A', float(z_a[:, d].mean()),
             float(z_a[:, d].var() + LATENT_COV)),
            ('posterior, branch B', float(z_b[:, d].mean()),
             float(z_b[:, d].var() + LATENT_COV)),
            ('aggregate posterior', agg_mu, agg_var),
            ('conditional prior', pri_mu, pri_var),
        )
        for i, (label, mu, var) in enumerate(rows_out):
            print(f"  {dim_name if i == 0 else '':<12} {label:<26} {mu:>10.3f} {var:>10.3f}")
        print(f"  {'':<12} {'  prior - aggregate':<26} {rows_out[3][1] - rows_out[2][1]:>+10.3f} "
              f"{rows_out[3][2] - rows_out[2][2]:>+10.3f}")
        print()

    # Per-cluster means: does the prior put its components where the posteriors are?
    print(f"{'dimension':<14} {'cluster':<26} {'prior mean':>12} {'posterior mean':>16} {'err':>8}")
    print('-' * 96)
    for d in range(n_vars):
        dim_name = f'z[{VAR_NAMES[d]}]'
        for m_idx, (name, z) in enumerate((('A (X->Y)', z_a), ('B (Y->X)', z_b))):
            sel = assignment == m_idx
            if sel.sum() == 0:
                continue
            ww = w[sel] / w[sel].sum()
            pmu = float((ww * m[sel, d]).sum())
            qmu = float(z[:, d].mean())
            print(f"  {dim_name if m_idx == 0 else '':<12} {name:<26} {pmu:>12.3f} "
                  f"{qmu:>16.3f} {pmu - qmu:>+8.3f}")
    print()
    return separating


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------

def _save(fig, prefix, name):
    for ext in ('png', 'pdf'):
        path = f'{prefix}_{name}.{ext}'
        fig.savefig(path, dpi=160 if ext == 'png' else None)
    plt.close(fig)
    print(f"Figure written to {prefix}_{name}.png and .pdf")


def _shared_bins(arrays, n=90, pad_frac=0.05):
    lo = min(a.min() for a in arrays)
    hi = max(a.max() for a in arrays)
    pad = pad_frac * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n)


def _plot_data_posterior(results, v, tv, outcome, prefix):
    ivar, ovar = VAR_NAMES[tv], VAR_NAMES[outcome]
    bins = _shared_bins([results[s][k] for s in (SCM_A, SCM_B)
                         for k in ('ground_truth', 'posterior', 'prior')])

    do = f'do({ivar} = {v})'
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for sid, col, branch in ((SCM_A, COL_A, 'A'), (SCM_B, COL_B, 'B')):
        ax.hist(results[sid]['posterior'], bins=bins, density=True, alpha=0.5, color=col,
                label=f'$p_\\theta({ovar} \\mid {do},\\ x_{branch},\\ dm)$')
        ax.hist(results[sid]['ground_truth'], bins=bins, density=True, histtype='step',
                color=col, lw=1.3, ls='--',
                label=f'$p_{branch}({ovar} \\mid {do})$   ground truth')
    ax.set_xlabel(ovar)
    ax.set_ylabel('density')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    _save(fig, prefix, 'data_posterior')


def _plot_data_prior(results, p_a, v, tv, outcome, prefix):
    ivar, ovar = VAR_NAMES[tv], VAR_NAMES[outcome]
    bins = _shared_bins([results[s][k] for s in (SCM_A, SCM_B)
                         for k in ('ground_truth', 'posterior', 'prior')])

    do = f'do({ivar} = {v})'
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(results[SCM_A]['prior'], bins=bins, density=True, alpha=0.55, color=COL_MIX,
            label=f'$p_\\theta({ovar} \\mid {do},\\ dm)$')
    # Each ground-truth branch is a density in its own right, so plotting it raw would
    # stand at twice the height of the corresponding mode of the prior path. Scaling each
    # by its simulator weight makes the two directly comparable: the scaled curves sum to
    # exactly the distribution the prior path should be reproducing.
    for sid, col, branch, w in ((SCM_A, COL_A, 'A', p_a), (SCM_B, COL_B, 'B', 1 - p_a)):
        gt = results[sid]['ground_truth']
        ax.hist(gt, bins=bins, histtype='step', color=col, lw=1.5, ls='--',
                weights=np.full(gt.size, w / gt.size / np.diff(bins)[0]),
                label=f'${w:.2f} \\cdot p_{branch}({ovar} \\mid {do})$   ground truth')
    ax.set_xlabel(ovar)
    ax.set_ylabel('density')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    _save(fig, prefix, 'data_prior')


def _plot_latent(results, p_a, separating_dims, v, tv, prefix):
    """One figure per separating latent dimension: both posteriors against the prior.

    Dimensions whose branch means coincide carry no information about SCM identity, so
    plotting them says nothing about the question; `_report_latent` reports them instead.
    """
    z_a = results[SCM_A]['z_posterior']
    z_b = results[SCM_B]['z_posterior']
    z_p = results[SCM_A]['z_prior']
    do = f'do({VAR_NAMES[tv]} = {v})'

    for d in separating_dims:
        zname = f'z_{{{VAR_NAMES[d]}}}'
        bins = _shared_bins([z_a[:, d], z_b[:, d], z_p[:, d]])
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.hist(z_p[:, d], bins=bins, density=True, alpha=0.55, color=COL_MIX,
                label=f'$p_\\theta({zname} \\mid dm,\\ {do})$')
        # Posteriors scaled by their branch rate, so the two curves sum to the aggregate
        # posterior and are directly comparable to the prior.
        for z, col, branch, w in ((z_a, COL_A, 'A', p_a), (z_b, COL_B, 'B', 1 - p_a)):
            ax.hist(z[:, d], bins=bins, histtype='step', color=col, lw=1.5, ls='--',
                    weights=np.full(z.shape[0], w / z.shape[0] / np.diff(bins)[0]),
                    label=f'${w:.2f} \\cdot q_\\theta({zname} \\mid x_{branch},\\ dm,\\ {do})$')
        ax.set_xlabel(f'$z_{{{VAR_NAMES[d]}}}$')
        ax.set_ylabel('density')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        _save(fig, prefix, f'latent_z{VAR_NAMES[d]}')


if __name__ == '__main__':
    main()
