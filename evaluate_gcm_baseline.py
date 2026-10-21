"""
Standalone evaluation of the nonlinear graph-privileged DoWhy-GCM baseline.

Runs independently of `evaluate_causal.py` (no trained ACTIVA/MACE-TNP checkpoint
is needed) but reuses its metric, bootstrap and table code verbatim, so the
numbers land in the same scale as ACTIVA's and can be pasted into the same table.
Per-task fitting is parallelised across evaluation tasks with joblib.

Usage (see `--help` for all flags):

    python evaluate_gcm_baseline.py \
      --data-path data/TMLR/8var_sergio_nodrop_test.npz \
      --intervention-value 0.0 --n-eval-scms 200 --n-jobs 32 --variant v1

Merge into an existing results JSON produced by evaluate_causal.py with
`--merge-into <results.json>`.

The baseline is a *privileged-information* comparison, not an oracle: it is
handed the ground-truth graph and the same observational data as ACTIVA, but its
mechanisms are estimated from data and do not match SERGIO's generative form.
"""

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import tyro

import gcm_baseline as gb
from utils import load_scm_data
# Metric/aggregation code is imported (not reimplemented) so the baseline is
# scored by exactly the code that produces ACTIVA's numbers.
from evaluate_causal import (
    _energy_dist_np,
    _oas_cov,
    compute_cate_ground_truth,
    compute_cate_metrics_with_ci,
    compute_dist_metrics_with_ci,
    compute_intcov_ground_truth,
    compute_intcov_metrics_with_ci,
    compute_metrics_with_ci,
    get_descendants,
    print_results_table,
)


@dataclass
class Args:
    data_path: str = 'data/TMLR/8var_sergio_nodrop_test.npz'
    """Path to the test dataset (.npz file)."""
    seed: int = 42
    """Random seed. Use the same value as evaluate_causal.py to evaluate the same SCMs."""
    n_eval_scms: int = 200
    """Number of SCMs (evaluation tasks) to evaluate."""
    intervention_value: float = 0.0
    """Value v of the atomic intervention do(V_i = v). SERGIO knockouts use 0.0."""
    n_pred_samples: int = 0
    """Interventional samples drawn per task*intervention. 0 = match the data's sample count."""
    n_density_samples: int = 1000
    """Samples drawn per task*intervention to estimate subset marginals (NLL on dim subsets)."""
    quality: str = 'GOOD'
    """Auto-assignment quality: GOOD | BETTER | BEST (variant v1 only)."""
    variant: str = 'v1'
    """v1 = per-task model selection (primary); v2 = model class fixed across tasks, refit per task."""
    v2_selection_tasks: int = 50
    """Tasks used for the V2 selection pass (majority vote over chosen regressors)."""
    cross_fit_folds: int = 5
    """K for out-of-fold residuals feeding the noise density. <2 disables cross-fitting."""
    density: str = 'kde'
    """1-D residual density estimator: 'kde' (CV bandwidth) or 'gmm' (BIC) as sensitivity check."""
    log_density_floor: float = -20.0
    """Per-coordinate floor on log p. The fraction of floored evaluations is reported."""
    n_jobs: int = 8
    """Parallel worker processes for per-task fitting."""
    n_bootstrap: int = 1000
    """Bootstrap resamples for 95% confidence intervals."""
    method_label: str = ''
    """Row label in the tables. Auto-generated from the variant if empty."""
    results_path: str = ''
    """Where to write results JSON. Auto-generated from data_path/variant if empty."""
    merge_into: str = ''
    """Optional existing results JSON (from evaluate_causal.py) to merge these rows into."""


def _method_label(args: Args) -> str:
    if args.method_label:
        return args.method_label
    return f'DoWhy-GCM ({args.variant.upper()})'


def main():
    args = tyro.cli(Args)
    np.random.seed(args.seed)

    label = _method_label(args)
    label_sd = f'{label} samp-dens'

    print("=" * 80)
    print("Nonlinear graph-privileged baseline (DoWhy-GCM)")
    print("=" * 80)

    # ------------------------------------------------------------------ data
    # Loaded through the same `load_scm_data` call, in the same order relative to
    # the `np.random.seed` / `np.random.choice` pair, as evaluate_causal.py's
    # main(). Reproducing the subset by call-order coincidence would be fragile, so
    # the RNG-consuming preamble is replicated exactly rather than approximated,
    # and the resulting indices are fingerprinted below for a direct comparison.
    print(f"\nLoading data from {args.data_path}...")
    data_loader = load_scm_data(args.data_path, mode='interventional')
    input_data = np.array(data_loader.arrays[0])
    target_data = np.array(data_loader.arrays[1])
    graphs = np.array(data_loader.arrays[-2])
    targets = np.array(data_loader.arrays[-1])

    n_scms, n_samples, n_vars = input_data.shape
    n_targets = target_data.shape[1]
    print(f"Data shape: {n_scms} SCMs, {n_samples} samples, {n_vars} variables, {n_targets} targets")

    n_eval_scms = min(args.n_eval_scms, n_scms)
    eval_indices = np.random.choice(n_scms, n_eval_scms, replace=False)
    idx_fingerprint = hashlib.md5(np.ascontiguousarray(eval_indices).tobytes()).hexdigest()[:12]
    print(f"eval_indices fingerprint: {idx_fingerprint}  (first 8: {eval_indices[:8].tolist()})")
    print("  Must match the line printed by evaluate_causal.py for the same "
          "--data-path/--seed/--n-eval-scms;\n  if it does not, the two runs are not "
          "evaluating the same SCMs and the comparison is unpaired.")

    input_eval = input_data[eval_indices]
    target_eval = target_data[eval_indices]
    graphs_eval = graphs[eval_indices]
    targets_eval = targets[eval_indices]
    print(f"Evaluating on {n_eval_scms} SCMs")

    # Draw as many samples as the ground truth has, not as many as the
    # observational set has: the energy distance and the OAS covariance are both
    # sensitive to a sample-count mismatch between the predicted and true sets.
    n_true_samples = target_eval.shape[2]
    n_pred = args.n_pred_samples if args.n_pred_samples > 0 else n_true_samples
    if n_true_samples != n_samples:
        print(f"  [NOTE] observational n ({n_samples}) != interventional n "
              f"({n_true_samples}). Drawing {n_pred} samples to match the ground "
              f"truth; evaluate_causal.py draws {n_samples} for ACTIVA, so pass "
              f"--n-pred-samples {n_samples} if you need to reproduce that choice.")

    # How atomic is the intervention really? Measured, not assumed: SERGIO knockouts
    # sit at exactly 0, but the linadd datasets carry a small spread on the
    # intervened coordinate.
    intervened_sd = float(np.mean([
        target_eval[k, j, :, int(np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)[0])].std()
        for k in range(min(200, n_eval_scms)) for j in range(n_targets)
        if np.any(targets_eval[k, 0, j, :] > 0.5)]))

    # The worker computes its subset NLL columns from its own descendant sets while
    # the driver computes the subset energy columns from evaluate_causal's; a silent
    # disagreement would score the two on different dimensions.
    for k in range(n_eval_scms):
        for v in range(n_vars):
            assert gb._descendants(graphs_eval[k], v) == get_descendants(graphs_eval[k], v), (
                f"descendant logic disagrees on SCM {k}, node {v}")

    # ------------------------------------------------- masks and ground truth
    obs_mean = input_eval.mean(axis=1)
    ate_gt = np.zeros((n_eval_scms, n_targets, n_vars))
    intvar_gt = np.zeros((n_eval_scms, n_targets, n_vars))
    for k in range(n_eval_scms):
        for j in range(n_targets):
            ate_gt[k, j] = target_eval[k, j].mean(axis=0) - obs_mean[k]
            intvar_gt[k, j] = target_eval[k, j].var(axis=0)

    desc_mask = np.zeros((n_eval_scms, n_targets, n_vars), dtype=bool)
    valid_mask = np.ones((n_eval_scms, n_targets, n_vars), dtype=bool)
    for k in range(n_eval_scms):
        adj = graphs_eval[k]
        for j in range(n_targets):
            for v in np.flatnonzero(targets_eval[k, 0, j, :] > 0.5):
                for y in get_descendants(adj, v):
                    desc_mask[k, j, y] = True
                valid_mask[k, j, v] = False
    valid_mask_full = np.ones((n_eval_scms, n_targets, n_vars), dtype=bool)

    print("  Computing interventional covariance ground truth (OAS estimator)...",
          end="", flush=True)
    intcov_gt = compute_intcov_ground_truth(target_eval, n_eval_scms, n_targets, n_vars)
    print(" done")

    print("  Computing CATE ground truth...", end="", flush=True)
    cate_gt = compute_cate_ground_truth(
        input_eval, target_eval, graphs_eval, targets_eval, n_eval_scms, n_targets, n_vars
    )
    print(" done")

    # ------------------------------------------------------- V2 selection pass
    from joblib import Parallel, delayed

    fixed_prototype = None
    v2_vote = None
    if args.variant == 'v2':
        n_sel = min(args.v2_selection_tasks, n_eval_scms)
        print(f"\nV2 selection pass on {n_sel} tasks (quality={args.quality})...")
        sel = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(gb.safe_select_mechanisms_task)(
                input_eval[k], graphs_eval[k], args.quality, args.seed + k
            )
            for k in range(n_sel)
        )
        counts = Counter()
        protos = {}
        for res in sel:
            for lab, proto in zip(res.get('labels', []), res.get('prototypes', [])):
                counts[lab] += 1
                protos.setdefault(lab, proto)
        if not counts:
            raise RuntimeError("V2 selection pass produced no mechanisms.")
        winner, _ = counts.most_common(1)[0]
        fixed_prototype = protos[winner]
        v2_vote = dict(counts)
        print(f"  Selected model class (majority over non-root nodes): {winner}")
        print(f"  Vote counts: {v2_vote}")

    # ---------------------------------------------------- per-task fit/predict
    print(f"\nFitting {n_eval_scms} tasks ({args.variant.upper()}, quality={args.quality}, "
          f"n_jobs={args.n_jobs})...")
    t_start = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(gb.safe_fit_and_predict_task)(
            input_eval[k],
            target_eval[k],
            graphs_eval[k],
            targets_eval[k, 0],
            args.intervention_value,
            args.quality,
            args.variant,
            fixed_prototype,
            n_pred,
            args.n_density_samples,
            args.cross_fit_folds,
            args.density,
            args.log_density_floor,
            args.seed + k,
        )
        for k in range(n_eval_scms)
    )
    wall = time.time() - t_start
    print(f"  Total wall-clock: {wall:.1f}s  ({wall / max(n_eval_scms, 1):.2f}s per task)")

    failures = [(k, r['failed']) for k, r in enumerate(results) if 'failed' in r]
    ok_mask = np.array(['failed' not in r for r in results])
    print(f"  Fitted tasks: {int(ok_mask.sum())}/{n_eval_scms}   failures: {len(failures)}")
    for k, msg in failures[:5]:
        print(f"    task {k}: {msg}")
    if ok_mask.sum() == 0:
        raise RuntimeError("Every task failed; nothing to report.")

    # -------------------------------------------------- assemble predictions
    pred_samples = np.zeros((n_eval_scms, n_targets, n_pred, n_vars), dtype=np.float32)
    logp_joint = np.full((n_eval_scms, n_targets, n_true_samples), np.nan)
    logp_nondesc = np.full((n_eval_scms, n_targets, n_true_samples), np.nan)
    nll_sampdens = np.full((n_eval_scms, n_targets, 3), np.nan)
    for k, r in enumerate(results):
        if 'failed' in r:
            continue
        pred_samples[k] = r['samples']
        logp_joint[k] = r['logp_joint']
        logp_nondesc[k] = r['logp_nondesc']
        nll_sampdens[k] = r['nll_sampdens']

    ate_pred = pred_samples.mean(axis=2) - obs_mean[:, None, :]
    intvar_pred = pred_samples.var(axis=2)
    intcov_pred = np.zeros((n_eval_scms, n_targets, n_vars, n_vars))
    for k in range(n_eval_scms):
        for j in range(n_targets):
            intcov_pred[k, j] = _oas_cov(pred_samples[k, j].astype(float))
    # CATE is computed from the predicted samples by the same routine that builds
    # the ground truth from the true interventional samples.
    cate_pred = compute_cate_ground_truth(
        input_eval, pred_samples, graphs_eval, targets_eval, n_eval_scms, n_targets, n_vars
    )

    # ------------------------------------------------ pointwise metric tables
    print("\nComputing metrics with bootstrap CIs...")
    keep = np.flatnonzero(ok_mask)

    def _sub(a):
        return a[keep]

    ate_results = {label: compute_metrics_with_ci(
        _sub(ate_gt), _sub(ate_pred), _sub(desc_mask), _sub(valid_mask),
        n_bootstrap=args.n_bootstrap, seed=args.seed, non_desc_vs_zero=True,
        valid_mask_full=_sub(valid_mask_full))}
    intvar_results = {label: compute_metrics_with_ci(
        _sub(intvar_gt), _sub(intvar_pred), _sub(desc_mask), _sub(valid_mask),
        n_bootstrap=args.n_bootstrap, seed=args.seed,
        valid_mask_full=_sub(valid_mask_full))}
    intcov_results = {label: compute_intcov_metrics_with_ci(
        _sub(intcov_gt), _sub(intcov_pred), _sub(desc_mask), _sub(valid_mask),
        n_bootstrap=args.n_bootstrap, seed=args.seed)}
    cate_results = {label: compute_cate_metrics_with_ci(
        _sub(cate_gt), _sub(cate_pred), _sub(desc_mask), _sub(valid_mask),
        n_bootstrap=args.n_bootstrap, seed=args.seed)}

    # ------------------------------------------- energy distance / NLL per (k,j)
    print("Computing distribution metrics (Energy Distance, NLL)...")
    # Per-(task, intervention) matrices rather than flat lists: NaN marks "not
    # defined for this pair", which keeps the rows alignable for the paired
    # comparison below and lets the CIs be clustered by task.
    nan_mat = lambda: np.full((n_eval_scms, n_targets), np.nan)  # noqa: E731
    energy_all_full, energy_all_vi = nan_mat(), nan_mat()
    energy_desc, energy_nondesc = nan_mat(), nan_mat()
    nll_fact_all_vi, nll_fact_nondesc = nan_mat(), nan_mat()
    nll_sd_all_vi, nll_sd_desc, nll_sd_nondesc = nan_mat(), nan_mat(), nan_mat()

    for k in range(n_eval_scms):
        if not ok_mask[k]:
            continue
        adj = graphs_eval[k]
        for j in range(n_targets):
            true_kj = np.asarray(target_eval[k, j], dtype=float)
            pred_kj = np.asarray(pred_samples[k, j], dtype=float)
            intervened = np.flatnonzero(targets_eval[k, 0, j, :] > 0.5)
            j_var = int(intervened[0]) if len(intervened) else j
            desc_set = get_descendants(adj, j_var)
            desc_dims = sorted(desc_set)
            non_desc_dims = sorted(set(range(n_vars)) - desc_set - {j_var})
            all_dims_full = list(range(n_vars))
            all_dims_vi = [d for d in range(n_vars) if d != j_var]

            energy_all_full[k, j] = _energy_dist_np(true_kj[:, all_dims_full],
                                                    pred_kj[:, all_dims_full])
            energy_all_vi[k, j] = _energy_dist_np(true_kj[:, all_dims_vi],
                                                  pred_kj[:, all_dims_vi])
            if desc_dims:
                energy_desc[k, j] = _energy_dist_np(true_kj[:, desc_dims],
                                                    pred_kj[:, desc_dims])
            if non_desc_dims:
                energy_nondesc[k, j] = _energy_dist_np(true_kj[:, non_desc_dims],
                                                       pred_kj[:, non_desc_dims])

            lp = logp_joint[k, j]
            if np.all(np.isfinite(lp)):
                nll_fact_all_vi[k, j] = float(-np.mean(lp))
            lp_nd = logp_nondesc[k, j]
            if non_desc_dims and np.all(np.isfinite(lp_nd)):
                nll_fact_nondesc[k, j] = float(-np.mean(lp_nd))
            nll_sd_all_vi[k, j] = nll_sampdens[k, j, 0]
            if desc_dims:
                nll_sd_desc[k, j] = nll_sampdens[k, j, 1]
            if non_desc_dims:
                nll_sd_nondesc[k, j] = nll_sampdens[k, j, 2]

    # The two NLL rows must be scored on the same (task, intervention) pairs, or
    # their difference conflates the estimator with the task set. Both are
    # restricted to the pairs on which both are finite.
    paired = np.isfinite(nll_fact_all_vi) & np.isfinite(nll_sd_all_vi)
    n_pairs_total = int(np.sum(ok_mask) * n_targets)
    n_pairs_paired = int(paired.sum())
    nll_fact_all_vi = np.where(paired, nll_fact_all_vi, np.nan)
    nll_sd_all_vi = np.where(paired, nll_sd_all_vi, np.nan)
    nll_fact_nondesc = np.where(paired, nll_fact_nondesc, np.nan)
    nll_sd_desc = np.where(paired, nll_sd_desc, np.nan)
    nll_sd_nondesc = np.where(paired, nll_sd_nondesc, np.nan)
    paired_delta = float(np.mean((nll_sd_all_vi - nll_fact_all_vi)[paired])) \
        if n_pairs_paired else float('nan')

    def _flat(mat):
        """Finite entries only — compute_dist_metrics_with_ci filters NaN itself,
        but flattening explicitly keeps the bootstrap sample size honest."""
        a = np.asarray(mat).ravel()
        return a[np.isfinite(a)].tolist()

    energy_results = {label: compute_dist_metrics_with_ci(
        _flat(energy_all_vi), _flat(energy_desc), _flat(energy_nondesc), 'Energy',
        n_bootstrap=args.n_bootstrap, seed=args.seed,
        vals_all_full=_flat(energy_all_full))}

    # Factorised NLL: the exact joint over all non-intervened coordinates
    # (NLL_all_vi), and the exact marginal over the non-descendants, which are
    # ancestrally closed so their factors alone integrate correctly. Descendants
    # have no closed-form marginal here and are left to the sample-density row.
    #
    # NLL_all is deliberately N/A rather than a copy of NLL_all_vi: ACTIVA's
    # NLL_all *includes* a density term on the intervened coordinate (6.70 vs 9.23
    # on the SERGIO test set — it gains 2.5 nats there), so a column that silently
    # omits that term is not comparable to ACTIVA's, only to the Linear Causal
    # baseline's. Compare methods on NLL_all_vi.
    nll_results = {label: compute_dist_metrics_with_ci(
        _flat(nll_fact_all_vi), [], _flat(nll_fact_nondesc), 'NLL',
        n_bootstrap=args.n_bootstrap, seed=args.seed)}
    # Sample-density NLL: a GMM (BIC-selected) fitted to the model's own
    # interventional samples, evaluated on any coordinate subset — the only route
    # to the descendant column, at the cost of a second estimation layer.
    nll_results[label_sd] = compute_dist_metrics_with_ci(
        _flat(nll_sd_all_vi), _flat(nll_sd_desc), _flat(nll_sd_nondesc), 'NLL',
        n_bootstrap=args.n_bootstrap, seed=args.seed)

    # Task-clustered CIs. The shared code bootstraps flattened (task, intervention)
    # pairs, which are not independent within a task, so those intervals are too
    # narrow. ACTIVA's numbers are computed the same way, so the printed tables stay
    # comparable; these wider intervals are recorded alongside them.
    def _clustered_ci(mat, n_bootstrap, seed):
        rng = np.random.default_rng(seed)
        rows = [r[np.isfinite(r)] for r in np.asarray(mat)]
        rows = [r for r in rows if len(r)]
        if not rows:
            return {'mean': float('nan'), 'ci_lower': float('nan'),
                    'ci_upper': float('nan'), 'n_clusters': 0}
        mean = float(np.concatenate(rows).mean())
        boot = [float(np.concatenate([rows[i] for i in
                                      rng.integers(0, len(rows), len(rows))]).mean())
                for _ in range(n_bootstrap)]
        return {'mean': mean, 'ci_lower': float(np.percentile(boot, 2.5)),
                'ci_upper': float(np.percentile(boot, 97.5)), 'n_clusters': len(rows)}

    clustered = {
        'Energy_all_vi': _clustered_ci(energy_all_vi, args.n_bootstrap, args.seed),
        'Energy_descendants': _clustered_ci(energy_desc, args.n_bootstrap, args.seed),
        'Energy_non_descendants': _clustered_ci(energy_nondesc, args.n_bootstrap, args.seed),
        'NLL_all_vi': _clustered_ci(nll_fact_all_vi, args.n_bootstrap, args.seed),
        'NLL_non_descendants': _clustered_ci(nll_fact_nondesc, args.n_bootstrap, args.seed),
    }

    # --------------------------------------------------------------- reporting
    print_results_table("ATE Estimation Results", ate_results,
                        ['MAE_all', 'MAE_all_vi', 'RMSE_all', 'MAE_descendants',
                         'MAE_non_descendants', 'MAE_non_desc_noise_floor'],
                        n_bootstrap=args.n_bootstrap)
    print_results_table("Interventional Variance Results", intvar_results,
                        ['MAE_all', 'MAE_all_vi', 'RMSE_all', 'MAE_descendants',
                         'RMSE_descendants', 'MAE_non_descendants'],
                        n_bootstrap=args.n_bootstrap)
    print_results_table("Interventional Covariance Results — full matrix MAE/RMSE",
                        intcov_results,
                        ['CovMAE_all', 'CovRMSE_all', 'CovMAE_both_desc',
                         'CovMAE_neither_desc'],
                        n_bootstrap=args.n_bootstrap)
    print_results_table("CATE Results", cate_results,
                        ['CATE_MAE_all', 'CATE_MAE_descendants',
                         'CATE_MAE_non_descendants'],
                        n_bootstrap=args.n_bootstrap)
    print_results_table("Energy Distance Results", energy_results,
                        ['Energy_all', 'Energy_all_vi', 'Energy_descendants',
                         'Energy_non_descendants'],
                        n_bootstrap=args.n_bootstrap)
    print_results_table("NLL Results", nll_results,
                        ['NLL_all', 'NLL_all_vi', 'NLL_descendants',
                         'NLL_non_descendants'],
                        n_bootstrap=args.n_bootstrap)

    # Reporting block: everything needed so the comparison cannot be dismissed
    # as a strawman (library version, quality, selected mechanisms, warnings,
    # extrapolation distance, floor rate, wall-clock).
    mech_counter = Counter()
    per_node_classes = {}
    all_warnings = Counter()
    floor_fracs, neg_fracs, fit_times, oof_fracs = [], [], [], []
    support_d, support_nd = [], []
    n_degenerate = 0
    for k, r in enumerate(results):
        if 'failed' in r:
            continue
        for node, lab in r['mech_labels'].items():
            mech_counter[lab] += 1
            per_node_classes.setdefault(node, Counter())[lab] += 1
        for w in r['warnings']:
            all_warnings[w] += 1
        floor_fracs.append(r['floor_frac'])
        neg_fracs.append(r['neg_frac'])
        support_d.append(r['support_desc'])
        support_nd.append(r['support_nondesc'])
        fit_times.append(r['fit_seconds'])
        oof_fracs.append(r['oof_fraction'])
        n_degenerate += r['n_degenerate']

    node_variability = {
        node: len(c) for node, c in per_node_classes.items()
    }

    print("\n" + "=" * 80)
    print("Baseline reporting block")
    print("=" * 80)
    print(f"  Library:                 dowhy {getattr(gb, 'DOWHY_REQUIRED_VERSION', '?')} "
          f"(dowhy.gcm), Blöbaum et al., JMLR 2024")
    print(f"  Variant / quality:       {args.variant.upper()} / {args.quality}")
    print(f"  SCM subset fingerprint:  {idx_fingerprint} (must match evaluate_causal.py)")
    print(f"  Representation space:    raw dataset values, as loaded by load_scm_data — "
          f"the same\n                           arrays ACTIVA's NLL is computed on "
          f"(models.py applies LayerNorm to\n                           activations "
          f"only; the decoder GMM is in data space, no transform)")
    print(f"  Observational n:         {n_samples} rows per task (same rows as ACTIVA)")
    print(f"  Evaluation samples:      {n_pred} drawn per task*intervention "
          f"(true: {n_true_samples})")
    print(f"  Intervention:            atomic do(V_i = {args.intervention_value}); the "
          f"intervened coordinate in the\n                           ground truth has "
          f"sd {intervened_sd:.4f} vs overall data sd {float(input_eval.std()):.3f} "
          f"({intervened_sd / max(float(input_eval.std()), 1e-12) * 100:.1f}%),\n"
          f"                           so it is treated as atomic — the same convention "
          f"as Linear Causal,\n                           whose mutilated SCM gives the "
          f"intervened node zero variance")
    print(f"  Residual density:        {args.density} "
          f"({'CV-selected bandwidth' if args.density == 'kde' else 'BIC-selected components'}), "
          f"cross-fit K={args.cross_fit_folds}")
    print(f"  Out-of-fold residuals:   {np.nanmean(oof_fracs) * 100:.1f}% of non-root nodes "
          f"(rest fell back to in-sample)")
    print(f"  Intervened coordinate:   excluded from the NLL. NLL_all is reported N/A "
          f"rather than\n                           copied from NLL_all_vi, because "
          f"ACTIVA's NLL_all includes a density\n                           term there. "
          f"Compare methods on NLL_all_vi.")
    print(f"  NLL task pairing:        {n_pairs_paired}/{n_pairs_total} "
          f"(task, intervention) pairs scored by both NLL rows;\n"
          f"                           paired samp-dens minus exact = "
          f"{paired_delta:+.4f} nats")
    print(f"  Degenerate coordinates:  {n_degenerate} (task, intervention) pairs dropped "
          f"from the NLL\n                           (constant coordinate — no density "
          f"defined; not substituted)")
    print(f"  log-density floor:       {args.log_density_floor}  "
          f"(binds on {np.mean(floor_fracs) * 100:.2f}% of node evaluations)")
    print(f"  Negative sampled values: {np.mean(neg_fracs) * 100:.2f}% "
          f"(not clipped — clipping would change the density)")
    print(f"  Extrapolation distance:  descendants {np.nanmean(support_d):.3f} vs "
          f"non-descendants {np.nanmean(support_nd):.3f}\n"
          f"                           (standardised distance from an evaluation parent "
          f"vector to the\n                           nearest observational one; only "
          f"descendants sit off-support)")
    print(f"  Clustered 95% CIs:       " + ", ".join(
        f"{k} {v['mean']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
        for k, v in list(clustered.items())[:2]))
    print(f"                           (bootstrap over tasks, not over "
          f"(task, intervention) pairs; the\n                           tables above use "
          f"the shared unclustered code, as ACTIVA's do)")
    print(f"  Mechanism classes:       {dict(mech_counter)}")
    print(f"  Distinct classes/node:   {node_variability}")
    print(f"  Fit warnings:            {sum(all_warnings.values())} total, "
          f"{len(all_warnings)} distinct")
    for msg, cnt in all_warnings.most_common(5):
        print(f"    [{cnt}x] {msg}")
    print(f"  Failed tasks:            {len(failures)}")
    print(f"  Wall-clock per task:     {np.mean(fit_times):.2f}s fit "
          f"(+ prediction; {wall / max(n_eval_scms, 1):.2f}s end-to-end at n_jobs={args.n_jobs})")
    print("=" * 80)
    print("\nThis baseline is a privileged-information comparison (it receives the "
          "ground-truth graph), not an oracle.")
    print("Causal-sufficiency caveat: the truncated factorisation treats root nodes as "
          "independent.\n  SERGIO's cell types set the master regulators' production "
          "rates jointly, so on graphs with\n  more than one root they share a latent "
          "cause that the GRN does not contain. This applies\n  identically to the "
          "Linear Causal baseline. The '_nodrop' datasets have outlier, library-size\n"
          "  and dropout effects disabled, so no further technical noise is present.\n")

    # ------------------------------------------------------------------ output
    def _to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (v != v) else v
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    all_results = {
        'ate': ate_results,
        'intvar': intvar_results,
        'intcov': intcov_results,
        'cate': cate_results,
        'energy': energy_results,
        'nll': nll_results,
    }
    serializable = _to_serializable(all_results)
    serializable['_meta'] = _to_serializable({
        'method': label,
        'library': f'dowhy {gb.DOWHY_REQUIRED_VERSION} (dowhy.gcm)',
        'citation': 'Blöbaum et al., DoWhy-GCM, JMLR 25(147), 2024',
        'data_path': args.data_path,
        'seed': args.seed,
        'n_eval_scms': n_eval_scms,
        'n_failed_tasks': len(failures),
        'failures': [f'{k}: {m}' for k, m in failures[:50]],
        'n_obs_samples': n_samples,
        'n_pred_samples': n_pred,
        'n_density_samples': args.n_density_samples,
        'eval_indices_fingerprint': idx_fingerprint,
        'n_true_samples': n_true_samples,
        'representation_space': ('raw dataset values as loaded by load_scm_data; '
                                 'ACTIVA computes its NLL on the same arrays'),
        'intervention': f'atomic do(V_i = {args.intervention_value})',
        'intervened_coordinate_sd_in_ground_truth': intervened_sd,
        'nll_intervened_coordinate': ('excluded; NLL_all reported N/A because '
                                      "ACTIVA's NLL_all includes a term there"),
        'nll_pairs_scored': n_pairs_paired,
        'nll_pairs_total': n_pairs_total,
        'nll_paired_sampdens_minus_exact': paired_delta,
        'nll_degenerate_pairs_dropped': n_degenerate,
        'clustered_ci': clustered,
        'extrapolation_distance_descendants': float(np.nanmean(support_d)),
        'extrapolation_distance_non_descendants': float(np.nanmean(support_nd)),
        'oof_fraction': float(np.nanmean(oof_fracs)),
        'variant': args.variant,
        'quality': args.quality,
        'v2_vote': v2_vote,
        'density_estimator': args.density,
        'cross_fit_folds': args.cross_fit_folds,
        'log_density_floor': args.log_density_floor,
        'floor_bind_fraction': float(np.mean(floor_fracs)),
        'negative_sample_fraction': float(np.mean(neg_fracs)),
        'mechanism_classes': dict(mech_counter),
        'distinct_classes_per_node': node_variability,
        'n_warnings': int(sum(all_warnings.values())),
        'warnings_top': [f'[{c}x] {m}' for m, c in all_warnings.most_common(10)],
        'wall_clock_total_s': wall,
        'wall_clock_per_task_s': wall / max(n_eval_scms, 1),
        'n_bootstrap': args.n_bootstrap,
        'intervention_value': args.intervention_value,
    })

    out_path = args.results_path
    if not out_path:
        data_stem = os.path.splitext(os.path.basename(args.data_path))[0]
        out_path = f'results_gcm_{args.variant}_{data_stem}.json'
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to: {out_path}")

    if args.merge_into:
        with open(args.merge_into) as f:
            merged = json.load(f)
        for section in ('ate', 'intvar', 'intcov', 'cate', 'energy', 'nll'):
            merged.setdefault(section, {}).update(serializable.get(section, {}))
        merged.setdefault('_meta', {})[f'gcm_baseline_{args.variant}'] = serializable['_meta']
        with open(args.merge_into, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Merged into: {args.merge_into}")

    return all_results


if __name__ == '__main__':
    main()
