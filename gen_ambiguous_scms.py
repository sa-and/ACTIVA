"""
Generate the non-identifiability dataset: two SCMs with an identical observational
joint but different interventional distributions.

The pair is an edge reversal in the standardized linear-Gaussian family:

    SCM A (X -> Y):  X = e_x,            e_x ~ N(0, 1)
                     Y = rho*X + e_y,    e_y ~ N(0, 1 - rho^2)

    SCM B (Y -> X):  Y = n_y,            n_y ~ N(0, 1)
                     X = rho*Y + n_x,    n_x ~ N(0, 1 - rho^2)

Both induce exactly N(0, [[1, rho], [rho, 1]]): each is a linear map of independent
Gaussians (hence jointly Gaussian), and a Gaussian is determined by its mean and
covariance, which match by construction. The interventional distributions do not:

    do(X = v):  A gives Y ~ N(rho*v, 1 - rho^2),   B gives Y ~ N(0, 1)
    do(Y = v):  A gives X ~ N(0, 1),               B gives X ~ N(rho*v, 1 - rho^2)

Every row of the emitted dataset carries the *same* observational context and the
*same* intervention target mask; rows differ only in which SCM produced their
interventional data. The obs context is therefore information-theoretically unable
to identify the SCM, so the Bayes-optimal predictive is the mixture over both
branches, weighted by --p-scm-a.

Note that this pair is non-identifiable precisely because the noise is Gaussian.
With non-Gaussian additive noise (e.g. the `f_linear_noise_beta` DGF used elsewhere
in this repo) the edge direction becomes identifiable from observational data alone.

Output matches the interventional NPZ layout produced by `generate_data.py`:
    input       : (n_scms, n_samples, n_vars)
    target_data : (n_interventions, n_scms, n_targets, n_samples, n_vars)
    graphs      : (n_scms, n_vars, n_vars)
    targets     : (n_scms, n_interventions, n_targets, n_vars)

Two extra keys are stored for analysis and ignored by `utils.load_scm_data`:
    scm_id      : (n_scms,)  0 = SCM A (X->Y), 1 = SCM B (Y->X)
    context     : (n_samples, n_vars)  the single shared observational context

Example (the configuration of Sec. 7.4 of the paper; see the README for training):
    python gen_ambiguous_scms.py --save-dir data/ambiguous_2var_75 \
        --p-scm-a 0.75 --n-scms 1024 --seed 0
"""

import argparse
import json

import numpy as np

# Variable order in every array: index 0 = X, index 1 = Y.
X, Y = 0, 1
N_VARS = 2
SCM_A, SCM_B = 0, 1

# Adjacency convention (matches generate_data.py / evaluate_causal.py):
# graph[i, j] == 1 means i -> j, i.e. i is a parent of j.
GRAPH_A = np.array([[0, 1], [0, 0]], dtype=np.float32)  # X -> Y
GRAPH_B = np.array([[0, 0], [1, 0]], dtype=np.float32)  # Y -> X


def sample_context(rng, n_samples, rho):
    """Draw the shared observational context directly from the common joint.

    Both SCMs induce N(0, [[1, rho], [rho, 1]]) exactly, so sampling the joint
    rather than routing through either SCM makes the symmetry explicit in code.
    """
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal(np.zeros(N_VARS), cov, size=n_samples)


def sample_interventional(rng, scm_id, target_var, n_samples, rho, value, value_std):
    """Draw n_samples from `scm_id` under do(target_var = value).

    The intervened variable is set to `value` with a small jitter, matching the
    `gauss(intervention_value, 0.01)` convention in generate_data.py. Downstream
    variables are computed from the *realized* intervened value.
    """
    noise_std = np.sqrt(1.0 - rho ** 2)
    out = np.empty((n_samples, N_VARS))
    forced = value + value_std * rng.standard_normal(n_samples)

    # Is the non-intervened variable a child of the intervened one in this SCM?
    # A is X->Y, so do(X) propagates; B is Y->X, so do(Y) propagates.
    propagates = (scm_id == SCM_A and target_var == X) or (scm_id == SCM_B and target_var == Y)
    other = Y if target_var == X else X

    out[:, target_var] = forced
    if propagates:
        out[:, other] = rho * forced + noise_std * rng.standard_normal(n_samples)
    else:
        # Intervening on a non-ancestor leaves the other variable at its marginal N(0, 1).
        out[:, other] = rng.standard_normal(n_samples)
    return out


def build_rows(rng, n_scms, n_samples, rho, value, value_std, p_scm_a):
    """Assign each row an SCM and draw its interventional datasets.

    Returns (target_data, graphs, scm_ids) with
        target_data : (n_scms, n_targets, n_samples, n_vars), n_targets = 2 (do(X), do(Y))
        graphs      : (n_scms, n_vars, n_vars)
        scm_ids     : (n_scms,)
    """
    # Deterministic counts so the mixture weight is exact rather than binomial —
    # with p=0.5 and n=512 a random draw would wobble the target ratio for no reason.
    n_a = int(round(p_scm_a * n_scms))
    scm_ids = np.concatenate([np.full(n_a, SCM_A), np.full(n_scms - n_a, SCM_B)])

    target_data = np.stack([
        np.stack([
            sample_interventional(rng, sid, tv, n_samples, rho, value, value_std)
            for tv in (X, Y)
        ])
        for sid in scm_ids
    ])
    graphs = np.stack([GRAPH_A if sid == SCM_A else GRAPH_B for sid in scm_ids])
    return target_data, graphs, scm_ids


def stratified_split(rng, scm_ids, train_fraction, val_fraction):
    """Shuffle and split row indices, keeping the A/B ratio intact in every split.

    A plain random split would almost certainly work at these sizes, but stratifying
    guarantees each split actually contains both branches — a split missing one of
    them would silently stop being ambiguous.
    """
    train_idx, val_idx, test_idx = [], [], []
    for sid in (SCM_A, SCM_B):
        idx = np.flatnonzero(scm_ids == sid)
        rng.shuffle(idx)
        n_train = int(train_fraction * len(idx))
        n_val = int((train_fraction + val_fraction) * len(idx))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_val])
        test_idx.append(idx[n_val:])

    out = []
    for parts in (train_idx, val_idx, test_idx):
        merged = np.concatenate(parts)
        rng.shuffle(merged)  # interleave A and B so batches see both
        out.append(merged)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--save-dir', type=str, default='data/ambiguous_2var',
                        help='Base path; _train.npz / _val.npz / _test.npz are appended.')
    parser.add_argument('--n-scms', type=int, default=512,
                        help='Number of rows. All share one context; half carry SCM A data, half SCM B. '
                             'Kept above batch_size so the P("batch") sharding in train_vae.py works.')
    parser.add_argument('--samples', type=int, default=50,
                        help='Samples per observational context and per interventional dataset.')
    parser.add_argument('--rho', type=float, default=0.8,
                        help='Correlation of the shared joint. Edge weight is rho and noise variance '
                             '1 - rho^2 in both SCMs, which is what makes the pair self-mirroring.')
    parser.add_argument('--intervention-value', type=float, default=5.0,
                        help='do(V_j = v). Matches the hardcoded 5.0 in generate_data.py.')
    parser.add_argument('--intervention-std', type=float, default=0.01,
                        help='Jitter on the intervened variable, matching generate_data.py.')
    parser.add_argument('--p-scm-a', type=float, default=0.5,
                        help='Fraction of rows drawn from SCM A. Set to e.g. 0.75 for the '
                             'calibration variant: the predictive mass ratio should track this.')
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # One observational context, shared byte-for-byte by every row. This is what makes
    # SCM identity unrecoverable from the model's inputs rather than merely hard to recover.
    context = sample_context(rng, args.samples, args.rho)
    input_data = np.broadcast_to(context, (args.n_scms, args.samples, N_VARS)).copy()

    target_data, graphs, scm_ids = build_rows(
        rng, args.n_scms, args.samples, args.rho,
        args.intervention_value, args.intervention_std, args.p_scm_a)

    # Intervention target masks, identical for every row:
    #   target 0 = do(X) -> [1, 0],  target 1 = do(Y) -> [0, 1]
    # Shape (n_scms, n_interventions=1, n_targets=2, n_vars=2).
    targets = np.tile(np.eye(N_VARS, dtype=np.int32)[None, None], (args.n_scms, 1, 1, 1))

    _self_check(input_data, target_data, scm_ids, args)

    splits = stratified_split(rng, scm_ids, args.train_fraction, args.val_fraction)
    for suffix, idx in zip(('_train', '_val', '_test'), splits):
        np.savez(
            args.save_dir + suffix,
            input=input_data[idx].astype(np.float32),
            # (n_interventions=1, n_scms, n_targets, n_samples, n_vars)
            target_data=target_data[idx].astype(np.float32)[None],
            graphs=graphs[idx].astype(np.float32),
            targets=targets[idx],
            scm_id=scm_ids[idx],
            context=context.astype(np.float32),
        )
        n_a = int((scm_ids[idx] == SCM_A).sum())
        print(f"  {args.save_dir}{suffix}.npz  rows={len(idx):5d}  "
              f"SCM_A={n_a:5d}  SCM_B={len(idx) - n_a:5d}")

    _save_metadata(args)


def _self_check(input_data, target_data, scm_ids, args):
    """Assert the invariants the experiment depends on, and print the ground truth."""
    assert np.all(input_data == input_data[0]), "obs contexts differ across rows"
    assert set(np.unique(scm_ids)) == {SCM_A, SCM_B}, "dataset does not contain both SCMs"

    emp_cov = np.cov(input_data[0].T)
    print(f"\nShared context ({args.samples} samples), empirical covariance:")
    print(f"  Var(X)={emp_cov[0, 0]:.3f}  Var(Y)={emp_cov[1, 1]:.3f}  "
          f"Cov={emp_cov[0, 1]:.3f}   (target: 1, 1, {args.rho})")

    # Empirical interventional moments per branch, against the analytic values.
    rho, v = args.rho, args.intervention_value
    noise_var = 1.0 - rho ** 2
    print("\nInterventional ground truth (empirical / analytic):")
    for sid, name in ((SCM_A, 'A (X->Y)'), (SCM_B, 'B (Y->X)')):
        rows = target_data[scm_ids == sid]
        for tv, tv_name in ((X, 'do(X=%.1f)' % v), (Y, 'do(Y=%.1f)' % v)):
            other = Y if tv == X else X
            obs = rows[:, tv, :, other].ravel()
            propagates = (sid == SCM_A and tv == X) or (sid == SCM_B and tv == Y)
            exp_mean, exp_var = (rho * v, noise_var) if propagates else (0.0, 1.0)
            print(f"  SCM {name}  {tv_name} -> {'XY'[other]}: "
                  f"mean={obs.mean():+.3f} var={obs.var():.3f}  |  "
                  f"analytic mean={exp_mean:+.3f} var={exp_var:.3f}")

    p = args.p_scm_a
    print(f"\nBayes-optimal predictive under do(X={v:.1f}) is the mixture:")
    print(f"  {p:.2f} * N({rho * v:+.2f}, {noise_var:.2f})  +  {1 - p:.2f} * N(+0.00, 1.00)   [over Y]")
    print(f"and under do(Y={v:.1f}):")
    print(f"  {p:.2f} * N(+0.00, 1.00)  +  {1 - p:.2f} * N({rho * v:+.2f}, {noise_var:.2f})   [over X]\n")


def _save_metadata(args):
    """Write the analytic ground truth alongside the data so eval need not re-derive it."""
    rho, v = args.rho, args.intervention_value
    noise_var = 1.0 - rho ** 2
    meta = {
        'generator': 'gen_ambiguous_scms.py',
        'args': vars(args),
        'var_order': ['X', 'Y'],
        'obs_joint': {'mean': [0.0, 0.0], 'cov': [[1.0, rho], [rho, 1.0]]},
        'interventional': {
            'do(X)': {'A': {'Y': [rho * v, noise_var]}, 'B': {'Y': [0.0, 1.0]}},
            'do(Y)': {'A': {'X': [0.0, 1.0]}, 'B': {'X': [rho * v, noise_var]}},
        },
        'mixture_weights': {'A': args.p_scm_a, 'B': 1.0 - args.p_scm_a},
    }
    path = args.save_dir + '_meta.json'
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {path}")


if __name__ == '__main__':
    main()
