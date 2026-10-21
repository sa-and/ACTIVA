"""
Nonlinear graph-privileged baseline built on DoWhy-GCM (PyWhy).

This module contains the *worker* side of the baseline: everything that is fitted
per evaluation task. It deliberately imports nothing from the JAX/Flax stack so
that it can be executed inside `joblib` worker processes without pulling in the
ACTIVA model code.

Baseline description
--------------------
For each evaluation task (one SCM) the baseline receives

  * the ground-truth causal graph (privileged information), and
  * exactly the observational sample set that ACTIVA receives,

assigns a flexible causal mechanism to every node with `dowhy.gcm.auto`
(root nodes: empirical distribution; non-root nodes: additive noise model
Y = f(pa(Y)) + N with a data-selected regressor for f), fits it, and then
produces, per intervention,

  * a set of post-interventional samples (used for the energy distance / ERG and
    for the moment-based causal quantities), and
  * a joint log-density of the held-out interventional samples under the
    truncated factorisation
        log p(v | do(V_i = v_i)) = sum_{y != i} log p_y(v_y | pa_y(v))
    (used for the NLL).

Unlike the existing Linear Causal baseline it does *not* impose linear-Gaussian
mechanisms; both baselines get the same graph and the same data, so the contrast
between them isolates mechanism flexibility.

Reference: Blöbaum et al., "DoWhy-GCM: An Extension of DoWhy for Causal
Inference in Graphical Causal Models", JMLR 25(147), 2024.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

from dowhy import gcm
from dowhy.graph import get_ordered_predecessors

# Verified against dowhy 0.12: `AssignmentQuality` (not `AutoAssignmentQuality`),
# `assign_causal_mechanisms(scm, df, quality=...)`, `gcm.fit(scm, df)`,
# `gcm.interventional_samples(scm, {node: callable}, num_samples_to_draw=n)`,
# `AdditiveNoiseModel.estimate_noise(target_samples, parent_samples)` and
# `dowhy.graph.get_ordered_predecessors`.
DOWHY_REQUIRED_VERSION = "0.12"

# Worker processes must not print a progress bar per task.
gcm.config.disable_progress_bars()

QUALITY_MAP = {
    "GOOD": gcm.auto.AssignmentQuality.GOOD,
    "BETTER": gcm.auto.AssignmentQuality.BETTER,
    "BEST": gcm.auto.AssignmentQuality.BEST,
}


def node_names(n_vars: int) -> List[str]:
    """Canonical node names, used for the graph, the DataFrame and every slice."""
    return [f"V{i}" for i in range(n_vars)]


def build_graph(adj: np.ndarray, names: Sequence[str]) -> nx.DiGraph:
    """
    Build a DiGraph from an adjacency matrix using the repo convention
    `adj[i, j] == 1` means edge i -> j (i is a parent of j).
    """
    g = nx.DiGraph()
    g.add_nodes_from(names)
    idx = np.argwhere(np.asarray(adj) == 1)
    for i, j in idx:
        g.add_edge(names[int(i)], names[int(j)])
    return g


def _prediction_model_label(mechanism) -> str:
    """Human-readable label of the regressor inside an AdditiveNoiseModel."""
    pm = getattr(mechanism, "prediction_model", None)
    if pm is None:
        return type(mechanism).__name__
    sk = getattr(pm, "sklearn_model", None)
    return type(sk).__name__ if sk is not None else type(pm).__name__


class _Density1D:
    """
    1-D density estimator used for the factorised log-density.

    Kernel density with a cross-validated bandwidth (`kind='kde'`), or a small
    Gaussian mixture selected by BIC (`kind='gmm'`) as a sensitivity check.
    Both expose `log_density(x)`; the caller applies the density floor.
    """

    def __init__(self, kind: str = "kde", seed: int = 0):
        self.kind = kind
        self.seed = seed
        self._model = None
        self.degenerate = False

    def fit(self, x: np.ndarray) -> "_Density1D":
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        x = x[np.isfinite(x[:, 0])]
        scale = float(np.std(x)) if len(x) > 1 else 0.0
        if len(x) < 2 or scale < 1e-10:
            # Degenerate coordinate (a node that is constant in this task, or a
            # mechanism with zero residual variance): a density is not defined.
            # Substituting an arbitrarily narrow Gaussian would hand the baseline
            # unearned log-density, so the coordinate is flagged and the caller
            # drops the affected (task, intervention) pair, reporting how often.
            self.degenerate = True
            return self

        if self.kind == "gmm":
            best, best_bic = None, np.inf
            for k in (1, 2, 3):
                if k > len(x):
                    break
                gm = GaussianMixture(n_components=k, covariance_type="full",
                                     reg_covar=1e-6, random_state=self.seed)
                gm.fit(x)
                bic = gm.bic(x)
                if bic < best_bic:
                    best, best_bic = gm, bic
            self._model = best
            return self

        # KDE with a CV-selected bandwidth (with n ~ 30 residuals the bandwidth
        # materially drives the NLL, so a rule of thumb is not good enough).
        grid = np.logspace(-1.5, 0.5, 12) * scale
        n_splits = int(min(5, len(x)))
        if n_splits >= 2:
            search = GridSearchCV(
                KernelDensity(kernel="gaussian"),
                {"bandwidth": grid},
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=self.seed),
                n_jobs=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(x)
            self._model = search.best_estimator_
            self.bandwidth = float(search.best_params_["bandwidth"])
        else:
            self._model = KernelDensity(kernel="gaussian",
                                        bandwidth=float(scale)).fit(x)
            self.bandwidth = float(scale)
        return self

    def log_density(self, x: np.ndarray) -> np.ndarray:
        if self.degenerate or self._model is None:
            raise ValueError("Density is undefined for a degenerate coordinate.")
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return np.asarray(self._model.score_samples(x), dtype=float)


def _fit_joint_density(samples: np.ndarray, seed: int = 0) -> Optional[GaussianMixture]:
    """
    Multivariate density fitted to the *model's own* interventional samples.

    Needed because the truncated factorisation gives the joint over all
    non-intervened coordinates but no closed-form marginal over an arbitrary
    subset of coordinates (e.g. descendants only). Components are selected by BIC.
    """
    X = np.asarray(samples, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] < 5 or X.shape[1] == 0:
        return None
    best, best_bic = None, np.inf
    for k in (1, 2, 3, 5):
        if k * X.shape[1] > X.shape[0]:
            break
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             reg_covar=1e-4, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gm.fit(X)
            except Exception:
                continue
        bic = gm.bic(X)
        if bic < best_bic:
            best, best_bic = gm, bic
    return best


def _oof_residuals(mechanism, X_parents: np.ndarray, y: np.ndarray,
                   n_folds: int, seed: int):
    """
    Cross-fitted (out-of-fold) residuals of a node's mechanism.

    In-sample residuals of a flexible f fitted on ~30 rows are systematically too
    small; a density fitted to them is too narrow and held-out interventional
    samples then land in its tails, inflating the baseline's NLL for reasons that
    have nothing to do with the method.

    Returns `(residuals, cross_fitted)`. The fallback to in-sample residuals is
    silent in effect but not in reporting: `cross_fitted` is False on that path so
    the caller can report the true rate rather than assuming success.
    """
    pm = getattr(mechanism, "prediction_model", None)
    n = len(y)
    n_splits = int(min(n_folds, n))
    if pm is None or n_splits < 2 or not hasattr(pm, "clone"):
        return np.asarray(mechanism.estimate_noise(y, X_parents)).reshape(-1), False

    resid = np.empty(n, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        for tr, te in kf.split(X_parents):
            fold_model = pm.clone()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_model.fit(X_parents[tr], y[tr])
            resid[te] = y[te] - np.asarray(fold_model.predict(X_parents[te])).reshape(-1)
    except Exception:
        return np.asarray(mechanism.estimate_noise(y, X_parents)).reshape(-1), False
    return resid, True


def _support_distance(eval_parents: np.ndarray, obs_parents: np.ndarray) -> np.ndarray:
    """
    Per-sample standardised distance from a parent vector to the nearest
    observational parent vector.

    Under a knockout the downstream regressions are evaluated far outside their
    observational support; a flexible mechanism extrapolates arbitrarily there.
    This quantifies how far outside, so an underperforming baseline can be
    explained rather than merely suspected of being undertuned.
    """
    sd = np.std(obs_parents, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    A = eval_parents / sd
    B = obs_parents / sd
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).mean(axis=-1), 0.0))
    return d.min(axis=1)


def select_mechanisms_task(obs: np.ndarray, adj: np.ndarray, quality: str = "GOOD",
                           seed: int = 0) -> Dict[str, Any]:
    """
    Variant V2, selection pass: run auto-assignment on one task and report the
    prediction-model prototype chosen for each non-root node.

    Returns a dict with `labels` (list of str) and `prototypes` (list of dowhy
    PredictionModel instances, cloneable and picklable).
    """
    n_vars = obs.shape[1]
    names = node_names(n_vars)
    g = build_graph(adj, names)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Graph is cyclic; truncated factorisation is invalid.")

    df = pd.DataFrame(np.asarray(obs, dtype=float), columns=names)
    scm = gcm.StructuralCausalModel(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gcm.auto.assign_causal_mechanisms(scm, df, quality=QUALITY_MAP[quality])

    labels, prototypes = [], []
    for name in names:
        mech = scm.causal_mechanism(name)
        pm = getattr(mech, "prediction_model", None)
        if pm is None:
            continue
        labels.append(_prediction_model_label(mech))
        prototypes.append(pm.clone())
    return {"labels": labels, "prototypes": prototypes}


def fit_and_predict_task(
    obs: np.ndarray,
    true_int: np.ndarray,
    adj: np.ndarray,
    targets_onehot: np.ndarray,
    intervention_value: float = 0.0,
    quality: str = "GOOD",
    variant: str = "v1",
    fixed_prototype: Any = None,
    n_pred_samples: int = 30,
    n_density_samples: int = 1000,
    cross_fit_folds: int = 5,
    density: str = "kde",
    log_density_floor: float = -20.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Fit the GCM baseline on one evaluation task and predict every intervention.

    Args:
        obs: (n_obs, n_vars) observational data — exactly what ACTIVA receives.
        true_int: (n_targets, n_true, n_vars) held-out interventional samples,
            used only as evaluation targets (never at fit time).
        adj: (n_vars, n_vars) ground-truth adjacency, adj[i, j] == 1 means i -> j.
        targets_onehot: (n_targets, n_vars) intervention target masks.
        intervention_value: value v of the atomic intervention do(V_i = v).
        quality: 'GOOD' | 'BETTER' | 'BEST' auto-assignment quality (V1 only).
        variant: 'v1' per-task model selection, 'v2' fixed model class + per-task refit.
        fixed_prototype: dowhy PredictionModel prototype used by variant 'v2'.
        n_pred_samples: interventional samples drawn per intervention (matched to
            the evaluation sample count used for ACTIVA).
        n_density_samples: extra samples used only to estimate subset marginals.
        cross_fit_folds: K for the out-of-fold residuals (<2 disables cross-fitting).
        density: '1-D residual density estimator: 'kde' or 'gmm'.
        log_density_floor: per-coordinate floor on log p, applied identically to
            every node; the fraction of floored evaluations is reported.
        seed: seed for the fold splits and mixture initialisations.

    Returns a dict of predictions and reporting diagnostics (see keys below).
    """
    obs = np.asarray(obs, dtype=float)
    true_int = np.asarray(true_int, dtype=float)
    n_obs, n_vars = obs.shape
    n_targets = true_int.shape[0]
    n_true = true_int.shape[1]
    names = node_names(n_vars)

    g = build_graph(adj, names)
    # Blocking check: truncated factorisation requires a DAG.
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Graph is cyclic; truncated factorisation is invalid.")

    df = pd.DataFrame(obs, columns=names)
    scm = gcm.StructuralCausalModel(g)

    t0 = time.time()
    caught: List[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        if variant == "v2":
            if fixed_prototype is None:
                # Falling through to auto-assignment here would run V1 while every
                # label, filename and metadata field still claimed V2.
                raise ValueError(
                    "variant='v2' requires a fixed_prototype from the selection pass."
                )
            # Fixed model class per node, parameters re-fitted per task.
            for name in names:
                if len(get_ordered_predecessors(g, name)) == 0:
                    scm.set_causal_mechanism(name, gcm.EmpiricalDistribution())
                else:
                    scm.set_causal_mechanism(
                        name, gcm.AdditiveNoiseModel(fixed_prototype.clone())
                    )
        else:
            gcm.auto.assign_causal_mechanisms(scm, df, quality=QUALITY_MAP[quality])
        gcm.fit(scm, df)
        caught = [str(w.message)[:200] for w in wlist]
    fit_seconds = time.time() - t0

    # Mechanism-type assertions: a discrete or post-nonlinear mechanism would make
    # the joint quantity incoherent and incomparable with ACTIVA's continuous NLL.
    mech_labels: Dict[str, str] = {}
    for name in names:
        mech = scm.causal_mechanism(name)
        parents = get_ordered_predecessors(g, name)
        if not parents:
            mech_labels[name] = type(mech).__name__
            continue
        if type(mech).__name__ != "AdditiveNoiseModel":
            raise TypeError(
                f"Node {name}: expected a continuous AdditiveNoiseModel, got "
                f"{type(mech).__name__}. Densities and probability masses cannot be "
                f"mixed across nodes."
            )
        mech_labels[name] = _prediction_model_label(mech)

    # --- residual densities for the non-intervened factors (fit on obs data only)
    rng = np.random.default_rng(seed)
    node_density: Dict[str, _Density1D] = {}
    node_parents: Dict[str, List[int]] = {}
    oof_used = 0
    n_nonroot = 0
    for y_idx, name in enumerate(names):
        parents = get_ordered_predecessors(g, name)  # dowhy's own ordering
        p_idx = [names.index(p) for p in parents]
        node_parents[name] = p_idx
        if not p_idx:
            # Root factor: EmpiricalDistribution has draw_samples but no density,
            # so fit a separate 1-D estimator on the observational column. The SCM's
            # own mechanisms are left untouched, so sampling is unaffected.
            node_density[name] = _Density1D(density, seed).fit(obs[:, y_idx])
            continue
        n_nonroot += 1
        mech = scm.causal_mechanism(name)
        Xp = obs[:, p_idx]
        yv = obs[:, y_idx]
        if cross_fit_folds >= 2:
            resid, cross_fitted = _oof_residuals(mech, Xp, yv, cross_fit_folds, seed)
            oof_used += int(cross_fitted)
        else:
            resid = np.asarray(mech.estimate_noise(yv, Xp)).reshape(-1)
        node_density[name] = _Density1D(density, seed).fit(resid)

    # --- per-intervention prediction
    pred_samples = np.zeros((n_targets, n_pred_samples, n_vars), dtype=np.float32)
    logp_joint = np.full((n_targets, n_true), np.nan, dtype=np.float64)
    logp_nondesc = np.full((n_targets, n_true), np.nan, dtype=np.float64)
    nll_sampdens = np.full((n_targets, 3), np.nan, dtype=np.float64)  # all_vi/desc/nondesc
    floor_hits = 0
    floor_total = 0
    n_degenerate = 0
    support_desc: List[float] = []
    support_nondesc: List[float] = []

    for j in range(n_targets):
        intervened = np.flatnonzero(np.asarray(targets_onehot[j]) > 0.5)
        j_var = int(intervened[0]) if len(intervened) else j
        j_name = names[j_var]
        v = float(intervention_value)

        interventions = {j_name: (lambda x, _v=v: _v)}
        drawn = gcm.interventional_samples(
            scm, interventions, num_samples_to_draw=max(n_pred_samples, n_density_samples)
        )[names].to_numpy(dtype=float)
        # First block is the ERG/moment sample set, the rest only feeds the
        # subset-marginal density; both come from the same fitted model.
        perm = rng.permutation(len(drawn))
        pred_samples[j] = drawn[perm[:n_pred_samples]].astype(np.float32)
        dens_samples = drawn

        desc = _descendants(adj, j_var)
        dims_all_vi = [d for d in range(n_vars) if d != j_var]
        dims_desc = sorted(desc)
        dims_nondesc = sorted(set(range(n_vars)) - desc - {j_var})

        # --- factorised joint log-density of the held-out interventional samples
        vtrue = true_int[j]
        lp = np.zeros(n_true, dtype=float)
        # Non-descendants of the intervened node are ancestrally closed (a parent of
        # a non-descendant cannot be a descendant, nor the intervened node itself),
        # so the sum of their factors is the *exact* marginal over those coordinates
        # — no sample-density layer needed for that column.
        lp_nd = np.zeros(n_true, dtype=float)
        ok = True
        for y_idx, name in enumerate(names):
            if y_idx == j_var:
                # Atomic intervention: the intervened coordinate is degenerate under
                # do(V_i = v) and is dropped from every condition.
                continue
            dens = node_density[name]
            if dens.degenerate:
                # A constant coordinate has no density; drop the whole (task,
                # intervention) pair rather than substituting an arbitrary width,
                # so the reported NLL always covers the full coordinate set.
                ok = False
                n_degenerate += 1
                break
            p_idx = node_parents[name]
            if not p_idx:
                resid_eval = vtrue[:, y_idx]
            else:
                mech = scm.causal_mechanism(name)
                try:
                    resid_eval = np.asarray(
                        mech.estimate_noise(vtrue[:, y_idx], vtrue[:, p_idx])
                    ).reshape(-1)
                except Exception:
                    ok = False
                    break
                # Extrapolation is only expected downstream of the intervention, so
                # the two subsets are tracked apart; pooling them dilutes the signal.
                sd = float(np.mean(_support_distance(vtrue[:, p_idx], obs[:, p_idx])))
                (support_desc if y_idx in desc else support_nondesc).append(sd)
            lj = dens.log_density(resid_eval)
            floor_total += len(lj)
            floor_hits += int(np.sum(lj < log_density_floor))
            lj = np.maximum(lj, log_density_floor)
            lp += lj
            if y_idx not in desc:
                lp_nd += lj
        if ok:
            logp_joint[j] = lp
            if dims_nondesc:
                logp_nondesc[j] = lp_nd

        # --- remaining subset marginals via a density fitted to the model's samples
        for slot, dims in enumerate((dims_all_vi, dims_desc, dims_nondesc)):
            if not dims:
                continue
            gm = _fit_joint_density(dens_samples[:, dims], seed=seed)
            if gm is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ls = gm.score_samples(vtrue[:, dims])
            ls = np.maximum(ls, log_density_floor * len(dims))
            nll_sampdens[j, slot] = float(-np.mean(ls))

    return {
        "samples": pred_samples,
        "logp_joint": logp_joint,
        "logp_nondesc": logp_nondesc,
        "nll_sampdens": nll_sampdens,
        "mech_labels": mech_labels,
        "fit_seconds": fit_seconds,
        "warnings": caught,
        "floor_frac": float(floor_hits / floor_total) if floor_total else 0.0,
        "neg_frac": float(np.mean(pred_samples < 0.0)),
        "support_desc": float(np.mean(support_desc)) if support_desc else float("nan"),
        "support_nondesc": (float(np.mean(support_nondesc))
                            if support_nondesc else float("nan")),
        "n_degenerate": int(n_degenerate),
        "oof_fraction": float(oof_used / n_nonroot) if n_nonroot else float("nan"),
    }


def safe_fit_and_predict_task(*args, **kwargs) -> Dict[str, Any]:
    """
    `fit_and_predict_task` with per-task failure isolation.

    A single unfittable task must not abort an evaluation over thousands of tasks;
    failures are returned so the driver can count and report them.
    """
    try:
        return fit_and_predict_task(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"failed": f"{type(exc).__name__}: {exc}"}


def safe_select_mechanisms_task(*args, **kwargs) -> Dict[str, Any]:
    """`select_mechanisms_task` with per-task failure isolation."""
    try:
        return select_mechanisms_task(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"labels": [], "prototypes": [], "failed": f"{type(exc).__name__}: {exc}"}


def _descendants(adj: np.ndarray, node: int) -> set:
    """Descendants of `node` (BFS), matching evaluate_causal.get_descendants."""
    adj = np.asarray(adj)
    out, queue, seen = set(), [node], {node}
    while queue:
        cur = queue.pop(0)
        if cur != node:
            out.add(cur)
        for child in np.flatnonzero(adj[cur] == 1):
            child = int(child)
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return out
