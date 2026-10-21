"""
Plot causal evaluation results from one or more evaluate_causal.py JSON outputs.

Usage:
    python plot_causal_results.py results1.json results2.json ... [options]

Each JSON file is treated as one dataset. The dataset label defaults to the
data_path stored in the file's _meta, but can be overridden with --labels.

Output: one PDF/PNG per metric type (ATE, IntVar, CATE), plus an
optional combined figure.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_ORDER = [
    "CVAE (ours)",
    "ACTIVA (ours)",
    "CVAE top-1",
    "CVAE top-2",
    "CVAE top-10",
    "ACTIVA best-comp",
    "MACE-TNP",
    "MACE-TNP top-1",
    "MACE-TNP top-2",
    "MACE-TNP top-10",
    "MACE-TNP best-comp",
    # "Bad (Obs. Mean)",
    "Linear Causal",
    "Gaussian Conditional",
    "Backdoor Adjustment",
    "NOTEARS",
    "Causal Forest (RF S-Learner)",
]

# Short display names for the legend (keeps figures readable)
METHOD_SHORT = {
    "Linear Causal":                  "Linear",
    "CVAE (ours)":                    "ACTIVA (ours)",
    "ACTIVA (ours)":                  "ACTIVA (ours)",
    "CVAE top-1":                     "ACTIVA top-1",
    "CVAE top-2":                     "ACTIVA top-2",
    "CVAE top-10":                    "ACTIVA top-10",
    "ACTIVA best-comp":               "ACTIVA best-comp",
    "MACE-TNP":                       "MACE-TNP",
    "MACE-TNP top-1":                 "MACE-TNP top-1",
    "MACE-TNP top-2":                 "MACE-TNP top-2",
    "MACE-TNP top-10":                "MACE-TNP top-10",
    "MACE-TNP best-comp":             "MACE-TNP best-comp",
    "Gaussian Conditional":           "Gaussian",
    "Bad (Obs. Mean)":         "Obs. Mean",
    "Backdoor Adjustment":            "Backdoor",
    "NOTEARS":                        "NOTEARS",
    "Causal Forest (RF S-Learner)":   "CausalForest",
}

METRIC_CONFIGS = {
    'ate': {
        'title': 'ATE',
        'full_title': 'ATE  —  $\\mathbb{E}[\\mathbf{V}\\,|\\,\\mathrm{do}(V_j{=}v_j)] - \\mathbb{E}[\\mathbf{V}\\,|\\,\\mathrm{do}(\\emptyset)]$',
        'panels': [
            ('MAE_all_vi',         'MAE (All/$V_i$)'),
            ('MAE_descendants',    'MAE (descendants)'),
            ('MAE_non_descendants','MAE (non-descendants)'),
        ],
    },
    # 'intmean': {  # commented out — redundant with ATE
    #     'title': 'IntMean',
    #     'full_title': 'Interventional Mean  —  $\\mathbb{E}[\\mathbf{V}\\,|\\,\\mathrm{do}(V_j{=}v_j)]$',
    #     'panels': [
    #         ('MAE_all',         'MAE (all pairs)'),
    #         ('MAE_descendants', 'MAE (descendants)'),
    #     ],
    # },
    'intvar': {
        'title': 'IntVar',
        'full_title': 'Interventional Variance  —  $\\mathrm{Var}[\\mathbf{V}\\,|\\,\\mathrm{do}(V_j{=}v_j)]$',
        'panels': [
            ('MAE_all_vi',         'MAE (All/$V_i$)'),
            ('MAE_descendants',    'MAE (descendants)'),
            ('MAE_non_descendants','MAE (non-descendants)'),
        ],
        'log_scale': True,
    },
    'cate': {
        'title': 'CATE',
        'full_title': 'CATE  —  Conditional Average Treatment Effect',
        'panels': [
            ('CATE_MAE_all',            'MAE (All/$V_i$)'),
            ('CATE_MAE_descendants',    'MAE (descendants)'),
            ('CATE_MAE_non_descendants','MAE (non-descendants)'),
        ],
    },
    'intcov': {
        'title': 'IntCov',
        'full_title': 'Interventional Covariance  —  $\\mathrm{Cov}[\\mathbf{V}\\,|\\,\\mathrm{do}(V_j{=}v_j)]$',
        'panels': [
            ('CovMAE_all',          'MAE (All/$V_i$)'),
            ('CovMAE_both_desc',    'MAE (both descendants)'),
            ('CovMAE_neither_desc', 'MAE (neither descendant)'),
        ],
        'log_scale': True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_dataset_label(meta_data_path: str) -> str:
    """Convert a data_path stored in _meta to a short readable label."""
    stem = os.path.splitext(os.path.basename(meta_data_path))[0]
    # Strip common suffixes like _test / _train
    for suffix in ('_test', '_train'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    # Apply custom label mapping if available
    return DATASET_LABEL_MAP.get(stem, stem)


def load_results(paths, labels):
    """Load and label each JSON results file."""
    datasets = []
    for i, p in enumerate(paths):
        with open(p) as f:
            data = json.load(f)
        if labels and i < len(labels):
            label = labels[i]
        else:
            meta = data.get('_meta', {})
            label = short_dataset_label(meta.get('data_path', os.path.basename(p)))
        datasets.append({'label': label, 'data': data, 'path': p})
    return datasets


def get_methods(datasets, metric_key):
    """Return ordered list of methods present in at least one dataset."""
    all_methods = set()
    for ds in datasets:
        all_methods.update(ds['data'].get(metric_key, {}).keys())
    return [m for m in METHOD_ORDER if m in all_methods]


def extract_values(datasets, metric_key, stat_key):
    """
    Returns (values, ci_lower, ci_upper) each of shape (n_methods, n_datasets).
    Missing entries become NaN.
    """
    methods = get_methods(datasets, metric_key)
    n_m, n_d = len(methods), len(datasets)
    vals  = np.full((n_m, n_d), np.nan)
    ci_lo = np.full((n_m, n_d), np.nan)
    ci_hi = np.full((n_m, n_d), np.nan)
    for di, ds in enumerate(datasets):
        metric_data = ds['data'].get(metric_key, {})
        for mi, method in enumerate(methods):
            if method not in metric_data:
                continue
            md = metric_data[method]
            v = md.get(stat_key)
            if v is not None:
                vals[mi, di] = v
            lo = md.get(f'{stat_key}_ci_lower')
            hi = md.get(f'{stat_key}_ci_upper')
            if lo is not None:
                ci_lo[mi, di] = lo
            if hi is not None:
                ci_hi[mi, di] = hi
    return methods, vals, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD',
]

DATASET_LABEL_MAP = {
    '2var_linadd_normal': 'Gauss 2',
    '2var_linadd_beta': 'Beta 2',
    '8var_linadd_normal': 'Gauss 8',
    '8var_linadd_beta': 'Beta 8',
    '8var_sergio_nodrop': 'SERGIO',
}


def plot_panel(ax, methods, vals, ci_lo, ci_hi, dataset_labels, panel_title,
               log_scale=False, show_legend=False):
    """Draw a grouped bar chart for one (metric_key, stat_key) panel.

    X-axis groups = datasets; bar colors = methods.
    """
    n_methods = len(methods)
    n_datasets = len(dataset_labels)

    group_w = 0.8
    bar_w = group_w / n_methods
    group_centers = np.arange(n_datasets)

    for mi, method in enumerate(methods):
        offsets = group_centers - group_w / 2 + bar_w * (mi + 0.5)
        color = METHOD_COLORS[mi % len(METHOD_COLORS)]
        v = vals[mi, :]
        lo = np.where(np.isfinite(ci_lo[mi, :]), ci_lo[mi, :], v)
        hi = np.where(np.isfinite(ci_hi[mi, :]), ci_hi[mi, :], v)
        yerr_lo = np.where(np.isfinite(v), np.abs(v - lo), 0)
        yerr_hi = np.where(np.isfinite(v), np.abs(hi - v), 0)

        short_name = METHOD_SHORT.get(method, method)
        ax.bar(
            offsets, v,
            width=bar_w * 0.9,
            color=color,
            label=short_name,
            alpha=0.85,
            zorder=3,
        )
        # Error bars only where we have finite values
        valid = np.isfinite(v)
        if valid.any():
            ax.errorbar(
                offsets[valid], v[valid],
                yerr=[yerr_lo[valid], yerr_hi[valid]],
                fmt='none',
                color='black',
                capsize=3,
                linewidth=1,
                zorder=4,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('MAE', fontsize=9)
    ax.set_title(panel_title, fontsize=10, pad=4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    if log_scale:
        ax.set_yscale('log')

    if show_legend:
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # Zero line
    ax.axhline(0, color='black', linewidth=0.5)


def make_figure(metric_key, config, datasets, out_path, log_scale=False, dpi=150):
    panels = config['panels']
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.2),
                              constrained_layout=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(config['full_title'], fontsize=12, fontweight='bold')

    dataset_labels = [ds['label'] for ds in datasets]
    methods = get_methods(datasets, metric_key)

    legend_added = False
    for ax, (stat_key, panel_title) in zip(axes, panels):
        _, vals, ci_lo, ci_hi = extract_values(datasets, metric_key, stat_key)
        show_legend = not legend_added
        plot_panel(ax, methods, vals, ci_lo, ci_hi, dataset_labels,
                   panel_title, log_scale=log_scale, show_legend=show_legend)
        legend_added = True

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_combined_figure(datasets, metric_keys, out_path, log_scale=False, dpi=150):
    """2×2 (or 1×N) grid with one panel per metric using MAE_all / CATE_MAE_all."""
    n = len(metric_keys)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    dataset_labels = [ds['label'] for ds in datasets]
    legend_added = False

    for i, mk in enumerate(metric_keys):
        ax = axes_flat[i]
        config = METRIC_CONFIGS[mk]
        stat_key = config['panels'][0][0]  # first panel = MAE_all
        methods = get_methods(datasets, mk)
        _, vals, ci_lo, ci_hi = extract_values(datasets, mk, stat_key)
        show_legend = not legend_added
        plot_panel(ax, methods, vals, ci_lo, ci_hi, dataset_labels,
                   config['title'] + ' — MAE (all)', log_scale=log_scale,
                   show_legend=show_legend)
        legend_added = True

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot causal evaluation results from evaluate_causal.py JSON outputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'results', nargs='+', metavar='results.json',
        help='One or more JSON result files produced by evaluate_causal.py.',
    )
    parser.add_argument(
        '--labels', nargs='+', metavar='LABEL',
        help='Dataset display labels (one per JSON file). Defaults to data_path in _meta.',
    )
    parser.add_argument(
        '--metrics', nargs='+', default=['ate', 'intvar', 'cate', 'intcov'],
        choices=['ate', 'intvar', 'cate', 'intcov'],
        help='Which metric types to plot (default: all).',
    )
    parser.add_argument(
        '--out-dir', default='.',
        help='Directory to write output figures (default: current dir).',
    )
    parser.add_argument(
        '--format', default='pdf', choices=['pdf', 'png', 'svg'],
        help='Output file format (default: pdf).',
    )
    parser.add_argument(
        '--log-scale', action='store_true',
        help='Use log scale on the Y-axis.',
    )
    parser.add_argument(
        '--combined', action='store_true', default=True,
        help='Also produce a combined 2×2 overview figure (default: true).',
    )
    parser.add_argument(
        '--no-combined', dest='combined', action='store_false',
        help='Skip the combined overview figure.',
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for raster formats (default: 150).',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {len(args.results)} result file(s)...")
    datasets = load_results(args.results, args.labels)
    for ds in datasets:
        print(f"  [{ds['label']}]  {ds['path']}")

    print(f"\nGenerating figures in '{args.out_dir}/'...")
    ext = args.format

    for mk in args.metrics:
        config = METRIC_CONFIGS[mk]
        out_path = os.path.join(args.out_dir, f'causal_{mk}.{ext}')
        log_scale = config.get('log_scale', args.log_scale)
        make_figure(mk, config, datasets, out_path,
                    log_scale=log_scale, dpi=args.dpi)

    if args.combined and len(args.metrics) > 1:
        out_path = os.path.join(args.out_dir, f'causal_overview.{ext}')
        make_combined_figure(datasets, args.metrics, out_path,
                              log_scale=args.log_scale, dpi=args.dpi)

    print("\nDone.")


if __name__ == '__main__':
    main()
