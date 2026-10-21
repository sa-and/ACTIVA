# Baselines and Comparisons

This directory contains the MACE-TNP baseline model and supporting scripts. Two additional analytical baselines (Gaussian conditional and linear causal) are implemented directly inside `evaluate_causal.py` and require no separate setup.

## Overview of baselines

| Baseline | Where implemented | Graph required | Notes |
|----------|-------------------|----------------|-------|
| Zero ATE | `evaluate_causal.py` | No | Lower bound reference |
| Gaussian conditional | `evaluate_causal.py` | No | Fits observational Gaussian; uses conditional distribution |
| Linear causal | `evaluate_causal.py` | Yes (oracle) | Topological OLS regression; optimal for linear SCMs |
| MACE-TNP | `comparisons/` | No | Neural process baseline; separate PyTorch environment |

The analytical baselines run automatically when you call `evaluate_causal.py` — no additional steps needed.

## MACE-TNP

MACE-TNP is a joint-multivariate extension of the MACE architecture from [CausalInferenceNeuralProcess](https://github.com/Anish144/CausalInferenceNeuralProcess). It uses a permutation-equivariant Transformer encoder (MACE) and a K-component full-covariance Gaussian mixture decoder (`MultivariateMoGPredictor`) to predict the joint interventional distribution over all variables simultaneously.

It runs in a **separate conda environment** (`mace_tnp`) using PyTorch instead of JAX.

### Architecture

- **Encoder**: MACE (Masked Amortized Causal Effect) — permutation-equivariant Transformer that processes observational context and interventional target data via multi-head cross-attention.
- **Decoder**: `MultivariateMoGPredictor` — decodes encoder outputs to a K-component full-covariance Gaussian mixture over all `n_vars` jointly.
- **Loss**: Joint multivariate NLL under the GMM.

Key design choices matched to ACTIVA for a fair comparison:

| Aspect | Setting | Rationale |
|--------|---------|-----------|
| `d_model` | 88 | Matched to ACTIVA parameter count (~746k params) |
| `num_mixture_components` | 20 | Matched to ACTIVA's prior components |
| Attention mode | MHCA (Multi-Head Cross-Attention) | Paper default |
| Cholesky parameterization | Full lower-triangular | Captures inter-variable correlations |
| Intervention value at test | Hardcoded to match data (e.g. 5.0 for linear/beta, 0.0 for SERGIO) | Original CITNP uses all-zeros, which after z-score normalization equals the population mean rather than the true intervention value |

### Setup (one-time)

```bash
conda create -n mace_tnp python=3.10 -y
conda activate mace_tnp
pip install -r comparisons/requirements_mace_tnp.txt
cd ~/CausalInferenceNeuralProcess && pip install -e . && cd -
```

### Usage

All commands are run from the **project root**, not from `comparisons/`.

**Step 1 — Convert data** (one-time per dataset):

```bash
python comparisons/convert_data_to_hdf5.py \
  --npz-train data/TMLR/8var_linadd_normal_train.npz \
  --npz-val   data/TMLR/8var_linadd_normal_val.npz \
  --npz-test  data/TMLR/8var_linadd_normal_test.npz \
  --output-dir data/mace_tnp/8var_linadd_normal
```

**Step 2 — Train:**

```bash
python comparisons/train_mace_tnp.py \
  --data-dir data/mace_tnp/8var_linadd_normal \
  --save-path models/mace_tnp_8var_linadd_normal.pt \
  --epochs 500 --batch-size 64 --n-vars 8 --track
```

**Step 3 — Evaluate:**

```bash
python comparisons/evaluate_mace_tnp.py \
  --data-path data/TMLR/8var_linadd_normal_test.npz \
  --model-path models/mace_tnp_8var_linadd_normal_best.pt \
  --n-samples 100 --n-eval-scms 200
```

Reports the same distribution metrics (MMD, Wasserstein, energy) and ATE metrics (MAE/RMSE, descendant/non-descendant split) as `evaluate_model.py` and `evaluate_causal.py`.

**SLURM submission:**

```bash
sbatch comparisons/launch_mace_tnp.sh
```

### Files

| File | Purpose |
|------|---------|
| `mace_tnp_model.py` | Model definition (`JointMoGCausalInferenceModel`) |
| `train_mace_tnp.py` | Training loop |
| `evaluate_mace_tnp.py` | Evaluation (PyTorch/NumPy, no JAX) |
| `convert_data_to_hdf5.py` | NPZ → HDF5 conversion for CITNP's dataset API |
| `conditional_distribution_baseline.py` | Standalone Gaussian conditional baseline (development use) |
| `launch_mace_tnp.sh` | SLURM submission script |
| `requirements_mace_tnp.txt` | PyTorch-based dependencies |
