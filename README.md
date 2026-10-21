# ACTIVA: Amortized Causal Effect Estimation via Variational Autoencoders

This repository contains the official code for the ACTIVA model, a Conditional VAE for amortized interventional distribution estimation in causal systems. Given observational data from a structural causal model (SCM) and an intervention label, ACTIVA predicts the full interventional distribution without retraining per intervention.

- Paper: [Transactions on Machine Learning Research (OpenReview)](https://openreview.net/forum?id=819pIIPuVl)
- Datasets and pre-trained checkpoints: [OSF](https://osf.io/5vebr/)

## Installation

Python 3.10+ is required.

```bash
conda create -n activa python=3.10 -y
conda activate activa
```

Install JAX with CUDA 12 support (for GPU; the cluster setup uses CUDA 12.3 on A100 GPUs):

```bash
pip install "jax[cuda12]==0.4.33"
```

For CPU-only (e.g. local development):

```bash
pip install "jax[cpu]==0.4.33"
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

**Weights & Biases** is used for experiment tracking. `train_vae.py` logs to it by default (`--track`), which needs `WANDB_API_KEY` in your environment. Pass `--no-track` to train without it.

**Note on `torch`**: PyTorch appears in `requirements.txt` because `generate_data.py` uses a small MLP (in PyTorch) as a random structural function for non-linear SCMs. It is not used during training or evaluation.

## Repository structure

```
.
├── models.py                  # ACTIVA model (ConditionalVAE, encoder, prior, decoder)
├── train_vae.py               # Training loop and CLI
├── evaluate_model.py          # Distribution-level evaluation (MMD, energy, NLL)
├── evaluate_causal.py         # Causal evaluation (ATE, CATE)
├── gcm_baseline.py            # DoWhy-GCM graph-privileged baseline (per-task fitting)
├── evaluate_gcm_baseline.py   # Evaluation runner for the DoWhy-GCM baseline
├── gen_ambiguous_scms.py      # Data for the non-identifiable setting (Sec. 7.4)
├── evaluate_ambiguous.py      # Evaluation and plots for the non-identifiable setting
├── plot_causal_results.py     # Plots from evaluate_causal.py JSON outputs
├── generate_data.py           # Synthetic SCM dataset generation
├── utils.py                   # Shared I/O, metrics, plotting utilities
├── launch.sh                  # SLURM submission script
├── data/TMLR/                 # Datasets used in the paper
├── models/TMLR/               # Pre-trained model checkpoints and result JSONs
└── comparisons/               # Baseline implementations (MACE-TNP, Gaussian, linear)
```

## Reproducing paper results

The datasets and pre-trained checkpoints are available on [OSF](https://osf.io/5vebr/). Place them under `data/TMLR/` and `models/TMLR/`, the paths used below.

### Datasets

| Name | Variables | SCM type | Noise |
|------|-----------|----------|-------|
| `2var_linadd_normal` | 2 | Linear additive | Gaussian |
| `2var_linadd_beta` | 2 | Linear additive | Beta |
| `8var_linadd_normal` | 8 | Linear additive | Gaussian |
| `8var_linadd_beta` | 8 | Linear additive | Beta |
| `8var_sergio_nodrop` | 8 | Non-linear (SERGIO gene-regulatory) | — |

### Evaluate a pre-trained model

**Distribution metrics** (MMD, Wasserstein, energy distance, NLL):

```bash
python evaluate_model.py \
  --data-path data/TMLR/8var_linadd_normal_test.npz \
  --model-path models/TMLR/8var_linadd_normal_peach_sun_best
```

**Causal metrics** (ATE MAE/RMSE, CATE; all baselines included):

```bash
python evaluate_causal.py \
  --data-path data/TMLR/8var_linadd_normal_test.npz \
  --model-path models/TMLR/8var_linadd_normal_peach_sun_best \
  --n-eval-scms 1000 --samples-from-prior 20 --n-model-samples 200
```

The paper evaluates 1000 SCMs on the 8-variable test sets and 200 on the 2-variable and `sergio_nodrop_sachs_ood` test sets, with seed 42 (the default). SERGIO interventions are knockouts: add `--intervention-value 0.0` for `8var_sergio_nodrop` and `sergio_nodrop_sachs_ood`.

Results are saved as JSON alongside the checkpoint. Pass `--results-path` to override the output path.

**Plot results** across datasets:

```bash
python plot_causal_results.py \
  models/TMLR/results_8var_linadd_normal_peach_sun_best_8var_linadd_normal_test.json \
  models/TMLR/results_8var_linadd_beta_youthful_resonance_best_8var_linadd_beta_test.json \
  models/TMLR/results_8var_sergio_nodrop_comfy_sea_best_8var_sergio_nodrop_test.json
```

### Pre-trained checkpoints

| Dataset | Checkpoint |
|---------|------------|
| `2var_linadd_normal` | `models/TMLR/2var_linadd_normal_pious_flower_best` |
| `2var_linadd_beta` | `models/TMLR/2var_linadd_beta_lively_spaceship_best` |
| `8var_linadd_normal` | `models/TMLR/8var_linadd_normal_peach_sun_best` |
| `8var_linadd_beta` | `models/TMLR/8var_linadd_beta_youthful_resonance_best` |
| `8var_sergio_nodrop` | `models/TMLR/8var_sergio_nodrop_comfy_sea_best` |

Checkpoints are dill files loaded with `utils.load_model_state(path)`.

### Non-identifiable setting (Sec. 7.4)

Two linear-Gaussian SCMs over (X, Y), M_A: X → Y and M_B: Y → X, both with edge weight 0.8 and noise variance 0.36, induce the same observational distribution N(0, [[1, 0.8], [0.8, 1]]). They differ under do(X = 5): Y ~ N(4.00, 0.36) under M_A and Y ~ N(0, 1) under M_B. All 1024 tasks share one observational dataset, and 75% of them carry interventional data from M_A, so the best possible prediction is the mixture 0.75·N(4.00, 0.36) + 0.25·N(0, 1).

```bash
# 1. Data: 1024 tasks, pi_A = 0.75, 80/10/10 split that preserves the A/B ratio.
#    Writes data/ambiguous_2var_75_{train,val,test}.npz and _meta.json (analytic ground truth).
python gen_ambiguous_scms.py --save-dir data/ambiguous_2var_75 \
  --p-scm-a 0.75 --n-scms 1024 --seed 0

# 2. Train a small model: hidden and embedding dimension 16, 2 encoder and 2 decoder
#    blocks, C = 20 prior and B = 2 decoder components; all other flags at their defaults.
python train_vae.py --data-path data/ambiguous_2var_75 \
  --save-path models/TMLR/ambiguous_2var_75 \
  --hidden-dim 16 --embedding-dim 16 --attention-heads 2 \
  --encoder-attention-layers 2 --decoder-attention-layers 2 \
  --prior-mixture-components 20 --decoder-mixture-components 2

# 3. Evaluate and plot: prior path (used at inference) vs posterior path, in data and
#    latent space, with the predicted mode weights, means and variances.
python evaluate_ambiguous.py --data-path data/ambiguous_2var_75 \
  --model-path models/TMLR/ambiguous_2var_75 --output-prefix ambiguous_75 \
  --samples-from-prior 50000
```

Step 3 prints the mode decomposition against the ground truth and writes `ambiguous_75_data_prior`, `ambiguous_75_data_posterior` and `ambiguous_75_latent_zY` as `.png` and `.pdf`.

## Running new experiments

### 1. Generate data

```bash
python generate_data.py \
  --save-dir data/TMLR/8var_linadd_normal \
  --samples 30 --n-endo 8 --n-graphs 500 --n-scms-per-graph 1 \
  --data-mode interventional
```

Produces three files: `<save-dir>_train.npz`, `<save-dir>_val.npz`, `<save-dir>_test.npz`. Data generation runs on CPU (the paper's datasets were generated on CPU nodes with `launchgendata.sh`). On a machine with a GPU, prefix the command with `JAX_PLATFORMS=cpu`: otherwise every worker process initialises JAX on the GPU and runs out of memory. Run with `--help` for all options including the structural functions and noise distributions (`--data-generating-function`).

### 2. Train

```bash
python train_vae.py \
  --data-path data/TMLR/8var_linadd_normal \
  --save-path models/my_model \
  --epochs 5000 --batch-size 256
```

`--data-path` is the base path (the script appends `_train.npz` and `_val.npz`). Checkpoints are saved every `--logging-freq` epochs, and the best checkpoint (by validation NLL) is saved with a `_best` suffix.

Key hyperparameters used in the paper:

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden-dim` | 256 | FFN width in Transformer blocks |
| `--embedding-dim` | 64 | Attention Q/K/V dimension |
| `--encoder-attention-layers` | 4 | Must be even (alternates sample/variable axes) |
| `--decoder-attention-layers` | 4 | |
| `--decoder-mixture-components` | 4 | GMM components in decoder |
| `--prior-mixture-components` | 20 | GMM components in prior |
| `--kl-beta` | 0.5 | Weight on KL divergence term |

Submit to SLURM:

```bash
sbatch launch.sh
```

The `launch*.sh` scripts were written for the DAS6 cluster. Their module loads and absolute paths (`/home/asr448/...`, `/var/scratch/asr448/...`) need adjusting before running elsewhere.

### 3. Evaluate

**Distribution metrics:**

```bash
python evaluate_model.py \
  --data-path data/TMLR/8var_linadd_normal_test.npz \
  --model-path models/my_model_best
```

Prints MMD, Wasserstein distance, energy distance, NLL, and descendant/non-descendant marginal breakdowns.

**Causal metrics:**

```bash
python evaluate_causal.py \
  --data-path data/TMLR/8var_linadd_normal_test.npz \
  --model-path models/my_model_best \
  --n-eval-scms 2000 --samples-from-prior 20
```

Compares ACTIVA against zero-ATE, Gaussian conditional, and linear causal baselines on ATE MAE/RMSE split by descendant/non-descendant pairs.

## Model architecture

ACTIVA is a Conditional VAE with three components:

- **Encoder** (`EquivariantNPTEncoder`): takes interventional samples + observational context → posterior `q(z | x, dm)`. Uses alternating Transformer attention over sample and variable axes to maintain permutation equivariance.
- **Prior** (`encode_prior`): takes observational context + intervention specification → GMM prior `p(z | intervention)`. Shares the observational encoder with the posterior.
- **Decoder** (`TransformerDecoder`): maps latent `z` (shape: `batch × n_vars × 1`) → Gaussian mixture over data space.

**Inference**: sample `z ~ p(z | intervention)` from the prior, decode to a GMM, combine `n_prior_samples` decoded mixtures into a two-level mixture approximating the interventional marginal.

**Loss**:
```
L = -E[log p(x | z)] + kl_beta * KL(q(z | x, dm) || p(z | intervention))
```

Parameters are `float32`; activations use `bfloat16`. Optimizer: AdamW via `optax.apply_if_finite` (NaN-safe).

## Baselines and comparisons

The `comparisons/` directory contains the MACE-TNP baseline and supporting scripts. Analytical baselines (Gaussian conditional, linear causal) are implemented directly in `evaluate_causal.py`.

See [`comparisons/README.md`](comparisons/README.md) for setup and usage.

### Nonlinear graph-privileged baseline (DoWhy-GCM)

`gcm_baseline.py` and `evaluate_gcm_baseline.py` implement a flexible graph-privileged
estimator built on [`dowhy.gcm`](https://github.com/py-why/dowhy) (Blöbaum et al.,
*DoWhy-GCM*, JMLR 25(147), 2024). Like the Linear Causal baseline it receives the
ground-truth graph and exactly the observational data ACTIVA sees, but it does not
impose linear-Gaussian mechanisms: root nodes get an empirical distribution and
non-root nodes an additive noise model `Y = f(pa(Y)) + N` whose regressor is selected
from data. It is a *privileged-information* comparison, not an oracle.

It runs independently of `evaluate_causal.py` (no ACTIVA/MACE-TNP checkpoint required)
but imports that script's metric, bootstrap and table code, so the numbers are directly
comparable. Per-task fitting runs in parallel across evaluation tasks with joblib.

```bash
# dowhy is kept out of the main environment, in a venv layered on top of it:
python -m venv --system-site-packages ~/.venvs/dowhy_gcm
~/.venvs/dowhy_gcm/bin/pip install dowhy==0.12
source ~/.venvs/dowhy_gcm/bin/activate

# V1 (primary): per-task model selection
python evaluate_gcm_baseline.py --data-path data/TMLR/8var_sergio_nodrop_test.npz \
  --intervention-value 0.0 --n-eval-scms 1000 --variant v1 --n-jobs 32

# V2 (robustness): model class fixed across tasks, parameters refit per task
python evaluate_gcm_baseline.py --data-path data/TMLR/8var_sergio_nodrop_test.npz \
  --intervention-value 0.0 --n-eval-scms 1000 --variant v2 --n-jobs 32
```

On the cluster, `sbatch launch_evaluate_gcm_baseline.sh <flags>` runs the same command, and
`sbatch test_evaluate_gcm_baseline.sh` is a smoke test on 8 tasks.

Use the same `--seed`, `--n-eval-scms` and `--intervention-value` as `evaluate_causal.py`
so both scripts evaluate the same SCMs. Both print an `eval_indices fingerprint` line,
which must match. `--merge-into <results.json>` adds the baseline's rows to an existing
results file. The estimator, the NLL construction and its diagnostics are documented in
`gcm_baseline.py`.

## Citation

```bibtex
@article{sauter2026activa,
  title   = {{ACTIVA}: Amortized Causal Effect Estimation via Variational Autoencoders},
  author  = {Sauter, Andreas and Salehkaleybar, Saber and van Harmelen, Frank and Plaat, Aske and Acar, Erman},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=819pIIPuVl}
}
```
