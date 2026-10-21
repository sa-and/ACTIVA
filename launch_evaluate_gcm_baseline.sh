#!/bin/bash
#SBATCH --job-name=gcm_baseline
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=gcm_baseline_%j.log

# Launch the nonlinear graph-privileged DoWhy-GCM baseline on the cluster.
# CPU-only: per-task fitting is parallelised across evaluation tasks with joblib.
#
# Usage:
#   sbatch launch_evaluate_gcm_baseline.sh --data-path <...> [other evaluate_gcm_baseline.py flags]
# Example (SERGIO, 8 variables, knockout interventions):
#   sbatch launch_evaluate_gcm_baseline.sh \
#     --data-path data/TMLR/8var_sergio_nodrop_test.npz \
#     --intervention-value 0.0 --n-eval-scms 200 --variant v1

cd /home/asr448/Amortized_Interventional_Distribution_Estimation

source ~/.bashrc
conda activate mace_tnp
# dowhy lives in a venv layered on the conda env, so the conda env is left untouched.
source ~/.venvs/dowhy_gcm/bin/activate
unset PYTHONPATH

# One BLAS thread per worker: joblib already uses every allocated core.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

N_JOBS=${SLURM_CPUS_PER_TASK:-8}
echo "Running with n_jobs=${N_JOBS}"

python evaluate_gcm_baseline.py --n-jobs "${N_JOBS}" "$@"
