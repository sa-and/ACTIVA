#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C A5000
##SBATCH -p fatq
#SBATCH --gres=gpu:1
# Load GPU drivers
module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.4.0.58
# Load shell configuration
source /home/asr448/.bashrc

# Ensure conda uses scratch for environments and builds
export CONDA_ENVS_PATH=/var/scratch/asr448/conda/envs
export TMPDIR=/var/scratch/asr448
export PIP_CACHE_DIR=/var/scratch/asr448/pip-cache

# Data conversion prerequisite (one-time, skip if HDF5 already exists):
# If needed, run manually from project root:
#   python comparisons/convert_data_to_hdf5.py \
#       --npz-train data/AAAI/8var_linadd_beta_train.npz \
#       --npz-test  data/AAAI/8var_linadd_beta_test.npz \
#       --output-dir data/mace_tnp/8var_linadd_beta

conda activate mace_tnp
# NOTE: Run from project root, not from comparisons/ directory
#   sbatch comparisons/launch_mace_tnp.sh
cd /home/asr448/Amortized_Interventional_Distribution_Estimation
srun python comparisons/train_mace_tnp.py \
  --data-dir /var/scratch/asr448/data/mace_tnp/8var_sergio_nodrop \
  --save-path ./models/mace_tnp_8var_sergio_nodrop.pt \
  --epochs 1000 --batch-size 256 --n-vars 8 \
  --intervention-value 0.0 \
  --track
