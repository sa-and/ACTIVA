#!/bin/bash
#SBATCH --job-name=test_eval_causal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test_eval_causal.log

# Test evaluate_causal.py on a small dataset
cd /home/asr448/Amortized_Interventional_Distribution_Estimation

# Load GPU drivers
module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.4.0.58

source ~/.bashrc
conda activate mace_tnp
unset PYTHONPATH

echo "=========================================="
echo "Testing evaluate_causal.py"
echo "=========================================="
echo ""

python evaluate_causal.py \
  --data-path data/TMLR/8var_sergio_nodrop_test.npz \
  --model-path models/TMLR/8var_sergio_nodrop_comfy_sea_best \
  --mace-tnp-model-path models/mace_tnp_8var_sergio_nodrop_best.pt \
  --n-eval-scms 1000 \
  --top-k-components 0 \
  --intervention-value 0.0 \
  --samples-from-prior 10 \
  --n-model-samples 200 \
  --seed 42

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
