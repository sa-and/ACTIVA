#!/bin/bash
#SBATCH --job-name=test_gcm_baseline
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=test_gcm_baseline.log

# Smoke test for evaluate_gcm_baseline.py on a handful of SERGIO tasks.

cd /home/asr448/Amortized_Interventional_Distribution_Estimation

source ~/.bashrc
conda activate mace_tnp
source ~/.venvs/dowhy_gcm/bin/activate
unset PYTHONPATH

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu

echo "=========================================="
echo "Testing evaluate_gcm_baseline.py (V1)"
echo "=========================================="
python evaluate_gcm_baseline.py \
  --data-path data/TMLR/8var_sergio_nodrop_test.npz \
  --intervention-value 0.0 \
  --n-eval-scms 8 \
  --n-jobs 8 \
  --n-bootstrap 50 \
  --variant v1 \
  --results-path /tmp/test_gcm_v1.json

echo ""
echo "=========================================="
echo "Testing evaluate_gcm_baseline.py (V2)"
echo "=========================================="
python evaluate_gcm_baseline.py \
  --data-path data/TMLR/8var_sergio_nodrop_test.npz \
  --intervention-value 0.0 \
  --n-eval-scms 8 \
  --v2-selection-tasks 4 \
  --n-jobs 8 \
  --n-bootstrap 50 \
  --variant v2 \
  --results-path /tmp/test_gcm_v2.json

echo ""
echo "=========================================="
echo "Guard: variant=v2 without a selection pass must raise, not silently run V1"
echo "=========================================="
python - <<'PY'
import numpy as np, gcm_baseline as gb
d = np.load('data/TMLR/8var_sergio_nodrop_test.npz')
r = gb.safe_fit_and_predict_task(
    np.array(d['input'][0], dtype=float), np.array(d['target_data'][0, 0], dtype=float),
    np.array(d['graphs'][0]), np.array(d['targets'][0, 0]),
    0.0, 'GOOD', 'v2', None, 30, 200, 5, 'kde', -20.0, 0)
assert 'failed' in r and 'fixed_prototype' in r['failed'], r
print('OK — raised:', r['failed'])
PY

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
