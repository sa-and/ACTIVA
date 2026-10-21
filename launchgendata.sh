#!/bin/sh
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -C cpunode 


source /home/asr448/.bashrc
conda activate amort

SAVE_DIR="data/TMLR/8var_linadd_normal"
SAMPLES=80
N_ENDO=8
N_GRAPHS=2000
N_SCMS_PER_GRAPH=10
DATA_MODE="interventional"
DGF="f_linear_noise_normal"
TRAIN_FRACTION=0.8
VAL_FRACTION=0.1

python3 generate_data.py \
    --save-dir "$SAVE_DIR" \
    --samples "$SAMPLES" \
    --n-endo "$N_ENDO" \
    --n-graphs "$N_GRAPHS" \
    --n-scms-per-graph "$N_SCMS_PER_GRAPH" \
    --data-mode "$DATA_MODE" \
    --data-generating-function "$DGF" \
    --train-fraction "$TRAIN_FRACTION" \
    --val-fraction "$VAL_FRACTION"
