#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C A6000
##SBATCH -p fatq
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:4
# Load GPU drivers
module load cuda12.3/toolkit/12.3
module load cuDNN/cuda12.3/9.4.0.58
# module load cuDNN/cuda10.1
# This loads the anaconda virtual environment with our packages
source /home/asr448/.bashrc
# Set common JAX environment variables
#export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
#export JAX_PLATFORM_NAME="gpu"

# Important: Set these environment variables explicitly
#export SLURM_PROCID=$SLURM_NODEID
#export JAX_PROCESS_COUNT=$SLURM_NTASKS

# Get the coordinator node's hostname and IP
#COORDINATOR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#COORDINATOR_IP=$(srun --nodes=1 --ntasks=1 -w $COORDINATOR hostname -i | tail -n 1)
#PORT=1234

#echo "Coordinator node: $COORDINATOR"
#echo "Coordinator IP: $COORDINATOR_IP"

# Use the IP address instead of hostname for the coordinator
#export JAX_COORDINATOR_ADDRESS="${COORDINATOR_IP}:${PORT}"
#echo "JAX Coordinator address: ${JAX_COORDINATOR_ADDRESS}"
#echo "Process ID: $SLURM_PROCID of $JAX_PROCESS_COUNT"
conda activate amort
srun --export=ALL  python evaluate_model.py

