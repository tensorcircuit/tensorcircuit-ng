#!/bin/bash

#SBATCH --job-name=vqe_test   
#SBATCH --output=vqe_test_%j.out  
#SBATCH --nodes=2                       
#SBATCH --ntasks-per-node=1             
#SBATCH --gres=gpu:1                    # consistent with the number of cards per node
#SBATCH --cpus-per-task=8               
#SBATCH --partition=qdagnormal          


export MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)

export MASTER_PORT=29500

echo "Node list: $SLURM_JOB_NODELIST"
echo "Master Node: $MASTER_NODE"
echo "Master Address (IP): $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

export NCCL_DEBUG=INFO

echo "--- Launching JAX script on all nodes ---"
srun  python /abs/path/slurm_vqe_with_path.py