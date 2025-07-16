#!/bin/bash

#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --job-name=finetune
#SBATCH -o logs/pjm/finetune/%x-%j.out

source ~/.bashrc
source /home/s1/jypark/anaconda3/bin/activate
conda activate snuenv

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8952

# model
# export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=16

srun --jobid $SLURM_JOBID bash -c 'torchrun \
 --nproc-per-node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
train2.py'
