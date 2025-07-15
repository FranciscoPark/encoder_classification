#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --job-name=eval
#SBATCH -o logs/eval/%x-%j.out

source ~/.bashrc
source /home/s1/jypark/anaconda3/bin/activate
conda activate snuenv

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8952

# model
# export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=16

python /home/s1/jypark/encoder_classification/eval.py
