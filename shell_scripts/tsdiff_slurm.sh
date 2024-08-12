#!/bin/bash

# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o slurm_output/JOB%j.out # File to which STDOUT will be written
#SBATCH -e slurm_output/JOB%j.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL

# Load up your conda environment
# Set up environment on watgpu.cs or in interactive session (use `source` keyword instead of `conda`)
source activate /u1/$USER/.conda/envs/tsdiff2/
source shell_scripts/env_vars.sh


python bin/clear_pykeops_cache.py

# Train models over different datasets
for ((i=0; i<=$DATASET_NUM; i++))
do
    python bin/train_model.py \
    -c configs/train_tsdiff/train_gbm.yaml \
    --dataset_path data/debug_gbm/gbm-$i.jsonl \
    --out_dir ./results/$EXP_NAME && \
    python bin/generation_experiment.py \
    -c configs/generation/gen_gbm.yaml \
    --dataset gbm-$i \
    --dataset_path data/debug_gbm/gbm-$i.jsonl \
    --ckpt ./results/$EXP_NAME/lightning_logs/version_0/best_checkpoint.ckpt \
    --out_dir ./results/$EXP_NAME
done

