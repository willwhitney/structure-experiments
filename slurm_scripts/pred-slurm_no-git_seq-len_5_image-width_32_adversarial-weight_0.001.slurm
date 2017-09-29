#!/bin/bash
#SBATCH --job-name=pred-slurm_no-git_seq-len_5_image-width_32_adversarial-weight_0.001
#SBATCH --output=slurm_logs/pred-slurm_no-git_seq-len_5_image-width_32_adversarial-weight_0.001.out
#SBATCH --error=slurm_logs/pred-slurm_no-git_seq-len_5_image-width_32_adversarial-weight_0.001.err
python adversarial_main.py --no-git --seq-len 5 --image-width 32 --adversarial-weight 0.001 --name pred-slurm_no-git_seq-len_5_image-width_32_adversarial-weight_0.001