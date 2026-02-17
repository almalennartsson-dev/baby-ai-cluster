#!/bin/bash
#SBATCH -A berzelius-2025-224
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH -J train_unet
#SBATCH --output=/proj/synthetic_alzheimer/users/x_almle/bobsrepository/logs/%j.out
#SBATCH --error=/proj/synthetic_alzheimer/users/x_almle/bobsrepository/logs/%j.err

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate /proj/synthetic_alzheimer/users/x_almle/.venvs/mri-sr-bob

python training_aug_new_less.py