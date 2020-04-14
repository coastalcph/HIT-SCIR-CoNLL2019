#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=20G

source ~/.bashrc
conda activate eud_37

python get_predicted_dev.py ~/data/train-dev/
