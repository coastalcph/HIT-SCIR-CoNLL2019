#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G

source ~/.bashrc
conda activate eud_37

python get_preprocessed_test_stanza.py /cluster/projects/nn9447k/mdelhoneux/test/
