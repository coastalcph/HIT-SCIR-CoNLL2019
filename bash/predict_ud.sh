#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=10G

module load Perl/5.30.0-GCCcore-8.3.0
checkpoint_dir=$1
preprocessed_file=$2
output_file=$3

allennlp predict \
    --output-file $checkpoint_dir/$output_file \
    --predictor transition_predictor_eud \
    --include-package utils \
    --include-package modules \
    --use-dataset-reader \
    --batch-size 32 \
    --silent \
    --cuda-device 0 \
    --override '{"model": {"output_null_nodes": true}}' \
    $checkpoint_dir \
    $preprocessed_file \
