#!/bin/bash
#SBATCH --job-name=eud
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1
##SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run longer than 30 min
#Note that a program will be killed once it exceeds this time! 
#SBATCH --time=0-05:00:00
#SBATCH --mem=50G
#Skipping many options! see man sbatch
# From here on, we can start our program

#hostname
#echo $CUDA_VISIBLE_DEVICES
#conda activate hit_scir
# examples of training commands
lang=$1
iso=$2
checkpoint_dir=$3

allennlp predict \
    --output-file $checkpoint_dir/dev_pred.conllu \
    --predictor transition_predictor_eud \
    --include-package utils \
    --include-package modules \
    --use-dataset-reader \
    --batch-size 32 \
    --silent \
    --cuda-device -1 \
    $checkpoint_dir \
    /image/nlp-datasets/iwpt20/train-dev/$lang/$iso-ud-dev.conllu \

