#!/bin/bash
#SBATCH --job-name=eud
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=1
#We expect that our program should not run longer than 30 min
#Note that a program will be killed once it exceeds this time! 
#SBATCH --time=0-05:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program

# examples of training commands
lang=$1
iso=$2
checkpoint_dir=checkpoints/test

rm -rf $checkpoint_dir
TRAIN_PATH=/image/nlp-datasets/iwpt20/train-dev/$lang/$iso-ud-train.conllu \
DEV_PATH=/image/nlp-datasets/iwpt20/train-dev/$lang/$iso-ud-dev.conllu \
WORD_DIM=1024 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
BERT_PATH=/image/nlp-letre/bert/wwm_cased_L-24_H-1024_A-16/ \
allennlp train \
-s $checkpoint_dir \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_eud.jsonnet

