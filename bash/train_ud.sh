#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=2-00:00:00

#conda activate hit_scir
# examples of training commands
lang=$1
iso=$2
checkpoint_dir=$3
datadir=${4:-/image/nlp-datasets/iwpt20/train-dev/}

rm -rf $checkpoint_dir
TRAIN_PATH=$datadir/$lang/$iso-ud-train.conllu \
DEV_PATH=$datadir/$lang/$iso-ud-dev.conllu \
WORD_DIM=768 \
LOWER_CASE=FALSE \
BATCH_SIZE=8 \
BERT_PATH=/image/nlp-letre/bert/multi_cased_L-12_H-768_A-12/ \
allennlp train \
-s $checkpoint_dir \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_eud_gpu.jsonnet
#config/transition_eud.jsonnet

