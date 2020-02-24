#!/bin/bash

# examples of training commands
lang=$1
iso=$2
checkpoint_dir=checkpoints/test

rm -rf $checkpoint_dir
# debug EUD
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

