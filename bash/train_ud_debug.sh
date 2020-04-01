lang=$1
iso=$2
checkpoint_dir=checkpoints_debug

rm -rf $checkpoint_dir
TRAIN_PATH=../data/$lang/$iso-ud-train.conllu \
DEV_PATH=../data/$lang/$iso-ud-dev.conllu \
WORD_DIM=768 \
LOWER_CASE=FALSE \
BATCH_SIZE=4 \
BERT_PATH=/cluster/projects/nn9447k/bert/multi_cased_L-12_H-768_A-12/ \
allennlp train \
-s $checkpoint_dir \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_eud_debug.jsonnet

