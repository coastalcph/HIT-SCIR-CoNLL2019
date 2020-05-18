lang=$1
iso=$2
checkpoint_dir=checkpoints_debug
n_sen_train=$( grep -c sent_id ../data/$lang/$iso-ud-train.conllu )
n_sen_train=$(( $n_sen_train + 0 ))
if [ $n_sen_train -lt 5000 ]
then
	ntrain=$n_sen_train
else
	ntrain=5000
fi
ndev=$(grep -c sent_id ../data/$lang/$iso-ud-dev.conllu )
ndev=$(( $ndev + 0 ))

rm -rf $checkpoint_dir
TRAIN_PATH=../data/$lang/$iso-ud-train.conllu \
DEV_PATH=../data/$lang/$iso-ud-dev.conllu \
WORD_DIM=768 \
LOWER_CASE=FALSE \
BATCH_SIZE=1 \
BERT_PATH=/cluster/projects/nn9447k/bert/multi_cased_L-12_H-768_A-12/ \
INSTANCES_PER_EPOCH_TRAIN=$ntrain \
INSTANCES_PER_EPOCH_DEV=$ndev \
allennlp train \
-s $checkpoint_dir \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_eud_debug.jsonnet

