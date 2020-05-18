#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=20G

source /cluster/home/mdelhoneux/.bashrc
workingdir='/cluster/work/users/mdelhoneux/hit_parser'
cd $workingdir
module load Perl/5.30.0-GCCcore-8.3.0
#module load Anaconda3/2019.03
conda activate hit_parser

lang=$1
iso=$2
checkpoint_dir=$3
datadir=${4:-/cluster/projects/nn9447k/mdelhoneux/train-dev}
train_path=$datadir/$lang/$iso-ud-train.conllu 
dev_path=$datadir/$lang/$iso-ud-dev.conllu 

#get number of instances
#sorry hacky will clean up
n_sen_train=$( grep -c sent_id $train_path )
n_sen_train=$(( $n_sen_train + 0 ))
ndev=$(grep -c sent_id $dev_path )
ndev=$(( $ndev + 0 ))
if [ $n_sen_train -lt 5000 ]
then
	ntrain=$n_sen_train
else
	ntrain=5000
fi

rm -rf $checkpoint_dir
TRAIN_PATH=$train_path \
DEV_PATH=$dev_path \
WORD_DIM=768 \
LOWER_CASE=FALSE \
BATCH_SIZE=8 \
INSTANCES_PER_EPOCH_TRAIN=$ntrain \
INSTANCES_PER_EPOCH_DEV=$ndev \
BERT_PATH=/cluster/projects/nn9447k/bert/multi_cased_L-12_H-768_A-12/ \
allennlp train \
-s $checkpoint_dir \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_eud_gpu.jsonnet
#config/transition_eud.jsonnet

