#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=20G

# examples of training commands
workingdir='/cluster/work/users/mdelhoneux/hit_parser'
cd $workingdir
module load Perl/5.30.0-GCCcore-8.3.0
module load Anaconda3/2019.03
conda activate hit_parser

lang=$1
iso=$2
checkpoint_dir=$3
datadir=${4:-/cluster/projects/nn9447k/mdelhoneux/train-dev}

rm -rf $checkpoint_dir
TRAIN_PATH=$datadir/$lang/$iso-ud-train.conllu \
DEV_PATH=$datadir/$lang/$iso-ud-dev.conllu \
WORD_DIM=768 \
LOWER_CASE=FALSE \
BATCH_SIZE=8 \
BERT_PATH=/cluster/projects/nn9447k/bert/multi_cased_L-12_H-768_A-12/ \
allennlp train \
-s $checkpoint_dir \
--include-package utils \
--include-package modules \
--file-friendly-logging \
config/transition_eud_gpu.jsonnet
#config/transition_eud.jsonnet

