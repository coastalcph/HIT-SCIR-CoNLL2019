#!/bin/bash

treebank_dir=/cluster/projects/nn9447k/mdelhoneux/train-dev
treebanks=$(ls $treebank_dir)
checkpoints=/cluster/projects/nn9447k/mdelhoneux/models/mbert

for tb in $treebanks; do
    preprocessed=$(find $treebank_dir/${tb}/ -name *preprocessed-stanza*)
    tb_code=$(find $treebank_dir/${tb}/ -name *preprocessed-stanza* -exec basename {} \; | sed 's/-.*//')
    sbatch --job-name ${tb_code} bash/predict_ud.sh \
           ${checkpoints}/${tb_code}/ \
           ${preprocessed} \
           ${tb_code}-preprocessed-stanza-dev.conllu
done


