#!/bin/bash

treebank_dir=/cluster/projects/nn9447k/mdelhoneux/train-dev
treebanks=(Arabic-PADT Bulgarian-BTB Czech-CAC Czech-FicTree Czech-PDT
           Dutch-Alpino Dutch-LassySmall English-EWT Estonian-EDT Estonian-EWT
           Finnish-TDT French-Sequoia Italian-ISDT Latvian-LVTB
           Lithuanian-ALKSNIS Polish-LFG Polish-PDB Russian-SynTagRus Slovak-SNK
           Swedish-Talbanken Tamil-TTB Ukrainian-IU)
checkpoints=/cluster/projects/nn9447k/mdelhoneux/models/mbert

for tb in "${treebanks[@]}"; do
    preprocessed=$(find $treebank_dir/UD_${tb}/ -name *preprocessed-udpipe*)
    tb_code=$(find $treebank_dir/UD_${tb}/ -name *preprocessed-udpipe* -exec basename {} \; | sed 's/-.*//')
    sbatch --job-name ${tb_code} bash/predict_ud.sh \
           ${checkpoints}/${tb_code}/ \
           ${preprocessed} \
           ${tb_code}-predicted-udpipe-dev.conllu
done
