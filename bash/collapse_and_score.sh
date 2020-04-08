#!/bin/bash

treebank_dir=/cluster/projects/nn9447k/mdelhoneux/train-dev
treebanks=$(ls $treebank_dir)
checkpoints=/cluster/projects/nn9447k/mdelhoneux/models/mbert

for tb in $treebanks; do
    gold=$(find $treebank_dir/${tb}/ -name *-ud-dev.conllu -exec basename {} \;)
    tb_code=$(echo $gold | sed 's/-.*//')
    pred=$(find ${checkpoints}/${tb_code}/ -name *preprocessed-stanza* -exec basename {} \;)
    #pred=$(find ${checkpoints}/${tb_code}/ -name *preprocessed-udpipe* -exec basename {} \;)
    tmp_gold=/tmp/gold.conllu
    tmp_pred=/tmp/pred.conllu
    perl tools/enhanced_collapse_empty_nodes.pl ${treebank_dir}/${tb}/${gold} > $tmp_gold
    perl tools/enhanced_collapse_empty_nodes.pl ${checkpoints}/${tb_code}/${pred} > $tmp_pred
    echo $tb
    #python metrics/iwpt20_xud_eval.py $tmp_gold $tmp_pred > ${checkpoints}/${tb_code}/dev_score_preprocessed-udpipe.txt
    python metrics/iwpt20_xud_eval.py $tmp_gold $tmp_pred > ${checkpoints}/${tb_code}/dev_score_preprocessed-stanza.txt
done
