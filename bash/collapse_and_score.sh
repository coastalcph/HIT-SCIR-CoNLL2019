#!/bin/bash

treebanks=(Arabic-PADT Bulgarian-BTB Czech-CAC Czech-FicTree Czech-PDT
           Dutch-Alpino Dutch-LassySmall English-EWT Estonian-EDT Estonian-EWT
           Finnish-TDT French-Sequoia Italian-ISDT Latvian-LVTB
           Lithuanian-ALKSNIS Polish-LFG Polish-PDB Russian-SynTagRus Slovak-SNK
           Swedish-Talbanken Tamil-TTB Ukrainian-IU)
treebank_dir=/cluster/projects/nn9447k/mdelhoneux/train-dev
checkpoints=/cluster/projects/nn9447k/mdelhoneux/models/mbert

touch out.txt
for tb in "${treebanks[@]}"; do
    gold=$(find $treebank_dir/UD_${tb}/ -name *-ud-dev.conllu -exec basename {} \;)
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
