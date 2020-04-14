#!/bin/bash

# treebanks=(Arabic-PADT Bulgarian-BTB Czech-CAC Czech-FicTree Czech-PDT
#            Dutch-Alpino Dutch-LassySmall English-EWT Estonian-EDT Estonian-EWT
#            Finnish-TDT French-Sequoia Italian-ISDT Latvian-LVTB
#            Lithuanian-ALKSNIS Polish-LFG Polish-PDB Russian-SynTagRus Slovak-SNK
#            Swedish-Talbanken Tamil-TTB Ukrainian-IU)

treebanks=(Czech-FicTree)
treebank_dir=/cluster/projects/nn9447k/mdelhoneux/train-dev
checkpoints=/cluster/projects/nn9447k/mdelhoneux/models/mbert

touch out.txt
for tb in "${treebanks[@]}"; do
    gold=$(find $treebank_dir/UD_${tb}/ -name *-ud-dev.conllu -exec basename {} \;)
    tb_code=$(echo $gold | sed 's/-.*//')
    echo -e $tb
    pred=$(find ${checkpoints}/${tb_code}/ -name *preprocessed-stanza-dev.conllu -exec basename {} \;)
    tmp_gold=/tmp/${tb_code}_gold.conllu
    tmp_pred=/tmp/${tb_code}_pred.conllu
    perl tools/enhanced_collapse_empty_nodes.pl ${treebank_dir}/UD_${tb}/${gold} > $tmp_gold
    perl tools/enhanced_collapse_empty_nodes.pl ${checkpoints}/${tb_code}/${pred} > $tmp_pred
    python metrics/iwpt20_xud_eval.py $tmp_gold $tmp_pred
done
