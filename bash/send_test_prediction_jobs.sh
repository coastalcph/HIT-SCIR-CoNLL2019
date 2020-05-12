#!/bin/bash

test_dir=/cluster/projects/nn9447k/mdelhoneux/test
checkpoint_dir=/cluster/projects/nn9447k/mdelhoneux/models/mbert

models=(
    ar_padt  en_ewt      et_ewt      it_isdt     nl_all        pl_all  ru_syntagrus  ta_ttb
    bg_btb   et_all      fi_tdt      lt_alksnis  sk_snk        uk_iu
    cs_all   fr_sequoia  lv_lvtb     sv_talbanken
#    cs_cac   cs_fictree  cs_pdt      nl_alpino  nl_lassysmall  pl_lfg  pl_pdb  et_edt
)


for model in "${models[@]}"; do
    iso=$(echo $model | grep -o '^[^_]*')
    preprocessed_stanza=$(find $test_dir/preprocessed/ -name ${iso}-ud-preprocessed-stanza-test.conllu)
    preprocessed_udpipe=$(find $test_dir/preprocessed/ -name ${iso}-ud-preprocessed-udpipe-test.conllu)
    gold_file=${test_dir}-gold/$iso.conllu

    sbatch --job-name ${model}_stanza bash/predict_ud.sh \
           ${checkpoint_dir}/${model}/ \
           ${preprocessed_stanza} \
           ${iso}-predicted-stanza-test.conllu \
           ${gold_file}

    sbatch --job-name ${model}_udpipe bash/predict_ud.sh \
           ${checkpoint_dir}/${model}/ \
           ${preprocessed_udpipe} \
           ${iso}-predicted-udpipe-test.conllu \
           ${gold_file}

done
