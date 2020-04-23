#!/bin/bash

test_dir=/cluster/projects/nn9447k/mdelhoneux/test
checkpoint_dir=/cluster/projects/nn9447k/mdelhoneux/models/mbert

models=(
    ar_padt  cs_cac      en_ewt  et_ewt      it_isdt     nl_all         pl_all  ru_syntagrus  ta_ttb
    bg_btb   cs_fictree  et_all  fi_tdt      lt_alksnis  nl_alpino      pl_lfg  sk_snk        uk_iu
    cs_all   cs_pdt      et_edt  fr_sequoia  lv_lvtb     nl_lassysmall  pl_pdb  sv_talbanken
)


for model in "${models[@]}"; do
    iso=$(echo $model | grep -o '^[^_]*')
    preprocessed_stanza=$(find $test_dir/preprocessed/ -name ${iso}-ud-preprocessed-stanza-test.conllu)
    preprocessed_udpipe=$(find $test_dir/preprocessed/ -name ${iso}-ud-preprocessed-udpipe-test.conllu)

    sbatch --job-name ${model}_stanza bash/predict_ud.sh \
           ${checkpoint_dir}/${model}/ \
           ${preprocessed_stanza} \
           ${iso}-predicted-stanza-test.conllu

    sbatch --job-name ${model}_udpipe bash/predict_ud.sh \
           ${checkpoint_dir}/${model}/ \
           ${preprocessed_udpipe} \
           ${iso}-predicted-udpipe-test.conllu

done
