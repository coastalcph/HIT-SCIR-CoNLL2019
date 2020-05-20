#!/bin/bash

test_dir=/cluster/projects/nn9447k/mdelhoneux/dev
checkpoint_dir=/cluster/projects/nn9447k/mdelhoneux/models/mbert

models=(
    ar_padt  en_ewt      it_isdt     nl_all        pl_all  ru_syntagrus  ta_ttb
    bg_btb   et_all      fi_tdt      lt_alksnis  sk_snk        uk_iu
    cs_all   fr_sequoia  lv_lvtb     sv_talbanken
#    cs_cac   cs_fictree  cs_pdt      nl_alpino  nl_lassysmall  pl_lfg  pl_pdb  et_edt et_ewt
)

top_output_dir="${output_dir:-.}"

for model in "${models[@]}"; do
    iso=$(echo $model | grep -o '^[^_]*')
    [ -n "$lang" ] && [ "$iso" != "$lang" ] && continue
    preprocessed_stanza=$(find $test_dir/preprocessed/ -name ${iso}-ud-preprocessed-stanza-dev.conllu)
    preprocessed_udpipe=$(find $test_dir/preprocessed/ -name ${iso}-ud-preprocessed-udpipe-dev.conllu)
    gold_file=${test_dir}-gold/$iso.conllu
    if [ "${iso}" == et ]; then
      output_null_nodes=false  # no null nodes for Estonian since collapsing them takes forever
    else
      output_null_nodes=true
    fi

    if [ -z "$preprocessor" ] || [ "$preprocessor" == stanza ]; then
      export output_dir="${top_output_dir:-.}/dev/stanza"
      mkdir -p "${output_dir}"
      sbatch --job-name ${model}_stanza -o "$output_dir/$iso-%j.out" bash/predict_ud.sh \
             ${checkpoint_dir}/${model}/ \
             ${preprocessed_stanza} \
             ${iso}.conllu \
             ${gold_file} \
             ${output_null_nodes}
    fi

    if [ -z "$preprocessor" ] || [ "$preprocessor" == udpipe ]; then
      export output_dir="${top_output_dir:-.}/dev/udpipe"
      mkdir -p "${output_dir}"
      sbatch --job-name ${model}_udpipe -o "$output_dir/$iso-%j.out" bash/predict_ud.sh \
             ${checkpoint_dir}/${model}/ \
             ${preprocessed_udpipe} \
             ${iso}.conllu \
             ${gold_file} \
             ${output_null_nodes}
    fi

    if [ -z "$preprocessor" ] || [ "$preprocessor" == gold ]; then
      export output_dir="${top_output_dir:-.}/dev/gold"
      mkdir -p "${output_dir}"
      sbatch --job-name ${model}_gold -o "$output_dir/$iso-%j.out" bash/predict_ud.sh \
             ${checkpoint_dir}/${model}/ \
             ${gold_file} \
             ${iso}.conllu \
             ${gold_file} \
             ${output_null_nodes}
    fi

done
