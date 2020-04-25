#!/bin/bash
cd "$(dirname $0)/.." || exit
cd submission/ || exit
for lang in `ls ../dev|sed -n '/\.txt/{s/\.txt//;p}'`; do
  printf "$lang\t"
  for preprocessor in stanza udpipe; do
    python ../tools/iwpt20_xud_eval.py ../collapsed/gold/dev/$lang.conllu ../collapsed/$preprocessor/dev/$lang.conllu 2>/dev/null | grep -zPo "(?<=ELAS F1 Score: ).*"
    printf "\t"
  done
  echo
done
