#!/bin/bash
#set -x
cd "$(dirname $0)/.." || exit
mkdir -p collapsed/{stanza,udpipe,gold}/{dev,test}
cd submission/ || exit
for div in dev test; do
#  [ $div == test ] || continue
  echo "$div"
  for lang in `ls ../$div|sed -n '/\.txt/{s/\.txt//;p}'`; do
#    [ $lang == et ] || continue
    printf "$lang\t"
    [ -f ../collapsed/gold/$div/$lang.conllu ] || perl ../tools/enhanced_collapse_empty_nodes.pl ../$div-gold/$lang.conllu > ../collapsed/gold/$div/$lang.conllu 2>../collapsed/gold/$div/$lang.log || head -v -n5 ../collapsed/gold/$div/$lang.log
    for preprocessor in stanza udpipe; do
#      [ $preprocessor == stanza ] || continue
      [ -f ../collapsed/$preprocessor/$div/$lang.conllu ] || perl ../tools/enhanced_collapse_empty_nodes.pl $preprocessor/$div/$lang.conllu > ../collapsed/$preprocessor/$div/$lang.conllu 2>../collapsed/$preprocessor/$div/$lang.log || head -v -n5 ../collapsed/$preprocessor/$div/$lang.log
      python ../tools/iwpt20_xud_eval.py ../collapsed/gold/$div/$lang.conllu ../collapsed/$preprocessor/$div/$lang.conllu 2>/dev/null | grep -zPo "(?<=ELAS F1 Score: ).*"
      printf "\t"
    done
    echo
  done
done