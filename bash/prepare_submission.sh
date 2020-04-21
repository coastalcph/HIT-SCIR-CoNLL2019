#!/bin/bash
cd $(dirname $0)/..
mkdir -p {submission,collapsed,validation,text_without_spaces}/{stanza,udpipe,gold}/{dev,test} dev test
if [ -n "$GET_PRED" ]; then
  scp saga:/cluster/projects/nn9447k/mdelhoneux/models/mbert/*/*-predicted-*.conllu submission/
fi
if [ -n "$GET_TEXT" ]; then
  for div in test dev; do
    wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-$div-blind.tgz
    cd $div
    tar xvzf ../iwpt2020-$div-blind.tgz
    cd ..
  done
fi
cd submission/
for f in *.conllu; do
  lang=${f%_*}
  basename=${f%.*}
  div=${basename##*-}
  basename=${basename%-*}
  preprocessor=${basename##*-}
  basename=${basename#*_}
  treebank_suffix=${basename%%-*}
  echo $lang $treebank_suffix $preprocessor $div $f
  if [ $treebank_suffix != all -a -f ${lang}_all-predicted-$preprocessor-$div.conllu ]; then
    echo Skipped because a concat model exists
    continue
  fi
  python ../tools/validate.py $f --lang $lang --level 2 > ../validation/$preprocessor/$div/$lang.txt 2>&1 &
  perl ../tools/enhanced_collapse_empty_nodes.pl $f > ../collapsed/$preprocessor/$div/$lang.conllu
  python ../tools/iwpt20_xud_eval.py ../collapsed/$preprocessor/$div/$lang.conllu ../collapsed/$preprocessor/$div/$lang.conllu
  perl ../tools/text_without_spaces.pl ../$div/$lang.txt > ../text_without_spaces/gold/$div/$lang.txt
  perl ../tools/conllu_to_text.pl $f | ../tools/text_without_spaces.pl > ../text_without_spaces/$preprocessor/$div/$lang.txt
  if ! diff -q ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt; then
    echo Text mismatch: $f
    wc -l ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head -n-1
    diff --color=always -dy ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head
  fi
  cp -v $f $preprocessor/$div/$lang.txt
done

wait < <(jobs -p)
tail -n1 ../validation/*
