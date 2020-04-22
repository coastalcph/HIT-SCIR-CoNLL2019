#!/bin/bash
cd $(dirname $0)/..
mkdir -p {submission,collapsed,validation,text_without_spaces}/{stanza,udpipe,gold}/{dev,test} dev dev-gold test
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
  wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-dev-gold.tgz
  cd dev-gold
  tar xvzf ../iwpt2020-dev-gold.tgz
  cd ..
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
  echo === $lang $treebank_suffix $preprocessor $div $f
  if [ $treebank_suffix != all -a -f ${lang}_all-predicted-$preprocessor-$div.conllu ]; then
    echo Skipped because a concat model exists
    continue
  fi
  # Workarounds for validation errors:
  sed -i 's/\([	|]0:\)\w*/\1root/g;s/0:root|0:root/0:root/g' $f

  python ../tools/validate.py $f --lang $lang --level 2 > ../validation/$preprocessor/$div/$lang.txt 2>&1 &
  timeout 10s "perl ../tools/enhanced_collapse_empty_nodes.pl $f" > ../collapsed/$preprocessor/$div/$lang.conllu 2>/dev/null
  python ../tools/iwpt20_xud_eval.py ../collapsed/$preprocessor/$div/$lang.conllu ../collapsed/$preprocessor/$div/$lang.conllu
  if [ $div == dev ]; then
    perl ../tools/enhanced_collapse_empty_nodes.pl ../dev-gold/$lang.conllu > ../collapsed/gold/$div/$lang.conllu 2>/dev/null
    python ../tools/iwpt20_xud_eval.py ../collapsed/gold/$div/$lang.conllu ../collapsed/$preprocessor/$div/$lang.conllu
  fi
  perl ../tools/text_without_spaces.pl ../$div/$lang.txt > ../text_without_spaces/gold/$div/$lang.txt
  perl ../tools/conllu_to_text.pl $f | ../tools/text_without_spaces.pl > ../text_without_spaces/$preprocessor/$div/$lang.txt
  if ! diff -q ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt; then
    echo Text mismatch: $f
    wc -l ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head -n-1
    diff -dy ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head
  fi
  cp -v $f $preprocessor/$div/$lang.conllu
done

wait < <(jobs -p)
tail -n1 ../validation/*/*/*.txt
