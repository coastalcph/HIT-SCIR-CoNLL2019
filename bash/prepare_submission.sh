#!/bin/bash
cd "$(dirname $0)/.." || exit
rm -rf collapsed validation text_without_spaces
mkdir -p {submission,collapsed,validation,text_without_spaces}/{stanza,udpipe,gold}/{dev,test} dev dev-gold test
if [ -n "$GET_PRED" ]; then
  scp saga:/cluster/projects/nn9447k/mdelhoneux/models/mbert/*/*-predicted-*.conllu submission/
  scp saga:/cluster/projects/nn9447k/mdelhoneux/models/mbert/*_all/*-predicted-*.conllu submission/
fi
if [ -n "$GET_TEXT" ]; then
  for div in test dev; do
    wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-$div-blind.tgz
    cd $div || exit
    tar xvzf ../iwpt2020-$div-blind.tgz
    cd ..
  done
  wget http://ufal.mff.cuni.cz/~zeman/soubory/iwpt2020-dev-gold.tgz
  cd dev-gold || exit
  tar xvzf ../iwpt2020-dev-gold.tgz
  cd ..
fi
cd submission/ || exit
for f in *.conllu; do
  lang=${f%%-*}
  lang=${lang%_*}
  basename=${f%.*}
  div=${basename##*-}
  basename=${basename%-*}
  preprocessor=${basename##*-}
  basename=${basename#*_}
  treebank_suffix=${basename%%-*}
  [ -n "$PREPROCESSOR" ] && [ $preprocessor != $PREPROCESSOR ] && continue
  [ -n "$DIV" ] && [ $div != $DIV ] && continue
  echo "=== $lang $treebank_suffix $preprocessor $div $f -> submission/$preprocessor/$div/$lang.conllu"
  if [ $treebank_suffix != all ] && [ -f ${lang}_all-predicted-$preprocessor-$div.conllu ]; then
    echo Skipped because a concat model exists
    continue
  fi
  if [ $lang != en ] && [ $lang != it ] && [ $lang != sv ]; then
    continue  # final run, TODO remove this
  fi
  # Workarounds for validation errors:
  # sed 's/\([	|]0:\)\w*/\1root/g;s/0:root|0:root/0:root/g' $f |
  # sed 's/\(0:root|\?\)\+/0:root/g' $f |
  perl ../tools/conllu-quick-fix.pl < $f > $preprocessor/$div/$lang.conllu
  if [ $preprocessor == udpipe ] && [ $div == test ] && [ $lang == sv ]; then
    # Workaround for weird udpipe issue resulting in space in lemma (L1 Format trailing-whitespace)
    sed -i 's/\(\tand\) /\1/' $preprocessor/$div/$lang.conllu
  fi

  python ../tools/validate.py $preprocessor/$div/$lang.conllu --lang $lang --level 2 > ../validation/$preprocessor/$div/$lang.txt 2>&1 &
  timeout 60s perl ../tools/enhanced_collapse_empty_nodes.pl $preprocessor/$div/$lang.conllu > ../collapsed/$preprocessor/$div/$lang.conllu 2>../collapsed/$preprocessor/$div/$lang.log || head -v -n5 ../collapsed/$preprocessor/$div/$lang.log
  echo -n Evaluating submission/$preprocessor/$div/$lang.conllu against itself " ... "
  python ../tools/iwpt20_xud_eval.py ../collapsed/$preprocessor/$div/$lang.conllu ../collapsed/$preprocessor/$div/$lang.conllu | grep ELAS
  if [ $div == dev ]; then
    perl ../tools/enhanced_collapse_empty_nodes.pl ../dev-gold/$lang.conllu > ../collapsed/gold/$div/$lang.conllu 2>../collapsed/gold/$div/$lang.log || head -v -n5 ../collapsed/gold/$div/$lang.log
    echo -n Evaluating submission/$preprocessor/$div/$lang.conllu against dev-gold/$lang.conllu " ... "
    python ../tools/iwpt20_xud_eval.py ../collapsed/gold/$div/$lang.conllu ../collapsed/$preprocessor/$div/$lang.conllu | grep ELAS
  fi
  perl ../tools/text_without_spaces.pl ../$div/$lang.txt > ../text_without_spaces/gold/$div/$lang.txt
  perl ../tools/conllu_to_text.pl $preprocessor/$div/$lang.conllu | ../tools/text_without_spaces.pl > ../text_without_spaces/$preprocessor/$div/$lang.txt
  if ! diff -q ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt; then
    echo "Text mismatch: $preprocessor/$div/$lang.conllu ($f)"
    wc -l ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head -n-1
    diff -dy ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head
  fi
done

wait < <(jobs -p)
tail -n1 ../validation/*/*/*.txt
