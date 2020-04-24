#!/bin/bash
cd "$(dirname $0)/.." || exit
mkdir -p preprocessed text_without_spaces/preprocessed-{udpipe,stanza}/{dev,test}
if [ -n "$GET_PRED" ]; then
  scp -r saga:/cluster/projects/nn9447k/mdelhoneux/*/preprocessed/* preprocessed/
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
cd preprocessed/ || exit
for f in *.conllu; do
  lang=${f%%-*}
  lang=${lang%_*}
  basename=${f%.*}
  div=${basename##*-}
  basename=${basename%-*}
  preprocessor=${basename##*-}
  basename=${basename#*_}
  treebank_suffix=${basename%%-*}
  [ $treebank_suffix != all -a -f ${lang}_all-preprocessed-$preprocessor-$div.conllu ] && continue
  printf "%-50s" $f
  perl ../tools/conllu_to_text.pl $f | ../tools/text_without_spaces.pl > ../text_without_spaces/preprocessed-$preprocessor/$div/$lang.txt
  diff -q ../text_without_spaces/{gold,preprocessed-$preprocessor}/$div/$lang.txt && echo OK
done

