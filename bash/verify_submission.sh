#!/bin/bash
cd submission/
for f in *.conllu; do
  lang=${f%_*}
  basename=${f%.*}
  div=${basename##*-}
  basename=${basename%-*}
  preprocessor=${basename##*-}
  basename=${basename#*_}
  treebank_suffix=${basename%%-*}
  [ $treebank_suffix != all -a -f ${lang}_all-predicted-$preprocessor-$div.conllu ] && continue
  printf "%-50s" $f
  diff -q ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt && echo OK
done

