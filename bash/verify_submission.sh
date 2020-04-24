#!/bin/bash
cd submission/
for f in *.conllu; do
  lang=${f%%-*}
  lang=${lang%_*}
  basename=${f%.*}
  div=${basename##*-}
  basename=${basename%-*}
  preprocessor=${basename##*-}
  basename=${basename#*_}
  treebank_suffix=${basename%%-*}
  [ $treebank_suffix != all -a -f ${lang}_all-predicted-$preprocessor-$div.conllu ] && continue
  printf "%-50s" $f
  perl ../tools/conllu_to_text.pl $f | ../tools/text_without_spaces.pl > ../text_without_spaces/$preprocessor/$div/$lang.txt
  diff -q ../text_without_spaces/{gold,$preprocessor}/$div/$lang.txt && echo OK
done

echo Unique errors:
sed -n '/^\[/{s/\[[^[]*][^[]*\[//;s/\].*//;p}' ../validation/*/*/* | sort | uniq -c
