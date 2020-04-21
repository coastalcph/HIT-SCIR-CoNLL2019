#!/bin/bash
cd $(dirname $0)/..
mkdir submission collapsed validation
scp saga:/cluster/projects/nn9447k/mdelhoneux/models/mbert/*/*-predicted-*.conllu submission/
cd submission/
mkdir {test,dev}/{stanza,udpipe}
for f in *.conllu; do
  lang=${f%_*}
  basename=${f%.*}
  div=${basename#*-}
  basename=${basename%-*}
  preprocessor=${basename#*-}
  basename=${basename%*_}
  treebank_suffix=${basename%-*}
  echo $lang $treebank_suffix $preprocessor $div $f
  python ../tools/validate.py $f --lang $lang --level 2 > ../validation/$f.txt
  perl ../tools/enhanced_collapse_empty_nodes.pl $f > ../collapsed/$f
  python ../tools/iwpt20_xud_eval.py ../collapsed/$f ../collapsed/$f
  perl ../tools/text_without_spaces.pl dev/ar.txt 
  perl ../tools/conllu_to_text.pl $f
done

