#!/bin/bash

cd $(dirname $0)/../tools
tools=$(pwd)
cd /cluster/projects/nn9447k/mdelhoneux/models/mbert
rm invalid.txt
for preprocessor in stanza udpipe; do
  for dir in *; do
    lang=${dir%_*}
    echo ==== $lang ====
    for pred in $dir/*predicted-$preprocessor-test.conllu; do
      echo === $pred ===
      python $tools/validate.py $pred --lang $lang || echo $pred >> invalid.txt
    done
  done |& tee validation_$preprocessor.txt
done
