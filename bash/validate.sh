#!/bin/bash

cd $(dirname $0)/../tools
tools=$(pwd)
cd /cluster/projects/nn9447k/mdelhoneux/models/mbert
for div in dev test; do
  for preprocessor in stanza udpipe; do
    for dir in *; do
      lang=${dir%_*}
      for pred in $dir/*predicted-$preprocessor-$div; do
        echo $pred ...
        python $tools/validate.py $pred --lang $lang --level 2 > $pred.validation.txt 2>&1 &
      done
    done
  done
  wait < <(jobs -p)
  tail -n1 /cluster/projects/nn9447k/mdelhoneux/models/mbert/*/*predicted-*-test.conllu.validation.txt
done
