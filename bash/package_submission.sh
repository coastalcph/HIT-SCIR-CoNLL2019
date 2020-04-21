#!/bin/bash
for preprocessor in stanza udpipe; do
  for div in dev test; do
    cd submission/$preprocessor/$div
    tar cvzf ../../../koebsala-$preprocessor_$div_`date '+%Y%m%d_%H%M%S'`.tgz *.conllu
    cd -
  done
done
