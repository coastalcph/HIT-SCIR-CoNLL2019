#!/bin/bash
cd "$(dirname $0)/.." || exit
rm -rf collapsed validation text_without_spaces
mkdir -p {fixed_submission,submission,collapsed,validation,text_without_spaces}/{stanza,udpipe,gold}/{dev,test} dev dev-gold test
for div in dev test; do
  for preprocessor in udpipe stanza; do
    for f in fixed/$div/$preprocessor; do
      lang=${f%%.*}
      # Workarounds for validation errors:
      # sed 's/\([	|]0:\)\w*/\1root/g;s/0:root|0:root/0:root/g' $f |
      # sed 's/\(0:root|\?\)\+/0:root/g' $f |
      perl tools/conllu-quick-fix.pl < $f > fixed_submission/$preprocessor/$div/$lang.conllu
      if [ $preprocessor == udpipe ] && [ $div == test ] && [ $lang == sv ]; then
        # Workaround for weird udpipe issue resulting in space in lemma (L1 Format trailing-whitespace)
        sed -i 's/\(\tand\) \(\t\)/\1\2/' fixed_submission/$preprocessor/$div/$lang.conllu
      fi

      python tools/validate.py fixed_submission/$preprocessor/$div/$lang.conllu --lang $lang --level 2 > validation/$preprocessor/$div/$lang.txt 2>&1 &
      timeout 60s perl tools/enhanced_collapse_empty_nodes.pl fixed_submission/$preprocessor/$div/$lang.conllu > collapsed/$preprocessor/$div/$lang.conllu 2>collapsed/$preprocessor/$div/$lang.log || head -v -n5 collapsed/$preprocessor/$div/$lang.log
      echo -n Evaluating fixed_submission/$preprocessor/$div/$lang.conllu against itself " ... "
      python tools/iwpt20_xud_eval.py collapsed/$preprocessor/$div/$lang.conllu collapsed/$preprocessor/$div/$lang.conllu | grep ELAS
      if [ $div == dev ]; then
        perl tools/enhanced_collapse_empty_nodes.pl dev-gold/$lang.conllu > collapsed/gold/$div/$lang.conllu 2>collapsed/gold/$div/$lang.log || head -v -n5 collapsed/gold/$div/$lang.log
        echo -n Evaluating fixed_submission/$preprocessor/$div/$lang.conllu against dev-gold/$lang.conllu " ... "
        python tools/iwpt20_xud_eval.py collapsed/gold/$div/$lang.conllu collapsed/$preprocessor/$div/$lang.conllu | grep ELAS
      fi
      perl tools/text_without_spaces.pl $div/$lang.txt > text_without_spaces/gold/$div/$lang.txt
      perl tools/conllu_to_text.pl fixed_submission/$preprocessor/$div/$lang.conllu | tools/text_without_spaces.pl > text_without_spaces/$preprocessor/$div/$lang.txt
      if ! diff -q text_without_spaces/{gold,$preprocessor}/$div/$lang.txt; then
        echo "Text mismatch: $preprocessor/$div/$lang.conllu ($f)"
        wc -l text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head -n-1
        diff -dy text_without_spaces/{gold,$preprocessor}/$div/$lang.txt | head
      fi
    done
  done
done

wait < <(jobs -p)
tail -n1 validation/*/*/*.txt
