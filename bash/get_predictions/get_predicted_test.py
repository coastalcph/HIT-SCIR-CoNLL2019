import logging
import os
import sys

import stanza

test_dir = sys.argv[1]
test_files = os.listdir(test_dir)

done = []
for test_file in test_files:
    lang_code = test_file.split(".")[0]
    if lang_code == "lt":
        stanza.download(lang_code, package="alksnis")
        nlp = stanza.Pipeline(lang_code, package="alksnis", use_gpu=True)
    else:
        stanza.download(lang_code)
        nlp = stanza.Pipeline(lang_code, use_gpu=True)
        with open(f'{test_dir}/{test_file}', 'r') as infile:
            raw_text = infile.read().strip()
        doc = nlp(raw_text)
        with open(f'{test_dir}/{lang_code}-ud-preprocessed-stanza-test.conllu', 'w') as infile:
            for i, sentence in enumerate(doc.sentences):
                for word in sentence.words:
                    line = f'{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats}\t{word.head}\t{word.deprel}\t_\t_\n'
                    infile.write(line)
                infile.write("\n")
                if i % 1000 == 0:
                    print(f"Processed {i} sentences.")
