import logging
import os
import sys

import stanza

treebank_dir = sys.argv[1]
treebanks = os.listdir(treebank_dir)

done = []
for treebank in treebanks:
    raw_dev = [f for f in os.listdir(f"{treebank_dir}/{treebank}") if f.endswith("dev.txt")]
    if raw_dev:
        lang_code = raw_dev[0].split("_")[0]
        treebank_code = raw_dev[0].split("-")[0]
        if lang_code == "lt":
            if lang_code not in done:
                stanza.download(lang_code, package="alksnis")
            nlp = stanza.Pipeline(lang_code, package="alksnis", use_gpu=True)
            with open(f'{treebank_dir}/{treebank}/{raw_dev[0]}', 'r') as infile:
                raw_text = infile.read().strip()
            doc = nlp(raw_text)
            with open(f'{treebank_dir}/{treebank}/{treebank_code}-ud-pred-dev.conllu', 'w') as infile:
                for i, sentence in enumerate(doc.sentences):
                    for word in sentence.words:
                        line = f'{word.id}\t{word.text}\t{word.lemma}\t{word.upos}\t{word.xpos}\t{word.feats}\t{word.head}\t{word.deprel}\t_\t_\n'
                        infile.write(line)
                    infile.write("\n")
                    if i % 1000 == 0:
                        print(f"Processed {i} sentences.")
