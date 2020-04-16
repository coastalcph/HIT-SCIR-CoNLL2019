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
        tb_code = raw_dev[0].split("-")[0]
        iso, tb = tuple(tb_code.split("_"))
        if iso == "lt":
            stanza.download(iso, package="alksnis")
            nlp = stanza.Pipeline(iso, package="alksnis", use_gpu=True)
        else:
            stanza.download(iso)
            nlp = stanza.Pipeline(iso, use_gpu=True)
        with open(f'{treebank_dir}/{treebank}/{raw_dev[0]}', 'r') as infile:
            raw_text = infile.read().strip()
        doc = nlp(raw_text)
        with open(f'{treebank_dir}/{treebank}/{tb_code}-ud-preprocessed-stanza-dev.conllu', 'w') as infile:
            for i, sentence in enumerate(doc.sentences):
                for token in sentence.tokens:
                    for word in token.to_dict():
                        line = f'{word["id"]}'
                        for key in ['text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']:
                            try:
                                line = line + f'\t{word[key]}'
                            except KeyError:
                                line = line + f'\t_'
                        line = line + "\t_\t_\n"
                        infile.write(line)
                infile.write("\n")
                if (i > 0) and (i % 1000) == 0:
                    print(f"Processed {i} sentences.")
