TODO:
- Run predictions with concat UDPipe/Stanza models

STRETCH GOAL:
- Find preprocessing models that do best on average on dev treebanks per language
- Train with preprocessed UDPipe/Stanza
- Repeat with lang-spec BERT
- Repeat with XLM-R large
- Treebank embeddings with "guessing" during test




train+dev: /cluster/projects/nn9447k/mdelhoneux/train-dev

test: /cluster/projects/nn9447k/mdelhoneux/test/*.txt

models: /cluster/projects/nn9447k/mdelhoneux/models/{mbert,langspecbert}/

1. Train TB with gold conllu
   bash/send_all_jobs.py, which calls bash/train_ud_saga.sh
2. Train TB with preprocessed-stanza conllu dev
	dev: /cluster/projects/nn9447k/mdelhoneux/train-dev/*-preprocessed-stanza*.conllu
3. Train TB with preprocessed-udpipe conllu dev
	dev: /cluster/projects/nn9447k/mdelhoneux/train-dev/*-preprocessed-udpipe*.conllu
4. Run TB predictions with preprocessed-stanza dev
	input: /cluster/projects/nn9447k/mdelhoneux/train-dev/*-preprocessed-stanza*.conllu
<!---
5. Run TB predictions with preprocessed-udpipe dev
	input: /cluster/projects/nn9447k/mdelhoneux/train-dev/*-preprocessed-udpipe*.conllu
--->    
6. Run TB predictions with preprocessed-stanza test
	input: /cluster/projects/nn9447k/mdelhoneux/test/*-preprocessed-stanza*.conllu
	output: /cluster/projects/nn9447k/mdelhoneux/test/*-pred-udpipe*.conllu
<!---
7. Run TB predictions with preprocessed-udpipe test
	input: /cluster/projects/nn9447k/mdelhoneux/test/*-preprocessed-udpipe*.conllu
	output: /cluster/projects/nn9447k/mdelhoneux/test/*-pred-udpipe*.conllu
-->
8. Collect full predictions into one directory

