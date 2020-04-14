TODO:
MUST:
Miryam 1. Train language-specific instead of treebank-specific models by joining the training treebanks
Artur 2. Fix predictions with preprocessed UDPipe+Stanza
  - Fix prediction bug
  - Both dev and test: preprocessing with Stanza+UDPipe with largest model
  - Both dev and test: parser model from joint language
Daniel 3. Evaluate on dev to make sure they are valid, update dev scores for all languages
  - Verify valid output for ar_padt, cs_cac, fi_tdt (long sentences are skipped)
Artur 4. Predictions on test with preprocessed UDPipe+Stanza
Daniel 5. Collect to directory for submission https://universaldependencies.org/iwpt20/submission.html

NICE TO HAVE:
- Train new UDPipe/Stanza models from joined language training sets instead of per treebank
- Find preprocessing models that do best on average on dev treebanks per language
- Repeat with lang-spec BERT

STRETCH GOAL:
- Submit training jobs with preprocessed UDPipe/Stanza
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

