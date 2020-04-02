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
# 5. Run TB predictions with preprocessed-udpipe dev
#	input: /cluster/projects/nn9447k/mdelhoneux/train-dev/*-preprocessed-udpipe*.conllu
6. Run TB predictions with preprocessed-stanza test
	input: /cluster/projects/nn9447k/mdelhoneux/test/*-preprocessed-stanza*.conllu
	output: /cluster/projects/nn9447k/mdelhoneux/test/*-pred-udpipe*.conllu
# 7. Run TB predictions with preprocessed-udpipe test
#	input: /cluster/projects/nn9447k/mdelhoneux/test/*-preprocessed-udpipe*.conllu
#	output: /cluster/projects/nn9447k/mdelhoneux/test/*-pred-udpipe*.conllu
8. Collect full predictions into one directory

TODO:
- Fix training for ar_padt (mem error, long sent), cz_cac, fi_tdt (mem error, long sent)
	Miryam and Daniel both trying to train
- Fix model path in training script
	Daniel
- Copy already trained models to new path
	Artur, Miryam
- Support preprocessed stanza in send_all_jobs - update paths
	Artur
- Update paths in predict_ud.sh
	Artur
- Create predict_ud_test.sh
	Artur
- Submit training jobs with preprocessed stanza
	Artur
- Predictions on dev with preprocessed stanza - add scores to spreadsheet
	Artur
- Predictions on test with preprocessed stanza - collect to directory for submission
- Repeat with UDPipe
	Elham
- Repeat with lang-spec BERT
- Repeat with XLM-R large

