nlp_datasets=/image/nlp-datasets
treebank_name=${1:UD_Arabic-PADT}
iso=${2:ar_padt}
train_set=${nlp_datasets}/iwpt20/train-dev/$treebank_name/$iso-ud-train.conllu
dev_set=${nlp_datasets}/iwpt20/train-dev/$treebank_name/$iso-ud-dev.conllu
model_save_path=models/noop_eud
config_file=config/noop_eud.jsonnet

rm -rf $model_save_path
CUDA_VISIBLE_DEVICES=-1 \
TRAIN_PATH=${train_set} \
DEV_PATH=${dev_set} \
LOWER_CASE=false \
BATCH_SIZE=2 \
    allennlp train \
        -s ${model_save_path} \
        --include-package utils \
        --include-package modules \
        --file-friendly-logging \
        ${config_file}
