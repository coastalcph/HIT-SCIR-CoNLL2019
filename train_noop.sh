nlp_datasets=/image/nlp-datasets
train_set=${nlp_datasets}/iwpt20/train-dev/UD_Arabic-PADT/ar_padt-ud-train.conllu
dev_set=${nlp_datasets}/iwpt20/train-dev/UD_Arabic-PADT/ar_padt-ud-dev.conllu
model_save_path=models/noop_eud
config_file=config/noop_eud.jsonnet

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
