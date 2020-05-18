#!/bin/bash
#SBATCH --account=nn9447k
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --time=09:00:00
#SBATCH --mem-per-cpu=50G

module load Perl/5.30.0-GCCcore-8.3.0
checkpoint_dir=$1
preprocessed_file=$2
output_file=$3
pred_file=${output_dir:-$checkpoint_dir}/$output_file

allennlp predict \
    --output-file $pred_file \
    --predictor transition_predictor_eud \
    --include-package utils \
    --include-package modules \
    --use-dataset-reader \
    --batch-size 32 \
    --silent \
    --cuda-device 0 \
    --override "{\"model\": {\"output_null_nodes\": true, \"max_heads\": 7, \"max_swaps_per_node\": 30, \"fix_unconnected_egraph\": true}}" \
    $checkpoint_dir \
    $preprocessed_file \

if [ $# -ge 4 ]; then
  gold_file=$4
  eval_dir=${output_dir:-$(mktemp)}
  mkdir -p $eval_dir/collapsed/{gold,pred}
  perl tools/enhanced_collapse_empty_nodes.pl $pred_file > $eval_dir/collapsed/pred/$output_file
  perl tools/enhanced_collapse_empty_nodes.pl $gold_file > $eval_dir/collapsed/gold/$output_file
  python tools/iwpt20_xud_eval.py $eval_dir/collapsed/gold/$output_file $eval_dir/collapsed/pred/$output_file | tee $eval_dir/${output_file%%-*}_eval.log
  grep -zPo "(?<=ELAS F1 Score: ).*" $eval_dir/${output_file%%-*}_eval.log > ${output_dir:-.}/${output_file%%-*}_elas.txt
fi
