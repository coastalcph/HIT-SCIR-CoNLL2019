{
  "vocabulary": {
    "non_padded_namespaces": []
  },
  "dataset_reader": {
      "type": "enhanced_universal_dependencies",
  "action_indexers": {
      "actions": {
          "type": "single_id",
          "namespace": "actions"
      }
  },
  "arc_tag_indexers": {
      "arc_tags": {
          "type": "single_id",
          "namespace": "arc_tags"
      }
  },
  },
  "train_data_path": std.extVar('TRAIN_PATH'),
  "validation_data_path": std.extVar('DEV_PATH'),
  "model": {
    "type": "noop_parser_ud"
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["words", "num_tokens"]],
    "batch_size": std.parseInt(std.extVar('BATCH_SIZE'))
  },
  "trainer": {
    "num_epochs": 2,
    "grad_norm": 5.0,
    "grad_clipping": 5.0,
    "patience": 50,
    "cuda_device": -1,
    "validation_metric": "+ELAS",
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.999],
      "lr": 1e-3
    }
  }
}
