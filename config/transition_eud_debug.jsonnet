{
  "vocabulary": {
    "non_padded_namespaces": [],
    "min_count":{
        "actions": 5
    }
  },
  "dataset_reader": {
      "type": "enhanced_universal_dependencies",
      "token_indexers": {
          "tokens": {
              "type": "bert-pretrained",
              "pretrained_model": std.extVar('BERT_PATH'),
              "do_lowercase": std.extVar('LOWER_CASE')=='TRUE',
              "truncate_long_sequences": false
          }
      },
  "action_indexers": {
      "actions": {
          "type": "single_id",
          "namespace": "actions"
      }
  },
  },
  "train_data_path": std.extVar('TRAIN_PATH'),
  "validation_data_path": std.extVar('DEV_PATH'),
  "model": {
    "type": "transition_parser_ud",
    "text_field_embedder": {
        "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": std.extVar('BERT_PATH'),
            "requires_grad": true,
            "top_layer_only": false
        },
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets", "tokens-type-ids"]
        },
        "allow_unmatched_keys": true
    },
    "action_embedding": {
      "embedding_dim": 50,
      "vocab_namespace": "actions"
    },
    "word_dim": std.parseInt(std.extVar('WORD_DIM')),
    "hidden_dim": 200,
    "action_dim": 50,
    "ratio_dim" : 1,
    "num_layers": 2,
    "recurrent_dropout_probability": 0.2,
    "layer_dropout_probability": 0.2,
    "same_dropout_mask_per_instance": true,
    "input_dropout": 0.2,
    "initializer": [
      ["p_.*weight", {"type": "xavier_uniform"}],
      ["p_.*bias", {"type": "zero"}],
      ["pempty_buffer_emb", {"type": "normal"}],
      ["proot_stack_emb", {"type": "normal"}],
      ["pempty_deque_emb", {"type": "normal"}],
      ["pempty_action_emb", {"type": "normal"}],
    ]
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
     "parameter_groups": [
        [[".*bert.*"], {"lr": 5e-5}],
        [["^((?!bert).)*$"], {}]
      ],
      "betas": [0.9, 0.999],
      "lr": 1e-3
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 2,
      "num_steps_per_epoch": 1000,
      "cut_frac": 0.1,
      "ratio": 32,
      "gradual_unfreezing": true,
      "discriminative_fine_tuning": true,
      "decay_factor": 1.0,
    },
    "num_serialized_models_to_keep": 50
  }
}
