import logging
from typing import Dict, Any, List

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model

from metrics.xud_score import XUDScore
from utils import eud_trans_outputs_into_conllu, annotation_to_conllu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("noop_parser_ud")
class NoopParser(Model):
    def __init__(self, vocab: Vocabulary) -> None:

        super(NoopParser, self).__init__(vocab)

        self._unlabeled_correct = 0
        self._labeled_correct = 0
        self._total_edges_predicted = 0
        self._total_edges_actual = 0
        self._exact_unlabeled_correct = 0
        self._exact_labeled_correct = 0
        self._total_sentences = 0

        self.num_actions = vocab.get_vocab_size('actions')
        self._xud_score = XUDScore()

        self.p = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def forward(self,
                words: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                enhanced_arc_tags: torch.LongTensor = None,
                ) -> Dict[str, torch.LongTensor]:

        batch_size = len(metadata)

        if self.training:
            return {"loss": 1/self.p}

        training_mode = self.training
        self.eval()
        self.train(training_mode)

        sent_len = [len(d['words']) for d in metadata]
        token_id = []
        for sent_idx in range(batch_size):
            token_id.append({i: sent_len[sent_idx] - i for i in range(1 + sent_len[sent_idx])})

        # prediction-mode
        output_dict = {
            'edge_list': [[(token_id[sent_idx][tok_metadata["id"]],
                            token_id[sent_idx][tok_metadata["head"]],
                            tok_metadata["deprel"])
                           for tok_metadata in sent_metadata["annotation"]
                           if isinstance(tok_metadata["id"], int)]
                          for sent_idx, sent_metadata in enumerate(metadata)],
            'null_node': [[tok_metadata for tok_metadata in sent_metadata["annotation"]
                           if "." in str(tok_metadata["id"])]
                          for sent_metadata in metadata],
            "multiwords": [sent_metadata['multiwords'] for sent_metadata in metadata],
            'loss': self.p
        }

        for k in "id", "form", "lemma", "upostag", "xpostag", "feats", "head", "deprel", "misc":
            output_dict[k] = [[tok_metadata[k] for tok_metadata in sent_metadata['annotation']]
                              for sent_metadata in metadata]

        # validation mode
        if gold_actions is not None:
            predicted_graphs_conllu = []
            for sent_idx in range(batch_size):
                predicted_graphs_conllu += eud_trans_outputs_into_conllu({
                    k: output_dict[k][sent_idx] for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
                                                          "deprel", "misc", "edge_list", "null_node", "multiwords"]
                })
            gold_graphs_conllu = [line for sent_metadata in metadata
                                  for line in annotation_to_conllu(sent_metadata['annotation'])]
            self._xud_score(predicted_graphs_conllu,
                            gold_graphs_conllu)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._xud_score is not None and not self.training:
            all_metrics.update(self._xud_score.get_metric(reset=reset))
        return all_metrics
