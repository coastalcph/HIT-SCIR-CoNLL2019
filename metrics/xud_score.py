import os
import sys
from typing import Dict, List, Any

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from metrics.iwpt20_xud_eval import evaluate, load_conllu_default
from conllu import string_to_file


@Metric.register("xud")
class XUDScore(Metric):
    def __init__(self):
        self.results = []

    @overrides
    def __call__(self,
                 predictions: List[str],
                 golds: List[str]):
        """
        Parameters
        ----------
        predictions : the predicted graph
        golds : the gold graph
        """

        #hack to avoid double new lines when joining
        predictions = [pred if pred != '\n' else '' for pred in predictions]
        golds = [gold if gold != '\n' else '' for gold in golds]
        try:
            pred_graphs = load_conllu_default(string_to_file('\n'.join(predictions)+'\n'))
        except Exception:
            raise Exception('\n'.join(string_to_file('\n'.join(predictions)).readlines()))
        gold_graphs = load_conllu_default(string_to_file('\n'.join(golds)+'\n'))
        self.results.append(evaluate(gold_graphs, pred_graphs))

    def get_metric(self, reset: bool = False) -> Dict[str,float]:
        results = {'ELAS':0}
        for result in self.results:
            results['ELAS'] += result['ELAS'].f1

        results['ELAS']/=len(self.results)
        if reset:
            self.results = []
        return results

