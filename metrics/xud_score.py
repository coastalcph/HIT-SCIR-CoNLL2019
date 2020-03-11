import os
import sys
from typing import Dict, List, Any
import tempfile

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
        string_pred = '\n'.join(predictions)+'\n'
        string_pred = string_pred.encode()
        pred_tmp_file = tempfile.NamedTemporaryFile(delete=False)
        pred_tmp_file.write(string_pred)
        pred_tmp_file.close()
        cmd = f'perl tools/enhanced_collapse_empty_nodes.pl {pred_tmp_file.name}'
        pred_collapsed = os.popen(cmd)
        preds = [pred if pred != '\n' else '' for pred in pred_collapsed]
        pred_graphs = load_conllu_default(string_to_file(''.join(preds)+'\n'))
        string_gold = '\n'.join(golds)+'\n'
        string_gold = string_gold.encode()
        gold_tmp_file = tempfile.NamedTemporaryFile(delete=False)
        gold_tmp_file.write(string_gold)
        gold_tmp_file.close()
        cmd = f'perl tools/enhanced_collapse_empty_nodes.pl {gold_tmp_file.name}'
        gold_collapsed = os.popen(cmd)
        golds_c = [gold if gold != '\n' else '' for gold in gold_collapsed]
        gold_graphs = load_conllu_default(string_to_file(''.join(golds_c)+'\n'))
        self.results.append(evaluate(gold_graphs, pred_graphs))

    def get_metric(self, reset: bool = False) -> Dict[str,float]:
        results = {'ELAS':0}
        for result in self.results:
            results['ELAS'] += result['ELAS'].f1

        results['ELAS']/=len(self.results)
        if reset:
            self.results = []
        return results

