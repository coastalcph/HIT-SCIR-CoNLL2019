import os
import sys
from typing import Dict, List, Any
import tempfile
import subprocess

from overrides import overrides

from allennlp.training.metrics.metric import Metric
from metrics.iwpt20_xud_eval import evaluate, load_conllu_default, UDError
from conllu import string_to_file


@Metric.register("xud")
class XUDScore(Metric):
    def __init__(self):
        self.results = []
        self._collapse = 'tools/enhanced_collapse_empty_nodes.pl'

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
        string_pred = '\n'.join(predictions)+'\n'
        string_pred = string_pred.encode("utf-8")
        pred_tmp_file = tempfile.NamedTemporaryFile(delete=False)
        pred_tmp_file.write(string_pred)
        pred_tmp_file.close()
        cmd = ["perl", self._collapse, pred_tmp_file.name]
        pred_collapsed = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True,
                check=True)
        preds = [pred for pred in pred_collapsed.stdout.partition("\n") ]
        try:
            pred_graphs = load_conllu_default(string_to_file(''.join(preds)))
        except UDError:
            print('string pred')
            print(string_pred)
            print('preds')
            print(preds)
            print(pred_tmp_file.name)
            return
        golds = [gold if gold != '\n' else '' for gold in golds]
        string_gold = '\n'.join(golds)+'\n'
        string_gold = string_gold.encode("utf-8")
        gold_tmp_file = tempfile.NamedTemporaryFile(delete=False)
        gold_tmp_file.write(string_gold)
        gold_tmp_file.close()
        cmd = ["perl", self._collapse, gold_tmp_file.name]
        gold_collapsed = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True,
                check=True)
        golds_c = [gold for gold in gold_collapsed.stdout.partition("\n")]
        try:
            gold_graphs = load_conllu_default(string_to_file(''.join(golds_c)))
        except UDError:
            print('sting gold')
            print(string_gold)
            print('golds_c')
            print(golds_c)
            print(gold_tmp_file.name)
            return

        self.results.append(evaluate(gold_graphs, pred_graphs))

    def get_metric(self, reset: bool = False) -> Dict[str,float]:
        results = {'ELAS':0}
        if self.results:
            for result in self.results:
                results['ELAS'] += result['ELAS'].f1

            results['ELAS']/=len(self.results)
        if reset:
            self.results = []
        return results

