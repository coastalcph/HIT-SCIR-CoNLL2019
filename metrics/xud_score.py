import os
import sys
from typing import Dict, List, Any
import tempfile
import subprocess
from retrying import retry

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
        #TODO: this code is ugly and possibly suboptimal
        #predictions
        predictions = [pred if pred != '\n' else '' for pred in predictions]
        string_pred = '\n'.join(predictions)+'\n'
        string_pred = string_pred.encode("utf-8")
        pred_tmp_file = tempfile.NamedTemporaryFile(delete=False)
        pred_tmp_file.write(string_pred)
        pred_tmp_file.close()
        try:
            pred_collapsed = self.collapse_conllu(pred_tmp_file.name)
        except subprocess.CalledProcessError:
            return

        preds = [pred for pred in pred_collapsed.stdout.partition("\n") ]
        pred_graphs = load_conllu_default(string_to_file(''.join(preds)))

        #gold
        golds = [gold if gold != '\n' else '' for gold in golds]
        string_gold = '\n'.join(golds)+'\n'
        string_gold = string_gold.encode("utf-8")
        gold_tmp_file = tempfile.NamedTemporaryFile(delete=False)
        gold_tmp_file.write(string_gold)
        gold_tmp_file.close()
        try:
            gold_collapsed = self.collapse_conllu(gold_tmp_file.name)
        except subprocess.CalledProcessError:
            return
        golds_c = [gold for gold in gold_collapsed.stdout.partition("\n")]
        gold_graphs = load_conllu_default(string_to_file(''.join(golds_c)))
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

    @retry(stop_max_attempt_number=3)
    def collapse_conllu(self,filename):
        cmd = ["perl", self._collapse, filename]
        return subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True,
                    check=True)

