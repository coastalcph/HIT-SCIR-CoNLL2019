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
import re


@Metric.register("xud")
class XUDScore(Metric):
    def __init__(self, collapse=True):
        self.predictions = []
        self.golds = []
        self._collapse = 'tools/enhanced_collapse_empty_nodes.pl'
        self.collapse = collapse

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
        self.predictions.append(predictions)
        self.golds.append(golds)

    def get_metric(self, reset: bool = False) -> Dict[str,float]:
        if self.predictions and self.golds:
            #predictions
            #flatten list
            preds = [item for sublist in self.predictions for item in sublist]
            #hack to avoid double new lines when joining
            preds = [pred if pred != '\n' else '' for pred in preds]
            if self.collapse:
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
                string_pred = ''.join(preds)
            else:
                string_pred = '\n'.join(preds)+'\n'

            pred_graphs = load_conllu_default(string_to_file(string_pred))

            #gold
            golds = [item for sublist in self.golds for item in sublist]
            golds = [gold if gold != '\n' else '' for gold in golds]
            if self.collapse:
                string_gold = '\n'.join(golds)+'\n'
                string_gold = string_gold.encode("utf-8")
                gold_tmp_file = tempfile.NamedTemporaryFile(delete=False)
                gold_tmp_file.write(string_gold)
                gold_tmp_file.close()
                try:
                    gold_collapsed = self.collapse_conllu(gold_tmp_file.name)
                except subprocess.CalledProcessError:
                    return
                golds = [gold for gold in gold_collapsed.stdout.partition("\n")]
                string_gold = ''.join(golds)
            else:
                string_gold = '\n'.join(golds)+'\n'
            gold_graphs = load_conllu_default(string_to_file(string_gold))
            results = evaluate(gold_graphs, pred_graphs)

        if reset:
            self.predictions = []
            self.golds = []
        #print(results['ELAS'].f1)
        return {"ELAS":results['ELAS'].f1}

    @retry(stop_max_attempt_number=3)
    def collapse_conllu(self,filename):
        cmd = ["perl", self._collapse, filename]
        return subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True,
                    check=True)

