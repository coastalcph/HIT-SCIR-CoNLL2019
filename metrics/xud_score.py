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
        results = {}
        if self.predictions and self.golds:
            string_pred = self.get_string(self.predictions)
            pred_graphs = load_conllu_default(string_to_file(string_pred))
            string_gold = self.get_string(self.golds)
            gold_graphs = load_conllu_default(string_to_file(string_gold))
            results = evaluate(gold_graphs, pred_graphs)

        if reset:
            self.predictions = []
            self.golds = []
        return {"ELAS":results['ELAS'].f1}

    @retry(stop_max_attempt_number=3)
    def collapse_conllu(self,filename):
        cmd = ["perl", self._collapse, filename]
        return subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True,
                    check=True)

    def get_string(self, conllu_list):
        #flatten list
        conllu_list = [item for sublist in conllu_list for item in sublist]
        #hack to avoid double new lines when joining
        conllu_list = [conllu_item if conllu_item != '\n' else '' for conllu_item in conllu_list]
        if self.collapse:
            string_conllu_list = '\n'.join(conllu_list)+'\n'
            string_conllu_list = string_conllu_list.encode("utf-8")
            conllu_list_tmp_file = tempfile.NamedTemporaryFile(delete=False)
            conllu_list_tmp_file.write(string_conllu_list)
            conllu_list_tmp_file.close()
            try:
                conllu_list_collapsed = self.collapse_conllu(conllu_list_tmp_file.name)
            except subprocess.CalledProcessError:
                return
            conllu_list = [conllu_item for conllu_item in conllu_list_collapsed.stdout.partition("\n") ]
            string_conllu_list = ''.join(conllu_list)
        else:
            string_conllu_list = '\n'.join(conllu_list)+'\n'
        return string_conllu_list

