import json
from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from utils.transition_sdp_reader import parse_sentence


@Predictor.register('transition_predictor_eud')
class EUDPParserPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a dependency parse for the given sentence.
        Parameters
        ----------
        sentence The sentence to parse.

        Returns
        -------
        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        """

        ret = parse_sentence(json.dumps(json_dict))

        tokens = ret["tokens"]
        meta_info = ret["meta_info"]
        tokens_range = ret["tokens_range"]

        return self._dataset_reader.text_to_instance(tokens=tokens, meta_info=[meta_info], tokens_range=tokens_range)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        ret_dict = eud_trans_outputs_into_conllu(outputs)

        return sanitize(ret_dict)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs_batch = self._model.forward_on_instances(instances)

        ret_dict_batch = [[] for i in range(len(outputs_batch))]
        for outputs_idx in range(len(outputs_batch)):
            try:
                ret_dict_batch[outputs_idx] = eud_trans_outputs_into_conllu(outputs_batch[outputs_idx])
            except:
                print('graph_id:' + json.loads(outputs_batch[outputs_idx]["meta_info"])['id'])

        return sanitize(ret_dict_batch)


def eud_trans_outputs_into_conllu(outputs):
    edge_list = outputs["edge_list"]
    tokens = outputs["words"]
    pos_tag = outputs["pos_tag"]

    word_count = len([word for word in outputs["words"]])
    lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
        for k in ["ids", "words", "lemmas", "upos", "xpos", "feats",
            "heads", "dependencies"]])

    multiword_map = None
    if outputs["multiword_ids"]:
        multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
        multiword_forms = outputs["multiword_forms"]
        multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}

    null_node_prefix = len(lines)
    #we might need to reverse this???
    token_index_to_id = {i: str(i) for i in range(len(lines))}
    null_node_id = {}
    if outputs["null_nodes"]:
        for i, node in enumerate(null_nodes,start=1):
            token_index_to_id[null_node_prefix + i] = null_node_id[i] = f'{null_node_prefix}.{i}'

    output_lines = []
    for i, line in enumerate(lines, start=1):
        line = [str(l) for l in line]

        # Handle multiword tokens
        if multiword_map and i in multiword_map:
            id_, form = multiword_map[i]
            row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
            output_lines.append(row)
        deps = "|".join([token_index_to_id[edge[1]] + ':' + edge[2] for edge in edge_list if edge[0] == i])

        row = "\t".join(line) + '\t' + deps +\
                "".join(["\t_"])
        output_lines.append(row)

    if outputs["null_nodes"]:
        for i, node in enumerate(null_nodes,start=1):
            deps = "|".join([token_index_to_id[edge[1]] + ':' + edge[2] for edge in edge_list if edge[0]-null_node_prefix == i])
            row = "\t".join([null_node_id[i]] + ['_']*7) + '\t' + deps+ '\t_'
        output_lines.append(row)

    return "\n".join(output_lines) + "\n\n"

