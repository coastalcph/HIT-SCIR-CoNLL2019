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
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return outputs

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return outputs

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        #TODO: add sentence ID?
        predictions= eud_trans_outputs_into_conllu(outputs)
        #avoids printing two new lines
        predictions = [pred if pred != '\n' else '' for pred in predictions]
        return '\n'.join(predictions)+'\n'


def eud_trans_outputs_into_conllu(outputs):
    return annotation_to_conllu(eud_trans_outputs_to_annotation(outputs))

def serialize_field(field,key):
    if isinstance(field, (str,int)):
        return str(field)
    elif field is None:
        return '_'
    elif key in ['feats', 'misc']:
        return '|'.join(f'{k}={v}' for k,v in field.items())
    elif key == 'deps':
        return "|".join(f"{v}:{k}" for k,v in field)
    else:
        raise ValueError(f"Type not known for {key}, value: {field}")

def annotation_to_conllu(annotation):
    output_lines = []

    for i, line in enumerate(annotation, start=1):
        line = [serialize_field(line.get(k), k) for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
            "deprel", "deps", "misc"]]

        row = "\t".join(line)

        output_lines.append(row)

    return output_lines + ['\n']

def eud_trans_outputs_to_annotation(outputs):
    edge_list = outputs["edge_list"]
    null_nodes = outputs["null_node"]

    word_count = len(outputs["form"])
    annotation = [{} for _ in range(word_count)]
    for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
        "deprel", "misc"]:
        for token_annotation_dict, token_annotation_for_k in zip(annotation,outputs[k]):
            token_annotation_dict[k] = token_annotation_for_k

    multiword_map = None
    if outputs["multiword_ids"]:
        multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
        multiword_forms = outputs["multiword_forms"]
        multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}

    null_node_prefix = len(annotation)+1
    token_index_to_id = {len(annotation):0}
    for i in range(len(annotation)):
        token_index_to_id[i]= i+1
    null_node_id = {}
    if null_nodes:
        for i, node in enumerate(null_nodes,start=1):
            token_index_to_id[null_node_prefix + i] = null_node_id[i] = f'{null_node_prefix}.{i}'

    output_annotation = []
    for i, line in enumerate(annotation,start=1):

        # Handle multiword tokens
        if multiword_map and i in multiword_map:
            id_, form = multiword_map[i]
            row = {"id":id_, "form":form}
            output_annotation.append(row)
        deps = "|".join([str(token_index_to_id[edge[1]]) + ':' + edge[2] for edge in edge_list if token_index_to_id[edge[0]] == i])
        #print(deps)
        if not deps:
            raise ValueError(f"No edge found for {annotation[i-1]}")

        line['deps'] = deps
        output_annotation.append(line)

    if null_nodes:
        for i, node in enumerate(null_nodes,start=1):
            deps_list = [token_index_to_id[edge[1]] + ':' + edge[2] for edge in edge_list if token_index_to_id[edge[0]]-null_node_prefix == i]
            if not deps_list:
                raise ValueError(f"deps empty")
            deps = "|".join(deps_list)
            row = {"id":null_node_id[i], "deps":deps}
            output_annotation.append(row)

    return output_annotation

