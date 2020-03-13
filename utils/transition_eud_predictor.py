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
    if field == '_' or field is None:
        return '_'
    elif key in ["id", "form", "lemma", "upostag", "xpostag", "head", "deprel"]:
        if isinstance(field, (str,int)):
            return str(field)
        elif isinstance(field,tuple):
            return ''.join(str(x) for x in field)
    elif key in ['feats', 'misc']:
        if isinstance(field,str):
            return field
        else:
            return '|'.join(f'{k}={v}' for k,v in field.items())
    elif key == 'deps':
        if isinstance(field,str):
            return field
        else:
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
    annotation = [{} for _ in range(len(outputs["form"]))]
    for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
        "deprel", "misc"]:
        for token_annotation_dict, token_annotation_for_k in zip(annotation,outputs[k]):
            token_annotation_dict[k] = token_annotation_for_k
    word_annotation = [x for x in annotation if isinstance(x['id'],int)]

    multiword_map = None
    if outputs["multiwords"]:
        multiword_map = {}
        for d in outputs['multiwords']:
            start,_ = d['id'].split("-")
            multiword_map[int(start)] = (d)

    null_node_prefix = len(word_annotation)
    token_index_to_id = {len(word_annotation):0}
    for i in range(len(word_annotation)):
        token_index_to_id[i]= i+1
    if null_nodes:
        for i, node in enumerate(null_nodes, start=1):
            token_index_to_id[null_node_prefix + i] = f'{null_node_prefix}.{i}'

    output_annotation = []
    #this part is a bit of a headache
    # index is the indexing in the parser
    # ID is the actual ID of words
    # i in the first enumerate is equivalent to ID so we need to convert edge[0] to ID 
    # i in the second enumerate is equivalent to index so no need to convert
    # EXAMPLE:
    # idx, ID,   i
    # 2     0    NA
    # 0     1    1
    # 1     2    2
    # 3     2.1  3
    # 4     2.2  4

    for i, line in enumerate(word_annotation,start=1):

        # Handle multiword tokens
        if multiword_map and i in multiword_map:
            output_annotation.append(multiword_map[i])
        deps = "|".join([str(token_index_to_id[edge[1]]) + ':' + edge[2] for edge in edge_list if token_index_to_id[edge[0]] == i])
        if not deps:
            raise ValueError(f"No edge found for {line}")

        line['deps'] = deps
        output_annotation.append(line)

    if null_nodes:
        for i, node in enumerate(null_nodes, start=null_node_prefix+1):
            deps = "|".join([str(token_index_to_id[edge[1]]) + ':' + edge[2] for edge in edge_list if edge[0] == i])
            if not deps:
                raise ValueError(f"No edge found for {node}")
            row = {"id":token_index_to_id[i], "deps":deps}
            output_annotation.append(row)

    return output_annotation

