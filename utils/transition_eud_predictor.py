import json
import re
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
        predictions= eud_trans_outputs_into_conllu(outputs)
        #avoids printing two new lines
        predictions = [pred if pred != '\n' else '' for pred in predictions]
        return '\n'.join(predictions)+'\n'


def eud_trans_outputs_into_conllu(outputs, output_null_nodes=True):
    return annotation_to_conllu(eud_trans_outputs_to_annotation(outputs), output_null_nodes)

NNODE_IN_DEPS = re.compile('(([0-9][0-9]*)\.[1-9][0-9]*)')

def serialize_field(field, key, output_null_nodes=True):
    #bit of an overkill but let's play it safe
    field_type_error = False
    if field == '_' or field is None:
        return '_'
    elif key in ["id", "form", "lemma", "upostag", "xpostag", "head", "deprel"]:
        if isinstance(field, (str,int,float)):
            return str(field)
        elif isinstance(field,tuple):
            return ''.join(str(x) for x in field)
        else:
            field_type_error = True
    elif key in ['feats', 'misc']:
        if isinstance(field,str):
            return field
        elif isinstance(field,dict):
            return '|'.join(f'{k}={v}' for k,v in field.items())
        else:
            field_type_error = True
    elif key == 'deps':
        if isinstance(field,str):
            if not output_null_nodes:
                field = re.sub(NNODE_IN_DEPS, r'\2', field)
            return field
        else:
            dep_list = []
            for i,(k,v) in enumerate(field):
                if isinstance(v,tuple):
                    if output_null_nodes:
                        dep_list.append((k,''.join(str(val) for val in v)))
                    else:
                        dep_list.append((k, v[0]))
                elif isinstance(v,int):
                    dep_list.append((k,v))
                else:
                    field_type_error = True

            return "|".join(f"{v}:{k}" for k,v in dep_list)
    if field_type_error:
        raise ValueError(f"Type not known for {key}, value: {field}")

NULL_NODE_ID = re.compile(r"^[0-9][0-9]*\.[1-9][0-9]*")
def annotation_to_conllu(annotation, output_null_nodes=True):
    output_lines = []
    for key in "sent_id", "text":
        if key in annotation[0] and annotation[0][key] is not None:
            output_lines.append(f'# {key} = {annotation[0][key]}'.encode(encoding='UTF-8',errors='strict').decode())

    for i, line in enumerate(annotation[1:], start=1):
        line = [serialize_field(line.get(k), k, output_null_nodes) for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
            "deprel", "deps", "misc"]]
        if output_null_nodes or not re.match(NULL_NODE_ID, str(line[0])):
            row = "\t".join(line).encode(encoding='UTF-8',errors='strict').decode()
            output_lines.append(row)

    return output_lines + ['\n']

def eud_trans_outputs_to_annotation(outputs, output_null_nodes = True):
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
            multiword_map[int(start)] = d

    null_node_prefix = len(word_annotation)
    token_index_to_id = {len(word_annotation):0}
    for i in range(len(word_annotation)):
        token_index_to_id[i]= i+1
    if null_nodes:
        for i, node in enumerate(null_nodes, start=1):
            if output_null_nodes:
                token_index_to_id[null_node_prefix + i] = f'{null_node_prefix}.{i}'
            else:
                #TODO: hacky! come up with something better
                #attach everything that is attached to a node to the last sentence item
                token_index_to_id[null_node_prefix + i] = null_node_prefix

    output_annotation = []
    output_annotation.append({'sent_id':outputs["sent_id"],'text':outputs['text']})
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
        deps =[(token_index_to_id[edge[1]], edge[2]) for edge in edge_list if token_index_to_id[edge[0]] == i ]
        deps = sorted(deps, key=lambda x:tuple(int(i) for i in str(x[0]).split(".")))
        string_deps = "|".join([str(dep[0]) + ':' + dep[1] for dep in deps])

        if not deps:
            raise ValueError(f"No edge found for {line}")

        line['deps'] = string_deps
        output_annotation.append(line)

    if null_nodes and output_null_nodes:
        for i, node in enumerate(null_nodes, start=null_node_prefix+1):
            deps = [(token_index_to_id[edge[1]], edge[2]) for edge in edge_list if edge[0] == i]
            deps = sorted(deps, key=lambda x:tuple(int(i) for i in str(x[0]).split(".")))
            string_deps = "|".join([str(dep[0]) + ':' + dep[1] for dep in deps])
            if not deps:
                raise ValueError(f"No edge found for {node}")
            row = {"id":token_index_to_id[i], "deps":string_deps}
            output_annotation.append(row)

    return output_annotation

