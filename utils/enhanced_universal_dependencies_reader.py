from typing import Dict, Tuple, List, Any
import logging

from overrides import overrides
from conllu import parse_incr, string_to_file

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from .enhanced_universal_dependencies_oracle import get_oracle_actions

logger = logging.getLogger(__name__)


@DatasetReader.register("enhanced_universal_dependencies")
class EnhancedUniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    tokenizer : `Tokenizer`, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        action_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        max_heads: int = None,
        max_sentence_length: int = None,
        lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._action_indexers = None
        self.max_heads = max_heads
        self.max_sentence_length = max_sentence_length
        if action_indexers is not None and len(action_indexers) > 0:
            self._action_indexers = action_indexers

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:

            conllu_string= conllu_file.read()
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(string_to_file(conllu_string)):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                word_annotation = [x for x in annotation if isinstance(x["id"], int)]
                #ignore long sentences
                if self.max_sentence_length and len(word_annotation) > self.max_sentence_length:
                    continue

                words = [x["form"] for x in word_annotation]
                enhanced_arc_indices = []
                enhanced_arc_tags = []
                multiwords = []

                null_node_ids = []
                node_ids = ['0']
                token_node_ids = []
                for x in annotation:
                    node_ids.append(str(x['id']))
                    if isinstance(x['id'],tuple):
                        if x['id'][1] == '.':
                            null_node_ids.append(str(x['id']))
                        elif x['id'][1] == '-':
                            multiwords.append({'id':''.join([str(tup) for tup in (x['id'])]),'form':x['form']})
                        else:
                            raise TypeError(f"Unknown token ID type: {x['id']}")

                    else:
                        token_node_ids.append(str(x['id']))

                    if not x['deps'] == None:
                        if self.max_heads:
                            x['deps'] = x['deps'][:self.max_heads]

                        for tag,ind2 in x['deps']:
                            enhanced_arc_indices.append((str(x['id']),str(ind2)))
                            enhanced_arc_tags.append(tag)

                gold_actions = get_oracle_actions(token_node_ids, enhanced_arc_indices, enhanced_arc_tags,
                                                  null_node_ids, node_ids) if enhanced_arc_indices else None

                if gold_actions and gold_actions[-2] == '-E-':
                    print('Oracle failed to complete the tree, actions:')
                    print(gold_actions)
                    continue

                yield self.text_to_instance(words, annotation, gold_actions, multiwords)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        annotation: List[Dict[str,Any]],
        gold_actions: List[str] = None,
        multiwords: List[Dict[str,str]]=None,
    ) -> Instance:

        """
        # Parameters

        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies : `List[Tuple[str, int]]`, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        # Returns

        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        text_field = TextField([Token(t) for t in words], self._token_indexers)
        meta_dict = {"words": words, "annotation":annotation, "multiwords": multiwords}
        fields["words"] = text_field

        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        fields["metadata"] = MetadataField(meta_dict)
        return Instance(fields)
