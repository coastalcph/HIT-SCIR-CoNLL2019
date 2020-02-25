from typing import Dict, Tuple, List
import logging

from overrides import overrides
from conllu import parse_incr

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
    use_language_specific_pos : `bool`, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    tokenizer : `Tokenizer`, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        action_indexers: Dict[str, TokenIndexer] = None,
        use_language_specific_pos: bool = False,
        arc_tag_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos
        self._action_indexers = None
        if action_indexers is not None and len(action_indexers) > 0:
            self._action_indexers = action_indexers
        self._arc_tag_indexers = None
        if arc_tag_indexers is not None and len(arc_tag_indexers) > 0:
            self._arc_tag_indexers = arc_tag_indexers

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by integers here as elided words have a non-integer word id,
                # as parsed by the conllu python library.
                word_annotation = [x for x in annotation if isinstance(x["id"], int)]
                #annotation = [x for x in annotation]

                heads = [x["head"] for x in word_annotation]
                tags = [x["deprel"] for x in word_annotation]
                words = [x["form"] for x in word_annotation]
                #eud = [x["deps"] for x in annotation]
                enhanced_arc_indices = []
                enhanced_arc_tags = []
                #TODO: Inserted nodes! Right now we ignore but should not I think

                null_node_ids = []
                node_ids = ['0']
                token_node_ids = []
                for x in annotation:
                    node_ids.append(str(x['id']))
                    if not isinstance(x['id'],int):
                        null_node_ids.append(str(x['id']))
                    else:
                        token_node_ids.append(str(x['id']))
                    for tag,ind2 in x['deps']:
                        enhanced_arc_indices.append((str(x['id']),str(ind2)))
                        enhanced_arc_tags.append(tag)

                #TODO: update the oracle to include null nodes
                gold_actions = get_oracle_actions(token_node_ids, enhanced_arc_indices, enhanced_arc_tags, null_node_ids, node_ids)

                if gold_actions[-1] == '-E-':
                    print('-E-')
                    continue

                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
                yield self.text_to_instance(words, pos_tags, list(zip(tags, heads)), enhanced_arc_indices, enhanced_arc_tags, gold_actions)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
        enhanced_arc_indices: List[Tuple[str, str]] = None,
        enhanced_arc_tags: List[str] = None,
        gold_actions: List[str] = None,
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
        meta_dict = {"words": words, "pos":upos_tags}
        fields["words"] = text_field
        #fields["pos_tag"] = SequenceLabelField(upos_tags, text_field, label_namespace="tags")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies], text_field, label_namespace="head_index_tags"
            )

        if enhanced_arc_tags is not None:
            #enhanced_arc_tags, enhanced_arc_indices = enhanced_dependencies
            fields["enhanced_arc_tags"] = TextField([Token(a) for a in enhanced_arc_tags], self._arc_tag_indexers)
            meta_dict["enhanced_arc_tags"] = enhanced_arc_tags
        if enhanced_arc_indices is not None:
            meta_dict["enhanced_arc_indices"] = enhanced_arc_indices
        if enhanced_arc_indices is not None and enhanced_arc_tags is not None:
            meta_dict["gold_graphs"] = [(mod, head, tag) for tag, (mod, head) in zip(enhanced_arc_tags, enhanced_arc_indices)]

        if gold_actions is not None:
            meta_dict["gold_actions"] = gold_actions
            fields["gold_actions"] = TextField([Token(a) for a in gold_actions], self._action_indexers)

        fields["metadata"] = MetadataField(meta_dict)
        return Instance(fields)
