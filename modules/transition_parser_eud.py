import logging
from typing import Dict, Optional, Any, List, Set
from copy import deepcopy

import sys
import torch
from allennlp.data import Vocabulary
from allennlp.models import SimpleTagger
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from torch.nn.modules import Dropout
from metrics.xud_score import XUDScore

from modules import StackRnn, SimpleTagger
from utils import eud_trans_outputs_into_conllu, annotation_to_conllu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("transition_parser_ud")
class TransitionParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 ratio_dim: int,
                 num_layers: int,
                 recurrent_dropout_probability: float = 0.0,
                 layer_dropout_probability: float = 0.0,
                 same_dropout_mask_per_instance: bool = True,
                 input_dropout: float = 0.0,
                 output_null_nodes: bool = True,
                 max_heads: int = None,
                 max_swaps_per_node: int = 3,
                 fix_unconnected_egraph: bool = True,
                 validate_every_n_instances: int = None,
                 action_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:

        super(TransitionParser, self).__init__(vocab, regularizer)

        self._total_validation_instances = 0

        self.num_actions = vocab.get_vocab_size('actions')
        self.text_field_embedder = text_field_embedder
        self.output_null_nodes = output_null_nodes
        self.max_heads = max_heads
        self.max_swaps_per_node = max_swaps_per_node
        self._fix_unconnected_egraph = fix_unconnected_egraph
        self.num_validation_instances = validate_every_n_instances
        self._xud_score = XUDScore(collapse=self.output_null_nodes)


        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.ratio_dim = ratio_dim
        self.action_dim = action_dim

        self.action_embedding = action_embedding

        if action_embedding is None:
            self.action_embedding = Embedding(num_embeddings=self.num_actions,
                                              embedding_dim=action_dim,
                                              trainable=False)


        # syntactic composition
        self.p_comp = torch.nn.Linear(self.hidden_dim * 5 + self.ratio_dim, self.word_dim)
        # parser state to hidden
        self.p_s2h = torch.nn.Linear(self.hidden_dim * 3 + self.ratio_dim, self.hidden_dim)
        # hidden to action
        self.p_act = torch.nn.Linear(self.hidden_dim + self.ratio_dim, self.num_actions)

        self.update_null_node = torch.nn.Linear(self.hidden_dim + self.ratio_dim, self.word_dim)

        self.pempty_buffer_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.proot_stack_emb = torch.nn.Parameter(torch.randn(self.word_dim))
        self.pempty_action_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))
        self.pempty_stack_emb = torch.nn.Parameter(torch.randn(self.hidden_dim))

        self._input_dropout = Dropout(input_dropout)

        self.buffer = StackRnn(input_size=self.word_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=num_layers,
                        recurrent_dropout_probability=recurrent_dropout_probability,
                        layer_dropout_probability=layer_dropout_probability,
                        same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.stack = StackRnn(input_size=self.word_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=num_layers,
                        recurrent_dropout_probability=recurrent_dropout_probability,
                        layer_dropout_probability=layer_dropout_probability,
                        same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        self.action_stack = StackRnn(input_size=self.action_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=num_layers,
                        recurrent_dropout_probability=recurrent_dropout_probability,
                        layer_dropout_probability=layer_dropout_probability,
                        same_dropout_mask_per_instance=same_dropout_mask_per_instance)

        initializer(self)

    def _greedy_decode(self,
                        batch_size: int,
                        sent_len: List[int],
                        embedded_text_input: torch.Tensor,
                        oracle_actions: Optional[List[List[str]]] = None,
                        ) -> Dict[str, Any]:

        self.buffer.reset_stack(batch_size)
        self.stack.reset_stack(batch_size)
        self.action_stack.reset_stack(batch_size)

        # We will keep track of all the losses we accumulate during parsing.
        # If some decision is unambiguous because it's the only thing valid given
        # the parser state, we will not model it. We only model what is ambiguous.

        losses = [[] for _ in range(batch_size)]
        ratio_factor_losses = [[] for _ in range(batch_size)]
        edge_list = [[] for _ in range(batch_size)]
        total_node_num = [0 for _ in range(batch_size)]
        action_list = [[] for _ in range(batch_size)]
        ret_node = [[] for _ in range(batch_size)]
        root_id = [[] for _ in range(batch_size)]
        num_of_generated_node= [[] for _ in range(batch_size)]
        generated_order = [{} for _ in range(batch_size)]
        head_count = [{} for _ in range(batch_size)] # keep track of the number of heads
        swap_count = [{} for _ in range(batch_size)] # keep track of the number of swaps
        reachable = [{} for _ in range(batch_size)]  # set of node ids reachable from each node (by id)
        # push the tokens onto the buffer (tokens is in reverse order)
        for token_idx in range(max(sent_len)):
            for sent_idx in range(batch_size):
                if sent_len[sent_idx] > token_idx:
                    try:
                        self.buffer.push(sent_idx,
                                        input=embedded_text_input[sent_idx][sent_len[sent_idx] - 1 - token_idx],
                                        extra={'token': sent_len[sent_idx] - token_idx - 1})
                    except IndexError:
                        raise IndexError(f"{sent_idx} {batch_size} {token_idx} {sent_len[sent_idx]} {embedded_text_input[sent_idx].dim()}")

        # init stack using proot_emb, considering batch
        for sent_idx in range(batch_size):
            root_id[sent_idx] = sent_len[sent_idx]
            reachable[sent_idx][root_id[sent_idx]] = {root_id[sent_idx]}  # only root is reachable from root so far
            generated_order[sent_idx][root_id[sent_idx]] = 0
            self.stack.push(sent_idx,
                            input=self.proot_stack_emb,
                            extra={'token': root_id[sent_idx]})

        # compute probability of each of the actions and choose an action
        # either from the oracle or if there is no oracle, based on the model
        trans_not_fin = True

        action_tag_for_terminate = [False] * batch_size
        action_sequence_length = [0] * batch_size

        null_node = {}
        for sent_idx in range(batch_size):
            null_node[sent_idx] = []

        while trans_not_fin:
            trans_not_fin = False
            for sent_idx in range(batch_size):
                if (action_sequence_length[sent_idx] > 10000 *
                        sent_len[sent_idx]) and oracle_actions is None:
                    raise RuntimeError(f"Too many actions for a sentence {sent_idx}. actions: {action_list}")
                total_node_num[sent_idx] = sent_len[sent_idx] + len(null_node[sent_idx])
                if not action_tag_for_terminate[sent_idx]:
                    trans_not_fin = True
                    valid_actions = self.calc_valid_actions(edge_list, generated_order, head_count, swap_count,
                                                            null_node, root_id, sent_idx, sent_len)

                    log_probs = None
                    action = valid_actions[0]
                    action_idx = self.vocab.get_token_index(action, namespace='actions')
                    ratio_factor = torch.tensor([len(null_node[sent_idx]) / (1.0 * sent_len[sent_idx])],
                                                device=self.pempty_action_emb.device)

                    if len(valid_actions) > 1:
                        action, action_idx, log_probs, valid_action_tbl = self.predict_action(ratio_factor, sent_idx,
                                                                                              valid_actions)

                    if oracle_actions is not None:
                        action = oracle_actions[sent_idx].pop(0)
                        action_idx = self.vocab.get_token_index(action, namespace='actions')

                    # push action into action_stack
                    self.action_stack.push(sent_idx,
                            input=self.action_embedding(
                                    torch.tensor(action_idx, device=embedded_text_input.device)),
                            extra={
                                    'token': action})
                    action_list[sent_idx].append(action)
                    #print(f'Sent ID: {sent_idx}, action {action}')

                    try:
                        UNK_ID = self.vocab.get_token_index('@@UNKNOWN@@')
                        if log_probs is not None and not (UNK_ID and action_idx == UNK_ID):
                            loss = log_probs[valid_action_tbl[action_idx]]
                            if not torch.isnan(loss):
                                losses[sent_idx].append(loss)
                    except KeyError:
                        raise KeyError(f'action: {action}, valid actions: {valid_actions}')

                    self.exec_action(action, action_sequence_length, action_tag_for_terminate, edge_list,
                                     generated_order, head_count, swap_count, null_node, num_of_generated_node, oracle_actions,
                                     ratio_factor, ratio_factor_losses, sent_idx, sent_len, total_node_num,
                                     reachable)

        if oracle_actions is None and self._fix_unconnected_egraph:
            # Fix edge_list if we are not training. If training, it's empty (no output)
            for sent_idx in range(batch_size):
                self.fix_unconnected_egraph(edge_list[sent_idx], reachable[sent_idx], root_id[sent_idx],
                                            generated_order[sent_idx])

        # categorical cross-entropy
        _loss_CCE = -torch.sum(
                        torch.stack([torch.sum(torch.stack(cur_loss)) for cur_loss in losses if len(cur_loss) > 0])) / \
                                        sum([len(cur_loss) for cur_loss in losses])

        _loss = _loss_CCE

        ret = {
                'loss': _loss,
                'losses': losses,
                }

        # extract null node list in batchmode
        for sent_idx in range(batch_size):
            ret_node[sent_idx] = null_node[sent_idx]

        ret["total_node_num"] = total_node_num

        if oracle_actions is None:
            ret['edge_list'] = edge_list
        ret['action_sequence'] = action_list
        ret["null_node"] = ret_node

        return ret

    def calc_valid_actions(self, edge_list, generated_order, head_count, swap_count, null_node, root_id, sent_idx, sent_len):
        valid_actions = []
        # given the buffer and stack, conclude the valid action list
        if self.buffer.get_len(sent_idx) == 0 and self.stack.get_len(sent_idx) == 1:
            valid_actions += ['FINISH']
        if self.buffer.get_len(sent_idx) > 0:
            valid_actions += ['SHIFT']
        if self.stack.get_len(sent_idx) > 0:
            s0 = self.stack.get_stack(sent_idx)[-1]['token']
            if s0 != root_id[sent_idx] and head_count[sent_idx][s0] > 0:
                valid_actions += ['REDUCE-0']
            if self.output_null_nodes and \
                    len(null_node[sent_idx]) < sent_len[sent_idx]:  # Max number of null nodes is the number of words
                # valid_actions += ['NODE']
                # Support legacy models with "NODE:*" actions that also create an edge:
                node_possible_actions = [a for a in self.vocab.get_token_to_index_vocabulary('actions').keys()
                                         if a.startswith('NODE')
                                         and (":" not in a or s0 == root_id[sent_idx] or a.split('NODE:')[1] != "root")]
                # if len(node_possible_actions) > 1:
                #     logger.warning(f"Possible node actions: {node_possible_actions}. "
                #                    f"NODE:* actions are deprecated - train a new model!")
                valid_actions += node_possible_actions

            if self.stack.get_len(sent_idx) > 1:
                s1 = self.stack.get_stack(sent_idx)[-2]['token']
                if s1 != root_id[sent_idx] and \
                        generated_order[sent_idx][s0] > generated_order[sent_idx][s1] and \
                        swap_count[sent_idx][s1] < self.max_swaps_per_node:
                    valid_actions += ['SWAP']
                if s1 != root_id[sent_idx] and head_count[sent_idx][s1] > 0:
                    valid_actions += ['REDUCE-1']

                # Hacky code to verify that we do not draw the same edge with the same label twice
                labels_left_edge = ['root']  # should not be in vocab anyway actually but be safe
                labels_right_edge = []
                for mod_tok, head_tok, label in edge_list[sent_idx]:
                    if (mod_tok, head_tok) == (s1, s0):
                        labels_left_edge.append(label)
                    if (mod_tok, head_tok) == (s0, s1):
                        labels_right_edge.append(label)
                if s1 != root_id[sent_idx] and (not self.max_heads or
                                                head_count[sent_idx][s1] < self.max_heads):
                    left_edge_possible_actions = \
                        [a for a in self.vocab.get_token_to_index_vocabulary('actions').keys()
                         if a.startswith('LEFT-EDGE') and
                         a.split('LEFT-EDGE:')[1] not in labels_left_edge]
                    valid_actions += left_edge_possible_actions

                if not self.max_heads or head_count[sent_idx][s0] < self.max_heads:
                    if s1 == root_id[sent_idx]:
                        right_edge_possible_actions = [] if "root" in labels_right_edge else ['RIGHT-EDGE:root']
                    else:
                        # hack to disable root
                        labels_right_edge += ['root']
                        right_edge_possible_actions = \
                            [a for a in self.vocab.get_token_to_index_vocabulary('actions').keys()
                             if a.startswith('RIGHT-EDGE')
                             and a.split('RIGHT-EDGE:')[1] not in labels_right_edge]
                    valid_actions += right_edge_possible_actions
        # remove unknown actions:
        vocab_actions = self.vocab.get_token_to_index_vocabulary('actions').keys()
        valid_actions = [valid_action for valid_action in valid_actions if valid_action in vocab_actions]
        if not valid_actions:
            valid_actions = ["FINISH"]
        return valid_actions

    def predict_action(self, ratio_factor, sent_idx, valid_actions):
        stack_emb = self.stack.get_output(sent_idx)
        buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
            else self.buffer.get_output(sent_idx)
        action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
            else self.action_stack.get_output(sent_idx)
        p_t = torch.cat([buffer_emb, stack_emb, action_emb])
        p_t = torch.cat([p_t, ratio_factor])
        h = torch.tanh(self.p_s2h(p_t))
        h = torch.cat([h, ratio_factor])
        valid_action_idx = [self.vocab.get_token_index(a, namespace='actions') for a in valid_actions]
        logits = self.p_act(h)[torch.tensor(valid_action_idx, dtype=torch.long, device=h.device)]
        valid_action_tbl = {a: i for i, a in enumerate(valid_action_idx)}
        log_probs = torch.log_softmax(logits, dim=0)
        action_idx = torch.max(log_probs, 0)[1].item()
        action_idx = valid_action_idx[action_idx]
        action = self.vocab.get_token_from_index(action_idx, namespace='actions')
        return action, action_idx, log_probs, valid_action_tbl

    def exec_action(self, action, action_sequence_length, action_tag_for_terminate, edge_list, generated_order,
                    head_count, swap_count, null_node, num_of_generated_node, oracle_actions, ratio_factor,
                    ratio_factor_losses, sent_idx, sent_len, total_node_num, reachable: List[Dict[int, Set[int]]]):
        # generate null node, recursive way
        if action.startswith("NODE"):  # Support legacy models with "NODE:*" actions that also create an edge
            null_node_token = len(null_node[sent_idx]) + sent_len[sent_idx] + 1
            null_node[sent_idx].append(null_node_token)
            head_count[sent_idx][null_node_token] = swap_count[sent_idx][null_node_token] = 0
            reachable[sent_idx][null_node_token] = {null_node_token}

            stack_emb = self.stack.get_output(sent_idx)

            stack_emb = torch.cat([stack_emb, ratio_factor])
            comp_rep = torch.tanh(self.update_null_node(stack_emb))

            node_input = comp_rep

            self.buffer.push(sent_idx,
                             input=node_input,
                             extra={'token': null_node_token})

            total_node_num[sent_idx] = sent_len[sent_idx] + len(null_node[sent_idx])
        # Support legacy models with "NODE:*" actions that also create an edge
        if action.startswith("NODE:") or action.startswith("LEFT-EDGE") or action.startswith("RIGHT-EDGE"):
            if action.startswith("NODE:"):
                logger.warning(f"Took action {action}")
                modifier = self.buffer.get_stack(sent_idx)[-1]
                head = self.stack.get_stack(sent_idx)[-1]
            elif action.startswith("LEFT-EDGE"):
                head = self.stack.get_stack(sent_idx)[-1]
                modifier = self.stack.get_stack(sent_idx)[-2]
            else:
                head = self.stack.get_stack(sent_idx)[-2]
                modifier = self.stack.get_stack(sent_idx)[-1]

            (head_rep, head_tok) = (head['stack_rnn_output'], head['token'])
            (mod_rep, mod_tok) = (modifier['stack_rnn_output'], modifier['token'])

            if oracle_actions is None:
                edge_list[sent_idx].append((mod_tok,
                                            head_tok,
                                            action.split(':', maxsplit=1)[1]))
            # propagate reachability
            reachable_from_mod = reachable[sent_idx][mod_tok]
            for tok, reachable_for_tok_set in reachable[sent_idx].items():
                if head_tok in reachable_for_tok_set:
                    reachable_for_tok_set |= reachable_from_mod

            action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
                else self.action_stack.get_output(sent_idx)

            stack_emb = self.pempty_stack_emb if self.stack.get_len(sent_idx) == 0 \
                else self.stack.get_output(sent_idx)

            buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                else self.buffer.get_output(sent_idx)

            # # compute composed representation
            comp_rep = torch.cat([head_rep, mod_rep, action_emb, buffer_emb, stack_emb, ratio_factor])
            comp_rep = torch.tanh(self.p_comp(comp_rep))

            if action.startswith("NODE:"):
                self.buffer.pop(sent_idx)
                self.buffer.push(sent_idx,
                                 input=comp_rep,
                                 extra={'token': mod_tok})
            elif action.startswith("LEFT-EDGE"):
                self.stack.pop(sent_idx)
                self.stack.push(sent_idx,
                                input=comp_rep,
                                extra={'token': head_tok})
            # RIGHT-EDGE
            else:
                stack_0_rep = self.stack.get_stack(sent_idx)[-1]['stack_rnn_input']
                self.stack.pop(sent_idx)
                self.stack.pop(sent_idx)

                self.stack.push(sent_idx,
                                input=comp_rep,
                                extra={'token': head_tok})

                self.stack.push(sent_idx,
                                input=stack_0_rep,
                                extra={'token': mod_tok})
            head_count[sent_idx][mod_tok] += 1

        # Execute the action to update the parser state
        elif action == "REDUCE-0":
            self.stack.pop(sent_idx)

        elif action == "REDUCE-1":
            stack_0 = self.stack.pop(sent_idx)
            self.stack.pop(sent_idx)
            self.stack.push(sent_idx,
                            input=stack_0['stack_rnn_input'],
                            extra={'token': stack_0['token']})

        elif action == "SHIFT":
            buffer = self.buffer.pop(sent_idx)
            self.stack.push(sent_idx,
                            input=buffer['stack_rnn_input'],
                            extra={'token': buffer['token']})
            s0 = self.stack.get_stack(sent_idx)[-1]['token']
            if s0 not in generated_order[sent_idx]:
                num_of_generated_node[sent_idx] = len(generated_order[sent_idx])
                generated_order[sent_idx][s0] = num_of_generated_node[sent_idx]
                head_count[sent_idx][s0] = swap_count[sent_idx][s0] = 0
                reachable[sent_idx][s0] = {s0}

        elif action == "SWAP":
            stack_penult = self.stack.pop_penult(sent_idx)
            self.buffer.push(sent_idx,
                             input=stack_penult['stack_rnn_input'],
                             extra={'token': stack_penult['token']})
            swap_count[sent_idx][stack_penult['token']] += 1

        elif action == "FINISH":
            action_tag_for_terminate[sent_idx] = True
            ratio_factor_losses[sent_idx] = ratio_factor
        action_sequence_length[sent_idx] += 1

    def fix_unconnected_egraph(self, edge_list, reachable, root_id, generated_order):
        """
        Detect cycles in edge_list, select arbitrary element, attach to root's dependent.
        Postprocessing to avoid validation errors (validate.py --level 2): L2 Enhanced unconnected-egraph
        Since REDUCE-{0,1} has a precondition requiring a head, the only way to get an unconnect graph
        (which means there are nodes unreachable from the root) is if there are cycles.
        """
        # Find the dependent of root since it will be the head of all orphans
        pred = None
        for mod_tok, head_tok, label in edge_list:
            if head_tok == root_id:
                pred = mod_tok
                break
        orphans = set(generated_order) - reachable[root_id]  # All nodes not reachable from the root
        while orphans:
            tok = max(orphans, key=lambda x: len(reachable[x]))  # Start from nodes with the most descendents: "roots"
            if pred is None:
                # print(f"No predicate (dependent of root) found: {edge_list}", file=sys.stderr)
                edge_list.append((tok, root_id, "root"))
                pred = tok
            else:
                edge_list.append((tok, pred, "orphan"))
            orphans -= reachable[tok]  # All cycle members were taken care of by doing this

    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def forward(self,
                words: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                ) -> Dict[str, torch.LongTensor]:

        batch_size = len(metadata)
        sent_len = [len(d['words']) for d in metadata]

        #oracle_actions = None
        if gold_actions is not None:
            oracle_actions  = deepcopy([d['gold_actions'] for d in metadata])

        embedded_text_input = self.text_field_embedder(words)
        embedded_text_input = self._input_dropout(embedded_text_input)

        if self.training:
            try:
                ret_train = self._greedy_decode(batch_size=batch_size,
                                                sent_len=sent_len,
                                                embedded_text_input=embedded_text_input,
                                                oracle_actions=oracle_actions)
            except IndexError:
                raise IndexError(f"{[d['words'] for d in metadata]}")

            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            return output_dict
        else:
            #reset
            if self.num_validation_instances and (not self._total_validation_instances or self._total_validation_instances == self.num_validation_instances):
                self._total_validation_instances = batch_size
            else:
                self._total_validation_instances += batch_size

        #print(f'{self._total_validation_instances}/{self.num_validation_instances}')
        training_mode = self.training
        self.eval()
        with torch.no_grad():
            ret_eval = self._greedy_decode(batch_size=batch_size,
                                           sent_len=sent_len,
                                           embedded_text_input=embedded_text_input)

        self.train(training_mode)

        edge_list = ret_eval['edge_list']
        null_node = ret_eval['null_node']

        _loss = ret_eval['loss']

        # prediction-mode
        output_dict = {
                'edge_list': edge_list,
                'null_node': null_node,
                'loss': _loss
        }

        for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
                        "deprel", "misc"]:
                output_dict[k] = [[token_metadata[k] for token_metadata in sentence_metadata['annotation']] for sentence_metadata in metadata]

        output_dict["multiwords"] = [sentence_metadata['multiwords'] for sentence_metadata in metadata]
        output_dict["sent_id"] = [sentence_metadata['sent_id'] for sentence_metadata in metadata]
        output_dict["text"] = [sentence_metadata['text'] for sentence_metadata in metadata]
        # validation mode
        # compute the accuracy when gold actions exist
        if gold_actions is not None:
            predicted_graphs = []
            gold_graphs_conllu = []

            for sent_idx in range(batch_size):
                predicted_graphs.append(eud_trans_outputs_into_conllu({
                        k:output_dict[k][sent_idx] for k in ["id", "form", "lemma", "upostag", "xpostag", "feats", "head",
                                "deprel", "misc", "edge_list", "null_node", "multiwords", "text", "sent_id"]
                }, self.output_null_nodes))
                gold_annotation = [{key: output_dict[key][sent_idx] for key in ("sent_id", "text")
                    if output_dict[key][sent_idx]}]
                for annotation in metadata[sent_idx]['annotation']:
                    gold_annotation.append(annotation)
                gold_graphs_conllu.append(annotation_to_conllu(gold_annotation, self.output_null_nodes))

            predicted_graphs_conllu = [line for lines in predicted_graphs for line in lines]
            gold_graphs_conllu = [line for lines in gold_graphs_conllu for line in lines]

            self._xud_score(predicted_graphs_conllu,
                            gold_graphs_conllu)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._xud_score is not None and not self.training:
            if self.num_validation_instances and self._total_validation_instances == self.num_validation_instances:
                all_metrics.update(self._xud_score.get_metric(reset=reset))
            elif not self.num_validation_instances:
                all_metrics.update(self._xud_score.get_metric(reset=True))
        return all_metrics
