import logging
from typing import Dict, Optional, Any, List

import torch
from allennlp.data import Vocabulary
from allennlp.models import SimpleTagger
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure
from torch.nn.modules import Dropout

from modules import StackRnn, SimpleTagger
from utils import eud_trans_outputs_into_conllu

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
                 lemma_text_field_embedder: TextFieldEmbedder = None,
                 pos_tag_embedding: Embedding = None,
                 action_embedding: Embedding = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None
                 ) -> None:

        super(TransitionParser, self).__init__(vocab, regularizer)

        self._unlabeled_correct = 0
        self._labeled_correct = 0
        self._total_edges_predicted = 0
        self._total_edges_actual = 0
        self._exact_unlabeled_correct = 0
        self._exact_labeled_correct = 0
        self._total_sentences = 0

        self.num_actions = vocab.get_vocab_size('actions')
        self.text_field_embedder = text_field_embedder
        self.pos_tag_embedding = pos_tag_embedding
        #this is most probably incorrect
        self._f1 = F1Measure(positive_label=1)


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

        self.update_concept_node = torch.nn.Linear(self.hidden_dim + self.ratio_dim, self.word_dim)

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
            oracle_actions: Optional[List[List[int]]] = None,
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
        # push the tokens onto the buffer (tokens is in reverse order)
        for token_idx in range(max(sent_len)):
            for sent_idx in range(batch_size):
                if sent_len[sent_idx] > token_idx:
                    self.buffer.push(sent_idx,
                            input=embedded_text_input[sent_idx][sent_len[sent_idx] - 1 - token_idx],
                            extra={'token': sent_len[sent_idx] - token_idx - 1})

                    # init stack using proot_emb, considering batch
        for sent_idx in range(batch_size):
            self.stack.push(sent_idx,
                    input=self.proot_stack_emb,
                    extra={'token': sent_len[sent_idx]})
            ret_node[sent_idx] = [sent_len[sent_idx]]

        action_id = {
                action_: [self.vocab.get_token_index(a, namespace='actions') for a in
                    self.vocab.get_token_to_index_vocabulary('actions').keys() if a.startswith(action_)]
                for action_ in
                ["SHIFT", "REDUCE", "NODE", "LEFT-EDGE", "RIGHT-EDGE", "SWAP", "FINISH", "-E-"]
                }

        # compute probability of each of the actions and choose an action
        # either from the oracle or if there is no oracle, based on the model
        trans_not_fin = True

        action_tag_for_terminate = [False] * batch_size
        action_sequence_length = [0] * batch_size

        null_node = {}
        for sent_idx in range(batch_size):
            null_node[sent_idx] = [sent_len[sent_idx]]

        while trans_not_fin:
            trans_not_fin = False
            for sent_idx in range(batch_size):
                if (len(null_node[sent_idx]) > 50 * sent_len[sent_idx] or action_sequence_length[sent_idx] > 50 *
                        sent_len[sent_idx]) and oracle_actions is None:
                    raise RuntimeError(f"Too many actions for a sentence {sent_idx}")
                total_node_num[sent_idx] = sent_len[sent_idx] + len(null_node[sent_idx])
                # if self.buffer.get_len(sent_idx) != 0:
                if action_tag_for_terminate[sent_idx] == False:
                    trans_not_fin = True
                    #not sure it is right but fixes the key error l230
                    valid_actions = action_id['-E-']
                    # given the buffer and stack, conclude the valid action list
                    if self.buffer.get_len(sent_idx) == 0:
                        valid_actions += action_id['FINISH']

                    if self.buffer.get_len(sent_idx) > 0:
                        valid_actions += action_id['SHIFT']

                    try:
                        if self.stack.get_len(sent_idx) > 0:
                            valid_actions += action_id['REDUCE']
                            valid_actions += action_id['NODE']
                    except:
                        pass
                    if self.stack.get_len(sent_idx) > 1:
                        valid_actions += action_id['SWAP']
                        valid_actions += action_id['LEFT-EDGE']
                        valid_actions += action_id['RIGHT-EDGE']

                    log_probs = None
                    action = valid_actions[0]
                    ratio_factor = torch.tensor([len(null_node[sent_idx]) / (1.0 * sent_len[sent_idx])],
                            device=self.pempty_action_emb.device)

                    if len(valid_actions) > 1:
                        stack_emb = self.stack.get_output(sent_idx)
                        buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                                else self.buffer.get_output(sent_idx)

                        action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
                                else self.action_stack.get_output(sent_idx)

                        p_t = torch.cat([buffer_emb, stack_emb, action_emb])
                        p_t = torch.cat([p_t, ratio_factor])

                        h = torch.tanh(self.p_s2h(p_t))
                        h = torch.cat([h, ratio_factor])

                        logits = self.p_act(h)[torch.tensor(valid_actions, dtype=torch.long, device=h.device)]
                        valid_action_tbl = {a: i for i, a in enumerate(valid_actions)}
                        log_probs = torch.log_softmax(logits, dim=0)

                        action_idx = torch.max(log_probs, 0)[1].item()
                        action = valid_actions[action_idx]

                    if oracle_actions is not None:
                        action = oracle_actions[sent_idx].pop(0)

                    # push action into action_stack
                    self.action_stack.push(sent_idx,
                            input=self.action_embedding(
                                torch.tensor(action, device=embedded_text_input.device)),
                            extra={
                                'token': self.vocab.get_token_from_index(action, namespace='actions')})
                    action_list[sent_idx].append(self.vocab.get_token_from_index(action, namespace='actions'))

                    if log_probs is not None:
                        losses[sent_idx].append(log_probs[valid_action_tbl[action]])

                    # generate concept node, recursive way
                    if action in action_id["NODE"] :
                        null_node_token = len(null_node[sent_idx]) + sent_len[sent_idx]
                        null_node[sent_idx].append(null_node_token)

                        stack_emb = self.stack.get_output(sent_idx)

                        stack_emb = torch.cat([stack_emb, ratio_factor])
                        comp_rep = torch.tanh(self.update_null_node(stack_emb))

                        node_input = comp_rep

                        self.buffer.push(sent_idx,
                                input=node_input,
                                extra={'token': null_node_token})

                        total_node_num[sent_idx] = sent_len[sent_idx] + len(null_node[sent_idx])

                    if action in action_id["NODE"] + action_id["LEFT-EDGE"] \
                            + action_id["RIGHT-EDGE"] :

                        if action in action_id["NODE"] :
                            head = self.buffer.get_stack(sent_idx)[-1]
                            modifier = self.stack.get_stack(sent_idx)[-1]

                        elif action in action_id["LEFT-EDGE"] :
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
                                self.vocab.get_token_from_index(action, namespace='actions')
                                .split(':', maxsplit=1)[1]))

                            # # compute composed representation

                        action_emb = self.pempty_action_emb if self.action_stack.get_len(sent_idx) == 0 \
                                else self.action_stack.get_output(sent_idx)

                        stack_emb = self.pempty_stack_emb if self.stack.get_len(sent_idx) == 0 \
                                else self.stack.get_output(sent_idx)

                        buffer_emb = self.pempty_buffer_emb if self.buffer.get_len(sent_idx) == 0 \
                                else self.buffer.get_output(sent_idx)

                        comp_rep = torch.cat([head_rep, mod_rep, action_emb, buffer_emb, stack_emb, ratio_factor])
                        comp_rep = torch.tanh(self.p_comp(comp_rep))

                        if action in action_id["NODE"] :
                            self.buffer.pop(sent_idx)
                            self.buffer.push(sent_idx,
                                    input=comp_rep,
                                    extra={'token': head_tok})


                        elif action in action_id["LEFT-EDGE"] :
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

                    # Execute the action to update the parser state
                    if action in action_id["REDUCE"]:
                        self.stack.pop(sent_idx)

                    elif action in action_id["SHIFT"]:
                        buffer = self.buffer.pop(sent_idx)
                        self.stack.push(sent_idx,
                                input=buffer['stack_rnn_input'],
                                extra={'token': buffer['token']})

                    elif action in action_id["SWAP"]:
                        stack_penult = self.stack.pop_penult(sent_idx)
                        self.buffer.push(sent_idx,
                                input=stack_penult['stack_rnn_input'],
                                extra={'token': stack_penult['token']})

                    elif action in action_id["FINISH"]:
                        action_tag_for_terminate[sent_idx] = True
                        ratio_factor_losses[sent_idx] = ratio_factor

                    action_sequence_length[sent_idx] += 1

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
        ret['node'] = ret_node
        ret["null_node"] = ret_node

        return ret
    # Returns an expression of the loss for the sequence of actions.
    # (that is, the oracle_actions if present or the predicted sequence otherwise)
    def forward(self,
                words: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]],
                gold_actions: Dict[str, torch.LongTensor] = None,
                #pos_tag: torch.LongTensor = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                enhanced_arc_tags: torch.LongTensor=None,
                ) -> Dict[str, torch.LongTensor]:

        batch_size = len(metadata)
        sent_len = [len(d['words']) for d in metadata]

        oracle_actions = None
        if gold_actions is not None:
            oracle_actions = [d['gold_actions'] for d in metadata]
            oracle_actions = [[self.vocab.get_token_index(s, namespace='actions') for s in l] for l in oracle_actions]

        embedded_text_input = self.text_field_embedder(words)
        embedded_text_input = self._input_dropout(embedded_text_input)

        if self.training:
            ret_train = self._greedy_decode(batch_size=batch_size,
                                            sent_len=sent_len,
                                            embedded_text_input=embedded_text_input,
                                            oracle_actions=oracle_actions)

            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            return output_dict

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
            'words': [d['words'] for d in metadata],
            'edge_list': edge_list,
            'null_node': null_node,
            'ids': [list(range(1,len(d['words'])+1)) for d in metadata],
            'loss': _loss
        }

        # prediction-mode
        if gold_actions is not None:
            gold_graphs = [x["gold_graphs"] for x in metadata]
            predicted_graphs = []

            for sent_idx in range(batch_size):
                    predicted_graphs.append(eud_trans_outputs_into_conllu({
                        'words': output_dict['words'][sent_idx],
                        'edge_list': output_dict['edge_list'][sent_idx],
                        'meta_info': output_dict['meta_info'][sent_idx],
                        'pos_tag': output_dict['pos_tag'][sent_idx],
                        'ids': output_dict['ids'][sent_idx],
                    }))

            self._f1(edge_list,
                    gold_graphs)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._f1 is not None and not self.training:
            all_metrics.update(self._f1.get_metric(reset=reset))
        return all_metrics