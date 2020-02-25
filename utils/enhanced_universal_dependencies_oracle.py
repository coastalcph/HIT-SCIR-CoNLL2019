def get_oracle_actions(token_ids, arc_indices, arc_tags, null_node_ids, node_ids):
    root_id='0'
    actions = []
    stack = [root_id]
    buffer = []
    null_node_ids = {i: False for i in null_node_ids}
    generated_order = {root_id: 0}

    for i in reversed(token_ids):
        buffer.append(i)

    graph = {}
    for token_idx in node_ids:
        graph[token_idx] = []

    # construct graph given directed_arc_indices and arc_tags
    # key: id_of_point
    # value: a list of tuples -> [(id_of_head1, label),(id_of_head2, label)，...]
    whole_graph = {j:{i:False for i in node_ids} for j in node_ids}
    for arc, arc_tag in zip(arc_indices, arc_tags):
        graph[arc[0]].append((arc[1], arc_tag))
        whole_graph[arc[0]][arc[1]] = True
    # i:head_point j:child_point›
    top_down_graph = {i:[] for i in node_ids}  # N real point, 1 root point, null_node_ids

    # i:child_point j:head_point ->Bool
    # partial graph during construction
    sub_graph = {j:{i:False for i in node_ids} for j in node_ids}
    sub_graph_arc_list = []

    for i in node_ids:
        for head_tuple_of_point_i in graph[i]:
            head = head_tuple_of_point_i[0]
            top_down_graph[head].append(i)

    def has_find_head(w0):
        if w0 ==-1:
            return False
        for node_info in graph[w0]:
            if sub_graph[w0][node_info[0]] == True:
                return True
        return False

    # return if w1 is one head of w0
    def has_head(w0, w1):
        if w0 == -1 or w1 ==-1:
            return False
        for w in graph[w0]:
            if w[0] == w1:
                return True
        return False

    def has_unfound_child(w):
        for child in top_down_graph[w]:
            if not sub_graph[child][w]:
                return True
        return False

    # return if w has any unfound head
    def lack_head(w):
        if w ==-1:
            return False
        head_num = 0
        for h in sub_graph[w]:
            if h:
                head_num += 1
        if head_num < len(graph[w]):
            return True
        return False

    # return the relation between child: w0, head: w1
    def get_arc_label(w0, w1):
        for h in graph[w0]:
            if h[0] == w1:
                return h[1]

    def get_node_id(w0):
        """
        return True only if find the new head+concept node of w0
        """
        if w0 ==-1:
            return -1
        for head_node_info_of_w0 in graph[w0]:
            head_node_id = head_node_info_of_w0[0]
            if sub_graph[w0][head_node_id] == False and head_node_id ==-1:
                if null_node_ids[head_node_id] == True:
                    return -1
                return head_node_id
        return -1

    def check_graph_finish():
        return whole_graph == sub_graph

    def check_sub_graph(w0, w1):
        if w0 ==-1 or w1 ==-1:
            return False
        else:
            return sub_graph[w0][w1] == False

    def get_oracle_actions_onestep(sub_graph, stack, buffer, actions):

        s0 = stack[-1] if len(stack) > 0 else -1
        s1 = stack[-2] if len(stack) > 1 else -1

        # RIGHT_EDGE
        if s0 != -1 and has_head(s0, s1) and check_sub_graph(s0, s1):
            actions.append("RIGHT-EDGE:" + get_arc_label(s0, s1))
            sub_graph[s0][s1] = True
            sub_graph_arc_list.append((s0, s1))
            return

            # LEFT_EDGE
        elif s1 != root_id and has_head(s1, s0) and check_sub_graph(s1, s0):
            actions.append("LEFT-EDGE:" + get_arc_label(s1, s0))
            sub_graph[s1][s0] = True
            sub_graph_arc_list.append((s1, s0))
            return

        # NODE
        elif s0 != root_id and get_node_id(s0) != -1 and has_head(s0, get_node_id(
            s0)) and not has_find_head(s0):
            null_node_id = get_node_id(s0)
            buffer.append(null_node_id)

            actions.append("NODE:" + get_arc_label(s0, null_node_id))

            null_node_ids[null_node_id] = True
            sub_graph[s0][null_node_id] = True
            sub_graph_arc_list.append((s0, null_node_id))

            return

            # REDUCE
        elif s0 != -1 and not has_unfound_child(s0) and not lack_head(s0):
            actions.append("REDUCE")
            stack.pop()
            return

            # SWAP
        elif len(stack) > 2 and generated_order[s0] > generated_order[s1] and (
                has_unfound_child(stack[-3]) or lack_head(stack[-3])):
            buffer.append(stack.pop(-2))
            actions.append("SWAP")
            return

            # SHIFT
        elif len(buffer) != 0:

            if buffer[-1] not in generated_order:
                num_of_generated_node = len(generated_order)
                generated_order[buffer[-1]] = num_of_generated_node

            stack.append(buffer.pop())
            actions.append("SHIFT")
            return

        else:
            remain_unfound_edge = set(arc_indices) - set(sub_graph_arc_list)
            actions.append('-E-')
            return

    while not (check_graph_finish() and len(buffer) == 0):
        get_oracle_actions_onestep(sub_graph, stack, buffer, actions)
        assert len(actions) <10000
        if actions[-1] == '-E-':
            break

    actions.append('FINISH')
    stack.pop()

    return actions


