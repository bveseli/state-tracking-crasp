import dfa
from dfa import generate_dfa_diagram
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import networkx as nx
from sympy import Matrix
import matplotlib.pyplot as plt

def nullspaces_equal(A, B):
    ns_A = A.nullspace()
    ns_B = B.nullspace()

    if len(ns_A) != len(ns_B):
        return False
    if len(ns_A) == 0:
        return True

    NA = Matrix.hstack(*ns_A)
    NB = Matrix.hstack(*ns_B)

    r = NA.rank()
    return Matrix.hstack(NA, NB).rank() == r

def automata_to_graph(my_dfa):
    # converts a dfa into a networkx graph
    # note that we use a MultiDiGraph
    states = my_dfa.states
    transitions = my_dfa.transitions
    G = nx.MultiDiGraph()
    state_list = list(states)
    state_mapping = {}
    for i in range(len(state_list)):
        state_mapping[state_list[i]] = i
    for q in states:
        for sym in transitions[q].keys():
            G.add_edge(state_mapping[q], state_mapping[transitions[q][sym]], key=sym, label=sym)
    return G

def show_graph(G):
    # show the graph
    # does not disambiguate multi-edges
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = data.get('label', '')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def show_graph_as_DFA(G):
    # show the graph as a DFA
    # disambiguates multi-edges by showing all labels on the same edge
    # edges may be labeled by tuples
    # convert to an automata dot format for visualization
    # display it and wait for the user to close it before proceeding
    dot_str = 'digraph G {\n'
    for u, v, data in G.edges(data=True):
        label = data.get('label', '')
        if isinstance(label, tuple):
            label = ','.join(map(str, label))
        dot_str += f'    {u} -> {v} [label="{label}"];\n'
    dot_str += '}'
    from graphviz import Source
    src = Source(dot_str)
    src.render('drawings/dfa_graph', format='svg', cleanup=True)
    src.view()
    # wait for the user to close the graph before proceeding
    input("Press Enter to continue...")

def get_path_edges(G, path):
    # path is a list of nodes in a multigraph
    # recursively computes all paths through the multigraph through the nodes in order
    # returns a list of lists of edges (n1, n2, symbol)
    if len(path) < 2:
        return []
    edges = []
    first = path[0]
    rest = path[1:]
    successors = list(G.successors(first))
    for succ in successors:
        if succ == rest[0]:
            for sym in G[first][succ]:
                label = G[first][succ][sym]['label']
                if isinstance(label, Matrix):
                    label = tuple(label)
                if len(rest) == 1:
                    edges.append([(first, succ, label)])
                else:
                    sub_paths = get_path_edges(G, rest)
                    for sp in sub_paths:
                        edges.append([(first, succ, label)] + sp)
    return edges

def get_simple_cycles_edges(G):
    # computes all simple cycles in G
    # returns a list of lists of edges (n1, n2, symbol)
    cycles = []
    for cycle in nx.simple_cycles(G):
        edges = get_path_edges(G, cycle + [cycle[0]])
        cycles.extend(edges)
    return cycles

def loop_equations(ordered_alphabet, cycles):
    # to maintain a balanced morphism, all loops must sum to 0
    # each loop defines a linear equation based on its Parikh image
    # we will then find the nullspace of this set of linear equations
    # this gives the basis of all balanced morphisms
    basis = []
    for cycle in cycles:
        vec = [0]*len(ordered_alphabet)
        for _,_,sym in cycle:
            index = ordered_alphabet.index(sym)
            vec[index] += 1
        basis.append(vec)
    return basis

def relabel(G, cycles, morphism):
    # morphism is a dict for relabling symbols
    # the labels will be vectors
    # choose an arbitrary start node
    # for each edge n1 -> n2 with label sym
    # pick a path from start to n1
    # the new label is the sum of morphism[sym]
    # and the sum of morphism[label] for edges in the path
    graph_copy = G.copy()
    start_node = list(graph_copy.nodes())[0]
    for u, v, data in G.edges(data=True):
        path = nx.shortest_path(G, source=start_node, target=u)
        path_edges = get_path_edges(G, path)
        label = data['label']
        vec = morphism[label].copy()
        if path_edges:
            # if there are paths, sum up the labels under the morphism
            # take the first (they all must end up the same)
            path_edges = path_edges[0]
            for e in path_edges:
                vec += morphism[e[2]]
        # update the label of the u -> v edge in the copy of the graph
        graph_copy[u][v][label]['label'] = tuple(vec)
    G.clear()
    for u, v, key, data in graph_copy.edges(data=True, keys=True):
        if isinstance(key, str):
            new_label = data['label']
        else:
            new_label = (data['label'], key)
        G.add_edge(u, v, label=new_label, key=new_label)


def separated(G):
    # separated nodes are those where the set of outgoing edge labels are different
    # check that all pairs of nodes are separated
    # do not consider the 'garbage' node
    nodes = [n for n in G.nodes() if n != 'garbage']
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1 = nodes[i]
            n2 = nodes[j]
            labels_n1 = set()
            for _, v, data in G.out_edges(n1, data=True):
                labels_n1.add(data['label'])
            labels_n2 = set()
            for _, v, data in G.out_edges(n2, data=True):
                labels_n2.add(data['label'])
            if labels_n1 == labels_n2:
                return False
    return True

def get_ordered_alphabet(G):
    labels = []
    for u in sorted(G.nodes(), key=str):
        for u, v, data in G.out_edges(u, data=True):
            label = data['label']
            if label not in labels:
                labels.append(label)
    return labels

def attack_scc(G):
    # attack an SCC (to decide membership in C-RASP)
    temp_G = G.copy()
    prev_basis = None
    # the algorithm converges once the null spaces don't change
    while True:
        # find all balanced morphisms using the nullspace of the loop equations
        ordered_alphabet = get_ordered_alphabet(temp_G)
        cycles = get_simple_cycles_edges(temp_G)
        basis = Matrix(loop_equations(ordered_alphabet, cycles))
        # print(f"Current basis: {basis}")
        if prev_basis is not None and nullspaces_equal(basis, prev_basis):
            # converged
            break
        if basis.nullspace() == []:
            # no more nontrivial balanced morphisms
            break
        else:
            prev_basis = basis
            # relabel the graph using the morphism defined by the nullspace basis
            morphism = {}
            for i in range(len(ordered_alphabet)):
                vec = []
                for nb in basis.nullspace():
                    vec.append(nb[i])
                morphism[ordered_alphabet[i]] = Matrix(vec)
            # apply the morphism to relabel the graph
            relabel(temp_G, cycles, morphism)
    # after convergence, return whether the morphism separates the nodes
    return separated(temp_G)

def decide_CRASP_membership(my_dfa):
    # decide membership in C-RASP by iterating over the SCCs in the dfa
    # the entire dfa is in C-RASP iff every SCC is in C-RASP
    G = automata_to_graph(my_dfa)
    sccs = nx.strongly_connected_components(G)
    for component in sccs:
        # duplicate component
        temp_G = G.subgraph(component).copy()
        garbage_node = 'garbage'
        temp_G.add_node(garbage_node)
        # print(f"Attacking SCC with nodes: {temp_G.nodes()} and edges: {[(u, v, data['label']) for u, v, data in temp_G.edges(data=True)]}")
        # make sure that every node has an outgoing edge for every symbol in the alphabet (add a garbage node for missing edges)
        # but don't add self loops to the garbage node
        alphabet = set()
        for _, _, data in temp_G.edges(data=True):
            label = data['label']
            label = tuple(label) if isinstance(label, Matrix) else label
            alphabet.add(label)
        for node in temp_G.nodes():
            for sym in alphabet:
                # Do not add self loops to the garbage node
                if node == garbage_node:
                    continue
                if not any((node, _, data) for _, _, data in temp_G.out_edges(node, data=True) if data['label'] == sym):
                    temp_G.add_edge(node, garbage_node, label=sym, key=sym)
        # show the graph of the SCC being attacked
        result = attack_scc(temp_G)
        if not result:
            return False
    return True

def decide_CRASP_membership_from_regex(regex_str):
    nfa = NFA.from_regex(regex_str.replace('+', '|'))
    my_dfa = DFA.from_nfa(nfa, minify=True)
    return decide_CRASP_membership(my_dfa)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decide C-RASP membership for a regular language.")
    parser.add_argument("--regex", type=str, nargs="+", required=True, help="One or more regular expressions (use + for union, e.g. '(cab+c)*')")
    parser.add_argument("--draw", action="store_true", help="Generate and open a DFA diagram")
    args = parser.parse_args()

    for regex_str in args.regex:
        print("regex:", regex_str)
        nfa = NFA.from_regex(regex_str.replace('+', '|'))
        my_dfa = DFA.from_nfa(nfa, minify=True)

        if args.draw:
            generate_dfa_diagram(my_dfa, filename='drawings/my_dfa', output_format='svg', auto_open=False)
            G = automata_to_graph(my_dfa)
            plt.show()

        membership = decide_CRASP_membership(my_dfa)
        print(f"C-RASP membership: {membership}")