import dfa
from dfa import generate_dfa_diagram
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import networkx as nx
from sympy import Matrix
import matplotlib.pyplot as plt
import decider as d
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("regex", type=str)
    args = parser.parse_args()

    regex_str = ''.join(args.regex)
    print("regex:", regex_str)
    nfa = NFA.from_regex(regex_str)
    my_dfa = DFA.from_nfa(nfa, minify=True)

    generate_dfa_diagram(my_dfa, filename='drawings/my_dfa', output_format='svg', auto_open=False)
    G = d.automata_to_graph(my_dfa)
    plt.show()

    membership = d.decide_CRASP_membership(my_dfa)
    print(f"CRASP membership: {membership}")

