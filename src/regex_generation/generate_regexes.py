import pysemigroup
import itertools
from graphviz import Source
import csv
import dfa
from dfa import generate_dfa_diagram
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import networkx as nx
from sympy import Matrix
import matplotlib.pyplot as plt
import decider as d
import argparse
import random
from typing import Dict, List
import csv
import os
from pysemigroup.ring import hash_matrix
import sys

def _patched_hash(self):
    try:
        self._hash = hash(self.tobytes())       # NumPy 2.0+
    except AttributeError:
        self._hash = hash(self.tostring())      # NumPy < 2.0 fallback
    return self._hash

hash_matrix.__hash__ = _patched_hash

def _patched_automaton(self):
    if self._regex == "":
        return pysemigroup.Automaton.from_empty_string(self.letters())
    # Build explicit namespace dict instead of relying on exec() locals
    local_vars = {
        letter: pysemigroup.Automaton.from_letter(letter, alphabet=self.letters())
        for letter in self.letters()
    }
    local_vars['_empty_word'] = pysemigroup.Automaton.from_empty_string(self.letters())
    local_vars['_star'] = "_star"
    return eval(self._regex, {}, local_vars)

pysemigroup.RegularLanguage.automaton = _patched_automaton


class CFG:
    """Context-Free Grammar sampler with customizable rule probabilities."""

    def __init__(self, rules: Dict[str, List[List[str]]], start_symbol: str = 'S', rule_probs: Dict[str, List[float]] = None):
        """
        rules: dict mapping nonterminals to list of productions (each a list of symbols)
        rule_probs: dict mapping nonterminals to list of probabilities (same length as rules[nonterminal])
        """
        self.rules = rules
        self.start_symbol = start_symbol
        self.rule_probs = rule_probs or {}

        # Validate probabilities if provided
        for nt, prods in self.rules.items():
            if nt in self.rule_probs:
                probs = self.rule_probs[nt]
                if len(probs) != len(prods):
                    raise ValueError(f"Probabilities for {nt} do not match number of productions")
                if not abs(sum(probs) - 1.0) < 1e-8:
                    raise ValueError(f"Probabilities for {nt} must sum to 1.0")

    def sample(self, symbol: str = None, max_depth: int = 50) -> str:
        if symbol is None:
            symbol = self.start_symbol

        if symbol not in self.rules:
            return symbol

        if max_depth <= 0:
            # When depth exhausted, force terminal-only productions
            terminal_prods = [p for p in self.rules[symbol]
                              if all(s not in self.rules for s in p)]
            if terminal_prods:
                production = random.choice(terminal_prods)
            else:
                # If no pure terminal production, pick shortest
                production = min(self.rules[symbol], key=len)
            # Expand remaining symbols with depth=0 (will force terminals)
            result = []
            for sym in production:
                if sym not in self.rules:
                    result.append(sym)
                else:
                    result.append(self.sample(sym, 0))
            return ''.join(result)

        prods = self.rules[symbol]
        if symbol in self.rule_probs:
            probs = self.rule_probs[symbol]
            idx = random.choices(range(len(prods)), weights=probs)[0]
            production = prods[idx]
        else:
            production = random.choice(prods)

        result = []
        for sym in production:
            expanded = self.sample(sym, max_depth - 1)
            result.append(expanded)

        return ''.join(result)

    def sample_n(self, n: int, symbol: str = None) -> List[str]:
        return [self.sample(symbol) for _ in range(n)]


regex_rules = {
    # Regex with + for alternation and * for Kleene star
    'R': [
        ['R', '+', 'R'],
        ['R', 'R'],
        ['(', 'R', ')', '*'],
        ['A']
    ],
    # Atoms (base symbols)
    'A': [
        ['a'],
        ['b'],
        ['c']
    ]
}

def check_R(semigroup):
    for x in semigroup.elements():
        R = semigroup.R_class_of_element(x)
        if len(R) > 1:
            return False
    return True

def check_R_infinity(semigroup):
    # print("Syntactic monoid elements:", list(semigroup.elements()))
    # dot = semigroup.graphviz_string()
    # # replace print(dot) with:
    # Source(dot).view()  # writes a temp file and opens the default viewer

    # print("Elements of the semigroup:", list(semigroup.elements()))
    # print("Idempotents of the semigroup:", list(semigroup.idempotents()))

    two_idempotents_in_the_same_R_class = False
    for el1, el2 in itertools.combinations(list(semigroup.idempotents()), 2):
        if semigroup.R_class_of_element(el1) == semigroup.R_class_of_element(el2):
            two_idempotents_in_the_same_R_class = True
            # print(f"Two idempotents in the same R-class: {el1}, {el2}")
            break
    # check aperiodic
    is_aperiodic = semigroup.is_Ap()
    return two_idempotents_in_the_same_R_class and is_aperiodic

if __name__ == "__main__":
    alphabet = {'a', 'b', 'c'}
    # # generate
    n = 2000
    # # Set probabilities for each nonterminal
    # regex_rule_probs = {
    #     'R': [0.1, 0.6, 0.2, 0.1],  # Adjust probabilities for R productions
    #     'A': [0.4, 0.4, 0.2]
    # }

    # cfg = CFG(regex_rules, 'R', rule_probs=regex_rule_probs)
    # os.makedirs('data', exist_ok=True)
    # os.makedirs('results', exist_ok=True)

    data_path = 'data/regex.txt'
    # output_rows = []
    # i = 0
    # while len(output_rows) < n:
    #     regex = cfg.sample(max_depth=3)
    #     if regex not in [row[1] for row in output_rows] and '*' in regex:
    #         output_rows.append([len(output_rows) + 1, regex])
    #     i += 1

    # with open(data_path, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in output_rows:
    #         writer.writerow(row)

    # classify into four bins (each of which is contained in the next):
    #  - R
    #  - C-RASP
    #  - R_infinity but not C-RASP
    #  - not in R_infinity

    input_file_path = data_path
    r_path = 'results/r.txt'
    crasp_path = 'results/crasp.txt'
    r_infty_not_crasp_path = 'results/r_infinity_not_crasp.txt'
    not_r_infty_path = 'results/not_r_infinity.txt'
    all_results_path = 'results/all_results.txt'
    errors_path = 'results/classify_errors.txt'


    counts = {'r': 0, 'crasp': 0, 'r_infty_not_crasp': 0, 'not_r_infty': 0, 'errors': 0}

    with open(errors_path, 'w', newline='') as err_file, \
         open(input_file_path, 'r', newline='') as csvfile, \
         open(r_path, 'w', newline='') as r_file, \
         open(crasp_path, 'w', newline='') as crasp_file, \
         open(r_infty_not_crasp_path, 'w', newline='') as rinf_file, \
         open(all_results_path, 'w', newline='') as all_file, \
         open(not_r_infty_path, 'w', newline='') as notr_file:

        reader = csv.reader(csvfile)
        r_writer = csv.writer(r_file)
        crasp_writer = csv.writer(crasp_file)
        rinf_writer = csv.writer(rinf_file)
        notr_writer = csv.writer(notr_file)
        all_writer = csv.writer(all_file)
        err_writer = csv.writer(err_file)

        # header rows (optional)
        r_writer.writerow(["index", "regex", "R", "C-RASP", "R_infinity"])
        crasp_writer.writerow(["index", "regex", "R", "C-RASP", "R_infinity"])
        rinf_writer.writerow(["index", "regex", "R", "C-RASP", "R_infinity"])
        notr_writer.writerow(["index", "regex", "R", "C-RASP", "R_infinity"])
        all_writer.writerow(["index", "regex", "R", "C-RASP", "R_infinity"])
        err_writer.writerow(["index", "regex", "error"])


        for row in reader:
            try:
                idx = row[0]
                regex_str = row[1]

                language = pysemigroup.RegularLanguage.from_easy_regex(regex_str, alphabet)

                # get the syntactic monoid of the language
                semigroup = language.syntactic_monoid()

                # check R membership first (since R is a subset of C-RASP)
                try:
                    r_membership = check_R(semigroup)
                except Exception as e:
                    err_writer.writerow([idx, f"R check error: {e}"])
                    counts['errors'] += 1
                    continue

                # check C-RASP membership
                try:
                    # build NFA/DFA
                    # note that we use + for alternation in the regex rules, but the NFA.from_regex expects |, so we replace it here
                    nfa = NFA.from_regex(regex_str.replace('+', '|'))
                    my_dfa = DFA.from_nfa(nfa, minify=True)
                    crasp_membership = d.decide_CRASP_membership(my_dfa)
                except Exception as e:
                    # If decider fails, record error and skip
                    err_writer.writerow([idx, f"C-RASP decider error: {e}"])
                    counts['errors'] += 1
                    continue

                # check R_infinity
                try:
                    r_infty = check_R_infinity(semigroup)
                except Exception as e:
                    err_writer.writerow([idx, f"R_infinity check error: {e}"])
                    counts['errors'] += 1
                    continue

                # binning
                if r_membership:
                    r_writer.writerow([idx, regex_str, True, crasp_membership, r_infty])
                    counts['r'] += 1
                elif crasp_membership:
                    crasp_writer.writerow([idx, regex_str, r_membership, True, r_infty])
                    counts['crasp'] += 1
                elif r_infty:
                    rinf_writer.writerow([idx, regex_str, r_membership, crasp_membership, True])
                    counts['r_infty_not_crasp'] += 1
                else:
                    notr_writer.writerow([idx, regex_str, r_membership, crasp_membership, r_infty])
                    counts['not_r_infty'] += 1

                all_writer.writerow([idx, regex_str, r_membership, crasp_membership, r_infty])

            except Exception as e:
                # catch any unexpected parsing/IO error for the row
                err_writer.writerow(row + [f"unexpected error: {e}"])
                counts['errors'] += 1

    # write a small summary
    summary_path = 'results/classify_summary.txt'
    with open(summary_path, 'w') as sf:
        sf.write(f"Total generated: {n}\n")
        sf.write(f"R: {counts['r']}\n")
        sf.write(f"C-RASP: {counts['crasp']}\n")
        sf.write(f"R_infinity but not C-RASP: {counts['r_infty_not_crasp']}\n")
        sf.write(f"Not R_infinity: {counts['not_r_infty']}\n")
        sf.write(f"Errors: {counts['errors']}\n")