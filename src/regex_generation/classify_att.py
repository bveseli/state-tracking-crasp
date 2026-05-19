"""
classify_att.py
---------------
Read a folder of .att DFA files and classify each language into:
  - R          : all R-classes in the syntactic monoid are singletons
  - C-RASP     : decided by decider.decide_CRASP_membership
  - R_infinity : no two idempotents share an R-class

Output CSV columns: id, name, R, C-RASP, R_infinity

Usage:
    python classify_att.py <att_dir> <output_csv>
"""

import os
import sys
import csv
import copy
import itertools

import pysemigroup
from pysemigroup import Automaton as PySemiAutomaton, TransitionSemiGroup
from pysemigroup.ring import hash_matrix
from automata.fa.dfa import DFA
import decider as d


# ---------------------------------------------------------------------------
# Patch hash_matrix for NumPy compatibility (from existing codebase)
# ---------------------------------------------------------------------------
def _patched_hash(self):
    try:
        self._hash = hash(self.tobytes())       # NumPy 2.0+
    except AttributeError:
        self._hash = hash(self.tostring())      # NumPy < 2.0
    return self._hash

hash_matrix.__hash__ = _patched_hash


# ---------------------------------------------------------------------------
# .att parser
# ---------------------------------------------------------------------------
_DEAD = '__dead__'

def parse_att(filepath: str):
    """
    Parse an OpenFst .att file (tab-separated, deterministic acceptor).

    Returns
    -------
    initial_state : str
    final_states  : set[str]
    arcs          : dict[str, dict[str, str]]   src -> {symbol -> dest}
    """
    arcs: dict = {}
    final_states: set = set()
    initial_state = None

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 4:                     # arc line
                src, dest, ilabel, _olabel = parts
                if initial_state is None:
                    initial_state = src
                arcs.setdefault(src, {})[ilabel] = dest
            elif len(parts) == 1 and parts[0].strip():   # final-state line
                final_states.add(parts[0].strip())

    return initial_state, final_states, arcs


def complete_dfa(initial_state, final_states, arcs):
    """
    Return a *complete* DFA by adding an explicit dead/sink state for any
    missing (state, symbol) transitions.  The input arcs dict is not mutated.

    Returns
    -------
    initial_state, final_states, arcs (completed copy), alphabet, states
    """
    # Collect alphabet and states
    alphabet: set = set()
    states: set = set()
    if initial_state:
        states.add(initial_state)
    states |= final_states
    for src, trans in arcs.items():
        states.add(src)
        for sym, dest in trans.items():
            alphabet.add(sym)
            states.add(dest)

    # Check whether any transition is missing
    needs_dead = any(
        arcs.get(state, {}).get(sym) is None
        for state in states
        for sym in alphabet
    )

    arcs = copy.deepcopy(arcs)   # don't modify the caller's dict

    if needs_dead:
        states.add(_DEAD)
        arcs.setdefault(_DEAD, {})
        for sym in alphabet:                       # dead loops to itself
            arcs[_DEAD][sym] = _DEAD
        for state in list(states):
            if state == _DEAD:
                continue
            for sym in alphabet:
                arcs.setdefault(state, {}).setdefault(sym, _DEAD)

    return initial_state, final_states, arcs, alphabet, states


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def to_pysemi_automaton(initial_state, final_states, arcs, alphabet, states):
    """Build a pysemigroup.Automaton from completed DFA data."""
    transitions = {}
    for src, trans in arcs.items():
        for sym, dest in trans.items():
            transitions[(src, sym)] = [dest]
    return PySemiAutomaton(
        transitions=transitions,
        initial_states=[initial_state],
        final_states=list(final_states),
        states=set(states),
        alphabet=set(alphabet),
    )


def to_automata_dfa(initial_state, final_states, arcs, alphabet, states):
    """Build an automata.fa.dfa.DFA from completed DFA data."""
    transitions = {state: {} for state in states}
    for src, trans in arcs.items():
        for sym, dest in trans.items():
            transitions[src][sym] = dest
    return DFA(
        states=set(states),
        input_symbols=set(alphabet),
        transitions=transitions,
        initial_state=initial_state,
        final_states=set(final_states),
    )


# ---------------------------------------------------------------------------
# Classification helpers (mirrored from existing codebase)
# ---------------------------------------------------------------------------

def check_R(semigroup) -> bool:
    """True iff every R-class in the syntactic monoid is a singleton."""
    for x in semigroup.elements():
        if len(semigroup.R_class_of_element(x)) > 1:
            return False
    return True


def check_R_infinity(semigroup) -> bool:
    """True iff no two distinct idempotents share an R-class."""
    idempotents = list(semigroup.idempotents())
    for e1, e2 in itertools.combinations(idempotents, 2):
        if semigroup.R_class_of_element(e1) == semigroup.R_class_of_element(e2):
            return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def classify_folder(input_dir: str, output_csv: str):
    att_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.att'))
    if not att_files:
        print(f"No .att files found in '{input_dir}'.")
        return

    print(f"Found {len(att_files)} .att files — classifying...")

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = len(att_files)
    counts = {'R': 0, 'C-RASP': 0, 'R_inf': 0, 'errors': 0}

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'name', 'R', 'C-RASP', 'R_infinity'])

        # just check the first 100 for a sanity check before doing the whole batch  
        for i, filename in enumerate(att_files[:100], 1):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(input_dir, filename)
            r_mem = crasp_mem = r_inf = 'ERROR'

            try:
                initial, finals, raw_arcs = parse_att(filepath)
                initial, finals, arcs, alphabet, states = complete_dfa(
                    initial, finals, raw_arcs
                )
                # ── Syntactic monoid (pysemigroup) → R and R_infinity ──────
                try:
                    psa = to_pysemi_automaton(initial, finals, arcs, alphabet, states)
                    sg  = TransitionSemiGroup(psa)
                    r_mem = check_R(sg)
                    r_inf = check_R_infinity(sg)
                    if r_mem:   counts['R']     += 1
                    if r_inf:   counts['R_inf'] += 1
                except Exception as e:
                    r_mem = r_inf = f'ERROR: {e}'
                    counts['errors'] += 1

                # ── automata.fa.dfa.DFA → C-RASP ──────────────────────────
                try:
                    auto_dfa  = to_automata_dfa(initial, finals, arcs, alphabet, states)
                    crasp_mem = d.decide_CRASP_membership(auto_dfa)
                    if crasp_mem: counts['C-RASP'] += 1
                except Exception as e:
                    crasp_mem = f'ERROR: {e}'
                    counts['errors'] += 1

            except Exception as e:
                r_mem = crasp_mem = r_inf = f'ERROR: {e}'
                counts['errors'] += 1

            writer.writerow([i, name, r_mem, crasp_mem, r_inf])

            if i % 50 == 0 or i == total:
                print(
                    f"  [{i:>4}/{total}] {name}"
                    f"  R={str(r_mem):<5}  C-RASP={str(crasp_mem):<5}  R∞={r_inf}"
                )

    print(f"\nDone. Output written to '{output_csv}'.")
    print(f"  R: {counts['R']}  |  C-RASP: {counts['C-RASP']}  |  "
          f"R∞: {counts['R_inf']}  |  Errors: {counts['errors']}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    classify_folder(sys.argv[1], sys.argv[2])