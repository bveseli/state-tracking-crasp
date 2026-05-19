import subprocess
from typing import Optional, Dict, Set
from automata.fa.dfa import DFA

def generate_dfa_diagram(
    dfa,
    filename: str = 'dfa_diagram',
    output_format: str = 'svg',
    auto_open: bool = True,
    transition_labels: Optional[Dict] = None,
    rankdir: str = 'LR'
) -> str:
    """
    Generate a Graphviz diagram for a DFA with optional customization.

    Args:
        dfa: The DFA object from automata-lib
        filename: Output filename (without extension)
        output_format: Output format ('svg', 'png', 'pdf', etc.)
        auto_open: Whether to automatically open the generated diagram
        transition_labels: Dict for custom transition labels (advanced)
        rankdir: Graph direction ('LR' for left-to-right, 'TB' for top-to-bottom)

    Returns:
        Path to the generated file
    """

    dot_string = _build_dot_string(
        dfa,
        transition_labels=transition_labels,
        rankdir=rankdir
    )

    # Write DOT file
    dot_file = f'{filename}.dot'
    with open(dot_file, 'w') as f:
        f.write(dot_string)

    # Generate output file
    output_file = f'{filename}.{output_format}'
    subprocess.run(['dot', f'-T{output_format}', dot_file, '-o', output_file])

    # Open if requested
    if auto_open:
        subprocess.run(['open', output_file])

    print(f"Diagram saved to {output_file}")
    return output_file


def _build_dot_string(
    dfa,
    transition_labels: Optional[Dict] = None,
    rankdir: str = 'LR'
) -> str:
    """Build the DOT format string for the DFA."""

    dot_lines = [
        'digraph DFA {',
        f'rankdir={rankdir};',
        'node [shape=circle];'
    ]

    # Mark final states
    for state in dfa.states:
        if state in dfa.final_states:
            shape = 'doublecircle'
        else:
            shape = 'circle'

        dot_lines.append(f'{state} [shape={shape}];')

    # Add transitions
    for from_state, transitions in dfa.transitions.items():
        for symbol, to_state in transitions.items():
            dot_lines.append(f'{from_state} -> {to_state} [label="{symbol}"];')

    # Mark initial state
    dot_lines.append('_start [shape=point, label=""];')
    dot_lines.append(f'_start -> {dfa.initial_state};')

    dot_lines.append('}')

    return '\n'.join(dot_lines)

my_dfa = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'},
    input_symbols={'a', 'b'},
    transitions={
        'q0': {'a': 'q1', 'b': 'q2'},
        'q1': {'a': 'q6', 'b': 'q0'},
        'q2': {'a': 'q6', 'b': 'q3'},
        'q3': {'a': 'q4', 'b': 'q6'},
        'q4': {'a': 'q5', 'b': 'q6'},
        'q5': {'a': 'q6', 'b': 'q2'},
        'q6': {'a': 'q6', 'b': 'q6'}
    },
    initial_state='q0',
    final_states={'q3'}
)

def get_dot_string(dfa, rankdir: str = 'LR') -> str:
    """Get the DOT format string without generating a file."""
    return _build_dot_string(dfa, rankdir=rankdir)
