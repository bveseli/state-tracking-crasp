import argparse
import json
import re
from pathlib import Path


def parse_arch(arch_str: str):
    """Parse '2l4h64d' -> (n_layer, n_head, d_model) tuple for size comparison."""
    m = re.fullmatch(r'(\d+)l(\d+)h(\d+)d', arch_str.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def make_arch_key(arch_str: str, lr: float) -> str:
    """Append 'smalllr' suffix when lr == 1e-4."""
    suffix = 'smalllr' if round(lr, 6) == round(1e-4, 6) else ''
    return arch_str + suffix


def parse_line(line: str):
    """
    Parse a single log line into (arch_str, lr, metrics).
    Example line:
        1l1h64d  >>early stop 24.0  eval_len0-50_acc: 1.0  ...  lr: 0.001
    Returns None if the line cannot be parsed.
    """
    line = line.strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split('\t\t')]
    if len(parts) < 3:
        return None

    arch_str = parts[0]
    if not re.fullmatch(r'\d+l\d+h\d+d', arch_str):
        return None

    lr_match = re.search(r'lr:\s*([\d.e+-]+)', parts[-1])
    if not lr_match:
        return None
    lr = float(lr_match.group(1))

    metrics = {}
    for part in parts[2:-1]:
        for m in re.finditer(r'(eval_len(\d+)-(\d+)_acc):\s*([\d.]+)', part):
            metrics[m.group(1)] = float(m.group(4))

    return arch_str, lr, metrics


def sorted_bins(metrics: dict):
    """Return bin keys sorted by upper bound descending, e.g. eval_len451-500_acc first."""
    bins = []
    for key in metrics:
        m = re.match(r'eval_len\d+-(\d+)_acc', key)
        if m:
            bins.append((int(m.group(1)), key))
    bins.sort(reverse=True)
    return [key for _, key in bins]


def select_best(runs: list):
    """
    Select the best run from a list of (arch_str, lr, metrics) tuples.

    Strategy:
      - Starting from the highest length bin, find all runs with acc == 1.0.
      - If multiple, pick the smallest architecture (layers → heads → d_model).
      - If none reach 1.0 on the highest bin, move to the next lower bin and repeat.
      - Returns None if no run achieves 1.0 on any bin.
    """
    if not runs:
        return None

    for bin_key in sorted_bins(runs[0][2]):
        candidates = [
            (arch_str, lr, metrics)
            for arch_str, lr, metrics in runs
            if metrics.get(bin_key, 0.0) == 1.0
        ]
        if candidates:
            best = min(candidates, key=lambda x: parse_arch(x[0]) or (999, 999, 999))
            return make_arch_key(best[0], best[1])

    return None


def collect_runs(task_dir: Path, nope: bool = False):
    """Read .txt files in task_dir and return a list of parsed runs.
    If nope=True, only files with 'nope' in the filename are read.
    """
    runs = []
    for txt_file in sorted(task_dir.glob('*.txt')):
        if nope and 'nope' not in txt_file.name:
            continue
        with open(txt_file, encoding='utf-8') as f:
            for line in f:
                result = parse_line(line)
                if result:
                    runs.append(result)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Select the best hyperparameter configuration per task and save to JSON."
    )
    parser.add_argument(
        '--results_dir', type=str, default='../../results/hyperparam_search',
        help='Root directory with per-task subdirectories of .txt log files'
    )
    parser.add_argument(
        '--output', type=str, default='../../best_hparams.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--tasks', nargs='*', default=None,
        help='Tasks to process (default: all subdirectories in results_dir)'
    )
    parser.add_argument(
        '--nope', action='store_true',
        help='Only read .txt files with "nope" in the filename'
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if args.tasks:
        task_dirs = [results_dir / t for t in args.tasks]
    else:
        task_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())

    best_hparams = {}

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            print(f"[{task_dir.name}] Directory not found, skipping.")
            continue

        task_name = task_dir.name
        runs = collect_runs(task_dir, nope=args.nope)

        if not runs:
            print(f"[{task_name}] No runs found.")
            continue

        best = select_best(runs)
        if best:
            best_hparams[task_name] = best
            print(f"[{task_name}] Best: {best}")
        else:
            print(f"[{task_name}] No run achieved 1.0 accuracy on any bin.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(best_hparams, f, indent=2)
    print(f"\nSaved {len(best_hparams)} entries to {output_path}")


if __name__ == '__main__':
    main()
