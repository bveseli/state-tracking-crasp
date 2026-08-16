import argparse
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from automata.fa.dfa import DFA
from automata.fa.nfa import NFA

rng = random.Random(42)


def _build_count_table(dfa, L):
    trans = dfa.transitions
    finals = set(dfa.final_states)
    outgoing = {s: list(mp.items()) for s, mp in trans.items()}
    counts = [defaultdict(int) for _ in range(L + 1)]

    for s in trans.keys():
        counts[0][s] = 1 if s in finals else 0

    for k in range(1, L + 1):
        ck = counts[k]
        cprev = counts[k - 1]
        for s in trans.keys():
            total = 0
            for _, t in outgoing.get(s, []):
                total += cprev[t]
            ck[s] = total
    return counts, outgoing


def sample_accepted_words(dfa, L, n, weighted=True):
    counts, outgoing = _build_count_table(dfa, L)
    start = dfa.initial_state

    if counts[L][start] == 0:
        return []

    results = []
    for _ in range(n):
        state = start
        word = []
        for step in range(L):
            remaining = L - step - 1
            choices = [(sym, nxt) for sym, nxt in outgoing.get(state, []) if counts[remaining][nxt] > 0]
            if weighted:
                weights = [counts[remaining][nxt] for _, nxt in choices]
                sym, nxt = rng.choices(choices, weights=weights, k=1)[0]
            else:
                sym, nxt = rng.choice(choices)

            word.append(sym)
            state = nxt
        if dfa.accepts_input("".join(word)):
            results.append("".join(word))
    return list(set(results))


def save_formal_language_info(regex, dfa, output_dir, bin_ranges):
    data = {
        "regex": regex,
        "states": ["q" + str(s) for s in dfa.states],
        "symbols": list(dfa.input_symbols),
        "bin_ranges_lengths": bin_ranges,
    }
    with open(f"{output_dir}/meta_data.json", "w") as f:
        json.dump(data, f)


def save_as_jsonl(data_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_languages(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    if "name" not in colmap or "regex" not in colmap:
        raise ValueError(f"Missing columns in {csv_path}: need 'name' and 'regex', found {list(df.columns)}")
    return df.rename(columns={colmap["name"]: "name", colmap["regex"]: "regex"})


def folder_name(name: str) -> str:
    return f"lang_id_{name}" if name.isdigit() else name


def resolve_tasks(df: pd.DataFrame, task_names: list[str]):
    names = [t.strip() for t in task_names if t and t.strip()]
    rows = [(str(r["name"]), str(r["regex"])) for _, r in df.iterrows()]
    if not names or "all" in names:
        return rows

    requested = set()
    for name in names:
        requested.add(name)
        if name.startswith("lang_id_") and name[len("lang_id_"):].isdigit():
            requested.add(name[len("lang_id_"):])

    selected = [(name, regex) for name, regex in rows if name in requested or folder_name(name) in requested]
    found = {name for name, _ in selected}
    found.update(folder_name(name) for name in found)
    missing = sorted(n for n in names if n not in found)
    if missing:
        available = ", ".join(sorted({folder_name(name) for name, _ in rows}))
        raise ValueError(f"Unknown task(s): {missing}. Available: {available}")
    return selected


def valid_lengths_up_to(dfa, max_len: int):
    counts, _ = _build_count_table(dfa, max_len)
    start = dfa.initial_state
    return [L for L in range(max_len + 1) if counts[L][start] > 0]


def _upsample(samples, target_size):
    if not samples:
        return []
    if len(samples) >= target_size:
        return random.sample(samples, target_size)
    out = list(samples)
    out.extend(random.choices(samples, k=target_size - len(samples)))
    return out


def _dedup_samples(samples):
    seen = set()
    uniq = []
    for s in samples:
        key = (s["word"], s.get("states"), s.get("final_states"))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


def get_test_train_split(final_data, train_size=None, test_size=None):
    train_test_data = {}
    bin_ranges = list(final_data.keys())
    bin_ranges.sort(key=min)

    for idx, bin_range in enumerate(bin_ranges):
        uniq_samples = _dedup_samples(final_data[bin_range])
        random.shuffle(uniq_samples)

        if idx == 0:
            split_idx = int(len(uniq_samples) * 0.8)
            train_uniq = uniq_samples[:split_idx]
            test_uniq = uniq_samples[split_idx:]
            train_out = _upsample(train_uniq, train_size) if train_size else train_uniq
            test_out = _upsample(test_uniq, test_size) if test_size else test_uniq
            train_test_data[f"train_{bin_range[0]}-{bin_range[1]}"] = train_out
            train_test_data[f"test_{bin_range[0]}-{bin_range[1]}"] = test_out
        else:
            test_out = _upsample(uniq_samples, test_size) if test_size else uniq_samples
            train_test_data[f"test_{bin_range[0]}-{bin_range[1]}"] = test_out

    return train_test_data


def main():
    parser = argparse.ArgumentParser(description="Generate datasets with concatenated traces.")
    parser.add_argument("--task", nargs="*", default=["all"], help="Task names from the languages CSV. Use 'all' or pass one/many names.")
    parser.add_argument("--languages_csv", type=str, required=True, help="Path to the languages CSV")
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument(
        "--bins",
        type=str,
        default="[(0, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300), (301, 350), (351, 400), (401, 450), (451, 500)]",
        help="Python list of (lo, hi) tuples. First-bin lo 0 means the DFA minimum word length.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory for generated datasets")
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Use uniform-over-transitions sampling (biased over words). Default is weighted/uniform over accepted words.",
    )
    args = parser.parse_args()
    weighted = not args.unweighted

    languages_csv = Path(args.languages_csv)
    output_root = Path(args.output)
    bins = ast.literal_eval(args.bins)
    train_len = bins[0][1]
    suite = languages_csv.stem.replace("_language_suite", "")
    suite_root = output_root / f"{suite}_train{train_len}"

    failed_tasks = []
    langs = load_languages(languages_csv)
    tasks = resolve_tasks(langs, args.task)

    for csv_name, regex in tasks:
        skip_task = False
        task_name = folder_name(csv_name)
        output = suite_root / task_name
        output.mkdir(parents=True, exist_ok=True)

        print(f"[{task_name}] Compiling Regex: {regex}")
        print(f"[{task_name}] Sampling mode: {'weighted (uniform over words)' if weighted else 'unweighted (uniform over transitions)'}")
        nfa = NFA.from_regex(regex)
        dfa = DFA.from_nfa(nfa).minify()
        if "_complement" in task_name:
            dfa = dfa.complement()
            print("Complemented DFA")

        actual_min = dfa.minimum_word_length()
        print(f"Detected Minimum Word Length: {actual_min}")

        bin_ranges = []
        for idx, (lo, hi) in enumerate(bins):
            s = actual_min if idx == 0 else lo
            all_valid = valid_lengths_up_to(dfa, hi)
            lengths_per_range = [L for L in all_valid if s <= L <= hi and L > 0]
            bin_ranges.append(lengths_per_range)

        all_seen_words = set()
        all_seen_s_traces = set()
        final_data = {}

        for idx, bin_range in enumerate(bin_ranges):
            min_len, max_len = bins[idx]
            bin_samples = []
            for length in bin_range:
                words = sample_accepted_words(
                    dfa,
                    length,
                    args.train_size if idx == 0 else args.test_size,
                    weighted=weighted,
                )
                if not words:
                    raise Exception(f"No words sampled for length {length} within range {(min_len, max_len)}")

                for w in words:
                    if w in all_seen_words:
                        continue
                    s_trace = "q" + "q".join(
                        [str(state) for state in list(dfa.read_input_stepwise(w, ignore_rejection=True))]
                    )
                    if s_trace in all_seen_s_traces:
                        continue
                    bin_samples.append({"word": w, "states": s_trace})
                    all_seen_words.add(w)
                    all_seen_s_traces.add(s_trace)

            if len(bin_samples) == 0:
                print(f"No samples for task {task_name}, bin {idx}; skipping this task.")
                skip_task = True
                break

            final_data[(min_len, max_len)] = bin_samples
            print("Bin samples: ", (min_len, max_len), "length of bin samples: ", len(final_data[(min_len, max_len)]))

        if skip_task:
            failed_tasks.append(task_name)
            continue

        train_test_data = get_test_train_split(
            final_data,
            train_size=args.train_size,
            test_size=args.test_size,
        )

        save_formal_language_info(regex, dfa, output, bin_ranges)
        for bin_range, samples in train_test_data.items():
            save_as_jsonl(samples, f"{output}/{bin_range}.jsonl")
        print(f"\nSuccess! Saved to {output}")

    if failed_tasks:
        failed_file = suite_root / "languages_with_empty_bins.txt"
        failed_file.parent.mkdir(parents=True, exist_ok=True)
        with failed_file.open("w", encoding="utf-8") as f:
            for name in failed_tasks:
                f.write(name + "\n")
        print(f"Failed tasks: {failed_tasks}; wrote {failed_file}")


if __name__ == "__main__":
    main()
