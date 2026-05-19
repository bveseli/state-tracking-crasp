import argparse
import random
import re
from automata.fa.nfa import NFA
from automata.fa.dfa import DFA
from pathlib import Path
from collections import defaultdict
from utils.utils import DatasetUtils
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
            results.append("".join(word)) #double check if the word is accepted by the DFA
    return list(set(results)) # Unique words only



def valid_lengths_up_to(dfa, max_len: int):
    counts, _ = _build_count_table(dfa, max_len)
    start = dfa.initial_state
    return [L for L in range(max_len + 1) if counts[L][start] > 0]


def _upsample(samples, target_size):
    if not samples:
        return []
    if len(samples) >= target_size:
        return rng.sample(samples, target_size)
    # keep split disjointness: only sample from within this split
    out = list(samples)
    out.extend(rng.choices(samples, k=target_size - len(samples)))
    return out



def get_test_train_split(final_data, train_size=None, test_size=None):
    train_test_data = {}

    bin_ranges = list(final_data.keys())
    bin_ranges.sort(key=min)

    for idx, bin_range in enumerate(bin_ranges):
        random.shuffle(uniq_samples)

        if idx == 0:
            # in-distribution split from unique pool
            split_idx = int(len(uniq_samples) * 0.8)
            train_uniq = uniq_samples[:split_idx]
            test_uniq = uniq_samples[split_idx:]

            # 2) upsample inside each split (no cross-split leakage)
            train_out = _upsample(train_uniq, train_size) if train_size else train_uniq
            test_out = _upsample(test_uniq, test_size) if test_size else test_uniq

            train_test_data[f"train_{bin_range[0]}-{bin_range[1]}"] = train_out
            train_test_data[f"test_{bin_range[0]}-{bin_range[1]}"] = test_out
        else:
            # OOD bins are test-only; optional upsample
            test_out = _upsample(uniq_samples, test_size) if test_size else uniq_samples
            train_test_data[f"test_{bin_range[0]}-{bin_range[1]}"] = test_out

    return train_test_data

def main():
    parser = argparse.ArgumentParser(description="Generate datasets with concatenated traces.")
    parser.add_argument("--task", nargs="*", default=["all"], help="Task names from languages.csv. Use 'all' or pass one/many names.")
    parser.add_argument("--languages_csv", type=str, default="../../languages.csv")
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument( "--bins", type=str, default="(min,50),(51,100),(101,150)", help="e.g. (min,20),(21,50)")
    parser.add_argument("--output", type=str, default="../../datas/")
    args = parser.parse_args()

    failed_tasks = []

    langs = DatasetUtils.load_languages(args.languages_csv)
    tasks = DatasetUtils.resolve_tasks(langs, args.task)
        
        
    bin_ranges = []
    matches = re.findall(r'\((min|\d+),\s*(\d+)\)', args.bins)

    for task_name, regex in tasks:

        skip_task = False

        output = Path(args.output) / f"n{args.train_size}-trainlen{matches[0][1]}" / task_name
        output.mkdir(parents=True, exist_ok=True)

        print(f"[{task_name}] Compiling Regex: {regex}")

        nfa = NFA.from_regex(regex)
        dfa = DFA.from_nfa(nfa).minify()
        if "_complement" in task_name:
            dfa = dfa.complement()
            print("Complemented DFA")

        actual_min = dfa.minimum_word_length()
        print(f"Detected Minimum Word Length: {actual_min}")

        for start_val, end_val in matches:
            s = actual_min if start_val == "min" else int(start_val)
            e = int(end_val)
            all_valid = valid_lengths_up_to(dfa, e)
            lengths_per_range = [L for L in all_valid if s <= L <= e and L > 0]

            bin_ranges.append(lengths_per_range)
        

        all_seen_words = set()
        final_data = {}

        for idx, bin_range in enumerate(bin_ranges):
            bin_samples = []
            for length in bin_range:

                words = sample_accepted_words(dfa, length, args.train_size if idx == 0 else args.test_size)
                if words:
                    for w in words:
                        if w not in all_seen_words:

                            s_trace = "q"+"q".join([str(state) for state in list(dfa.read_input_stepwise(w, ignore_rejection=True))])

                            bin_samples.append({
                                "word": w, 
                                "states": s_trace, 
                            })

                            all_seen_words.add(w)
                else:
                    raise Exception(f"No words sampled for length {length}")

            if len(bin_samples) == 0:
                print(f"No samples for task {task_name}, bin {idx}; skipping this task.")
                skip_task = True
                break  # break out of the bin loop

            min_len,max_len = 0 if idx==0 else int(matches[idx][0]),int(matches[idx][1])
            final_data[(min_len,max_len)] = bin_samples
            print("Bin samples: ", (min_len,max_len), "length of bin samples: ", len(final_data[(min_len,max_len)]))

        if skip_task:
            failed_tasks.append(task_name)
            continue

        train_test_data = get_test_train_split(
            final_data,
            train_size=args.train_size,   
            test_size=args.test_size      
        )

        DatasetUtils.save_formal_language_info(regex, dfa, output, bin_ranges)

        for bin_range, samples in train_test_data.items():
            DatasetUtils.save_as_jsonl(samples, f'{output}/{bin_range}.jsonl')
        print(f"\nSuccess! Saved to {output}")

    # After processing all tasks, save failed ones
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}; No samples for these tasks")

if __name__ == "__main__":
    main()

