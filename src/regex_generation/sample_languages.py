import argparse
import pandas as pd
import random

FILES = ["r.txt", "r_infinity_not_crasp.txt", "crasp.txt", "not_r_infinity.txt"]


def sample_languages(results_dir: str, n: int, seed: int, output: str):
    rng = random.Random(seed)
    rows = []

    for fname in FILES:
        stem = fname.replace(".txt", "")
        df = pd.read_csv(f"{results_dir}/{fname}")

        sample = rng.sample(range(len(df)), min(n, len(df)))
        for i, idx in enumerate(sample):
            row = df.iloc[idx]
            regex_raw = str(row["regex"])
            rows.append({
                "Name": f"{i}_{stem}",
                "Formal Language": regex_raw,
                "Regex": regex_raw.replace("+", "|"),
                "R": str(row["R"]),
                "C-RASP": str(row["C-RASP"]),
                "R_infinity": str(row["R_infinity"]),
            })

    out_df = pd.DataFrame(rows, columns=["Name", "Formal Language", "Regex", "R", "C-RASP", "R_infinity"])
    out_df.to_csv(output, sep=";", index=False)
    print(f"Saved {len(out_df)} languages to {output}")


if __name__ == "__main__":
    """
    Sample languages from the results of the language generation process.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--n", type=int, required=True, help="Number of languages to sample per file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="../../languages.csv")
    args = parser.parse_args()

    sample_languages(args.results_dir, args.n, args.seed, args.output)
