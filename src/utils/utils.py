import json
import pandas as pd

class DatasetUtils:

    @staticmethod
    def save_as_jsonl(data_list, filename):
        """
        Saves a list of dictionaries into a .jsonl file.
        Each dictionary becomes a single line in the file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in data_list:
                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + '\n')
            
    @staticmethod
    def load_languages(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, sep=";")
        df.columns = [c.strip() for c in df.columns]
        required = {"Name", "Regex"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in languages.csv: {missing}")
        return df

    @staticmethod
    def resolve_tasks(df: pd.DataFrame, task_names: list[str]):
        names = [t.strip() for t in task_names if t and t.strip()]
        if not names or "all" in names:
            return [(r["Name"], r["Regex"]) for _, r in df.iterrows()]

        requested = set(names)
        selected = df[df["Name"].isin(requested)]

        found = set(selected["Name"].astype(str).tolist())
        missing = sorted(requested - found)
        if missing:
            available = ", ".join(sorted(df["Name"].astype(str).tolist()))
            raise ValueError(f"Unknown task(s): {missing}. Available: {available}")

        return [(r["Name"], r["Regex"]) for _, r in selected.iterrows()]


    @staticmethod
    def save_formal_language_info(
        regex,
        dfa,
        output_dir,
        bin_ranges
    ):
        data = {}
        data["regex"] = regex
        data["states"] = ["q"+str(s) for s in dfa.states]
        data["symbols"] = list(dfa.input_symbols)
        data["bin_ranges_lengths"] = bin_ranges

        with open(f"{output_dir}/meta_data.json", "w") as f:
            json.dump(data, f)


