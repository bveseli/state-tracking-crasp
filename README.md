# Algebraic Decomposition Theory for Transformer Length Generalization

This repository corresponds to the paper "Algebraic Decomposition Theory for Transformer Length Generalization" by Yang et al. 2026.

## Pipeline: Step-by-Step

### Step 1 — Languages

You need a `languages.csv` file listing the regular languages to experiment on. You have two options:

**Option A: Use the provided languages** (recommended to reproduce paper results)

A pre-built `languages.csv` is included in the repository root. Skip to Step 2.

**Option B: Generate your own languages**

First, generate and classify 2000 random regular expressions over `{a, b, c}`:
```bash
cd scripts && sbatch generate_languages.sh
```
This classifies each regex into **R**, **C-RASP**, and **R∞** and saves the results to `src/regex_generation/results/`.

Then sample `n` languages from each class:
```bash
sbatch sample_languages.sh
# or directly:
python src/regex_generation/sample_languages.py --n 20 --output languages.csv
```

---

### Step 2 — Datasets

You need word datasets for each language. Again, two options:

**Option A: Use the provided datasets**

Pre-built datasets are included under `datasets/`. Skip to Step 3.

**Option B: Generate datasets from `languages.csv`**

For each language, `create_words.py` samples accepted words at different lengths, records the DFA state trace for each word, and writes JSONL files grouped by length bin.

```bash
cd scripts && sbatch create_words.sh
# or directly:
python src/data/create_words.py \
  --languages_csv languages.csv \
  --output datasets/ \
  --bins "(min,50),(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(401,450),(451,500)"
```

#### Length bins and sampling defaults

Bins are passed via `--bins` as comma-separated ranges, e.g. `(min,50),(51,100),...`. The **first bin** sets the **train length range**; every **later bin** is a **test-only** length range for out-of-distribution (OOD) length generalization evaluation.

| Bin | Role | Default sampling | Output files |
|-----|------|------------------|--------------|
| First (e.g. `(min,50)`) | Train length range | Up to `--train_size` words (default **10,000**), then **80/20** train / in-distribution test | `train_{min}-{max}.jsonl`, `test_{min}-{max}.jsonl` |
| Remaining bins | OOD test length ranges | Up to `--test_size` words per bin (default **1,000**) | `test_{min}-{max}.jsonl` only |

With defaults (`--train_size 10000`, `--test_size 1000`): the first bin yields roughly 8,000 training and 2,000 in-distribution test examples (after the 80/20 split; upsampled if fewer unique words are available). Each additional bin contributes up to 1,000 test examples at longer lengths.

Output layout: `datasets/n{train_size}-trainlen{upper_bound_of_first_bin}/{language_name}/` (e.g. `datasets/n10000-trainlen50/{language_name}/`).

Each language subdirectory contains:
- `train_{min}-{max}.jsonl` — training data (first bin only)
- `test_{min}-{max}.jsonl` — in-distribution test (first bin) or OOD test (later bins)
- `meta_data.json` — DFA states, alphabet, and bin configuration (used to build the per-language tokenizer)

Each record has the form:
```json
{"word": "abba", "states": "q0q1q2q1q0"}
```

---

### Step 3 — Hyperparameter Search

Run a sweep over GPT-2 architectures (layers, heads, model dimension, learning rate) to find the best configuration per language. Training stops early once 100% in-distribution accuracy is reached:

```bash
python src/training/state_prediction_ntp.py \
  --task <language_name> \
  --dataset_root datasets/
```

> **Note:** `<language_name>` must match the `Name` column in `languages.csv` exactly. This same identifier is used as `--task` across all training scripts and corresponds to the subdirectory name created under `datasets/` during dataset generation.

Sweep logs are written as `.txt` files to `results/hyperparam_search/{language_name}/`. Once the sweep is done, run `hparam_selection.py` to parse the logs and select the best architecture per language, saved as a JSON file used in the next step:

```bash
cd scripts && sbatch hparam_selection.sh
# or directly:
python src/utils/hparam_selection.py \
  --results_dir results/hyperparam_search \
  --output best_hparams.json
```

Add `--nope` if the sweep was run with NoPE (No Positional Encoding), which will restrict selection to logs with `nope` in the filename and save to `best_hparams_nope.json`.

---

### Step 4 — Multi-Seed Evaluation

Train the best architecture for each language across multiple random seeds to get robust generalisation results:

```bash
python src/training/run_multiple_seeds_ntp.py \
  --tasks <language_name> \
  --dataset_root datasets/ \
  --save_path results/
```

Results are saved to `results/` and can then be used for evaluation and plotting.

---

### Utility/Extras — Check C-RASP Membership

To check whether a regex belongs to C-RASP independently of the main pipeline:

```bash
python src/regex_generation/decider.py --regex '(cab+c)*'
# multiple regexes at once:
python src/regex_generation/decider.py --regex '(cab+c)*' '(ab+ba)*' '(aabb)*'
```

Add `--draw` to also generate a DFA diagram.

## Requirements

```
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---|---|
| `automata` | 0.1.4 | 
| `pysemigroup` | 0.3b3 | 
| `torch` | 2.2.2 | 
| `transformers` | 4.47.0 | 
| `networkx` | 3.1 | 
| `sympy` | 1.12 | 
| `graphviz` | 0.21 |


> **Compatibility note:** This script customises GPT-2's inputs, outputs, loss masks, and attention masks in ways that may break across transformers versions. If you upgrade the library, verify that everything still behaves as expected — and in particular that the new version's default GPT-2 does not silently overwrite any of these.


## Folder Structure

```
state-tracking-crasp/
├── datasets/                        # Generated word datasets (created at runtime)
│   ├── n10000-trainlen50/      # 10k training samples, train length range up to 50
│   ├── n10000-trainlen200/     # 10k training samples, train length range up to 200
│   ├── n100000-trainlen50/     # 100k training samples, train length range up to 50
│   └── n100000-trainlen200/    # 100k training samples, train length range up to 200
├── scripts/                     # SLURM batch scripts
│   ├── generate_languages.sh
│   ├── sample_languages.sh
│   ├── create_words.sh
│   ├── hparam_selection.sh
│   └── decider.sh
├── src/
│   ├── regex_generation/        # Language generation and classification
│   │   ├── generate_regexes.py  # Generate random regexes and classify them
│   │   ├── classify_att.py      # Classify DFAs from .att files
│   │   ├── decider.py           # Decide C-RASP membership for a regex
│   │   ├── sample_languages.py  # Sample languages from classification results
│   │   ├── dfa.py               # DFA utilities and diagram generation
│   │   ├── crasp_reg.py         # C-RASP algebraic helpers
│   │   └── results/             # Pre-computed classification results
│   │       ├── r.txt            # Generated languages in R
│   │       ├── crasp.txt        # Generated languages in C-RASP
│   │       ├── r_infinity_not_crasp.txt        # Generated languages in R-infinity, not C-RASP
│   │       ├── not_r_infinity.txt              # Generated languages not in R-infinity
│   │       └── all_results.txt                 # All generated languages
│   ├── data/
│   │   └── create_words.py      # Sample words from regex and build datasets
│   ├── training/
│   │   ├── state_prediction_ntp.py      # Hyperparameter sweep
│   │   └── run_multiple_seeds_ntp.py    # Multi-seed training
│   └── utils/
│       ├── utils.py                    # Shared I/O utilities
│       └── hparam_selection.py         # Select best architecture from sweep logs
├── languages.csv                       # Language list for experiments (optional if generating your own)
└── requirements.txt
```