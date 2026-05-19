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
This classifies each regex into three algebraic classes — **R**, **C-RASP**, and **R∞** — and saves the results to `src/regex_generation/results/`.

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

Pre-built datasets are included under `data/`. Skip to Step 3.

**Option B: Generate datasets from `languages.csv`**

For each language, this samples words of varying lengths, computes DFA state traces, and splits them into train/test bins by length:
```bash
cd scripts && sbatch create_words.sh
# or directly:
python src/data/create_words.py \
  --languages_csv languages.csv \
  --output data/ \
  --bins "(min,50),(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(401,450),(451,500)"
```

Each language gets a subdirectory under `data/` with:
- `train_{min}-{max}.jsonl` — training split (first bin only)
- `test_{min}-{max}.jsonl` — in-distribution test split
- `test_{a}-{b}.jsonl` — OOD test splits (longer lengths)
- `meta_data.json` — DFA states, alphabet, and bin configuration. The *Tokenizer* per language will be created based on this file.

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
  --dataset_root data/
```

Sweep logs are written as `.txt` files to `results/hyperparam_search/{task}/`. Once the sweep is done, run `hparam_selection.py` to parse the logs and select the best architecture per language, saved as a JSON file used in the next step:

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
  --dataset_root data/ \
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

| Package | Version | Purpose |
|---|---|---|
| `automata` | 0.1.4 | NFA/DFA construction from regex |
| `pysemigroup` | 0.3b3 | Syntactic monoid computation |
| `torch` | 2.2.2 | Model training |
| `transformers` | 4.47.0 | GPT-2 architecture and Trainer |
| `networkx` | 3.1 | Graph algorithms for C-RASP decision |
| `sympy` | 1.12 | Nullspace computation for balanced morphisms |
| `graphviz` | 0.21 | DFA diagram generation |


Compatibility note: 
his script customises GPT-2's inputs, outputs, loss masks, and attention masks in ways that may break across transformers versions. If you upgrade the library, verify that everything still behaves as expected — and in particular that the new version's default GPT-2 does not silently overwrite any of these.


## Folder Structure

```
state-tracking-crasp/
├── data/                        # Generated word datasets (created at runtime)
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
│       └── utils.py                    # Utilities, e.g. I/O functions
│       └── hparam_selection.py         # Parses hyperparameter sweep logs to select the best architecture 
└── requirements.txt      
```