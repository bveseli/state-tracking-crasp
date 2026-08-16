#!/usr/bin/env python3
"""Verify best_hparams_by_task_nope_8k_len200train.json against summary.txt.

Secondary summaries use coarse bins. Selection rule (adapted from larger-train):
  1. Only configs with eval_len0-200_acc == 1.0  (short-range filter)
  2. Maximize accuracy longest-first:
       401-500 -> 301-400 -> 201-300
  3. On remaining ties, smaller model: layers -> heads -> dims
  4. Then prefer larger lr; smalllr suffix when lr < 5e-4
"""
import json
import re
from pathlib import Path

HP = Path(__file__).resolve().parent
JSON_PATH = HP / "best_hparams_by_task_nope_8k_len200train.json"

RANGES = ["401-500", "301-400", "201-300", "0-200"]
CASCADE = RANGES[:-1]
RANGE_RE = {r: re.compile(rf"eval_len{re.escape(r)}_acc:\s*([0-9.]+)") for r in RANGES}
CFG_RE = re.compile(r"^(\S+)")
LR_RE = re.compile(r"lr:\s*([0-9.eE+-]+)")
SIZE_RE = re.compile(r"^(\d+)l(\d+)h(\d+)d(?:smalllr)?$")


def config_name(cfg, lr):
    return cfg + ("smalllr" if lr < 5e-4 else "")


def model_size(name):
    m = SIZE_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {name}")
    return tuple(int(x) for x in m.groups())


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    m = CFG_RE.match(line)
    if not m:
        return None
    cfg = m.group(1)
    accs = {}
    for r, rx in RANGE_RE.items():
        mm = rx.search(line)
        if not mm:
            return None
        accs[r] = float(mm.group(1))
    lm = LR_RE.search(line)
    if not lm:
        return None
    lr = float(lm.group(1))
    name = config_name(cfg, lr)
    return {
        "cfg": cfg,
        "lr": lr,
        "name": name,
        "size": model_size(name),
        "accs": accs,
    }


def selection_key(run):
    layers, heads, dims = run["size"]
    return (
        tuple(run["accs"][r] for r in CASCADE),
        -layers,
        -heads,
        -dims,
        run["lr"],
        run["name"],
    )


def load_runs(lang_dir):
    runs = []
    fails = []
    for p in sorted(lang_dir.glob("*.txt")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parsed = parse_line(line)
            if parsed is None:
                fails.append((p.name, i, line[:160]))
                continue
            parsed["source"] = f"{p.name}:{i}"
            runs.append(parsed)
    return runs, fails


def main():
    claimed = json.loads(JSON_PATH.read_text())
    mismatches = []
    matches = 0
    no_eligible = []
    missing = []
    parse_fails = []
    size_tiebreaks = 0
    computed = {}

    for lang, claim in claimed.items():
        lang_dir = HP / lang
        if not lang_dir.is_dir():
            missing.append(lang)
            continue
        runs, fails = load_runs(lang_dir)
        parse_fails.extend((lang, *f) for f in fails)
        eligible = [r for r in runs if r["accs"]["0-200"] >= 1.0 - 1e-12]
        if not eligible:
            no_eligible.append(lang)
            computed[lang] = None
            continue
        best = max(eligible, key=selection_key)
        best_ck = tuple(best["accs"][r] for r in CASCADE)
        acc_optima = sorted(
            {
                r["name"]
                for r in eligible
                if tuple(r["accs"][x] for x in CASCADE) == best_ck
            }
        )
        if len(acc_optima) > 1:
            size_tiebreaks += 1
        computed[lang] = best["name"]
        if claim == best["name"]:
            matches += 1
        else:
            claimed_runs = [r for r in runs if r["name"] == claim]
            claimed_best = (
                max(claimed_runs, key=selection_key) if claimed_runs else None
            )
            mismatches.append(
                {
                    "lang": lang,
                    "claimed": claim,
                    "computed": best["name"],
                    "computed_accs": [best["accs"][r] for r in RANGES],
                    "computed_size": list(best["size"]),
                    "computed_source": best["source"],
                    "acc_tied": acc_optima,
                    "claimed_accs": (
                        [claimed_best["accs"][r] for r in RANGES]
                        if claimed_best
                        else None
                    ),
                    "claimed_size": (
                        list(claimed_best["size"]) if claimed_best else None
                    ),
                    "claimed_source": (
                        claimed_best["source"] if claimed_best else None
                    ),
                }
            )

    folders = {p.name for p in HP.iterdir() if p.is_dir()}
    lines = [
        "RULE: 0-200==1.0; cascade 401-500->301-400->201-300; size L->H->D; prefer larger lr",
        f"json langs: {len(claimed)}",
        f"matches: {matches}",
        f"mismatches: {len(mismatches)}",
        f"size_tiebreaks: {size_tiebreaks}",
        f"no_eligible: {no_eligible}",
        f"missing folders: {missing}",
        f"parse fails: {len(parse_fails)}",
        f"folders not in json: {sorted(folders - set(claimed))}",
        f"json missing folder: {sorted(set(claimed) - folders)}",
        "",
        "=== MISMATCHES ===",
    ]
    for m in mismatches:
        lines.append(
            f"{m['lang']}: claimed={m['claimed']}  computed={m['computed']}"
        )
        lines.append(
            f"  computed accs [401-500,301-400,201-300,0-200]: "
            f"{m['computed_accs']} size={m['computed_size']} @{m['computed_source']}"
        )
        if len(m["acc_tied"]) > 1:
            lines.append(f"  acc-tied pre-size: {m['acc_tied']}")
        if m["claimed_accs"] is not None:
            lines.append(
                f"  claimed  accs: {m['claimed_accs']} size={m['claimed_size']} "
                f"@{m['claimed_source']}"
            )
        else:
            lines.append("  claimed NOT FOUND")

    out = HP / "best_hparams_verification.txt"
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    report = HP / "best_hparams_verification.json"
    report.write_text(
        json.dumps(
            {
                "matches": matches,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "size_tiebreaks": size_tiebreaks,
                "no_eligible": no_eligible,
                "missing": missing,
                "computed": computed,
            },
            indent=2,
        )
        + "\n"
    )
    print(text)
    print(f"Wrote {out}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
