#!/usr/bin/env python3
"""2026-09-05: Controlled temporal-decay experiment (critique.md P0 / Benchmark B,
Panel A: confidence vs. memory age).

Calls the REAL, unmodified production reliability function
(memory_layer.reliability.compute_reliability) with synthetic elapsed times,
rather than reimplementing the C_i formula -- this is exactly the code every
batch run in this project already exercises, just with `now` swept across a
controlled range instead of real wall-clock passage. No simulation needed for
this panel: C_i is a pure function of (entity state, now, mobility, alpha,
lambda_base), and entity state / alpha / lambda_base are held fixed while we
vary `now` and mobility.

Real, non-invented parameters used throughout:
  - alpha=0.5, lambda_base=0.001/s, theta_direct=0.2 (configs/memory.yaml,
    the actual values every reported batch run uses)
  - mobility: radiator=0.03 (low), chair=0.3 (medium), phone=0.85 (high) --
    real cached values from configs/mobility_priors.json, not invented
  - relevance s(q,d_i) in {0.11, 0.15, 0.19} -- the real range the paper's
    own Section 5.2/7.1 report CLIP similarity actually produces for
    genuine matches in this pipeline ("~0.11-0.19"), not an arbitrary guess
  - n_i in {1, 2}: 1 is the actual production default
    (min_confirming_observations=1, retuned 2026-09-02); 2 matches the
    paper's existing Table 1 worked example, included for direct
    comparability with that table

Does NOT cover Benchmark B's other two panels (navigation performance vs.
age, stale-memory false-find rate vs. age) -- those need a live episode with
an artificially-aged memory entry injected before querying, which is a
separate, heavier experiment (see PROGRESS.md). This script is the fast,
purely-computational half that needed no simulation to do honestly.

Usage: python scripts/controlled_temporal_decay_experiment.py
Writes: results/controlled_temporal_decay.json (machine-readable)
        results/controlled_temporal_decay.md (human-readable table)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "memory_layer"))

from memory_layer.reliability import compute_reliability  # noqa: E402
from memory_layer.models import MemoryEntity  # noqa: E402

import yaml  # noqa: E402

with open(ROOT / "configs" / "memory.yaml") as f:
    mem_cfg = yaml.safe_load(f)
ALPHA = mem_cfg["reliability"]["alpha"]
LAMBDA_BASE = mem_cfg["reliability"]["lambda_base"]
THETA_DIRECT = mem_cfg["recall"]["theta_direct"]

with open(ROOT / "configs" / "mobility_priors.json") as f:
    mobility_table = json.load(f)["categories"]

MOBILITY_CATEGORIES = {
    "radiator (low)": mobility_table["radiator"],
    "chair (medium)": mobility_table["chair"],
    "phone (high)": mobility_table["phone"],
}

AGES_SECONDS = [
    ("0s", 0), ("30s", 30), ("1min", 60), ("5min", 300), ("10min", 600),
    ("30min", 1800), ("1h", 3600), ("6h", 21600), ("24h", 86400),
]

RELEVANCE_VALUES = [0.11, 0.15, 0.19]  # real observed range, see module docstring
N_I_VALUES = [1, 2]


def run():
    rows = []
    for n_i in N_I_VALUES:
        for cat_label, mobility in MOBILITY_CATEGORIES.items():
            entity = MemoryEntity(
                entity_id=0, category=cat_label, description=cat_label,
                c_i=1.0, n_i=n_i, v_i=1.0, first_observed=0.0, last_observed=0.0,
            )
            for age_label, age_s in AGES_SECONDS:
                c_i = compute_reliability(entity, now=age_s, mobility=mobility,
                                           alpha=ALPHA, lambda_base=LAMBDA_BASE)
                for relevance in RELEVANCE_VALUES:
                    r_i = relevance * c_i
                    direct_recall_fires = r_i >= THETA_DIRECT
                    rows.append({
                        "n_i": n_i, "category": cat_label, "mobility": mobility,
                        "age_label": age_label, "age_seconds": age_s,
                        "C_i": round(c_i, 6), "relevance": relevance,
                        "R_i": round(r_i, 6), "theta_direct": THETA_DIRECT,
                        "direct_recall_fires": direct_recall_fires,
                    })
    return rows


def write_outputs(rows):
    out_json = ROOT / "results" / "controlled_temporal_decay.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "params": {"alpha": ALPHA, "lambda_base": LAMBDA_BASE,
                       "theta_direct": THETA_DIRECT,
                       "mobility_used": {k: v for k, v in MOBILITY_CATEGORIES.items()}},
            "rows": rows,
        }, f, indent=2)

    lines = [
        "# Controlled temporal-decay experiment",
        "",
        "Real production `compute_reliability` (memory_layer/reliability.py),",
        f"real configured params: alpha={ALPHA}, lambda_base={LAMBDA_BASE}/s,",
        f"theta_direct={THETA_DIRECT}. Mobility values from configs/mobility_priors.json.",
        "",
        "## C_i vs. memory age, by mobility category (n_i=1, the real production default)",
        "",
        "| age | radiator (m=0.03) | chair (m=0.3) | phone (m=0.85) |",
        "|---|---|---|---|",
    ]
    by_key = {(r["n_i"], r["category"], r["age_label"]): r for r in rows if r["relevance"] == 0.15}
    for age_label, _ in AGES_SECONDS:
        vals = [f"{by_key[(1, cat, age_label)]['C_i']:.4f}" for cat in MOBILITY_CATEGORIES]
        lines.append(f"| {age_label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Direct-recall firing (R_i >= theta_direct) vs. age, n_i=1, at real relevance=0.15",
        "",
        "| age | radiator | chair | phone |",
        "|---|---|---|---|",
    ]
    for age_label, _ in AGES_SECONDS:
        vals = ["YES" if by_key[(1, cat, age_label)]["direct_recall_fires"] else "no"
                for cat in MOBILITY_CATEGORIES]
        lines.append(f"| {age_label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Direct-recall firing across the full real relevance range (0.11-0.19), n_i=1",
        "",
        "| category | age | relevance=0.11 | relevance=0.15 | relevance=0.19 |",
        "|---|---|---|---|---|",
    ]
    by_full = {(r["n_i"], r["category"], r["age_label"], r["relevance"]): r for r in rows}
    for cat in MOBILITY_CATEGORIES:
        for age_label, _ in AGES_SECONDS:
            vals = ["YES" if by_full[(1, cat, age_label, rel)]["direct_recall_fires"] else "no"
                    for rel in RELEVANCE_VALUES]
            lines.append(f"| {cat} | {age_label} | " + " | ".join(vals) + " |")

    out_md = ROOT / "results" / "controlled_temporal_decay.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    rows = run()
    write_outputs(rows)
