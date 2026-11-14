"""Post-Paper-Draft Test 2.2 (part 1) -- what the memory-influence probe actually
counts, from EXISTING logs only. Offline; reads batch_metrics.json + stdout logs,
writes to data/results_postdraft/factorial_influence/.

Probe semantics (packages/frontier_agent/frontier_agent/agent.py, fresh-replan block,
unchanged since commit 6880c8c 2026-08-24):
    full_memory_best = _select_target(store, q, now)                       # all entries
    live_best        = _select_target(store, q, now, entity_ids_pool=live_entity_ids)
    memory_influenced = None if both None else (best_id != live_id)        # TOP-1 id change
    value_without = live_best r_i, or 0.0 if live_best is None
live_entity_ids (orchestrator run_episode) = ids contained in THIS STEP's batch of
DAAAM grounding updates -- not "entities currently in view".

Run from repo root:
    ~/miniconda3/envs/rpv_mem/bin/python scripts/postdraft/influence_probe_analysis.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_log_parser import load_run_evals, classify_eval  # noqa: E402
from factorial_analysis import RUNS as FACTORIAL_RUNS, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results_postdraft/factorial_influence"

# RQ3 figure (acra_paper/figures/fig_memory_influence.pdf) is produced by
# scripts/generate_paper_figures.py::fig_memory_influence from
# results/batch_runs/phase4_batch_minconf1default and
# results/batch_runs/phase4_batch_persistent_minconf1default (byte-identical, md5-checked,
# to the data/runs/ copies). The _rep2 runs are the paper's stated repeats.
LOG_FINAL = "logs/phase4_final_minconf1default.log"
RQ3_RUNS = {
    "RQ3 full memory (fig)": ("phase4_batch_minconf1default", LOG_FINAL),
    "RQ3 persistent (fig)": ("phase4_batch_persistent_minconf1default", LOG_FINAL),
    "RQ3 full memory rep2": ("phase4_batch_minconf1default_rep2", LOG_FINAL),
    "RQ3 persistent rep2": ("phase4_batch_persistent_minconf1default_rep2", LOG_FINAL),
}
CASES = ["not_influenced", "influenced_target_changed", "influenced_live_empty_live0",
         "influenced_live_empty_liveN_noqualifying"]


def summarise(label, run, log):
    bm = json.load(open(ROOT / "data/runs" / run / "batch_metrics.json"))
    parsed = load_run_evals(run, ROOT / log if log else None)
    ep_rows, allc = [], Counter()
    first_eval = Counter()
    for e in bm["episodes"]:
        ev = parsed[e["episode_id"]]["evals"]
        c = Counter(classify_eval(x) for x in ev)
        allc.update(c)
        n = len(ev)
        with_alt = c["not_influenced"] + c["influenced_target_changed"]
        if ev:
            first_eval[classify_eval(ev[0])] += 1
        ep_rows.append({
            "group": label, "run": run, "episode_id": e["episode_id"], "query": e["query"],
            "outcome": e["outcome"], "n_evals": n, **{k: c[k] for k in CASES},
            "paper_rate_top1_changed_incl_empty": round((n - c["not_influenced"]) / n, 4) if n else None,
            "rate_target_changed_given_live_alternative": round(c["influenced_target_changed"] / with_alt, 4) if with_alt else None,
            "share_evals_with_no_live_alternative": round((c["influenced_live_empty_live0"] + c["influenced_live_empty_liveN_noqualifying"]) / n, 4) if n else None,
            "first_eval_case": classify_eval(ev[0]) if ev else None,
            "first_eval_step": ev[0]["step"] if ev else None,
            "logged_influence_classification": e.get("memory_influence_classification"),
        })
    n = sum(allc.values())
    infl = n - allc["not_influenced"]
    with_alt = allc["not_influenced"] + allc["influenced_target_changed"]
    empty = allc["influenced_live_empty_live0"] + allc["influenced_live_empty_liveN_noqualifying"]
    run_row = {
        "group": label, "run": run, "n_evals": n, **{k: allc[k] for k in CASES},
        "paper_rate": round(infl / n, 4), "paper_rate_wilson95": wilson(infl, n),
        "influenced_that_are_live_empty": f"{empty}/{infl}",
        "share_influenced_live_empty": round(empty / infl, 4) if infl else None,
        "strict_rate_given_live_alternative": round(allc["influenced_target_changed"] / with_alt, 4) if with_alt else None,
        "strict_rate_wilson95": wilson(allc["influenced_target_changed"], with_alt),
        "n_with_live_alternative": with_alt,
        "per_episode_paper_rate_range": [min(r["paper_rate_top1_changed_incl_empty"] for r in ep_rows if r["n_evals"]),
                                         max(r["paper_rate_top1_changed_incl_empty"] for r in ep_rows if r["n_evals"])],
        "first_eval_cases": dict(first_eval),
    }
    return run_row, ep_rows


def main():
    groups = dict(RQ3_RUNS)
    for (cond, rep), (run, log, _) in FACTORIAL_RUNS.items():
        groups[f"Factorial {cond}{rep}"] = (run, log)
    run_rows, ep_rows = [], []
    for label, (run, log) in groups.items():
        r, e = summarise(label, run, log)
        run_rows.append(r); ep_rows.extend(e)

    # pooled by factorial condition
    pooled = []
    for cond in "ABCD":
        rows = [r for r in run_rows if r["group"].startswith(f"Factorial {cond}")]
        tot = {k: sum(r[k] for r in rows) for k in CASES}
        n = sum(tot.values()); infl = n - tot["not_influenced"]
        wa = tot["not_influenced"] + tot["influenced_target_changed"]
        pooled.append({"condition": cond, "n_evals": n, **tot, "paper_rate": round(infl / n, 4),
                       "paper_rate_wilson95": wilson(infl, n),
                       "strict_rate_given_live_alternative": round(tot["influenced_target_changed"] / wa, 4) if wa else None,
                       "strict_rate_wilson95": wilson(tot["influenced_target_changed"], wa), "n_with_live_alternative": wa})

    # existing logged episode-level classification (823f98b, only in runs made after 11:05 2026-09-05)
    logged = Counter(r["logged_influence_classification"] for r in ep_rows if r["logged_influence_classification"])
    logged_by_run = {}
    for r in ep_rows:
        if r["logged_influence_classification"]:
            logged_by_run.setdefault(r["run"], Counter())[r["logged_influence_classification"]] += 1

    out = {
        "_doc": "Memory-influence probe decomposition. Cases are derived from the printed probe lines "
                "(value_without==0.0000 <=> live-only selection returned None). Counts reconciled "
                "exactly against batch_metrics.json memory_probe for every episode.",
        "definition_from_code": {
            "memory_influenced": "top-1 selected entity id differs (best_id != live_id); NOT any ranking disagreement",
            "live_only_empty": "live_id=None while full-memory target exists => memory_influenced=True (counted as influenced)",
            "both_empty": "memory_influenced=None => not counted in evaluated",
            "live_set": "entity ids in THIS step's DAAAM grounding-update batch (not 'in view')",
        },
        "per_run": run_rows, "pooled_factorial": pooled,
        "logged_episode_classification_total": dict(logged),
        "logged_episode_classification_by_run": {k: dict(v) for k, v in logged_by_run.items()},
    }
    json.dump(out, open(OUT / "influence_probe_decomposition.json", "w"), indent=1)
    with open(OUT / "influence_probe_per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys())); w.writeheader(); w.writerows(ep_rows)

    # ---- figure: composition of probe evaluations per run group ----
    labels = [r["group"].replace("Factorial ", "") for r in run_rows]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    bottom = np.zeros(len(run_rows))
    colors = {"not_influenced": "#bdbdbd", "influenced_target_changed": "#b2182b",
              "influenced_live_empty_live0": "#92c5de", "influenced_live_empty_liveN_noqualifying": "#2166ac"}
    names = {"not_influenced": "not influenced", "influenced_target_changed": "influenced: top-1 target changed",
             "influenced_live_empty_live0": "influenced: no live update this step (live set empty)",
             "influenced_live_empty_liveN_noqualifying": "influenced: live set non-empty but no qualifying live target"}
    for k in CASES:
        v = np.array([r[k] / r["n_evals"] for r in run_rows])
        ax.bar(range(len(run_rows)), v, bottom=bottom, color=colors[k], label=names[k])
        bottom += v
    ax.set_xticks(range(len(run_rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("share of probe evaluations")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    ax.set_title("What 'memory-influenced' decisions actually are (from run logs)", fontsize=10)
    fig.savefig(OUT / "fig_influence_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
