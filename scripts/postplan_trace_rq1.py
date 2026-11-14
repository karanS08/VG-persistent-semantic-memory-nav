"""Post-Paper Test Plan 0.1/0.2: recompute the paper's RQ1 figure from raw
batch_metrics.json under an explicit strict and lenient audit, and emit a
machine-readable trace of each paper number -> run dir -> audit rule.
Labels (per episode_id) are the 2026-09-28 re-audit; see EXPERIMENT_RESULTS.md."""
import json
from pathlib import Path
S, L, W = "strict", "lenient", "wrong"
RUNS = {  # condition -> {run_dir: {episode_id: label}} (successes only)
  "memory_off": {"phase4_batch_memory_off_minconf1default": {3: S}},
  "full_memory": {"phase4_batch_minconf1default": {0: L, 1: S, 3: S, 4: W, 5: S, 7: W},
                  "phase4_batch_minconf1default_rep2": {0: L, 1: S, 3: S, 5: S}},
  "uniform_decay": {"phase4_batch_no_mobility_minconf1default": {1: S, 3: S, 5: S, 6: S, 7: S}},
  "no_verify_discount": {"phase4_batch_no_verify_discount_minconf1default": {1: S, 3: W, 5: S, 6: S, 7: S}},
  "persistent": {"phase4_batch_persistent_minconf1default": {1: S, 2: S, 3: L, 4: S, 5: S, 6: W},
                 "phase4_batch_persistent_minconf1default_rep2": {0: L, 1: S, 2: W, 3: S, 4: S, 5: S, 7: W}},
}
PAPER = {"memory_off": (0.125, 0.062), "full_memory": (0.50, 0.092), "uniform_decay": (0.625, 0.131),
         "no_verify_discount": (0.50, 0.154), "persistent": (0.625, 0.310)}
NOTES = {L: "lenient = part-of object or region caption naming it; includes cushion->'bench with cushioned seat' and 'wall ... features a large rectangular mirror'"}
trace = {"_doc": __doc__, "notes": NOTES, "rows": []}
for cond, runs in RUNS.items():
    for run, labels in runs.items():
        eps = json.load(open(f"results/batch_runs/{run}/batch_metrics.json"))["episodes"]
        n = len(eps)
        row = {"condition": cond, "run_dir": f"results/batch_runs/{run}", "n": n,
               "raw_success": round(sum(e["outcome"] == "success" for e in eps) / n, 3)}
        for rule, ok in (("strict", {S}), ("lenient", {S, L})):
            good = [e for e in eps if labels.get(e["episode_id"]) in ok]
            row[f"{rule}_success"] = round(len(good) / n, 3)
            row[f"{rule}_spl"] = round(sum(e["spl"] for e in good) / n, 3)
        if not run.endswith("_rep2"):
            row["paper_value_success_spl"] = PAPER[cond]
            row["paper_matches"] = ("lenient" if (row["lenient_success"], row["lenient_spl"]) == PAPER[cond]
                                    else "strict" if (row["strict_success"], row["strict_spl"]) == PAPER[cond] else "neither")
        trace["rows"].append(row)
        print({k: v for k, v in row.items() if k != "run_dir"}, run)
Path("results/post_paper/rq1_trace.json").write_text(json.dumps(trace, indent=2))
