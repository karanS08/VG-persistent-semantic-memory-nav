"""Post-Paper-Draft Test 2.1 -- 2x2 Memory x Position factorial, OFFLINE re-analysis
of already-completed runs. Reads only data/runs/*/batch_metrics.json, logs/*.log and
results/post_paper/factorial_audit_labels.json; writes only to
data/results_postdraft/factorial_influence/. Launches nothing, imports nothing from
packages/.

Run from repo root:
    ~/miniconda3/envs/rpv_mem/bin/python scripts/postdraft/factorial_analysis.py

Conditions (verified from batch_metrics.json flags + driver logs, see REPORT.md):
  A reset memory / reset position   : reset5, reset5_rep2, reset5_rep3
  B persist memory / reset position : persist_memory_only_persistmemoryonly5_rep{1,2,3}
  C reset memory / persist position : persist_position_only_persistpositiononly5_rep{1,2,3}
  D persist memory / persist position: persistent_continue5, _rep2, _rep3
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_log_parser import load_run_evals, classify_eval  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/results_postdraft/factorial_influence"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260928)
N_BOOT = 20000

QUERIES = ["a large window", "a wall-mounted radiator", "a cabinet", "a wall vent", "a bench"]

# (condition, rep) -> (run dir, stdout log or None for logs/phase4_<dir>.log, code epoch)
# Code epochs from run timestamps (log timestamps / file mtimes) vs git history:
#   E1 = 2026-09-02 ~17:00 (before 9807034 lexical/category verification fix of 09-03,
#        before f72edd4 coverage memory of 09-04; multi-anchor fusion state uncertain --
#        it was 'uncommitted-until' 9d2879b on 09-04 per PROGRESS.md)
#   E2 = 2026-09-05 09:51-10:34 (after 9e01c92 / 4df6e00)
#   E3 = 2026-09-05 12:45-17:07 (after 0544e40, 823f98b, 41436da; defaults unchanged,
#        logging-only additions)
RUNS = {
    ("A", 1): ("phase4_batch_reset5", "logs/phase4_reset_vs_continue.log", "E1"),
    ("A", 2): ("phase4_batch_reset5_rep2", None, "E2"),
    ("A", 3): ("phase4_batch_reset5_rep3", None, "E2"),
    ("B", 1): ("phase4_batch_persist_memory_only_persistmemoryonly5_rep1", None, "E2"),
    ("B", 2): ("phase4_batch_persist_memory_only_persistmemoryonly5_rep2", None, "E3"),
    ("B", 3): ("phase4_batch_persist_memory_only_persistmemoryonly5_rep3", None, "E3"),
    ("C", 1): ("phase4_batch_persist_position_only_persistpositiononly5_rep1", None, "E3"),
    ("C", 2): ("phase4_batch_persist_position_only_persistpositiononly5_rep2", None, "E3"),
    ("C", 3): ("phase4_batch_persist_position_only_persistpositiononly5_rep3", None, "E3"),
    ("D", 1): ("phase4_batch_persistent_continue5", "logs/phase4_reset_vs_continue.log", "E1"),
    ("D", 2): ("phase4_batch_persistent_continue5_rep2", None, "E2"),
    ("D", 3): ("phase4_batch_persistent_continue5_rep3", None, "E2"),
}
EXPECTED_FLAGS = {  # condition -> (persistent_across_queries, persist_memory, persist_position)
    "A": (False, None, None), "B": (False, True, False), "C": (False, False, True), "D": (True, None, None),
}
METRICS = ["success", "spl", "steps", "path_length", "wall_clock_s", "coverage", "influence_rate",
           "n_contextual_anchor", "n_fallback", "n_direct_recall"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def load_episodes():
    audit = json.load(open(ROOT / "results/post_paper/factorial_audit_labels.json"))["labels"] \
        if (ROOT / "results/post_paper/factorial_audit_labels.json").exists() else {}
    eps, checks = [], []
    for (cond, rep), (run, log, epoch) in RUNS.items():
        bm = json.load(open(ROOT / "data/runs" / run / "batch_metrics.json"))
        # --- settings checks ---
        exp = EXPECTED_FLAGS[cond]
        chk = {
            "run": run, "cond": cond, "rep": rep, "epoch": epoch,
            "queries_ok": bm["queries_used"] == QUERIES,
            "scene": bm["scene_path"],
            "persistent_across_queries": bm["persistent_across_queries"],
            "persist_memory": bm.get("persist_memory"), "persist_position": bm.get("persist_position"),
            "theta_direct": bm["theta_direct_used"], "min_conf": bm["min_confirming_observations"],
            "vqa": bm.get("vqa_verification", "absent(pre-09-04 code)"),
            "coverage_memory": bm.get("coverage_memory_enabled", "absent(pre-09-04 code)"),
            "anchor_weight": bm.get("anchor_weight_used", "absent"),
            "gates": {k: bm.get(k, "absent") for k in ("fresh_evidence_guard_enabled", "relevance_gate_enabled", "lexical_gate_enabled")},
        }
        chk["flags_ok"] = (bm["persistent_across_queries"] == exp[0]
                           and (exp[1] is None or bm.get("persist_memory") == exp[1])
                           and (exp[2] is None or bm.get("persist_position") == exp[2]))
        checks.append(chk)
        parsed = load_run_evals(run, ROOT / log if log else None)
        for e in bm["episodes"]:
            evals = parsed[e["episode_id"]]["evals"]
            assert (sum(x["influenced"] for x in evals), len(evals)) == \
                (e["memory_probe"]["influenced"], e["memory_probe"]["evaluated"]), run
            cases = [classify_eval(x) for x in evals]
            ugc = e["usable_growth_curve"]
            step0 = next((c for s, c in ugc if s == 0), 0)
            vd = e.get("verification_detail") or {}
            key = f"{cond}{rep}.{e['episode_id']}"
            succ = e["outcome"] == "success"
            eps.append({
                "key": key, "cond": cond, "rep": rep, "session": f"{cond}{rep}", "epoch": epoch, "run": run,
                "episode_id": e["episode_id"], "query": e["query"],
                "success": int(succ), "reason": e["reason"], "spl": e["spl"], "steps": e["steps_taken"],
                "path_length": e["path_length"], "wall_clock_s": e["wall_clock_seconds"],
                "coverage": e["coverage_ratio_at_end"],
                "n_direct_recall": e["branch_counts"]["direct_recall"],
                "n_contextual_anchor": e["branch_counts"]["contextual_anchor"],
                "n_fallback": e["branch_counts"]["fallback"],
                "memory_size_at_end": e["memory_size_at_end"],
                "usable_at_step0": step0,
                "new_usable_after_step0": int(any(s > 0 for s, _ in ugc)),
                "probe_evaluated": len(evals), "probe_influenced": sum(x["influenced"] for x in evals),
                "influence_rate": e["memory_influence_rate"],
                "n_not_influenced": cases.count("not_influenced"),
                "n_infl_target_changed": cases.count("influenced_target_changed"),
                "n_infl_live_empty_live0": cases.count("influenced_live_empty_live0"),
                "n_infl_live_empty_liveN": cases.count("influenced_live_empty_liveN_noqualifying"),
                "logged_influence_classification": e.get("memory_influence_classification", "not_logged(pre-823f98b)"),
                "verified_entity_id": vd.get("entity_id") if succ else None,
                "verified_relevance": vd.get("relevance") if succ else None,
                "verified_caption": vd.get("description") if succ else None,
                "rrd_exists": (ROOT / e["rrd_path"]).exists() if e.get("rrd_path") else False,
                "rrd_path": e.get("rrd_path"),
                "agent_label_strict": (audit.get(key, [None])[0] == "strict") if succ and key in audit else (False if succ else None),
                "agent_label_lenient": (audit.get(key, [None])[0] in ("strict", "lenient")) if succ and key in audit else (False if succ else None),
            })
    return eps, checks


# ---------------------------------------------------------------- statistics
def per_query_means(eps, cond, metric, queries):
    out = []
    for q in queries:
        v = [e[metric] for e in eps if e["cond"] == cond and e["query"] == q and e[metric] is not None]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


def contrast_value(eps, weights, metric, queries):
    """sum_c w_c * (mean over queries of per-query condition mean)."""
    return float(sum(w * np.nanmean(per_query_means(eps, c, metric, queries)) for c, w in weights.items()))


def exact_stratified_perm(eps, cx, cy, metric, queries):
    """Exact two-sided permutation test of mean_q(meanY_q - meanX_q): within each query,
    all re-assignments of the pooled X/Y episodes to groups of the original sizes
    (query-stratified; treats reps within a query as exchangeable -- ignores
    within-session dependence, see session-level test)."""
    per_q_vals, obs = [], 0.0
    for q in queries:
        x = [e[metric] for e in eps if e["cond"] == cx and e["query"] == q]
        y = [e[metric] for e in eps if e["cond"] == cy and e["query"] == q]
        pool = np.array(x + y, float)
        ny = len(y)
        vals = []
        for idx in itertools.combinations(range(len(pool)), ny):
            m = np.zeros(len(pool), bool); m[list(idx)] = True
            vals.append(pool[m].mean() - pool[~m].mean())
        per_q_vals.append(np.array(vals))
        obs += np.mean(y) - np.mean(x)
    dist = np.zeros(1)
    for v in per_q_vals:
        dist = (dist[:, None] + v[None, :]).ravel()
    dist /= len(queries); obs /= len(queries)
    p = float(np.mean(np.abs(dist) >= abs(obs) - 1e-12))
    return round(obs, 4), round(p, 5), int(dist.size)


def session_perm(eps, cx, cy, metric, queries):
    """Exact permutation over whole sessions (the truly independent unit for B/C/D,
    whose episodes are chained). Session value = mean over `queries`."""
    def sess_vals(c):
        out = []
        for s in sorted({e["session"] for e in eps if e["cond"] == c}):
            out.append(np.mean([e[metric] for e in eps if e["session"] == s and e["query"] in queries]))
        return out
    x, y = sess_vals(cx), sess_vals(cy)
    pool = np.array(x + y); ny = len(y)
    obs = np.mean(y) - np.mean(x)
    ds = []
    for idx in itertools.combinations(range(len(pool)), ny):
        m = np.zeros(len(pool), bool); m[list(idx)] = True
        ds.append(pool[m].mean() - pool[~m].mean())
    ds = np.array(ds)
    return round(float(obs), 4), round(float(np.mean(np.abs(ds) >= abs(obs) - 1e-12)), 4), len(ds)


def signflip_queries(eps, cx, cy, metric, queries):
    d = per_query_means(eps, cy, metric, queries) - per_query_means(eps, cx, metric, queries)
    obs = d.mean()
    ds = [np.mean(d * np.array(s)) for s in itertools.product([1, -1], repeat=len(d))]
    return [round(float(v), 4) for v in d], round(float(np.mean(np.abs(ds) >= abs(obs) - 1e-12)), 4)


def boot_ci(eps, weights, metric, queries, n=N_BOOT):
    """Query-stratified bootstrap: resample episodes (reps) with replacement within
    each (condition, query) cell."""
    cells = {(c, q): np.array([e[metric] for e in eps if e["cond"] == c and e["query"] == q], float)
             for c in weights for q in queries}
    stats = np.zeros(n)
    for c, w in weights.items():
        qm = np.zeros((n, len(queries)))
        for j, q in enumerate(queries):
            v = cells[(c, q)]
            qm[:, j] = v[RNG.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
        stats += w * qm.mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


CONTRASTS = {
    "B-A (memory, position reset)": {"B": 1, "A": -1},
    "C-A (position, memory reset)": {"C": 1, "A": -1},
    "D-A (combined)": {"D": 1, "A": -1},
    "D-C (memory, position persistent)": {"D": 1, "C": -1},
    "D-B (position, memory persistent)": {"D": 1, "B": -1},
    "interaction D-B-C+A": {"D": 1, "B": -1, "C": -1, "A": 1},
}


def analyse(eps, subset_name, queries):
    rows = []
    for cname, w in CONTRASTS.items():
        for metric in ["success", "spl", "steps", "path_length", "wall_clock_s", "coverage", "influence_rate"]:
            if any(e[metric] is None for e in eps if e["cond"] in w and e["query"] in queries):
                continue
            est = contrast_value(eps, w, metric, queries)
            ci = boot_ci(eps, w, metric, queries)
            row = {"subset": subset_name, "contrast": cname, "metric": metric, "estimate": round(est, 4),
                   "boot95_lo": ci[0], "boot95_hi": ci[1]}
            if len(w) == 2:
                (cy,), (cx,) = [k for k, v in w.items() if v == 1], [k for k, v in w.items() if v == -1]
                _, p_strat, nperm = exact_stratified_perm(eps, cx, cy, metric, queries)
                _, p_sess, nsess = session_perm(eps, cx, cy, metric, queries)
                perq, p_sign = signflip_queries(eps, cx, cy, metric, queries)
                row.update({"p_exact_query_stratified": p_strat, "n_perms_strat": nperm,
                            "p_exact_session_level": p_sess, "n_perms_session": nsess,
                            "p_signflip_query_means": p_sign, "per_query_diffs": perq})
            else:
                row.update({"p_exact_query_stratified": None, "p_exact_session_level": None,
                            "p_signflip_query_means": None,
                            "note": "no exact test for the interaction at this n; bootstrap CI only"})
            rows.append(row)
    return rows


def cond_summary(eps, queries, label):
    out = []
    for c in "ABCD":
        E = [e for e in eps if e["cond"] == c and e["query"] in queries]
        k, n = sum(e["success"] for e in E), len(E)
        r = {"subset": label, "cond": c, "n": n, "success_k": k, "success_rate": round(k / n, 4),
             "success_wilson95": wilson(k, n)}
        for m in ["spl", "steps", "path_length", "wall_clock_s", "coverage", "influence_rate",
                  "n_contextual_anchor", "n_fallback", "n_direct_recall", "usable_at_step0"]:
            vals = [e[m] for e in E if e[m] is not None]
            r[f"mean_{m}"] = round(float(np.mean(vals)), 4)
        r["mean_spl_boot95"] = boot_ci(eps, {c: 1}, "spl", queries)
        # agent (NOT human) rule-based labels -- sensitivity only
        for lab in ("agent_label_strict", "agent_label_lenient"):
            kk = sum(1 for e in E if e["success"] and e[lab])
            r[f"{lab}_success_k"] = kk
            r[f"{lab}_success_wilson95"] = wilson(kk, n)
        out.append(r)
    return out


def session_table(eps):
    rows = []
    for (c, rep), (run, _, epoch) in RUNS.items():
        E = [e for e in eps if e["cond"] == c and e["rep"] == rep]
        rows.append({
            "session": f"{c}{rep}", "run": run, "epoch": epoch,
            "success": f"{sum(e['success'] for e in E)}/{len(E)}",
            "mean_spl": round(np.mean([e["spl"] for e in E]), 4),
            "total_steps": sum(e["steps"] for e in E),
            "total_path_m": round(sum(e["path_length"] for e in E), 2),
            "total_wall_s": round(sum(e["wall_clock_s"] for e in E), 1),
            "mean_coverage": round(np.mean([e["coverage"] for e in E]), 4),
            "branch_DR/CA/FB": "/".join(str(sum(e[k] for e in E)) for k in ("n_direct_recall", "n_contextual_anchor", "n_fallback")),
            "probe_infl/eval": f"{sum(e['probe_influenced'] for e in E)}/{sum(e['probe_evaluated'] for e in E)}",
            "logged_influence_class": ",".join(e["logged_influence_classification"][:4] for e in E),
        })
    return rows


def main():
    eps, checks = load_episodes()
    json.dump(checks, open(OUT / "factorial_settings_check.json", "w"), indent=1)
    with open(OUT / "factorial_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0].keys())); w.writeheader(); w.writerows(eps)

    q_all, q_14 = QUERIES, QUERIES[1:]
    matched = [e for e in eps if e["epoch"] != "E1"]
    results = {
        "_doc": "Test 2.1 factorial. Every number derived from data/runs/<run>/batch_metrics.json and "
                "logs (see factorial_episodes.csv 'run' column). success = RAW pipeline success "
                "(no human audit exists for 10 of 12 sessions). agent_label_* = Claude rule-based "
                "caption labels from results/post_paper/factorial_audit_labels.json -- NOT a human audit.",
        "settings_checks": checks,
        "condition_summary": cond_summary(eps, q_all, "all_queries") + cond_summary(eps, q_14, "queries_1_4")
                             + cond_summary(matched, q_all, "code_matched_all_queries")
                             + cond_summary(matched, q_14, "code_matched_queries_1_4"),
        "contrasts": analyse(eps, "all_queries", q_all) + analyse(eps, "queries_1_4", q_14)
                     + analyse(matched, "code_matched_all_queries", q_all)
                     + analyse(matched, "code_matched_queries_1_4", q_14),
        "sessions": session_table(eps),
    }
    # query-0 noise floor: all four conditions are identical by design at query 0
    q0 = [e for e in eps if e["episode_id"] == 0]
    results["query0_noise_floor"] = {
        "n": len(q0), "success": f"{sum(e['success'] for e in q0)}/{len(q0)}",
        "spl_min_max": [min(e["spl"] for e in q0), max(e["spl"] for e in q0)],
        "steps_min_max": [min(e["steps"] for e in q0), max(e["steps"] for e in q0)],
        "by_session": {e["session"]: {"success": e["success"], "spl": e["spl"], "steps": e["steps"]} for e in q0},
    }
    # condition-C leakage diagnostic: usable entities already present at step 0 of a query
    results["step0_carryover"] = {
        c: {e["key"]: {"usable_at_step0": e["usable_at_step0"], "new_usable_after_step0": e["new_usable_after_step0"],
                       "success": e["success"], "steps": e["steps"], "verified_entity_id": e["verified_entity_id"]}
            for e in eps if e["cond"] == c and e["episode_id"] > 0}
        for c in "ABCD"
    }
    json.dump(results, open(OUT / "factorial_results.json", "w"), indent=1, default=str)
    with open(OUT / "factorial_contrasts.csv", "w", newline="") as f:
        keys = sorted({k for r in results["contrasts"] for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(results["contrasts"])

    # ---- audit sheet (for a HUMAN; label columns intentionally blank) ----
    with open(OUT / "factorial_audit_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "run_dir", "episode_id", "query", "verified_entity_id", "verified_at_step",
                    "clip_relevance", "matched_entry_caption_FULL", "rrd_for_visual_check",
                    "prior_audit_record", "HUMAN_correct_object(Y/N/partial)", "HUMAN_notes"])
        for e in eps:
            if not e["success"]:
                continue
            prior = ("PROGRESS.md 2026-09-02 entry: 'both conditions hit 5/5, all audited genuine' "
                     "(auditor not identified in the entry)") if e["epoch"] == "E1" else "none"
            w.writerow([e["key"], e["run"], e["episode_id"], e["query"], e["verified_entity_id"], e["steps"],
                        e["verified_relevance"], e["verified_caption"],
                        e["rrd_path"] if e["rrd_exists"] else "deleted (disk cleanup)", prior, "", ""])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
