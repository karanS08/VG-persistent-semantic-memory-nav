"""Post-Paper Test Plan 2.1: 2x2 memory x position factorial -- re-analysis of
the EXISTING runs (A=reset5*, B=persistmemoryonly5, C=persistpositiononly5,
D=continue5*), with the rule-based audit in
results/post_paper/factorial_audit_labels.json. Writes
results/post_paper/factorial_summary.json. No runs are modified."""
import glob, json, random, statistics as st
from pathlib import Path

CONDS = {
    "A": ("reset / reset", "data/runs/phase4_batch_reset5*/batch_metrics.json"),
    "B": ("memory persists / position resets", "data/runs/phase4_batch_persist_memory_only_persistmemoryonly5_rep*/batch_metrics.json"),
    "C": ("memory resets / position persists", "data/runs/phase4_batch_persist_position_only_persistpositiononly5_rep*/batch_metrics.json"),
    "D": ("both persist", "data/runs/phase4_batch_persistent_continue5*/batch_metrics.json"),
}
labels = json.load(open("results/post_paper/factorial_audit_labels.json"))["labels"]


def boot_ci(xs, n=10000, seed=0):
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choice(xs) for _ in xs) for _ in range(n))
    return [round(ms[int(0.025 * n)], 3), round(ms[int(0.975 * n)], 3)]


out = {"_doc": __doc__, "conditions": {}}
per_query = {}
for c, (desc, pat) in CONDS.items():
    files = sorted(glob.glob(pat))
    eps = []
    for i, f in enumerate(files):
        for j, e in enumerate(json.load(open(f))["episodes"]):
            key = f"{c}{i+1}.{j}"
            lab = labels.get(key, [None])[0] if e["outcome"] == "success" else None
            eps.append(dict(key=key, run=Path(f).parent.name, query=e["query"], raw=e["outcome"] == "success",
                            strict=lab == "strict", lenient=lab in ("strict", "lenient"), spl=e["spl"],
                            steps=e["steps_taken"], wall=e["wall_clock_seconds"],
                            infl=e.get("memory_influence_rate")))
    # query 0 is identical across all conditions (nothing carried over yet), so
    # the conditions can only differ on queries 1-4 -- report both.
    later = [e for e in eps if not e["key"].endswith(".0")]
    def agg(es):
        return {
            "n": len(es),
            "raw_success": round(st.mean(e["raw"] for e in es), 3),
            "audited_success_strict": round(st.mean(e["strict"] for e in es), 3),
            "audited_success_lenient": round(st.mean(e["lenient"] for e in es), 3),
            "raw_spl": round(st.mean(e["spl"] for e in es), 3),
            "audited_spl_strict": round(st.mean(e["spl"] * e["strict"] for e in es), 3),
            "audited_spl_strict_ci95": boot_ci([e["spl"] * e["strict"] for e in es]),
            "mean_steps": round(st.mean(e["steps"] for e in es), 1),
            "mean_wall_clock_s": round(st.mean(e["wall"] for e in es), 1),
            "mean_memory_influence_rate": round(st.mean(e["infl"] for e in es if e["infl"] is not None), 3),
        }
    out["conditions"][c] = {"description": desc, "runs": [Path(f).parent.name for f in files],
                            "all_queries": agg(eps), "queries_1_to_4": agg(later), "episodes": eps}

A, B, C, D = (out["conditions"][k]["queries_1_to_4"] for k in "ABCD")
out["effects_queries_1_to_4"] = {
    m: {"memory_B_minus_A": round(B[m] - A[m], 3), "position_C_minus_A": round(C[m] - A[m], 3),
        "combined_D_minus_A": round(D[m] - A[m], 3),
        "interaction_D-B-C+A": round(D[m] - B[m] - C[m] + A[m], 3)}
    for m in ("audited_success_strict", "audited_success_lenient", "audited_spl_strict", "mean_steps", "mean_wall_clock_s")
}
Path("results/post_paper/factorial_summary.json").write_text(json.dumps(out, indent=2))
for c in "ABCD":
    for scope in ("all_queries", "queries_1_to_4"):
        a = out["conditions"][c][scope]
        print(f"{c} {scope:15s} n={a['n']:2d} raw={a['raw_success']:.3f} strict={a['audited_success_strict']:.3f} "
              f"lenient={a['audited_success_lenient']:.3f} aSPL={a['audited_spl_strict']:.3f} {a['audited_spl_strict_ci95']} "
              f"steps={a['mean_steps']} wall={a['mean_wall_clock_s']}")
print(json.dumps(out["effects_queries_1_to_4"], indent=1))
