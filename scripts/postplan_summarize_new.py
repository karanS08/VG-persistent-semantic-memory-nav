"""Aggregate the 2026-09-28/29 post-plan runs by experiment, using the rule-based
audit in results/post_paper/postplan_audit_labels.json (successes only; an
unlabelled success is an error). Old paper runs are included where a group
compares against them, with labels from scripts/postplan_trace_rq1.py.
Writes results/post_paper/new_runs_summary.json."""
import json
from pathlib import Path

LAB = json.load(open("results/post_paper/postplan_audit_labels.json"))["labels"]
S, L, W = "strict", "lenient", "wrong"
OLD = {  # paper runs (results/batch_runs), labels from postplan_trace_rq1.py
    "phase4_batch_minconf1default": {0: L, 1: S, 3: S, 4: W, 5: S, 7: W},
    "phase4_batch_minconf1default_rep2": {0: L, 1: S, 3: S, 5: S},
    "phase4_batch_memory_off_minconf1default": {3: S},
    "phase4_batch_persistent_minconf1default": {1: S, 2: S, 3: L, 4: S, 5: S, 6: W},
    "phase4_batch_persistent_minconf1default_rep2": {0: L, 1: S, 2: W, 3: S, 4: S, 5: S, 7: W},
}
p4 = "phase4_batch_"
GROUPS = {
    "RQ1 memory-off, plain (live = this step's DAAAM batch)": [p4 + "memory_off_minconf1default"] + [p4 + f"memory_off_lagbase_rep{r}" for r in (1, 2, 3)],
    "RQ1 memory-off, 30 s live window (lag-matched)": [p4 + f"memory_off_livewin30_lagbase_rep{r}" for r in (1, 2, 3)],
    "RQ1 full memory (prefix gate)": [p4 + "minconf1default", p4 + "minconf1default_rep2", p4 + "pp_full_rep1", p4 + "pp_full_rep2"],
    "RQ1 persistent (prefix gate)": [p4 + "persistent_minconf1default", p4 + "persistent_minconf1default_rep2"],
    "caption_head full memory": [p4 + f"lexcaption_head_chrq1v2_rep{r}" for r in (1, 2, 3)],
    "caption_head memory-off": [p4 + f"memory_off_lexcaption_head_chrq1v2_rep{r}" for r in (1, 2, 3)],
    "caption_head persistent": [p4 + f"persistent_lexcaption_head_chrq1v2_rep{r}" for r in (1, 2, 3)],
    "decay on, 24 h age (persistent)": [p4 + f"persistent_pp_decay24h_rep{r}" for r in (1, 2)],
    "decay off, 24 h age (persistent)": [p4 + "persistent_lambda0.0_pp_nodecay24h_rep1"],
    "factorial B memory-only, rep4 (instrumented)": [p4 + "persist_memory_only_pp_memonly5_rep1"],
    "held-out 00337 memory-off": [p4 + "memory_off_heldout00337_memoff"],
    "held-out 00337 reset rep2": [p4 + "heldout00337_baseline_rep2"],
    "held-out 00337 persistent rep2": [p4 + "persistent_heldout00337_persistent_rep2"],
    "verif. all gates rep2": [p4 + "semantic00770mined_minconf1default_leakfix_rep2"],
    "verif. no lexical rep2": [p4 + "no_lexical_gate_verifyloo_nolexical_rep2"],
    "verif. no relevance rep2": [p4 + "no_relevance_gate_verifyloo_norelevance_rep2"],
    "verif. no reconfirmation rep2": [p4 + "minconf0_verifyloo_nomultiobs_rep2"],
    "verif. no fresh-evidence guard rep2": [p4 + "persistent_no_fresh_evidence_guard_verifyloo_nofreshguard_rep2"],
    "verif. ALL gates removed rep2": [p4 + "persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification_rep2"],
    "absent objects (8 absent queries)": [p4 + "mq_absent_rep1"],
    "00861 GT pilot (caption audit only)": [p4 + "mq_gt861_rep1"],
}


def load(run):
    for base in ("data/runs", "results/batch_runs"):
        f = Path(base) / run / "batch_metrics.json"
        if f.exists():
            return json.load(open(f))["episodes"]
    raise FileNotFoundError(run)


out = {}
for g, runs in GROUPS.items():
    n = raw = s = l = w = 0
    spl_s = 0.0
    for run in runs:
        for e in load(run):
            n += 1
            if e["outcome"] != "success":
                continue
            raw += 1
            lab = OLD[run][e["episode_id"]] if run in OLD else LAB[f"{run}/{e['episode_id']}"]
            s += lab == S; l += lab in (S, L); w += lab == W
            spl_s += e["spl"] * (lab == S)
    out[g] = {"runs": runs, "n": n, "raw": raw, "strict": s, "lenient": l, "false_finds": w,
              "raw_rate": round(raw / n, 3), "strict_rate": round(s / n, 3), "lenient_rate": round(l / n, 3),
              "strict_spl": round(spl_s / n, 3), "precision_strict": round(s / raw, 3) if raw else None}
    o = out[g]
    print(f"{g:55s} n={n:3d} raw={raw:2d} strict={s:2d} ({o['strict_rate']:.2f}) lenient={l:2d} ({o['lenient_rate']:.2f}) "
          f"FP={w:2d} sSPL={o['strict_spl']:.3f} prec={o['precision_strict']}")
Path("results/post_paper/new_runs_summary.json").write_text(json.dumps({"_doc": __doc__, "groups": out}, indent=2))
