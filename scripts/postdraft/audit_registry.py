"""Shared registry for the post-draft GT/audit/precision analysis (Test 3.4 + Task B).

Maps every audited rate in acra_paper/main.tex to the run directory it came from,
and encodes the EXISTING author audit verdicts per reported-success episode, with
the file each verdict was taken from. Nothing here is a new rating: every verdict
below is transcribed from an existing audit record, or (for the ablation figure,
which has no per-episode record file) reconstructed from the paper's published
audited success AND audited SPL, which together pin the set down uniquely
(checked in precision_and_intervals.py::check_reconstruction).

Verdict codes:
  "correct"  author counted it as a genuine match
  "wrong"    author counted it as a false positive
  "lenient"  author counted it correct in the published number, but either the
             paper's own stated rule or the author's own audit note calls it a
             partial / stale match (cushion->cushioned bench; step-1 stale bench).
             The "strict" precision variant scores these as wrong.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/runs"


def run_path(short):
    return RUNS / f"phase4_batch_{short}" / "batch_metrics.json"


# group -> list of (condition label, run short name, role)
GROUPS = {
    "fig_ablation_audited": [
        ("memory-off", "memory_off_minconf1default", "figure"),
        ("full memory", "minconf1default", "figure"),
        ("no verification discount", "no_verify_discount_minconf1default", "figure"),
        ("uniform decay", "no_mobility_minconf1default", "figure"),
        ("persistent", "persistent_minconf1default", "figure"),
        ("full memory (repeat 2)", "minconf1default_rep2", "cited-repeat"),
        ("persistent (repeat 2)", "persistent_minconf1default_rep2", "cited-repeat"),
    ],
    "fig_verification_ablation": [
        ("all gates, non-persistent (reference 0.5)", "semantic00770mined_minconf1default_leakfix", "baseline"),
        ("all gates, persistent", "persistent_semantic00770mined_minconf1default_leakfix", "baseline"),
        ("relevance gate removed", "no_relevance_gate_verifyloo_norelevance", "figure"),
        ("reconfirmation removed", "minconf0_verifyloo_nomultiobs", "figure"),
        ("lexical gate removed", "no_lexical_gate_verifyloo_nolexical", "figure"),
        ("fresh-evidence guard removed (persistent)", "persistent_no_fresh_evidence_guard_verifyloo_nofreshguard", "figure"),
        ("all four gates removed (persistent)",
         "persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification", "text"),
    ],
    "fig_memory_age_sweep": [
        ("age 0h", "persistent_memage0s", "figure"),
        ("age 1h", "persistent_memage1h", "figure"),
        ("age 3h", "persistent_memage3h", "figure"),
        ("age 6h", "persistent_memage6h", "figure"),
        ("age 24h", "persistent_memage24h", "figure"),
    ],
    "heldout_00337": [
        ("held-out reset", "heldout00337_baseline", "text"),
        ("held-out persistent", "persistent_heldout00337_persistent", "text"),
    ],
    "tab_reset_vs_continue": [
        ("Reset", "reset5", "table"),
        ("Continue", "persistent_continue5", "table"),
    ],
    "minconf_claim": [
        ("min_confirming_observations=1 (catgate era)", "minconf1_catgate", "text"),
    ],
}

AUDIT_SOURCES = {
    "memory_off_minconf1default": "reconstructed: acra_paper/main.tex fig:ablation-audited (0.125/0.062) + PROGRESS.md 2026-09-02 sweep audit",
    "minconf1default": "reconstructed: main.tex fig:ablation-audited audited 0.50 / SPL 0.092 (unique subset) + PROGRESS.md 2026-09-02",
    "no_verify_discount_minconf1default": "reconstructed: main.tex audited 0.50 / SPL 0.131 (unique subset)",
    "no_mobility_minconf1default": "reconstructed: main.tex audited 0.625 / SPL 0.154 (unique subset)",
    "persistent_minconf1default": "reconstructed: main.tex audited 0.625 / SPL 0.310 + paper/main.tex tab:ablation-audited caption ('doorway'->cabinet)",
    "minconf1default_rep2": "reconstructed: main.tex 'repeat runs landed on exactly these same audited rates' (0.50)",
    "persistent_minconf1default_rep2": "reconstructed: main.tex 'repeat runs ... same audited rates' (0.625)",
    "semantic00770mined_minconf1default_leakfix": "main.tex tab:semantic-adversarial (5/8 real, 0/2 adv) + results/verification_leaveoneout_audit.md baseline",
    "persistent_semantic00770mined_minconf1default_leakfix": "main.tex tab:semantic-adversarial (3/8, 0/2) + results/full_no_verification_audit.md",
    "no_relevance_gate_verifyloo_norelevance": "results/verification_leaveoneout_audit.md (Gate 2)",
    "minconf0_verifyloo_nomultiobs": "results/verification_leaveoneout_audit.md (Gate 3)",
    "no_lexical_gate_verifyloo_nolexical": "results/verification_leaveoneout_audit.md (Gate 4 table)",
    "persistent_no_fresh_evidence_guard_verifyloo_nofreshguard": "results/verification_leaveoneout_audit.md (Gate 1)",
    "persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification": "results/full_no_verification_audit.md",
    "persistent_memage0s": "results/memory_age_sweep_audit.md",
    "persistent_memage1h": "results/memory_age_sweep_audit.md",
    "persistent_memage3h": "results/memory_age_sweep_audit.md",
    "persistent_memage6h": "results/memory_age_sweep_audit.md",
    "persistent_memage24h": "results/memory_age_sweep_audit.md",
    "heldout00337_baseline": "results/held_out_test_scene00337_results.md",
    "persistent_heldout00337_persistent": "results/held_out_test_scene00337_results.md",
    "reset5": "PROGRESS.md 2026-09-02 Reset-vs-Continue ('both 5/5, all audited genuine')",
    "persistent_continue5": "PROGRESS.md 2026-09-02 ('2 matches with minor imprecision ... neither a wrong-category FP')",
    "minconf1_catgate": "paper/main.tex sec:results-minconf ('every one of those five successes a genuine match')",
}

# (run short, episode_id) -> verdict ; everything not listed is "correct".
EXCEPTIONS = {
    ("minconf1default", 0): ("lenient", "cushion -> cushioned bench, counted correct (needed for 0.50/0.092)"),
    ("minconf1default", 4): ("wrong", "dining table -> portable heater"),
    ("minconf1default", 7): ("wrong", "framed painting -> window"),
    ("minconf1default_rep2", 0): ("lenient", "cushion -> bench, counted correct (needed for 0.50)"),
    ("no_verify_discount_minconf1default", 3): ("wrong", "mirror -> window"),
    ("persistent_minconf1default", 6): ("wrong", "doorway -> wall-mounted cabinet (named in paper/main.tex caption)"),
    ("persistent_minconf1default_rep2", 0): ("lenient", "cushion -> bench, counted correct (needed for 0.625)"),
    ("persistent_minconf1default_rep2", 2): ("wrong", "wooden chair -> air conditioner"),
    ("persistent_minconf1default_rep2", 7): ("wrong", "framed painting -> window"),
    ("no_lexical_gate_verifyloo_nolexical", 0): ("wrong", "cabinet -> radiator"),
    ("no_lexical_gate_verifyloo_nolexical", 1): ("wrong", "trash bin -> radiator"),
    ("no_lexical_gate_verifyloo_nolexical", 2): ("wrong", "wall vent -> radiator"),
    ("no_lexical_gate_verifyloo_nolexical", 3): ("wrong", "curtain -> window"),
    ("no_lexical_gate_verifyloo_nolexical", 4): ("wrong", "fireplace -> radiator"),
    ("no_lexical_gate_verifyloo_nolexical", 5): ("wrong", "lamp -> door"),
    ("no_lexical_gate_verifyloo_nolexical", 6): ("wrong", "wall clock -> door"),
    ("no_lexical_gate_verifyloo_nolexical", 9): ("wrong", "bicycle (adversarial) -> marble wall"),
    ("persistent_no_fresh_evidence_guard_verifyloo_nofreshguard", 7):
        ("lenient", "bench verified at step 1 from carried-over memory; audit md: strict 0.4 / lenient 0.5"),
    ("persistent_memage0s", 0): ("lenient", "cushion -> bench ('standard bench-with-cushion partial match')"),
    ("persistent_memage1h", 0): ("lenient", "cushion -> bench"),
    ("persistent_memage3h", 0): ("lenient", "cushion -> bench"),
    ("persistent_memage6h", 0): ("lenient", "cushion -> bench"),
    ("persistent_memage3h", 5): ("wrong", "radiator -> mirror reflecting a radiator"),
    ("persistent_memage6h", 1): ("wrong", "window -> mirror reflecting a window"),
    ("persistent_memage6h", 7): ("wrong", "painting -> mirror with peeling paint"),
    ("persistent_heldout00337_persistent", 6): ("wrong", "pillow -> bed with a pillow (compound caption)"),
}
# whole-run overrides
ALL_WRONG = {"persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification"}

ADVERSARIAL = {"a staircase", "a bicycle", "a laptop"}


def author_verdict(short, ep):
    if short in ALL_WRONG:
        return "wrong", "all 10 wrong per results/full_no_verification_audit.md"
    return EXCEPTIONS.get((short, ep), ("correct", "counted genuine"))
