"""Task B: verification precision per condition + interval statistics for every
quantitative rate in acra_paper/main.tex. Read-only over data/runs, results/, logs/.

Outputs (data/results_postdraft/gt_audit_precision/):
  per_episode_audit.csv      every episode of every registered run + author verdict
  precision_by_condition.csv raw/audited success and verification precision, Wilson 95% CIs
  paired_tests.csv           exact McNemar (per-query pairs) + Fisher + Wilcoxon (SPL)
  claims_table.csv           every number in main.tex: value, n, source, CI, status
  reconstruction_check.json  proves the ablation-figure verdicts are the unique set
                             consistent with the published audited success AND SPL
  all_numbers.json           everything above in one file

Usage: ~/miniconda3/envs/rpv_mem/bin/python scripts/postdraft/precision_and_intervals.py
"""
import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_registry import ADVERSARIAL, AUDIT_SOURCES, GROUPS, ROOT, author_verdict, run_path  # noqa: E402

OUT = ROOT / "data/results_postdraft/gt_audit_precision"
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "n/a"


def boot_mean_ci(x, B=20000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return (round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4))


def mcnemar_exact(a, b):
    """a, b: equal-length 0/1 lists paired by query. Two-sided exact binomial on discordant pairs."""
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = b10 + b01
    p = 1.0 if n == 0 else stats.binomtest(b10, n, 0.5).pvalue
    return b10, b01, float(p)


# ---------------------------------------------------------------- load runs
def load_all():
    runs = {}
    rows = []
    for group, entries in GROUPS.items():
        for cond, short, role in entries:
            d = json.load(open(run_path(short)))
            eps = []
            for e in d["episodes"]:
                ok = e["outcome"] == "success"
                vd = e.get("verification_detail") or {}
                av, note = author_verdict(short, e["episode_id"]) if ok else ("", "")
                lenient = int(ok and av in ("correct", "lenient"))
                strict = int(ok and av == "correct")
                rec = dict(group=group, condition=cond, run=f"data/runs/phase4_batch_{short}/batch_metrics.json",
                           run_short=short, episode_id=e["episode_id"], query=e["query"],
                           adversarial=e["query"] in ADVERSARIAL, reported_success=int(ok),
                           author_verdict=av, author_note=note, audited_lenient=lenient, audited_strict=strict,
                           steps_taken=e["steps_taken"], spl_raw=e["spl"],
                           spl_audited_lenient=e["spl"] if lenient else 0.0,
                           spl_audited_strict=e["spl"] if strict else 0.0,
                           description=(vd.get("description") or "")[:160],
                           memory_influence_rate=e.get("memory_influence_rate"),
                           probe_evaluated=(e.get("memory_probe") or {}).get("evaluated"),
                           probe_influenced=(e.get("memory_probe") or {}).get("influenced"),
                           wall_clock_seconds=e.get("wall_clock_seconds"))
                eps.append(rec)
                rows.append(rec)
            runs[short] = dict(condition=cond, group=group, role=role, episodes=eps,
                               success_rate_file=d["success_rate"], mean_spl_file=d["mean_spl"],
                               persistent=bool(d.get("persistent_across_queries")))
    return runs, rows


def summarize(short, r):
    eps = r["episodes"]
    n = len(eps)
    rep = sum(e["reported_success"] for e in eps)
    len_ = sum(e["audited_lenient"] for e in eps)
    st = sum(e["audited_strict"] for e in eps)
    adv = [e for e in eps if e["adversarial"]]
    adv_fp = sum(e["reported_success"] for e in adv)
    real = [e for e in eps if not e["adversarial"]]
    return dict(
        group=r["group"], condition=r["condition"], run=f"data/runs/phase4_batch_{short}", role=r["role"],
        n_episodes=n, n_real=len(real), n_adversarial=len(adv),
        reported_successes=rep, raw_success=round(rep / n, 4), raw_success_ci=wilson(rep, n),
        audited_success_lenient=round(len_ / n, 4), audited_lenient_k=len_, audited_lenient_ci=wilson(len_, n),
        audited_success_strict=round(st / n, 4), audited_strict_k=st, audited_strict_ci=wilson(st, n),
        precision_lenient=(round(len_ / rep, 4) if rep else None), precision_lenient_ci=wilson(len_, rep),
        precision_strict=(round(st / rep, 4) if rep else None), precision_strict_ci=wilson(st, rep),
        false_positives_lenient=rep - len_, false_positives_strict=rep - st,
        adversarial_false_verifications=f"{adv_fp}/{len(adv)}" if adv else "",
        mean_spl_raw=round(float(np.mean([e["spl_raw"] for e in eps])), 4),
        mean_spl_raw_file=round(r["mean_spl_file"], 4),
        mean_spl_audited_lenient=round(float(np.mean([e["spl_audited_lenient"] for e in eps])), 4),
        mean_spl_audited_lenient_boot95=boot_mean_ci([e["spl_audited_lenient"] for e in eps]),
        mean_spl_audited_strict=round(float(np.mean([e["spl_audited_strict"] for e in eps])), 4),
        audit_source=AUDIT_SOURCES[short],
    )


# ------------------------------------------------- reconstruction uniqueness
PUBLISHED_ABLATION = {  # main.tex fig:ablation-audited via scripts/generate_acra_paper_figures.py
    "memory_off_minconf1default": (0.125, 0.062),
    "minconf1default": (0.50, 0.092),
    "no_verify_discount_minconf1default": (0.50, 0.131),
    "no_mobility_minconf1default": (0.625, 0.154),
    "persistent_minconf1default": (0.625, 0.310),
}


def check_reconstruction(runs):
    out = {}
    for short, (sr, spl) in PUBLISHED_ABLATION.items():
        eps = runs[short]["episodes"]
        n = len(eps)
        succ = [e for e in eps if e["reported_success"]]
        k = round(sr * n)
        matches = []
        for comb in itertools.combinations(succ, k):
            m = sum(e["spl_raw"] for e in comb) / n
            if round(m, 3) == round(spl, 3) or abs(m - spl) < 0.0015:
                matches.append(sorted(e["episode_id"] for e in comb))
        ours = sorted(e["episode_id"] for e in eps if e["audited_lenient"])
        out[short] = dict(published_audited_success=sr, published_audited_spl=spl, k=k,
                          n_reported=len(succ), subsets_consistent=matches,
                          registry_subset=ours, registry_subset_is_unique_match=(matches == [ours]))
    return out


# ---------------------------------------------------------------- log parsing
def parse_influence_log():
    """logs/phase4_final_minconf1default.log: per-decision probe lines for the 7 minconf1default batches."""
    path = ROOT / "logs/phase4_final_minconf1default.log"
    res, batch, ep = {}, None, None
    for line in open(path):
        m = re.match(r"=== \[(\d)/7\] (.*) ===", line)
        if m:
            batch = m.group(2)
            continue
        m = re.match(r"=== (?:Episode|Query) (\d+)", line)
        if m:
            ep = int(m.group(1))
            res.setdefault(batch, {}).setdefault(ep, [])
            continue
        m = re.search(r"contextual-anchor eval: live=(\d+) remembered=(\d+) .*influenced=(True|False)", line)
        if m and batch is not None and ep is not None:
            res[batch][ep].append((int(m.group(1)), int(m.group(2)), m.group(3) == "True"))
    out = {}
    for b, eps in res.items():
        dec = [x for v in eps.values() for x in v]
        out[b] = dict(
            n_decisions=len(dec), n_influenced=sum(x[2] for x in dec),
            n_influenced_with_zero_live_candidates=sum(1 for x in dec if x[2] and x[0] == 0),
            n_decisions_with_live_candidates=sum(1 for x in dec if x[0] > 0),
            n_influenced_with_live_candidates=sum(1 for x in dec if x[2] and x[0] > 0),
            first_decision_influenced=[(v[0][2] if v else None) for _, v in sorted(eps.items())],
            per_episode=[(sum(x[2] for x in v), len(v)) for _, v in sorted(eps.items())],
        )
    return out


def parse_gap_log():
    gaps = [float(m.group(1)) for m in re.finditer(r"gap=([\d.]+)s", open(ROOT / "logs/phase4_timestamp_gap_evidence.log").read())]
    g = np.array(gaps)
    return dict(n=len(g), mean=round(float(g.mean()), 3), max=round(float(g.max()), 3), min=round(float(g.min()), 3),
                sd=round(float(g.std(ddof=1)), 3), boot95_mean=boot_mean_ci(g), source="logs/phase4_timestamp_gap_evidence.log")


def parse_pilot_log():
    t = open(ROOT / "logs/phase4_grounding_tuning2.log").read()
    return [(int(a), int(b)) for a, b in re.findall(r"memory probe: influenced (\d+)/(\d+)", t)]


# ------------------------------------------------------------------- main
def main():
    runs, rows = load_all()
    summ = {s: summarize(s, r) for s, r in runs.items()}
    recon = check_reconstruction(runs)
    infl = parse_influence_log()
    gap = parse_gap_log()
    pilot = parse_pilot_log()

    def vec(short, key="audited_lenient"):
        return [e[key] for e in runs[short]["episodes"]]

    def spl(short, key="spl_audited_lenient"):
        return [e[key] for e in runs[short]["episodes"]]

    pairs = [
        ("full memory vs memory-off (audited)", "minconf1default", "memory_off_minconf1default", "main.tex L489"),
        ("persistent vs full memory (audited)", "persistent_minconf1default", "minconf1default", "main.tex L490"),
        ("persistent vs memory-off (audited)", "persistent_minconf1default", "memory_off_minconf1default", "main.tex L698 (conclusion)"),
        ("uniform decay vs full memory (audited)", "no_mobility_minconf1default", "minconf1default", "main.tex L557"),
        ("no verification discount vs full memory (audited)", "no_verify_discount_minconf1default", "minconf1default", "main.tex L599"),
        ("full memory rep1 vs rep2 (audited)", "minconf1default", "minconf1default_rep2", "main.tex L481"),
        ("persistent rep1 vs rep2 (audited)", "persistent_minconf1default", "persistent_minconf1default_rep2", "main.tex L481"),
        ("held-out persistent vs reset (audited)", "persistent_heldout00337_persistent", "heldout00337_baseline", "main.tex L494"),
        ("Continue vs Reset (success)", "persistent_continue5", "reset5", "main.tex tab:reset-vs-continue"),
        ("no lexical gate vs all gates, non-persistent (audited)", "no_lexical_gate_verifyloo_nolexical", "semantic00770mined_minconf1default_leakfix", "main.tex L579"),
        ("no relevance gate vs all gates, non-persistent (audited)", "no_relevance_gate_verifyloo_norelevance", "semantic00770mined_minconf1default_leakfix", "main.tex L584"),
        ("no reconfirmation vs all gates, non-persistent (audited)", "minconf0_verifyloo_nomultiobs", "semantic00770mined_minconf1default_leakfix", "main.tex L584"),
        ("no fresh-evidence guard vs all gates, persistent (audited)", "persistent_no_fresh_evidence_guard_verifyloo_nofreshguard", "persistent_semantic00770mined_minconf1default_leakfix", "main.tex L582"),
        ("all gates removed vs all gates, persistent (audited)", "persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification", "persistent_semantic00770mined_minconf1default_leakfix", "main.tex L566-569"),
        ("tab:semantic-adversarial persistent vs non-persistent (real queries)", "persistent_semantic00770mined_minconf1default_leakfix", "semantic00770mined_minconf1default_leakfix", "main.tex tab:semantic-adversarial"),
        ("memory age 1h vs 0h (audited)", "persistent_memage1h", "persistent_memage0s", "main.tex fig:memory-age-sweep"),
        ("memory age 24h vs 0h (audited)", "persistent_memage24h", "persistent_memage0s", "main.tex fig:memory-age-sweep"),
    ]
    paired = []
    for name, a, b, where in pairs:
        va, vb = vec(a), vec(b)
        qa = [e["query"] for e in runs[a]["episodes"]]
        qb = [e["query"] for e in runs[b]["episodes"]]
        same_q = qa == qb
        if "real queries" in name:
            keep = [i for i, q in enumerate(qa) if q not in ADVERSARIAL]
            va, vb = [va[i] for i in keep], [vb[i] for i in keep]
        b10, b01, p_mc = mcnemar_exact(va, vb) if same_q else (None, None, None)
        ka, kb, n = sum(va), sum(vb), len(va)
        _, p_f = stats.fisher_exact([[ka, n - ka], [kb, n - kb]])
        sa, sb = spl(a), spl(b)
        if "real queries" in name:
            sa, sb = [sa[i] for i in keep], [sb[i] for i in keep]
        diff = np.array(sa) - np.array(sb)
        p_w = float(stats.wilcoxon(sa, sb).pvalue) if same_q and np.any(diff != 0) else None
        paired.append(dict(comparison=name, a=f"data/runs/phase4_batch_{a}", b=f"data/runs/phase4_batch_{b}", paper=where,
                           n_pairs=n, same_queries_same_order=same_q, k_a=ka, k_b=kb,
                           rate_a=round(ka / n, 3), rate_b=round(kb / n, 3),
                           discordant_a_only=b10, discordant_b_only=b01,
                           mcnemar_exact_p=None if p_mc is None else round(p_mc, 4),
                           fisher_exact_p=round(float(p_f), 4),
                           mean_audited_spl_a=round(float(np.mean(sa)), 4), mean_audited_spl_b=round(float(np.mean(sb)), 4),
                           wilcoxon_signed_rank_p_spl=None if p_w is None else round(p_w, 4)))

    # strict-variant McNemar for the headline RQ1 pairs
    for name, a, b in [("STRICT full memory vs memory-off", "minconf1default", "memory_off_minconf1default"),
                       ("STRICT persistent vs full memory", "persistent_minconf1default", "minconf1default")]:
        va, vb = vec(a, "audited_strict"), vec(b, "audited_strict")
        b10, b01, p = mcnemar_exact(va, vb)
        paired.append(dict(comparison=name, a=f"data/runs/phase4_batch_{a}", b=f"data/runs/phase4_batch_{b}",
                           paper="variant: cushion->bench counted as FP", n_pairs=len(va), same_queries_same_order=True,
                           k_a=sum(va), k_b=sum(vb), rate_a=round(sum(va) / len(va), 3), rate_b=round(sum(vb) / len(vb), 3),
                           discordant_a_only=b10, discordant_b_only=b01, mcnemar_exact_p=round(p, 4),
                           fisher_exact_p=round(float(stats.fisher_exact([[sum(va), len(va) - sum(va)], [sum(vb), len(vb) - sum(vb)]])[1]), 4)))

    # ---------------------------------------------------------------- influence
    fb = infl.get("baseline (minconf=1 default)", {})
    pb = infl.get("persistent (minconf=1 default)", {})
    pb2 = infl.get("persistent repeat #2 (minconf=1 default, determinism check)", {})
    fb2 = infl.get("baseline repeat #2 (minconf=1 default, determinism check)", {})
    first_p = [x for x in pb.get("first_decision_influenced", []) + pb2.get("first_decision_influenced", []) if x is not None]
    first_np = [x for x in fb.get("first_decision_influenced", []) + fb2.get("first_decision_influenced", []) if x is not None]
    rates_full = [e["memory_influence_rate"] for e in runs["minconf1default"]["episodes"] if e["memory_influence_rate"] is not None]
    rates_pers = [e["memory_influence_rate"] for e in runs["persistent_minconf1default"]["episodes"] if e["memory_influence_rate"] is not None]
    mw = stats.mannwhitneyu(rates_pers, rates_full, alternative="two-sided")

    def S(short):
        return summ[short]

    # ------------------------------------------------------------- claims table
    C = []

    def claim(cid, line, text, value, k=None, n=None, source="", status="traced", note="", test=""):
        ci = fmt_ci(k, n) if (k is not None and n) else ""
        C.append(dict(id=cid, main_tex_line=line, claim=text, value=value, k=k, n=n, wilson95=ci,
                      source=source, status=status, test=test, note=note))

    ab = ["memory_off_minconf1default", "minconf1default", "no_verify_discount_minconf1default",
          "no_mobility_minconf1default", "persistent_minconf1default"]
    names = ["memory-off", "full memory", "no verification discount", "uniform decay", "persistent"]
    for sh, nm in zip(ab, names):
        s = S(sh)
        claim(f"abl_raw_{sh}", "481-483", f"fig:ablation raw success, {nm}", s["raw_success"], s["reported_successes"], s["n_episodes"], s["run"])
        claim(f"abl_aud_{sh}", "473-474 (figure)", f"fig:ablation audited success, {nm}", s["audited_success_lenient"],
              s["audited_lenient_k"], s["n_episodes"], s["run"], "reconstructed" if True else "",
              note=f"strict (cushion->bench = FP): {s['audited_strict_k']}/{s['n_episodes']} {fmt_ci(s['audited_strict_k'], s['n_episodes'])}")
        claim(f"abl_prec_{sh}", "(new)", f"verification precision, {nm}", s["precision_lenient"], s["audited_lenient_k"], s["reported_successes"], s["run"],
              "new", note=f"strict: {s['audited_strict_k']}/{s['reported_successes']} {fmt_ci(s['audited_strict_k'], s['reported_successes'])}")
        claim(f"abl_splraw_{sh}", "483", f"fig:ablation raw SPL, {nm}", s["mean_spl_raw"], source=s["run"],
              note=f"bootstrap95 audited-SPL {s['mean_spl_audited_lenient_boot95']}")
        claim(f"abl_splaud_{sh}", "figure", f"fig:ablation audited SPL, {nm}", s["mean_spl_audited_lenient"], source=s["run"],
              note=f"bootstrap95 {s['mean_spl_audited_lenient_boot95']}")
    for p in paired:
        pass
    claim("rq1_4x", "489", "audited success four times higher with memory (0.50 vs 0.125)", "0.50 vs 0.125", source="minconf1default vs memory_off_minconf1default",
          test=next(f"McNemar exact p={p['mcnemar_exact_p']} (discordant {p['discordant_a_only']}:{p['discordant_b_only']})" for p in paired if p['comparison'].startswith("full memory vs memory-off")))
    claim("rq1_persist", "490-491", "persistent 0.625 audited; SPL 0.310 vs 0.092", "0.625; 0.310 vs 0.092", source="persistent_minconf1default vs minconf1default",
          test=next(f"McNemar p={p['mcnemar_exact_p']}; Wilcoxon SPL p={p['wilcoxon_signed_rank_p_spl']}" for p in paired if p['comparison'].startswith("persistent vs full")))
    h1, h0 = S("persistent_heldout00337_persistent"), S("heldout00337_baseline")
    claim("heldout_p", "494", "held-out persistent audited 7/10", h1["audited_success_lenient"], h1["audited_lenient_k"], 10, h1["run"])
    claim("heldout_r", "494", "held-out reset audited 6/10", h0["audited_success_lenient"], h0["audited_lenient_k"], 10, h0["run"],
          test=next(f"McNemar p={p['mcnemar_exact_p']} (discordant {p['discordant_a_only']}:{p['discordant_b_only']})" for p in paired if p['comparison'].startswith("held-out")))
    r5, c5 = S("reset5"), S("persistent_continue5")
    claim("rc_reset", "515", "Reset 5/5, mean SPL 0.188, 456.1 s", r5["raw_success"], 5, 5, r5["run"],
          note="rep2 = 4/5 (SPL 0.200, 563.0 s), rep3 = 3/5 (SPL 0.179, 568.0 s) exist and are not reported")
    claim("rc_cont", "516", "Continue 5/5, mean SPL 0.569, 331.0 s", c5["raw_success"], 5, 5, c5["run"],
          note="rep2 = 4/5 (SPL 0.333, 439.8 s), rep3 = 4/5 (SPL 0.544, 408.7 s) exist and are not reported")
    claim("rc_27", "502", "Continue 27% faster (331.0 vs 456.1 s)", round(1 - 331.0 / 456.1, 4), source="sum wall_clock_seconds, reset5 / persistent_continue5",
          note="across 3 reps: reset 456.1/563.0/568.0 s vs continue 331.0/439.8/408.7 s -> 27%/22%/28% faster")
    claim("rc_3x", "502-503", "three times the mean SPL (0.569 vs 0.188)", "0.569 vs 0.188", source="reset5, persistent_continue5",
          test=next(f"Wilcoxon SPL p={p['wilcoxon_signed_rank_p_spl']} (n=5)" for p in paired if p['comparison'].startswith("Continue")))
    claim("decay_ci", "532", "stale entry reliability collapses to 3e-6", 2.73e-06, source="results/controlled_decay_decision_flip.json (C_i_stale=2.7258e-06)")
    claim("perm_p", "559", "mobility trend permutation p=0.109/0.233", "0.109 / 0.233", source="results/mobility_sensitivity_permutation.json")
    claim("dec_equal", "557", "mobility-aware vs uniform 0.50 vs 0.625 audited", "0.50 vs 0.625", source="minconf1default vs no_mobility_minconf1default",
          test=next(f"McNemar p={p['mcnemar_exact_p']}" for p in paired if p['comparison'].startswith("uniform decay")))
    for sh, lab in [("persistent_memage0s", "0h"), ("persistent_memage1h", "1h"), ("persistent_memage3h", "3h"),
                    ("persistent_memage6h", "6h"), ("persistent_memage24h", "24h")]:
        s = S(sh)
        claim(f"age_{lab}", "fig memory-age", f"memory-age {lab} audited success", s["audited_success_lenient"], s["audited_lenient_k"], 8, s["run"],
              note=f"raw {s['reported_successes']}/8; precision {s['audited_lenient_k']}/{s['reported_successes']} {fmt_ci(s['audited_lenient_k'], s['reported_successes'])}; audited SPL {s['mean_spl_audited_lenient']}")
    claim("age_fp", "540-541", "three false positives at 3 h and 6 h (mirror)", 3, source="persistent_memage3h/6h + results/memory_age_sweep_audit.md")
    nv = S("persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification")
    claim("nover_raw", "568-569", "all gates removed: reported success 1.0", nv["raw_success"], 10, 10, nv["run"])
    claim("nover_aud", "569", "all gates removed: audited 0/10", nv["audited_success_lenient"], 0, 10, nv["run"],
          note="precision 0/10, Wilson [0, 0.278]")
    claim("nover_step1", "573-574", "six of ten verify at step 1", sum(1 for e in runs["persistent_minconf0_no_fresh_evidence_guard_no_relevance_gate_no_lexical_gate_full_no_verification"]["episodes"] if e["steps_taken"] == 1), source=nv["run"])
    for sh, lab, line in [("no_lexical_gate_verifyloo_nolexical", "lexical removed", "579"),
                          ("no_relevance_gate_verifyloo_norelevance", "relevance removed", "584"),
                          ("minconf0_verifyloo_nomultiobs", "reconfirmation removed", "584"),
                          ("persistent_no_fresh_evidence_guard_verifyloo_nofreshguard", "fresh-evidence guard removed", "582")]:
        s = S(sh)
        claim(f"loo_{sh}", line, f"verification LOO, {lab}: raw / audited", f"{s['raw_success']} / {s['audited_success_lenient']}",
              s["audited_lenient_k"], 10, s["run"],
              note=f"raw {s['reported_successes']}/10 {fmt_ci(s['reported_successes'], 10)}; precision {s['audited_lenient_k']}/{s['reported_successes']} "
                   f"{fmt_ci(s['audited_lenient_k'], s['reported_successes'])}; strict audited {s['audited_strict_k']}/10")
    claim("loo_fig_045", "fig:verification-ablation", "figure plots 0.45 audited for fresh-evidence-guard removal", 0.45,
          source="scripts/generate_paper_figures.py (hard-coded 'midpoint of 0.4-0.5')", status="NOT AN OBSERVED VALUE",
          note="observed: 4/10 strict or 5/10 lenient; 0.45 cannot occur with n=10")
    b0 = S("semantic00770mined_minconf1default_leakfix")
    claim("loo_base", "579", "baseline audited success 0.5 (all gates)", b0["audited_success_lenient"], b0["audited_lenient_k"], 10, b0["run"],
          note="the fresh-guard ablation is persistent-mode; its matched baseline is the persistent leakfix run (3/10), not 0.5")
    claim("loo_8of9", "580", "eight of nine reported successes wrong (lexical removed)", "8/9", 8, 9, "no_lexical_gate_verifyloo_nolexical")
    bp = S("persistent_semantic00770mined_minconf1default_leakfix")
    claim("adv_np", "610", "non-persistent real 5/8, adv f.p. 0/2, 131.5 s", "5/8", 5, 8, b0["run"], note="adv 0/2 Wilson [0, 0.658]")
    claim("adv_p", "611", "persistent real 3/8, adv f.p. 0/2, 241.6 s", "3/8", 3, 8, bp["run"], note="adv 0/2 Wilson [0, 0.658]")
    claim("adv_pooled", "597", "neither adversarial query ever verified (all gates)", "0/4", 0, 4, "both leakfix runs",
          note="pooled with held-out (0/4 more): 0/8, Wilson [0, 0.324]")
    claim("nvd_050", "599", "no verification discount leaves audited success at 0.50", S("no_verify_discount_minconf1default")["audited_success_lenient"],
          S("no_verify_discount_minconf1default")["audited_lenient_k"], 8, S("no_verify_discount_minconf1default")["run"],
          note="but reported successes 5 vs 6 and precision 4/5 vs 4/6; rep2 of this condition: raw 6/8")
    claim("infl_full", "624", "71-90% of contextual-anchor decisions memory-influenced (full memory)",
          f"{min(rates_full):.3f}-{max(rates_full):.3f}", fb.get("n_influenced"), fb.get("n_decisions"), "minconf1default batch_metrics + logs/phase4_final_minconf1default.log",
          note=f"pooled {fb.get('n_influenced')}/{fb.get('n_decisions')}; of the influenced decisions, {fb.get('n_influenced_with_zero_live_candidates')} had ZERO live candidates; "
               f"with >=1 live candidate: {fb.get('n_influenced_with_live_candidates')}/{fb.get('n_decisions_with_live_candidates')} "
               f"{fmt_ci(fb.get('n_influenced_with_live_candidates', 0), fb.get('n_decisions_with_live_candidates', 0))}; decisions are clustered within episodes")
    claim("infl_pers", "625", "93-100% in persistent mode", f"{min(rates_pers):.3f}-{max(rates_pers):.3f}", pb.get("n_influenced"), pb.get("n_decisions"),
          "persistent_minconf1default + log",
          note=f"zero-live influenced: {pb.get('n_influenced_with_zero_live_candidates')}; with live candidates: {pb.get('n_influenced_with_live_candidates')}/{pb.get('n_decisions_with_live_candidates')} "
               f"{fmt_ci(pb.get('n_influenced_with_live_candidates', 0), pb.get('n_decisions_with_live_candidates', 0))}",
          test=f"per-episode rates persistent vs full: Mann-Whitney p={mw.pvalue:.4f}")
    claim("infl_pilot", "630", "single-room pilot influence 0-15%", str(pilot), sum(a for a, _ in pilot), sum(b for _, b in pilot),
          "logs/phase4_grounding_tuning2.log", note="2 episodes, an older pipeline configuration")
    claim("first_target", "633-635 / 701-702", "first contextual-anchor target differed in 14 of 16 persistent trials (88%)",
          f"{sum(first_p)}/{len(first_p)}", sum(first_p), len(first_p), "logs/phase4_final_minconf1default.log (persistent + persistent repeat #2)",
          note=f"proxy used: first probe decision per query has influenced=True. non-persistent: {sum(first_np)}/{len(first_np)}")
    claim("minconf3", "661", ">=3 confirmations drops success to 0.0", "0/8 (x3 runs)", 0, 8,
          "phase4_batch_minconf3_catgate, minconf4_confirmext4, minconf5_confirmext5", note="each 0/8; pooled 0/24 Wilson [0, 0.138]; runs predate or differ from the reported default config")
    mc = S("minconf1_catgate")
    claim("minconf1_0625", "662-663", "single confirmation reached best audited success (0.625) with no false positives",
          mc["audited_success_lenient"], mc["audited_lenient_k"], 8, mc["run"], status="traced to an older configuration",
          note="true for phase4_batch_minconf1_catgate (catgate-era, pre-leakfix). Under the reported default (min_conf=1, fig:ablation) "
               "full memory = 0.50 with 2 FPs (precision 4/6) and persistent = 0.625 with 1 FP (5/6)")
    claim("lag", "372-373", "DAAAM lag 12.9 s mean (max 21.2 s)", f"{gap['mean']} (max {gap['max']})", source=gap["source"],
          note=f"n={gap['n']} entities, ONE episode; bootstrap95 mean {gap['boot95_mean']}")
    claim("steps", "369", "up to 500 steps", 500, source="batch runs --max-steps 500 (logs/phase4_final_minconf1default.log)")
    claim("area", "433", "299 m^2", 299.44, source=".deps/habitat-matterport-3dresearch/metadata.csv navigable_area (floor_space=572.98)",
          note="299.44 is HM3D 'navigable_area', not floor area")
    claim("concl_10", "696-697", "reported success 10/10, independent audit 0/10", "10/10 vs 0/10", 0, 10, nv["run"],
          status="wording", note="the audit is the author's own, not independent")
    claim("concl_0125_0625", "698", "persistent memory improved audited success 0.125 -> 0.625", "0.125 -> 0.625", source="memory_off_minconf1default -> persistent_minconf1default",
          test=next(f"McNemar p={p['mcnemar_exact_p']} (discordant {p['discordant_a_only']}:{p['discordant_b_only']})" for p in paired if p['comparison'].startswith("persistent vs memory-off")),
          note="compares non-persistent memory-off with persistent (memory + position carry-over)")
    claim("concl_spl", "700", "tripling mean SPL (0.19 to 0.57) and cutting session time by 27%", "0.19 -> 0.57", source="reset5 / persistent_continue5")
    claim("variance25", "673-674", "run-to-run variance up to 25 points", 0.25, source="PROGRESS.md (old min_conf=2 default: 0.125/0.25/0.375 audited)",
          note="under the reported default, raw success varied 0.75 vs 0.50 (full) and 0.75 vs 0.875 (persistent) between repeats")

    # ---------------------------------------------------------------- write
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_episode_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    prec_rows = []
    for s in summ.values():
        r = dict(s)
        for k in list(r):
            if isinstance(r[k], tuple):
                r[k] = f"[{r[k][0]}, {r[k][1]}]" if r[k][0] is not None else ""
        prec_rows.append(r)
    with open(OUT / "precision_by_condition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prec_rows[0].keys()))
        w.writeheader()
        w.writerows(prec_rows)
    keys = sorted({k for p in paired for k in p})
    with open(OUT / "paired_tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(paired)
    with open(OUT / "claims_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(C[0].keys()))
        w.writeheader()
        w.writerows(C)
    (OUT / "reconstruction_check.json").write_text(json.dumps(recon, indent=1))
    (OUT / "all_numbers.json").write_text(json.dumps(dict(
        precision_by_condition=summ, paired_tests=paired, claims=C, reconstruction=recon,
        influence_log=infl, grounding_lag=gap, pilot=pilot,
        first_target_persistent=first_p, first_target_nonpersistent=first_np,
        influence_rates=dict(full=rates_full, persistent=rates_pers, mannwhitney_p=float(mw.pvalue))), indent=1, default=str))

    print("== precision by condition (lenient = as published; strict = cushion/stale counted FP)")
    for s in summ.values():
        print(f"{s['condition'][:44]:44} rep {s['reported_successes']:2}/{s['n_episodes']:2} aud {s['audited_lenient_k']:2} "
              f"prec {s['precision_lenient']} {s['precision_lenient_ci']} strict {s['precision_strict']} "
              f"splaud {s['mean_spl_audited_lenient']}")
    print("== reconstruction"); [print(k, v["subsets_consistent"], v["registry_subset_is_unique_match"]) for k, v in recon.items()]
    print("== paired")
    for p in paired:
        print(f"{p['comparison'][:60]:60} {p['k_a']}/{p['n_pairs']} vs {p['k_b']}/{p['n_pairs']} mcn={p['mcnemar_exact_p']} fis={p['fisher_exact_p']} wil={p.get('wilcoxon_signed_rank_p_spl')}")
    print("== influence", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'per_episode'} for k, v in infl.items()}, indent=0)[:3000])
    print("== gap", gap, "pilot", pilot, "first", sum(first_p), len(first_p), sum(first_np), len(first_np))


if __name__ == "__main__":
    main()
