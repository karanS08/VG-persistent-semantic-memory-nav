#!/usr/bin/env python3
"""2026-09-04: aggregates every coverage-memory ON vs OFF batch run so far --
both the interactive ones from this afternoon (longsession10_covmem*,
longsession10_samedaybaseline*) and scripts/run_coverage_memory_megabatch.sh's
output (megabatch_on_rep*, megabatch_off_rep*) -- into one comparison, since
they're all independent repeats of the identical config (same 10-query list/
order, same scene, --persistent-memory, only --no-coverage-memory differing).

Usage: python scripts/coverage_memory_ab_summary.py
"""
import glob
import json
import statistics

ON_PATTERNS = [
    # NOT "..._longsession10_covmem" (no suffix) -- that's v1, which hit the
    # memory_size_at_end=0 bridge-hangup bug (see logs/failures.md) and is
    # not a valid data point, only its _v2/_rep* successors are.
    "data/runs/phase4_batch_persistent_longsession10_covmem_v2",
    "data/runs/phase4_batch_persistent_longsession10_covmem_rep*",
    "data/runs/phase4_batch_persistent_megabatch_on_rep*",
]
OFF_PATTERNS = [
    "data/runs/phase4_batch_persistent_no_coverage_memory_longsession10_samedaybaseline",
    "data/runs/phase4_batch_persistent_no_coverage_memory_longsession10_samedaybaseline_rep*",
    # NOTE: --no-coverage-memory is appended to ablation_suffixes BEFORE
    # --run-tag in run_phase4_batch.py, so the real dir is
    # phase4_batch_persistent_no_coverage_memory_<tag>, NOT
    # phase4_batch_persistent_<tag>_no_coverage_memory or
    # phase4_batch_persistent_megabatch_off_rep* (verified 2026-09-04 via
    # `ls data/runs/*megabatch*` after the first version of this pattern
    # silently dropped all 10 megabatch OFF runs from the comparison).
    "data/runs/phase4_batch_persistent_no_coverage_memory_megabatch_off_rep*",
]
# NOTE on the ON list: the very first covmem run (pre-crash, "v1") is
# EXCLUDED on purpose -- that one hit the memory_size_at_end=0 bridge-hangup
# bug (see logs/failures.md), so it's not a valid data point for this
# comparison, not a coverage-memory result at all.


def collect(patterns):
    rows = []
    seen = set()
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            f = d + "/batch_metrics.json"
            if d in seen:
                continue
            seen.add(d)
            try:
                with open(f) as fh:
                    m = json.load(fh)
            except FileNotFoundError:
                continue
            rows.append((d, m["success_rate"], m["mean_spl"]))
    return rows


def summarize(label, rows):
    print(f"\n{label} (n={len(rows)}):")
    for d, sr, spl in rows:
        print(f"  {d:<80} success_rate={sr:.2f}  mean_spl={spl:.4f}")
    if not rows:
        print("  (none yet)")
        return None, None
    srs = [r[1] for r in rows]
    spls = [r[2] for r in rows]
    sr_mean = statistics.mean(srs)
    spl_mean = statistics.mean(spls)
    sr_sd = statistics.stdev(srs) if len(srs) > 1 else float("nan")
    spl_sd = statistics.stdev(spls) if len(spls) > 1 else float("nan")
    print(f"  MEAN success_rate={sr_mean:.4f} (sd={sr_sd:.4f})   MEAN mean_spl={spl_mean:.4f} (sd={spl_sd:.4f})")
    return (sr_mean, sr_sd, len(srs), srs), (spl_mean, spl_sd, len(spls), spls)


def welch_t(a, b):
    """a, b: (mean, sd, n, values). Returns (t, approx_df) or None if underpowered."""
    m1, s1, n1, _ = a
    m2, s2, n2, _ = b
    if n1 < 2 or n2 < 2 or s1 != s1 or s2 != s2:  # nan check
        return None
    se = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 0.5
    if se == 0:
        return None
    t = (m1 - m2) / se
    # Welch-Satterthwaite df
    num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
    den = (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
    df = num / den if den > 0 else float("nan")
    return t, df


if __name__ == "__main__":
    on_rows = collect(ON_PATTERNS)
    off_rows = collect(OFF_PATTERNS)
    on_sr, on_spl = summarize("COVERAGE MEMORY ON", on_rows)
    off_sr, off_spl = summarize("COVERAGE MEMORY OFF", off_rows)

    print("\n=== Welch's t (rough, unequal-variance) ===")
    for name, a, b in [("success_rate", on_sr, off_sr), ("mean_spl", on_spl, off_spl)]:
        if a is None or b is None:
            print(f"  {name}: not enough data yet")
            continue
        result = welch_t(a, b)
        if result is None:
            print(f"  {name}: n too small (need >=2 per side with nonzero variance)")
        else:
            t, df = result
            print(f"  {name}: ON mean={a[0]:.4f} (n={a[2]}) vs OFF mean={b[0]:.4f} (n={b[2]}) -- t={t:.3f}, df~{df:.1f}"
                  f" (rule of thumb: |t|>~2 starts to look real at df>~10, need |t|>~3-4 at small df)")
