"""Sensitivity check for the mobility-prior validation in
results/mobility_validation_analysis.md (2026-09-06), answering the paper's
own caveat there ("not a controlled test") and closing the main.tex TODO at
Section 7.4 ("sensitivity run with perturbed or shuffled mobility priors").

Re-running the full 1550/1570-episode dataset under new mobility values would
mean re-simulating ~1550 real closed-loop episodes, which is infeasible to
redo just for a sensitivity check. Instead this reuses the SAME real,
already-collected per-query (mobility, n, success_rate) table
analyze_mobility_validation.py already produced, and asks the question that
script's own "Important causal caveat" left open: is the observed monotonic
low->medium->high success-rate trend actually driven by the real mobility
assignment, or would shuffling which query gets which mobility value produce
a similarly strong trend just by chance (i.e. is 14 data points enough for
one lucky/unlucky permutation to look this clean)?

Method: permutation test. Fix each of the 14 well-sampled queries' own (n,
successes) exactly as observed -- only the mobility VALUE attached to each
query is randomly reassigned (a permutation of the same 14 real values across
the same 14 queries), many times. For each shuffle, recompute the low/medium/
high buckets (same thresholds as the real analysis: <=0.1, 0.1-0.3, >0.3) and
the Spearman rank correlation between per-query mobility and per-query
success rate. Compare the real assignment's statistics against this null
distribution to get an exact permutation p-value.
"""
import json
import random
import re
from pathlib import Path

random.seed(20260927)

MOBILITY = json.load(open("configs/mobility_priors.json"))["categories"]
ARTICLES = {"a", "an", "the"}


def head_noun(query):
    words = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in ARTICLES]
    return words[-1] if words else None


def mobility_for(query):
    noun = head_noun(query)
    return MOBILITY.get(noun)


# Reuse the exact same data source as analyze_mobility_validation.py so this
# operates on the identical 1570-episode real dataset, not a re-derivation.
import glob

paths = set(glob.glob("data/runs/**/batch_metrics.json", recursive=True)) | \
        set(glob.glob("results/batch_runs/**/batch_metrics.json", recursive=True))

by_query = {}
for p in paths:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    for e in d.get("episodes") or []:
        q = e.get("query")
        if not q:
            continue
        m = mobility_for(q)
        if m is None:
            continue
        rec = by_query.setdefault(q, {"mobility": m, "n": 0, "succ": 0, "spls": []})
        rec["n"] += 1
        if e.get("outcome") == "success":
            rec["succ"] += 1
        if e.get("spl") is not None:
            rec["spls"].append(e["spl"])

# Restrict to well-sampled queries only, matching the original analysis'
# own robustness check ("restricting to the 14 individual queries with >=20
# real episodes each -- no singleton/n=4 noise").
MIN_N = 20
queries = sorted([q for q, v in by_query.items() if v["n"] >= MIN_N],
                  key=lambda q: by_query[q]["mobility"])
n_total = sum(by_query[q]["n"] for q in queries)
print(f"{len(queries)} queries with n>={MIN_N} real episodes each, {n_total} episodes total")
for q in queries:
    v = by_query[q]
    print(f"  mobility={v['mobility']:.2f}  n={v['n']:>4}  success_rate={v['succ']/v['n']:.3f}  {q!r}")

real_mobilities = [by_query[q]["mobility"] for q in queries]
ns = [by_query[q]["n"] for q in queries]
succs = [by_query[q]["succ"] for q in queries]
rates = [s / n for s, n in zip(succs, ns)]


def bucket_rates(mobilities):
    buckets = {"low": [0, 0], "medium": [0, 0], "high": [0, 0]}  # [succ, n]
    for m, s, n in zip(mobilities, succs, ns):
        key = "low" if m <= 0.1 else ("medium" if m <= 0.3 else "high")
        buckets[key][0] += s
        buckets[key][1] += n
    return {k: (v[0] / v[1] if v[1] else None) for k, v in buckets.items()}


def spearman(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


real_buckets = bucket_rates(real_mobilities)
real_spread = real_buckets["low"] - real_buckets["high"]
real_rho = spearman(real_mobilities, rates)
print("\nReal assignment:")
print(f"  buckets: {real_buckets}")
print(f"  low-high spread: {real_spread:.4f}")
print(f"  Spearman rho(mobility, success_rate): {real_rho:.4f}  "
      f"(negative = higher mobility -> lower success, the prior's predicted direction)")

N_PERM = 200000
spreads = []
rhos = []
for _ in range(N_PERM):
    shuffled = real_mobilities[:]
    random.shuffle(shuffled)
    b = bucket_rates(shuffled)
    spreads.append(b["low"] - b["high"])
    rhos.append(spearman(shuffled, rates))

# two-sided permutation p-value: how often does a random reassignment produce
# a spread/rho at least as extreme (in the observed direction) as the real one
p_spread = sum(1 for s in spreads if s >= real_spread) / N_PERM
p_rho = sum(1 for r in rhos if r <= real_rho) / N_PERM  # real_rho is negative

spreads_sorted = sorted(spreads)
rhos_sorted = sorted(rhos)


def percentile_rank(sorted_vals, x):
    import bisect
    return bisect.bisect_left(sorted_vals, x) / len(sorted_vals)


print(f"\nPermutation null distribution ({N_PERM} shuffles of the mobility "
      f"value <-> query mapping, success-rate table held fixed):")
print(f"  spread: mean={sum(spreads)/N_PERM:.4f}  "
      f"[5th,95th]=[{spreads_sorted[int(0.05*N_PERM)]:.4f}, {spreads_sorted[int(0.95*N_PERM)]:.4f}]")
print(f"  rho:    mean={sum(rhos)/N_PERM:.4f}  "
      f"[5th,95th]=[{rhos_sorted[int(0.05*N_PERM)]:.4f}, {rhos_sorted[int(0.95*N_PERM)]:.4f}]")
print(f"\n  real spread {real_spread:.4f} is at percentile "
      f"{percentile_rank(spreads_sorted, real_spread)*100:.2f} of the null distribution "
      f"(one-sided p = {p_spread:.5f} for spread >= real)")
print(f"  real rho {real_rho:.4f} is at percentile "
      f"{percentile_rank(rhos_sorted, real_rho)*100:.2f} of the null distribution "
      f"(one-sided p = {p_rho:.5f} for rho <= real)")

out = {
    "min_n_per_query": MIN_N,
    "n_queries": len(queries),
    "n_episodes_total": n_total,
    "per_query": {q: {"mobility": by_query[q]["mobility"], "n": by_query[q]["n"],
                       "success_rate": by_query[q]["succ"] / by_query[q]["n"]}
                  for q in queries},
    "real_buckets": real_buckets,
    "real_spread_low_minus_high": real_spread,
    "real_spearman_rho": real_rho,
    "n_permutations": N_PERM,
    "permutation_seed": 20260927,
    "null_spread_mean": sum(spreads) / N_PERM,
    "null_spread_p5_p95": [spreads_sorted[int(0.05 * N_PERM)], spreads_sorted[int(0.95 * N_PERM)]],
    "null_rho_mean": sum(rhos) / N_PERM,
    "null_rho_p5_p95": [rhos_sorted[int(0.05 * N_PERM)], rhos_sorted[int(0.95 * N_PERM)]],
    "p_value_spread_one_sided": p_spread,
    "p_value_rho_one_sided": p_rho,
}
Path("results/mobility_sensitivity_permutation.json").write_text(json.dumps(out, indent=2))
print("\nwrote results/mobility_sensitivity_permutation.json")
