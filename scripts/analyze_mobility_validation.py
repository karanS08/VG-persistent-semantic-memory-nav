"""Mobility-prior validation (critique.md SS31/Benchmark D), via analysis of
already-collected data -- no new bridge experiments needed. Pulls every real
episode across every batch_metrics.json produced this project (data/runs/ and
results/batch_runs/), maps each query's head noun to its configs/
mobility_priors.json entry, and checks whether verification outcomes actually
correlate with mobility the way the prior assumes (low-mobility categories
should be easier to keep verified/found reliably; high-mobility categories
should show more misses/instance-confusion, since a real object may have
moved since it was last remembered).

This is NOT the same as validating the decay *rate* itself (already covered
by the real controlled_temporal_decay_experiment.py) -- it is the genuinely
separate question CLAUDE.md Section 6 flags: does the offline LLM-generated
mobility number for a category actually predict anything about this
project's real observed outcomes, or is it just a plausible-sounding guess
never checked against data.
"""
import glob
import json
import re

MOBILITY = json.load(open("configs/mobility_priors.json"))["categories"]
ARTICLES = {"a", "an", "the"}


def head_noun(query):
    words = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in ARTICLES]
    return words[-1] if words else None


def mobility_for(query):
    # try the full head noun, then strip common compound prefixes DAAAM/this
    # project's queries use (wall-mounted X, wall X) to find a real dictionary hit
    noun = head_noun(query)
    if noun in MOBILITY:
        return MOBILITY[noun]
    return None


paths = set(glob.glob("data/runs/**/batch_metrics.json", recursive=True)) | \
        set(glob.glob("results/batch_runs/**/batch_metrics.json", recursive=True))

rows = []
for p in paths:
    try:
        d = json.load(open(p))
    except Exception:
        continue
    episodes = d.get("episodes")
    if not episodes:
        continue
    for e in episodes:
        q = e.get("query")
        if not q:
            continue
        m = mobility_for(q)
        if m is None:
            continue
        rows.append({
            "query": q, "mobility": m, "outcome": e.get("outcome"),
            "spl": e.get("spl"), "source": p,
        })

print(f"Total episodes with a resolvable mobility prior: {len(rows)}")
print(f"(from {len(paths)} batch files scanned)\n")

# bucket by mobility: low <=0.1, medium 0.1-0.3, high >0.3 -- matches the
# spec's own low/medium/high framing used in controlled_temporal_decay_experiment.py
buckets = {"low (<=0.1)": [], "medium (0.1-0.3)": [], "high (>0.3)": []}
for r in rows:
    if r["mobility"] <= 0.1:
        buckets["low (<=0.1)"].append(r)
    elif r["mobility"] <= 0.3:
        buckets["medium (0.1-0.3)"].append(r)
    else:
        buckets["high (>0.3)"].append(r)

print(f"{'Bucket':<20} {'n':>5} {'success_rate':>13} {'mean_spl':>10}")
for name, items in buckets.items():
    n = len(items)
    if n == 0:
        print(f"{name:<20} {n:>5}     -             -")
        continue
    succ = sum(1 for i in items if i["outcome"] == "success") / n
    spls = [i["spl"] for i in items if i["spl"] is not None]
    mean_spl = sum(spls) / len(spls) if spls else None
    print(f"{name:<20} {n:>5} {succ:>13.3f} {mean_spl:>10.3f}" if mean_spl is not None
          else f"{name:<20} {n:>5} {succ:>13.3f}          -")

print("\nPer-query breakdown (query, mobility, n, success_rate):")
by_query = {}
for r in rows:
    by_query.setdefault(r["query"], {"mobility": r["mobility"], "n": 0, "succ": 0})
    by_query[r["query"]]["n"] += 1
    if r["outcome"] == "success":
        by_query[r["query"]]["succ"] += 1

for q, v in sorted(by_query.items(), key=lambda kv: kv[1]["mobility"]):
    rate = v["succ"] / v["n"]
    print(f"  mobility={v['mobility']:.2f}  n={v['n']:>3}  success_rate={rate:.3f}  {q!r}")

json.dump({"rows": rows, "buckets": {k: len(v) for k, v in buckets.items()}},
          open("results/mobility_validation_analysis.json", "w"), indent=2)
print("\nwrote results/mobility_validation_analysis.json")
