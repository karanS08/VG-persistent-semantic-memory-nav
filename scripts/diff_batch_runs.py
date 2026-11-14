#!/usr/bin/env python
"""Diffs two batch_metrics.json files episode-by-episode on the fields that
matter for a determinism/variance check: outcome, steps_taken, branch_counts,
and which entity (if any) triggered verification. Written for TEST_PLAN.md's
Tier 2 (repeated identical configs) -- run_phase4_batch.py seeds
random.seed(episode_id) per episode, but DAAAM's grounding is a live,
timing-dependent async VLM call, so whether repeats are actually
bit-identical downstream is an open, unverified question this script answers
directly instead of assuming either way.

Usage: python scripts/diff_batch_runs.py <run_a>/batch_metrics.json <run_b>/batch_metrics.json
"""
from __future__ import annotations

import json
import sys


FIELDS = ["outcome", "reason", "steps_taken", "spl", "branch_counts"]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    a, b = load(sys.argv[1]), load(sys.argv[2])
    eps_a = {e["episode_id"]: e for e in a["episodes"]}
    eps_b = {e["episode_id"]: e for e in b["episodes"]}

    print(f"A: {sys.argv[1]} (success_rate={a['success_rate']}, mean_spl={a['mean_spl']})")
    print(f"B: {sys.argv[2]} (success_rate={b['success_rate']}, mean_spl={b['mean_spl']})")
    print()

    identical_count = 0
    for eid in sorted(set(eps_a) | set(eps_b)):
        ea, eb = eps_a.get(eid), eps_b.get(eid)
        if ea is None or eb is None:
            print(f"episode {eid}: MISSING from one run")
            continue

        diffs = []
        for field in FIELDS:
            if ea.get(field) != eb.get(field):
                diffs.append(f"{field}: {ea.get(field)!r} != {eb.get(field)!r}")

        vid_a = (ea.get("verification_detail") or {}).get("entity_id")
        vid_b = (eb.get("verification_detail") or {}).get("entity_id")
        if vid_a != vid_b:
            diffs.append(f"verification entity_id: {vid_a!r} != {vid_b!r}")

        if diffs:
            print(f"episode {eid} ({ea['query']!r}): DIFFERS")
            for d in diffs:
                print(f"    {d}")
        else:
            identical_count += 1

    n = len(set(eps_a) | set(eps_b))
    print(f"\n{identical_count}/{n} episodes identical on {FIELDS} + verification entity_id.")
    if identical_count == n:
        print("=> Fully deterministic on the fields checked for this pair of runs.")
    else:
        print("=> NOT deterministic -- see per-episode diffs above.")


if __name__ == "__main__":
    main()
