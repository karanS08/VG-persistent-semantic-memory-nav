"""Post-Paper-Draft Test 2.2 (part 2) -- decision-level helpful / harmful / neutral /
undetermined classification of memory-influenced decisions, by navmesh geodesic
distance. OFFLINE: loads only a PathFinder from the scene's .navmesh (no renderer,
no Simulator, no DAAAM bridge).

INPUT REQUIREMENT: per-decision logs data/runs/<run>/decisions_ep<k>.jsonl, written by
the 2026-09-28 logging-only instrumentation in run_phase4_batch.py (fields
memory_target_position, live_target_position, agent_position, memory_influenced) and
verification_detail.entity_position_world in batch_metrics.json. Runs made before
2026-09-28 do NOT have these fields -> this script refuses them (exit code 2) rather
than guessing targets.

Criterion (documented in REPORT.md):
  G   = navmesh-snapped position of the entity that ended the episode, used ONLY if the
        episode succeeded AND a human audit marked that match correct (--audit CSV,
        column 'HUMAN_correct_object(Y/N/partial)' == 'Y'); otherwise every influenced
        decision in the episode is 'undetermined' (failed / not audited / audited wrong).
  d(x)= habitat_sim geodesic distance from snap(x) to G.
  If a live-only target existed:   delta = d(memory_target) - d(live_target)
  If the live-only set was empty:  delta = d(memory_target) - d(agent_position)
        (the live-only counterfactual is frontier fallback, whose goal is not logged;
         this asks whether memory's target moves the goal toward the eventual find
         relative to staying put -- reported as a separate sub-type)
  helpful if delta < -margin, harmful if delta > +margin, neutral otherwise.
  margin default 1.5 m = verify_distance in configs/run_defaults.yaml.

Usage:
  python scripts/postdraft/classify_influence_geodesic.py --runs <run_dir> [...] \
      --audit data/results_postdraft/factorial_influence/<audit>.csv [--margin 1.5] \
      [--out data/results_postdraft/factorial_influence/geodesic_influence.json]
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_pathfinder(scene_glb_path):
    import habitat_sim  # verified API (habitat_sim 0.3.3): nav.PathFinder.load_nav_mesh,
    # snap_point, find_path(ShortestPath) -> ShortestPath.geodesic_distance
    navmesh = Path(scene_glb_path).with_suffix(".navmesh")
    if not navmesh.is_absolute():
        navmesh = ROOT / navmesh
    pf = habitat_sim.nav.PathFinder()
    if not pf.load_nav_mesh(str(navmesh)):
        raise RuntimeError(f"failed to load {navmesh}")
    return pf


def snap(pf, p):
    s = pf.snap_point(np.array(p, dtype=np.float32))
    s = [float(x) for x in s]
    return None if any(math.isnan(x) for x in s) else s


def geodesic(pf, a, b):
    import habitat_sim
    sa, sb = snap(pf, a), snap(pf, b)
    if sa is None or sb is None:
        return math.inf
    sp = habitat_sim.ShortestPath()
    sp.requested_start = sa
    sp.requested_end = sb
    if not pf.find_path(sp):
        return math.inf
    return float(sp.geodesic_distance)


def load_audit(path):
    ok = {}
    if path is None:
        return ok
    with open(path) as f:
        for r in csv.DictReader(f):
            lab = (r.get("HUMAN_correct_object(Y/N/partial)") or "").strip().upper()
            if lab:
                ok[(r["run_dir"], int(r["episode_id"]))] = lab
    return ok


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def classify_run(run, pf_cache, audit, margins):
    rd = ROOT / "data/runs" / run
    bm = json.load(open(rd / "batch_metrics.json"))
    files = sorted(rd.glob("decisions_ep*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{run}: no decisions_ep*.jsonl -- run predates 2026-09-28 instrumentation")
    scene = bm["scene_path"]
    pf = pf_cache.setdefault(scene, load_pathfinder(scene))
    ep_by_id = {e["episode_id"]: e for e in bm["episodes"]}
    rows = []
    for fp in files:
        for line in open(fp):
            d = json.loads(line)
            if not d.get("memory_influenced"):
                continue
            ep = ep_by_id[d["episode_id"]]
            vd = ep.get("verification_detail") or {}
            lab = audit.get((run, d["episode_id"]), "")
            row = {"run": run, "episode_id": d["episode_id"], "query": d["query"], "step": d["step"],
                   "memory_target_entity_id": d["memory_target_entity_id"],
                   "live_target_entity_id": d["live_target_entity_id"],
                   "memory_target_pre_query": d.get("memory_target_pre_query"),
                   "comparison": "vs_live_target" if d.get("live_target_position") else "vs_current_position(live_empty)"}
            if ep["outcome"] != "success":
                row["reason"] = "episode_failed"
            elif lab != "Y":
                row["reason"] = "not_human_audited" if lab == "" else f"human_audit={lab}"
            elif not vd.get("entity_position_world"):
                row["reason"] = "verified_position_not_logged"
            else:
                G = vd["entity_position_world"]
                d_mem = geodesic(pf, d["memory_target_position"], G)
                ref = d["live_target_position"] or d["agent_position"]
                d_ref = geodesic(pf, ref, G)
                row.update({"d_memory_target_m": d_mem, "d_reference_m": d_ref})
                if math.isinf(d_mem) or math.isinf(d_ref):
                    row["reason"] = "unreachable_on_navmesh"
                else:
                    row["delta_m"] = round(d_mem - d_ref, 3)
                    row["reason"] = "classified"
            for m in margins:
                if row["reason"] != "classified":
                    row[f"label@{m}"] = "undetermined"
                else:
                    dl = row["delta_m"]
                    row[f"label@{m}"] = "helpful" if dl < -m else ("harmful" if dl > m else "neutral")
            rows.append(row)
    return rows


def summarise(rows, margins):
    out = {}
    for m in margins:
        key = f"label@{m}"
        per_dec = Counter(r[key] for r in rows)
        n = len(rows)
        # de-duplicated by (run, episode, memory target): the same target is re-chosen
        # on many consecutive replans; majority label per unique target
        uniq = defaultdict(list)
        for r in rows:
            uniq[(r["run"], r["episode_id"], r["memory_target_entity_id"], r["comparison"])].append(r[key])
        per_target = Counter(Counter(v).most_common(1)[0][0] for v in uniq.values())
        out[key] = {
            "per_decision": {k: {"n": per_dec[k], "rate": round(per_dec[k] / n, 4) if n else None,
                                 "wilson95": wilson(per_dec[k], n)} for k in ("helpful", "harmful", "neutral", "undetermined")},
            "per_unique_target": dict(per_target),
            "by_comparison": {c: dict(Counter(r[key] for r in rows if r["comparison"] == c))
                              for c in {r["comparison"] for r in rows}},
        }
    out["undetermined_reasons"] = dict(Counter(r["reason"] for r in rows if r["reason"] != "classified"))
    out["n_influenced_decisions"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--audit", default=None, help="human audit CSV (factorial_audit_sheet.csv format)")
    ap.add_argument("--margins", nargs="+", type=float, default=[1.5, 0.5, 3.0])
    ap.add_argument("--out", default=str(ROOT / "data/results_postdraft/factorial_influence/geodesic_influence.json"))
    a = ap.parse_args()
    audit = load_audit(a.audit)
    pf_cache, rows, refused = {}, [], {}
    for run in a.runs:
        try:
            rows.extend(classify_run(run, pf_cache, audit, a.margins))
        except FileNotFoundError as e:
            refused[run] = str(e)
    if not rows:
        print(json.dumps({"refused": refused}, indent=1))
        sys.exit(2)
    res = {"runs": a.runs, "refused": refused, "margins": a.margins, "summary": summarise(rows, a.margins)}
    json.dump(res, open(a.out, "w"), indent=1)
    with open(Path(a.out).with_suffix(".csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
