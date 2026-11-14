"""Post-Paper Test Plan 2.2 -- helpful vs harmful memory influence.

Reads decisions_ep*.jsonl (logged since 2026-09-28) from the given run dirs.
For every memory-influenced contextual-anchor decision (with-memory target !=
live-only target) it classifies, in the x-z plane, with R = 2.0 m:
  helpful : memory target within R of a reference location of the queried
            object AND the live-only target is not (or there was none)
  harmful : live-only target within R of a reference AND memory target is not
  neutral : both or neither within R
  undetermined : no reference location known for this query
Reference locations = entity_position_world of every audited-correct (strict)
verified entity for the same query in the same scene, across all instrumented
runs passed in (audit file: results/post_paper/postplan_audit_labels.json,
keyed "<run_dir>/<episode_id>": "strict" | "lenient" | "wrong").
The rule was fixed before any instrumented run finished.
Usage: python scripts/analyze_postplan_influence.py <run_dir> [...]
"""
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path

R = 2.0
runs = [Path(p) for p in sys.argv[1:]]
audit = json.load(open("results/post_paper/postplan_audit_labels.json"))["labels"]

refs = defaultdict(list)  # (scene, query) -> [xz]
for run in runs:
    m = json.load(open(run / "batch_metrics.json"))
    for e in m["episodes"]:
        v = e.get("verification_detail") or {}
        if e["outcome"] == "success" and audit.get(f"{run.name}/{e['episode_id']}") == "strict" \
                and v.get("entity_position_world"):
            p = v["entity_position_world"]; refs[(m["scene_path"], e["query"])].append((p[0], p[2]))


def near(pos, ref_list):
    if pos is None:
        return False
    return min(math.dist((pos[0], pos[2]), r) for r in ref_list) <= R


out = {}
for run in runs:
    m = json.load(open(run / "batch_metrics.json"))
    counts = defaultdict(int); stale = defaultdict(int)
    for f in sorted(run.glob("decisions_ep*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d["memory_influenced"] is None:
                continue
            counts["eligible"] += 1
            if d.get("memory_target_pre_query"):
                stale["memory_target_pre_query"] += 1
            if not d["memory_influenced"]:
                continue
            counts["influenced"] += 1
            # influenced only because nothing is live this step (memory keeps a
            # target seen moments ago) vs. memory overriding a real live option
            counts["influenced_live_pool_empty" if d["live_target_entity_id"] is None
                   else "influenced_overriding_live"] += 1
            if d.get("memory_target_pre_query"):
                counts["influenced_by_pre_query_memory"] += 1
            ref = refs.get((m["scene_path"], d["query"]))
            if not ref:
                counts["undetermined"] += 1; continue
            mem_ok, live_ok = near(d["memory_target_position"], ref), near(d["live_target_position"], ref)
            counts["helpful" if mem_ok and not live_ok else "harmful" if live_ok and not mem_ok else "neutral"] += 1
    c = dict(counts); det = c.get("influenced", 0) - c.get("undetermined", 0)
    c["influence_rate"] = round(c.get("influenced", 0) / c["eligible"], 3) if c.get("eligible") else None
    for k in ("helpful", "harmful", "neutral"):
        c[f"{k}_rate_of_influenced"] = round(c.get(k, 0) / c["influenced"], 3) if c.get("influenced") else None
        c[f"{k}_rate_of_determined"] = round(c.get(k, 0) / det, 3) if det > 0 else None
    c["stale_target_rate_of_eligible"] = round(stale["memory_target_pre_query"] / c["eligible"], 3) if c.get("eligible") else None
    out[run.name] = c
    print(run.name, json.dumps(c))
Path("results/post_paper/influence_summary.json").write_text(json.dumps(
    {"_doc": __doc__, "R_m": R, "n_reference_queries": len(refs), "runs": out}, indent=2))
