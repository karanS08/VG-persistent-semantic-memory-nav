"""Score a THOR moved-object matrix (v7, v8, v8a, ...) against THOR ground truth. Same rules as score_v7.py:

True success: the verified entity lies within TOL m of a Sofa footprint at its CURRENT position. A runner 'success'
failing this is a false success. Failures: 'perception' = came within REACH m of a current sofa but never verified;
'budget' = never came that close. Memory-driven steps = steps on the direct_recall branch.
Added: stale steps = steps whose goal is a PRE-EPISODE (seeded) memory, identified by first_observed (seeds are
stamped at seed time; entities seen this episode are stamped >= 300 s later) -- not by id, since seeded ids are
renumbered on load. pipeline-v8 diagnostics (revisit discounts, image-check rejections) are read from the runner log.

Usage: python scripts/thor/score_thor.py v8      -> results/post_paper/v8_scored.json"""
import collections, glob, json, math, os, re, sys

VER = sys.argv[1] if len(sys.argv) > 1 else "v7"
TOL = 1.5
REACH = 1.5
CELLS = ["fresh_static", "a5_static", "a60_static", "fresh_moved", "a5_moved", "a60_moved", "a60_moved_nodecay"]


def d2box(p, o):
    return math.hypot(max(abs(p[0] - o["pos_h"][0]) - o["size"]["x"] / 2, 0), max(abs(p[2] - o["pos_h"][2]) - o["size"]["z"] / 2, 0))


rows = collections.defaultdict(dict)
for log in sorted(glob.glob(f"logs/thor_{VER}_*_s[0-9].log")):
    tag = log[len("logs/thor_"):-4]
    m = re.match(rf"{re.escape(VER)}_(.+)_s(\d)$", tag)
    if not m or m.group(1) not in CELLS:
        continue
    cell, start = m.group(1), int(m.group(2))
    txt = open(log).read()
    o = re.search(r"outcome=(\w+) reason=\w+ steps=(\d+) spl=[\d.]+ branches=\{'direct_recall': (\d+)", txt)
    if not o:
        continue
    runner_ok, steps, mem_steps = o.group(1) == "success", int(o.group(2)), int(o.group(3))
    snap = json.load(open(glob.glob(f"data/runs/phase4_batch_*{tag}/memory_snapshot_ep0.json")[0]))
    gt = json.load(open(glob.glob(f"data/runs/thor_gt_{tag}/gt_*.json")[0]))
    ents = {e["id"]: e for e in snap["entities"]}
    true_ok, dist = False, None
    v = re.findall(r"VERIFIED by entity (\d+)", txt)
    if runner_ok and v and int(v[-1]) in ents:
        dist = min(d2box(ents[int(v[-1])]["position_world"], q) for q in gt if q["type"] == "Sofa")
        true_ok = dist <= TOL
    sofas = [q for q in gt if q["type"] == "Sofa"]
    traj = [json.loads(l) for l in open(glob.glob(f"data/runs/phase4_batch_*{tag}/trajectory_ep0.jsonl")[0])]
    closest = min(d2box([t["xy"][0], 0, t["xy"][1]], q) for t in traj for q in sofas)
    t_seed = min((e["first_observed"] for e in ents.values()), default=0)
    pre = {i for i, e in ents.items() if e["first_observed"] < t_seed + 60} if "fresh" not in cell else set()
    stale_steps = sum(1 for t in traj if t.get("goal_entity_id") in pre)
    kind = "true" if true_ok else "false" if runner_ok else "perception" if closest <= REACH else "budget"
    rows[cell][start] = dict(kind=kind, closest_approach=round(closest, 2), runner=runner_ok, true=true_ok,
                             false=runner_ok and not true_ok, steps=steps, mem_steps=mem_steps, stale_steps=stale_steps,
                             dist=None if dist is None else round(dist, 2),
                             revisit_discounts=len(re.findall(r"\[revisit\] entity", txt)),
                             image_rejects=len(re.findall(r"\[image\] entity \d+ rejected", txt)))

summary = {}
print(f"[{VER}] {'cell':18s} n  true  false  P/B  steps(true,med)  stale steps/ep (mean)  per-start")
for c in CELLS:
    r = rows.get(c, {})
    if not r:
        continue
    n = len(r); t = [x for x in r.values() if x["true"]]
    med = sorted(x["steps"] for x in t)[len(t) // 2] if t else None
    kinds = [x["kind"] for x in r.values()]
    stale = sum(x["stale_steps"] for x in r.values()) / n
    per = " ".join(f"s{k}:{ {'true': 'Y', 'false': 'F', 'perception': 'P', 'budget': 'B'}[r[k]['kind']]}{r[k]['steps']}" for k in sorted(r))
    print(f"[{VER}] {c:18s} {n}  {len(t)}/{n}  {kinds.count('false')}/{n}   {kinds.count('perception')}/{kinds.count('budget')}  "
          f"{str(med):>6s}           {stale:7.1f}               {per}")
    summary[c] = dict(n=n, true_success=len(t), false_success=sum(x["false"] for x in r.values()), median_steps_true=med,
                      mean_stale_steps=round(stale, 1), per_start=r,
                      failure_kinds={k: kinds.count(k) for k in ("perception", "budget")})
tot = [x for r in rows.values() for x in r.values()]
if tot:
    print(f"[{VER}] TOTAL true {sum(x['true'] for x in tot)}/{len(tot)}  false {sum(x['false'] for x in tot)}  "
          f"revisit discounts {sum(x['revisit_discounts'] for x in tot)}  image rejects {sum(x['image_rejects'] for x in tot)}")
json.dump(summary, open(f"results/post_paper/{VER}_scored.json", "w"), indent=1)
if os.path.exists(f"logs/{VER}.excluded"):
    ex = open(f"logs/{VER}.excluded").read().splitlines()
    print(f"\nexcluded attempts (infrastructure rule, re-run): {len(ex)}"); [print("  " + l) for l in ex]
