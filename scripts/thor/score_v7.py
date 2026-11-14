"""Score the v7 moved-object matrix (scripts/thor/launch_v7.sh; derived from score_v6.py) against THOR ground truth.

True success: the verified entity lies within TOL m of a Sofa footprint at its CURRENT position (1.5 m = the verify
distance; DAAAM's median position error after the camera fix is ~0.6 m). A runner 'success' failing this is a false
success. Memory-driven steps = steps on the direct_recall branch (walking to a remembered/seen entry).
Failures are classified from the per-step trajectory: 'perception' = the agent came within REACH m of a current sofa
but never verified it (looked at it, not labelled a sofa); 'budget' = it never came that close. Attempts excluded by the
launcher's pre-set infrastructure rule (< 0.1 captions/step) are listed from logs/v7.excluded.
Writes results/post_paper/v7_summary.json."""
import collections, glob, json, math, re

TOL = 1.5
REACH = 1.5
CELLS = ["fresh_static", "a5_static", "a60_static", "fresh_moved", "a5_moved", "a60_moved", "a60_moved_nodecay"]


def d2box(p, o):
    return math.hypot(max(abs(p[0] - o["pos_h"][0]) - o["size"]["x"] / 2, 0), max(abs(p[2] - o["pos_h"][2]) - o["size"]["z"] / 2, 0))


rows = collections.defaultdict(dict)
for log in sorted(glob.glob("logs/thor_v7_*_s[0-9].log")):
    tag = log[len("logs/thor_"):-4]
    m = re.match(r"v7_(.+)_s(\d)$", tag)
    cell, start = m.group(1), int(m.group(2))
    txt = open(log).read()
    o = re.search(r"outcome=(\w+) reason=\w+ steps=(\d+) spl=[\d.]+ branches=\{'direct_recall': (\d+)", txt)
    if not o:
        continue
    runner_ok, steps, mem_steps = o.group(1) == "success", int(o.group(2)), int(o.group(3))
    true_ok, dist = False, None
    v = re.findall(r"VERIFIED by entity (\d+)", txt)
    if runner_ok and v:
        snap = json.load(open(glob.glob(f"data/runs/phase4_batch_*{tag}/memory_snapshot_ep0.json")[0]))
        gt = json.load(open(glob.glob(f"data/runs/thor_gt_{tag}/gt_*.json")[0]))
        e = [x for x in snap["entities"] if x["id"] == int(v[-1])]
        if e:
            dist = min(d2box(e[0]["position_world"], q) for q in gt if q["type"] == "Sofa")
            true_ok = dist <= TOL
    gt = json.load(open(glob.glob(f"data/runs/thor_gt_{tag}/gt_*.json")[0]))
    sofas = [q for q in gt if q["type"] == "Sofa"]
    traj = [json.loads(l)["xy"] for l in open(glob.glob(f"data/runs/phase4_batch_*{tag}/trajectory_ep0.jsonl")[0])]
    closest = min(d2box([x, 0, z], q) for x, z in traj for q in sofas)
    kind = "true" if true_ok else "false" if (runner_ok and not true_ok) else "perception" if closest <= REACH else "budget"
    rows[cell][start] = dict(kind=kind, closest_approach=round(closest, 2), runner=runner_ok, true=true_ok, false=runner_ok and not true_ok, steps=steps,
                             mem_steps=mem_steps, dist=None if dist is None else round(dist, 2))

summary = {}
print(f"{'cell':20s} n  true  false  fail(perc/budget)  steps(true, median)  memory-driven steps (mean)   per-start")
for c in CELLS:
    r = rows.get(c, {})
    if not r:
        continue
    n = len(r); t = [x for x in r.values() if x["true"]]
    med = sorted(x["steps"] for x in t)[len(t) // 2] if t else None
    ms = sum(x["mem_steps"] for x in r.values()) / n
    per = " ".join(f"s{k}:{ {'true': 'Y', 'false': 'F', 'perception': 'P', 'budget': 'B'}[r[k]['kind']]}{r[k]['steps']}" for k in sorted(r))
    kinds = [x["kind"] for x in r.values()]
    print(f"{c:20s} {n}  {len(t)}/{n}  {kinds.count('false')}/{n}  {kinds.count('perception')}/{kinds.count('budget'):<14d} {str(med):>6s}               {ms:6.1f}                 {per}")
    summary[c] = dict(n=n, true_success=len(t), false_success=sum(x["false"] for x in r.values()), median_steps_true=med,
                      mean_memory_driven_steps=round(ms, 1), per_start=r, failure_kinds={k: kinds.count(k) for k in ("perception", "budget")})
json.dump(summary, open("results/post_paper/v7_summary.json", "w"), indent=1)
import os
if os.path.exists("logs/v7.excluded"):
    ex = open("logs/v7.excluded").read().splitlines()
    print(f"\nexcluded attempts (infrastructure rule, re-run): {len(ex)}"); [print("  " + l) for l in ex]
print("\nY=true success, F=false success (verified entity not on a sofa), P=perception miss (came within 1.5 m, never verified), B=budget (never came within 1.5 m); number = steps used")
