"""Score the v6 moved-object matrix (scripts/thor/launch_v6.sh) against THOR ground truth.

True success: the verified entity lies within TOL m of a Sofa footprint at its CURRENT position (1.5 m = the verify
distance; DAAAM's median position error after the camera fix is ~0.6 m). A runner 'success' failing this is a false
success. Memory-driven steps = steps on the direct_recall branch (walking to a remembered/seen entry).
Writes results/post_paper/v6_summary.json."""
import collections, glob, json, math, re

TOL = 1.5
CELLS = ["fresh_static", "a5_static", "a60_static", "fresh_moved", "a5_moved", "a60_moved", "a60_moved_nodecay"]


def d2box(p, o):
    return math.hypot(max(abs(p[0] - o["pos_h"][0]) - o["size"]["x"] / 2, 0), max(abs(p[2] - o["pos_h"][2]) - o["size"]["z"] / 2, 0))


rows = collections.defaultdict(dict)
for log in sorted(glob.glob("logs/thor_v6_*_s[0-9].log")):
    tag = log[len("logs/thor_"):-4]
    m = re.match(r"v6_(.+)_s(\d)$", tag)
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
    rows[cell][start] = dict(runner=runner_ok, true=true_ok, false=runner_ok and not true_ok, steps=steps,
                             mem_steps=mem_steps, dist=None if dist is None else round(dist, 2))

summary = {}
print(f"{'cell':20s} n  true  false  steps(true, median)  memory-driven steps (mean)   per-start true/steps")
for c in CELLS:
    r = rows.get(c, {})
    if not r:
        continue
    n = len(r); t = [x for x in r.values() if x["true"]]
    med = sorted(x["steps"] for x in t)[len(t) // 2] if t else None
    ms = sum(x["mem_steps"] for x in r.values()) / n
    per = " ".join(f"s{k}:{'Y' if r[k]['true'] else ('F' if r[k]['false'] else '-')}{r[k]['steps']}" for k in sorted(r))
    print(f"{c:20s} {n}  {len(t)}/{n}  {sum(x['false'] for x in r.values())}/{n}   {str(med):>6s}               {ms:6.1f}                 {per}")
    summary[c] = dict(n=n, true_success=len(t), false_success=sum(x["false"] for x in r.values()), median_steps_true=med,
                      mean_memory_driven_steps=round(ms, 1), per_start=r)
json.dump(summary, open("results/post_paper/v6_summary.json", "w"), indent=1)
print("\nY=true success, F=false success (verified entity not on a sofa), -=no success; number = steps used")
