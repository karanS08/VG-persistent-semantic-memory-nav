"""Re-score moved-sofa runs: a runner 'success' counts only if the verified entity lies within TOL m of a Sofa footprint at its CURRENT GT position."""
import glob, json, math, re, collections
TOL = 1.0
def d2box(p, o):
    return math.hypot(max(abs(p[0]-o["pos_h"][0]) - o["size"]["x"]/2, 0), max(abs(p[2]-o["pos_h"][2]) - o["size"]["z"]/2, 0))
rows = collections.defaultdict(list)
for log in sorted(glob.glob("logs/thor_ms_*.log")):
    tag = log.split("thor_")[1][:-4]; cond = tag.split("_")[1]
    txt = open(log).read()
    m = re.search(r"outcome=(\w+) reason=(\w+) steps=(\d+) spl=([\d.]+)", txt)
    if not m: continue  # run still in progress
    runner_ok = m.group(1) == "success"; steps = int(m.group(3)); spl = float(m.group(4))
    v = re.findall(r"VERIFIED by entity (\d+)", txt); true_ok, where, stale = False, None, False
    if runner_ok and v:
        snap = json.load(open(glob.glob(f"data/runs/phase4_batch_*{tag}/memory_snapshot_ep0.json")[0]))
        gt = json.load(open(glob.glob(f"data/runs/thor_gt_{tag}/gt_*.json")[0]))
        sofas = [o for o in gt if o["type"] == "Sofa"]
        e = [x for x in snap["entities"] if x["id"] == int(v[-1])]
        if e:
            stale = e[0]["id"] >= 900_000_000
            p = e[0]["position_world"]; dd = min(d2box(p, o) for o in sofas)
            true_ok = dd <= TOL; where = (round(p[0], 1), round(p[2], 1), round(dd, 2))
    rows[cond].append((tag, runner_ok, true_ok, steps, spl, where, stale))
for c in ("static", "fresh", "full", "nodecay", "memoff"):
    if c not in rows: continue
    r = rows[c]
    print(f"\n{c}: runner-success {sum(x[1] for x in r)}/{len(r)}  true-success (verified entity on a sofa at its current position) {sum(x[2] for x in r)}/{len(r)}")
    for t, ro, to, st, sp, w, sl in r: print(f"   {t:16s} runner={ro!s:5} true={to!s:5} steps={st:3d} spl={sp:.2f} verified_at(x,z,dist_to_sofa)={w} stale_seed_entity={sl}")
    print(f"   verified a STALE seed entity: {sum(x[6] for x in r)}/{len(r)}   median steps (true successes): {sorted(x[3] for x in r if x[2])[len([1 for x in r if x[2]])//2] if any(x[2] for x in r) else None}")
    ok = [x for x in r if x[2]]
    print(f"   mean SPL over all 4 (failures=0, true-success only): {sum(x[4] for x in ok)/len(r):.2f}")
json.dump({c: rows[c] for c in rows}, open("results/post_paper/movedsofa_summary.json", "w"), indent=1)
