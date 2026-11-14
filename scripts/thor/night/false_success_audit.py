"""Classify each false success: stale (verified a pre-episode memory near the OLD sofa spot), near-old-spot (live entry
near old spot), or mis-caption elsewhere. Usage: python false_success_audit.py VER [VER...]"""
import glob, json, math, re, sys, collections
for VER in sys.argv[1:]:
    house = {"t2": 1, "t1l0": 1, "t1l018": 1, "t1l055": 1, "v7": 1}.get(VER) or int(re.match(r"h(\d+)", VER).group(1))
    if house != 1:
        cfg = json.load(open(f"configs/thor_night/house_{house}.json")); ox, oz = cfg["sofa_orig_thor_xz"]
    else:
        ox = oz = None
    cnt = collections.Counter()
    for log in sorted(glob.glob(f"logs/thor_{VER}_*_s[0-9].log")):
        tag = log[len("logs/thor_"):-4]; txt = open(log).read()
        if not re.search(r"outcome=success", txt): continue
        v = re.findall(r"VERIFIED by entity (\d+)", txt)
        snap = json.load(open(glob.glob(f"data/runs/phase4_batch_*{tag}/memory_snapshot_ep0.json")[0]))
        gt = json.load(open(glob.glob(f"data/runs/thor_gt_{tag}/gt_*.json")[0]))
        ents = {e["id"]: e for e in snap["entities"]}; e = ents.get(int(v[-1])) if v else None
        if e is None: print(VER, tag, "verified entity missing"); continue
        p = e["position_world"]; sofas = [q for q in gt if q["type"] == "Sofa"]
        dnow = min(math.hypot(max(abs(p[0]-s["pos_h"][0])-s["size"]["x"]/2,0), max(abs(p[2]-s["pos_h"][2])-s["size"]["z"]/2,0)) for s in sofas)
        if dnow <= 1.5: continue
        t_seed = min(x["first_observed"] for x in ents.values())
        stale = "fresh" not in tag and e["first_observed"] < t_seed + 60
        dold = math.hypot(p[0]-ox, -p[2]-oz) if ox is not None else None
        near = [ (q["type"], round(math.hypot(p[0]-q["pos_h"][0], p[2]-q["pos_h"][2]),2)) for q in gt]
        near = sorted(near, key=lambda x: x[1])[:2]
        k = "STALE" if stale else ("live@oldspot" if dold is not None and dold < 2 else "miscaption")
        cnt[k] += 1
        print(f"{VER:7s} {tag[len(VER)+1:]:24s} {k:12s} d_now={dnow:.1f} d_old={'-' if dold is None else round(dold,1)} nearest={near} | {e['description'][:60]}")
    print(VER, dict(cnt)); print()
