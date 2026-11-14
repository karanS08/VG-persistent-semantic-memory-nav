"""Score the moving-object runs (launch_movedobj.sh). Per run & episode: outcome, steps, whether the verified entity lies on the
object's TRUE position at that time (<=1.5 m, xz) or on its STALE (original) position, and how many decisions targeted a remembered
position near the stale spot. Writes results/post_paper/movedobj_summary.json and prints a condition table.
Frames: all positions are Habitat-convention (z_h = -z_thor)."""
import glob, json, math, os, statistics as st, sys
OIDS = {"tbs": "TeddyBear|surface|8|76", "tb": "TeddyBear|surface|8|76", "wm": "WashingMachine|11|3"}  # target per run-tag prefix; stale/true positions come from the GT dumps
def d2(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
rows = []
for d in sorted(glob.glob("data/runs/phase4_batch_*_r[0-9]")):
    import re
    mt = re.search(r"((?:tbs|tb|wm)_[a-z]+_r\d)$", os.path.basename(d))  # runner prefixes the folder with the mode/flags
    if not (mt and os.path.exists(f"{d}/batch_metrics.json")): continue
    tag = mt.group(1)
    cond, rep = tag.rsplit("_r", 1); m = json.load(open(f"{d}/batch_metrics.json"))
    for e in m["episodes"]:
        k = e["episode_id"]; OID = OIDS[tag.split("_")[0]]
        def pos_at(i):
            f = f"data/runs/thor_gt_{tag}/gt_{i}.json"
            o = next((o for o in json.load(open(f)) if o["id"] == OID), None) if os.path.exists(f) else None
            return (o["pos_h"][0], o["pos_h"][2]) if o else None
        true = pos_at(k + 1)
        STALE = (lambda f: pos_at(1) if tag.split("_")[0] != "tbs" else (lambda o: (o["pos_h"][0], o["pos_h"][2]))(next(o for o in json.load(open("data/runs/thor_gt_tbs_static_r1/gt_1.json")) if o["id"] == OID)))(None)
        v = e.get("verification_detail") or {}; p = v.get("entity_position_world")
        ok = e["outcome"] == "success" and p is not None
        dec = [json.loads(l) for l in open(f"{d}/decisions_ep{k}.jsonl")] if os.path.exists(f"{d}/decisions_ep{k}.jsonl") else []
        stale_t = sum(1 for x in dec if x.get("memory_target_position") and d2((x["memory_target_position"][0], x["memory_target_position"][2]), STALE) <= 2.0)
        rows.append({"cond": cond, "target": tag.split("_")[0], "rep": int(rep), "episode": k, "outcome": e["outcome"], "steps": e["steps_taken"], "spl": e["spl"],
                     "found_on_true_pos": bool(ok and true and d2((p[0], p[2]), true) <= 1.5),
                     "found_on_stale_pos": bool(ok and d2((p[0], p[2]), STALE) <= 1.5),
                     "decisions": len(dec), "stale_targeted_decisions": stale_t, "true_pos": true})
json.dump(rows, open("results/post_paper/movedobj_summary.json", "w"), indent=1)
print(f"{'cond':11s} {'q':2s} {'n':>2s} {'success':>8s} {'true-pos':>8s} {'stale-pos':>9s} {'med steps':>9s} {'stale-targeted decisions (mean)':>32s}")
for cond in sorted({r["cond"] for r in rows}):  # cond = <target>_<condition>
    for k in (0, 1):
        rs = [r for r in rows if r["cond"] == cond and r["episode"] == k]
        if not rs: continue
        print(f"{cond:11s} {k:2d} {len(rs):2d} {sum(r['outcome']=='success' for r in rs):>6d}/{len(rs)} {sum(r['found_on_true_pos'] for r in rs):>6d}/{len(rs)} "
              f"{sum(r['found_on_stale_pos'] for r in rs):>7d}/{len(rs)} {st.median(r['steps'] for r in rs):>9.0f} {st.mean(r['stale_targeted_decisions'] for r in rs):>32.1f}")
