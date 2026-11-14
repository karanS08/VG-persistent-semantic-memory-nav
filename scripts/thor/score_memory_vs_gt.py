"""Score memory_snapshot entity positions (Habitat-frame world xz) against THOR ground-truth objects
(gt_*.json from run_thor_batch.py). An entity is 'on an object' if its xz is within `tol` m of the
object's AABB footprint. Also reports the nearest compact object's type for caption sanity checks."""
import json, math, sys, collections
snap = json.load(open(sys.argv[1])); gt = json.load(open(sys.argv[2])); tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
ents = snap["entities"] if isinstance(snap, dict) and "entities" in snap else snap
if isinstance(ents, dict): ents = list(ents.values())
def dist_to_box(p, o):
    cx, cz = o["pos_h"][0], o["pos_h"][2]; hx, hz = o["size"]["x"] / 2, o["size"]["z"] / 2
    return math.hypot(max(abs(p[0] - cx) - hx, 0), max(abs(p[2] - cz) - hz, 0))
compact = [o for o in gt if max(o["size"].values()) < 1.5 and o["type"] not in ("Wall", "Floor", "Door", "Window")]
res, rows = [], []
for e in ents:
    p = e.get("position_world")
    if not p: continue
    dall = min(dist_to_box(p, o) for o in gt)
    nc = min(compact, key=lambda o: dist_to_box(p, o)); dc = dist_to_box(p, nc)
    res.append((dall, dc)); rows.append((e.get("description", "")[:60], nc["type"], round(dc, 2)))
n = len(res)
print(f"entities scored: {n}")
print(f"within {tol} m of ANY scene object footprint: {sum(d <= tol for d, _ in res)/n:.2f}")
print(f"within {tol} m of a COMPACT object footprint: {sum(c <= tol for _, c in res)/n:.2f}")
ds = sorted(c for _, c in res); print(f"nearest-compact-object distance: median {ds[n//2]:.2f} m  p90 {ds[int(n*.9)]:.2f} m")
for r in rows[:12]: print(r)

# ---- type-matched scoring: caption keyword -> THOR object types (any size) ----
KW = {"refrigerator": ["Fridge"], "television": ["Television"], "sofa": ["Sofa"], "couch": ["Sofa"], "bed": ["Bed"],
      "toilet": ["Toilet"], "sink": ["Sink", "SinkBasin"], "chair": ["Chair", "ArmChair", "Stool"], "table": ["DiningTable", "SideTable", "CoffeeTable", "Desk"],
      "painting": ["Painting"], "lamp": ["FloorLamp", "DeskLamp", "Lamp"], "plant": ["HousePlant"], "window": ["Window"], "door": ["Door"],
      "microwave": ["Microwave"], "oven": ["StoveBurner", "StoveKnob", "Oven"], "stove": ["StoveBurner"], "bathtub": ["Bathtub"], "shelf": ["Shelf", "ShelvingUnit"],
      "dresser": ["Dresser"], "cabinet": ["Cabinet"], "mirror": ["Mirror"], "vase": ["Vase"], "bowl": ["Bowl"], "book": ["Book"]}
def first_kw(desc):
    d = desc.lower(); best = None
    for k in KW:
        i = d.find(k)
        if i >= 0 and (best is None or i < best[0]): best = (i, k)
    return best[1] if best else None
tm = []; per = collections.defaultdict(list)
for e in ents:
    p = e.get("position_world"); k = first_kw(e.get("description", ""))
    if not p or not k: continue
    cands = [o for o in gt if o["type"] in KW[k]]
    if not cands: continue
    d = min(dist_to_box(p, o) for o in cands); tm.append(d); per[k].append(d)
tm.sort(); m = len(tm)
print(f"\ntype-matched entities (caption names a THOR-mappable object that exists in the house): {m}/{n}")
print(f"within {tol} m of the nearest object OF THAT TYPE: {sum(d <= tol for d in tm)/m:.2f}   within 1.0 m: {sum(d <= 1.0 for d in tm)/m:.2f}")
print(f"distance to nearest same-type object: median {tm[m//2]:.2f} m  p90 {tm[int(m*.9)]:.2f} m")
for k, v in sorted(per.items(), key=lambda kv: -len(kv[1]))[:8]: print(f"  {k:12s} n={len(v):3d} median {sorted(v)[len(v)//2]:.2f} m")
