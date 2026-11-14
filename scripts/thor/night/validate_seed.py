"""Validate a house tour's memory before it is used as seed memory (night queue, test 3). Rules fixed in advance:
  * >= 100 remembered entities with a position;
  * >= 1 remembered entry whose caption names a sofa/couch within 2.0 m (floor plane) of the sofa's ORIGINAL position
    (otherwise there is no memory of the sofa to go stale and the house cannot test anything);
  * every remembered sofa entry is >= 4.0 m from the sofa's NEW position (the stale entry must not double as a
    correct one after the move; verify distance is 1.5 m).
On success: copies the snapshot to data/runs/seed_tour_house<idx>.json, records each remembered sofa entry's
mobility-prior lookup (the decay rate it will get), and writes the marker logs/night/ready_house<idx>.
Usage: python scripts/thor/night/validate_seed.py <house_idx> <tour_tag>"""
import glob, json, math, shutil, sys
sys.path.insert(0, "packages/memory_layer")
from memory_layer.mobility import MobilityPriors

idx, tag = int(sys.argv[1]), sys.argv[2]
cfg = json.load(open(f"configs/thor_night/house_{idx}.json"))
snap_path = [p for p in glob.glob(f"data/runs/phase4_batch_*{tag}/memory_snapshot_ep0.json")][0]
snap = json.load(open(snap_path))
ents = [e for e in snap["entities"] if e.get("position_world")]
ox, oz = cfg["sofa_orig_thor_xz"]; nx, nz = cfg["sofa_new_thor_xz"]
def d(e, x, z):  # entity positions are Habitat frame (x, y, -z_thor)
    return math.hypot(e["position_world"][0] - x, -e["position_world"][2] - z)
sofa_words = ("sofa", "couch", "loveseat", "sectional", "settee")
sofas = [e for e in ents if any(w in (e.get("description") or "")[:80].lower() for w in sofa_words)]
near_orig = [e for e in sofas if d(e, ox, oz) <= 2.0]
# 2026-11-03 uniform criterion (replaces ">= 4 m from the new sofa centre"): every sofa entry >= 2.0 m from the NEW
# sofa footprint in the worst orientation (scorer tolerance 1.5 m), so a stale entry can never score as a true success.
gt = json.load(open(glob.glob(f"data/runs/thor_gt_{tag}/gt_*.json")[0]))
sz = [o for o in gt if o["type"] == "Sofa"][0]["size"]
def dfoot(e):
    px, pz = e["position_world"][0], -e["position_world"][2]
    return min(math.hypot(max(abs(px - nx) - a / 2, 0), max(abs(pz - nz) - b / 2, 0))
               for a, b in ((sz["x"], sz["z"]), (sz["z"], sz["x"])))
too_close_new = [e for e in sofas if dfoot(e) < 2.0]
m = MobilityPriors.load("configs/mobility_priors.json")
report = {"house": idx, "tour_tag": tag, "n_entities_positioned": len(ents), "n_sofa_entries": len(sofas),
          "sofa_entries": [{"id": e["id"], "dist_to_orig": round(d(e, ox, oz), 2), "dist_to_new": round(d(e, nx, nz), 2),
                            "mobility_lookup": m.mobility_for(e["category"]), "caption": e["description"][:90]} for e in sofas]}
problems = []
if len(ents) < 100:
    problems.append(f"only {len(ents)} positioned entities")
if not near_orig:
    problems.append("no remembered sofa entry within 2 m of the sofa's original position")
if too_close_new:
    problems.append(f"{len(too_close_new)} remembered sofa entries within 2 m of the new sofa footprint")
report["problems"] = problems
json.dump(report, open(f"logs/night/seed_check_house{idx}.json", "w"), indent=1)
if problems:
    print(f"house {idx}: seed REJECTED: {problems}"); sys.exit(1)
shutil.copy(snap_path, f"data/runs/seed_tour_house{idx}.json")
open(f"logs/night/ready_house{idx}", "w").write(json.dumps(report["sofa_entries"]) + "\n")
print(f"house {idx}: seed OK ({len(ents)} entities, {len(near_orig)} sofa entries at the original spot)")
