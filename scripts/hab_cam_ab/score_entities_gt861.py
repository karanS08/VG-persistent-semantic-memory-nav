"""Entity-level ground-truth check on HM3D 00861 (semantic annotations): for every stored memory entity whose caption
names an object category, distance (xz) from the entity's position to the nearest GT instance of that category.
Instance centres come from score_gt861.instance_centres (semantic+depth back-projection). Usage:
  python scripts/hab_cam_ab/score_entities_gt861.py <out.json> <run_dir> [<run_dir> ...]"""
import glob, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "scripts")
import score_gt861 as S
K = {"chair": ["chair"], "armchair": ["chair"], "sofa": ["sofa", "couch"], "couch": ["sofa", "couch"], "bed": ["bed"], "pillow": ["pillow", "cushion"],
     "cushion": ["pillow", "cushion"], "lamp": ["lamp"], "toilet": ["toilet"], "sink": ["sink"], "mirror": ["mirror"], "refrigerator": ["refrigerator", "fridge"],
     "table": ["table"], "desk": ["desk", "table"], "painting": ["picture", "painting"], "television": ["tv", "television"], "plant": ["plant"],
     "cabinet": ["cabinet"], "shelf": ["shelf", "shelves"], "bathtub": ["bathtub", "bath"], "window": ["window"], "door": ["door"], "vase": ["vase"], "rug": ["rug", "mat"]}
def kw(desc):
    d = desc.lower(); b = None
    for k in K:
        i = d.find(k)
        if i >= 0 and (b is None or i < b[0]): b = (i, k)
    return b[1] if b else None
sim = S.make_sim(); cats = {}
for o in sim.semantic_scene.objects:
    if o is not None and o.category is not None:
        try: cats[int(o.id.split("_")[-1])] = o.category.name().lower()
        except ValueError: pass
centres = S.instance_centres(sim); sim.close()
out = {"_doc": __doc__, "runs": {}}
for run in sys.argv[2:]:
    ds, n_all = [], 0
    for f in sorted(glob.glob(f"{run}/memory_snapshot_ep*.json")):
        sn = json.load(open(f)); ents = sn["entities"] if isinstance(sn, dict) and "entities" in sn else sn
        if isinstance(ents, dict): ents = list(ents.values())
        for e in ents:
            p = e.get("position_world"); k = kw(e.get("description", ""))
            n_all += 1
            if not p or not k: continue
            cand = [np.linalg.norm((c - np.array(p))[[0, 2]]) for i, c in centres.items() if any(t in cats.get(i, "") for t in K[k])]
            if cand: ds.append(min(cand))
    ds = np.array(sorted(ds))
    out["runs"][run] = {"entities": n_all, "typed_with_gt_instance": len(ds),
        "within_1m": float((ds <= 1).mean()) if len(ds) else None, "within_2m": float((ds <= 2).mean()) if len(ds) else None,
        "median_m": float(np.median(ds)) if len(ds) else None, "p90_m": float(np.percentile(ds, 90)) if len(ds) else None}
    print(run, out["runs"][run])
json.dump(out, open(sys.argv[1], "w"), indent=1)
