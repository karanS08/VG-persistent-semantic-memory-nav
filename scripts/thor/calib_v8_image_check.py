"""Calibrate the pipeline-v8 image check's decision rule on ProcTHOR houses that are NEVER used for evaluation
(train houses 2..40; house 1 is the THOR test house). For each visible object of a common furniture type, one view
from a reachable cell 0.8-1.5 m away (the verification zone), facing the object; the crop is scored for the
object's own type (should pass) and for two other types (should fail). Writes every similarity vector so rules
(top-k, margin) can be compared offline:  results/post_paper/v8_image_check_calib.json

Usage (rpv_thor env, repo root): python scripts/thor/calib_v8_image_check.py [first_house] [last_house]
"""
import json, math, random, sys
import numpy as np
for p in ("sim_bridge", "memory_layer", "frontier_agent"):
    sys.path.insert(0, f"packages/{p}")
import prior, yaml
from sim_bridge.thor_env import ThorEnv, t2h_point
from memory_layer.relevance import ClipTextEncoder
from frontier_agent.image_check import ImageChecker

PHRASE = {"Sofa": "a sofa", "Chair": "a chair", "ArmChair": "an armchair", "Bed": "a bed",
          "DiningTable": "a dining table", "Desk": "a desk", "Toilet": "a toilet", "Fridge": "a refrigerator",
          "Dresser": "a dresser", "FloorLamp": "a floor lamp", "Television": "a television", "Stool": "a stool",
          "CoffeeTable": "a coffee table", "SideTable": "a side table", "ShelvingUnit": "a shelf",
          "Bathtub": "a bathtub", "HousePlant": "a plant", "GarbageCan": "a trash can", "TVStand": "a tv stand"}
CAM_H = 1.576  # THOR camera height above floor (measured); GT centres are true positions, so use the true height
first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2, 40)
assert first > 1, "house 1 is the evaluation house"
random.seed(0)
mem = yaml.safe_load(open("configs/memory.yaml"))
ic = ImageChecker(ClipTextEncoder(mem["embedding"]["clip_model_name"], mem["embedding"]["clip_model_dataset"]))
intr = {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0}
cfg = {"sensor": {"rgb_resolution": [480, 640], "hfov": 90, "height": 1.5},
       "action_space": {"turn_left": {"amount": 10.0}}, "render_instance_segmentation": True}
ds = prior.load_dataset("procthor-10k")["train"]
out = []
for hi in range(first, last + 1):
    try:
        env = ThorEnv(cfg, house=ds[hi]); env.reset()
    except Exception as exc:  # noqa: BLE001
        print(f"house {hi}: skipped ({exc})"); continue
    c_ = env.controller
    reach = c_.step("GetReachablePositions").metadata["actionReturn"] or []
    objs = [o for o in c_.last_event.metadata["objects"] if o["objectType"] in PHRASE]
    for o in objs:
        c = o["axisAlignedBoundingBox"]["center"]
        cand = [r for r in reach if 0.8 <= math.hypot(r["x"] - c["x"], r["z"] - c["z"]) <= 1.5]
        random.shuffle(cand)
        for r in cand[:4]:
            yaw = math.degrees(math.atan2(c["x"] - r["x"], c["z"] - r["z"]))
            c_.step("Teleport", position=r, rotation={"x": 0, "y": yaw, "z": 0}, horizon=0, standing=True)
            m = c_.last_event.instance_masks.get(o["objectId"])
            if m is None or m.sum() < 800:
                continue
            f = env._frame()
            gt = [float(c["x"]), float(c["y"]), -float(c["z"])]  # true centre height (t2h_point puts y on the floor)
            crop = ic.crop(gt, f.rgb, f.depth, f.pose, intr, CAM_H)
            if crop is None:
                continue
            others = random.sample([t for t in PHRASE if t != o["objectType"]], 2)
            for q, own in [(PHRASE[o["objectType"]], True)] + [(PHRASE[t], False) for t in others]:
                labels, sims = ic.scores(q, crop)
                rank = int((sims > sims[0]).sum()) + 1
                out.append({"house": hi, "type": o["objectType"], "query": q, "own": own, "rank": rank,
                            "margin": float(sims[0] - np.max(sims[1:])), "dist": round(math.hypot(r["x"] - c["x"], r["z"] - c["z"]), 2),
                            "mask_px": int(m.sum()), "labels": labels, "sims": [round(float(s), 5) for s in sims]})
            break
    env.close()
    n_own = sum(1 for x in out if x["own"])
    print(f"house {hi}: {len(objs)} objects, total crops {n_own}", flush=True)
json.dump(out, open("results/post_paper/v8_image_check_calib.json", "w"))
print("wrote", len(out), "rows")
