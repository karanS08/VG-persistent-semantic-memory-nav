"""pipeline-v8 sanity check on real THOR renders (mechanism check, nothing tuned):
(1) camera-height / projection: projected GT object centre vs its instance-mask centroid;
(2) image check: real CLIP ViT-L/14 on crops at GT object centres, for sofas (should pass) and chairs/armchairs
    queried as "a sofa" (should fail) and as their own type (should pass)."""
import math, sys, json
import numpy as np
for p in ("sim_bridge", "memory_layer", "frontier_agent"):
    sys.path.insert(0, f"packages/{p}")
import prior, yaml
from sim_bridge.thor_env import ThorEnv, t2h_point
from memory_layer.relevance import ClipTextEncoder
from memory_layer.geometry import project_point
from frontier_agent.image_check import ImageChecker

cfg = {"sensor": {"rgb_resolution": [480, 640], "hfov": 90, "height": 1.5},
       "action_space": {"turn_left": {"amount": 10.0}}, "render_instance_segmentation": True}
house = prior.load_dataset("procthor-10k")["train"][1]
env = ThorEnv(cfg, house=house); env.reset()
intr = {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0}
mem = yaml.safe_load(open("configs/memory.yaml"))
enc = ClipTextEncoder(mem["embedding"]["clip_model_name"], mem["embedding"]["clip_model_dataset"])
ic = ImageChecker(enc)
QUERY = {"Sofa": "a sofa", "Chair": "a chair", "ArmChair": "an armchair"}
reach = env.controller.step("GetReachablePositions").metadata["actionReturn"]
rows, vres = [], []
objs = [o for o in env.controller.last_event.metadata["objects"] if o["objectType"] in QUERY]
for o in objs:
    c = o["axisAlignedBoundingBox"]["center"]
    # stand 1.2-2.0 m away, facing the object (THOR coords)
    cands = sorted(reach, key=lambda r: abs(math.hypot(r["x"] - c["x"], r["z"] - c["z"]) - 1.3))[:6]
    for r in cands[:3]:
        yaw = math.degrees(math.atan2(c["x"] - r["x"], c["z"] - r["z"]))
        env.controller.step("Teleport", position=r, rotation={"x": 0, "y": yaw, "z": 0}, horizon=0, standing=True)
        f = env._frame(); e = env.controller.last_event
        m = e.instance_masks.get(o["objectId"])
        if m is None or m.sum() < 500:
            continue
        gt = [float(c["x"]), float(c["y"]), -float(c["z"])]  # true centre height
        pr = project_point(gt, f.pose, intr, 1.576)
        ys, xs = np.nonzero(m)
        vres.append((pr[0] - xs.mean(), pr[1] - ys.mean()))
        for q in {"a sofa", QUERY[o["objectType"]]}:
            res = ic.check(q, gt, f.rgb, f.depth, f.pose, intr, 1.576)
            rows.append((o["objectType"], q, res.get("status"), res.get("top_label"), res.get("margin")))
        break
v = np.array(vres)
print(f"projection vs mask centroid over {len(v)} objects: median |du|={np.median(abs(v[:,0])):.1f}px "
      f"median dv={np.median(v[:,1]):.1f}px (positive = projected below centroid)")
for r in rows:
    print(r)
from collections import Counter
print(Counter((t, q == 'a sofa', s) for t, q, s, _, _ in rows))
env.close()
