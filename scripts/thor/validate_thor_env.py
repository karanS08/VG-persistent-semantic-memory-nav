"""Projection test for sim_bridge.thor_env: back-project visible objects' mask pixels
with (a) horizontal-FOV and (b) vertical-FOV intrinsics through the emitted Habitat-frame
pose and compare with THOR's ground-truth object centres (same frame conversion)."""
import math, sys, json, numpy as np
sys.path.insert(0, "packages/sim_bridge")
import prior
from sim_bridge.thor_env import ThorEnv, t2h_point

cfg = {"sensor": {"rgb_resolution": [480, 640], "hfov": 90, "height": 1.5},
       "action_space": {"turn_left": {"amount": 10.0}}, "render_instance_segmentation": True}
house = prior.load_dataset("procthor-10k")["train"][int(sys.argv[1]) if len(sys.argv) > 1 else 1]
env = ThorEnv(cfg, house=house); f = env.reset()
H, W = 480, 640
errs = {"hfov(runner intrinsics)": []}
for k in range(36):
    f = env.step("turn_left")
    e = env.controller.last_event
    pos, q = f.pose["position"], f.pose["rotation"]
    phi = 2 * math.atan2(q[2], q[0])  # rotation about +y
    objs = {o["objectId"]: o for o in e.metadata["objects"]}
    for oid, m in e.instance_masks.items():
        o = objs.get(oid)
        if o is None or m.sum() < 400: continue
        sz = o["axisAlignedBoundingBox"]["size"]
        if max(sz.values()) > 1.2 or oid.split('|')[0] in ('wall', 'floor', 'ceiling', 'door', 'window'): continue  # compact objects only
        ys, xs = np.nonzero(m); d = np.median(f.depth[ys, xs]); u, v = xs.mean(), ys.mean()
        if d > 15: continue
        gt = t2h_point(o["axisAlignedBoundingBox"]["center"])
        for name, fx in (("hfov(runner intrinsics)", (W / 2) / math.tan(math.radians(90) / 2)),):
            xc, zc = (u - W / 2) / fx * d, -d              # camera frame: +x right, forward -z
            xw = pos[0] + xc * math.cos(phi) + zc * math.sin(phi)   # rotate about +y by phi
            zw = pos[2] - xc * math.sin(phi) + zc * math.cos(phi)
            errs[name].append(math.hypot(xw - gt[0], zw - gt[2]))
for n, v in errs.items():
    v = np.array(v); print(f"{n}: n={len(v)} median err {np.median(v):.3f} m  p90 {np.percentile(v, 90):.3f} m")
print("pathfinder: islands", env.pathfinder.num_islands, "cells", len(env.pathfinder._keys))
env.close()
