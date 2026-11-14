"""Real camera frames of the five spread-demo targets from the house-tour recording
(2026-09-29). For each target (tour-memory entity whose caption names it, see
render_spread_objects.py -> spread/spread_objects.json), keep tour steps where the
remembered position is 1-3.5 m in front of the camera within +-12 deg of the view
axis; save the 4 closest-to-centre frames. These are the images DAAAM saw.
Out: acra_paper/teaser_assets/spread/tour_obj_<k>_<v>.png"""
import json
import numpy as np
import quaternion  # noqa: F401
import rerun.dataframe as rd
from PIL import Image

RRD = "data/runs/phase4_batch_persist_memory_only_house_tour/episode_0.rrd"
OUT = "acra_paper/teaser_assets/spread"
objs = json.load(open(f"{OUT}/spread_objects.json"))
rec = rd.load_recording(RRD)
t = rec.view(index="step", contents="/world/agent").select().read_all()
steps = t.column("step").to_pylist()
tr = t.column("/world/agent:Translation3D").to_pylist()
qs = t.column("/world/agent:RotationQuat").to_pylist()
pose = {s: (np.array(p[0]), np.array(q[0])) for s, p, q in zip(steps, tr, qs) if p and q}
for o in objs:
    T = np.array(o["position_world"]); cand = []
    for s, (p, q) in pose.items():
        R = quaternion.as_rotation_matrix(np.quaternion(q[3], q[0], q[1], q[2]))   # rerun xyzw
        fwd = R @ np.array([0.0, 0.0, -1.0])
        d = (T - p)[[0, 2]]; dist = np.linalg.norm(d)
        f2 = fwd[[0, 2]] / (np.linalg.norm(fwd[[0, 2]]) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(d @ f2 / (dist + 1e-9), -1, 1)))
        if 1.0 < dist < 3.5 and ang < 12:
            cand.append((ang, s))
    cand.sort()
    picked, used = [], []
    for ang, s in cand:
        if all(abs(s - u) > 8 for u in used):
            used.append(s); picked.append(s)
        if len(picked) == 4:
            break
    for v, s in enumerate(picked):
        im = rec.view(index="step", contents="/world/agent/rgb").filter_index_values([s]).select().read_all()
        buf = np.array(im.column("/world/agent/rgb:ImageBuffer").to_pylist()[0][0], dtype=np.uint8)
        fmt = im.column("/world/agent/rgb:ImageFormat").to_pylist()[0][0]
        img = buf.reshape(fmt["height"], fmt["width"], -1)[:, :, :3]
        Image.fromarray(img).save(f"{OUT}/tour_obj_{o['k']}_{v}.png")
    print(o["k"], o["query"], "candidates", len(cand), "picked steps", picked)
