"""Real DAAAM detections for a corner-of-the-room view of 00770 (2026-10-03).
Stands in a corner of the blue south-east room, looking at the opposite corner; pushes that frame (640x480, hfov 90,
the pipeline's normal input) through the real bridge/DAAAM, keeps pushing it while captions come back asynchronously
(pull_updates), and saves: the tracks (2D boxes, 640x480 px), the captions by semantic id, and a 1920x1440 render of
the SAME pose (boxes scale exactly 3x). Labels in the figure are therefore DAAAM's own, correct by construction.
Out: acra_paper/figures/corner_grounding/{raw_3x.png, tracks.json, captions.json}
Usage: python scripts/capture_corner_grounding.py [x z look_x look_z]"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import quaternion  # noqa: F401
import requests
import yaml
from habitat_sim import AgentState
from PIL import Image

sys.path.insert(0, "packages/sim_bridge")
sys.path.insert(0, "packages/daaam_adapter")
from sim_bridge.habitat_env import HabitatEnv  # noqa: E402
from daaam_adapter.bridge_client import DaaamBridgeClient  # noqa: E402

OUT = Path("acra_paper/figures/corner_grounding"); OUT.mkdir(parents=True, exist_ok=True)
x, z, lx, lz = ([float(v) for v in sys.argv[1:5]] if len(sys.argv) >= 5 else [4.2, 9.5, 9.0, 14.0])
cfg = yaml.safe_load(open("configs/habitat.yaml"))


def place(env):
    env.reset()
    pf = env.pathfinder
    pos = np.array(pf.snap_point(np.array([x, 0.107, z], dtype=np.float32)))
    d = np.array([lx, 0.0, lz]) - pos
    yaw = math.atan2(-d[0], -d[2])
    st = AgentState(); st.position = pos; st.rotation = quaternion.from_rotation_vector([0, yaw, 0])
    env._agent.set_state(st)
    return env.current_frame()


def intr(w, h, hfov):
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    return {"fx": f, "fy": f, "cx": w / 2, "cy": h / 2}


env = HabitatEnv(cfg)
frame = place(env)
print("pose", [round(v, 2) for v in frame.pose["position"]])
Image.fromarray(frame.rgb[:, :, :3]).save(OUT / "raw_1x.png")
client = DaaamBridgeClient()
if client.health().get("episode_running"):
    requests.post("http://localhost:8765/episode/stop", timeout=30); time.sleep(2)
client.start_episode()
w_, h_ = cfg["sensor"]["rgb_resolution"][1], cfg["sensor"]["rgb_resolution"][0]
w, xq, y, zq = frame.pose["rotation"]
transform = [*frame.pose["position"], xq, y, zq, w]
captions, tracks_by_id, t0 = {}, {}, time.time()
intrs = intr(w_, h_, cfg["sensor"]["hfov"])


def push(fr, i):
    tf = [*fr.pose["position"], fr.pose["rotation"][1], fr.pose["rotation"][2], fr.pose["rotation"][3], fr.pose["rotation"][0]]
    return client.push_frame(frame_id=i, timestamp=time.time(), rgb=fr.rgb[:, :, :3], depth=fr.depth, transform=tf, camera_intrinsics=intrs)


# 1) the figure's view: push it a few times and RECORD its tracks (ids + boxes)
for i in range(6):
    resp = push(frame, i)
    time.sleep(0.5)
tracks_by_id = {t["id"]: t["bbox"] for t in resp.get("tracks", [])}
print("tracks of the figure view:", len(tracks_by_id), flush=True)
# 2) DAAAM captions objects once the camera has moved: spin a full circle in place (36 x 10 deg) pushing every frame
fr = frame
for k in range(36):
    fr = env.step("turn_left")
    push(fr, 100 + k)
    for u in client.pull_updates():
        captions[u["semantic_id"]] = u["semantic_label"]
    time.sleep(0.5)
# 3) back at the starting orientation: keep pushing while the captions come back
i = 200
while time.time() - t0 < 240:
    push(fr, i); i += 1
    for u in client.pull_updates():
        captions[u["semantic_id"]] = u["semantic_label"]
    if i % 10 == 0:
        print(f"t={time.time() - t0:5.1f}s captions {len(captions)} (ids matching figure tracks: {len(set(captions) & set(tracks_by_id))})", flush=True)
    if len(set(captions) & set(tracks_by_id)) >= 6 and time.time() - t0 > 80:
        break
    time.sleep(0.5)
json.dump({str(k): v for k, v in tracks_by_id.items()}, open(OUT / "tracks.json", "w"), indent=1)
json.dump({str(k): v for k, v in captions.items()}, open(OUT / "captions.json", "w"), indent=1)
requests.post("http://localhost:8765/episode/stop", timeout=30)
env.close()

cfg3 = dict(cfg); cfg3["sensor"] = dict(cfg["sensor"]); cfg3["sensor"]["rgb_resolution"] = [1440, 1920]; cfg3["sensor"]["depth_resolution"] = [1440, 1920]
env3 = HabitatEnv(cfg3)
f3 = place(env3)
Image.fromarray(f3.rgb[:, :, :3]).save(OUT / "raw_3x.png")
env3.close()
print("saved", OUT, "| tracks", len(tracks_by_id), "captions", len(captions), "| ids with both:", sorted(set(tracks_by_id) & set(captions)))
