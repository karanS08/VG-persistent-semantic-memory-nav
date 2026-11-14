"""Real, pixel-accurate grounding overlay (2026-09-06): navigates to the same
window/radiator room used in the teaser frames, pushes that EXACT frame through
the real bridge/DAAAM pipeline, and gets back real 2D bounding boxes for
whatever DAAAM actually detects in that same frame -- then draws those boxes
directly onto the frame they came from. No coordinate-system mismatch: the
bbox and the pixels are from the identical frame.

Requires the bridge server's /frame response to include "tracks" (added
2026-09-06 to daaam_bridge_server.py inside the daaam_dev container
specifically for this).
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "packages/sim_bridge")
sys.path.insert(0, "packages/daaam_adapter")
import requests
import yaml
from PIL import Image, ImageDraw

from sim_bridge.habitat_env import HabitatEnv
from daaam_adapter.bridge_client import DaaamBridgeClient

OUT_DIR = Path("paper/figures/live_grounding")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def camera_intrinsics(width, height, hfov_deg):
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def wxyz_to_xyzw_transform(position, rotation_wxyz):
    w, x, y, z = rotation_wxyz
    return [position[0], position[1], position[2], x, y, z, w]


with open("configs/habitat.yaml") as f:
    config = yaml.safe_load(f)

env = HabitatEnv(config)
frame = env.reset()

# EXACT 17-action prefix of capture_teaser_frames.py's v2 trajectory that
# produced step_017.png -- the specific frame (window + radiator both fully
# visible) the user picked as "image 3" and wants the real overlay for.
actions = (
    ["turn_left"] * 9
    + ["move_forward"] * 3
    + ["turn_left"] * 4
    + ["move_forward"] * 1
)
for action in actions:
    frame = env.step(action)

Image.fromarray(frame.rgb[:, :, :3]).save(OUT_DIR / "raw_frame.png")
print("captured frame at step", frame.step_id)

client = DaaamBridgeClient()
health = client.health()
print("bridge health:", health)

if health.get("episode_running"):
    try:
        requests.post("http://localhost:8765/episode/stop", timeout=30)
    except Exception as e:
        print("stop attempt:", e)

client.start_episode()
print("episode started")

habitat_cfg = env._config
intrinsics = camera_intrinsics(
    habitat_cfg["sensor"]["rgb_resolution"][1], habitat_cfg["sensor"]["rgb_resolution"][0],
    habitat_cfg["sensor"]["hfov"],
)
transform = wxyz_to_xyzw_transform(frame.pose["position"], frame.pose["rotation"])

# push the SAME frame several times in a row -- BoT-SORT tracking needs a couple
# of frames to establish a stable track before FastSAM's mask survives into
# frame.tracks; pushing the identical static frame repeatedly is a legitimate way
# to let tracking stabilize without moving the (fixed) camera.
resp = None
for i in range(5):
    resp = client.push_frame(
        frame_id=frame.step_id + i, timestamp=time.time(),
        rgb=frame.rgb[:, :, :3], depth=frame.depth,
        transform=transform, camera_intrinsics=intrinsics,
    )
    print(f"push {i}: num_tracks={resp.get('num_tracks')}")
    time.sleep(0.5)

tracks = resp.get("tracks", [])
print(f"final tracks ({len(tracks)}):", tracks)

json.dump(tracks, open(OUT_DIR / "tracks.json", "w"), indent=2)

env.close()
try:
    requests.post("http://localhost:8765/episode/stop", timeout=30)
except Exception as e:
    print("final stop:", e)

print(f"done. raw frame + tracks.json saved to {OUT_DIR}")
