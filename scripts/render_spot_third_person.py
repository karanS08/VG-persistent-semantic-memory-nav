"""Third-person render of the robot (Spot model) in scene 00770 at the exact
camera pose of acra_paper/figures/live_grounding/raw_frame_matched.png (window
and radiator in view), reproduced by replaying the same deterministic action
prefix as scripts/capture_live_grounding_overlay.py. Reuses the lit Spot +
chase-camera setup from capture_spot_teaser_v2.py. The evaluation agent is a
bodiless Habitat agent; Spot is shown for illustration. Spot's body faces its
+X axis while the Habitat agent faces -Z, hence the +90 deg yaw offset
(confirmed visually against the agent's own camera frame).
Usage: python scripts/render_spot_third_person.py [extra_yaw_deg] [cam_offset dx,dy,dz] [advance_m] [WxH]
Paper figure (2026-09-29): args 0 "-1.8,1.7,0.8" 1.6 (robot 1.6 m into the room, camera
behind-above; clear of the doorframe)."""
import math
import sys
from pathlib import Path

import magnum as mn
import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")
from capture_spot_teaser_v2 import build_sim, spawn_spot  # noqa: E402

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
OUT = Path("acra_paper/figures/spot_third_person.png")
ACTIONS = ["turn_left"] * 9 + ["move_forward"] * 3 + ["turn_left"] * 4 + ["move_forward"] * 1

import yaml  # noqa: E402
sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.habitat_env import HabitatEnv  # noqa: E402
env = HabitatEnv(yaml.safe_load(open("configs/habitat.yaml")))
frame = env.reset()
for act in ACTIONS:
    frame = env.step(act)
pos = list(frame.pose["position"])
w, x, y, z = frame.pose["rotation"]                             # wxyz
env.close()
yaw = 2 * math.atan2(y, w) + math.radians(90)                   # agent -Z -> Spot +X
yaw += math.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)

RES = [int(v) for v in sys.argv[4].split("x")][::-1] if len(sys.argv) > 4 else (720, 1280)  # "WxH"
sim = build_sim(SCENE, resolution=RES)
spot = spawn_spot(sim)
ADVANCE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0  # metres forward along heading
fwd = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0)).transform_vector(mn.Vector3(1, 0, 0))
spot.base_pos = sim.pathfinder.snap_point(mn.Vector3(*pos) + fwd * ADVANCE)
spot.base_rot = yaw
if len(sys.argv) > 2:  # optional chase-camera offset "dx,dy,dz" in Spot's base frame
    spot.params.cameras["teaser"].cam_offset_pos = mn.Vector3(*[float(v) for v in sys.argv[2].split(",")])
spot.update()
obs = sim.get_sensor_observations()
Image.fromarray(obs["teaser_rgb"][:, :, :3]).save(OUT)
sim.close()
print("pose", np.round(pos, 2), "yaw_deg", round(math.degrees(yaw), 1), "->", OUT)
