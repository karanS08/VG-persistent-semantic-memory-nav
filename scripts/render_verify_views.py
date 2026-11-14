"""Two views of the SAME radiator in the blue south-east room of 00770 for the teaser's Verify card (2026-10-03):
  far  = the earlier sighting (camera in the room corner, pose of capture_corner_grounding.py), looking at the radiator;
  near = the fresh look (1.8 m from it, facing it).
Radiator = DAAAM box 13 of the corner view (tracks.json), unprojected with the depth of that frame. These are renders of the
mechanism in the same scene, NOT frames from an evaluated run (the caption must say so).
Out: acra_paper/figures/verify_views/{far.png,near.png,info.json}"""
import json, math, sys
import numpy as np, quaternion, yaml  # noqa: F401
from habitat_sim import AgentState
from PIL import Image
sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.habitat_env import HabitatEnv

OUT = "acra_paper/figures/verify_views"
import os; os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open("configs/habitat.yaml"))
cfg["sensor"]["rgb_resolution"] = [1440, 1920]; cfg["sensor"]["depth_resolution"] = [1440, 1920]
env = HabitatEnv(cfg); env.reset(); pf = env.pathfinder


def place(pos, look):
    pos = np.array(pf.snap_point(np.array(pos, dtype=np.float32)))
    d = np.array(look) - pos; yaw = math.atan2(-d[0], -d[2])
    st = AgentState(); st.position = pos; st.rotation = quaternion.from_rotation_vector([0, yaw, 0]); env._agent.set_state(st)
    return env.current_frame(), pos


# 1) locate radiator (box 13) from the corner frame + depth
fr, pos0 = place([4.2, 0.107, 9.5], [9.0, 0.0, 14.0])
T = json.load(open("acra_paper/figures/corner_grounding/tracks.json"))
x0, y0, x1, y1 = [v * 3 for v in T["13"]]; u, v = int((x0 + x1) / 2), int((y0 + y1) / 2)
depth = float(np.median(fr.depth[v - 8:v + 8, u - 8:u + 8]))
f = (1920 / 2) / math.tan(math.radians(cfg["sensor"]["hfov"]) / 2)
pc = np.array([(u - 960) / f * depth, -(v - 720) / f * depth, -depth])
w, x, y, z = fr.pose["rotation"]; R = quaternion.as_rotation_matrix(np.quaternion(w, x, y, z))
cam = np.array(fr.pose["position"]) + np.array([0, cfg["sensor"]["height"], 0])
rad = R @ pc + cam; rad_floor = np.array([rad[0], 0.107, rad[2]]); print("radiator ~", np.round(rad, 2), "depth", round(depth, 2))
# 2) far view = from the corner, looking at it ; near view = 1.8 m in front of it, toward the room
far, _ = place([4.2, 0.107, 9.5], rad_floor)
Image.fromarray(far.rgb[:, :, :3]).save(f"{OUT}/far.png")
dirn = (np.array([pos0[0], 0, pos0[2]]) - rad_floor); dirn /= np.linalg.norm(dirn)
cand = rad_floor + dirn * 1.8
near, p1 = place(cand.tolist(), [rad[0], 0.5, rad[2]])
Image.fromarray(near.rgb[:, :, :3]).save(f"{OUT}/near.png")
json.dump({"radiator_world": rad.tolist(), "far_pos": far.pose["position"], "near_pos": near.pose["position"]}, open(f"{OUT}/info.json", "w"), indent=1)
env.close()
