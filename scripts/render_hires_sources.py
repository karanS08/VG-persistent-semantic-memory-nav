"""Re-render the teaser's photographic sources at 2x their previous resolution (2026-10-03), same poses so the stored
DAAAM boxes (640x480 px) scale exactly: 6x -> 3840x2880.
  1) acra_paper/figures/live_grounding/raw_frame_matched_6x.png  (window + radiator view; render_highres_view.py pose)
  2) acra_paper/figures/corner_grounding/raw_6x.png              (blue-room corner view; capture_corner_grounding.py pose)
Spot render: python scripts/render_spot_third_person.py 0 "-1.8,1.7,0.8" 1.6 3840x2160"""
import math, sys
import numpy as np, quaternion, yaml  # noqa: F401
from habitat_sim import AgentState
from PIL import Image
sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.habitat_env import HabitatEnv

cfg = yaml.safe_load(open("configs/habitat.yaml"))
cfg["sensor"]["rgb_resolution"] = [2880, 3840]; cfg["sensor"]["depth_resolution"] = [2880, 3840]

env = HabitatEnv(cfg); f = env.reset()
for a in ["turn_left"] * 9 + ["move_forward"] * 3 + ["turn_left"] * 4 + ["move_forward"] * 1:
    f = env.step(a)
Image.fromarray(f.rgb[:, :, :3]).save("acra_paper/figures/live_grounding/raw_frame_matched_6x.png"); print("frame pose", np.round(f.pose["position"], 2))

env.reset()
pos = np.array(env.pathfinder.snap_point(np.array([4.2, 0.107, 9.5], dtype=np.float32)))
d = np.array([9.0, 0.0, 14.0]) - pos; yaw = math.atan2(-d[0], -d[2])
st = AgentState(); st.position = pos; st.rotation = quaternion.from_rotation_vector([0, yaw, 0]); env._agent.set_state(st)
f = env.current_frame()
Image.fromarray(f.rgb[:, :, :3]).save("acra_paper/figures/corner_grounding/raw_6x.png"); print("corner pose", np.round(f.pose["position"], 2))
env.close()
