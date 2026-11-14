"""High-resolution re-render of the robot camera view used in the teaser/scene
figure (2026-09-29): same deterministic action prefix and camera (height 1.5 m,
hfov 90) as capture_live_grounding_overlay.py, at 3x the 640x480 resolution, so
DAAAM's track boxes (tracks_matched.json, 640x480 pixels) scale by exactly 3.
Out: acra_paper/figures/live_grounding/raw_frame_matched_3x.png"""
import sys
import yaml
from PIL import Image
sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.habitat_env import HabitatEnv

ACTIONS = ["turn_left"] * 9 + ["move_forward"] * 3 + ["turn_left"] * 4 + ["move_forward"] * 1  # as render_spot_third_person.py
cfg = yaml.safe_load(open("configs/habitat.yaml"))
cfg["sensor"]["rgb_resolution"] = [1440, 1920]; cfg["sensor"]["depth_resolution"] = [1440, 1920]
env = HabitatEnv(cfg)
f = env.reset()
for a in ACTIONS:
    f = env.step(a)
Image.fromarray(f.rgb[:, :, :3]).save("acra_paper/figures/live_grounding/raw_frame_matched_3x.png")
print("pose", f.pose["position"]); env.close()
