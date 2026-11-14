"""Capture a short sequence of real RGB frames from scene 00770 for the paper's
teaser figure (2026-09-06). Reuses sim_bridge.habitat_env.HabitatEnv exactly as
run_phase1.py does -- same scene, same environment class, no shortcuts.

v2: turn away from the bright entrance-hall windows first, then walk into the
interior, saving every frame along a short deliberate path so they can be
inspected and picked by hand rather than auto-selected by a brightness metric
(which picked uninformative blank-wall frames on the first attempt).
"""
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.habitat_env import HabitatEnv  # noqa: E402

OUT_DIR = Path("paper/figures/teaser_frames_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in OUT_DIR.glob("*.png"):
    f.unlink()

with open("configs/habitat.yaml") as f:
    config = yaml.safe_load(f)

env = HabitatEnv(config)
frame = env.reset()

actions = (
    ["turn_left"] * 9        # turn ~90 deg away from the bright windows
    + ["move_forward"] * 3
    + ["turn_left"] * 4      # angle further into the interior
    + ["move_forward"] * 6
    + ["turn_right"] * 3
    + ["move_forward"] * 5
)

Image.fromarray(frame.rgb[:, :, :3]).save(OUT_DIR / "step_000.png")
for i, action in enumerate(actions, start=1):
    frame = env.step(action)
    Image.fromarray(frame.rgb[:, :, :3]).save(OUT_DIR / f"step_{i:03d}.png")

env.close()
print(f"done, {len(actions) + 1} frames saved to {OUT_DIR}")
