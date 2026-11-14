"""Capture frames from several distinct rooms in scene 00770, for the infographic's
'3-4 different scenes' strip (2026-09-06). Longer, more varied walk than the earlier
single-room captures, sampling frames periodically so distinct rooms can be picked
by hand from a contact sheet.
"""
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.habitat_env import HabitatEnv  # noqa: E402

OUT_DIR = Path("paper/figures/room_survey")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in OUT_DIR.glob("*.png"):
    f.unlink()

with open("configs/habitat.yaml") as f:
    config = yaml.safe_load(f)

env = HabitatEnv(config)
frame = env.reset()
Image.fromarray(frame.rgb[:, :, :3]).save(OUT_DIR / "step_000.png")

# A long, varied walk: alternating longer forward runs with turns, to pass through
# several different rooms rather than circling one.
import random
random.seed(42)
actions = []
for _ in range(14):
    actions += ["move_forward"] * random.randint(4, 9)
    actions += [random.choice(["turn_left", "turn_right"])] * random.randint(2, 8)

for i, action in enumerate(actions, start=1):
    frame = env.step(action)
    if i % 3 == 0:  # sample every 3rd step
        Image.fromarray(frame.rgb[:, :, :3]).save(OUT_DIR / f"step_{i:03d}.png")

env.close()
print(f"done, saved frames to {OUT_DIR}")
