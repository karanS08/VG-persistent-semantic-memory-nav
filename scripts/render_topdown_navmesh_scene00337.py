"""Top-down navmesh visualization for scene 00337 (33 rooms, 4 floors).

Same real methodology the paper's own Environment section already cites for
scene selection (pathfinder.get_topdown_view()) -- extended here with an
orthographic top-down RGB render of the actual scene geometry underneath the
navmesh, composited with a green overlay, matching the reference style the
user asked for (real textured geometry visible, green = navigable).

Real floor heights found by probing get_topdown_view()'s navigable-pixel
fraction across the scene's full vertical extent (a clean, distinct peak per
floor, separated by near-zero valleys at the walls/stairwells between them):
-2.80, 0.34, 3.48, 5.58 (metres, Habitat's Y-up convention).

One PNG per floor, plus a combined 2x2 grid figure for the paper.
"""
from __future__ import annotations

import numpy as np
import habitat_sim
from PIL import Image, ImageDraw, ImageFont

SCENE_PATH = "data/hm3d/example/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb"
OUT_DIR = "paper/figures/topdown_navmesh_00337"
FLOOR_HEIGHTS = [-2.80, 0.34, 3.48, 5.58]  # real, probed -- see module docstring
METERS_PER_PIXEL = 0.02
CAMERA_CLEARANCE = 0.5  # metres above each floor's navigable height, for the ortho camera

import os
os.makedirs(OUT_DIR, exist_ok=True)

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE_PATH
sim_cfg.enable_physics = False

rgb_sensor = habitat_sim.CameraSensorSpec()
rgb_sensor.uuid = "rgb_topdown"
rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
rgb_sensor.resolution = [1024, 1024]
rgb_sensor.orientation = np.array([-np.pi / 2, 0.0, 0.0], dtype=np.float32)  # look straight down (-Y)

# ortho_scale must be known before the sim/sensor is constructed -- probe bounds
# with a throwaway sim first (cheap, no rendering needed for get_bounds()).
_probe_cfg = habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()])
_probe_sim = habitat_sim.Simulator(_probe_cfg)
lo, hi = _probe_sim.pathfinder.get_bounds()
_probe_sim.close()
x_extent = hi[0] - lo[0]
z_extent = hi[2] - lo[2]
x_center = (hi[0] + lo[0]) / 2.0
z_center = (hi[2] + lo[2]) / 2.0
rgb_sensor.ortho_scale = max(x_extent, z_extent) / 2.0 * 1.02  # slight margin

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_sensor]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
pf = sim.pathfinder
agent = sim.get_agent(0)

panel_imgs = []
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
except Exception:
    font = ImageFont.load_default()

for floor_idx, floor_h in enumerate(FLOOR_HEIGHTS, start=1):
    cam_y = floor_h + 3.0  # well above this floor's ceiling, still below the floor above
    state = habitat_sim.AgentState()
    state.position = np.array([x_center, cam_y, z_center], dtype=np.float32)
    state.rotation = np.quaternion(1, 0, 0, 0)
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    rgb = obs["rgb_topdown"][:, :, :3].astype(np.uint8)
    render_h, render_w = rgb.shape[0], rgb.shape[1]

    navmesh = pf.get_topdown_view(METERS_PER_PIXEL, floor_h)  # shape (rows along Z, cols along X), True=navigable
    nav_img = Image.fromarray((navmesh * 255).astype(np.uint8)).resize((render_w, render_h), Image.NEAREST)
    nav_arr = np.array(nav_img) > 127

    composite = rgb.copy()
    green = np.array([46, 204, 64], dtype=np.uint8)
    alpha = 0.55
    composite[nav_arr] = (composite[nav_arr].astype(np.float32) * (1 - alpha) + green.astype(np.float32) * alpha).astype(np.uint8)

    area_m2 = float(navmesh.sum()) * (METERS_PER_PIXEL ** 2)
    im = Image.fromarray(composite)
    draw = ImageDraw.Draw(im)
    label = f"Floor {floor_idx}  (y={floor_h:.2f} m, navigable {area_m2:.1f} m2)"
    draw.rectangle([0, 0, im.width, 46], fill=(0, 0, 0))
    draw.text((10, 6), label, fill=(255, 255, 255), font=font)

    out_path = f"{OUT_DIR}/floor_{floor_idx}.png"
    im.save(out_path)
    panel_imgs.append(im)
    print(f"Floor {floor_idx}: y={floor_h:.2f}  navigable_area={area_m2:.2f} m^2  -> {out_path}")

sim.close()

# combined 2x2 grid for the paper
w, h = panel_imgs[0].size
grid = Image.new("RGB", (w * 2, h * 2), (255, 255, 255))
for i, im in enumerate(panel_imgs):
    grid.paste(im, ((i % 2) * w, (i // 2) * h))
grid_path = f"{OUT_DIR}/all_floors_grid.png"
grid.save(grid_path)
print(f"Combined grid -> {grid_path}")
