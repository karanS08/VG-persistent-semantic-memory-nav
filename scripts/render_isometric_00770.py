"""Isometric render of HM3D 00770 (2026-09-29): orthographic camera, yaw 45 deg,
pitch -35.26 deg (true isometric), placed far back along the view direction so
the whole floor is in front of it. Ceilings face down and are back-face culled,
so rooms read as an open cut-away. Real scene geometry only.
Usage: python scripts/render_isometric_00770.py [yaw_deg] [pitch_deg]
Out: acra_paper/teaser_assets/isometric_00770_yaw<yaw>.png"""
import math
import sys
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
YAW = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
PITCH = float(sys.argv[2]) if len(sys.argv) > 2 else -35.26
W, H = 1920, 1080
FLOOR_Y = 0.107

c = habitat_sim.SimulatorConfiguration(); c.scene_id = SCENE
probe = habitat_sim.Simulator(habitat_sim.Configuration(c, [habitat_sim.agent.AgentConfiguration()]))
pts = np.array([probe.pathfinder.get_random_navigable_point() for _ in range(4000)])
probe.close()
pts = pts[np.abs(pts[:, 1] - FLOOR_Y) < 0.3]
center = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, FLOOR_Y + 1.0, (pts[:, 2].min() + pts[:, 2].max()) / 2])
extent = np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 2]))            # house diagonal, metres

s = habitat_sim.CameraSensorSpec(); s.uuid = "iso"; s.sensor_type = habitat_sim.SensorType.COLOR
s.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC; s.resolution = [H, W]
s.position = [0.0, 0.0, 0.0]
s.orientation = np.array([math.radians(PITCH), 0.0, 0.0], dtype=np.float32)
s.ortho_scale = 0.5 / (extent / 2 * 1.05) * (W / H) ** 0   # visible half-width = 1/(2*scale) (fitted earlier)
s.far = 500.0
a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = [s]
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))
yaw = math.radians(YAW)
fwd = np.array([-math.sin(yaw) * math.cos(math.radians(PITCH)), math.sin(math.radians(PITCH)),
                -math.cos(yaw) * math.cos(math.radians(PITCH))])                 # camera looks along fwd
st = habitat_sim.AgentState()
st.position = center - fwd * 60.0                                                 # far back and up
st.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
sim.get_agent(0).set_state(st, reset_sensors=True, infer_sensor_states=True)
rgb = sim.get_sensor_observations()["iso"][:, :, :3]
out = Path("acra_paper/teaser_assets") / f"isometric_00770_yaw{int(YAW)}.png"
Image.fromarray(rgb).save(out)
print("wrote", out, "mean", round(float(rgb.mean()), 1), "diag_m", round(float(extent), 1))
sim.close()

# ---- world -> pixel mapping for overlays (camera basis + fitted scale/offset) ----
import json  # noqa: E402
p = math.radians(PITCH)
f = np.array([-math.sin(yaw) * math.cos(p), math.sin(p), -math.cos(yaw) * math.cos(p)])
r = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
u = np.cross(-f, r)
cam = st.position
floor = rgb.sum(2) > 30
sim2 = habitat_sim.Simulator(habitat_sim.Configuration(c, [habitat_sim.agent.AgentConfiguration()]))
nav = np.array([sim2.pathfinder.get_random_navigable_point() for _ in range(3000)]); sim2.close()
nav = nav[np.abs(nav[:, 1] - FLOOR_Y) < 0.3]
a_r, a_u = (nav - cam) @ r, (nav - cam) @ u
BLK = 24
fb = floor[: H // BLK * BLK, : W // BLK * BLK].reshape(H // BLK, BLK, W // BLK, BLK).mean((1, 3)) > 0.2
best = (-1, None)
ppm0 = W / 2 * 2 * s.ortho_scale   # habitat ortho: visible half-width = 1/(2*ortho_scale) m -> W/2 px
for sr in (1, -1):
    for su in (1, -1):
        for ppm in ppm0 * np.array([0.9, 0.95, 1.0, 1.05, 1.1]):
            for dc in range(-300, 301, 12):
                for dr in range(-300, 301, 12):
                    cols = (W / 2 + sr * a_r * ppm + dc).astype(int); rows = (H / 2 - su * a_u * ppm + dr).astype(int)
                    ok = (cols >= 0) & (cols < W // BLK * BLK) & (rows >= 0) & (rows < H // BLK * BLK)
                    m = np.zeros_like(fb); m[rows[ok] // BLK, cols[ok] // BLK] = True
                    iou = (m & fb).sum() / max((m | fb).sum(), 1)
                    if iou > best[0]:
                        best = (iou, (float(ppm), dc, dr, sr, su))
iou, (ppm, dc, dr, sr, su) = best
r, u = r * sr, u * su
a_r, a_u = (nav - cam) @ r, (nav - cam) @ u
json.dump({"yaw": YAW, "pitch": PITCH, "W": W, "H": H, "cam": cam.tolist(), "right": r.tolist(), "up": u.tolist(),
           "px_per_m": ppm, "dcol": dc, "drow": dr, "fit_iou": round(float(iou), 3),
           "note": "col = W/2 + (P-cam).right*px_per_m + dcol ; row = H/2 - (P-cam).up*px_per_m + drow"},
          open(out.with_suffix(".json"), "w"), indent=1)
chk = rgb.copy()
cols = (W / 2 + a_r * ppm + dc).astype(int); rows = (H / 2 - a_u * ppm + dr).astype(int)
for rr, cc in zip(rows, cols):
    if 0 <= rr < H and 0 <= cc < W:
        chk[max(rr - 2, 0):rr + 2, max(cc - 2, 0):cc + 2] = [255, 0, 0]
Image.fromarray(chk).save(str(out).replace(".png", "_check.png"))
print("mapping fit iou", round(float(iou), 3), "px_per_m", ppm, dc, dr)
