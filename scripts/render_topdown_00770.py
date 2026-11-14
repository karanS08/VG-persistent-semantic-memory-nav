"""Orthographic top-down RGB render of scene 00770 for the teaser figure.
Same method as render_topdown_navmesh_scene00337.py (ortho camera looking
straight down), single floor at the robot's floor height. Writes the image and
the world->pixel mapping, verified by overlaying the navmesh.
Out: acra_paper/figures/teaser/topdown_00770.png, topdown_00770.json"""
import json
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
OUT = Path("acra_paper/figures/teaser"); OUT.mkdir(parents=True, exist_ok=True)
FLOOR_Y = 0.107      # robot floor height in every 00770 run
CAM_H = 2.3          # camera height above the floor, below the ceiling
SENSOR_Y = 1.5       # CameraSensorSpec default offset above the agent base
RES = 1600

sim_cfg = habitat_sim.SimulatorConfiguration(); sim_cfg.scene_id = SCENE
spec = habitat_sim.CameraSensorSpec()
spec.uuid = "top"; spec.sensor_type = habitat_sim.SensorType.COLOR
spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
spec.resolution = [RES, RES]
spec.orientation = np.array([-np.pi / 2, 0.0, 0.0], dtype=np.float32)

probe = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()]))
nav_pts = np.array([probe.pathfinder.get_random_navigable_point() for _ in range(4000)])
nav_pts = nav_pts[np.abs(nav_pts[:, 1] - FLOOR_Y) < 0.3]
probe.close()
# centre on the navigable area (the bounding-box centre lies outside the house
# and renders black); half-extent from the navigable points, not the bbox
x0, z0 = nav_pts[:, [0, 2]].min(0); x1, z1 = nav_pts[:, [0, 2]].max(0)
half = max(x1 - x0, z1 - z0) / 2 * 1.05
# the ortho camera must sit over the house interior (over empty space it renders
# black): use the navigable point closest to the navigable-area centre
c = np.array([(x0 + x1) / 2, (z0 + z1) / 2])
xc, zc = nav_pts[np.argmin(((nav_pts[:, [0, 2]] - c) ** 2).sum(1)), [0, 2]]
spec.ortho_scale = 0.5 / half  # habitat-sim: visible half-width = 1/(2*ortho_scale); mapping is re-fitted below
agent_cfg = habitat_sim.agent.AgentConfiguration(); agent_cfg.sensor_specifications = [spec]
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
st = habitat_sim.AgentState(); st.position = np.array([xc, FLOOR_Y + CAM_H - SENSOR_Y, zc], dtype=np.float32)
st.rotation = np.quaternion(1, 0, 0, 0); sim.get_agent(0).set_state(st)
rgb = sim.get_sensor_observations()["top"][:, :, :3]
Image.fromarray(rgb).save(OUT / "topdown_00770.png")

# mapping check: navigable points should land on floor pixels
mapping = {"xc": float(xc), "zc": float(zc), "half_extent_m": float(half), "res": RES,
           "px_per_m": RES / (2 * half), "note": "col=(x-xc)*px_per_m+res/2, row=(z-zc)*px_per_m+res/2"}
pts = np.array([sim.pathfinder.get_random_navigable_point() for _ in range(400)])
cols = ((pts[:, 0] - xc) * mapping["px_per_m"] + RES / 2).astype(int)
rows = ((pts[:, 2] - zc) * mapping["px_per_m"] + RES / 2).astype(int)
ok = (cols >= 0) & (cols < RES) & (rows >= 0) & (rows < RES)
check = rgb.copy()
for r, c in zip(rows[ok], cols[ok]):
    check[max(r - 3, 0):r + 3, max(c - 3, 0):c + 3] = [255, 0, 0]
Image.fromarray(check).save(OUT / "topdown_00770_mappingcheck.png")


# ---- fit the world->pixel mapping empirically (habitat-sim's ortho scale
# convention is not documented; a wrong guess silently misplaces every overlay).
# Navigable floor points must land on rendered (non-black) floor pixels.
floor = rgb.sum(2) > 30
pts = np.array([sim.pathfinder.get_random_navigable_point() for _ in range(3000)])
pts = pts[np.abs(pts[:, 1] - FLOOR_Y) < 0.3][:, [0, 2]]
best = (-1, None)
BLK = 32  # IoU on a coarse grid: rewards covering the whole floor, not just landing on it
floor_blk = floor.reshape(RES // BLK, BLK, RES // BLK, BLK).mean((1, 3)) > 0.3
def score(swap, sx, sz, ppm, dc, dr):
    u, v = (pts[:, 1], pts[:, 0]) if swap else (pts[:, 0], pts[:, 1])
    uc, vc = (zc, xc) if swap else (xc, zc)
    cols = (sx * (u - uc) * ppm + RES / 2 + dc).astype(int)
    rows = (sz * (v - vc) * ppm + RES / 2 + dr).astype(int)
    ok = (cols >= 0) & (cols < RES) & (rows >= 0) & (rows < RES)
    m = np.zeros_like(floor_blk)
    m[rows[ok] // BLK, cols[ok] // BLK] = True
    return (m & floor_blk).sum() / max((m | floor_blk).sum(), 1)
for swap in (0, 1):
    for sx in (1, -1):
        for sz in (1, -1):
            for ppm in np.arange(20, 121, 4):
                for dc in range(-400, 401, 40):
                    for dr in range(-400, 401, 40):
                        h = score(swap, sx, sz, ppm, dc, dr)
                        if h > best[0]:
                            best = (h, (swap, sx, sz, ppm, dc, dr))
swap, sx, sz, ppm0, dc0, dr0 = best[1]
for ppm in np.arange(ppm0 - 4, ppm0 + 4.01, 0.25):   # refine
    for dc in range(dc0 - 40, dc0 + 41, 4):
        for dr in range(dr0 - 40, dr0 + 41, 4):
            h = score(swap, sx, sz, ppm, dc, dr)
            if h > best[0]:
                best = (h, (swap, sx, sz, float(ppm), dc, dr))
hit, (swap, sx, sz, ppm, dc, dr) = best

mapping.update({"swap_xz": int(swap), "sign_u": sx, "sign_v": sz, "px_per_m": ppm, "dcol": dc, "drow": dr,
                "fit_hit_rate": round(float(hit), 4),
                "note": "(u,v)=(z,x) if swap_xz else (x,z); col=sign_u*(u-uc)*px_per_m+res/2+dcol, "
                        "row=sign_v*(v-vc)*px_per_m+res/2+drow, (uc,vc) likewise from (xc,zc) (fitted)"})
json.dump(mapping, open(OUT / "topdown_00770.json", "w"), indent=1)
check = rgb.copy()
u, v = (pts[:, 1], pts[:, 0]) if swap else (pts[:, 0], pts[:, 1]); uc, vc = (zc, xc) if swap else (xc, zc)
cols = (sx * (u - uc) * ppm + RES / 2 + dc).astype(int); rows = (sz * (v - vc) * ppm + RES / 2 + dr).astype(int)
for r, c in zip(rows, cols):
    if 0 <= r < RES and 0 <= c < RES:
        check[max(r - 3, 0):r + 3, max(c - 3, 0):c + 3] = [255, 0, 0]
Image.fromarray(check).save(OUT / "topdown_00770_mappingcheck.png")
print("fitted", mapping)
sim.close()
