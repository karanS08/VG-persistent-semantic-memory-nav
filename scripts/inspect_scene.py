"""Quick geometry check of an HM3D scene (2026-09-29): floor levels (clusters of
navigable-point heights), walkable area, footprint, and an orthographic top-down
render per floor. Usage: python scripts/inspect_scene.py <scene.glb> <out_dir>"""
import json, sys
import numpy as np, habitat_sim
from PIL import Image

glb, out = sys.argv[1], sys.argv[2]
c = habitat_sim.SimulatorConfiguration(); c.scene_id = glb
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [habitat_sim.agent.AgentConfiguration()]))
pf = sim.pathfinder
pts = np.array([pf.get_random_navigable_point() for _ in range(20000)])
ys = np.sort(pts[:, 1]); levels = [[ys[0]]]
for y in ys[1:]:
    (levels[-1].append(y) if y - levels[-1][-1] < 0.5 else levels.append([y]))
levels = [(float(np.median(l)), len(l)) for l in levels if len(l) > 200]   # ignore stairs/ledges
info = {"navigable_area_m2": round(pf.navigable_area, 1), "floors": []}
sim.close()
for i, (fy, n) in enumerate(levels):
    P = pts[np.abs(pts[:, 1] - fy) < 0.3]
    x0, x1, z0, z1 = P[:, 0].min(), P[:, 0].max(), P[:, 2].min(), P[:, 2].max()
    half = max(x1 - x0, z1 - z0) / 2 * 1.05
    s = habitat_sim.CameraSensorSpec(); s.uuid = "top"; s.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    s.resolution = [1200, 1200]; s.position = [0.0, 0.0, 0.0]; s.orientation = np.array([-np.pi / 2, 0, 0], np.float32)
    s.ortho_scale = 0.5 / half; s.far = 100.0
    a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = [s]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))
    cen = np.array([(x0 + x1) / 2, fy, (z0 + z1) / 2]); cen = np.array(sim.pathfinder.snap_point(cen))
    st = habitat_sim.AgentState(); st.position = cen + np.array([0, 2.0, 0])   # just under the ceiling
    sim.get_agent(0).set_state(st)
    Image.fromarray(sim.get_sensor_observations()["top"][:, :, :3]).save(f"{out}/topdown_floor{i}.png")
    sim.close()
    info["floors"].append({"floor_y": round(fy, 2), "share_of_walkable_points": round(n / len(pts), 2),
                           "footprint_m": [round(float(x1 - x0), 1), round(float(z1 - z0), 1)]})
json.dump(info, open(f"{out}/scene_info.json", "w"), indent=1); print(json.dumps(info))
