"""High-resolution, path-free maps of HM3D 00770 (2026-09-29): the same
orthographic top-down camera as render_topdown_00770.py (same centre and
extent, so overlays still line up after scaling by RES/1600), cropped to the
house on a transparent background; plus 4K isometric renders.
Out: acra_paper/teaser_assets/highres/"""
import json, math
from pathlib import Path
import habitat_sim
import numpy as np
from PIL import Image

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
OUT = Path("acra_paper/teaser_assets/highres"); OUT.mkdir(parents=True, exist_ok=True)
m = json.load(open("acra_paper/figures/teaser/topdown_00770.json"))
RES = 6400
c = habitat_sim.SimulatorConfiguration(); c.scene_id = SCENE
s = habitat_sim.CameraSensorSpec(); s.uuid = "top"; s.sensor_type = habitat_sim.SensorType.COLOR
s.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC; s.resolution = [RES, RES]
s.orientation = np.array([-np.pi / 2, 0.0, 0.0], dtype=np.float32); s.ortho_scale = 0.5 / m["half_extent_m"]
a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = [s]
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))
st = habitat_sim.AgentState(); st.position = np.array([m["xc"], 0.107 + 2.3 - 1.5, m["zc"]], dtype=np.float32)
st.rotation = np.quaternion(1, 0, 0, 0); sim.get_agent(0).set_state(st)
rgb = sim.get_sensor_observations()["top"][:, :, :3]; sim.close()
mask = rgb.sum(2) > 30
r, cc = np.where(mask); pad = 40
r0, r1, c0, c1 = max(r.min() - pad, 0), r.max() + pad, max(cc.min() - pad, 0), cc.max() + pad
rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])[r0:r1, c0:c1]
Image.fromarray(rgba, "RGBA").save(OUT / "map_topdown_house_transparent.png")
Image.fromarray(rgb[r0:r1, c0:c1]).save(OUT / "map_topdown_house_black.png")
white = rgb[r0:r1, c0:c1].copy(); white[~mask[r0:r1, c0:c1]] = 255
Image.fromarray(white).save(OUT / "map_topdown_house_white.png")
json.dump({"render_res": RES, "crop_origin_px": [int(c0), int(r0)], "scale_vs_1600_render": RES / 1600,
           "note": "pixel = (1600-render pixel) * scale - crop_origin; 1600-render mapping in figures/teaser/topdown_00770.json"},
          open(OUT / "map_topdown_house_highres.json", "w"), indent=1)
print("topdown", rgba.shape, "px_per_m ~", round(m["px_per_m"] * RES / 1600, 1))
