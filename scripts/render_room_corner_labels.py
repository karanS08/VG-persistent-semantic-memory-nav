"""Corner-of-the-room perspective view of HM3D 00770 with remembered objects projected on top (2026-10-03, mock).
Camera: standing in a room corner, looking toward the opposite corner (hfov 100). Remembered objects: the scripted
house-tour memory (real DAAAM captions, estimated 3D positions). A point is drawn only if it is in front of the visible
surface (depth test, 1.2 m tolerance for DAAAM's 1-2 m position error). Labels = head noun of the caption.
Usage: python scripts/render_room_corner_labels.py [n_candidates]  -> scratchpad/corner_cand_<k>.png and a JSON of scores
"""
import json, math, re, sys
import numpy as np, habitat_sim, quaternion  # noqa: F401
from PIL import Image, ImageDraw, ImageFont

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
TOUR = "data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-1004/-home-ksharma-mem-ws/228ff7f4-4e94-4d56-9832-54bf4b393403/scratchpad"
W, H, HFOV = 1600, 900, 100.0
KEEP = {"radiator", "window", "door", "doorway", "cabinet", "mirror", "painting", "curtain", "lamp", "chandelier", "table", "bench",
        "vent", "fireplace", "chair", "sculpture", "clock", "switch", "tapestry", "vase", "rug", "drawer", "shelf", "sink", "bathtub", "toilet"}
STOP = {"with", "featuring", "features", "that", "which", "is", "are", "has", "of", "on", "in", "and", "mounted", "set", "made", "framed", "adorned", "covered", "depicts"}


def head(c):
    out = []
    for t in re.findall(r"[a-z]+", re.split(r"[.;:()]", c.lower())[0]):
        if t in STOP:
            break
        if t not in ("a", "an", "the"):
            out.append(t)
    return out[-1] if out else None


ents = [e for e in json.load(open(TOUR))["entities"] if e["position_world"]]
lab = [(head(e["description"]), np.array(e["position_world"])) for e in ents]
lab = [(h, p) for h, p in lab if h in KEEP]

c = habitat_sim.SimulatorConfiguration(); c.scene_id = SCENE
specs = []
for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
    s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = st; s.resolution = [H, W]
    s.position = [0.0, 1.6, 0.0]; s.hfov = HFOV; specs.append(s)
a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = specs
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a])); pf = sim.pathfinder
f = (W / 2) / math.tan(math.radians(HFOV) / 2)
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)


def view(pos, look_at):
    pos = np.array(pf.snap_point(np.array(pos, dtype=np.float32)))
    d = np.array(look_at) - pos; yaw = math.atan2(-d[0], -d[2])
    st = habitat_sim.AgentState(); st.position = pos
    st.rotation = quaternion.from_rotation_vector([0, yaw, 0])
    sim.get_agent(0).set_state(st)
    o = sim.get_sensor_observations()
    ss = sim.get_agent(0).get_state().sensor_states["rgb"]
    R = quaternion.as_rotation_matrix(ss.rotation)
    return o["rgb"][:, :, :3], o["depth"], np.array(ss.position), R


def project(rgb, depth, cam, R):
    pts = []
    for h, p in lab:
        pc = R.T @ (p - cam)            # camera frame: x right, y up, -z forward
        z = -pc[2]
        if z < 0.8 or z > 14:
            continue
        u = W / 2 + f * pc[0] / z; v = H / 2 - f * pc[1] / z
        if not (60 < u < W - 60 and 60 < v < H - 60):
            continue
        if z > depth[int(v), int(u)] + 1.2:     # behind a wall
            continue
        pts.append((h, u, v, z))
    return pts


def draw(rgb, pts, k=12):
    im = Image.fromarray(rgb).convert("RGB"); d = ImageDraw.Draw(im, "RGBA")
    chosen = []
    for h, u, v, z in sorted(pts, key=lambda t: t[3]):
        if all(math.hypot(u - cu, v - cv) > 200 for _, cu, cv, _ in chosen):
            chosen.append((h, u, v, z))
        if len(chosen) >= k:
            break
    for h, u, v, z in chosen:
        d.ellipse([u - 9, v - 9, u + 9, v + 9], fill=(255, 215, 0, 255), outline=(0, 0, 0, 255), width=3)
        t = h.capitalize(); tw = d.textlength(t, font=font)
        d.rectangle([u - tw / 2 - 8, v - 58, u + tw / 2 + 8, v - 18], fill=(0, 0, 0, 215))
        d.line([(u, v - 18), (u, v - 9)], fill=(255, 215, 0, 255), width=3)
        d.text((u - tw / 2, v - 54), t, fill="white", font=font)
    return im, chosen



CLS = {"radiator": "#f28e2b", "lamp": "#f28e2b", "chandelier": "#f28e2b", "vent": "#f28e2b", "switch": "#f28e2b", "sink": "#f28e2b",
       "bathtub": "#f28e2b", "toilet": "#f28e2b", "cabinet": "#4e79a7", "table": "#4e79a7", "bench": "#4e79a7", "chair": "#4e79a7",
       "drawer": "#4e79a7", "shelf": "#4e79a7", "window": "#17becf", "door": "#17becf", "doorway": "#17becf", "curtain": "#17becf",
       "painting": "#b07aa1", "mirror": "#b07aa1", "fireplace": "#b07aa1", "sculpture": "#b07aa1", "clock": "#b07aa1", "vase": "#b07aa1",
       "rug": "#b07aa1", "tapestry": "#b07aa1"}
if len(sys.argv) > 1 and sys.argv[1] == "final":
    pos, la = [4.2, 0.107, 9.5], [9.0, 1.0, 14.0]
    rgb, depth, cam, R = view(pos, la)
    TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    pts = []
    for h, p_ in lab:
        pc = R.T @ (p_ - cam); z = -pc[2]
        if z < 0.8 or z > 14:
            continue
        u = W / 2 + f * pc[0] / z; v = H / 2 - f * pc[1] / z
        if not (80 < u < W - 80 and 200 < v < 780):
            continue
        if abs(z - depth[int(v), int(u)]) > TOL:          # must lie ON the visible surface (strict depth test)
            continue
        pts.append((h, u, v, z))
    im = Image.fromarray(rgb).convert("RGB"); d = ImageDraw.Draw(im, "RGBA"); chosen = []
    for h, u, v, z in sorted(pts, key=lambda t: t[3]):
        if all(math.hypot(u - cu, v - cv) > 230 for _, cu, cv, _ in chosen):
            chosen.append((h, u, v, z))
        if len(chosen) >= 9:
            break
    for h, u, v, z in chosen:
        col = CLS.get(h, "#999999")
        d.ellipse([u - 10, v - 10, u + 10, v + 10], fill=col, outline="white", width=4)
        t = h.capitalize(); tw = d.textlength(t, font=font)
        d.rectangle([u - tw / 2 - 10, v - 62, u + tw / 2 + 10, v - 20], fill=col, outline="white", width=2)
        d.line([(u, v - 20), (u, v - 10)], fill="white", width=3)
        d.text((u - tw / 2, v - 58), t, fill="white", font=font)
    im = im.crop((0, 170, W, 790)); im.save("acra_paper/figures/corner_view_labels.png")
    print("final labels:", [h for h, *_ in chosen], "candidates passing depth test:", len(pts), "of", len(lab)); sim.close(); sys.exit(0)

# candidate corners of the blue south-east room (radiators on its south/east walls) looking to the opposite corner
cands = []
for cx in (3.4, 4.2, 5.0):
    for cz in (9.3, 10.0, 10.7):
        cands.append(([cx, 0.107, cz], [9.0, 1.0, 14.0]))
for cx in (9.4, 8.8):
    for cz in (9.4, 10.2):
        cands.append(([cx, 0.107, cz], [4.0, 1.0, 14.0]))
res = []
for k, (pos, la) in enumerate(cands[: int(sys.argv[1]) if len(sys.argv) > 1 else len(cands)]):
    rgb, depth, cam, R = view(pos, la)
    pts = project(rgb, depth, cam, R)
    im, chosen = draw(rgb, pts)
    im.save(f"{OUT}/corner_cand_{k}.png")
    res.append({"k": k, "pos": [round(x, 2) for x in cam.tolist()], "look_at": la, "n_visible": len(pts), "labelled": [h for h, *_ in chosen]})
    print(k, [round(x, 1) for x in cam.tolist()], "visible", len(pts), [h for h, *_ in chosen])
json.dump(res, open(f"{OUT}/corner_cands.json", "w"), indent=1)
sim.close()
