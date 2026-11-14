"""Close-up renders of the five spread-demo targets (2026-09-29): for each query,
the tour-memory entity near its spot whose caption names the target; camera at a
navigable point 1.2-2.5 m away, looking at the remembered position, keeping the
view whose centre depth agrees with the distance (line of sight). Several
candidates per object saved so a person can pick. Real scene renders.
Out: acra_paper/teaser_assets/spread/obj_<k>_<v>.png + spread_objects.json"""
import json, math, sys
import numpy as np, yaml, habitat_sim
from PIL import Image
from habitat_sim.utils.common import quat_from_two_vectors
sys.path.insert(0, "packages/memory_layer")
from memory_layer.relevance import ClipTextEncoder

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
OUT = "acra_paper/teaser_assets/spread"
SPOTS = [("a black vase", "NE corner [1]"), ("a faucet", "north annex [2]"), ("a sink", "NW room [3]"),
         ("a whiteboard", "SW room [4]"), ("a painting of the sea", "SE corner [5]")]
cfg = yaml.safe_load(open("configs/memory.yaml"))["embedding"]
enc = ClipTextEncoder(cfg["clip_model_name"], cfg["clip_model_dataset"])
wps = {w["name"]: w["xyz"] for w in json.load(open("configs/tours/00770_full_tour_named.json"))}
snap = json.load(open("data/runs/phase4_batch_persist_memory_only_house_tour/memory_snapshot_ep0.json"))
E = [e for e in snap["entities"] if e.get("embedding") and e["position_world"]]
M = np.array([e["embedding"] for e in E]); M /= np.linalg.norm(M, axis=1, keepdims=True)
P = np.array([[e["position_world"][0], e["position_world"][2]] for e in E])

c = habitat_sim.SimulatorConfiguration(); c.scene_id = SCENE
specs = []
for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
    s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = st; s.resolution = [480, 640]
    s.position = [0.0, 0.0, 0.0]; s.hfov = 75.0; specs.append(s)
a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = specs
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))
meta = []
for k, (q, spot) in enumerate(SPOTS):
    v = enc.embed(q); v /= np.linalg.norm(v); r = M @ v
    d = np.linalg.norm(P - np.array([wps[spot][0], wps[spot][2]]), axis=1)
    noun = {"a black vase": "vase", "a faucet": "faucet", "a sink": "sink", "a whiteboard": "whiteboard",
            "a painting of the sea": "sea"}[q]
    named = np.array([f" {noun}" in " " + e["description"][:40].lower() for e in E])
    i = int(np.argmax(np.where((d < 4.5) & named, r, -1)))  # the remembered object that IS the target
    e = E[i]; T = np.array(e["position_world"], float)
    views = []
    for j in range(16):
        ang = 2 * math.pi * j / 16
        for rad in (1.6, 2.2, 1.2):
            nav = np.array(sim.pathfinder.snap_point(T + np.array([rad * math.cos(ang), 0, rad * math.sin(ang)])))
            if not np.all(np.isfinite(nav)) or abs(nav[1] - 0.107) > 0.5:
                continue
            cam = nav + np.array([0, 1.3, 0]); dv = T - cam; dist = np.linalg.norm(dv)
            if not 0.8 < dist < 3.0:
                continue
            st = habitat_sim.AgentState(); st.position = cam
            st.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), dv / dist)
            sim.get_agent(0).set_state(st, reset_sensors=True, infer_sensor_states=False)
            o = sim.get_sensor_observations()
            dc = float(np.median(o["depth"][230:250, 310:330]))
            views.append((abs(dc - dist) / dist, o["rgb"][:, :, :3]))  # best line-of-sight first
            break
    views.sort(key=lambda t: t[0])
    for vi, (_, img) in enumerate(views[:4]):
        Image.fromarray(img).save(f"{OUT}/obj_{k}_{vi}.png")
    meta.append({"k": k, "query": q, "spot": spot, "entity_id": e["id"], "relevance": round(float(r[i]), 3),
                 "position_world": e["position_world"], "caption": e["description"], "n_views": len(views)})
    print(k, q, e["id"], round(float(r[i]), 3), len(views), e["description"][:70])
json.dump(meta, open(f"{OUT}/spread_objects.json", "w"), indent=1)
sim.close()
