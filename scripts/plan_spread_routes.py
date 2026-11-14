"""Planned (illustrative) memory-ON routes for the spread infographic (2026-09-29):
navmesh shortest paths from the episode spawn to each of the user's five marked spots
(configs/tours/00770_full_tour_named.json), i.e. the route a correct memory recall of
an object there produces. NOT agent runs. Routes go to the spots, not to the DAAAM
object positions, which are 1-2 m off (some behind walls). Eye-level room views from
each spot in 8 directions for the cards.
Out: acra_paper/teaser_assets/spread/planned_routes.json, card_<k>_<v>.png"""
import json, math
import numpy as np, habitat_sim
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis

SCENE = "data/hm3d/example/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb"
OUT = "acra_paper/teaser_assets/spread"
START = np.array([-9.7289, 0.10685, 9.7313])   # spawn used by every 00770 run (decisions_ep0 step 0)
objs = json.load(open(f"{OUT}/spread_objects.json"))
WPS = {w["name"]: w["xyz"] for w in json.load(open("configs/tours/00770_full_tour_named.json"))}
c = habitat_sim.SimulatorConfiguration(); c.scene_id = SCENE
s = habitat_sim.CameraSensorSpec(); s.uuid = "rgb"; s.resolution = [480, 640]; s.position = [0.0, 1.25, 0.0]; s.hfov = 80.0
a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = [s]
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))
start = np.array(sim.pathfinder.snap_point(START))
routes = []
for o in objs:
    g = np.array(sim.pathfinder.snap_point(np.array(WPS[o["spot"]])))
    sp = habitat_sim.ShortestPath(); sp.requested_start = start; sp.requested_end = g
    assert sim.pathfinder.find_path(sp), o["spot"]
    L, pts = sp.geodesic_distance, [list(map(float, p)) for p in sp.points]
    for v in range(8):
        st = habitat_sim.AgentState(); st.position = g
        st.rotation = quat_from_angle_axis(2 * math.pi * v / 8, np.array([0.0, 1.0, 0.0]))
        sim.get_agent(0).set_state(st)
        Image.fromarray(sim.get_sensor_observations()["rgb"][:, :, :3]).save(f"{OUT}/card_{o['k']}_{v}.png")
    routes.append({"k": o["k"], "query": o["query"], "spot": o["spot"], "geodesic_m": round(L, 2), "goal": g.tolist(), "path_world": pts})
    print(o["k"], o["query"], "route", round(L, 1), "m")
json.dump({"_doc": __doc__, "start": start.tolist(), "routes": routes}, open(f"{OUT}/planned_routes.json", "w"), indent=1)
sim.close()
