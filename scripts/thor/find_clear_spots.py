"""Enumerate clear-floor placements for each sofa in ProcTHOR house 1 (thor_env.relocate_clear's rules), keeping spots
>= MIN_D m (5 m; verify distance is 1.5 m) from every remembered sofa entry in the seed tour. Output: results/post_paper/thor_clear_sofa_spots.json."""
import json, math, sys, yaml, numpy as np, prior
from sim_bridge.thor_env import ThorEnv

MIN_D, STEP = 5.0, 1.0
cfg = yaml.safe_load(open("configs/habitat.yaml"))
env = ThorEnv(cfg, house=prior.load_dataset("procthor-10k")["train"][1])
seed = json.load(open("data/runs/seed_alarm_tour.json"))["entities"]
mem = [e["position_world"] for e in seed if e.get("position_world") and any(k in e["description"][:70].lower() for k in ("sofa", "couch"))]
arr = env.pathfinder._arr.copy()
grid = sorted({(round(x / STEP) * STEP, round(z / STEP) * STEP) for x, z in arr})
out = {}
for oid in ("Sofa|3|3", "Sofa|5|2"):
    ok = []
    for gx, gz in grid:
        # memory positions are Habitat frame (x, y, -z_thor)
        if min(math.hypot(gx - m[0], gz + m[2]) for m in mem) < MIN_D:
            continue
        pos = env.relocate_clear(oid, gx, gz, max_drift=0.5)
        if pos:
            ok.append([round(pos[0], 2), round(pos[1], 2)])
    out[oid] = ok
    print(oid, len(ok), ok, flush=True)
    env.relocate_clear(oid, *ok[0], max_drift=0.5) if ok else None  # leave it somewhere clear for the next sofa's search
json.dump(out, open("results/post_paper/thor_clear_sofa_spots.json", "w"), indent=1)
env.close()
