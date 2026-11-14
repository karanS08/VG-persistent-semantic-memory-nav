"""Night-queue setup for the multi-house test (test 3), run once BEFORE the queue (needs THOR, not DAAAM).

For each candidate house (ProcTHOR train idx; chosen offline: exactly one sofa, >=5 rooms, floor area > 1.2x house 1,
never house 1 or the image-check calibration houses 2-40):
  * relocation spot: the first reachable cell (in order of distance, starting at >= MIN_D m from the sofa's original
    position) where thor_env.relocate_clear succeeds (clear floor, house stays connected, agent can stand nearby);
    re-checked by repeating the relocation in a fresh environment (deterministic placement at run time);
  * 6 start poses: reachable in BOTH the original and the moved layout, >= MIN_START m from both sofa positions,
    spread by farthest-point sampling (seeded), yaw from {0, 90, 180, 270} (seeded);
  * tour waypoints: one per room (room-polygon centroid snapped to the nearest reachable cell), greedy order from spawn,
    in the Habitat frame used by --tour-waypoints.
Writes configs/thor_night/house_<idx>.json; exits non-zero for a house that fails any check (that house is then not
used by the queue).  Usage (rpv_thor env, repo root): python scripts/thor/night/prepare_houses.py 72 99 314
"""
import json, math, random, sys
from pathlib import Path
import yaml, prior
sys.path.insert(0, "packages/sim_bridge")
from sim_bridge.thor_env import ThorEnv

MIN_D, MIN_START, N_START = 5.0, 3.0, 6
cfg = yaml.safe_load(open("configs/habitat.yaml"))
ds = prior.load_dataset("procthor-10k")["train"]
out_dir = Path("configs/thor_night"); out_dir.mkdir(parents=True, exist_ok=True)


def walk(o):
    yield o
    for c in o.get("children", []) or []:
        yield from walk(c)


def centroid(poly):
    return sum(p["x"] for p in poly) / len(poly), sum(p["z"] for p in poly) / len(poly)


def reachable(env):
    r = env.controller.step("GetReachablePositions").metadata["actionReturn"] or []
    return {(round(p["x"], 2), round(p["z"], 2)) for p in r}


ok_all = True
for idx in map(int, sys.argv[1:]):
    try:
        house = ds[idx]
        sofas = [x for o in house["objects"] for x in walk(o) if x["id"].split("|")[0] == "Sofa"]
        assert len(sofas) == 1, f"expected 1 sofa, found {len(sofas)}"
        env = ThorEnv(cfg, house=house); env.reset()
        meta = {o["objectId"]: o for o in env.controller.last_event.metadata["objects"]}
        sofa_id = next(k for k in meta if k.startswith("Sofa"))
        c0 = meta[sofa_id]["axisAlignedBoundingBox"]["center"]
        orig = (c0["x"], c0["z"])
        reach0 = reachable(env)
        spawn = (env._spawn[0]["x"], env._spawn[0]["z"])
        # tour waypoints (static layout)
        wps = []
        for room in house["rooms"]:
            cx, cz = centroid(room["floorPolygon"])
            near = min(reach0, key=lambda p: math.hypot(p[0] - cx, p[1] - cz))
            if math.hypot(near[0] - cx, near[1] - cz) < 2.5 and near not in wps:
                wps.append(near)
        order, here = [], spawn
        while wps:
            nxt = min(wps, key=lambda p: math.hypot(p[0] - here[0], p[1] - here[1])); wps.remove(nxt); order.append(nxt); here = nxt
        assert len(order) >= 4, f"only {len(order)} tour waypoints"
        # relocation spot
        cands = sorted((p for p in reach0 if math.hypot(p[0] - orig[0], p[1] - orig[1]) >= MIN_D),
                       key=lambda p: math.hypot(p[0] - orig[0], p[1] - orig[1]))
        spot = None
        for p in cands[:400]:
            pos = env.relocate_clear(sofa_id, p[0], p[1], max_drift=0.5)
            if pos:
                spot = (round(pos[0], 3), round(pos[1], 3)); break
        assert spot, "no clear relocation spot >= 5 m"
        reach1 = reachable(env)
        env.close()
        # determinism re-check in a fresh environment
        env = ThorEnv(cfg, house=house); env.reset()
        pos2 = env.relocate_clear(sofa_id, spot[0], spot[1])
        assert pos2 and math.hypot(pos2[0] - spot[0], pos2[1] - spot[1]) < 0.05, f"re-placement drifted: {pos2} vs {spot}"
        env.close()
        # start poses valid in both layouts
        both = [p for p in reach0 & reach1 if math.hypot(p[0] - orig[0], p[1] - orig[1]) >= MIN_START
                and math.hypot(p[0] - spot[0], p[1] - spot[1]) >= MIN_START]
        assert len(both) >= 50, f"too few start cells ({len(both)})"
        rng = random.Random(idx)
        starts = [max(both, key=lambda p: math.hypot(p[0] - orig[0], p[1] - orig[1]))]
        while len(starts) < N_START:
            starts.append(max(both, key=lambda p: min(math.hypot(p[0] - s[0], p[1] - s[1]) for s in starts)))
        starts = [f"{x},{z},{rng.choice([0, 90, 180, 270])}" for x, z in starts]
        cfg_out = {"house": idx, "sofa_id": sofa_id, "sofa_orig_thor_xz": [round(orig[0], 3), round(orig[1], 3)],
                   "sofa_new_thor_xz": list(spot), "reloc_spec": f"0:{sofa_id}:{spot[0]}:{spot[1]}",
                   "move_distance_m": round(math.hypot(spot[0] - orig[0], spot[1] - orig[1]), 2),
                   "starts": starts, "tour_waypoints_habitat": [[x, 0.0, -z] for x, z in order],
                   "n_reachable_static": len(reach0), "n_reachable_moved": len(reach1)}
        json.dump(cfg_out["tour_waypoints_habitat"], open(out_dir / f"tour_house{idx}.json", "w"))
        json.dump(cfg_out, open(out_dir / f"house_{idx}.json", "w"), indent=1)
        print(f"house {idx}: OK sofa {sofa_id} moved {cfg_out['move_distance_m']} m, {len(order)} tour waypoints, "
              f"starts {starts}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- one bad house must not stop the others
        ok_all = False
        print(f"house {idx}: FAILED setup ({exc})", flush=True)
sys.exit(0 if ok_all else 1)
