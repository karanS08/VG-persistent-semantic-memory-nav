"""Corrected test 3 (2026-11-03). The first tours of houses 72/99 never saw the sofa at its original spot (house 72:
closest approach 3.8 m, and the 3000-step budget ran out before the last 3 rooms), so both seeds were rejected.
Fix: the tour additionally visits the navmesh point nearest the sofa (snap_point of its centre; inserted after the
nearest room waypoint) and gets a 6000-step budget. Everything else (episodes, seed rules, scoring) is unchanged.
Disclosure: the seed memory is a tour that includes a look at the sofa (the test needs a memory of it)."""
import json, math
old = json.load(open("configs/thor_night/jobs.json"))
jobs = []
for idx in (72, 99):
    cfg = json.load(open(f"configs/thor_night/house_{idx}.json"))
    ox, oz = cfg["sofa_orig_thor_xz"]; sofa = [ox, 0.0, -oz]
    wps = list(cfg["tour_waypoints_habitat"])
    k = min(range(len(wps)), key=lambda i: math.hypot(wps[i][0] - sofa[0], wps[i][2] - sofa[2]))
    wps.insert(k + 1, sofa)
    json.dump(wps, open(f"configs/thor_night2/tour2_house{idx}.json", "w"))
    jobs.append({"kind": "tour", "tag": f"tour2_house{idx}", "house": idx, "age": -1,
                 "args": ["--queries", "a bicycle", "--persist-memory-only", "--tour-waypoints",
                          f"configs/thor_night2/tour2_house{idx}.json", "--tour-max-steps", "6000", "--max-steps", "100"],
                 "post": f"$HOME/miniconda3/envs/rpv_thor/bin/python scripts/thor/night/validate_seed.py {idx} tour2_house{idx}"})
    jobs += [j for j in old if j["tag"].startswith(f"h{idx}_")]
jobs += [j for j in old if j["tag"].startswith(("h72v8_", "h99v8_"))]
json.dump(jobs, open("configs/thor_night2/jobs.json", "w"), indent=1)
print(len(jobs), "jobs;", [j["tag"] for j in jobs if j["kind"] == "tour"])
