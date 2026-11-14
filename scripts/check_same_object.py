"""Same-object check for a memory-on/off pair (2026-09-29): for every search both
runs verified, compare the verified object coordinates (verification_detail.
entity_position_world), where the robot stopped (last decision position), and the
captions. A pair counts as the SAME object only if the verified coordinates are
within 1.5 m on the floor plane.
Exit code 0 = COMPLETE (at least one search after the shared first one found the
same object in both runs), 3 = not complete (caller re-runs the pair).
Usage: python scripts/check_same_object.py <on_run_dir> <off_run_dir>"""
import json, sys
import numpy as np

TOL = 1.5


def load(d):
    out = {}
    for e in json.load(open(f"{d}/batch_metrics.json"))["episodes"]:
        v = e.get("verification_detail") or {}
        lines = open(f"{d}/decisions_ep{e['episode_id']}.jsonl").read().split("\n")
        last = json.loads([l for l in lines if l.strip()][-1])
        out[e["episode_id"]] = (e, v.get("entity_position_world"), last["agent_position"], (v.get("description") or "")[:70])
    return out


on, off = load(sys.argv[1]), load(sys.argv[2])
rows = []
for k in on:
    eo, po, ro, co = on[k]; ef, pf, rf, cf = off[k]
    ok = eo["outcome"] == ef["outcome"] == "success" and po and pf
    d_obj = float(np.linalg.norm(np.array(po)[[0, 2]] - np.array(pf)[[0, 2]])) if ok else None
    d_rob = float(np.linalg.norm(np.array(ro)[[0, 2]] - np.array(rf)[[0, 2]]))
    same = ok and d_obj <= TOL
    rows.append({"episode": k, "query": eo["query"], "steps_on": eo["steps_taken"], "steps_off": ef["steps_taken"],
                 "outcome_on": eo["outcome"], "outcome_off": ef["outcome"], "object_dist_m": d_obj and round(d_obj, 2),
                 "robot_end_dist_m": round(d_rob, 2), "same_object": bool(same), "caption_on": co, "caption_off": cf})
    print(f"ep{k} {eo['query']:16s} on {eo['steps_taken']:3d} {eo['outcome'][:4]} | off {ef['steps_taken']:3d} {ef['outcome'][:4]} | "
          f"object {rows[-1]['object_dist_m']} m, robot ends {d_rob:.1f} m apart -> {'SAME' if same else 'different/NA'}\n"
          f"      on : {co}\n      off: {cf}")
json.dump(rows, open(f"{sys.argv[1]}/same_object_check.json", "w"), indent=1)
done = [r for r in rows if r["same_object"] and r["episode"] > 0]
print("COMPLETE:" if done else "NOT COMPLETE: no search after the first found the same object in both runs",
      [(r["query"], r["steps_on"], r["steps_off"]) for r in done])
sys.exit(0 if done else 3)
