"""Ground-truth scoring of verified successes on HM3D 00861 (Test 3.4, 2026-09-29).

00861 is the only scene here with HM3D-Semantics annotations. In this format
objects carry no bounding boxes (instance ids live in a semantic texture), so
ground truth is read by rendering: for each verified object position P, a
semantic + depth camera is placed at up to 8 navigable points 1.0-2.2 m from P,
looking straight at P; the annotated instances in a small window around P's
pixel whose depth agrees with the distance to P (+-0.6 m) are collected, and the
category votes across views give the ground-truth label at P.
A success is ground-truth correct if the query noun (or a listed synonym) is
one of the categories found at P. Caption-audit labels are compared.
Out: results/post_paper/gt861_scoring.json"""
import json
import math
from collections import Counter
from pathlib import Path

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_two_vectors

CFG = "data/hm3d/example/hm3d_annotated_example_basis.scene_dataset_config.json"
SCENE = "00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb"
RUNS = ["phase4_batch_mq_gt861_rep1", "phase4_batch_memory_off_mq_gt861_rep1"]
SYN = {"pillow": {"pillow", "cushion"}, "bed": {"bed"}, "chair": {"chair", "armchair", "office chair", "dining chair"},
       "lamp": {"lamp", "table lamp", "floor lamp"}, "sink": {"sink"}, "mirror": {"mirror"}, "toilet": {"toilet"},
       "refrigerator": {"refrigerator", "fridge"}, "bicycle": {"bicycle"}, "piano": {"piano"}}
RES, HFOV = 256, 90.0
AUDIT = json.load(open("results/post_paper/postplan_audit_labels.json"))["labels"]


def make_sim():
    c = habitat_sim.SimulatorConfiguration(); c.scene_dataset_config_file = CFG; c.scene_id = SCENE
    c.use_semantic_textures = False  # verified 2026-09-29: True renders all-zero ids here; False renders instances
    specs = []
    for uuid, st in (("sem", habitat_sim.SensorType.SEMANTIC), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = st; s.resolution = [RES, RES]
        s.position = [0.0, 0.0, 0.0]; s.hfov = HFOV; specs.append(s)
    a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))


def gt_at(sim, P, cats):
    P = np.array(P, dtype=float)
    votes, views = Counter(), 0
    for k in range(8):
        ang = 2 * math.pi * k / 8
        for r in (1.4, 1.0, 2.2):
            cand = P + np.array([r * math.cos(ang), 0.0, r * math.sin(ang)])
            nav = np.array(sim.pathfinder.snap_point(cand))
            if not np.all(np.isfinite(nav)) or abs(nav[1] - P[1]) > 2.0:
                continue
            cam = nav + np.array([0.0, 1.5, 0.0])
            d = P - cam; dist = np.linalg.norm(d)
            if not 0.5 < dist < 3.5:
                continue
            st = habitat_sim.AgentState(); st.position = cam
            st.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), d / dist)
            sim.get_agent(0).set_state(st, reset_sensors=True, infer_sensor_states=False)
            obs = sim.get_sensor_observations()
            c0, w = RES // 2, 6
            sem = obs["sem"][c0 - w:c0 + w + 1, c0 - w:c0 + w + 1].ravel()
            dep = obs["depth"][c0 - w:c0 + w + 1, c0 - w:c0 + w + 1].ravel()
            ok = np.abs(dep * 1.0 - dist) < 0.6   # depth is along the optical axis ~ distance at centre
            ids = sem[ok]
            if len(ids) == 0:
                continue
            views += 1
            for iid, n in Counter(ids.tolist()).items():
                votes[cats.get(int(iid), "Unknown")] += n
            break
    return votes, views


def instance_centres(sim, n_points=250, step=4):
    """3D centre of every annotated instance, by back-projecting semantic+depth
    renders from many navigable viewpoints (instances carry no boxes here)."""
    rng = np.random.default_rng(0)
    acc = {}
    f = (RES / 2) / math.tan(math.radians(HFOV) / 2)
    vv, uu = np.mgrid[0:RES:step, 0:RES:step]
    for _ in range(n_points):
        nav = np.array(sim.pathfinder.get_random_navigable_point())
        for k in range(4):
            yaw = k * math.pi / 2
            st = habitat_sim.AgentState(); st.position = nav + np.array([0.0, 1.5, 0.0])
            st.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), np.array([math.sin(yaw), 0.0, -math.cos(yaw)]))
            sim.get_agent(0).set_state(st, reset_sensors=True, infer_sensor_states=False)
            o = sim.get_sensor_observations()
            dep = o["depth"][vv, uu]; ids = o["sem"][vv, uu]
            ok = (dep > 0.1) & (dep < 6)
            x = (uu[ok] - RES / 2) / f * dep[ok]; y = -(vv[ok] - RES / 2) / f * dep[ok]; z = -dep[ok]
            R = np.array(sim.get_agent(0).get_state().sensor_states["depth"].rotation.to_rotation_matrix()) \
                if hasattr(sim.get_agent(0).get_state().sensor_states["depth"].rotation, "to_rotation_matrix") else None
            q = sim.get_agent(0).get_state().sensor_states["depth"].rotation
            import quaternion  # noqa: F401  (numpy-quaternion, installed with habitat-sim)
            Rm = quaternion.as_rotation_matrix(q)
            world = (Rm @ np.vstack([x, y, z])).T + np.array(sim.get_agent(0).get_state().sensor_states["depth"].position)
            for iid, pw in zip(ids[ok].tolist(), world):
                a = acc.setdefault(int(iid), [np.zeros(3), 0]); a[0] += pw; a[1] += 1
    return {i: (s / n) for i, (s, n) in acc.items() if n >= 5}


def main():
    sim = make_sim()
    cats = {}
    for o in sim.semantic_scene.objects:
        if o is not None and o.category is not None:
            try:
                cats[int(o.id.split("_")[-1])] = o.category.name()
            except ValueError:
                pass
    present = Counter(cats.values())
    centres = instance_centres(sim)
    rows = []
    for run in RUNS:
        m = json.load(open(f"data/runs/{run}/batch_metrics.json"))
        for e in m["episodes"]:
            q = e["query"].replace("a ", "", 1)
            row = {"run": run, "episode": e["episode_id"], "query": q, "reported": e["outcome"] == "success",
                   "instances_in_scene": sum(present[c] for c in SYN.get(q, {q}))}
            v = e["verification_detail"] or {}
            if row["reported"]:
                votes, views = gt_at(sim, v["entity_position_world"], cats)
                top = [c for c, _ in votes.most_common(4) if c != "Unknown"]
                P = np.array(v["entity_position_world"])
                near = sorted(((float(np.linalg.norm((c - P)[[0, 2]])), cats.get(i, "?")) for i, c in centres.items()))
                qdist = min((d_ for d_, c in near if c in SYN.get(q, {q})), default=None)
                row.update({"nearest_query_instance_m": None if qdist is None else round(qdist, 2),
                            "gt_correct_within_2m": qdist is not None and qdist <= 2.0,
                            "nearest_instances": [(c, round(d_, 2)) for d_, c in near[:5]]})
                row.update({"caption": v["description"][:90], "gt_categories_at_target": top, "views": views,
                            "gt_correct": bool(set(top[:2]) & SYN.get(q, {q})),
                            "caption_audit": AUDIT.get(f"{run}/{e['episode_id']}")})
            rows.append(row)
    sim.close()
    succ = [r for r in rows if r["reported"]]
    summary = {"reported_successes": len(succ), "gt_correct_at_point": sum(r["gt_correct"] for r in succ),
               "gt_correct_within_2m": sum(r["gt_correct_within_2m"] for r in succ),
               "instances_located": len(centres),
               "agree_with_strict_audit": sum(r["gt_correct_within_2m"] == (r["caption_audit"] == "strict") for r in succ),
               "agree_with_lenient_audit": sum(r["gt_correct_within_2m"] == (r["caption_audit"] in ("strict", "lenient")) for r in succ),
               "absent_queries_falsely_verified": sum(r["reported"] for r in rows if r["instances_in_scene"] == 0)}
    Path("results/post_paper/gt861_scoring.json").write_text(json.dumps({"_doc": __doc__, "summary": summary, "rows": rows}, indent=1))
    for r in rows:
        print({k: r[k] for k in r if k not in ("run",)} , r["run"][13:])
    print(summary)


if __name__ == "__main__":
    main()
