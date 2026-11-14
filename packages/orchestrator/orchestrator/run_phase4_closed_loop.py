"""Phase 4 Step 4.2 verification: close the full loop --
sim -> DAAAM -> memory -> frontier -> action -> sim -- for real, on a live episode.
Logs the action sequence chosen and WHY (which branch) at every step, per the
spec's explicit requirement.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import yaml

from daaam_adapter.bridge_client import DaaamBridgeClient
from sim_bridge.habitat_env import HabitatEnv
from frontier_agent.agent import FrontierAgent
from memory_layer.mobility import MobilityPriors
from memory_layer.relevance import ClipTextEncoder
from memory_layer.store import MemoryStore, Observation

QUERY = "a soft cushion"


def camera_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def wxyz_to_xyzw_transform(position, rotation_wxyz) -> list:
    w, x, y, z = rotation_wxyz
    return [position[0], position[1], position[2], x, y, z, w]


def main() -> None:
    out_dir = Path("data/runs/phase4_closed_loop")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/habitat.yaml") as f:
        habitat_cfg = yaml.safe_load(f)
    with open("configs/memory.yaml") as f:
        mem_cfg = yaml.safe_load(f)

    client = DaaamBridgeClient()
    print("health:", client.health())
    print("episode started:", client.start_episode())

    env = HabitatEnv(habitat_cfg)
    frame = env.reset()

    store = MemoryStore(
        initial_v=mem_cfg["reliability"]["initial_v"],
        verification_failure_discount=mem_cfg["reliability"]["verification_failure_discount"],
    )
    mobility_priors = MobilityPriors.load("configs/mobility_priors.json")
    print(f"embedding CLIP text encoder ({mem_cfg['embedding']['clip_model_name']})...")
    encoder = ClipTextEncoder(mem_cfg["embedding"]["clip_model_name"], mem_cfg["embedding"]["clip_model_dataset"])

    agent = FrontierAgent(
        pathfinder=env.pathfinder,
        clip_encoder=encoder,
        mobility_priors=mobility_priors,
        alpha=mem_cfg["reliability"]["alpha"],
        lambda_base=mem_cfg["reliability"]["lambda_base"],
        theta_direct=mem_cfg["recall"]["theta_direct"],
        anchor_weight=mem_cfg["contextual_anchor"]["anchor_weight"],
        value_map_sigma=mem_cfg["contextual_anchor"]["value_map_sigma"],
    )

    intrinsics = camera_intrinsics(
        habitat_cfg["sensor"]["rgb_resolution"][1], habitat_cfg["sensor"]["rgb_resolution"][0], habitat_cfg["sensor"]["hfov"],
    )

    n_steps = 5
    step_log = []
    for i in range(n_steps):
        transform = wxyz_to_xyzw_transform(frame.pose["position"], frame.pose["rotation"])
        push_resp = client.push_frame(
            frame_id=frame.step_id, timestamp=frame.timestamp,
            rgb=frame.rgb[:, :, :3], depth=frame.depth,
            transform=transform, camera_intrinsics=intrinsics,
        )

        # incorporate any live DAAAM updates into memory before deciding
        updates = client.pull_updates()
        live_ids = set()
        if updates:
            positions = client.get_positions()
            observations = []
            for u in updates:
                sid = u["semantic_id"]
                pos_list = positions.get(str(sid))
                position_world = pos_list[-1]["position_world"] if pos_list else None
                observations.append(Observation(
                    entity_id=sid, category=u["semantic_label"], description=u["semantic_label"],
                    embedding=u.get("embedding"), confidence=1.0,  # flat c_i, per the Phase 3 decision
                    room_label=None,
                    # true last-observed time, not arrival time -- see PROGRESS.md 2026-08-24
                    timestamp=u.get("last_observed", u["timestamp"]),
                    first_observed=u.get("first_observed"),
                    position_world=position_world,
                ))
                live_ids.add(sid)
            store.update(observations, [])

        decision = agent.decide(
            agent_position=frame.pose["position"], agent_rotation_wxyz=frame.pose["rotation"],
            query=QUERY, memory_store=store, live_entity_ids=live_ids, now=frame.timestamp,
        )

        step_log.append({
            "step": i, "frame_id": frame.step_id, "num_tracks": push_resp.get("num_tracks"),
            "new_updates_this_step": len(updates), "memory_size": len(store.entities),
            "branch": decision.branch, "action": decision.action, "goal_position": decision.goal_position,
            "goal_entity_id": decision.goal_entity_id,
        })
        print(f"step {i}: branch={decision.branch:20s} action={decision.action:12s} "
              f"memory_size={len(store.entities):3d} new_updates={len(updates)}")

        frame = env.step(decision.action if decision.action != "stop" else "move_forward")

    env.close()
    stop_resp = client.stop_episode()
    print(f"episode stopped, final corrections: {len(stop_resp.get('corrections', []))}")

    with open(out_dir / "step_log.json", "w") as f:
        json.dump(step_log, f, indent=2)
    print(f"saved step log to {out_dir}/step_log.json")
    print(f"\nran {n_steps} closed-loop steps without crashing")


if __name__ == "__main__":
    main()
