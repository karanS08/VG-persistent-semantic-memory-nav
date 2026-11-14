"""Reuses run_spot_frontier_exploration.py's setup (same real SpotRobot spawn +
FrontierAgent-driven exploration) to save a handful of head_rgb + third_person_rgb
PNG candidates at spaced-out steps, for picking a teaser image -- not another full
episode recording, just standalone frame grabs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spot_frontier_exploration import (  # noqa: E402
    apply_action,
    build_sim,
    spawn_spot,
    spot_pose,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "frontier_agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "memory_layer"))
from frontier_agent.agent import FrontierAgent  # noqa: E402
from memory_layer.mobility import MobilityPriors  # noqa: E402
from memory_layer.relevance import ClipTextEncoder  # noqa: E402
from memory_layer.store import MemoryStore  # noqa: E402

OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/spot_demo/teaser_frames")
CAPTURE_STEPS = [40, 70, 100, 130, 160, 190, 220]
MAX_STEPS = max(CAPTURE_STEPS) + 1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open("configs/habitat.yaml") as f:
        habitat_cfg = yaml.safe_load(f)
    with open("configs/memory.yaml") as f:
        mem_cfg = yaml.safe_load(f)
    with open("configs/run_defaults.yaml") as f:
        run_defaults = yaml.safe_load(f)

    sim = build_sim(habitat_cfg["scene_path"])
    spot = spawn_spot(sim)
    print("spawned at", spot.base_pos)

    mobility_priors = MobilityPriors.load("configs/mobility_priors.json")
    encoder = ClipTextEncoder(mem_cfg["embedding"]["clip_model_name"], mem_cfg["embedding"]["clip_model_dataset"])
    store = MemoryStore(
        initial_v=mem_cfg["reliability"]["initial_v"],
        verification_failure_discount=mem_cfg["reliability"]["verification_failure_discount"],
    )
    agent = FrontierAgent(
        pathfinder=sim.pathfinder, clip_encoder=encoder, mobility_priors=mobility_priors,
        alpha=mem_cfg["reliability"]["alpha"], lambda_base=mem_cfg["reliability"]["lambda_base"],
        theta_direct=mem_cfg["recall"]["theta_direct"],
        anchor_weight=mem_cfg["contextual_anchor"]["anchor_weight"],
        value_map_sigma=mem_cfg["contextual_anchor"]["value_map_sigma"],
        n_frontier_candidates=run_defaults["episode"]["n_frontier_candidates"],
        periodic_replan_interval=run_defaults["episode"]["periodic_replan_interval"],
    )

    capture_set = set(CAPTURE_STEPS)
    for step in range(MAX_STEPS):
        position, rotation_wxyz = spot_pose(spot)
        decision = agent.decide(
            agent_position=position, agent_rotation_wxyz=rotation_wxyz,
            query="explore", memory_store=store, live_entity_ids=set(), now=time.time(),
        )
        action = decision.action if decision.action != "stop" else "move_forward"
        apply_action(sim, spot, action)

        if step in capture_set:
            obs = sim.get_sensor_observations()
            Image.fromarray(obs["third_rgb"][:, :, :3]).save(OUT_DIR / f"third_person_step{step:04d}.png")
            Image.fromarray(obs["head_rgb"][:, :, :3]).save(OUT_DIR / f"head_step{step:04d}.png")
            print(f"[step {step}] saved, branch={decision.branch} action={decision.action}")

    sim.close()
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
