"""Standalone (no cross-module import, to sidestep a reproducible segfault hit when
reusing run_spot_frontier_exploration.py's functions via import -- see PROGRESS.md/
conversation) teaser-image capture for Spot exploring an HM3D scene. Two real fixes
over the first teaser attempt, both visible problems in the first batch of extracted
frames:

1. Lighting: SpotRobot's meshesColored assets logged "incompatible light setup,
   instance will not be correctly lit" during construction (confirmed by reading
   the actual first-batch frames -- Spot rendered as a near-black silhouette while
   the room around it was fully lit/textured). Root cause: this HM3D scene's own
   dataset config specifies a "no_lights" layout (also seen in the same run's
   warnings), which the room's baked/vertex-shaded materials don't need but Spot's
   Phong-shaded meshes do. Fixed via SimulatorConfiguration.override_scene_light_
   defaults + scene_light_setup=DEFAULT_LIGHTING_KEY, a real habitat_sim config
   flag (verified present via `dir()` on the installed package), not a guess.

2. Camera framing: the built-in "third" camera (SpotRobot's own default chase-cam
   offset, cam_offset_pos=(-0.5,1.7,-0.5)) is tuned for interactive following at
   close range -- in the first extracted frames it mostly showed floor/walls with
   just Spot's rear legs at the very bottom edge. A custom "teaser" camera prefix
   is added instead, offset further back/up for a full-body three-quarter view.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import magnum as mn
import numpy as np
import yaml
from omegaconf import DictConfig
from PIL import Image

import habitat_sim
from habitat.articulated_agents.mobile_manipulator import ArticulatedAgentCameraParams
from habitat.articulated_agents.robots.spot_robot import SpotRobot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "frontier_agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "memory_layer"))
from frontier_agent.agent import FrontierAgent
from memory_layer.mobility import MobilityPriors
from memory_layer.relevance import ClipTextEncoder
from memory_layer.store import MemoryStore

ROBOT_URDF = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/spot_demo/teaser_frames_v2")
CAPTURE_STEPS = [60, 110, 160, 210]
MAX_STEPS = max(CAPTURE_STEPS) + 1


class TeaserSpotRobot(SpotRobot):
    def _get_spot_params(self):
        params = super()._get_spot_params()
        params.cameras["teaser"] = ArticulatedAgentCameraParams(
            attached_link_id=-1,
            cam_offset_pos=mn.Vector3(-2.0, 2.1, -2.0),
            cam_look_at_pos=mn.Vector3(0.6, 0.35, 0.6),
        )
        return params


def build_sim(scene_path: str, gpu_device_id: int = 0, resolution=(720, 1280)) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = gpu_device_id
    # Fix 1 (see module docstring): force real (Phong) lighting instead of this
    # scene's own "no_lights" layout, so Spot's meshes aren't flat black.
    backend_cfg.override_scene_light_defaults = True
    backend_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    def cam_spec(uuid, sensor_type, resolution, hfov):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = resolution
        spec.hfov = mn.Deg(hfov)
        return spec

    sensors = [cam_spec("teaser_rgb", habitat_sim.SensorType.COLOR, list(resolution), 90)]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    sim.initialize_agent(0)
    return sim


def spawn_spot(sim: habitat_sim.Simulator) -> TeaserSpotRobot:
    agent_config = DictConfig({"articulated_agent_urdf": ROBOT_URDF})
    spot = TeaserSpotRobot(agent_config, sim, fixed_base=True)
    spot.reconfigure()
    start = sim.pathfinder.get_random_navigable_point()
    spot.base_pos = mn.Vector3(*[float(x) for x in start])
    spot.base_rot = 0.0
    spot.leg_joint_pos = list(spot.params.leg_init_params)
    spot.update()
    return spot


def spot_pose(spot):
    pos = spot.base_pos
    position = [float(pos[0]), float(pos[1]), float(pos[2])]
    yaw = spot.base_rot
    q = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
    rotation_wxyz = [float(q.scalar), float(q.vector.x), float(q.vector.y), float(q.vector.z)]
    return position, rotation_wxyz


def apply_action(sim, spot, action):
    if action == "move_forward":
        yaw = spot.base_rot
        rot = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
        forward = rot.transform_vector(mn.Vector3(0, 0, -1))
        target = spot.base_pos + forward * 0.25
        spot.base_pos = sim.pathfinder.snap_point(target)
    elif action == "turn_left":
        spot.base_rot = spot.base_rot + np.radians(10.0)
    elif action == "turn_right":
        spot.base_rot = spot.base_rot - np.radians(10.0)
    spot.update()


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
            Image.fromarray(obs["teaser_rgb"][:, :, :3]).save(OUT_DIR / f"teaser_step{step:04d}.png")
            print(f"[step {step}] saved, branch={decision.branch} action={decision.action}")

    sim.close()
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
