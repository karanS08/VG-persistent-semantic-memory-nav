"""Spawn habitat-lab's real SpotRobot (hab_spot_arm URDF, Bullet-physics legged
base) into an HM3D scene, drive it with this project's REAL FrontierAgent
(empty MemoryStore, no live entities -> every decide() call takes the
"fallback" branch, i.e. pure frontier exploration, no query target), and
stream head + third-person RGB-D plus the trajectory to Rerun, saved as an
.rrd for offline review.

Design decisions, logged here rather than silently:

- Physics enabled (Bullet) to load Spot as an articulated object. This is a
  NEW, separate entry point -- packages/sim_bridge/sim_bridge/habitat_env.py
  (bare AgentConfiguration, no physics, used by every existing paper result)
  is untouched.

- A custom "head" camera prefix is added to SpotRobot's own camera-param dict
  via a local subclass (not editing the installed habitat-lab source) because
  the robot's built-in cameras are either stereo/narrow (head_stereo_left/
  right: 212x120px, 58deg hfov, angled for terrain sensing, not a clean
  forward view) or arm-mounted. "head" is the documented convention itself --
  ArticulatedAgentCameraParams's own docstring: a "head" key matches sensors
  "head_rgb"/"head_depth" (see mobile_manipulator.py MobileManipulatorParams
  docstring, verified by reading the installed source).

- The built-in "third" camera prefix (cam_offset_pos=(-0.5,1.7,-0.5),
  look-at=(1,0,0.75), attached to the base) is used as-is for third-person
  view -- Meta's own chase-cam default for this exact robot, not reinvented.

- Navigation drives the REAL FrontierAgent.decide() (agent.py) rather than a
  reimplemented walk loop, reusing the real path-planning/stuck-detection
  logic. This needs a real (lightweight, text-only) CLIP encoder just to
  embed the placeholder query string -- with an empty memory store the query
  text itself is inert (decide() always falls through to fallback/
  exploration since there is nothing to recall), so any placeholder query
  works; "explore" is used for clarity in logs, not because its content
  matters here.

- Spot's base is moved kinematically (base_pos/base_rot setters) using the
  SAME action vocabulary and step sizes as configs/habitat.yaml
  (move_forward=0.25m, turn=10deg) that the point-agent pipeline already
  uses -- frontier_agent needed zero changes to drive this.

- Known simplification, flagged not assumed: the scene's navmesh is NOT
  recomputed for Spot's footprint (radius 0.25m per habitat-lab's spot.yaml)
  -- it's still the navmesh baked at configs/habitat.yaml's agent.radius
  (0.1m). Fine for a visual demo; would need recompute_navmesh with Spot's
  real radius before trusting collision-tightness for anything quantitative.

Usage:
    python3 scripts/run_spot_frontier_exploration.py \
        --max-steps 500 --rrd-out data/runs/spot_demo/spot_exploration.rrd
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import magnum as mn
import numpy as np
import rerun as rr
import yaml
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "frontier_agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "memory_layer"))

import habitat_sim  # noqa: E402
from habitat.articulated_agents.mobile_manipulator import ArticulatedAgentCameraParams  # noqa: E402
from habitat.articulated_agents.robots.spot_robot import SpotRobot  # noqa: E402

from frontier_agent.agent import FrontierAgent  # noqa: E402
from memory_layer.mobility import MobilityPriors  # noqa: E402
from memory_layer.relevance import ClipTextEncoder  # noqa: E402
from memory_layer.store import MemoryStore  # noqa: E402

ROBOT_URDF = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
MOVE_FORWARD_M = 0.25
TURN_DEG = 10.0


class HeadCamSpotRobot(SpotRobot):
    """SpotRobot + one added forward-facing head camera (see module docstring)."""

    def _get_spot_params(self):
        params = super()._get_spot_params()
        params.cameras["head"] = ArticulatedAgentCameraParams(
            attached_link_id=-1,
            cam_offset_pos=mn.Vector3(0.35, 0.55, 0.0),
            cam_look_at_pos=mn.Vector3(2.35, 0.4, 0.0),
        )
        return params


def build_sim(scene_path: str, gpu_device_id: int = 0) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = gpu_device_id

    def cam_spec(uuid, sensor_type, resolution, hfov):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = resolution
        spec.hfov = mn.Deg(hfov)
        # position/orientation left at defaults -- SpotRobot.update() overwrites
        # this sensor node's transform every call based on params.cameras (see
        # manipulator.py's update(), verified by reading the installed source).
        return spec

    sensors = [
        cam_spec("head_rgb", habitat_sim.SensorType.COLOR, [480, 640], 90),
        cam_spec("head_depth", habitat_sim.SensorType.DEPTH, [480, 640], 90),
        cam_spec("third_rgb", habitat_sim.SensorType.COLOR, [540, 960], 90),
    ]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    sim.initialize_agent(0)
    return sim


def spawn_spot(sim: habitat_sim.Simulator) -> HeadCamSpotRobot:
    agent_config = DictConfig({"articulated_agent_urdf": ROBOT_URDF})
    spot = HeadCamSpotRobot(agent_config, sim, fixed_base=True)
    spot.reconfigure()

    start = sim.pathfinder.get_random_navigable_point()
    spot.base_pos = mn.Vector3(*[float(x) for x in start])
    spot.base_rot = 0.0
    spot.leg_joint_pos = list(spot.params.leg_init_params)
    spot.update()
    return spot


def spot_pose(spot: HeadCamSpotRobot):
    pos = spot.base_pos
    position = [float(pos[0]), float(pos[1]), float(pos[2])]
    yaw = spot.base_rot
    q = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
    rotation_wxyz = [float(q.scalar), float(q.vector.x), float(q.vector.y), float(q.vector.z)]
    return position, rotation_wxyz


def apply_action(sim: habitat_sim.Simulator, spot: HeadCamSpotRobot, action: str) -> None:
    if action == "move_forward":
        yaw = spot.base_rot
        rot = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
        forward = rot.transform_vector(mn.Vector3(0, 0, -1))
        target = spot.base_pos + forward * MOVE_FORWARD_M
        spot.base_pos = sim.pathfinder.snap_point(target)
    elif action == "turn_left":
        spot.base_rot = spot.base_rot + np.radians(TURN_DEG)
    elif action == "turn_right":
        spot.base_rot = spot.base_rot - np.radians(TURN_DEG)
    # "stop"/anything else: no base movement this step.
    spot.update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", default=None, help="overrides configs/habitat.yaml's scene_path")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--rrd-out", default="data/runs/spot_demo/spot_frontier_exploration.rrd")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    args = parser.parse_args()

    with open("configs/habitat.yaml") as f:
        habitat_cfg = yaml.safe_load(f)
    scene_path = args.scene_path or habitat_cfg["scene_path"]

    with open("configs/memory.yaml") as f:
        mem_cfg = yaml.safe_load(f)
    with open("configs/run_defaults.yaml") as f:
        run_defaults = yaml.safe_load(f)

    Path(args.rrd_out).parent.mkdir(parents=True, exist_ok=True)

    print(f"scene: {scene_path}")
    sim = build_sim(scene_path, args.gpu_device_id)
    print("sim constructed, physics enabled:", sim.get_physics_simulation_library())

    spot = spawn_spot(sim)
    print("spot spawned at", spot.base_pos)

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

    rr.init("rpv_mem_spot_exploration", spawn=False)

    action_counts = {"move_forward": 0, "turn_left": 0, "turn_right": 0, "stop": 0}
    t0 = time.time()
    for step in range(args.max_steps):
        position, rotation_wxyz = spot_pose(spot)
        decision = agent.decide(
            agent_position=position, agent_rotation_wxyz=rotation_wxyz,
            query="explore", memory_store=store, live_entity_ids=set(), now=time.time(),
        )
        action = decision.action if decision.action != "stop" else "move_forward"
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        apply_action(sim, spot, action)

        obs = sim.get_sensor_observations()
        position, rotation_wxyz = spot_pose(spot)

        rr.set_time_sequence("step", step)
        rr.log(
            "world/spot",
            rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=[*rotation_wxyz[1:], rotation_wxyz[0]]),
            ),
        )
        rr.log("world/spot/head_rgb", rr.Image(obs["head_rgb"][:, :, :3]))
        rr.log("world/spot/head_depth", rr.DepthImage(obs["head_depth"]))
        rr.log("world/spot/third_person_rgb", rr.Image(obs["third_rgb"][:, :, :3]))
        rr.log("world/spot/position", rr.Points3D([position]))

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"[step {step}] branch={decision.branch} action={decision.action} "
                  f"pos={[round(p, 2) for p in position]} elapsed={elapsed:.1f}s")

    print("action counts:", action_counts)
    rr.save(args.rrd_out)
    print(f"saved recording to {args.rrd_out}")
    sim.close()


if __name__ == "__main__":
    main()
