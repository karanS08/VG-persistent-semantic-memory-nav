"""Thin wrapper around habitat_sim.Simulator exposing the sim_bridge -> daaam_adapter
per-step contract from CLAUDE.md Section 7: {rgb, depth, pose, timestamp, episode_id,
step_id}.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import habitat_sim
import magnum as mn
import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    pose: Dict[str, Any]  # {position: [x,y,z], rotation: [w,x,y,z] quaternion}
    timestamp: float
    episode_id: str
    step_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rgb": self.rgb,
            "depth": self.depth,
            "pose": self.pose,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
        }


class HabitatEnv:
    """Scripted-agent Habitat-Sim wrapper. No RL/gym action-selection policy here —
    step() takes an explicit action name; callers (a random-walk script now, later
    frontier_agent) decide what to do.
    """

    def __init__(self, config: Dict[str, Any], episode_id: str = "episode_0"):
        self._config = config
        self.episode_id = episode_id
        self._step_id = 0
        self._sim = habitat_sim.Simulator(self._build_sim_config(config))
        self._agent = self._sim.initialize_agent(0)

    def _build_sim_config(self, config: Dict[str, Any]) -> habitat_sim.Configuration:
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = config["scene_path"]
        backend_cfg.gpu_device_id = config.get("gpu_device_id", 0)

        sensor_cfg = config["sensor"]
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = sensor_cfg["rgb_resolution"]
        rgb_spec.position = [0.0, sensor_cfg["height"], 0.0]
        rgb_spec.hfov = mn.Deg(sensor_cfg["hfov"])

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = sensor_cfg["depth_resolution"]
        depth_spec.position = [0.0, sensor_cfg["height"], 0.0]
        depth_spec.hfov = mn.Deg(sensor_cfg["hfov"])

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.radius = config["agent"]["radius"]
        agent_cfg.height = config["agent"]["height"]

        action_space = {}
        for name, params in config["action_space"].items():
            if name in ("move_forward",):
                action_space[name] = habitat_sim.agent.ActionSpec(
                    name, habitat_sim.agent.ActuationSpec(amount=params["amount"])
                )
            elif name in ("turn_left", "turn_right"):
                action_space[name] = habitat_sim.agent.ActionSpec(
                    name, habitat_sim.agent.ActuationSpec(amount=params["amount"])
                )
        agent_cfg.action_space = action_space

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def _current_frame(self) -> Frame:
        obs = self._sim.get_sensor_observations()
        state = self._agent.get_state()
        rot = state.rotation  # numpy-quaternion `quaternion.quaternion`, fields w/x/y/z
        pose = {
            "position": [float(x) for x in state.position],
            "rotation": [float(rot.w), float(rot.x), float(rot.y), float(rot.z)],
        }
        frame = Frame(
            rgb=obs["rgb"],
            depth=obs["depth"],
            pose=pose,
            timestamp=time.time(),
            episode_id=self.episode_id,
            step_id=self._step_id,
        )
        return frame

    def reset(self) -> Frame:
        self._step_id = 0
        self._sim.reset()
        return self._current_frame()

    def current_frame(self) -> Frame:
        """Read the agent's current pose/observations without moving or resetting
        it -- used by the persistent-memory-across-queries run mode, where a new
        query should resume from wherever the agent already is, not the scene's
        spawn pose."""
        return self._current_frame()

    def step(self, action: str) -> Frame:
        self._sim.step(action)
        self._step_id += 1
        return self._current_frame()

    def close(self) -> None:
        self._sim.close()

    @property
    def pathfinder(self):
        return self._sim.pathfinder
