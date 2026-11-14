"""Converts a goal position into a discrete Habitat action (move_forward /
turn_left / turn_right / stop), per CLAUDE.md Section 7's required
frontier_agent -> sim_bridge output: "next action in Habitat's action space".

The forward-direction convention (local -Z, rotated by the agent's quaternion) was
verified empirically against a real Habitat-Sim `move_forward` step, not assumed
from Habitat docs -- see PROGRESS.md Phase 4 for the verification.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import quaternion


def world_forward_xz(rotation_wxyz: List[float]) -> Tuple[float, float]:
    """World-frame forward direction (x, z) -- Habitat is Y-up, local forward is
    -Z, rotated by the agent's quaternion."""
    w, x, y, z = rotation_wxyz
    q = np.quaternion(w, x, y, z)
    R = quaternion.as_rotation_matrix(q)
    local_forward = np.array([0.0, 0.0, -1.0])
    world_forward = R @ local_forward
    return float(world_forward[0]), float(world_forward[2])


def bearing_to_goal_xz(agent_pos_xz: Tuple[float, float], goal_pos_xz: Tuple[float, float]) -> float:
    """Angle (radians) of the vector from agent to goal, in the same atan2(x,z)
    convention as world_forward_xz, so the two are directly comparable."""
    dx = goal_pos_xz[0] - agent_pos_xz[0]
    dz = goal_pos_xz[1] - agent_pos_xz[1]
    return math.atan2(dx, dz)


def signed_angle_diff(a: float, b: float) -> float:
    """a - b, wrapped to [-pi, pi]."""
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi


def choose_action(
    agent_position: List[float],
    agent_rotation_wxyz: List[float],
    goal_position: List[float],
    turn_tolerance_rad: float = math.radians(10.0),
    stop_distance: float = 0.3,
) -> str:
    agent_xz = (agent_position[0], agent_position[2])
    goal_xz = (goal_position[0], goal_position[2])

    distance = math.sqrt((agent_xz[0] - goal_xz[0]) ** 2 + (agent_xz[1] - goal_xz[1]) ** 2)
    if distance <= stop_distance:
        return "stop"

    forward_xz = world_forward_xz(agent_rotation_wxyz)
    forward_angle = math.atan2(forward_xz[0], forward_xz[1])
    bearing = bearing_to_goal_xz(agent_xz, goal_xz)

    diff = signed_angle_diff(bearing, forward_angle)
    if abs(diff) <= turn_tolerance_rad:
        return "move_forward"
    # empirically verified (see PROGRESS.md Phase 4): Habitat's turn_right action
    # DECREASES the atan2(x,z) forward angle, so a negative diff (goal's bearing is
    # below current forward angle) means turn_right closes the gap, not turn_left.
    return "turn_right" if diff < 0 else "turn_left"
