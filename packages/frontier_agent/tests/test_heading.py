import math

import pytest

from frontier_agent.heading import (
    world_forward_xz,
    bearing_to_goal_xz,
    signed_angle_diff,
    choose_action,
)


def test_world_forward_xz_matches_real_habitat_measurement():
    """These exact numbers came from a real Habitat-Sim move_forward step (see
    PROGRESS.md Phase 4): initial rotation [0.9158252477645874, 0.0,
    0.40157702565193176, 0.0], and the resulting move_forward delta normalized to
    [-0.73554961, 0., -0.67747087]. Not synthetic -- an empirical regression check."""
    rotation = [0.9158252477645874, 0.0, 0.40157702565193176, 0.0]
    fx, fz = world_forward_xz(rotation)
    assert fx == pytest.approx(-0.73554876, abs=1e-4)
    assert fz == pytest.approx(-0.67747178, abs=1e-4)


def test_world_forward_xz_identity_rotation_faces_negative_z():
    fx, fz = world_forward_xz([1.0, 0.0, 0.0, 0.0])
    assert fx == pytest.approx(0.0, abs=1e-9)
    assert fz == pytest.approx(-1.0, abs=1e-9)


def test_signed_angle_diff_wraps_correctly():
    # 350 degrees - 0 degrees should wrap to -10 degrees, not +350
    a = math.radians(350)
    b = math.radians(0)
    assert signed_angle_diff(a, b) == pytest.approx(math.radians(-10), abs=1e-6)


def test_signed_angle_diff_simple_case():
    assert signed_angle_diff(math.radians(30), math.radians(10)) == pytest.approx(math.radians(20))


def test_choose_action_stop_when_close():
    action = choose_action(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        goal_position=[0.1, 0.0, 0.1], stop_distance=0.3,
    )
    assert action == "stop"


def test_choose_action_move_forward_when_goal_directly_ahead():
    # identity rotation faces -Z; goal directly at -Z should be move_forward
    action = choose_action(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        goal_position=[0.0, 0.0, -5.0],
    )
    assert action == "move_forward"


def test_choose_action_turns_when_goal_behind():
    # identity rotation faces -Z; goal at +Z (behind) must turn, not move_forward
    action = choose_action(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        goal_position=[0.0, 0.0, 5.0],
    )
    assert action in ("turn_left", "turn_right")


def test_choose_action_turn_direction_is_consistent():
    """Goal to the agent's right (+X, facing -Z) should turn right; goal to the
    left (-X) should turn left."""
    right_action = choose_action(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        goal_position=[5.0, 0.0, -0.5],
    )
    left_action = choose_action(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        goal_position=[-5.0, 0.0, -0.5],
    )
    assert right_action == "turn_right"
    assert left_action == "turn_left"
