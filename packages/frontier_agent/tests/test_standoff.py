import math

import numpy as np
import pytest

from frontier_agent.standoff import find_standoff_pose


class FakePathfinder:
    """Duck-typed stand-in for habitat_sim's PathFinder, isolating the unit test
    from needing a real simulator."""
    def __init__(self, snap_result):
        self._snap_result = snap_result

    def snap_point(self, point):
        return self._snap_result


def test_returns_snapped_point_when_close_enough():
    pf = FakePathfinder(snap_result=np.array([1.1, 0.0, 2.0]))
    result = find_standoff_pose(pf, target_position=[1.0, 0.0, 2.0], max_snap_distance=2.0)
    assert result == pytest.approx([1.1, 0.0, 2.0])


def test_returns_none_when_too_far():
    pf = FakePathfinder(snap_result=np.array([10.0, 0.0, 10.0]))
    result = find_standoff_pose(pf, target_position=[0.0, 0.0, 0.0], max_snap_distance=2.0)
    assert result is None


def test_returns_none_on_nan_snap():
    pf = FakePathfinder(snap_result=np.array([float("nan"), float("nan"), float("nan")]))
    result = find_standoff_pose(pf, target_position=[0.0, 0.0, 0.0], max_snap_distance=2.0)
    assert result is None


def test_returns_none_when_pathfinder_gives_none():
    pf = FakePathfinder(snap_result=None)
    result = find_standoff_pose(pf, target_position=[0.0, 0.0, 0.0], max_snap_distance=2.0)
    assert result is None
