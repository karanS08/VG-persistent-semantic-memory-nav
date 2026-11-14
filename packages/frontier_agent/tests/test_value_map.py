import math

import pytest

from frontier_agent.value_map import Signal, ValueMap, gaussian_falloff, euclidean


def test_gaussian_falloff_at_zero_distance_is_one():
    assert gaussian_falloff(0.0, sigma=1.0) == pytest.approx(1.0)


def test_gaussian_falloff_hand_computed():
    # exp(-(2^2)/(2*1^2)) = exp(-2) = 0.1353352...
    assert gaussian_falloff(2.0, sigma=1.0) == pytest.approx(math.exp(-2.0))


def test_euclidean_hand_computed():
    assert euclidean([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)


def test_value_map_single_live_signal_at_query_point():
    """A single live signal exactly at the query point, no anchors: value should
    equal its raw strength (falloff(0)=1)."""
    vm = ValueMap(anchor_weight=1.0, sigma=2.0)
    signal = Signal(position_xy=[5.0, 5.0], strength=0.8)
    value = vm.value_at([5.0, 5.0], live_signals=[signal], remembered_anchors=[])
    assert value == pytest.approx(0.8)


def test_value_map_hand_computed_fusion():
    """Hand-computed: one live signal, one remembered anchor, at different
    distances from the query point, with anchor_weight != 1 to verify it's
    actually applied (not silently ignored)."""
    vm = ValueMap(anchor_weight=0.5, sigma=2.0)
    query = [0.0, 0.0]
    live = Signal(position_xy=[1.0, 0.0], strength=1.0)  # distance 1 from query
    anchor = Signal(position_xy=[4.0, 0.0], strength=1.0)  # distance 4 from query

    expected_live = 1.0 * gaussian_falloff(euclidean(query, [1.0, 0.0]), sigma=2.0)
    expected_anchor = 0.5 * (1.0 * gaussian_falloff(euclidean(query, [4.0, 0.0]), sigma=2.0))
    expected = expected_live + expected_anchor

    value = vm.value_at(query, live_signals=[live], remembered_anchors=[anchor])
    assert value == pytest.approx(expected, rel=1e-9)
    # sanity: the live signal (distance 1) should dominate over the anchor
    # (distance 4, plus halved by anchor_weight) given the same sigma/strength
    assert expected_live > expected_anchor


def test_value_map_anchor_weight_zero_ignores_anchors():
    vm = ValueMap(anchor_weight=0.0, sigma=1.0)
    anchor = Signal(position_xy=[0.0, 0.0], strength=100.0)  # huge strength, right at query
    value = vm.value_at([0.0, 0.0], live_signals=[], remembered_anchors=[anchor])
    assert value == 0.0


def test_value_map_far_signal_contributes_near_zero():
    vm = ValueMap(anchor_weight=1.0, sigma=1.0)
    far_signal = Signal(position_xy=[1000.0, 1000.0], strength=1.0)
    value = vm.value_at([0.0, 0.0], live_signals=[far_signal], remembered_anchors=[])
    assert value == pytest.approx(0.0, abs=1e-9)


def test_value_map_multiple_signals_sum():
    """No falloff-related cross terms -- multiple signals at the query point
    should sum linearly."""
    vm = ValueMap(anchor_weight=1.0, sigma=1.0)
    signals = [Signal(position_xy=[0.0, 0.0], strength=0.3) for _ in range(3)]
    value = vm.value_at([0.0, 0.0], live_signals=signals, remembered_anchors=[])
    assert value == pytest.approx(0.9)
