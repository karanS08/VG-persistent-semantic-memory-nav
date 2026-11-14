import pytest

from memory_layer.recall import evaluate_direct_recall, RecallOutcome


def test_below_threshold_not_relevant_enough():
    r = evaluate_direct_recall(entity_id=1, relevance=0.1, reliability=0.5, has_standoff_pose=True, theta_direct=0.2)
    assert r.r_i == pytest.approx(0.05)
    assert r.outcome == RecallOutcome.NOT_RELEVANT_ENOUGH
    assert not r.is_direct_recall


def test_above_threshold_but_no_standoff_pose_is_distinct_outcome():
    """The spec explicitly requires this NOT be conflated with low confidence --
    same r_i as a passing case, but distinguished purely by pose availability."""
    r = evaluate_direct_recall(entity_id=2, relevance=0.5, reliability=0.5, has_standoff_pose=False, theta_direct=0.2)
    assert r.r_i == pytest.approx(0.25)
    assert r.r_i >= 0.2  # would pass the threshold on its own
    assert r.outcome == RecallOutcome.NO_STANDOFF_POSE
    assert not r.is_direct_recall


def test_above_threshold_with_standoff_pose_is_direct_recall():
    r = evaluate_direct_recall(entity_id=3, relevance=0.5, reliability=0.5, has_standoff_pose=True, theta_direct=0.2)
    assert r.outcome == RecallOutcome.DIRECT_RECALL
    assert r.is_direct_recall


def test_exactly_at_threshold_passes():
    r = evaluate_direct_recall(entity_id=4, relevance=0.4, reliability=0.5, has_standoff_pose=True, theta_direct=0.2)
    assert r.r_i == pytest.approx(0.2)
    assert r.outcome == RecallOutcome.DIRECT_RECALL


def test_zero_relevance_or_reliability_fails_threshold():
    r1 = evaluate_direct_recall(entity_id=5, relevance=0.0, reliability=0.9, has_standoff_pose=True, theta_direct=0.2)
    r2 = evaluate_direct_recall(entity_id=6, relevance=0.9, reliability=0.0, has_standoff_pose=True, theta_direct=0.2)
    assert r1.outcome == RecallOutcome.NOT_RELEVANT_ENOUGH
    assert r2.outcome == RecallOutcome.NOT_RELEVANT_ENOUGH
