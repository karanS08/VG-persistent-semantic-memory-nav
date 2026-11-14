import pytest

from frontier_agent.frontier import FrontierCandidate
from frontier_agent.scoring import score_candidates, best_candidate
from frontier_agent.value_map import Signal, ValueMap


def test_scoring_picks_candidate_nearest_strong_signal():
    vm = ValueMap(anchor_weight=1.0, sigma=1.0)
    candidates = [
        FrontierCandidate(position_xyz=[0.0, 0.0, 0.0]),
        FrontierCandidate(position_xyz=[10.0, 0.0, 10.0]),
        FrontierCandidate(position_xyz=[10.0, 0.0, 0.0]),
    ]
    # a strong live signal right at [10, _, 0] (candidate 2's ground-plane position)
    signal = Signal(position_xy=[10.0, 0.0], strength=1.0)

    scored = score_candidates(candidates, vm, live_signals=[signal], remembered_anchors=[])
    best = best_candidate(scored)

    assert best.candidate.position_xyz == [10.0, 0.0, 0.0]
    assert best.value == pytest.approx(1.0)  # falloff(0)=1
    # candidates ordered descending by value
    assert scored[0].value >= scored[1].value >= scored[2].value


def test_scoring_no_signals_gives_zero_value_to_all():
    vm = ValueMap(anchor_weight=1.0, sigma=1.0)
    candidates = [FrontierCandidate(position_xyz=[0.0, 0.0, 0.0]), FrontierCandidate(position_xyz=[5.0, 0.0, 5.0])]
    scored = score_candidates(candidates, vm, live_signals=[], remembered_anchors=[])
    assert all(s.value == 0.0 for s in scored)


def test_best_candidate_empty_list_returns_none():
    assert best_candidate([]) is None
