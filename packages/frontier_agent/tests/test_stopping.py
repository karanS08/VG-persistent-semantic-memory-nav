import pytest

from frontier_agent.stopping import check_stopping, EpisodeOutcome


def test_episode_expected_to_succeed():
    """Target verified partway through, with budgets nowhere near exhausted."""
    result = check_stopping(
        target_verified=True,
        elapsed_time=30.0, time_budget=120.0,
        path_length=5.0, path_budget=50.0,
        coverage_ratio=0.2, coverage_threshold=0.9,
    )
    assert result.outcome == EpisodeOutcome.SUCCESS
    assert result.reason == "target_visually_verified"


def test_episode_expected_to_hit_budget_failure():
    """Target never verified, time budget exhausted -- must terminate FAILURE, not
    silently continue or claim success."""
    result = check_stopping(
        target_verified=False,
        elapsed_time=120.0, time_budget=120.0,  # exactly at budget
        path_length=5.0, path_budget=50.0,
        coverage_ratio=0.2, coverage_threshold=0.9,
        best_guess_position=[1.0, 2.0, 3.0],
    )
    assert result.outcome == EpisodeOutcome.FAILURE
    assert result.reason == "time_budget_exhausted"
    # best-guess position is logged as diagnostic but the outcome is still FAILURE,
    # never a disguised success
    assert result.best_guess_position == [1.0, 2.0, 3.0]


def test_continues_when_nothing_triggered():
    result = check_stopping(
        target_verified=False,
        elapsed_time=10.0, time_budget=120.0,
        path_length=2.0, path_budget=50.0,
        coverage_ratio=0.1, coverage_threshold=0.9,
    )
    assert result.outcome == EpisodeOutcome.CONTINUE


def test_path_budget_failure():
    result = check_stopping(
        target_verified=False,
        elapsed_time=10.0, time_budget=120.0,
        path_length=50.0, path_budget=50.0,
        coverage_ratio=0.1, coverage_threshold=0.9,
    )
    assert result.outcome == EpisodeOutcome.FAILURE
    assert result.reason == "path_budget_exhausted"


def test_coverage_threshold_failure():
    result = check_stopping(
        target_verified=False,
        elapsed_time=10.0, time_budget=120.0,
        path_length=2.0, path_budget=50.0,
        coverage_ratio=0.9, coverage_threshold=0.9,
    )
    assert result.outcome == EpisodeOutcome.FAILURE
    assert result.reason == "coverage_threshold_reached"


def test_success_wins_even_if_budget_also_exhausted_same_step():
    """Success is checked first and unconditionally -- 'at any point, including
    mid-movement' per the spec -- so it must win a same-step race against a
    budget failure, not be shadowed by it."""
    result = check_stopping(
        target_verified=True,
        elapsed_time=120.0, time_budget=120.0,  # budget also exhausted this step
        path_length=5.0, path_budget=50.0,
        coverage_ratio=0.2, coverage_threshold=0.9,
    )
    assert result.outcome == EpisodeOutcome.SUCCESS
