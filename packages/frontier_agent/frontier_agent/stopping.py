"""Stopping condition, exactly per CLAUDE.md Section 6:

    Failure  <=>  (time/path budget exhausted) OR (coverage_ratio >= threshold)
    Success  <=>  target visually verified (at any point, including mid-movement)

Evaluated continuously, not just at frontier completion -- callers should call
check_stopping() every step, not just when a frontier goal is reached. An
unverified best-guess location may be logged as diagnostic output on failure but
must NEVER be counted as a success in metrics -- that's why StoppingResult has no
"best guess" success state, only SUCCESS/FAILURE/CONTINUE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EpisodeOutcome(Enum):
    CONTINUE = "continue"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class StoppingResult:
    outcome: EpisodeOutcome
    reason: str
    best_guess_position: Optional[list] = None  # diagnostic only, see module docstring


def check_stopping(
    target_verified: bool,
    elapsed_time: float,
    time_budget: float,
    path_length: float,
    path_budget: float,
    coverage_ratio: float,
    coverage_threshold: float,
    best_guess_position: Optional[list] = None,
) -> StoppingResult:
    # success check comes first and is unconditional -- "at any point, including
    # mid-movement" per the spec, so it must win even if a budget was also hit
    # this same step.
    if target_verified:
        return StoppingResult(EpisodeOutcome.SUCCESS, reason="target_visually_verified")

    if elapsed_time >= time_budget:
        return StoppingResult(
            EpisodeOutcome.FAILURE, reason="time_budget_exhausted", best_guess_position=best_guess_position
        )
    if path_length >= path_budget:
        return StoppingResult(
            EpisodeOutcome.FAILURE, reason="path_budget_exhausted", best_guess_position=best_guess_position
        )
    if coverage_ratio >= coverage_threshold:
        return StoppingResult(
            EpisodeOutcome.FAILURE, reason="coverage_threshold_reached", best_guess_position=best_guess_position
        )

    return StoppingResult(EpisodeOutcome.CONTINUE, reason="in_progress")
