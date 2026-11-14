"""Direct-recall criterion, per CLAUDE.md Section 6:

    R_i(q) = s(q, d_i) * C_i >= theta_direct

Direct recall additionally requires a reachable standoff pose with line of sight --
a SEPARATE boolean condition, not folded into the threshold. The spec explicitly
requires logging "relevance*reliability passes but no standoff pose" as a distinct,
separately-tracked outcome, not conflated with "low confidence" -- that's the whole
reason RecallOutcome has 3 states instead of a single bool.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RecallOutcome(Enum):
    NOT_RELEVANT_ENOUGH = "not_relevant_enough"  # R_i(q) < theta_direct
    NO_STANDOFF_POSE = "no_standoff_pose"  # R_i(q) >= theta_direct but no reachable pose
    DIRECT_RECALL = "direct_recall"  # both conditions satisfied


@dataclass
class RecallResult:
    entity_id: int
    relevance: float
    reliability: float
    r_i: float  # relevance * reliability
    outcome: RecallOutcome

    @property
    def is_direct_recall(self) -> bool:
        return self.outcome == RecallOutcome.DIRECT_RECALL


def evaluate_direct_recall(
    entity_id: int,
    relevance: float,
    reliability: float,
    has_standoff_pose: bool,
    theta_direct: float,
) -> RecallResult:
    r_i = relevance * reliability

    if r_i < theta_direct:
        outcome = RecallOutcome.NOT_RELEVANT_ENOUGH
    elif not has_standoff_pose:
        outcome = RecallOutcome.NO_STANDOFF_POSE
    else:
        outcome = RecallOutcome.DIRECT_RECALL

    return RecallResult(
        entity_id=entity_id,
        relevance=relevance,
        reliability=reliability,
        r_i=r_i,
        outcome=outcome,
    )
