"""Scores frontier candidates by the fused value map and picks the best one."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .frontier import FrontierCandidate
from .value_map import Signal, ValueMap


@dataclass
class ScoredCandidate:
    candidate: FrontierCandidate
    value: float


def score_candidates(
    candidates: List[FrontierCandidate],
    value_map: ValueMap,
    live_signals: List[Signal],
    remembered_anchors: List[Signal],
) -> List[ScoredCandidate]:
    scored = [
        ScoredCandidate(
            candidate=c,
            value=value_map.value_at(
                [c.position_xyz[0], c.position_xyz[2]],  # ground-plane X,Z (Habitat is Y-up)
                live_signals,
                remembered_anchors,
            ),
        )
        for c in candidates
    ]
    scored.sort(key=lambda s: s.value, reverse=True)
    return scored


def best_candidate(scored: List[ScoredCandidate]) -> Optional[ScoredCandidate]:
    return scored[0] if scored else None
