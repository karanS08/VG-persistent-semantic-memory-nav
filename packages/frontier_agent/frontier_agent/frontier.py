"""Frontier candidate generation.

Simplification, logged explicitly: this is NOT a real occupancy-grid frontier
detector (the boundary between explored-free and unexplored space) like
vlfm/mapping/obstacle_map.py's FMM-based frontier extraction in rvp_semnav. Building
that is out of scope for this research prototype (CLAUDE.md's own instruction:
"prioritize correctness and instrumentation over polish"). Instead, candidates are
sampled directly from Habitat-Sim's own navmesh via the PathFinder API -- points
Habitat already knows are navigable, at increasing radii from the agent. This is a
legitimate proxy for "unexplored-ish nearby space" in a single-episode prototype,
not a claim of equivalence to real frontier exploration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class FrontierCandidate:
    position_xyz: List[float]


def sample_frontier_candidates(
    pathfinder,
    agent_position: List[float],
    n_candidates: int = 12,
    min_radius: float = 1.5,
    max_radius: float = 6.0,
) -> List[FrontierCandidate]:
    """Samples n_candidates navigable points near the agent via Habitat's own
    PathFinder.get_random_navigable_point_near -- real navmesh data, not invented
    coordinates, but not a real frontier boundary either (see module docstring).
    """
    import numpy as np

    candidates = []
    seen = set()
    attempts = 0
    max_attempts = n_candidates * 10
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        radius = min_radius + (max_radius - min_radius) * (attempts % n_candidates) / max(1, n_candidates - 1)
        point = pathfinder.get_random_navigable_point_near(
            circle_center=np.array(agent_position, dtype=np.float32),
            radius=radius,
        )
        if point is None or any(np.isnan(point)):
            continue
        key = tuple(round(float(x), 2) for x in point)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(FrontierCandidate(position_xyz=[float(x) for x in point]))

    return candidates


def coverage_ratio(visited_xy: List[List[float]], reference_candidates: List[FrontierCandidate], visited_radius: float) -> float:
    """Proxy for CLAUDE.md Section 6's stopping-condition `coverage_ratio`: fraction
    of `reference_candidates` (a fixed sample of navigable points, generated once at
    episode start) that fall within `visited_radius` of ANY point in the agent's
    trajectory so far. Same "not a real occupancy-grid coverage metric" caveat as
    sample_frontier_candidates -- a lightweight, explainable stand-in appropriate for
    a research prototype, not a claim of metric equivalence to a real coverage map.
    """
    if not reference_candidates:
        return 0.0
    from .value_map import euclidean

    covered = 0
    for c in reference_candidates:
        c_xy = [c.position_xyz[0], c.position_xyz[2]]  # Habitat is Y-up; use X,Z as the ground plane
        if any(euclidean(c_xy, v) <= visited_radius for v in visited_xy):
            covered += 1
    return covered / len(reference_candidates)
