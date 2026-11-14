"""Real A* waypoint pathfinding via Habitat's own navmesh, used to actually DRIVE
navigation -- not just score it after the fact for SPL, which is all
run_phase4_batch.py used habitat_sim.ShortestPath for previously.

2026-08-24: replaced the "sample ~12 random nearby points, score them by a fused
Gaussian value map, walk toward whichever scores highest" navigation strategy
(frontier.py/scoring.py/value_map.py) with this: pick ONE target (the best-scoring
remembered/live entity, or an unexplored point when nothing scores), plan a real
navmesh-constrained path to it, and walk that path waypoint by waypoint. This
directly fixes two separate problems patched around before: bearing-only
choose_action couldn't tell it was walking into a wall (needed a stuck-detection
hack), and summing Gaussian pulls from many nearby remembered entities created a
"gravity well" back toward already-explored territory (needed a visited-position
mask). A real path respects walls by construction and only targets one place at a
time, so neither patch is needed for path-following itself (stuck detection is
kept as a safety net for navmesh-snapping edge cases; the visited mask is kept
only for fallback point selection, unrelated to entities).

Verified live against the real scene before relying on it (PROGRESS.md 2026-08-24):
a real 12.5m query returned an 8-waypoint path that correctly bent around
geometry, using the exact same habitat_sim.ShortestPath API already in
run_phase4_batch.py's SPL computation, plus a confirmed real `.points` attribute
(a list of `_magnum.Vector3`, indexable and float-convertible).
"""
from __future__ import annotations

from typing import List, Optional


def compute_path(pathfinder, start_xyz: List[float], goal_xyz: List[float]) -> Optional[List[List[float]]]:
    """Real navmesh-constrained shortest path from start to goal. Returns a list of
    waypoint [x, y, z] positions (start and goal included), or None if find_path
    fails or returns a degenerate (<2 point) result."""
    import habitat_sim

    sp = habitat_sim.ShortestPath()
    sp.requested_start = start_xyz
    sp.requested_end = goal_xyz
    found = pathfinder.find_path(sp)
    if not found or not sp.points or len(sp.points) < 2:
        return None
    return [[float(p[0]), float(p[1]), float(p[2])] for p in sp.points]
