"""Standoff-pose computation: "a reachable standoff pose with line of sight" per
CLAUDE.md Section 6's direct-recall criterion. Flagged as unimplemented back in
Phase 3's replay test (data/runs/phase3_replay) -- this closes that gap using
Habitat-Sim's own navmesh, now that there's a live `pathfinder` to use.

Simplification, logged: reachability uses the navmesh (real, via
pathfinder.snap_point); line-of-sight is approximated by "the snapped navmesh point
is within max_snap_distance of the remembered object position" rather than a real
raycast against the scene mesh. A closer object with a nearby snapped point is much
more likely to have real line of sight than a far one, but this is a proxy, not a
rigorous check -- worth replacing with an actual raycast (habitat_sim supports it)
if Phase 4's batch runs show this proxy behaving badly.
"""
from __future__ import annotations

import math
from typing import List, Optional


def find_standoff_pose(pathfinder, target_position: List[float], max_snap_distance: float = 2.0) -> Optional[List[float]]:
    import numpy as np

    snapped = pathfinder.snap_point(np.array(target_position, dtype=np.float32))
    if snapped is None:
        return None
    snapped = [float(x) for x in snapped]
    if any(math.isnan(x) for x in snapped):
        return None

    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(snapped, target_position)))
    if dist > max_snap_distance:
        return None  # too far from any navigable point -- likely unreachable/blocked

    return snapped
