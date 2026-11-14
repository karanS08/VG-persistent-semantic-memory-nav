"""Revisit evidence (pipeline-v8, --revisit-evidence): a remembered entry whose location the robot looks at and finds
EMPTY has its v_i discounted -- the paper's stated rule ("reliability drops when the robot revisits the location and
does not find the object"), which the submitted pipeline applied only on a failed verification check.

"Empty" is decided from the depth image, as free-space evidence in an occupancy grid, not from DAAAM's captions:
DAAAM captions only a fraction of what it sees, so "no matching caption" is not evidence of absence (a first version
that used captions discounted 71 entries in one episode, mostly windows, doors and paintings that had not moved).
Per step, for each REMEMBERED entry (last observed before the current query started, has a position) within
0.5-3 m and inside the image, `geometry.depth_probe` compares the measured depth along the ray through the remembered
position with the predicted depth:
  * measured > predicted + see_through   -> the ray passes through the spot: inspected-empty (pending);
  * measured < predicted - occlusion_tol -> hidden behind something: no information;
  * otherwise                            -> something occupies the spot: pending cleared.
A pending entry is discounted ONCE per query, multiplicatively (never deleted, never zero), if it is still pending
after `grace_s` and no entry of the same kind (`same_kind`) observed during this query lies within `assoc_radius` of
it -- the second condition covers a re-detected object whose remembered point is slightly off its surface.
Nothing is ever written back as a re-observation, so live-only verification semantics are unchanged.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Tuple

from .geometry import depth_probe
from .models import MemoryEntity


class RevisitEvidence:
    def __init__(self, discount: float, same_kind: Callable[[MemoryEntity, MemoryEntity], bool],
                 assoc_radius: float = 1.5, grace_s: float = 10.0, min_range: float = 0.5, max_range: float = 3.0,
                 see_through: float = 0.5, occlusion_tol: float = 0.4):
        self.discount = discount
        self.same_kind = same_kind
        self.assoc_radius = assoc_radius
        self.grace_s = grace_s
        self.min_range = min_range
        self.max_range = max_range
        self.see_through = see_through
        self.occlusion_tol = occlusion_tol
        self.begin_query(0.0)

    def begin_query(self, query_start: float) -> None:
        self.query_start = query_start
        self.pending: Dict[int, float] = {}     # entity id -> time first seen empty
        self.discounted: Dict[int, float] = {}  # entity id -> time discounted (once per query)
        self.n_inspections = 0
        self.n_empty_looks = 0
        self.n_occupied_looks = 0

    def _near(self, a: List[float], b: List[float]) -> bool:
        return math.hypot(a[0] - b[0], a[2] - b[2]) <= self.assoc_radius

    def step(self, entities: Dict[int, MemoryEntity], now: float, pose: Dict, depth, intrinsics: Dict,
             cam_height: float, track_positions: Iterable[Tuple[int, List[float]]] = ()) -> List[int]:
        """Returns the ids discounted at this step. `track_positions` is accepted for interface stability (unused:
        occupancy now comes from depth, not from DAAAM tracks)."""
        for e in entities.values():
            if (e.last_observed >= self.query_start or e.position_world is None
                    or e.entity_id in self.discounted):
                continue
            probe = depth_probe(e.position_world, pose, depth, intrinsics, cam_height,
                                min_range=self.min_range, max_range=self.max_range)
            if probe is None:
                continue
            pred, measured = probe
            if measured < pred - self.occlusion_tol:
                continue  # hidden
            self.n_inspections += 1
            if measured > pred + self.see_through:
                self.n_empty_looks += 1
                self.pending.setdefault(e.entity_id, now)
            else:
                self.n_occupied_looks += 1
                self.pending.pop(e.entity_id, None)
        live = [e for e in entities.values() if e.last_observed >= self.query_start and e.position_world is not None]
        out = []
        for eid, t0 in list(self.pending.items()):
            e = entities.get(eid)
            if e is None:
                del self.pending[eid]
                continue
            if any(self._near(l.position_world, e.position_world) and self.same_kind(e, l) for l in live if l is not e):
                del self.pending[eid]  # re-detected nearby: supported, no discount
                continue
            if now - t0 >= self.grace_s:
                e.apply_verification(False, self.discount)
                self.discounted[eid] = now
                del self.pending[eid]
                out.append(eid)
        return out
