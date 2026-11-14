"""MemoryStore: M_{t+1} = Update(M_t, O_t, V_t), per CLAUDE.md Section 6. Runs on
every episode outcome, not just success. O_t = newly observed info (including
incidental objects unrelated to the query target); V_t = verification outcomes
(positive or negative evidence). Failed verification discounts v_i, never deletes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import MemoryEntity


@dataclass
class Observation:
    """One item of O_t -- a single detection from daaam_adapter, per the Section 7
    contract, whether or not it's the current query's target (incidental objects
    are explicitly meant to be incorporated too)."""
    entity_id: int  # DAAAM's semantic_id -- already a persistent per-episode ID
    category: str
    description: str
    embedding: Optional[List[float]]
    confidence: float  # c_i (flat 1.0 per the decision logged in PROGRESS.md)
    room_label: Optional[str]
    # The TRUE last-observed time (when the object was actually last seen), NOT
    # when this Observation was constructed/arrived. Callers bridging from DAAAM
    # must pass DAAAM's own temporal.last_observed here, not the SemanticUpdate's
    # arrival-time timestamp -- see PROGRESS.md's 2026-08-24 entry on this exact
    # bug: using arrival time made every freshly-grounded (but possibly long-since
    # unseen) entity look like it was observed "just now" for reliability-decay
    # purposes, silently understating staleness whenever grounding is delayed.
    timestamp: float
    # The TRUE first-observed time, if known (DAAAM's temporal.first_observed).
    # Falls back to `timestamp` if not provided -- only affects a brand-new
    # entity's first_observed field, not the decay-relevant last_observed.
    first_observed: Optional[float] = None
    position_world: Optional[List[float]] = None
    standoff_pose: Optional[List[float]] = None


@dataclass
class Verification:
    """One item of V_t."""
    entity_id: int
    success: bool


class MemoryStore:
    def __init__(self, initial_v: float, verification_failure_discount: float):
        self.entities: Dict[int, MemoryEntity] = {}
        self._initial_v = initial_v
        self._verification_failure_discount = verification_failure_discount

    def update(self, observations: List[Observation], verifications: List[Verification]) -> None:
        """M_{t+1} = Update(M_t, O_t, V_t). Order matters: new/confirming
        observations are folded in first, then verification outcomes are applied on
        top -- so a verification in the same update as a fresh observation still
        discounts v_i correctly rather than being overwritten by the observation step.
        """
        for obs in observations:
            self._incorporate_observation(obs)

        for v in verifications:
            entity = self.entities.get(v.entity_id)
            if entity is not None:
                entity.apply_verification(v.success, self._verification_failure_discount)

    def _incorporate_observation(self, obs: Observation) -> None:
        existing = self.entities.get(obs.entity_id)
        if existing is None:
            self.entities[obs.entity_id] = MemoryEntity(
                entity_id=obs.entity_id,
                category=obs.category,
                description=obs.description,
                embedding=obs.embedding,
                position_world=obs.position_world,
                room_label=obs.room_label,
                c_i=obs.confidence,
                n_i=1,
                v_i=self._initial_v,
                first_observed=obs.first_observed if obs.first_observed is not None else obs.timestamp,
                last_observed=obs.timestamp,
                standoff_pose=obs.standoff_pose,
            )
        else:
            existing.record_observation(obs.timestamp)
            # a fresh observation can update description/embedding/position (the
            # detection may be a better view than the original), but never resets
            # n_i or v_i -- those are cumulative across the entity's whole history
            if obs.description:
                existing.description = obs.description
            if obs.embedding is not None:
                existing.embedding = obs.embedding
            if obs.position_world is not None:
                existing.position_world = obs.position_world
            if obs.standoff_pose is not None:
                existing.standoff_pose = obs.standoff_pose

    def backfill_positions(self, positions_by_id: Dict[str, List[Dict[str, Any]]]) -> int:
        """Fills in position_world for already-known entities that don't have one
        yet. Needed because an entity's position (DAAAM's object_3d_positions) can
        become available AFTER its one grounding event already streamed through
        _incorporate_observation with position_world=None -- grounding fires once
        per track (see PROGRESS.md Phase 4), but depth-valid position observations
        keep accumulating independently and can lag behind it. Without this, an
        entity can be stuck with position_world=None forever even though a real
        position exists moments later, silently excluding it from the frontier
        agent's value map. `positions_by_id` keys are string semantic_ids, matching
        the daaam_bridge_server /positions response shape.
        """
        backfilled = 0
        for entity in self.entities.values():
            if entity.position_world is not None:
                continue
            obs_list = positions_by_id.get(str(entity.entity_id))
            if obs_list:
                entity.position_world = obs_list[-1]["position_world"]
                backfilled += 1
        return backfilled

    def fuse_positions(self, positions_by_id: Dict[str, List[Dict[str, Any]]], id_offset: int = 0) -> int:
        """pipeline-v8 (--fuse-positions): set each live entity's position_world to the component-wise
        MEDIAN of DAAAM's whole per-frame position history for that track (mask centroid + median depth per
        frame), refreshed every call. The default path keeps the single latest point at grounding time
        (`pos_list[-1]`), so one oblique or edge-of-mask view fixes the stored position; v7's THOR runs had
        5/11 false successes whose entry sat 1.6-1.9 m from the sofa the robot was standing next to.
        Entities not tracked in the current DAAAM session (e.g. remembered from earlier sessions) are left
        untouched. Returns the number of entities updated."""
        updated = 0
        for key, obs_list in positions_by_id.items():
            entity = self.entities.get(int(key) + id_offset)
            if entity is None or not obs_list:
                continue
            pts = np.asarray([o["position_world"] for o in obs_list], dtype=float)
            entity.position_world = [float(x) for x in np.median(pts, axis=0)]
            updated += 1
        return updated

    def diff_summary(self, before: Dict[int, MemoryEntity]) -> Dict[str, Any]:
        """Before/after diff for the Phase 3 verification gate's required
        before/after memory-state log."""
        added = [eid for eid in self.entities if eid not in before]
        updated = [
            eid for eid in self.entities
            if eid in before and (
                self.entities[eid].n_i != before[eid].n_i
                or self.entities[eid].v_i != before[eid].v_i
            )
        ]
        return {"added": added, "updated": updated, "total_entities": len(self.entities)}
