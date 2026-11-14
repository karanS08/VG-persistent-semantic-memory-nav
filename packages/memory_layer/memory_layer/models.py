"""Data model for a stored memory entity, per CLAUDE.md Section 6/7."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryEntity:
    entity_id: int
    category: str
    description: str
    embedding: Optional[List[float]] = None
    position_world: Optional[List[float]] = None
    room_label: Optional[str] = None

    c_i: float = 1.0  # quality of the original observation (flat constant per decision, see PROGRESS.md)
    n_i: int = 1  # number of independent confirming observations
    v_i: float = 1.0  # historical verification reliability, discounted (not deleted) on failed verification

    first_observed: float = 0.0
    last_observed: float = 0.0

    standoff_pose: Optional[List[float]] = None  # reachable pose with line-of-sight, if known

    def record_observation(self, timestamp: float) -> None:
        """A new confirming observation of an already-known entity."""
        self.n_i += 1
        self.last_observed = timestamp

    def apply_verification(self, success: bool, discount: float) -> None:
        """Verification outcome: success leaves v_i unchanged, failure discounts it
        (never deletes the entry) -- CLAUDE.md Section 6: "Failed verification
        discounts v_i; it does not delete the memory entry."
        """
        if not success:
            self.v_i *= discount
