"""Contextual anchor search value map, per CLAUDE.md Section 6:

    "Contextual anchor search: fuse live detections with weighted remembered
    anchor locations into a single value map; frontier scoring prefers regions
    near high-weight anchors. Anchor weight and fusion formula are not yet fully
    specified in the source paper... implement a clearly-labeled placeholder...
    but do not treat the placeholder as final."

THIS FILE IS THAT PLACEHOLDER. The fusion formula (linear weighted sum of
Gaussian-falloff-weighted signal strengths from live detections and remembered
anchors) is a reasonable-looking choice, not derived from the paper -- treat any
downstream behavior that depends on its exact shape as provisional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Signal:
    """One contributor to the value map -- either a live detection this step or a
    remembered anchor from memory_layer. `strength` is relevance*reliability
    (r_i) in both cases; the only difference is where they came from and whether
    anchor_weight applies.
    """
    position_xy: List[float]
    strength: float


def gaussian_falloff(distance: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if distance == 0 else 0.0
    return math.exp(-(distance ** 2) / (2 * sigma ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class ValueMap:
    def __init__(self, anchor_weight: float, sigma: float):
        self.anchor_weight = anchor_weight
        self.sigma = sigma

    def value_at(
        self,
        point_xy: List[float],
        live_signals: List[Signal],
        remembered_anchors: List[Signal],
    ) -> float:
        live_value = sum(
            s.strength * gaussian_falloff(euclidean(point_xy, s.position_xy), self.sigma)
            for s in live_signals
        )
        anchor_value = sum(
            s.strength * gaussian_falloff(euclidean(point_xy, s.position_xy), self.sigma)
            for s in remembered_anchors
        )
        return live_value + self.anchor_weight * anchor_value
