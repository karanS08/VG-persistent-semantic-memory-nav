"""Per-category mobility priors, per CLAUDE.md Section 6. The table itself
(configs/mobility_priors.json) is the "single batch LLM prompt" output the spec
calls for -- see that file's _meta for how/why. This module loads it and matches
DAAAM's free-text descriptions (not a fixed category taxonomy) against it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class MobilityPriors:
    def __init__(self, categories: Dict[str, float], fallback: float = 0.5):
        self._categories = categories
        self._fallback = fallback

    @classmethod
    def load(cls, path: str = "configs/mobility_priors.json") -> "MobilityPriors":
        with open(path) as f:
            data = json.load(f)
        fallback = data.get("_meta", {}).get("unmatched_category_fallback", 0.5)
        return cls(data["categories"], fallback)

    def mobility_for(self, category_or_description: str) -> float:
        """DAAAM's grounding output is free-text (e.g. "A rectangular cushion with
        a smooth, light gray fabric cover"), not a fixed category label, so this
        does substring matching against the vocab rather than requiring an exact
        key -- a real limitation (ambiguous on multi-object descriptions, no
        matching at all on completely novel object types), logged as such rather
        than silently assumed to be robust. Falls back to the neutral prior
        (configs/mobility_priors.json's unmatched_category_fallback) when nothing
        matches.
        """
        text = category_or_description.lower()
        if text in self._categories:
            return self._categories[text]

        # longest-match-first substring scan, so "kitchen island" beats "island"-less
        # generic terms and multi-word categories win over shorter ones they contain
        for category in sorted(self._categories, key=len, reverse=True):
            if category in text:
                return self._categories[category]

        return self._fallback


def update_mobility_from_verification(
    priors: MobilityPriors, category: str, verification_succeeded: bool
) -> None:
    """Hook required by CLAUDE.md Section 6 ("leave a hook for later updating
    mobility per-category from observed verification-failure rates in this
    specific environment, even if that online update isn't implemented until a
    later phase"). Deliberately NOT implemented yet -- this is that stub, not a
    real online-update mechanism. A real implementation would need a running
    per-category failure-rate estimate and a principled way to blend it with the
    cold-start prior above, which is out of scope for Phase 3.
    """
    raise NotImplementedError(
        "Online mobility-prior updates from verification outcomes are not "
        "implemented -- this is the hook CLAUDE.md Section 6 asks to leave for a "
        "later phase, not a bug."
    )
