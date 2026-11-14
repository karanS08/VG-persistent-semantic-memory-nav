"""Controlled decision-flip test for temporal decay (2026-09-07).

Answers a specific question the earlier synthetic formula sweep
(controlled_temporal_decay_experiment.py, results/controlled_temporal_decay.md)
could not: does decay ever change which entity the REAL pipeline code picks,
not just whether C_i's number goes down?

Scenario, using this project's own real config (configs/mobility_priors.json,
configs/memory.yaml) and real production code (memory_layer.reliability.
compute_reliability, memory_layer.relevance.compute_relevance) -- not a
standalone reimplementation of the formula:

  Query: "a cushion" (cushion mobility=0.55, a real, fairly mobile category
  in this project's own priors -- not a cherry-picked extreme).

  Entity STALE: relevance 0.19 (the real pipeline's observed high end,
  Section 5.3/5.2), last_observed 6 hours ago -- e.g. the agent saw a
  cushion here on an earlier visit.
  Entity FRESH: relevance 0.15 (still inside the pipeline's real observed
  range, just lower), last_observed now -- e.g. a weaker but current
  sighting.

  Both otherwise identical (n_i=1, c_i=1.0, v_i=1.0 -- the real production
  defaults every other result in this paper uses).

Under decay ON (lambda_base=0.001/s, the real default): does the fresher,
weaker sighting outrank the older, stronger one, correctly discounting
6-hour-old evidence for a mobile category? Under decay OFF (lambda_base=0,
the no-decay ablation already used elsewhere in this project, e.g. Table 2's
row structure): does raw relevance alone decide, regardless of age?
"""
from __future__ import annotations

import json
import math

from memory_layer.mobility import MobilityPriors
from memory_layer.reliability import compute_reliability
from memory_layer.relevance import compute_relevance
from memory_layer.models import MemoryEntity

ALPHA = 0.5  # real configs/memory.yaml default
LAMBDA_BASE_DEFAULT = 0.001  # real configs/memory.yaml default
STALE_AGE_S = 6 * 3600  # 6 hours

mobility_priors = MobilityPriors.load("configs/mobility_priors.json")
cushion_mobility = mobility_priors.mobility_for("cushion")


def vec_at_cosine(cos_theta: float):
    """2D unit vector whose cosine similarity with [1,0] is exactly cos_theta --
    same technique packages/frontier_agent/tests/test_verification.py already
    uses for controlled relevance values, applied here to the real
    compute_relevance function rather than hand-computing R_i."""
    theta = math.acos(cos_theta)
    return [math.cos(theta), math.sin(theta)]


query_embedding = [1.0, 0.0]
stale = MemoryEntity(
    entity_id=1, category="cushion", description="a real cushion, seen 6h ago",
    embedding=vec_at_cosine(0.19), position_world=[0.0, 0.0, 0.0],
    c_i=1.0, n_i=1, v_i=1.0, last_observed=-STALE_AGE_S,
)
fresh = MemoryEntity(
    entity_id=2, category="cushion", description="a real cushion, seen just now",
    embedding=vec_at_cosine(0.15), position_world=[0.0, 0.0, 0.0],
    c_i=1.0, n_i=1, v_i=1.0, last_observed=0.0,
)

rel_stale = compute_relevance(query_embedding, stale.embedding)
rel_fresh = compute_relevance(query_embedding, fresh.embedding)

results = {"cushion_mobility": cushion_mobility, "relevance_stale": rel_stale,
           "relevance_fresh": rel_fresh, "regimes": {}}

for label, lambda_base in [("decay_on (production default)", LAMBDA_BASE_DEFAULT),
                            ("decay_off (lambda_base=0 ablation)", 0.0)]:
    c_stale = compute_reliability(stale, now=0.0, mobility=cushion_mobility,
                                   alpha=ALPHA, lambda_base=lambda_base)
    c_fresh = compute_reliability(fresh, now=0.0, mobility=cushion_mobility,
                                   alpha=ALPHA, lambda_base=lambda_base)
    r_stale = rel_stale * c_stale
    r_fresh = rel_fresh * c_fresh
    winner = "STALE (6h old)" if r_stale > r_fresh else "FRESH (just observed)"
    results["regimes"][label] = {
        "C_i_stale": c_stale, "C_i_fresh": c_fresh,
        "R_i_stale": r_stale, "R_i_fresh": r_fresh, "winner": winner,
    }
    print(f"\n=== {label} ===")
    print(f"  C_i(stale, 6h old)  = {c_stale:.6f}  ->  R_i = {r_stale:.6f}")
    print(f"  C_i(fresh, 0s old)  = {c_fresh:.6f}  ->  R_i = {r_fresh:.6f}")
    print(f"  Winner (higher R_i): {winner}")

with open("results/controlled_decay_decision_flip.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/controlled_decay_decision_flip.json")
