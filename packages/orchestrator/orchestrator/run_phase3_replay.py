"""Phase 3 Step 3.3/3.4 verification: replay Phase 2's real logged DAAAM detections
(data/runs/phase2_smoke/final_corrections.json, 33 real DAM-3B-grounded objects with
real CLIP embeddings) through memory_layer, offline -- no live sim/DAAAM needed.

Known scoping boundary (not a bug): standoff-pose computation (navmesh-based
reachability + line-of-sight) isn't implemented anywhere in this codebase yet --
that's Phase 4's frontier/navigation domain. So `has_standoff_pose` is False for
every replayed entity here, meaning this replay can only exercise the
NOT_RELEVANT_ENOUGH and NO_STANDOFF_POSE recall outcomes, not DIRECT_RECALL.
DIRECT_RECALL's branch logic is separately verified in
packages/memory_layer/tests/test_recall.py's synthetic cases.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from memory_layer.mobility import MobilityPriors
from memory_layer.reliability import compute_reliability
from memory_layer.relevance import ClipTextEncoder, compute_relevance
from memory_layer.recall import evaluate_direct_recall
from memory_layer.store import MemoryStore, Observation

QUERY = "a soft cushion"


def main() -> None:
    detections_path = Path("data/runs/phase2_smoke/final_corrections.json")
    out_dir = Path("data/runs/phase3_replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(detections_path) as f:
        detections = json.load(f)
    print(f"loaded {len(detections)} real detections from {detections_path}")

    with open("configs/memory.yaml") as f:
        mem_cfg = yaml.safe_load(f)

    mobility_priors = MobilityPriors.load("configs/mobility_priors.json")

    # before-state snapshot for the required before/after diff
    store = MemoryStore(
        initial_v=mem_cfg["reliability"]["initial_v"],
        verification_failure_discount=mem_cfg["reliability"]["verification_failure_discount"],
    )
    before_state = dict(store.entities)

    # c_i = flat constant 1.0, per the decision logged in PROGRESS.md -- DAAAM's own
    # `confidence` field is always 0.0 (dam_grounding.py never sets it), which would
    # zero out C_i for everything if used directly.
    C_I_FLAT_CONSTANT = 1.0
    observations = [
        Observation(
            entity_id=d["semantic_id"],
            category=d["semantic_label"],  # DAAAM gives free text, not a fixed category -- see mobility.py's substring matching
            description=d["semantic_label"],
            embedding=d.get("embedding"),
            confidence=C_I_FLAT_CONSTANT,
            room_label=None,
            # last_observed (true sighting time) if present, else fall back to
            # arrival time -- see PROGRESS.md 2026-08-24; this file's saved JSON
            # predates the bridge server exposing first_observed/last_observed,
            # so the fallback is what actually applies here today.
            timestamp=d.get("last_observed", d["timestamp"]),
            first_observed=d.get("first_observed"),
        )
        for d in detections
    ]
    store.update(observations, [])

    diff = store.diff_summary(before_state)
    print(f"memory update diff: {diff}")

    # simulate one failed verification and one successful one on real entities,
    # to exercise M_{t+1}=Update(M_t,O_t,V_t)'s verification path on top of the
    # observation path, per Step 3.4's requirement
    from memory_layer.store import Verification
    ids = list(store.entities.keys())
    before_v = {eid: store.entities[eid].v_i for eid in ids[:2]}
    store.update([], [Verification(entity_id=ids[0], success=False), Verification(entity_id=ids[1], success=True)])
    print(f"verification test: entity {ids[0]} v_i {before_v[ids[0]]} -> {store.entities[ids[0]].v_i} (failed verification)")
    print(f"verification test: entity {ids[1]} v_i {before_v[ids[1]]} -> {store.entities[ids[1]].v_i} (successful verification)")

    print(f"\nembedding the query '{QUERY}' via {mem_cfg['embedding']['clip_model_name']}/{mem_cfg['embedding']['clip_model_dataset']}...")
    encoder = ClipTextEncoder(mem_cfg["embedding"]["clip_model_name"], mem_cfg["embedding"]["clip_model_dataset"])
    query_embedding = encoder.embed(QUERY).tolist()

    now = max(o.timestamp for o in observations) + 5.0  # "now" = shortly after the episode ended

    results = []
    for entity in store.entities.values():
        mobility = mobility_priors.mobility_for(entity.category)
        reliability = compute_reliability(
            entity, now, mobility,
            alpha=mem_cfg["reliability"]["alpha"],
            lambda_base=mem_cfg["reliability"]["lambda_base"],
        )
        relevance = compute_relevance(query_embedding, entity.embedding)
        recall = evaluate_direct_recall(
            entity.entity_id, relevance, reliability,
            has_standoff_pose=False,  # see module docstring
            theta_direct=mem_cfg["recall"]["theta_direct"],
        )
        results.append({
            "entity_id": entity.entity_id,
            "description": entity.description[:100],
            "mobility": round(mobility, 3),
            "reliability_C_i": round(reliability, 4),
            "relevance_s": round(relevance, 4),
            "r_i": round(recall.r_i, 4),
            "outcome": recall.outcome.value,
        })

    results.sort(key=lambda r: r["r_i"], reverse=True)

    print(f"\ntop 10 by r_i for query '{QUERY}':")
    for r in results[:10]:
        print(f"  r_i={r['r_i']:.4f} relevance={r['relevance_s']:.4f} C_i={r['reliability_C_i']:.4f} mobility={r['mobility']:.2f} outcome={r['outcome']:>20} | {r['description']}")

    with open(out_dir / "replay_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "memory_diff.json", "w") as f:
        json.dump(diff, f, indent=2)

    print(f"\nsaved {len(results)} scored results to {out_dir}/replay_results.json")


if __name__ == "__main__":
    main()
