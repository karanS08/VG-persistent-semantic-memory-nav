"""Reviewer 1 (computational cost): per-step CPU cost of the memory layer vs memory size, on real HM3D memory entries.
Times, per call, exactly the per-step work the agent does over the whole store:
  (a) goal scoring: mobility lookup + reliability + CLIP-embedding relevance (+ lexical gate) for every entry
      (agent._select_target minus the navmesh standoff query; no lexical goal gate, as in the HM3D runs);
  (b) verification check: check_target_verified_detailed over the whole store.
Entries are real (largest HM3D snapshot); sizes beyond its count are reached by resampling entries with new ids.
Usage: python scripts/cost/bench_memory.py [snapshot.json]  -> results/post_paper/cost_memory_bench.json"""
import json, random, statistics, sys, time, os, platform
sys.path[:0] = ["packages/memory_layer", "packages/frontier_agent"]
from memory_layer.models import MemoryEntity
from memory_layer.store import MemoryStore
from memory_layer.mobility import MobilityPriors
from memory_layer.reliability import compute_reliability
from memory_layer.relevance import compute_relevance
from frontier_agent.verification import check_target_verified_detailed, _lexically_consistent

snap_path = sys.argv[1] if len(sys.argv) > 1 else "data/runs/phase4_batch_persist_memory_only_spread_on/memory_snapshot_ep3.json"
snap = json.load(open(snap_path))
ents = [e for e in snap["entities"] if e.get("embedding") and e.get("position_world")]
mp = MobilityPriors.load("configs/mobility_priors.json")
q = next(e["embedding"] for e in ents if "sofa" in e["description"].lower() or "chair" in e["description"].lower())
now = max(e["last_observed"] for e in ents) + 600
rng = random.Random(0)
alpha, lam = 0.5, 0.001

def make_store(n):
    s = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    for i in range(n):
        e = ents[i] if i < len(ents) else rng.choice(ents)
        s.entities[i] = MemoryEntity(entity_id=i, category=e["category"], description=e["description"], embedding=e["embedding"],
                                     position_world=e["position_world"], c_i=e.get("c_i", 1.0), n_i=e["n_i"], v_i=e["v_i"],
                                     first_observed=e["first_observed"], last_observed=e["last_observed"])
    return s

def score_all(s):
    best = 0.0
    for e in s.entities.values():
        r = compute_reliability(e, now, mp.mobility_for(e.category), alpha, lam) * compute_relevance(q, e.embedding)
        best = max(best, r)
    return best

import numpy as np
def score_np(s):
    # same relevance computation, vectorised (embedding matrix kept in memory): what a matrix implementation would cost
    E = s._E; qq = np.asarray(q, dtype=np.float32)
    rel = E @ qq / (np.linalg.norm(E, axis=1) * np.linalg.norm(qq) + 1e-9)
    return float(rel.max())

def verify(s):
    return check_target_verified_detailed([0, 0, 0], s, q, 0.18, 1.5, now, mp, alpha, lam, query_text="a sofa")

def t(fn, s, reps):
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(s); xs.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(xs), 2)

out = {"snapshot": snap_path, "n_real_entries": len(ents), "cpu": platform.processor() or platform.machine(),
       "python": platform.python_version(), "rows": []}
for n in (250, 500, 1000, 2000, 3000):
    s = make_store(n)
    s._E = np.asarray([e.embedding for e in s.entities.values()], dtype=np.float32)
    row = {"entries": n, "score_ms": t(score_all, s, 20), "verify_ms": t(verify, s, 20), "relevance_vectorised_ms": t(score_np, s, 50)}
    row["total_ms"] = round(row["score_ms"] + row["verify_ms"], 2)
    out["rows"].append(row); print(row, flush=True)
json.dump(out, open("results/post_paper/cost_memory_bench.json", "w"), indent=1)
