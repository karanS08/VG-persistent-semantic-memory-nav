"""Phase 4 Step 4.4 verification: a small batch of full episodes end-to-end, real
DAAAM grounding, real memory, real frontier decisions -- logging SPL, success rate,
and per-branch usage counts. Phase 4 gate also requires at least one full episode's
Rerun recording, produced here for episode 0.

Scope note: "time_budget"/"path_budget" are expressed in STEPS here (1 unit per
env.step call), not wall-clock seconds -- simpler to reason about for a scripted
episode and avoids conflating DAAAM's real (and here, deliberately slow/batched)
grounding latency with the agent's own budget. `configs/memory.yaml` doesn't specify
episode-length units, so this is a scoping choice, logged as such.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import rerun as rr
import yaml

from daaam_adapter.bridge_client import DaaamBridgeClient
from sim_bridge.habitat_env import HabitatEnv
from sim_bridge import rerun_stream
from frontier_agent.agent import FrontierAgent
from frontier_agent.frontier import sample_frontier_candidates, coverage_ratio
from frontier_agent.stopping import check_stopping, EpisodeOutcome
from frontier_agent.verification import check_target_verified_detailed
from frontier_agent.value_map import euclidean
from memory_layer.mobility import MobilityPriors
from memory_layer.relevance import ClipTextEncoder
from memory_layer.store import MemoryStore, Observation, Verification

# pipeline-v8 options (all off by default -> behaviour identical to tag acra2026-submitted). Set once in main()
# from --fuse-positions / --revisit-evidence / --image-check; read by run_episode. A module-level setting instead of
# new run_episode parameters keeps both call sites (and every existing caller) untouched.
V8 = {"fuse_positions": False, "revisit": None, "image_checker": None, "image_holdout_steps": 20}

QUERIES = [
    "a soft cushion", "a large window", "a wooden chair", "a wall-mounted mirror",
    "a dining table", "a wall-mounted radiator", "a doorway", "a framed painting",
]
# 2026-08-24: "a potted plant" and "a bed with pillows" were replaced -- across
# 1492 real distinct DAAAM captions mined from the completed .rrd recordings
# (data/runs/phase4_batch/episode_*.rrd, the 00770-NBg5UqG3di3 scene), NEITHER
# object was ever captioned even once, while every other query had multiple real
# near-matches. Both consistently failed (budget-exhausted) across the memory-on
# and memory-off batches for this reason, not a pipeline bug. The replacements
# were chosen the same way, not guessed: "radiator" appears in 137 real captions,
# "painting" in 47, both with descriptions closely matching the query phrasing
# (e.g. "A beige, vertical, wall-mounted radiator with a flat front panel...").
# See PROGRESS.md 2026-08-24.


def camera_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2}


def _provenance() -> dict:
    """2026-09-28 Post-Paper Test Plan 2.4: git commit + dirty state + argv."""
    import subprocess, sys
    def _git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:  # noqa: BLE001 -- provenance must never kill a run
            return f"unavailable: {exc}"
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_dirty_files": _git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "argv": sys.argv, "started_or_written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def wxyz_to_xyzw_transform(position, rotation_wxyz) -> list:
    w, x, y, z = rotation_wxyz
    return [position[0], position[1], position[2], x, y, z, w]


def run_episode(episode_id, query, client, env, agent, encoder, store, max_steps, log_rerun,
                 n_frontier_candidates=30, coverage_threshold=0.95, verify_relevance_threshold=0.18,
                 verify_distance=1.5, post_loop_grounding_wait_s=30, out_dir="data/runs/phase4_batch",
                 reset_env=True, close_env=True, min_confirming_observations=1, vqa_verification=False,
                 coverage_memory_enabled=True, disable_lexical_gate=False,
                 disable_fresh_evidence_guard=False, disable_relevance_gate=False,
                 inject_latency_s=0.0, lexical_mode="prefix", id_offset=0, tour=None, instance_target=None,
                 verification_feedback=False, local_verification=False, live_verification=False):
    # id_offset (2026-09-29 fix): DAAAM numbers objects per session from scratch. When
    # memory persists across DAAAM restarts (memory-only mode), a new session's object
    # #k used to overwrite the unrelated remembered object #k. Callers pass a distinct
    # offset per DAAAM session so remembered IDs stay unique.
    random.seed(episode_id)
    episode_start_time = time.time()  # 2026-08-28: wall-clock retrieval latency, not just
    # step count -- a real deployment cares how long the user actually waits, and step
    # count alone hides DAAAM grounding latency (real, timing-dependent async VLM
    # calls) and the fixed post_loop_grounding_wait_s below. Includes everything from
    # episode start through the post-loop grounding wait, i.e. "time until this
    # query's answer -- success or failure -- is actually final."
    # 2026-09-02: persist_coverage=not reset_env -- the exact same signal that
    # already decides whether the memory store and agent position carry over
    # governs whether spatial coverage memory does too (see agent.reset()'s own
    # docstring for why this specific piece of state, unlike goal-path state or
    # per-query exclusions, should track the session rather than the query).
    # 2026-09-04: `and coverage_memory_enabled` added after the first real batch
    # validation came back inconclusive (two identical-config repeats disagreed
    # with each other more than either disagreed with the pre-feature baseline --
    # see PROGRESS.md). Needed a same-day, same-environment ablation control,
    # which the --persistent-memory flag alone can no longer provide now that
    # coverage memory is unconditionally bundled into it -- this flag is that
    # control, not a new user-facing mode.
    agent.reset(persist_coverage=(not reset_env) and coverage_memory_enabled)  # clear any goal-persistence state
    # from a prior use of this agent instance -- NOT the memory store, which is a
    # separate object and (in persistent mode) is deliberately left untouched so
    # it carries over into this query.
    # reset_env=False (persistent-across-queries mode, all but the first query in
    # the session): resume from wherever the agent physically is, rather than
    # teleporting back to the scene's spawn pose -- see PROGRESS.md 2026-08-25.
    frame = env.reset() if reset_env else env.current_frame()
    query_start_timestamp = frame.timestamp  # 2026-09-28: stale-target logging below
    start_position = frame.pose["position"]

    habitat_cfg = env._config
    intrinsics = camera_intrinsics(
        habitat_cfg["sensor"]["rgb_resolution"][1], habitat_cfg["sensor"]["rgb_resolution"][0], habitat_cfg["sensor"]["hfov"],
    )
    reference_candidates = sample_frontier_candidates(env.pathfinder, start_position, n_candidates=n_frontier_candidates, min_radius=1.0, max_radius=8.0)
    visited_xy = [[start_position[0], start_position[2]]]
    path_length = 0.0
    branch_counts = {"direct_recall": 0, "contextual_anchor": 0, "fallback": 0}
    # "does memory help": how often, of the decisions where the contextual-anchor
    # block actually ran (memory_influenced is not None), did including remembered
    # (past, non-live) signals change which candidate won vs. live-only scoring?
    memory_probe = {"evaluated": 0, "influenced": 0}
    # 2026-09-05: critique.md SS3/SS13/SS39 -- "memory changed the action" is not
    # the same claim as "memory helped." Track every entity a memory-influenced
    # decision ever chose as this episode's replan target; at episode end, compare
    # against whichever entity actually got verified (if any) to classify this
    # episode's influence, not just count it. See the classification logic below
    # (near `influence_rate`) for the full 4-way outcome and why it's computed at
    # episode granularity, not per-decision: the same memory-influenced entity is
    # typically the replan target across many consecutive steps, so attributing
    # "helpful" to one specific step among those would be arbitrary.
    memory_influenced_entity_ids = set()
    # critique.md SS8: "direct recall" (branch=="direct_recall") is a candidate
    # target selection, not a confirmed find -- track every entity id it ever
    # targeted this episode so we can classify correct-vs-incorrect against the
    # episode's actual verified outcome below (near memory_influence_classification),
    # rather than just counting how often the branch fired.
    direct_recall_entity_ids = set()
    usable_growth_curve = [(-1, 0)]  # (step, usable_entity_count) whenever it changes -- see PROGRESS.md Phase 4
    # 2026-09-06: critique.md "asynchronous DAAAM latency needs experimental
    # validation" -- injects ADDITIONAL synthetic delay on top of DAAAM's real
    # ~7-21s grounding latency, so the effect of latency itself (not just its
    # real, uncontrolled magnitude) can be measured. Updates that arrive from
    # pull_updates() are held in this buffer and only released into the memory
    # store once frame.timestamp >= release_time; at inject_latency_s=0.0 (the
    # default) release_time == arrival time, so this buffer changes nothing
    # about existing results -- confirmed by the full test suite staying green.
    pending_updates = []  # list of (release_time, update_dict)
    query_embedding = encoder.embed(query).tolist()

    if log_rerun:
        rerun_stream.init(app_id=f"rpv_mem_phase4_ep{episode_id}", serve=False)
        rerun_stream.log_frame(frame)

    outcome = None
    verified_entity_id = None
    verification_detail = None
    step = 0
    # 2026-09-29 tour mode: drive a fixed route through the listed waypoints
    # (navmesh shortest paths between them) with a 360-degree look-around at each,
    # instead of the frontier agent -- a "show the robot the house once" pass that
    # builds memory. Perception, memory updates and verification run as usual.
    tour_plan = None
    if tour:
        from types import SimpleNamespace
        from frontier_agent.heading import choose_action
        from frontier_agent.pathing import compute_path
        # plan items carry their waypoint index so a blocked waypoint can be skipped (2026-11-03 stuck fix below)
        def _plan_tour(start, wps):
            plan, here = [], list(start)
            for wi, wp in wps:
                goal = list(env.pathfinder.snap_point(wp))
                seg = compute_path(env.pathfinder, here, goal)
                if seg is None:
                    print(f"  tour: waypoint {wp} unreachable, skipped"); continue
                plan += [("goto", pt, wi) for pt in seg[1:]] + [("spin", 36, wi)]
                here = goal
            return plan
        tour_wps = list(enumerate(tour))
        tour_plan = _plan_tour(frame.pose["position"], tour_wps)
        tour_last_pos, tour_stuck, tour_last_act, tour_retries = None, 0, None, {}
        print(f"  tour: {len(tour)} waypoints -> {len(tour_plan)} plan items")
    # 2026-10-30 --verification-feedback (off by default; THOR moved-object test only): a VQA rejection is written
    # back to memory as a failed verification (v_i discount, the paper's own rule, which the closed loop otherwise
    # never applies), and the rejected entry is not re-sent to VQA until it has been observed again (n_i grows) --
    # without this, a stale entry is re-queried every step and one VQA false "yes" in ~30 calls ends the episode.
    vqa_rejected_at_n = {}
    # pipeline-v8 state (inert unless the V8 flags are set in main())
    cam_height = habitat_cfg["sensor"].get("height", 1.5)
    revisit = V8["revisit"]
    if revisit is not None:
        revisit.begin_query(episode_start_time)
    revisit_log = []      # (step, entity_id, description) for each v_i discount from revisit evidence
    image_log = []        # one dict per image check
    image_holdout = {}    # entity_id -> step until which it is held out after a failed image check
    for step in range(max_steps):
        transform = wxyz_to_xyzw_transform(frame.pose["position"], frame.pose["rotation"])
        push_resp = client.push_frame(
            frame_id=frame.step_id, timestamp=frame.timestamp,
            rgb=frame.rgb[:, :, :3], depth=frame.depth,
            transform=transform, camera_intrinsics=intrinsics,
        )

        new_updates = client.pull_updates()
        # fetched every step (not just when updates arrive) so backfill_positions
        # below can catch already-known entities whose position only becomes
        # available after their one grounding event already streamed through with
        # position_world=None -- see PROGRESS.md Phase 4 / MemoryStore.backfill_positions.
        # (Position availability itself is not delayed by inject_latency_s -- the
        # critique this flag answers is specifically about DAAAM's captioning
        # latency, not the separate /positions endpoint.)
        positions = client.get_positions()
        for u in new_updates:
            pending_updates.append((frame.timestamp + inject_latency_s, u))
        ready_updates = [u for release_time, u in pending_updates if release_time <= frame.timestamp]
        if ready_updates:
            pending_updates = [(rt, u) for rt, u in pending_updates if rt > frame.timestamp]
        updates = ready_updates
        live_ids = set()
        if updates:
            observations = []
            for u in updates:
                sid = u["semantic_id"] + id_offset
                pos_list = positions.get(str(u["semantic_id"]))
                position_world = pos_list[-1]["position_world"] if pos_list else None
                if pos_list and V8["fuse_positions"]:  # v8: median over the track's whole position history
                    position_world = [float(x) for x in np.median([o["position_world"] for o in pos_list], axis=0)]
                # Use DAAAM's own real observation window (first_observed/last_observed,
                # derived from per-frame timestamps recorded WHILE the object was
                # actually tracked) rather than u["timestamp"] (when this grounding
                # result ARRIVED, which can be 80-100+ steps later -- see
                # PROGRESS.md's 2026-08-24 entry on this exact bug). Falls back to
                # arrival time only if DAAAM didn't report temporal data at all.
                observations.append(Observation(
                    entity_id=sid, category=u["semantic_label"], description=u["semantic_label"],
                    embedding=u.get("embedding"), confidence=1.0, room_label=None,
                    timestamp=u.get("last_observed", u["timestamp"]),
                    first_observed=u.get("first_observed"),
                    position_world=position_world,
                ))
                live_ids.add(sid)
            store.update(observations, [])
        store.backfill_positions({str(int(k) + id_offset): v for k, v in positions.items()} if id_offset else positions)
        if V8["fuse_positions"]:
            store.fuse_positions(positions, id_offset)
        if revisit is not None:
            tracks = [(int(k) + id_offset, v[-1]["position_world"]) for k, v in positions.items() if v]
            for eid in revisit.step(store.entities, frame.timestamp, frame.pose, frame.depth, intrinsics,
                                    cam_height, tracks):
                e = store.entities[eid]
                revisit_log.append((step, eid, (e.description or "")[:80]))
                print(f"    [revisit] entity {eid} seen-empty -> v_i {e.v_i:.3f}: {(e.description or '')[:60]}")

        usable_now = sum(1 for e in store.entities.values() if e.position_world is not None and e.embedding is not None)
        if usable_now != usable_growth_curve[-1][1]:
            usable_growth_curve.append((step, usable_now))
            print(f"    [step {step}] usable (positioned+embedded) entity count: {usable_now}")

        if log_rerun:
            # Option B (see PROGRESS.md / conversation): bake DAAAM's real
            # detections into the recording itself, not just the console --
            # borrows the entity-path convention from run_phase2_rerun.py, but
            # simpler here since MemoryStore already merges description+position
            # for us (no separate corrections/positions cross-referencing needed).
            rr.set_time_sequence("step", frame.step_id)
            for u in updates:
                rr.log(f"world/daaam/live_captions/{u['semantic_id']}", rr.TextLog(u["semantic_label"]))
            positioned = [e for e in store.entities.values() if e.position_world is not None]
            if positioned:
                rr.log(
                    "world/daaam/objects",
                    rr.Points3D(
                        [e.position_world for e in positioned],
                        labels=[e.description[:60] for e in positioned],
                        radii=0.08,
                    ),
                )

        # 2026-09-05: critique.md SS5/SS33 leave-one-gate-out ablation support.
        # Gate 3 (multi-observation reconfirmation) is already ablatable via
        # --min-confirming-observations 0 (n_i < 0 is never true, so nothing is
        # ever excluded) -- no new parameter needed for it. Gates 2 and 4 below.
        # 2026-10-03 demo-only instance gate (--instance-target; off by default, not used in
        # any paper experiment): for one query, verification only considers entries within
        # `radius` m (floor plane) of a given target, so memory-on and memory-off searches must
        # end at the same physical object. Filtering BEFORE verification matters: verification
        # returns a single winner, so rejecting an off-target winner afterwards would hide an
        # on-target candidate (seen in the first instance-gated run, 2026-10-03).
        vstore = store
        if instance_target and query == instance_target["query"]:
            tx, tz, rad = instance_target["xyz"][0], instance_target["xyz"][2], instance_target["radius"]
            vstore = copy.copy(store)
            vstore.entities = {k: e for k, e in store.entities.items() if e.position_world is not None
                               and math.hypot(e.position_world[0] - tx, e.position_world[2] - tz) <= rad}
        if live_verification:
            # 2026-10-30 --live-verification: only entries observed during THIS episode may verify; remembered entries
            # still steer navigation but cannot end the search (verification by re-observation). Replaces VQA for
            # the THOR moved-object test: a second DAM-3B for VQA does not fit beside DAAAM's own on a 24 GB GPU.
            vstore = copy.copy(vstore)
            vstore.entities = {k: e for k, e in vstore.entities.items() if e.last_observed >= episode_start_time}
        if local_verification:
            # 2026-10-30 --local-verification (VQA runs only): the best candidate WITHIN verify_distance is sent to
            # VQA, instead of only the globally top-ranked entry (which blocked verifying a sofa the agent stood at
            # while a higher-ranked remembered sofa sat elsewhere). VQA on the live frame is the final authority.
            ax, az = frame.pose["position"][0], frame.pose["position"][2]
            vstore = copy.copy(vstore)
            vstore.entities = {k: e for k, e in vstore.entities.items() if e.position_world is not None
                               and math.hypot(e.position_world[0] - ax, e.position_world[2] - az) <= verify_distance}
        if verification_feedback and vqa_rejected_at_n:
            vstore = copy.copy(vstore)
            vstore.entities = {k: e for k, e in vstore.entities.items()
                               if not (k in vqa_rejected_at_n and e.n_i <= vqa_rejected_at_n[k])}
        if image_holdout:
            # v8: an entry that failed the image check sits out for a few steps, so a mis-captioned chair ranked
            # above a real sofa within verify_distance cannot block the sofa from being checked. Time-limited
            # (n_i rarely grows, so "until re-observed" would be forever) and never written to memory.
            vstore = copy.copy(vstore)
            vstore.entities = {k: e for k, e in vstore.entities.items() if image_holdout.get(k, -1) < step}
        verification = check_target_verified_detailed(
            frame.pose["position"], vstore, query_embedding,
            # -1.0 is a guaranteed pass-through: CLIP cosine similarity is
            # bounded in [-1, 1], so no real relevance score can fail this.
            relevance_threshold=(-1.0 if disable_relevance_gate else verify_relevance_threshold),
            verify_distance=verify_distance,
            now=frame.timestamp, mobility_priors=agent.mobility_priors,
            alpha=agent.alpha, lambda_base=agent.lambda_base,
            mobility_scaling_enabled=agent.mobility_scaling_enabled,
            min_confirming_observations=min_confirming_observations,
            query_text=(None if disable_lexical_gate else query),
            lexical_mode=lexical_mode,
        )
        verified, vid = verification.verified, verification.entity_id
        if verified and step == 0 and not reset_env and not disable_fresh_evidence_guard:
            # Persistent-mode false-positive guard (PROGRESS.md 2026-08-27): at
            # query start in persistent mode, `frame` is the carried-over position
            # from the PREVIOUS query -- no action has been taken for THIS query
            # yet, so a "verified" hit here is necessarily against a stale memory
            # entity from earlier exploration, not evidence this query's own
            # search made progress. Concretely caught: episode 4 ("a dining
            # table") verified in 1 step / 0m movement against entity 1001, whose
            # actual description was "a long, narrow, white marble countertop" --
            # a false positive that only existed because verification is checked
            # against the WHOLE accumulated memory pool from step 0, with no
            # requirement that this query did any of the finding. Suppress this
            # one step's verification and require >=1 real step this query first.
            print(f"    [step 0, persistent] suppressing stale-memory verification "
                  f"(entity {vid} would have triggered) -- requires >=1 real step "
                  f"this query first")
            verified, vid = False, None

        if verified and V8["image_checker"] is not None and store.entities.get(vid) is not None:
            # v8 image check: the verified entry must look like the query in the CURRENT frame (CLIP image crop,
            # top-1 over a fixed household vocabulary). not_visible -> just wait for a better view.
            ic = V8["image_checker"].check(query, store.entities[vid].position_world, frame.rgb, frame.depth,
                                           frame.pose, intrinsics, cam_height)
            image_log.append({"step": step, "entity_id": vid, **ic})
            if ic["status"] != "pass":
                if ic["status"] == "fail":
                    image_holdout[vid] = step + V8["image_holdout_steps"]
                    print(f"    [image] entity {vid} rejected: looks like {ic['top_label']!r} "
                          f"(query rank {ic['rank']}, margin {ic['margin']})")
                verified, vid = False, None
            else:
                print(f"    [image] entity {vid} confirmed (margin {ic['margin']} over {ic['runner_up']!r})")

        vqa_raw_answer = None
        if verified and vqa_verification:
            # 2026-09-02: final confirmation authority is a real VQA judgment on
            # the CURRENT frame, not the CLIP-threshold+gates check above --
            # see daaam_bridge_server.py's /verify docstring for the full
            # motivation (CLIP similarity at this pipeline's operating point is
            # inside CLIP's own noise floor; every false-positive mechanism
            # found this project traces back to trusting that number). The
            # cheap gates above still decide WHEN it's worth paying a
            # synchronous DAM-3B call's latency; they no longer decide the
            # outcome by themselves.
            vqa_result = client.verify_region(frame.rgb[:, :, :3], query)
            vqa_raw_answer = vqa_result.get("raw_answer")
            if not vqa_result.get("verified"):
                print(f"    [VQA] entity {vid} rejected -- DAM-3B said: {vqa_raw_answer!r}")
                if verification_feedback and vid in store.entities:
                    store.update([], [Verification(entity_id=vid, success=False)])
                    vqa_rejected_at_n[vid] = store.entities[vid].n_i
                    print(f"    [feedback] entity {vid} v_i -> {store.entities[vid].v_i:.3f}; held out until re-observed")
                verified, vid = False, None
            else:
                print(f"    [VQA] entity {vid} confirmed -- DAM-3B said: {vqa_raw_answer!r}")

        cov = coverage_ratio(visited_xy, reference_candidates, visited_radius=1.0)
        stop = check_stopping(
            target_verified=verified,
            elapsed_time=float(step), time_budget=float(max_steps),
            path_length=path_length, path_budget=1e9,  # not the binding constraint in this batch
            coverage_ratio=cov, coverage_threshold=(2.0 if tour else coverage_threshold),  # tour: no coverage stop
            best_guess_position=frame.pose["position"],
        )
        if stop.outcome != EpisodeOutcome.CONTINUE:
            outcome = stop
            verified_entity_id = vid if verified else None
            if verified:
                verification_detail = {
                    "entity_id": verification.entity_id,
                    "entity_position_world": (list(store.entities[verification.entity_id].position_world)
                                              if verification.entity_id in store.entities
                                              and store.entities[verification.entity_id].position_world is not None
                                              else None),
                    "description": verification.description,
                    "relevance": round(verification.relevance, 4),
                    "distance": round(verification.distance, 3),
                    "reliability": round(verification.reliability, 4),
                    "r_i": round(verification.r_i, 4),
                    "n_i": verification.n_i,
                    "n_eligible": verification.n_eligible,
                    "n_excluded_by_confirmation_gate": verification.n_excluded_by_confirmation_gate,
                    "n_excluded_by_category_gate": verification.n_excluded_by_category_gate,
                    "best_relevance_in_memory": round(verification.best_relevance_in_memory, 4),
                    "best_relevance_description": verification.best_relevance_description,
                    "threshold_used": verify_relevance_threshold,
                    "vqa_raw_answer": vqa_raw_answer,
                }
                print(f"  VERIFIED by entity {verification.entity_id} "
                      f"(relevance={verification.relevance:.4f}, r_i={verification.r_i:.4f}, "
                      f"n_i={verification.n_i}, dist={verification.distance:.2f}m, "
                      f"{verification.n_eligible} eligible)")
                print(f"    triggered on: {verification.description[:100]}")
                print(f"    best in memory: {verification.best_relevance_in_memory:.4f} -> "
                      f"{(verification.best_relevance_description or '')[:100]}")
            elif verification.best_relevance_in_memory is not None:
                # Failure diagnostic gap (2026-08-24): verification_detail used to
                # stay None on every failed episode, so there was no way to tell
                # "nothing in the scene matched" from "the best match existed but was
                # never reached." Recording it on failure too, using the LAST step's
                # snapshot -- what best_relevance_in_memory ended up being tells us
                # whether the rank-based verification fix (see PROGRESS.md
                # 2026-08-24) is being too strict about reachability, versus DAAAM
                # genuinely never producing a good-enough caption for the query.
                verification_detail = {
                    "entity_id": None,
                    "description": None,
                    "relevance": None,
                    "distance": None,
                    "n_eligible": 0,
                    "n_excluded_by_confirmation_gate": verification.n_excluded_by_confirmation_gate,
                    "n_excluded_by_category_gate": verification.n_excluded_by_category_gate,
                    "best_relevance_in_memory": round(verification.best_relevance_in_memory, 4),
                    "best_relevance_description": verification.best_relevance_description,
                    "threshold_used": verify_relevance_threshold,
                    "vqa_raw_answer": vqa_raw_answer,
                }
                print(f"  NOT VERIFIED at episode end -- best in memory: "
                      f"{verification.best_relevance_in_memory:.4f} -> "
                      f"{(verification.best_relevance_description or '')[:100]}")
            break

        if tour_plan is not None:
            act = None
            # 2026-11-03 stuck fix (tour mode only): THOR's reachable grid can route through a spot the agent cannot
            # pass (house 72 tour: 3200 blocked move_forward steps at one spot). After 15 goto steps without moving
            # 2 cm, skip the current waypoint and re-plan the remaining ones from where the agent is.
            pos_now = list(frame.pose["position"])
            if tour_plan and tour_plan[0][0] == "goto" and tour_last_pos is not None and tour_last_act == "move_forward" and \
                    math.hypot(pos_now[0] - tour_last_pos[0], pos_now[2] - tour_last_pos[2]) < 0.02:
                tour_stuck += 1
            else:
                tour_stuck = 0
            tour_last_pos = pos_now
            if tour_stuck >= 15:
                # block the grid cell the agent cannot enter and re-plan to the SAME waypoint (as the search agent
                # does on collision); skip the waypoint only after 6 failed re-plans
                wi, target = tour_plan[0][2], tour_plan[0][1]
                blocked = env.pathfinder.block_point(target) if hasattr(env.pathfinder, "block_point") else False
                tour_retries[wi] = tour_retries.get(wi, 0) + 1
                keep = tour_retries[wi] <= 6
                print(f"  tour: stuck at step {step} near {[round(v, 2) for v in pos_now]} (cell blocked={blocked}), "
                      f"{'re-planning to' if keep else 'skipping'} waypoint {wi}")
                tour_plan = _plan_tour(pos_now, [w for w in tour_wps if (w[0] >= wi if keep else w[0] > wi)])
                tour_stuck = 0
            while tour_plan and act is None:
                kind, arg, wi_ = tour_plan[0]
                if kind == "spin":
                    act = "turn_left"
                    tour_plan[0] = ("spin", arg - 1, wi_)
                    if arg - 1 <= 0:
                        tour_plan.pop(0)
                else:
                    a_ = choose_action(frame.pose["position"], frame.pose["rotation"], arg)
                    if a_ == "stop":
                        tour_plan.pop(0)
                    else:
                        act = a_
            if act is None:
                print(f"  tour complete at step {step}")
                break
            tour_last_act = act
            decision = SimpleNamespace(action=act, branch="fallback", goal_entity_id=None, memory_influenced=None)
        else:
            decision = agent.decide(
                agent_position=frame.pose["position"], agent_rotation_wxyz=frame.pose["rotation"],
                query=query, memory_store=store, live_entity_ids=live_ids, now=frame.timestamp,
            )
        branch_counts[decision.branch] += 1
        if decision.branch == "direct_recall" and decision.goal_entity_id is not None:
            direct_recall_entity_ids.add(decision.goal_entity_id)
        if decision.memory_influenced is not None:
            memory_probe["evaluated"] += 1
            if decision.memory_influenced:
                memory_probe["influenced"] += 1
                if decision.goal_entity_id is not None:
                    memory_influenced_entity_ids.add(decision.goal_entity_id)
            # 2026-09-28 Post-Paper Test Plan 2.2: per-decision record for the
            # offline helpful/harmful influence classifier (logging only).
            with open(Path(out_dir) / f"decisions_ep{episode_id}.jsonl", "a") as _df:
                _df.write(json.dumps({
                    "episode_id": episode_id, "query": query, "step": step,
                    "agent_position": list(frame.pose["position"]), "branch": decision.branch,
                    "memory_influenced": decision.memory_influenced,
                    "goal_entity_id": decision.goal_entity_id,
                    "memory_target_entity_id": decision.memory_target_entity_id,
                    "memory_target_position": decision.memory_target_position,
                    "live_target_entity_id": decision.live_target_entity_id,
                    "live_target_position": decision.live_target_position,
                    # age of the WITH-memory target and whether it was last seen
                    # before THIS query began (i.e. carried-over, possibly stale)
                    "memory_target_age_s": (frame.timestamp - store.entities[decision.memory_target_entity_id].last_observed
                                            if decision.memory_target_entity_id in store.entities else None),
                    "memory_target_pre_query": (store.entities[decision.memory_target_entity_id].last_observed < query_start_timestamp
                                                if decision.memory_target_entity_id in store.entities else None),
                    "value_with": decision.best_value_with_memory,
                    "value_without": decision.best_value_without_memory,
                    "live_signal_count": decision.live_signal_count,
                    "remembered_signal_count": decision.remembered_signal_count,
                }) + "\n")
            print(f"    [step {step}] contextual-anchor eval: live={decision.live_signal_count} "
                  f"remembered={decision.remembered_signal_count} "
                  f"value_with={decision.best_value_with_memory:.4f} "
                  f"value_without={decision.best_value_without_memory:.4f} "
                  f"influenced={decision.memory_influenced}")

        prev_xy = [frame.pose["position"][0], frame.pose["position"][2]]
        # 2026-10-31: per-step trajectory (logging only) -- failure classification needs where the agent went
        with open(Path(out_dir) / f"trajectory_ep{episode_id}.jsonl", "a") as _tf:
            _tf.write(json.dumps({"step": step, "xy": prev_xy, "action": decision.action, "branch": decision.branch,
                                  "goal_entity_id": decision.goal_entity_id}) + "\n")
        frame = env.step(decision.action if decision.action != "stop" else "move_forward")
        new_xy = [frame.pose["position"][0], frame.pose["position"][2]]
        path_length += euclidean(prev_xy, new_xy)
        visited_xy.append(new_xy)

        if log_rerun:
            rerun_stream.log_frame(frame)

    if outcome is None:
        # exhausted max_steps without any stop condition firing (e.g. budgets were
        # set loose relative to max_steps) -- treat as a failure, not a silent
        # success, per the spec's "never count an unverified best guess as success"
        outcome_type = EpisodeOutcome.FAILURE
        reason = "max_steps_exhausted"
        # 2026-08-28 diagnostic-gap fix: this path bypasses the in-loop
        # `stop.outcome != CONTINUE` branch entirely (found auditing Tier 4 on
        # scene 00861 -- EVERY episode fell through here, leaving
        # verification_detail permanently None with no way to tell "genuinely
        # never seen" from "seen but blocked by a gate"). Run one last
        # verification check against the final memory state so failures via
        # this path get the same best_relevance_in_memory diagnostic as
        # failures that trip via check_stopping's own budget/coverage checks.
        final_check = check_target_verified_detailed(
            frame.pose["position"], store, query_embedding,
            relevance_threshold=verify_relevance_threshold, verify_distance=verify_distance,
            now=frame.timestamp, mobility_priors=agent.mobility_priors,
            alpha=agent.alpha, lambda_base=agent.lambda_base,
            mobility_scaling_enabled=agent.mobility_scaling_enabled,
            min_confirming_observations=min_confirming_observations, query_text=query,
            lexical_mode=lexical_mode,
        )
        if final_check.best_relevance_in_memory is not None:
            verification_detail = {
                "entity_id": None, "description": None, "relevance": None, "distance": None,
                "n_eligible": 0,
                "n_excluded_by_confirmation_gate": final_check.n_excluded_by_confirmation_gate,
                "n_excluded_by_category_gate": final_check.n_excluded_by_category_gate,
                "best_relevance_in_memory": round(final_check.best_relevance_in_memory, 4),
                "best_relevance_description": final_check.best_relevance_description,
                "threshold_used": verify_relevance_threshold,
            }
            print(f"  NOT VERIFIED (max_steps_exhausted path) -- best in memory: "
                  f"{final_check.best_relevance_in_memory:.4f} -> "
                  f"{(final_check.best_relevance_description or '')[:100]}")
    else:
        outcome_type = outcome.outcome
        reason = outcome.reason

    spl = 0.0
    if outcome_type == EpisodeOutcome.SUCCESS and verified_entity_id is not None:
        import habitat_sim
        goal_position = store.entities[verified_entity_id].position_world
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = start_position
        shortest_path.requested_end = goal_position
        env.pathfinder.find_path(shortest_path)
        shortest = shortest_path.geodesic_distance
        if shortest is not None and not math.isinf(shortest) and max(path_length, shortest) > 0:
            spl = shortest / max(path_length, shortest)

    # give in-flight assignment/grounding a chance to finish before the caller
    # calls stop_episode() -- without this, a late-triggering assignment (which
    # happened at frame 49/59/60 in an earlier run of this exact script, well past
    # the nominal query_interval_frames=20) races stop()'s drain and its results
    # never make it into corrections. Matches the wait used in Phase 2's smoke
    # test; omitting it there was a real gap in an earlier version of this script.
    print(f"  waiting {post_loop_grounding_wait_s}s for in-flight grounding to finish...")
    time.sleep(post_loop_grounding_wait_s)

    # 2026-09-29: logging-only memory snapshot at episode end (per-object last-seen
    # times etc.), used for the teaser figure; never read by the controller.
    with open(Path(out_dir) / f"memory_snapshot_ep{episode_id}.json", "w") as _mf:
        json.dump({"now": frame.timestamp, "query": query, "entities": [
            {"id": e.entity_id, "category": e.category, "description": e.description,
             "position_world": e.position_world, "n_i": e.n_i, "v_i": e.v_i, "c_i": e.c_i,
             "embedding": e.embedding,
             "first_observed": e.first_observed, "last_observed": e.last_observed}
            for e in store.entities.values()]}, _mf)
    rrd_path = None
    if log_rerun:
        rrd_path = f"{out_dir}/episode_{episode_id}.rrd"
        rr.save(rrd_path)

    if close_env:
        env.close()

    n_with_position = sum(1 for e in store.entities.values() if e.position_world is not None)
    n_with_embedding = sum(1 for e in store.entities.values() if e.embedding is not None)
    print(f"  diagnostic: {n_with_position}/{len(store.entities)} entities have position_world, "
          f"{n_with_embedding}/{len(store.entities)} have embedding")
    influence_rate = (memory_probe["influenced"] / memory_probe["evaluated"]) if memory_probe["evaluated"] else None
    print(f"  memory probe: influenced {memory_probe['influenced']}/{memory_probe['evaluated']} "
          f"contextual-anchor evaluations"
          + (f" ({influence_rate:.0%})" if influence_rate is not None else " (block never ran)"))

    # critique.md SS3/SS13/SS39: classify this episode's memory influence rather
    # than only counting it. "not_applicable" = memory was never influential this
    # episode (memory_probe["influenced"]==0); "helpful" = the episode succeeded
    # AND the verified entity is one memory's inclusion had chosen as a replan
    # target at some point; "superseded" = the episode succeeded but via an
    # entity memory was never influential in choosing (the influence that did
    # occur didn't end up mattering to the actual find); "unresolved" = the
    # episode failed, so no success exists to attribute any decision's influence
    # to either way.
    verified_entity_id_final = verification_detail.get("entity_id") if verification_detail else None
    if memory_probe["influenced"] == 0:
        memory_influence_classification = "not_applicable"
    elif outcome_type.value == "success" and verified_entity_id_final in memory_influenced_entity_ids:
        memory_influence_classification = "helpful"
    elif outcome_type.value == "success":
        memory_influence_classification = "superseded"
    else:
        memory_influence_classification = "unresolved"

    # critique.md SS8: correctness proxy for direct recall re-uses the paper's
    # own verification criterion rather than a new ground-truth definition --
    # a direct-recall target is "correct" iff the episode succeeded via
    # verification of that exact entity id (not merely "the episode succeeded
    # by some means"), since direct recall commits the agent's path the moment
    # it fires and a wrong target cannot self-correct into a right one without
    # some other branch re-targeting later, which would already show up as a
    # different verified_entity_id_final.
    if not direct_recall_entity_ids:
        direct_recall_classification = "not_applicable"
    elif outcome_type.value == "success" and verified_entity_id_final in direct_recall_entity_ids:
        direct_recall_classification = "correct"
    else:
        direct_recall_classification = "incorrect"

    visited_x = [p[0] for p in visited_xy]
    visited_z = [p[1] for p in visited_xy]
    # room-crossing diagnostic for the 2026-08-24 multi-room scene switch: how far
    # the agent's actual trajectory spans, so "did it leave the starting room" can
    # be checked post-hoc without re-opening the .rrd -- see PROGRESS.md.
    traversal_span = {
        "start_xy": visited_xy[0], "end_xy": visited_xy[-1],
        "x_range": [round(min(visited_x), 3), round(max(visited_x), 3)],
        "z_range": [round(min(visited_z), 3), round(max(visited_z), 3)],
        "straight_line_start_to_end_m": round(
            math.hypot(visited_xy[-1][0] - visited_xy[0][0], visited_xy[-1][1] - visited_xy[0][1]), 3
        ),
    }

    wall_clock_seconds = round(time.time() - episode_start_time, 2)
    print(f"  wall_clock_seconds={wall_clock_seconds}")

    return {
        "episode_id": episode_id, "query": query, "outcome": outcome_type.value, "reason": reason,
        "steps_taken": step + 1, "path_length": round(path_length, 3), "spl": round(spl, 4),
        "branch_counts": branch_counts, "memory_size_at_end": len(store.entities),
        "n_with_position": n_with_position, "n_with_embedding": n_with_embedding,
        "memory_probe": memory_probe, "memory_influence_rate": influence_rate,
        "memory_influence_classification": memory_influence_classification,
        "direct_recall_classification": direct_recall_classification,
        "direct_recall_entity_ids": sorted(direct_recall_entity_ids),
        "verification_detail": verification_detail, "usable_growth_curve": usable_growth_curve[1:],
        "coverage_ratio_at_end": round(cov, 3), "rrd_path": rrd_path,
        "traversal_span": traversal_span, "wall_clock_seconds": wall_clock_seconds,
        "v8": {"fuse_positions": V8["fuse_positions"],
               "revisit": None if revisit is None else {"discounts": revisit_log, "n_inspections": revisit.n_inspections,
                                                         "n_empty_looks": revisit.n_empty_looks, "n_occupied_looks": revisit.n_occupied_looks},
               "image_check": None if V8["image_checker"] is None else image_log},
    }


def main() -> None:
    with open("configs/run_defaults.yaml") as f:
        run_defaults_full = yaml.safe_load(f)
    run_defaults = run_defaults_full["episode"]
    # CLAUDE.md Section 9 Step 4.5's three ablations, defaulting to fully-enabled
    # (baseline) if the section is missing from an older config file.
    ablation_cfg = run_defaults_full.get("ablations", {
        "memory_enabled": True, "mobility_scaling_enabled": True, "verification_discount_enabled": True,
    })
    persistent_default = run_defaults_full.get("persistent_across_queries", False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=run_defaults["max_steps"])
    parser.add_argument("--persistent-memory", action="store_true",
                         help="Force persistent_across_queries on regardless of the "
                              "yaml default -- one MemoryStore/DAAAM session/agent "
                              "position shared across the whole QUERIES loop.")
    parser.add_argument("--n-queries", type=int, default=None,
                         help="Only run the first N of QUERIES (for a quick smoke "
                              "test, e.g. of persistent-memory mode) instead of all "
                              "of them.")
    # CLAUDE.md Section 9 Step 4.5 ablation: "memory-off (live-only, fallback
    # always)" -- see FrontierAgent.__init__'s memory_enabled comment. Writes to a
    # separate directory so it never clobbers the memory-on batch's results.
    parser.add_argument("--memory-off", action="store_true",
                         help="Ablation: agent only ever acts on THIS step's live "
                              "entities, never anything remembered from earlier steps.")
    # CLAUDE.md Section 9 Step 4.5's other two ablations, same pattern as
    # --memory-off: a flag that changes one thing FrontierAgent/MemoryStore does,
    # with a separate output directory per active combination so results never
    # clobber each other or the non-ablated batch.
    parser.add_argument("--no-mobility-scaling", action="store_true",
                         help="Ablation: flat lambda_i=lambda_base for every "
                              "entity, ignoring the per-category mobility prior.")
    parser.add_argument("--no-verification-discount", action="store_true",
                         help="Ablation: v_i fixed at 1.0 -- failed verification "
                              "never discounts it.")
    # 2026-08-27 test-plan additions: cheap, code-free knobs for the
    # theta_direct calibration fix (PROGRESS.md 2026-08-24 diagnosis) and the
    # verification confirmation-count gate (PROGRESS.md 2026-08-27 fix), plus
    # a free-text tag so repeated/variant runs of the SAME ablation combo get
    # their own output directory instead of clobbering each other.
    parser.add_argument("--theta-direct", type=float, default=None,
                         help="Override configs/memory.yaml's recall.theta_direct "
                              "(direct-recall threshold on relevance*reliability) "
                              "for this run only.")
    parser.add_argument("--min-confirming-observations", type=int, default=1,
                         help="Verification confirmation-count gate: an entity "
                              "needs at least this many independent observations "
                              "(n_i) to be eligible to trigger verification. "
                              "Default changed 2->1 on 2026-09-02: once the "
                              "lexical category gate (2026-08-28 fix) is active, "
                              "it does the real discriminative work, and n_i>=2 "
                              "was found to discard true positives more than it "
                              "prevented false ones -- see PROGRESS.md.")
    parser.add_argument("--goal-lexical-gate", action="store_true",
                        help="Goal selection applies verification's head-noun check (off by default; see FrontierAgent).")
    parser.add_argument("--theta-gates-goals", action="store_true",
                        help="Entries with R_i < theta_direct are not taken as goals (off by default; see FrontierAgent).")
    parser.add_argument("--arrival-fix", action="store_true",
                        help="Re-plan onto the goal the agent is already at counts as arrival (off by default).")
    parser.add_argument("--live-verification", action="store_true",
                        help="Only entries observed during the current episode may verify (off by default; see run_episode).")
    parser.add_argument("--local-verification", action="store_true",
                        help="With --vqa-verification: verify the best candidate within verify_distance, not only the "
                             "global top-ranked entry (off by default; see run_episode).")
    parser.add_argument("--verification-feedback", action="store_true",
                        help="Write VQA rejections back to memory (v_i discount) and hold a rejected entry out of "
                             "verification until it is re-observed. Off by default; see run_episode.")
    parser.add_argument("--fuse-positions", action="store_true",
                        help="pipeline-v8: entity position = median of DAAAM's per-frame position history, refreshed "
                             "every step (default: the latest point at grounding time).")
    parser.add_argument("--revisit-evidence", action="store_true",
                        help="pipeline-v8: discount v_i of a remembered entry whose location is seen (in view, "
                             "0.5-3 m, unoccluded) with nothing of the same kind there (memory_layer/revisit.py).")
    parser.add_argument("--revisit-grace-s", type=float, default=10.0,
                        help="pipeline-v8: seconds to wait for a matching live detection before discounting.")
    parser.add_argument("--image-check", action="store_true",
                        help="pipeline-v8: verified entry must rank top-1 for the query among a fixed household "
                             "vocabulary in a CLIP image crop of the current frame (frontier_agent/image_check.py).")
    parser.add_argument("--image-check-topk", type=int, default=1,
                        help="pipeline-v8: pass if the query ranks within the top k (k chosen on ProcTHOR houses "
                             "2-40, never the evaluation house; see scripts/thor/calib_v8_image_check.py).")
    parser.add_argument("--vqa-verification", action="store_true",
                         help="2026-09-02: gate final verification acceptance on a "
                              "real synchronous VQA judgment ('is this a {query}?', "
                              "DAM-3B on the current frame via daaam_bridge_server's "
                              "/verify) rather than only the CLIP-threshold+gates "
                              "check. See PROGRESS.md 'verification redesign'.")
    parser.add_argument("--no-coverage-memory", action="store_true",
                         help="Ablation control, added 2026-09-04: coverage memory "
                              "(FrontierAgent._visited_positions persisting across "
                              "queries in --persistent-memory mode) is ON by default "
                              "whenever --persistent-memory is set -- there is no "
                              "separate flag to opt into it, per the design decision "
                              "to bundle it into the existing flag. This flag exists "
                              "only so a same-day, same-environment baseline can "
                              "still be run for comparison now that --persistent-memory "
                              "alone can't provide one. See PROGRESS.md.")
    # 2026-09-05: critique.md SS5/SS33 leave-one-gate-out verification ablation.
    # Gate 3 (multi-observation reconfirmation) already has a full ablation via
    # --min-confirming-observations 0. These three cover the remaining gates:
    # gate 1 (genuine per-query evidence), gate 2 (relevance threshold), gate 4
    # (lexical consistency). See paper Section 7.1 for what each gate targets.
    parser.add_argument("--no-fresh-evidence-guard", action="store_true",
                         help="Disables gate 1 (genuine per-query evidence): the "
                              "step-0 stale-verification suppression that normally "
                              "requires >=1 real step of THIS query's own search "
                              "before a hit counts, in persistent-position mode.")
    parser.add_argument("--no-relevance-gate", action="store_true",
                         help="Disables gate 2 (relevance threshold) by forcing "
                              "relevance_threshold to -1.0 (a guaranteed pass, since "
                              "CLIP cosine similarity is bounded in [-1,1]) for this "
                              "run only.")
    parser.add_argument("--no-lexical-gate", action="store_true",
                         help="Disables gate 4 (lexical consistency with the "
                              "entity's description) by passing query_text=None to "
                              "check_target_verified_detailed for this run only -- "
                              "the same as never having supplied it.")
    parser.add_argument("--tour-waypoints", default=None,
                         help="JSON file with a list of [x,y,z] world waypoints: the FIRST query of the "
                              "session drives this route (with a 360-degree look-around at each) instead "
                              "of searching, to build memory. Memory-only mode.")
    parser.add_argument("--tour-max-steps", type=int, default=3000, help="step budget for the tour episode only")
    parser.add_argument("--verify-distance", type=float, default=None,
                         help="DEMO ONLY: verification range in metres (run_defaults.yaml: 1.5). DAAAM "
                              "positions are 1-2 m off, so a robot standing at the right spot can be just out "
                              "of range. Off (config default) unless given.")
    parser.add_argument("--scan-steps", type=int, default=None,
                         help="DEMO ONLY: steps the agent turns in place on arriving at a target "
                              "(agent default 6, ~3 s). Longer lets late DAAAM captions arrive before it "
                              "moves on. Off (agent default) unless given.")
    parser.add_argument("--instance-target", default=None,
                         help='DEMO ONLY: JSON {"query": ..., "xyz": [x,y,z], "radius": m}; that query succeeds only '
                              'on an entity within radius of xyz (same physical object). Off by default.')
    parser.add_argument("--init-memory-snapshot", default=None,
                         help="memory-only mode: start the session from a saved memory_snapshot_ep*.json "
                              "(needs embeddings; written by runs from 2026-09-29 on).")
    parser.add_argument("--no-rrd", action="store_true", default=os.environ.get("RPV_NO_RRD") == "1",
                         help="Skip Rerun logging and the per-episode .rrd recording "
                              "(~0.7 GB/episode). Metrics and decisions_ep*.jsonl are unaffected.")
    parser.add_argument("--lambda-base", type=float, default=None,
                         help="Override configs/memory.yaml reliability.lambda_base for this "
                              "run only (Post-Paper Test Plan 1.4/3.2); 0 = no temporal decay.")
    parser.add_argument("--lexical-mode", choices=["prefix", "caption_head"], default=None,
                         help="Gate 4 matching rule; overrides configs/run_defaults.yaml "
                              "episode.lexical_mode. 'prefix' = legacy rule used for all "
                              "results before 2026-09-28.")
    parser.add_argument("--anchor-weight", type=float, default=None,
                         help="Override configs/memory.yaml's contextual_anchor."
                              "anchor_weight for this run only -- added 2026-09-05 "
                              "so multi-anchor fusion (top-k=3, Gaussian spatial "
                              "falloff, Section 4) can finally be independently "
                              "ablated against w=0 (solo-argmax, the pre-fusion "
                              "behavior) rather than being live in every run with "
                              "no comparison point. See PROGRESS.md / paper "
                              "Limitations 'multi-anchor fusion is untuned'.")
    parser.add_argument("--inject-latency-s", type=float, default=0.0,
                         help="Additional synthetic delay (seconds), on top of "
                              "DAAAM's real grounding latency, before a pulled "
                              "update is released into the memory store -- added "
                              "2026-09-06 per critique.md 'asynchronous DAAAM "
                              "latency needs experimental validation'. 0.0 (the "
                              "default) is a no-op: updates release on the same "
                              "step they arrive, identical to pre-existing "
                              "behavior.")
    parser.add_argument("--inject-memory-age-s", type=float, default=0.0,
                         help="2026-09-07: synthetic time-skip (seconds) applied to "
                              "every already-remembered entity's last_observed at "
                              "each query BOUNDARY in --persistent-memory mode -- "
                              "simulates 'N seconds/hours passed since the previous "
                              "query' on top of whatever real elapsed time the "
                              "episode itself takes, so temporal decay (Eq. 1) can "
                              "be tested at timescales (hours) a live episode's own "
                              "~250s runtime could never reach on its own. Applied "
                              "once per boundary (episode_id>0), not continuously -- "
                              "a discrete jump, not a faster clock. 0.0 (the "
                              "default) is a no-op, identical to pre-existing "
                              "behavior. No effect when memory does not persist "
                              "across queries (fresh store every query has nothing "
                              "to age).")
    parser.add_argument("--anchor-sigma", type=float, default=None,
                         help="Override configs/memory.yaml's contextual_anchor."
                              "value_map_sigma (metres) for this run only -- added "
                              "2026-09-06 alongside --anchor-weight so the fusion "
                              "Gaussian's spatial falloff can be swept "
                              "independently, per critique.md's 'multi-anchor "
                              "fusion parameters are unvalidated' (sigma in "
                              "{2,4,6,8,10}m).")
    parser.add_argument("--run-tag", type=str, default=None,
                         help="Free-text suffix appended to the output directory "
                              "name, so repeated runs of an identical config (for "
                              "determinism/variance checks) or new sensitivity "
                              "sweeps (theta_direct, min-confirming-observations) "
                              "don't overwrite each other's batch_metrics.json.")
    # 2026-08-28: real-semantic-data + adversarial-negative test additions.
    parser.add_argument("--scene-path", type=str, default=None,
                         help="Override configs/habitat.yaml's scene_path for this "
                              "run only -- e.g. to use a scene with real HM3D "
                              "ground-truth semantic annotations (00770, the "
                              "default, has none; 00861-GLAQ4DNUx5U does).")
    parser.add_argument("--queries", type=str, default=None,
                         help="Comma-separated query list overriding the hardcoded "
                              "QUERIES for this run only, e.g. for queries drawn "
                              "from a scene's real semantic ground truth plus "
                              "adversarial (known-absent) negative controls.")
    # 2026-09-05: critique.md SS4/SS32 -- the original --persistent-memory flag
    # couples TWO separate things (does the MemoryStore carry over between
    # queries, and does the agent's physical position/env session carry over)
    # into one boolean, so "persistent memory helps" was confounded with
    # "not re-teleporting to spawn helps." These two new flags make each axis
    # independently selectable, completing the 2x2: --persistent-memory alone
    # is condition D (both persist); neither flag is condition A (both reset,
    # the ordinary default); --persist-memory-only is condition B (memory
    # persists, position resets each query -- the critique's own "key
    # comparison" against condition A, isolating memory persistence cleanly);
    # --persist-position-only is condition C (position persists, memory
    # resets each query, isolating physical-carryover-only benefit).
    parser.add_argument("--persist-memory-only", action="store_true",
                         help="Factorial persistence condition B: MemoryStore "
                              "carries over across queries in this session, but "
                              "the agent/environment resets to spawn every query "
                              "(unlike --persistent-memory, which also keeps "
                              "position). Do not combine with --persistent-memory "
                              "or --persist-position-only.")
    parser.add_argument("--persist-position-only", action="store_true",
                         help="Factorial persistence condition C: agent position "
                              "and the DAAAM session carry over across queries, "
                              "but the MemoryStore resets fresh every query "
                              "(unlike --persistent-memory, which also keeps "
                              "memory). Do not combine with --persistent-memory "
                              "or --persist-memory-only.")
    args = parser.parse_args()
    lexical_mode = args.lexical_mode or run_defaults.get("lexical_mode", "prefix")
    if sum([args.persistent_memory, args.persist_memory_only, args.persist_position_only]) > 1:
        parser.error("--persistent-memory, --persist-memory-only, and "
                      "--persist-position-only are mutually exclusive -- pick the "
                      "one factorial condition this run tests.")

    # configs/run_defaults.yaml's `ablations:` section sets the default for each
    # flag; the matching CLI flag can additionally force one off (it can't force
    # one back on over a config that already says false -- see the yaml comment).
    memory_enabled = ablation_cfg["memory_enabled"] and not args.memory_off
    mobility_scaling_enabled = ablation_cfg["mobility_scaling_enabled"] and not args.no_mobility_scaling
    verification_discount_enabled = ablation_cfg["verification_discount_enabled"] and not args.no_verification_discount
    print(f"ablations: memory_enabled={memory_enabled} "
          f"mobility_scaling_enabled={mobility_scaling_enabled} "
          f"verification_discount_enabled={verification_discount_enabled}")

    persist_memory = persistent_default or args.persistent_memory or args.persist_memory_only
    persist_position = persistent_default or args.persistent_memory or args.persist_position_only
    # kept for backward compat: this name/semantics (True only when BOTH axes
    # persist, i.e. condition D) is what every archived batch_metrics.json and
    # the paper's own tables already key off of.
    persistent_across_queries = persist_memory and persist_position
    print(f"persist_memory={persist_memory} persist_position={persist_position} "
          f"(persistent_across_queries={persistent_across_queries})")
    base_queries = [q.strip() for q in args.queries.split(",")] if args.queries is not None else QUERIES
    queries = base_queries[:args.n_queries] if args.n_queries is not None else base_queries

    ablation_suffixes = []
    if not memory_enabled:
        ablation_suffixes.append("memory_off")
    if not mobility_scaling_enabled:
        ablation_suffixes.append("no_mobility")
    if not verification_discount_enabled:
        ablation_suffixes.append("no_verify_discount")
    if persist_memory and persist_position:
        ablation_suffixes.append("persistent")
    elif persist_memory:
        ablation_suffixes.append("persist_memory_only")
    elif persist_position:
        ablation_suffixes.append("persist_position_only")
    if args.theta_direct is not None:
        ablation_suffixes.append(f"theta{args.theta_direct}")
    if args.min_confirming_observations != 1:
        ablation_suffixes.append(f"minconf{args.min_confirming_observations}")
    if args.vqa_verification:
        ablation_suffixes.append("vqa")
    if args.no_coverage_memory:
        ablation_suffixes.append("no_coverage_memory")
    if args.anchor_weight is not None:
        ablation_suffixes.append(f"anchorw{args.anchor_weight}")
    if args.anchor_sigma is not None:
        ablation_suffixes.append(f"anchorsigma{args.anchor_sigma}")
    if args.no_fresh_evidence_guard:
        ablation_suffixes.append("no_fresh_evidence_guard")
    if args.no_relevance_gate:
        ablation_suffixes.append("no_relevance_gate")
    if args.no_lexical_gate:
        ablation_suffixes.append("no_lexical_gate")
    if args.lambda_base is not None:
        ablation_suffixes.append(f"lambda{args.lambda_base}")
    if args.lexical_mode is not None:
        ablation_suffixes.append(f"lex{args.lexical_mode}")
    if args.run_tag:
        ablation_suffixes.append(args.run_tag)
    out_dir = Path(
        "data/runs/phase4_batch_" + "_".join(ablation_suffixes) if ablation_suffixes
        else "data/runs/phase4_batch"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/habitat.yaml") as f:
        habitat_cfg = yaml.safe_load(f)
    if args.scene_path is not None:
        print(f"overriding scene_path: {habitat_cfg['scene_path']} -> {args.scene_path}")
        habitat_cfg["scene_path"] = args.scene_path
    with open("configs/memory.yaml") as f:
        mem_cfg = yaml.safe_load(f)
    if args.theta_direct is not None:
        print(f"overriding theta_direct: {mem_cfg['recall']['theta_direct']} -> {args.theta_direct}")
        mem_cfg["recall"]["theta_direct"] = args.theta_direct
    print(f"min_confirming_observations={args.min_confirming_observations}")

    client = DaaamBridgeClient()
    print("health:", client.health())

    mobility_priors = MobilityPriors.load("configs/mobility_priors.json")
    encoder = ClipTextEncoder(mem_cfg["embedding"]["clip_model_name"], mem_cfg["embedding"]["clip_model_dataset"])
    # pipeline-v8 switches (all off by default)
    V8["fuse_positions"] = args.fuse_positions
    if args.revisit_evidence:
        from memory_layer.revisit import RevisitEvidence
        from frontier_agent.verification import _caption_subject_nouns, _singular
        from memory_layer.relevance import cosine_similarity

        def _nouns(e):
            return {_singular(p[-1].split("-")[-1]) for p in _caption_subject_nouns(e.description or "") if p}

        def same_kind(a, b):
            na, nb = _nouns(a), _nouns(b)
            if na and nb:
                return bool(na & nb)
            # no parsable subject noun: fall back to caption-embedding similarity (0.95 ~ 95th pct of
            # different-kind pairs on the HM3D tuning scene 00770 memory)
            return (a.embedding is not None and b.embedding is not None
                    and cosine_similarity(a.embedding, b.embedding) >= 0.95)

        V8["revisit"] = RevisitEvidence(
            discount=mem_cfg["reliability"]["verification_failure_discount"], same_kind=same_kind,
            grace_s=args.revisit_grace_s)
    if args.image_check:
        from frontier_agent.image_check import ImageChecker
        V8["image_checker"] = ImageChecker(encoder, top_k=args.image_check_topk)
    print(f"pipeline-v8: fuse_positions={args.fuse_positions} revisit_evidence={args.revisit_evidence} "
          f"image_check={args.image_check}")

    def make_agent(env):
        return FrontierAgent(
            pathfinder=env.pathfinder, clip_encoder=encoder, mobility_priors=mobility_priors,
            alpha=mem_cfg["reliability"]["alpha"],
            lambda_base=(args.lambda_base if args.lambda_base is not None
                         else mem_cfg["reliability"]["lambda_base"]),
            theta_direct=mem_cfg["recall"]["theta_direct"],
            anchor_weight=(args.anchor_weight if args.anchor_weight is not None
                           else mem_cfg["contextual_anchor"]["anchor_weight"]),
            value_map_sigma=(args.anchor_sigma if args.anchor_sigma is not None
                              else mem_cfg["contextual_anchor"]["value_map_sigma"]),
            n_frontier_candidates=run_defaults["n_frontier_candidates"],
            periodic_replan_interval=run_defaults["periodic_replan_interval"],
            memory_enabled=memory_enabled,
            mobility_scaling_enabled=mobility_scaling_enabled,
            goal_lexical_gate=args.goal_lexical_gate, arrival_fix=args.arrival_fix,
            theta_gates_goals=args.theta_gates_goals,
            **({"scan_steps": args.scan_steps} if args.scan_steps is not None else {}),
        )

    def inject_memory_age(store, age_s):
        """2026-09-07: simulate 'age_s seconds passed' since the previous query by
        jumping every remembered entity's last_observed backward once, at a query
        boundary -- see --inject-memory-age-s help text above. No-op if age_s<=0
        or store is None (nothing to age)."""
        if store is None or age_s <= 0:
            return
        for entity in store.entities.values():
            entity.last_observed -= age_s

    def make_store():
        return MemoryStore(
            initial_v=mem_cfg["reliability"]["initial_v"],
            # no-verification-discount ablation: discount=1.0 means v_i *= 1.0 on
            # failure, i.e. v_i never changes from initial_v -- see
            # MemoryEntity.apply_verification.
            verification_failure_discount=(
                mem_cfg["reliability"]["verification_failure_discount"]
                if verification_discount_enabled else 1.0
            ),
        )

    results = []
    daaam_total_corrections = None

    # 2026-09-05: unified loop parameterized independently by persist_position
    # (does the DAAAM session/env/agent-position carry over between queries)
    # and persist_memory (does the MemoryStore carry over) -- replaces the old
    # two-branch persistent_across_queries split, which could only ever
    # express "both persist" or "neither persists" (conditions D and A),
    # never conditions B or C. See critique.md SS4/SS32.
    if persist_position:
        # One continuous DAAAM session/env/agent for the whole loop -- reset_env
        # only on the very first query -- regardless of whether memory itself
        # persists (that's store's own lifecycle, handled per-iteration below).
        client.start_episode()
        env = HabitatEnv(habitat_cfg)
        env._config = habitat_cfg
        agent = make_agent(env)
        store = make_store() if persist_memory else None
        for episode_id, query in enumerate(queries):
            if not persist_memory:
                store = make_store()  # condition C: fresh memory every query
            elif episode_id > 0:
                inject_memory_age(store, args.inject_memory_age_s)
            label = "persistent" if persist_memory else "position-only-persistent"
            print(f"\n=== Query {episode_id} ({label} session): query='{query}' ===")
            result = run_episode(
                episode_id, query, client, env, agent, encoder, store,
                max_steps=args.max_steps, log_rerun=not args.no_rrd,
                coverage_threshold=run_defaults["coverage_threshold"],
                verify_relevance_threshold=run_defaults["verify_relevance_threshold"],
                verify_distance=(args.verify_distance if args.verify_distance is not None else run_defaults["verify_distance"]),
                post_loop_grounding_wait_s=run_defaults["post_loop_grounding_wait_s"],
                out_dir=str(out_dir),
                reset_env=(episode_id == 0),
                close_env=(episode_id == len(queries) - 1),
                min_confirming_observations=args.min_confirming_observations,
                vqa_verification=args.vqa_verification,
                coverage_memory_enabled=not args.no_coverage_memory,
                disable_lexical_gate=args.no_lexical_gate,
                disable_fresh_evidence_guard=args.no_fresh_evidence_guard,
                verification_feedback=args.verification_feedback,
                local_verification=args.local_verification,
                live_verification=args.live_verification,
                disable_relevance_gate=args.no_relevance_gate,
                inject_latency_s=args.inject_latency_s,
                lexical_mode=lexical_mode,
                instance_target=(json.loads(args.instance_target) if args.instance_target else None),
            )
            print(f"outcome={result['outcome']} reason={result['reason']} steps={result['steps_taken']} "
                  f"spl={result['spl']} branches={result['branch_counts']} "
                  f"memory_size_at_end={result['memory_size_at_end']}")
            result["daaam_final_corrections"] = None  # only known once the shared session stops, below
            results.append(result)
        stop_resp = client.stop_episode()
        daaam_total_corrections = len(stop_resp.get("corrections", []))
    else:
        store = make_store() if persist_memory else None
        if persist_memory and args.init_memory_snapshot:
            # start from a saved memory (memory_snapshot_ep*.json with embeddings);
            # IDs offset so they can never collide with this session's DAAAM IDs
            from memory_layer.models import MemoryEntity
            snap = json.load(open(args.init_memory_snapshot))
            for e in snap["entities"]:
                if e.get("embedding") is None:
                    continue
                store.entities[900_000_000 + e["id"]] = MemoryEntity(
                    entity_id=900_000_000 + e["id"], category=e["category"], description=e["description"],
                    embedding=e["embedding"], position_world=e["position_world"], c_i=e.get("c_i", 1.0),
                    n_i=e["n_i"], v_i=e["v_i"], first_observed=e["first_observed"], last_observed=e["last_observed"])
            print(f"loaded {len(store.entities)} remembered objects from {args.init_memory_snapshot}")
        for episode_id, query in enumerate(queries):
            if not persist_memory:
                store = make_store()  # condition A: fresh memory every query
            else:
                # condition B -- reuse the store created at episode_id==0
                # across every fresh env/agent below, even though position resets.
                if episode_id > 0:
                    inject_memory_age(store, args.inject_memory_age_s)
            label = "memory-only-persistent" if persist_memory else "reset"
            print(f"\n=== Episode {episode_id} ({label}): query='{query}' ===")
            client.start_episode()
            env = HabitatEnv(habitat_cfg)
            env._config = habitat_cfg  # for camera_intrinsics inside run_episode

            agent = make_agent(env)

            result = run_episode(
                episode_id, query, client, env, agent, encoder, store,
                max_steps=(args.tour_max_steps if args.tour_waypoints and episode_id == 0 else args.max_steps), log_rerun=not args.no_rrd,
                coverage_threshold=run_defaults["coverage_threshold"],
                verify_relevance_threshold=run_defaults["verify_relevance_threshold"],
                verify_distance=(args.verify_distance if args.verify_distance is not None else run_defaults["verify_distance"]),
                post_loop_grounding_wait_s=run_defaults["post_loop_grounding_wait_s"],
                out_dir=str(out_dir),
                id_offset=(episode_id * 1_000_000 if persist_memory else 0),  # new DAAAM session per query
                tour=(json.load(open(args.tour_waypoints)) if args.tour_waypoints and episode_id == 0 else None),
                min_confirming_observations=args.min_confirming_observations,
                vqa_verification=args.vqa_verification,
                coverage_memory_enabled=not args.no_coverage_memory,
                disable_lexical_gate=args.no_lexical_gate,
                disable_fresh_evidence_guard=args.no_fresh_evidence_guard,
                verification_feedback=args.verification_feedback,
                local_verification=args.local_verification,
                live_verification=args.live_verification,
                disable_relevance_gate=args.no_relevance_gate,
                inject_latency_s=args.inject_latency_s,
                lexical_mode=lexical_mode,
                instance_target=(json.loads(args.instance_target) if args.instance_target else None),
            )
            print(f"outcome={result['outcome']} reason={result['reason']} steps={result['steps_taken']} "
                  f"spl={result['spl']} branches={result['branch_counts']}")

            stop_resp = client.stop_episode()
            result["daaam_final_corrections"] = len(stop_resp.get("corrections", []))
            results.append(result)

    success_rate = sum(1 for r in results if r["outcome"] == "success") / len(results)
    mean_spl = sum(r["spl"] for r in results) / len(results)
    mean_wall_clock_seconds = sum(r["wall_clock_seconds"] for r in results) / len(results)
    # Wall-clock among successes only -- "how long did a real find actually take"
    # vs. mean_wall_clock_seconds, which is dominated by max_steps_exhausted
    # failures running the full step budget every time.
    success_wall_clock = [r["wall_clock_seconds"] for r in results if r["outcome"] == "success"]
    mean_wall_clock_seconds_on_success = (
        sum(success_wall_clock) / len(success_wall_clock) if success_wall_clock else None
    )

    # critique.md SS3/SS13/SS39: batch-level helpful-influence rate, computed
    # over episodes where memory was actually influential at least once (not
    # over all episodes -- "not_applicable" episodes had nothing to be helpful
    # or unhelpful about). Reported alongside the raw classification counts so
    # the denominator is never hidden -- see SS13's "do not average percentages
    # without explaining the weighting."
    influence_classifications = [r["memory_influence_classification"] for r in results]
    n_influential_episodes = sum(1 for c in influence_classifications if c != "not_applicable")
    n_helpful_episodes = sum(1 for c in influence_classifications if c == "helpful")
    helpful_influence_rate = (n_helpful_episodes / n_influential_episodes) if n_influential_episodes else None

    # critique.md SS8: direct-recall precision, same "not_applicable" denominator
    # convention as the memory-influence rate above -- only episodes where the
    # branch fired at least once count toward precision.
    direct_recall_classifications = [r["direct_recall_classification"] for r in results]
    n_direct_recall_fired = sum(1 for c in direct_recall_classifications if c != "not_applicable")
    n_direct_recall_correct = sum(1 for c in direct_recall_classifications if c == "correct")
    direct_recall_precision = (
        (n_direct_recall_correct / n_direct_recall_fired) if n_direct_recall_fired else None
    )

    metrics = {
        "n_episodes": len(results), "success_rate": success_rate, "mean_spl": mean_spl,
        "mean_wall_clock_seconds": round(mean_wall_clock_seconds, 2),
        "mean_wall_clock_seconds_on_success": (
            round(mean_wall_clock_seconds_on_success, 2) if mean_wall_clock_seconds_on_success is not None else None
        ),
        "n_episodes_memory_influential": n_influential_episodes,
        "n_episodes_memory_helpful": n_helpful_episodes,
        "helpful_influence_rate": helpful_influence_rate,
        "n_episodes_direct_recall_fired": n_direct_recall_fired,
        "n_episodes_direct_recall_correct": n_direct_recall_correct,
        "direct_recall_precision": direct_recall_precision,
        "persistent_across_queries": persistent_across_queries,  # True only for condition D (both axes) -- kept for backward-compat with existing archived runs/analysis
        "persist_memory": persist_memory,
        "persist_position": persist_position,
        "theta_direct_used": mem_cfg["recall"]["theta_direct"],
        "min_confirming_observations": args.min_confirming_observations,
        "vqa_verification": args.vqa_verification,
        "coverage_memory_enabled": not args.no_coverage_memory,
        "anchor_weight_used": (args.anchor_weight if args.anchor_weight is not None
                                else mem_cfg["contextual_anchor"]["anchor_weight"]),
        "anchor_sigma_used": (args.anchor_sigma if args.anchor_sigma is not None
                               else mem_cfg["contextual_anchor"]["value_map_sigma"]),
        "inject_latency_s": args.inject_latency_s,
        "inject_memory_age_s": args.inject_memory_age_s,
        "fresh_evidence_guard_enabled": not args.no_fresh_evidence_guard,
        "verification_feedback": args.verification_feedback,
        "goal_lexical_gate": args.goal_lexical_gate, "arrival_fix": args.arrival_fix,
        "theta_gates_goals": args.theta_gates_goals, "local_verification": args.local_verification,
        "live_verification": args.live_verification,
        "fuse_positions": args.fuse_positions, "revisit_evidence": args.revisit_evidence,
        "revisit_grace_s": args.revisit_grace_s, "image_check": args.image_check, "image_check_topk": args.image_check_topk,
        "relevance_gate_enabled": not args.no_relevance_gate,
        "lexical_gate_enabled": not args.no_lexical_gate,
        "lexical_mode": lexical_mode,
        "init_memory_snapshot": args.init_memory_snapshot,
        "daaam_id_offset_fix": True,  # 2026-09-29: per-session ID offset in memory-only mode
        "scene_path": habitat_cfg["scene_path"],
        "queries_used": queries,
        "daaam_total_corrections": daaam_total_corrections,
        "lambda_base_used": (args.lambda_base if args.lambda_base is not None
                             else mem_cfg["reliability"]["lambda_base"]),
        "provenance": _provenance(),
        "resolved_configs": {"run_defaults": run_defaults_full, "memory": mem_cfg},
        "episodes": results,
    }
    with open(out_dir / "batch_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # 2026-09-28: uncommitted code is common in this repo, so the commit hash
    # alone cannot reproduce a run -- save the exact diff next to the metrics.
    try:
        import subprocess
        diff = subprocess.run(["git", "diff", "HEAD", "--", "packages", "configs", "scripts"],
                              capture_output=True, text=True, timeout=30).stdout
        (out_dir / "code_diff_vs_head.patch").write_text(diff)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not save code diff: {exc}")

    print(f"\n=== Batch summary: {len(results)} episodes, success_rate={success_rate:.2f}, "
          f"mean_spl={mean_spl:.4f}, mean_wall_clock_seconds={mean_wall_clock_seconds:.1f} ===")
    print(f"saved metrics to {out_dir}/batch_metrics.json")


if __name__ == "__main__":
    main()
