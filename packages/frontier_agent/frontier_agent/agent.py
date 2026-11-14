"""FrontierAgent: one decision step of the closed loop, per CLAUDE.md Section 6/7 --
direct-recall / contextual-anchor-search / fallback branches, producing a Habitat
action. Memory mutation (M_{t+1}=Update(...)) is the CALLER's responsibility
(memory_layer.store), not this class's -- decide() only reads memory_store's
current state and scores/selects an action, keeping the two concerns separate.

2026-08-24 navigation redesign: previously, "contextual anchor search" meant
sampling ~12 random nearby navmesh points each replan and scoring them by a fused
Gaussian value map summed over every live/remembered entity in range (see
value_map.py/scoring.py/frontier.py, still present but no longer used by decide()
below). Two structural problems came from that: (1) choose_action was purely
bearing-based with no notion of walls, so the agent could get physically stuck
walking into geometry for hundreds of consecutive steps; (2) summing Gaussian
pulls from every remembered entity meant already-explored, entity-dense areas only
ever pulled harder, while genuinely unexplored space always scored exactly 0.0 --
a "gravity well" that trapped the agent near wherever it had already been,
regardless of the actual query target.

Both are fixed by construction with the design here: decide() picks ONE target
(a live/remembered entity for direct-recall or contextual-anchor, or an
unexplored point for fallback), computes a REAL navmesh-constrained path to it
via pathing.compute_path (Habitat's own A*, the same API run_phase4_batch.py
already used to score SPL after the fact, now used to actually drive movement),
and walks that path waypoint by waypoint. A real path respects walls by
construction. See PROGRESS.md 2026-08-24.

2026-09-02 multi-anchor fusion: _select_target no longer always takes the
single highest-scoring entity outright. It gathers the top `top_k_anchors`
candidates by r_i, then re-ranks that small set by a fused value rewarding
proximity to OTHER good candidates (value_map.py's Gaussian-falloff math,
reused -- previously inert dead code after the 2026-08-24 redesign dropped it
entirely). The candidate SET stays bounded to top-K, and the agent still walks
a real A* path to exactly one entity's standoff pose -- this does not sum over
the whole memory store the way the pre-08-24 design did, so it does not
reintroduce the gravity well. The goal is a cheaper failure mode when the #1
solo-scoring candidate turns out to be wrong: if it sits near other plausible
candidates, the agent is already in a useful area rather than committed to one
specific far-off point with nothing nearby to fall back on. With
anchor_weight=0 this reduces to the pre-09-02 solo-argmax behaviour exactly.
See PROGRESS.md 2026-09-02 for the motivating evidence (episodes where the
right answer was in memory but never reached) and the validation run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from memory_layer.models import MemoryEntity
from memory_layer.reliability import compute_reliability
from memory_layer.relevance import compute_relevance
from memory_layer.store import MemoryStore

from .frontier import sample_frontier_candidates
from .heading import choose_action
from .pathing import compute_path
from .standoff import find_standoff_pose
from .value_map import euclidean, gaussian_falloff
from .verification import _lexically_consistent


@dataclass
class ActionDecision:
    action: str  # move_forward / turn_left / turn_right / stop
    branch: str  # "direct_recall" / "contextual_anchor" / "fallback"
    goal_position: Optional[List[float]]  # final destination, not the current waypoint
    goal_entity_id: Optional[int] = None
    # Did including REMEMBERED (past, non-live) entities change which target was
    # selected, vs. selecting from only this step's live entities? Zero-extra-cost
    # causal probe for "did persistent memory actually change this decision" --
    # see PROGRESS.md Phase 4 "does memory help" investigation. None only when
    # neither pool found any candidate at all (nothing to compare).
    memory_influenced: Optional[bool] = None
    best_value_with_memory: Optional[float] = None
    best_value_without_memory: Optional[float] = None
    live_signal_count: Optional[int] = None
    remembered_signal_count: Optional[int] = None
    # 2026-09-28 (Post-Paper Test Plan 2.2, helpful/harmful influence):
    # logging-only fields, never read by the controller. Target the WITH-memory
    # selection pointed at vs. the live-only counterfactual selection.
    memory_target_entity_id: Optional[int] = None
    memory_target_position: Optional[List[float]] = None
    live_target_entity_id: Optional[int] = None
    live_target_position: Optional[List[float]] = None


class FrontierAgent:
    def __init__(
        self,
        pathfinder,
        clip_encoder,
        mobility_priors,
        alpha: float,
        lambda_base: float,
        theta_direct: float,
        anchor_weight: float,
        value_map_sigma: float = 2.0,
        n_frontier_candidates: int = 12,
        standoff_max_snap_distance: float = 2.0,
        goal_reached_distance: float = 0.5,
        periodic_replan_interval: int = 10,
        scan_steps: int = 6,
        exploration_radius: float = 3.0,
        memory_enabled: bool = True,
        mobility_scaling_enabled: bool = True,
        top_k_anchors: int = 3,
        goal_lexical_gate: bool = False,
        arrival_fix: bool = False,
        theta_gates_goals: bool = False,
    ):
        # 2026-10-30 (both off by default; THOR moved-object test): goal_lexical_gate applies verification's own
        # head-noun check to goal selection, so the agent only walks to entries that could ever verify (CLIP gives
        # unrelated objects ~0.2 relevance, so ungated goals visit trash cans and cushions for a sofa query until the
        # budget runs out). arrival_fix counts a re-plan onto the goal the agent is already standing at as arrival
        # (scan, then exclude); otherwise a stream of new captions forces re-plans that never register arrival and
        # the agent re-picks the same entity in place indefinitely.
        self.goal_lexical_gate = goal_lexical_gate
        self.arrival_fix = arrival_fix
        self._stuck_resets: dict = {}  # arrival_fix: goal key -> stuck-reset count
        # theta_gates_goals: an entry with R_i < theta_direct is not taken as a goal at all (the agent explores
        # instead). Without it theta_direct only labels the branch, and a decayed entry is still walked to whenever
        # nothing better exists, so decay cannot change behaviour.
        self.theta_gates_goals = theta_gates_goals
        self._current_query: Optional[str] = None
        # CLAUDE.md Section 9 Step 4.5's "no mobility scaling: flat lambda across
        # all categories" ablation: when False, _score_entity uses mobility=1.0 for
        # every entity instead of mobility_priors.mobility_for(category), so
        # lambda_i = lambda_base * 1.0 = lambda_base uniformly -- the same
        # lambda_base as the non-ablated run, just with the per-category multiplier
        # removed rather than swapped for a new hyperparameter.
        self.mobility_scaling_enabled = mobility_scaling_enabled
        # CLAUDE.md Section 9 Step 4.5's "memory-off (live-only, fallback always)"
        # ablation: when False, target selection only ever considers entities in
        # live_entity_ids (this step's fresh DAAAM update), never anything
        # remembered from an earlier step. Persisted entities still accumulate in
        # the MemoryStore (unaffected -- that's the caller's concern, not this
        # class's) but decide() never navigates toward them. Since DAAAM's
        # grounding is asynchronous and lags observation (Section 5.2), most steps
        # have no live entities at all, so this ablation spends most of its time in
        # "fallback", occasionally taking "contextual_anchor" only when something
        # happens to be captioned in the very same step -- "direct_recall" is
        # reachable too if a live entity is strong enough, but in practice a single
        # observation rarely clears theta_direct. See PROGRESS.md 2026-08-24.
        self.memory_enabled = memory_enabled
        self.pathfinder = pathfinder
        self.clip_encoder = clip_encoder
        self.mobility_priors = mobility_priors
        self.alpha = alpha
        self.lambda_base = lambda_base
        self.theta_direct = theta_direct
        # 2026-09-02: anchor_weight/value_map_sigma are used again -- see
        # _select_target's multi-anchor fusion (module docstring, "Multi-anchor
        # fusion" section). Previously inert here since the 2026-08-24 redesign
        # dropped value_map.py's ValueMap entirely; that was too blunt a fix --
        # the real problem was summing over EVERY entity unconditionally (the
        # gravity well), not the idea of spatial fusion itself. anchor_weight
        # keeps its original meaning: how much a candidate's proximity to OTHER
        # good anchors (vs. its own solo score) should influence which top-K
        # candidate gets chosen.
        self.anchor_weight = anchor_weight
        self.value_map_sigma = value_map_sigma
        self.top_k_anchors = top_k_anchors
        self.n_frontier_candidates = n_frontier_candidates
        self.standoff_max_snap_distance = standoff_max_snap_distance
        self.goal_reached_distance = goal_reached_distance

        # Path-following state: the real waypoint path currently being walked (see
        # module docstring). _path_final_goal is reported as ActionDecision's
        # goal_position on every step of a given path (not the intermediate
        # waypoint), matching the pre-redesign external contract.
        self._current_path: Optional[List[List[float]]] = None
        self._path_index: int = 0
        self._path_final_goal: Optional[List[float]] = None
        self._path_branch: Optional[str] = None
        self._path_target_entity_id: Optional[int] = None

        # An entity the agent already walked to and did a depth-search scan at,
        # without verification catching it, is excluded from future target
        # selection for the rest of the episode -- otherwise it would just get
        # re-picked (still the best-scoring remembered match) and the agent would
        # loop back to the same spot forever. No expiry/un-exclusion implemented;
        # documented limitation, not expected to matter within one episode's budget.
        self._excluded_entity_ids: Set[int] = set()

        # "Depth search": once the agent arrives at a real target (direct-recall or
        # contextual-anchor), spend a few steps turning in place before moving on,
        # so verification.py's per-step check (which runs regardless of navigation
        # branch) gets a fair chance to actually see the target, rather than the
        # agent immediately walking off toward the next thing the instant it's
        # technically "arrived." Not run for fallback-point arrivals (nothing
        # specific is expected to be there).
        #
        # 2026-09-02: tried raising this 6 -> 60 to let DAAAM's async grounding
        # (average caption-arrival lag 12.90s, max 21.22s, n=64) actually land
        # before the agent gives up on a candidate -- real motivation
        # (best_relevance_in_memory repeatedly pointed at the right entity while
        # episodes still failed), but tested rather than assumed, and REJECTED:
        # a real 4-batch re-run showed a net regression, 50%->32% genuine-match
        # rate combined, concentrated almost entirely in persistent mode (5/8->2/8,
        # 3/10->1/10) where the larger accumulated memory pool makes wrong-
        # candidate visits far more common, and each one now costs 10x the step
        # budget -- later persistent queries ran out of their 500-step budget on
        # costly wrong guesses before ever reaching the right one. Non-persistent
        # mode was roughly a wash (some genuine wins, e.g. "fireplace" -- a
        # category DAAAM's classifier handles poorly -- succeeded for the first
        # time anywhere in this project, offset by other regressions). Reverted
        # to 6; see PROGRESS.md for the full per-episode before/after. The
        # underlying idea (verification is time-starved relative to grounding
        # latency) is still believed correct -- what's rejected is "spend more
        # wall-clock time per candidate" as the fix, since it doesn't account for
        # how many candidates an episode needs to get through. A budget-aware or
        # pool-size-aware scan duration, or a genuinely faster/synchronous
        # re-grounding call scoped to just the current candidate, are more
        # promising follow-ups than a flat constant.
        self.scan_steps = scan_steps
        self._scan_steps_remaining: int = 0

        # Re-plan safety net (PROGRESS.md Phase 4, 2026-08-21/23): re-plan whenever
        # fresh live evidence arrives, whenever the count of USABLE
        # (positioned+embedded) entities in memory has grown since the last plan,
        # or -- for non-fallback paths only -- every periodic_replan_interval steps
        # regardless of timing. Fallback paths are exempt from the periodic trigger
        # (2026-08-24): nothing memory-related can go stale when there's no memory
        # signal behind the goal in the first place, and cutting fallback paths off
        # early left the agent unable to ever walk far enough to explore.
        self.periodic_replan_interval = periodic_replan_interval
        self._last_usable_entity_count = 0
        self._steps_since_replan = 0

        # Stuck/collision-recovery (2026-08-24): kept as a safety net even with
        # real pathfinding -- navmesh snapping imprecision or a moving-target
        # standoff pose could still rarely produce a waypoint the agent can't quite
        # reach. If move_forward produces no real displacement for stuck_threshold
        # consecutive calls, abandon the current path and force a fresh plan.
        self._last_position: Optional[List[float]] = None
        self._last_action: Optional[str] = None
        self._stuck_move_forward_count: int = 0
        self.stuck_displacement_epsilon = 0.05  # metres; less than one 0.25m step
        self.stuck_threshold = 3  # consecutive no-progress move_forward calls

        # Visited-position history, used ONLY by fallback point selection now (see
        # _pick_fallback_point) to avoid re-picking points near where the agent has
        # already been -- the entity-driven "gravity well" this originally fixed
        # (PROGRESS.md 2026-08-24) can no longer occur at all under single-target
        # real-path navigation, since there's no summed multi-entity value map left
        # to pull the agent back into a dense already-explored area.
        self._visited_positions: List[List[float]] = []
        self.exploration_radius = exploration_radius

        # Query text -> embedding cache. Keyed on the query string (not cleared by
        # reset()) since the embedding is a pure function of the text -- re-running
        # the same query in a later episode should still hit the cache.
        self._query_embedding_cache: dict = {}

    def reset(self, persist_coverage: bool = False) -> None:
        """Clears persisted state -- call at the start of each new episode so state
        from a previous episode never leaks into a fresh one. Does NOT clear the
        query-embedding cache (see __init__ comment).

        2026-09-02 coverage memory: `_visited_positions` -- "roughly where has the
        agent already physically been" -- used to be cleared unconditionally here,
        meaning even in persistent-across-queries mode the agent's spatial
        familiarity reset every query while its object memory (the entity list,
        a separate object entirely, never touched by this method) did not. That
        mismatch is a real, direct analogue of a human re-entering a room they
        already searched five minutes ago and not remembering they'd looked in
        the corner already. `_excluded_entity_ids` (candidates already tried and
        failed) deliberately still resets every query regardless -- an entity
        excluded because it didn't match query 1 says nothing about query 2, so
        that state is genuinely per-query, unlike raw spatial coverage.
        persist_coverage=True keeps _visited_positions across this reset (pass
        the same signal that already decides whether the memory store and agent
        position carry over -- see run_phase4_batch.py's reset_env)."""
        self._current_path = None
        self._path_index = 0
        self._path_final_goal = None
        self._path_branch = None
        self._path_target_entity_id = None
        self._excluded_entity_ids = set()
        self._scan_steps_remaining = 0
        self._last_usable_entity_count = 0
        self._steps_since_replan = 0
        self._last_position = None
        self._last_action = None
        self._stuck_move_forward_count = 0
        if not persist_coverage:
            self._visited_positions = []

    def _embed_query(self, query: str) -> List[float]:
        if query not in self._query_embedding_cache:
            self._query_embedding_cache[query] = self.clip_encoder.embed(query).tolist()
        return self._query_embedding_cache[query]

    def _track(self, agent_position: List[float], decision: "ActionDecision") -> "ActionDecision":
        """Records what happened this call so the next call can detect a stuck
        (blocked-but-still-issuing-move_forward) agent. See __init__ comment."""
        self._last_position = agent_position
        self._last_action = decision.action
        return decision

    def _score_entity(self, entity, query_embedding, now):
        mobility = self.mobility_priors.mobility_for(entity.category) if self.mobility_scaling_enabled else 1.0
        reliability = compute_reliability(entity, now, mobility, self.alpha, self.lambda_base)
        relevance = compute_relevance(query_embedding, entity.embedding)
        return reliability, relevance

    def _select_target(
        self,
        memory_store: MemoryStore,
        query_embedding: List[float],
        now: float,
        entity_ids_pool: Optional[Set[int]] = None,
    ) -> Optional[Tuple[MemoryEntity, List[float], float]]:
        """Multi-anchor selection among the top-K scoring (relevance*reliability),
        reachable, non-excluded entities. entity_ids_pool restricts consideration
        to that id set (used by the memory-influence probe to score live-only).
        Returns (entity, standoff_pose, r_i) or None if nothing qualifies.

        2026-09-02 multi-anchor fusion: rather than always taking the single
        highest-r_i entity outright, gather the top `top_k_anchors` candidates by
        r_i, then re-rank them by a fused value that also rewards proximity to
        OTHER good candidates -- fused(c) = r_i(c) + anchor_weight * sum over the
        REST of the top-K of r_i(other) * gaussian_falloff(dist(c, other),
        value_map_sigma). A candidate sitting near several other plausible
        matches (e.g. two "window" entities in the same room) is preferred over
        a solo highest-scorer sitting far from any support -- reusing
        value_map.py's own Gaussian-falloff math on a bounded top-K set, not
        summing over the whole memory store, so this does not reintroduce the
        2026-08-24 gravity well (see module docstring): the candidate SET is
        still exactly one one of the top-K entities' own standoff poses, walked
        to via a real A* path exactly as before, just chosen with more context
        than "which single entity scored highest in isolation." With
        anchor_weight=0 this is exactly the old solo-argmax behaviour."""
        scored = []  # (r_i, entity, standoff, position_xy)
        for entity in memory_store.entities.values():
            if entity.entity_id in self._excluded_entity_ids:
                continue
            if entity.position_world is None:
                continue
            if entity_ids_pool is not None and entity.entity_id not in entity_ids_pool:
                continue
            if self.goal_lexical_gate and self._current_query and not _lexically_consistent(self._current_query, entity):
                continue
            reliability, relevance = self._score_entity(entity, query_embedding, now)
            r_i = relevance * reliability
            if r_i <= 0.0:
                continue
            if self.theta_gates_goals and r_i < self.theta_direct:
                continue
            standoff = find_standoff_pose(self.pathfinder, entity.position_world, self.standoff_max_snap_distance)
            if standoff is None:
                continue
            position_xy = [entity.position_world[0], entity.position_world[2]]
            scored.append((r_i, entity, standoff, position_xy))
        if not scored:
            return None

        scored.sort(key=lambda t: t[0], reverse=True)
        top_k = scored[: self.top_k_anchors]
        if len(top_k) == 1 or self.anchor_weight == 0.0:
            r_i, entity, standoff, _ = top_k[0]
            return (entity, standoff, r_i)

        best = None
        for i, (r_i, entity, standoff, position_xy) in enumerate(top_k):
            support = sum(
                other_r_i * gaussian_falloff(euclidean(position_xy, other_xy), self.value_map_sigma)
                for j, (other_r_i, _, _, other_xy) in enumerate(top_k)
                if j != i
            )
            fused = r_i + self.anchor_weight * support
            if best is None or fused > best[0]:
                best = (fused, entity, standoff, r_i)
        _, entity, standoff, r_i = best
        return (entity, standoff, r_i)

    def _pick_fallback_point(self, agent_position: List[float]) -> Optional[List[float]]:
        """No usable memory target exists -- pick somewhere to explore. Prefers a
        freshly-sampled navmesh point outside exploration_radius of anywhere the
        agent has already planned a path from; falls back to the farthest sampled
        point if every candidate is already-explored territory."""
        candidates = sample_frontier_candidates(self.pathfinder, agent_position, self.n_frontier_candidates)
        if not candidates:
            return None
        agent_xy = [agent_position[0], agent_position[2]]

        def is_fresh(c) -> bool:
            cxy = [c.position_xyz[0], c.position_xyz[2]]
            return not any(euclidean(cxy, p) <= self.exploration_radius for p in self._visited_positions)

        fresh = [c for c in candidates if is_fresh(c)]
        pool = fresh if fresh else candidates
        best = max(pool, key=lambda c: euclidean(agent_xy, [c.position_xyz[0], c.position_xyz[2]]))
        return best.position_xyz

    def _arrive_at_path_end(self) -> None:
        if self._path_target_entity_id is not None:
            self._excluded_entity_ids.add(self._path_target_entity_id)
            self._scan_steps_remaining = self.scan_steps
        self._current_path = None
        self._path_index = 0

    def _start_path(
        self, path: List[List[float]], branch: str, final_goal: List[float], entity_id: Optional[int],
        agent_position: List[float],
    ) -> None:
        self._current_path = path
        self._path_index = 1 if len(path) > 1 else 0
        self._path_final_goal = final_goal
        self._path_branch = branch
        self._path_target_entity_id = entity_id
        self._visited_positions.append([agent_position[0], agent_position[2]])

    def decide(
        self,
        agent_position: List[float],
        agent_rotation_wxyz: List[float],
        query: str,
        memory_store: MemoryStore,
        live_entity_ids: Set[int],
        now: float,
    ) -> ActionDecision:
        query_embedding = self._embed_query(query)
        self._current_query = query
        agent_xy = [agent_position[0], agent_position[2]]

        # --- 0. stuck detection: safety net for navmesh-snapping edge cases (a
        # real path should never walk into a wall, but a waypoint too close to
        # geometry could still occasionally be unreachable in practice).
        if self._last_action == "move_forward" and self._last_position is not None:
            displacement = euclidean(agent_xy, [self._last_position[0], self._last_position[2]])
            self._stuck_move_forward_count = (
                self._stuck_move_forward_count + 1 if displacement < self.stuck_displacement_epsilon else 0
            )
        else:
            self._stuck_move_forward_count = 0
        if self._stuck_move_forward_count >= self.stuck_threshold:
            if self.arrival_fix and self._current_path is not None and hasattr(self.pathfinder, "block_point"):
                # the waypoint the agent keeps colliding at is not enterable: drop it from the grid so re-planning
                # routes around it (THOR's reachable grid admits cells the off-grid agent cannot reach)
                self.pathfinder.block_point(self._current_path[self._path_index])
            if self.arrival_fix:
                # progress safeguard: a goal that blocks the agent 3 times (THOR collision on a path the reachable
                # grid allows) is treated as unreachable instead of re-planned into the same obstacle forever
                key = self._path_target_entity_id
                self._stuck_resets[key] = self._stuck_resets.get(key, 0) + 1
                if self._stuck_resets[key] >= 3:
                    if key is not None:
                        self._excluded_entity_ids.add(key)
                    elif self._path_final_goal is not None:
                        self._visited_positions.append([self._path_final_goal[0], self._path_final_goal[2]])
                    self._stuck_resets[key] = 0
            self._current_path = None
            self._path_index = 0
            self._stuck_move_forward_count = 0

        # --- 0.5 depth-search scan: just arrived at a real target -- look around
        # briefly before moving on. See __init__ comment.
        if self._scan_steps_remaining > 0:
            self._scan_steps_remaining -= 1
            return self._track(agent_position, ActionDecision(
                action="turn_left", branch=self._path_branch or "contextual_anchor",
                goal_position=self._path_final_goal, goal_entity_id=self._path_target_entity_id,
            ))

        # --- 1. path-following persistence ---
        usable_entity_count = sum(
            1 for e in memory_store.entities.values()
            if e.position_world is not None and e.embedding is not None
        )
        new_memory_available = usable_entity_count > self._last_usable_entity_count
        self._steps_since_replan += 1
        periodic_due = self._steps_since_replan >= self.periodic_replan_interval
        fallback_exempt_from_periodic = self._path_branch == "fallback"

        can_persist = (
            self._current_path is not None
            and not live_entity_ids
            and not new_memory_available
            and (fallback_exempt_from_periodic or not periodic_due)
        )

        if can_persist:
            waypoint = self._current_path[self._path_index]
            wp_xy = [waypoint[0], waypoint[2]]
            if euclidean(agent_xy, wp_xy) <= self.goal_reached_distance:
                self._path_index += 1
                if self._path_index >= len(self._current_path):
                    self._arrive_at_path_end()
                    if self._scan_steps_remaining > 0:
                        # start the depth-search scan immediately, same call --
                        # otherwise this call would fall through to a fresh plan
                        # first and the scan wouldn't actually begin until next
                        # call, one step later than "just arrived."
                        self._scan_steps_remaining -= 1
                        return self._track(agent_position, ActionDecision(
                            action="turn_left", branch=self._path_branch or "contextual_anchor",
                            goal_position=self._path_final_goal, goal_entity_id=self._path_target_entity_id,
                        ))
                    can_persist = False  # fall through to a fresh (re)plan below
                else:
                    waypoint = self._current_path[self._path_index]
            if can_persist:
                action = choose_action(agent_position, agent_rotation_wxyz, waypoint)
                return self._track(agent_position, ActionDecision(
                    action=action, branch=self._path_branch, goal_position=self._path_final_goal,
                    goal_entity_id=self._path_target_entity_id,
                ))

        # --- 2. fresh (re)plan: select ONE target, path to it for real ---
        self._current_path = None
        full_memory_best = self._select_target(memory_store, query_embedding, now)
        live_best = self._select_target(memory_store, query_embedding, now, entity_ids_pool=live_entity_ids)
        # memory-off ablation (CLAUDE.md Section 9 Step 4.5): act on live_best only,
        # never on anything remembered from an earlier step -- see __init__ comment.
        # full_memory_best is still computed either way so memory_influenced keeps
        # reporting what persistent memory WOULD have changed, even under the
        # ablation, which is itself a useful diagnostic of how much this ablation
        # is actually suppressing.
        best = full_memory_best if self.memory_enabled else live_best

        best_id = full_memory_best[0].entity_id if full_memory_best is not None else None
        live_id = live_best[0].entity_id if live_best is not None else None
        memory_influenced = None if full_memory_best is None and live_best is None else (best_id != live_id)
        # both always reflect the hypothetical WITH/WITHOUT comparison, regardless
        # of which one `best` (the actually-acted-on selection) resolves to under
        # the memory-off ablation -- otherwise these two fields would collapse to
        # the same value whenever memory_enabled=False, destroying the diagnostic.
        best_value_with = full_memory_best[2] if full_memory_best is not None else 0.0
        best_value_without = live_best[2] if live_best is not None else 0.0
        probe_fields = dict(
            memory_target_entity_id=best_id,
            memory_target_position=(list(full_memory_best[0].position_world) if full_memory_best is not None else None),
            live_target_entity_id=live_id,
            live_target_position=(list(live_best[0].position_world) if live_best is not None else None),
        )
        live_count = sum(
            1 for e in memory_store.entities.values()
            if e.entity_id in live_entity_ids and e.position_world is not None
        )
        remembered_count = sum(
            1 for e in memory_store.entities.values()
            if e.entity_id not in live_entity_ids and e.position_world is not None
        )

        self._steps_since_replan = 0
        self._last_usable_entity_count = usable_entity_count

        if best is not None and self.arrival_fix:
            entity, standoff, r_i = best
            if euclidean(agent_xy, [standoff[0], standoff[2]]) <= self.goal_reached_distance:
                branch = "direct_recall" if r_i >= self.theta_direct else "contextual_anchor"
                self._start_path([standoff], branch, standoff, entity.entity_id, agent_position)
                self._arrive_at_path_end()
                self._scan_steps_remaining -= 1
                return self._track(agent_position, ActionDecision(
                    action="turn_left", branch=branch, goal_position=standoff, goal_entity_id=entity.entity_id,
                    memory_influenced=memory_influenced,
                    best_value_with_memory=best_value_with, best_value_without_memory=best_value_without,
                    live_signal_count=live_count, remembered_signal_count=remembered_count,
                    **probe_fields,
                ))

        if best is not None:
            entity, standoff, r_i = best
            path = compute_path(self.pathfinder, agent_position, standoff)
            if path is None:
                # unreachable despite a valid standoff snap -- exclude, treat as
                # not-found this call (a later call may pick the next-best entity)
                self._excluded_entity_ids.add(entity.entity_id)
                best = None

        if best is not None:
            entity, standoff, r_i = best
            branch = "direct_recall" if r_i >= self.theta_direct else "contextual_anchor"
            self._start_path(path, branch, standoff, entity.entity_id, agent_position)
            waypoint = self._current_path[self._path_index]
            action = choose_action(agent_position, agent_rotation_wxyz, waypoint)
            return self._track(agent_position, ActionDecision(
                action=action, branch=branch, goal_position=standoff, goal_entity_id=entity.entity_id,
                memory_influenced=memory_influenced,
                best_value_with_memory=best_value_with, best_value_without_memory=best_value_without,
                live_signal_count=live_count, remembered_signal_count=remembered_count,
                **probe_fields,
            ))

        # --- 3. fallback: nothing in memory worth going to -- explore ---
        fallback_point = self._pick_fallback_point(agent_position)
        if fallback_point is None:
            self._current_path = None
            self._path_branch = None
            return self._track(agent_position, ActionDecision(action="stop", branch="fallback", goal_position=None))

        path = compute_path(self.pathfinder, agent_position, fallback_point)
        if path is None:
            return self._track(agent_position, ActionDecision(action="stop", branch="fallback", goal_position=None))

        self._start_path(path, "fallback", fallback_point, None, agent_position)
        waypoint = self._current_path[self._path_index]
        action = choose_action(agent_position, agent_rotation_wxyz, waypoint)
        return self._track(agent_position, ActionDecision(
            action=action, branch="fallback", goal_position=fallback_point, goal_entity_id=None,
            memory_influenced=memory_influenced,
            best_value_with_memory=best_value_with, best_value_without_memory=best_value_without,
            live_signal_count=live_count, remembered_signal_count=remembered_count,
            **probe_fields,
        ))
