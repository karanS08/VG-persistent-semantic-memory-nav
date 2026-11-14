import math

import numpy as np
import pytest

from frontier_agent.agent import FrontierAgent
from memory_layer.mobility import MobilityPriors
from memory_layer.models import MemoryEntity
from memory_layer.store import MemoryStore


class FakeEncoder:
    """Deterministic stand-in for ClipTextEncoder -- avoids loading a real CLIP
    model in unit tests. Embeds any text to a fixed 2D vector so relevance is
    controllable per-test via the entity embeddings chosen. Counts calls so tests
    can assert on caching behavior (see test_query_embedding_is_cached_per_query)."""
    def __init__(self):
        self.embed_calls = []

    def embed(self, text):
        self.embed_calls.append(text)
        return np.array([1.0, 0.0])


class FakePathfinder:
    """2026-08-24: gained find_path() alongside the pre-existing
    get_random_navigable_point_near/snap_point, since decide() now drives real
    A* waypoint following (see pathing.compute_path) instead of single-bearing
    movement. find_path() writes directly onto the REAL habitat_sim.ShortestPath
    object pathing.compute_path constructs (verified settable -- PROGRESS.md
    2026-08-24), producing a straight 2-point [start, goal] path by default, which
    keeps most pre-existing test assertions meaningful (a 2-point path's single
    waypoint IS the final goal). `unreachable_goals` lets a test simulate a target
    find_path can't reach."""
    def __init__(self, candidates=None, unreachable_goals=None):
        self._candidates = candidates or []
        self._i = 0
        self._unreachable_goals = {tuple(round(float(x), 3) for x in g) for g in (unreachable_goals or [])}

    def get_random_navigable_point_near(self, circle_center, radius):
        if self._i >= len(self._candidates):
            return None
        p = self._candidates[self._i]
        self._i += 1
        return np.array(p, dtype=np.float32)

    def snap_point(self, point):
        return np.array(point, dtype=np.float32)  # always "reachable" for these tests

    def find_path(self, shortest_path):
        start = [float(x) for x in shortest_path.requested_start]
        end = [float(x) for x in shortest_path.requested_end]
        key = tuple(round(x, 3) for x in end)
        if key in self._unreachable_goals:
            return False
        shortest_path.points = [start, end]
        return True


def make_agent(pathfinder, n_candidates=3, periodic_replan_interval=10, memory_enabled=True,
                mobility_scaling_enabled=True):
    return FrontierAgent(
        pathfinder=pathfinder,
        clip_encoder=FakeEncoder(),
        mobility_priors=MobilityPriors({"chair": 0.3}, fallback=0.5),
        alpha=0.5, lambda_base=0.001, theta_direct=0.2, anchor_weight=1.0,
        value_map_sigma=2.0, n_frontier_candidates=n_candidates,
        periodic_replan_interval=periodic_replan_interval,
        memory_enabled=memory_enabled,
        mobility_scaling_enabled=mobility_scaling_enabled,
    )


def test_multi_anchor_fusion_prefers_clustered_support_over_isolated_top_scorer():
    """2026-09-02: _select_target should not always take the single highest-r_i
    entity outright -- a slightly-lower-scoring entity with support from another
    good nearby candidate should be able to win over an isolated top scorer, per
    the multi-anchor fusion design (agent.py module docstring)."""
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf)  # anchor_weight=1.0, value_map_sigma=2.0 (see make_agent)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    # A: highest solo r_i (n_i=6 -> reliability~=0.950), but isolated.
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="entity A",
        embedding=[1.0, 0.0], position_world=[100.0, 0.0, 100.0],
        c_i=1.0, n_i=6, v_i=1.0, last_observed=0.0,
    )
    # B and C: lower solo r_i (n_i=3 -> reliability~=0.777 each), but close together
    # (distance ~1.41, well within sigma=2.0) -- each should boost the other's
    # fused value past A's isolated solo score.
    store.entities[2] = MemoryEntity(
        entity_id=2, category="chair", description="entity B",
        embedding=[1.0, 0.0], position_world=[0.0, 0.0, 0.0],
        c_i=1.0, n_i=3, v_i=1.0, last_observed=0.0,
    )
    store.entities[3] = MemoryEntity(
        entity_id=3, category="chair", description="entity C",
        embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0],
        c_i=1.0, n_i=3, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.goal_entity_id in (2, 3)  # B or C wins, NOT the isolated top scorer A


def test_anchor_weight_zero_reproduces_solo_argmax_behavior():
    """Same three-entity setup as above, but anchor_weight=0 -- fusion should have
    no effect, and the isolated top solo-scorer (A) wins, exactly as it did before
    the 2026-09-02 multi-anchor change."""
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf)
    agent.anchor_weight = 0.0
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="entity A",
        embedding=[1.0, 0.0], position_world=[100.0, 0.0, 100.0],
        c_i=1.0, n_i=6, v_i=1.0, last_observed=0.0,
    )
    store.entities[2] = MemoryEntity(
        entity_id=2, category="chair", description="entity B",
        embedding=[1.0, 0.0], position_world=[0.0, 0.0, 0.0],
        c_i=1.0, n_i=3, v_i=1.0, last_observed=0.0,
    )
    store.entities[3] = MemoryEntity(
        entity_id=3, category="chair", description="entity C",
        embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0],
        c_i=1.0, n_i=3, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.goal_entity_id == 1


def test_fallback_branch_when_memory_is_empty():
    pf = FakePathfinder(candidates=[[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    agent = make_agent(pf, n_candidates=2)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.branch == "fallback"
    assert decision.action in ("move_forward", "turn_left", "turn_right", "stop")


def test_contextual_anchor_branch_when_relevant_memory_exists():
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    # a real, positioned, embedded entity with WEAK relevance (cosine([1,0],[0.3,0.9])
    # ~=0.316) and a single observation (n_i=1 caps C_i~=0.393 at alpha=0.5) --
    # r_i ~= 0.124, deliberately below theta_direct=0.2 so this exercises the
    # contextual-anchor branch specifically, not direct recall (a stronger match
    # would legitimately qualify for direct recall too, as an earlier version of
    # this test's own data accidentally did -- fixed after a real test failure).
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[0.3, 0.9], position_world=[1.0, 0.0, 1.0],
        c_i=1.0, n_i=1, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.branch == "contextual_anchor"
    assert decision.goal_entity_id == 1
    assert decision.best_value_with_memory > 0.0


def test_goal_persists_across_calls_until_reached():
    """The real bug this guards against: decide() used to resample fresh random
    candidates every call, so in fallback mode the agent picked near-random noise
    each step and never made real progress (see PROGRESS.md Phase 4). With
    persistence, a second decide() call before the first goal is reached should
    keep walking the SAME path, not plan a freshly sampled one.
    """
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    # still far from the goal -- a second call should NOT resample
    d2 = agent.decide(
        agent_position=[0.1, 0.0, 0.1], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=1.0,
    )
    assert d1.goal_position == d2.goal_position == pytest.approx([5.0, 0.0, 5.0])
    # pathfinder's internal counter should NOT have advanced on the second call
    assert pf._i == 1


def test_goal_replanned_once_reached():
    pf = FakePathfinder(candidates=[[0.1, 0.0, 0.1], [9.0, 0.0, 9.0]])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.goal_position == pytest.approx([0.1, 0.0, 0.1])
    # agent is now AT the goal (within goal_reached_distance) -- next call should replan
    d2 = agent.decide(
        agent_position=[0.1, 0.0, 0.1], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=1.0,
    )
    assert d2.goal_position == pytest.approx([9.0, 0.0, 9.0])
    assert pf._i == 2


def test_reset_clears_persisted_goal():
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0], [1.0, 0.0, 1.0]])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert agent._current_path is not None
    agent.reset()
    assert agent._current_path is None
    assert agent._path_branch is None
    assert agent._path_final_goal is None
    assert agent._excluded_entity_ids == set()


def test_direct_recall_preempts_persisted_fallback_goal():
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0]])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.branch == "fallback"

    # a strongly relevant, reachable entity now appears in memory -- this alone
    # (new_memory_available) forces a fresh plan even though it's not "live" this
    # step, per the 2026-08-21/23 timing fix (still in effect under the 2026-08-24
    # navigation redesign).
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[3.0, 0.0, 3.0],
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )
    d2 = agent.decide(
        agent_position=[0.1, 0.0, 0.1], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=1.0,
    )
    assert d2.branch == "direct_recall"
    assert d2.goal_entity_id == 1


def test_fresh_evidence_forces_replan_even_before_goal_reached():
    """The other half of the goal-persistence fix: persistence shouldn't starve
    target selection of a chance to run just because a goal hasn't been reached
    yet -- fresh evidence (live_entity_ids non-empty) should force an immediate
    re-plan."""
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.goal_position == pytest.approx([5.0, 0.0, 5.0])

    # still far from d1's goal, but fresh evidence just arrived -- should replan
    # (pathfinder's counter advances -> a DIFFERENT fallback candidate) rather than persist
    d2 = agent.decide(
        agent_position=[0.1, 0.0, 0.1], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids={42}, now=1.0,
    )
    assert d2.goal_position == pytest.approx([1.0, 0.0, 1.0])
    assert pf._i == 2


def test_memory_influenced_true_when_remembered_signal_changes_winner():
    """2026-08-24: under the target-selection redesign, the memory-influence probe
    compares the best entity found across ALL entities vs. the best found among
    only LIVE ones -- much more directly than the old candidate-position tie-break
    dance this test used to need (see git history / PROGRESS.md if curious)."""
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    # a REMEMBERED (not live) entity, weak enough to stay under theta_direct=0.2
    store.entities[7] = MemoryEntity(
        entity_id=7, category="chair", description="a chair",
        embedding=[0.3, 0.9], position_world=[1.0, 0.0, 1.0],
        c_i=1.0, n_i=1, v_i=1.0, last_observed=0.0,
    )
    # entity_id 7 is NOT in live_entity_ids -> it's "remembered", not "live"
    d = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d.branch == "contextual_anchor"
    assert d.memory_influenced is True
    assert d.best_value_with_memory > 0.0
    assert d.best_value_without_memory == 0.0  # no live signals at all in this test


def test_memory_influenced_none_when_nothing_found_at_all():
    """2026-08-24 refinement: when neither the full pool nor the live-only pool
    finds ANY entity (pure fallback, nothing in memory at all), memory_influenced
    is None (not False) -- there's no entity-driven decision to have been
    influenced or not; a fallback-with-nothing-in-memory step is excluded from the
    probe's denominator entirely, which is a more precise match to the metric's
    intent ("of entity-driven decisions, how often did memory change the target")
    than the old design's False for this case."""
    pf = FakePathfinder(candidates=[[1.0, 0.0, 1.0], [9.0, 0.0, 9.0]])
    agent = make_agent(pf, n_candidates=2)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    d = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d.branch == "fallback"
    assert d.memory_influenced is None


def test_fallback_branch_is_exempt_from_periodic_replan():
    """2026-08-24 (navigation redesign) / originally 2026-08-23: periodic_replan_interval
    exists to catch MEMORY going stale, so a pure "fallback" goal (zero live/
    remembered signal behind it, by definition) has nothing to go stale and must
    NOT be cut off by the timer -- doing so was verified (a real 300-step
    instrumented run, pre-redesign) to leave the agent oscillating in a ~2m pocket.
    Fallback goals persist until reached or until real evidence (live detection or
    memory growth) appears."""
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0], [9.0, 0.0, 9.0]])
    agent = make_agent(pf, n_candidates=1, periodic_replan_interval=3)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.branch == "fallback"
    assert d1.goal_position == pytest.approx([5.0, 0.0, 5.0])

    # even well past periodic_replan_interval (3), the fallback goal must persist
    # -- no new evidence, no reason to abandon it.
    for i in range(5):
        d = agent.decide(
            agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
            query="a chair", memory_store=store, live_entity_ids=set(), now=float(i + 1),
        )
        assert d.goal_position == pytest.approx([5.0, 0.0, 5.0])
    assert pf._i == 1  # still no new candidate sampled


def test_periodic_replan_still_fires_for_contextual_anchor_without_new_evidence():
    """The periodic safety net (PROGRESS.md Phase 4, 2026-08-23) still applies to a
    memory-driven contextual_anchor goal, which CAN go stale (reliability decays
    with elapsed time) -- only pure fallback is exempt (see
    test_fallback_branch_is_exempt_from_periodic_replan).

    2026-08-24 note: under the target-selection redesign, re-selecting the SAME
    best-scoring entity naturally produces the SAME goal (unlike the old random-
    candidate-resampling design, where a periodic replan could coincidentally land
    on a different candidate position for the same entity) -- so this asserts the
    replan actually ran via the internal steps-since-replan counter, which is now
    the correct observable signature of "did a replan happen," not a changed goal.
    """
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf, periodic_replan_interval=3)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[7] = MemoryEntity(
        entity_id=7, category="chair", description="a chair",
        embedding=[0.3, 0.9], position_world=[5.0, 0.0, 5.0],
        c_i=1.0, n_i=1, v_i=1.0, last_observed=0.0,
    )

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.branch == "contextual_anchor"
    assert d1.goal_position == pytest.approx([5.0, 0.0, 5.0])
    assert agent._steps_since_replan == 0

    for i in range(2):
        d = agent.decide(
            agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
            query="a chair", memory_store=store, live_entity_ids=set(), now=float(i + 1),
        )
        assert d.goal_position == pytest.approx([5.0, 0.0, 5.0])
    assert agent._steps_since_replan == 2  # persisting, no replan yet

    # 4th call: periodic interval (3) reached -- must force a fresh plan even though
    # the remembered entity itself hasn't changed.
    d4 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=3.0,
    )
    assert d4.goal_position == pytest.approx([5.0, 0.0, 5.0])
    assert agent._steps_since_replan == 0  # proves a replan actually ran


def test_new_usable_memory_forces_replan_without_live_evidence():
    """The other half of the timing fix: an entity that gains a position+embedding
    (e.g. via backfill) between decide() calls must trigger a re-plan even if it
    was never in live_entity_ids at the moment it became usable.

    2026-08-24 note: under the target-selection redesign, the observable proof of
    "a replan happened" is switching from fallback exploration to the newly-usable
    entity's real location, not a newly-sampled random fallback point."""
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0]])
    agent = make_agent(pf, periodic_replan_interval=100)  # won't fire
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.branch == "fallback"
    assert d1.goal_position == pytest.approx([5.0, 0.0, 5.0])

    d2 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=1.0,
    )
    assert d2.goal_position == pytest.approx([5.0, 0.0, 5.0])  # still persisting
    assert pf._i == 1

    # an entity becomes usable "out of band" -- NOT flagged as live this step,
    # simulating a backfilled position landing after the entity's one grounding
    # event already streamed through
    store.entities[99] = MemoryEntity(
        entity_id=99, category="chair", description="a chair",
        embedding=[0.3, 0.9], position_world=[1.0, 0.0, 1.0],
        c_i=1.0, n_i=1, v_i=1.0, last_observed=0.0,
    )

    d3 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=2.0,
    )
    assert d3.branch == "contextual_anchor"  # forced a fresh plan, found the entity
    assert d3.goal_position == pytest.approx([1.0, 0.0, 1.0])
    assert pf._i == 1  # no NEW fallback candidate needed -- the entity was found instead


def test_fallback_prefers_unexplored_point_over_already_visited_one():
    """2026-08-24: _pick_fallback_point still avoids re-picking a point near
    territory the agent has already planned a path FROM -- the one surviving piece
    of the original 'gravity well' visited-position mask, now scoped to
    fallback-only, since entity-driven navigation can no longer exhibit that bug
    at all under single-target real-path selection (see module docstring)."""
    pf = FakePathfinder(candidates=[
        [1.0, 0.0, 1.0], [2.0, 0.0, 2.0],  # call 1: both "fresh" -- farthest wins
        [1.5, 0.0, 1.5], [9.0, 0.0, 9.0],  # call 2: one near-visited, one fresh
    ])
    agent = make_agent(pf, n_candidates=2)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.goal_position == pytest.approx([2.0, 0.0, 2.0])  # farther of the two

    # force a fresh plan without reaching the goal
    d2 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids={42}, now=1.0,
    )
    # [1.5,1.5] is within exploration_radius (3.0) of the visited [0,0] start point
    # -- masked out in favor of the fresh, far [9,9]
    assert d2.goal_position == pytest.approx([9.0, 0.0, 9.0])


def test_reset_persist_coverage_keeps_visited_positions_across_queries():
    """2026-09-02: reset(persist_coverage=True) -- the persistent-across-queries
    case -- must NOT clear _visited_positions, so spatial coverage from an
    earlier query still masks out already-explored fallback territory on a
    later query in the same session. Same setup as
    test_fallback_prefers_unexplored_point_over_already_visited_one, but with
    an agent.reset() in between standing in for 'this is a new query now'."""
    pf = FakePathfinder(candidates=[
        [2.0, 0.0, 2.0],  # query 1: visited from [0,0]
        [1.5, 0.0, 1.5], [9.0, 0.0, 9.0],  # query 2: one near query 1's visited spot, one fresh
    ])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )  # records [0,0] as visited via _start_path

    agent.reset(persist_coverage=True)  # "new query, same session" -- NOT a fresh episode
    agent.n_frontier_candidates = 2
    d2 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a wall clock", memory_store=store, live_entity_ids=set(), now=10.0,
    )
    # [1.5,1.5] is still within exploration_radius of [0,0], masked from a
    # PREVIOUS query -- the fresh [9,9] must win despite this being a new query
    assert d2.goal_position == pytest.approx([9.0, 0.0, 9.0])


def test_reset_without_persist_coverage_clears_visited_positions():
    """Contrast case: reset()'s default (persist_coverage=False, the
    non-persistent/fresh-episode case) clears _visited_positions exactly as
    before 2026-09-02 -- a near-[0,0] candidate is fair game again once the
    prior query's coverage is gone."""
    pf = FakePathfinder(candidates=[
        [2.0, 0.0, 2.0],  # query 1: visited from [0,0]
        [1.5, 0.0, 1.5],  # query 2 (fresh episode): only a near-[0,0] candidate offered
    ])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )

    agent.reset()  # default: a genuinely fresh episode, coverage does NOT carry over
    d2 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a wall clock", memory_store=store, live_entity_ids=set(), now=10.0,
    )
    # with no visited-position memory left, the only candidate offered is
    # accepted even though it would have been masked had coverage persisted
    assert d2.goal_position == pytest.approx([1.5, 0.0, 1.5])


def test_direct_recall_branch_when_strongly_relevant_and_reachable():
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    # high n_i pushes C_i's diminishing-returns term close to 1, combined with
    # relevance=1.0 (matching FakeEncoder's fixed embedding) this should clear
    # theta_direct=0.2 easily
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[3.0, 0.0, 3.0],
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.branch == "direct_recall"
    assert decision.goal_entity_id == 1


def test_unreachable_target_is_excluded_and_falls_back():
    """2026-08-24: if compute_path can't reach a scored entity's standoff pose
    despite a valid snap (e.g. disconnected navmesh island), the entity is
    excluded and the agent falls back to exploring instead of getting stuck
    repeatedly trying to path to something it can never reach."""
    pf = FakePathfinder(candidates=[[9.0, 0.0, 9.0]], unreachable_goals=[[3.0, 0.0, 3.0]])
    agent = make_agent(pf, n_candidates=1)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[3.0, 0.0, 3.0],
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.branch == "fallback"
    assert decision.goal_position == pytest.approx([9.0, 0.0, 9.0])
    assert 1 in agent._excluded_entity_ids


def test_depth_search_scan_runs_after_arriving_at_a_target():
    """2026-08-24: arriving at a real (non-fallback) target's standoff pose should
    trigger a few steps of turning in place ("depth search") before the agent
    moves on to anything else, giving verification.py's per-step check a fair
    chance to see the target."""
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf)
    agent.scan_steps = 3
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[0.1, 0.0, 0.1],  # already ~within goal_reached_distance
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )

    # first call: plans the path (2 points: start, standoff~=[0.1,0,0.1]); the
    # agent starts at [0,0,0], so the single waypoint is already within
    # goal_reached_distance (0.5) -- but arrival is only detected on the NEXT call
    # that checks distance-to-waypoint via the persistence block, so drive it there
    d1 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert d1.branch == "direct_recall"

    # second call, still effectively at the same spot -- waypoint is within
    # goal_reached_distance, so this call detects arrival and starts the scan
    d2 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=1.0,
    )
    assert d2.action == "turn_left"
    assert 1 in agent._excluded_entity_ids  # arrival excludes the target entity
    assert agent._scan_steps_remaining == 2

    d3 = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=2.0,
    )
    assert d3.action == "turn_left"
    assert agent._scan_steps_remaining == 1


def test_query_embedding_is_cached_per_query():
    """Real-time-optimization fix: decide() used to call clip_encoder.embed(query)
    on every single call even though the query text doesn't change within an
    episode. Now cached per query string on the FrontierAgent instance."""
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    encoder = FakeEncoder()
    agent = FrontierAgent(
        pathfinder=pf, clip_encoder=encoder,
        mobility_priors=MobilityPriors({"chair": 0.3}, fallback=0.5),
        alpha=0.5, lambda_base=0.001, theta_direct=0.2, anchor_weight=1.0,
        value_map_sigma=2.0, n_frontier_candidates=1,
    )
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert encoder.embed_calls == ["a chair"]

    # second call, same query, fresh evidence forces a re-plan -- embed() must NOT
    # be called again
    agent.decide(
        agent_position=[0.1, 0.0, 0.1], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids={99}, now=1.0,
    )
    assert encoder.embed_calls == ["a chair"]  # still just one call total


def test_query_embedding_cache_keyed_by_query_text():
    """A different query string should still trigger a fresh embed() call --
    caching must be keyed on the text, not a blanket "embed once ever"."""
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0], [1.0, 0.0, 1.0]])
    encoder = FakeEncoder()
    agent = FrontierAgent(
        pathfinder=pf, clip_encoder=encoder,
        mobility_priors=MobilityPriors({"chair": 0.3}, fallback=0.5),
        alpha=0.5, lambda_base=0.001, theta_direct=0.2, anchor_weight=1.0,
        value_map_sigma=2.0, n_frontier_candidates=1,
    )
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)

    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a window", memory_store=store, live_entity_ids={99}, now=1.0,
    )
    assert encoder.embed_calls == ["a chair", "a window"]


def test_memory_off_ablation_ignores_remembered_entity():
    """CLAUDE.md Section 9 Step 4.5's 'memory-off (live-only, fallback always)'
    ablation: with memory_enabled=False, a strongly-relevant REMEMBERED (not live)
    entity must be completely ignored -- the agent falls back to exploration
    instead of navigating to it, even though the same entity would trigger
    direct_recall with memory enabled (see test_direct_recall_branch_when_strongly_relevant_and_reachable)."""
    pf = FakePathfinder(candidates=[[9.0, 0.0, 9.0]])
    agent = make_agent(pf, n_candidates=1, memory_enabled=False)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[3.0, 0.0, 3.0],
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    assert decision.branch == "fallback"
    assert decision.goal_position == pytest.approx([9.0, 0.0, 9.0])
    # the probe still reports what full memory WOULD have done, even though the
    # ablation didn't act on it -- a useful diagnostic of how much is suppressed
    assert decision.memory_influenced is True
    assert decision.best_value_with_memory > 0.0
    assert decision.best_value_without_memory == 0.0


def test_memory_off_ablation_still_uses_live_entity():
    """The same ablation must still act on a LIVE entity (this step's fresh
    detection) -- "memory-off" means ignoring PERSISTED/remembered evidence, not
    ignoring current perception entirely."""
    pf = FakePathfinder(candidates=[])
    agent = make_agent(pf, memory_enabled=False)
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[3.0, 0.0, 3.0],
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )

    decision = agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids={1}, now=0.0,
    )
    assert decision.branch == "direct_recall"
    assert decision.goal_entity_id == 1


def test_no_mobility_scaling_ablation_uses_flat_lambda():
    """CLAUDE.md Section 9 Step 4.5's 'no mobility scaling: flat lambda across all
    categories' ablation: with mobility_scaling_enabled=False, _score_entity must
    use mobility=1.0 for every entity regardless of category (make_agent's
    MobilityPriors gives "chair" a real mobility of 0.3), so reliability decays
    faster (mobility=1.0 -> lambda_i=lambda_base) than the non-ablated case
    (mobility=0.3 -> lambda_i=0.3*lambda_base)."""
    pf = FakePathfinder(candidates=[])
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    entity = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0],
        c_i=1.0, n_i=20, v_i=1.0, last_observed=0.0,
    )
    store.entities[1] = entity
    now = 1000.0  # large elapsed time so the decay difference is clearly visible

    scaled_agent = make_agent(pf, mobility_scaling_enabled=True)
    flat_agent = make_agent(pf, mobility_scaling_enabled=False)

    scaled_reliability, _ = scaled_agent._score_entity(entity, [1.0, 0.0], now)
    flat_reliability, _ = flat_agent._score_entity(entity, [1.0, 0.0], now)

    # flat (mobility=1.0, faster decay) must be strictly lower than scaled
    # (mobility=0.3, slower decay) at the same elapsed time
    assert flat_reliability < scaled_reliability
    confirmed_quality = 1.0 - math.exp(-0.5 * 20)  # alpha=0.5, n_i=20, c_i=v_i=1.0
    assert flat_reliability == pytest.approx(confirmed_quality * math.exp(-0.001 * 1000.0))  # lambda_i = lambda_base
    assert scaled_reliability == pytest.approx(confirmed_quality * math.exp(-0.0003 * 1000.0))  # lambda_i = 0.3*lambda_base


def test_reset_does_not_clear_query_embedding_cache():
    pf = FakePathfinder(candidates=[[5.0, 0.0, 5.0]])
    encoder = FakeEncoder()
    agent = FrontierAgent(
        pathfinder=pf, clip_encoder=encoder,
        mobility_priors=MobilityPriors({"chair": 0.3}, fallback=0.5),
        alpha=0.5, lambda_base=0.001, theta_direct=0.2, anchor_weight=1.0,
        value_map_sigma=2.0, n_frontier_candidates=1,
    )
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    agent.decide(
        agent_position=[0.0, 0.0, 0.0], agent_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        query="a chair", memory_store=store, live_entity_ids=set(), now=0.0,
    )
    agent.reset()
    assert "a chair" in agent._query_embedding_cache
