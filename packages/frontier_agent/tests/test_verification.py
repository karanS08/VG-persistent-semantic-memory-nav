import pytest

from memory_layer.mobility import MobilityPriors
from memory_layer.models import MemoryEntity
from memory_layer.store import MemoryStore

from frontier_agent.verification import (
    check_target_verified, check_target_verified_detailed, _query_head_noun,
)

# Fixed reliability inputs shared by every test below: fallback mobility=1.0 and
# lambda_base=0.0 make temporal decay a no-op (exp(-0*delta_t)=1) regardless of
# category or elapsed time, so tests can reason about relevance/distance/n_i in
# isolation without reliability's time-decay term also moving.
_MOBILITY = MobilityPriors({}, fallback=1.0)
_RELIABILITY_KWARGS = dict(now=0.0, mobility_priors=_MOBILITY, alpha=0.5, lambda_base=0.0)


def make_store_with_entity(embedding, position_world, n_i=2):
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = MemoryEntity(
        entity_id=1, category="chair", description="a chair",
        embedding=embedding, position_world=position_world,
        c_i=1.0, n_i=n_i, v_i=1.0, last_observed=0.0,
    )
    return store


def test_verified_when_relevant_and_close():
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    verified, entity_id = check_target_verified(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, **_RELIABILITY_KWARGS,
    )
    assert verified
    assert entity_id == 1


def test_not_verified_when_too_far():
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[100.0, 0.0, 100.0])
    verified, entity_id = check_target_verified(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, **_RELIABILITY_KWARGS,
    )
    assert not verified
    assert entity_id is None


def test_not_verified_when_irrelevant():
    store = make_store_with_entity(embedding=[0.0, 1.0], position_world=[0.0, 0.0, 0.0])
    verified, _ = check_target_verified(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, **_RELIABILITY_KWARGS,
    )
    assert not verified


def test_no_entities_never_verified():
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    verified, entity_id = check_target_verified(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, **_RELIABILITY_KWARGS,
    )
    assert not verified
    assert entity_id is None


def _entity(eid, embedding, position_world, description="obj", n_i=2):
    return MemoryEntity(
        entity_id=eid, category="chair", description=description,
        embedding=embedding, position_world=position_world,
        c_i=1.0, n_i=n_i, v_i=1.0, last_observed=0.0,
    )


def test_detailed_reports_what_actually_triggered():
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [0.5, 0.0, 0.5], "a soft cushion")

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 1
    assert r.description == "a soft cushion"
    assert r.relevance == pytest.approx(1.0)
    assert r.distance == pytest.approx(0.7071, abs=1e-3)
    assert r.n_eligible == 1
    assert r.n_i == 2


def test_detailed_picks_most_relevant_not_first_in_dict_order():
    """The original implementation returned on the FIRST match in dict iteration
    order, making the reported 'what was found' arbitrary among eligible entities.
    Insert a weak match first so first-in-order != most-relevant. Since the
    2026-08-24 rank-based fix, only entities tied for the GLOBAL best relevance
    count as eligible, so the weak match (0.6) no longer counts as "eligible"
    just for clearing the absolute threshold -- n_eligible reflects only the
    top-ranked entity here."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [0.6, 0.8], [0.5, 0.0, 0.0], "a weakly matching thing")
    store.entities[2] = _entity(2, [1.0, 0.0], [0.6, 0.0, 0.0], "the actual target")

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 2
    assert r.description == "the actual target"
    assert r.n_eligible == 1  # only the top-ranked entity is eligible now


def test_detailed_reports_best_in_memory_even_when_not_verified():
    """Key diagnostic: if nothing verifies, we still want to know how good the
    best available match was -- distinguishes 'nothing relevant here' from
    'something relevant existed but was out of range'."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [100.0, 0.0, 100.0], "a far away perfect match")

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert not r.verified
    assert r.entity_id is None
    assert r.best_relevance_in_memory == pytest.approx(1.0)
    assert r.best_relevance_description == "a far away perfect match"


def test_nearby_weaker_match_does_not_verify_over_a_farther_better_match():
    """Regression test for the 2026-08-24 false-positive bug found in real batch
    data: a weaker-relevance entity that happens to be in range must NOT trigger
    verification while a stronger semantic match exists elsewhere in memory but
    hasn't been reached yet. Previously any entity clearing the absolute
    threshold and in range would win, regardless of whether a better match
    existed out of range."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    # weaker match, but right next to the agent
    store.entities[1] = _entity(1, [0.8, 0.6], [0.1, 0.0, 0.1], "a vaguely related nearby thing")
    # stronger match (true target), but far away -- not yet reached
    store.entities[2] = _entity(2, [1.0, 0.0], [50.0, 0.0, 50.0], "the actual target")

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert not r.verified
    assert r.entity_id is None
    assert r.best_relevance_in_memory == pytest.approx(1.0)
    assert r.best_relevance_description == "the actual target"


def test_verifies_once_agent_reaches_the_actual_best_match():
    """Same setup as above, but the agent has now reached the true best match --
    verification should succeed against entity 2, not entity 1."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [0.8, 0.6], [0.1, 0.0, 0.1], "a vaguely related nearby thing")
    store.entities[2] = _entity(2, [1.0, 0.0], [50.0, 0.0, 50.0], "the actual target")

    r = check_target_verified_detailed(
        agent_position=[50.0, 0.0, 50.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 2
    assert r.description == "the actual target"


def test_wrapper_stays_backwards_compatible():
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [0.5, 0.0, 0.5])
    verified, entity_id = check_target_verified(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert verified is True
    assert entity_id == 1


# --- 2026-08-27: confirmation-count (n_i) gate regression tests ---

def test_single_observation_does_not_verify_when_gate_requires_two():
    """Regression test for the persistent-mode false-positive batch
    (data/runs/phase4_batch_persistent/): a single grounding call's caption --
    n_i=1, never independently reconfirmed -- must not be enough to trigger
    verification when min_confirming_observations=2, even if it clears
    relevance_threshold and the agent is standing right next to it. This is
    exactly the shape of the chair->table, dining table->floor,
    doorway->cabinet false positives found in that batch. NOTE: the
    pipeline's actual default is min_confirming_observations=1 as of
    2026-09-02 (see verification.py's module docstring) -- this test passes
    the stricter value explicitly to keep covering the gate mechanism
    itself, independent of whatever the current default is."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [0.5, 0.0, 0.5], "a coincidentally-worded caption", n_i=1)

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, min_confirming_observations=2, **_RELIABILITY_KWARGS,
    )
    assert not r.verified
    assert r.entity_id is None
    assert r.n_excluded_by_confirmation_gate == 1
    # still surfaced as the best in-memory match, for diagnostic purposes
    assert r.best_relevance_in_memory == pytest.approx(1.0)


def test_single_observation_does_verify_under_the_current_default():
    """The pipeline's actual default (min_confirming_observations=1, since
    2026-09-02) must let a single, category-consistent observation verify --
    the category gate, not the confirmation-count gate, is expected to carry
    the real precision burden now. Same entity as above, default kwargs."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [0.5, 0.0, 0.5], "a single-glimpse caption", n_i=1)

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 1


def test_reconfirmed_observation_does_verify():
    """Same entity as above, but reconfirmed once (n_i=2) -- should now verify,
    demonstrating the gate is a genuine n_i>=2 cutoff, not a blanket rejection."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [0.5, 0.0, 0.5], "a reconfirmed caption", n_i=2)

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 1
    assert r.n_excluded_by_confirmation_gate == 0
    assert r.r_i == pytest.approx(r.relevance * r.reliability)


# --- 2026-08-28: lexical head-noun consistency (category-confusion) gate ---

def test_category_confused_match_blocked_when_query_text_supplied():
    """Regression test for the real per-episode audit finding: 'a wall-mounted
    mirror' repeatedly verified against a WINDOW entity across nearly every
    re-run batch, because raw CLIP relevance doesn't discriminate
    rectangular-framed-glass things from each other. Same entity/embedding
    setup as test_verified_when_relevant_and_close (which verifies fine
    without query_text) -- but with query_text='a wall-mounted mirror'
    supplied and the entity's actual category/description being a window,
    it must now be blocked."""
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].category = "window"
    store.entities[1].description = "a tall rectangular window with a white frame"

    r = check_target_verified_detailed(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, query_text="a wall-mounted mirror",
        **_RELIABILITY_KWARGS,
    )
    assert not r.verified
    assert r.entity_id is None
    assert r.n_excluded_by_category_gate == 1


def test_category_consistent_match_still_verifies():
    """Same setup, but the query's head noun ('window') actually matches the
    entity's category/description -- must still verify normally."""
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].category = "window"
    store.entities[1].description = "a tall rectangular window with a white frame"

    r = check_target_verified_detailed(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, query_text="a large window",
        **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 1
    assert r.n_excluded_by_category_gate == 0


def test_category_field_alone_no_longer_passes_when_description_disagrees():
    """Regression test for the real 2026-09-02 finding: an adversarial query
    'a staircase' (an object confirmed absent from the real scene) verified
    against an entity whose `category` field said 'staircase' but whose
    `description` plainly read 'a wall-mounted air conditioner' -- DAAAM's
    coarse category classifier is wrong here even though its free-text
    caption is accurate. The lexical gate must key off `description` only,
    not `category`, so this can no longer pass."""
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].category = "staircase"
    store.entities[1].description = "a wall-mounted air conditioner with a smooth, flat front panel"

    r = check_target_verified_detailed(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, query_text="a staircase",
        **_RELIABILITY_KWARGS,
    )
    assert not r.verified
    assert r.entity_id is None
    assert r.n_excluded_by_category_gate == 1


def test_lexical_gate_skipped_when_query_text_omitted():
    """query_text defaults to None -- old callers get the pre-08-28 behavior
    exactly, including the category-confused match that a real query_text
    would have blocked."""
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].category = "window"
    store.entities[1].description = "a tall rectangular window with a white frame"

    r = check_target_verified_detailed(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.n_excluded_by_category_gate == 0


def test_reliability_weighted_ranking_prefers_more_corroborated_entity():
    """Two entities both clear relevance_threshold and the n_i gate, and both
    are in range -- but entity 2 has been independently confirmed far more
    times than entity 1 (higher n_i => higher reliability C_i), even though
    entity 1 has marginally higher raw relevance. Ranking by R_i=relevance*
    reliability (not raw relevance) should prefer entity 2, the better-
    corroborated one, matching CLAUDE.md Section 6's direct-recall formula."""
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.entities[1] = _entity(1, [1.0, 0.0], [0.1, 0.0, 0.1], "barely-confirmed, slightly higher relevance", n_i=2)
    store.entities[2] = _entity(2, [0.99, 0.01], [0.1, 0.0, 0.2], "heavily-confirmed, slightly lower relevance", n_i=20)

    r = check_target_verified_detailed(
        agent_position=[0.0, 0.0, 0.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=2.0, **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 2


# --- 2026-09-07 relational-query fix ---------------------------------------
# Regression tests for the real, code-verified bug in
# results/lexical_rule_policy.md ("Category 5"): the query "a lamp near a
# window" verified against real entity_id=82, described "A tall,
# rectangular window with a white frame..." with zero mention of a lamp.

@pytest.mark.parametrize("query_text,expected_head", [
    ("a lamp near a window", "lamp"),          # the real reported bug
    ("a lamp on a nightstand", "lamp"),
    ("a plant next to the couch", "plant"),
    ("a mirror above the sink", "mirror"),
    # no relation marker present -> unchanged, whole-string behavior (must
    # not regress any query this project has ever actually run)
    ("a wall-mounted mirror", "mirror"),
    ("a dining table", "table"),
    ("a doorway", "doorway"),
    ("a soft cushion", "cushion"),
])
def test_query_head_noun_uses_target_not_reference_phrase(query_text, expected_head):
    assert _query_head_noun(query_text) == expected_head


def test_relational_query_no_longer_matches_bare_reference_entity():
    """The exact real failure: 'a lamp near a window' must not verify
    against an entity that is only a window, with no lamp mentioned."""
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].category = "window"
    store.entities[1].description = "A tall, rectangular window with a white frame"

    r = check_target_verified_detailed(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, query_text="a lamp near a window",
        **_RELIABILITY_KWARGS,
    )
    assert not r.verified
    assert r.entity_id is None
    assert r.n_excluded_by_category_gate == 1


def test_relational_query_matches_true_target_entity():
    """A real lamp entity -- one whose description happens to also mention
    being near a window -- must still verify normally; the fix should only
    stop the reference noun from being treated as the target, not make the
    gate stricter for genuine matches."""
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].category = "lamp"
    store.entities[1].description = "A tall floor lamp standing near a window"

    r = check_target_verified_detailed(
        agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
        relevance_threshold=0.5, verify_distance=1.0, query_text="a lamp near a window",
        **_RELIABILITY_KWARGS,
    )
    assert r.verified
    assert r.entity_id == 1


# --- 2026-09-28: caption_head lexical mode ---

from frontier_agent.verification import _caption_head_consistent  # noqa: E402


class _Desc:
    def __init__(self, description):
        self.description = description


@pytest.mark.parametrize("query,caption", [
    # Real audited false positives the legacy prefix rule accepted.
    ("a soft cushion", "A long, rectangular bench with a light beige, textured cushion on top."),
    ("a soft cushion", "A long, narrow bench with a light beige cushioned seat and a white wooden frame."),
    ("a doorway", "A modern built-in oven featuring a stainless steel finish with a large glass door."),
    ("a doorway", "A tall, narrow white cabinet with a single visible door featuring a raised panel."),
    ("a framed painting", "A tall, rectangular window with a white frame and some peeling paint."),
    ("a wall vent", "The ceiling is painted white and features a series of recessed lights."),
    ("a staircase", "A beige, wall-mounted air conditioner with a smooth, flat front panel."),
])
def test_caption_head_rejects_modifier_and_mention_matches(query, caption):
    assert not _caption_head_consistent(query, _Desc(caption))


@pytest.mark.parametrize("query,caption", [
    ("a doorway", "A tall, white wooden door featuring a raised panel design."),
    ("a wall-mounted mirror", "A rectangular, vertically oriented mirror with a thin, light-colored frame."),
    ("a wall-mounted mirror", "The wall is painted white and features a large rectangular mirror with a decorative frame."),
    ("a large window", "A tall, rectangular window with a white frame and a single vertical pane."),
    ("a cushion", "Two beige cushions with a textured surface."),
    ("a lamp", "A white, triangular-shaped lampshade with a smooth surface."),
    ("a dining table", "A long, narrow, white marble table with four legs."),
    ("a lamp near a window", "A tall floor lamp with a conical shade."),
    ("a wall-mounted mirror", "A white, rectangular wall panel with a central mirror framed by a decorative border."),
    ("a wall-mounted mirror", "The wall is adorned with a large rectangular mirror framed by an ornate molding."),
    ("a rug", "The floor is covered with a large, irregularly shaped cowhide rug."),
    ("a cabinet", "The wall is covered with white tiles, featuring a large, rectangular, built-in cabinet."),
])
def test_caption_head_accepts_subject_matches(query, caption):
    assert _caption_head_consistent(query, _Desc(caption))


def test_lexical_mode_caption_head_blocks_cushioned_bench_end_to_end():
    store = make_store_with_entity(embedding=[1.0, 0.0], position_world=[1.0, 0.0, 1.0])
    store.entities[1].description = "A long, rectangular bench with a light beige cushioned seat."
    kwargs = dict(agent_position=[1.0, 0.0, 1.0], memory_store=store, query_embedding=[1.0, 0.0],
                  relevance_threshold=0.5, verify_distance=1.0, query_text="a soft cushion",
                  **_RELIABILITY_KWARGS)
    assert check_target_verified_detailed(**kwargs, lexical_mode="prefix").verified
    r = check_target_verified_detailed(**kwargs, lexical_mode="caption_head")
    assert not r.verified
    assert r.n_excluded_by_category_gate == 1
