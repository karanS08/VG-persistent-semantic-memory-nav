import pytest

from memory_layer.store import MemoryStore, Observation, Verification


def make_store():
    return MemoryStore(initial_v=1.0, verification_failure_discount=0.5)


def test_new_observation_creates_entity():
    store = make_store()
    obs = Observation(
        entity_id=1, category="phone", description="a black phone",
        embedding=[1.0, 0.0], confidence=1.0, room_label="office", timestamp=10.0,
    )
    store.update([obs], [])
    assert 1 in store.entities
    e = store.entities[1]
    assert e.n_i == 1
    assert e.v_i == 1.0
    assert e.first_observed == 10.0
    assert e.last_observed == 10.0


def test_repeat_observation_increments_n_i_not_v_i():
    store = make_store()
    obs1 = Observation(entity_id=1, category="phone", description="a phone", embedding=None, confidence=1.0, room_label=None, timestamp=10.0)
    obs2 = Observation(entity_id=1, category="phone", description="a phone, better view", embedding=[1.0], confidence=1.0, room_label=None, timestamp=20.0)
    store.update([obs1], [])
    store.update([obs2], [])
    e = store.entities[1]
    assert e.n_i == 2
    assert e.v_i == 1.0
    assert e.first_observed == 10.0  # unchanged
    assert e.last_observed == 20.0  # updated
    assert e.description == "a phone, better view"  # updated
    assert e.embedding == [1.0]  # updated


def test_incidental_objects_incorporated_alongside_target():
    """O_t explicitly includes incidental objects unrelated to the query target."""
    store = make_store()
    target = Observation(entity_id=1, category="phone", description="target phone", embedding=None, confidence=1.0, room_label=None, timestamp=0.0)
    incidental = Observation(entity_id=2, category="chair", description="an unrelated chair", embedding=None, confidence=1.0, room_label=None, timestamp=0.0)
    store.update([target, incidental], [])
    assert set(store.entities.keys()) == {1, 2}


def test_failed_verification_discounts_v_i_but_does_not_delete():
    store = make_store()
    obs = Observation(entity_id=1, category="phone", description="a phone", embedding=None, confidence=1.0, room_label=None, timestamp=0.0)
    store.update([obs], [])
    store.update([], [Verification(entity_id=1, success=False)])
    e = store.entities[1]
    assert e.v_i == pytest.approx(0.5)
    assert 1 in store.entities  # not deleted


def test_successful_verification_leaves_v_i_unchanged():
    store = make_store()
    obs = Observation(entity_id=1, category="phone", description="a phone", embedding=None, confidence=1.0, room_label=None, timestamp=0.0)
    store.update([obs], [])
    store.update([], [Verification(entity_id=1, success=True)])
    assert store.entities[1].v_i == pytest.approx(1.0)


def test_repeated_failed_verifications_compound():
    store = make_store()
    obs = Observation(entity_id=1, category="phone", description="a phone", embedding=None, confidence=1.0, room_label=None, timestamp=0.0)
    store.update([obs], [])
    store.update([], [Verification(entity_id=1, success=False)])
    store.update([], [Verification(entity_id=1, success=False)])
    assert store.entities[1].v_i == pytest.approx(0.25)  # 1.0 * 0.5 * 0.5


def test_backfill_positions_fills_missing_position():
    store = make_store()
    obs = Observation(entity_id=1, category="chair", description="a chair", embedding=None, confidence=1.0, room_label=None, timestamp=0.0, position_world=None)
    store.update([obs], [])
    assert store.entities[1].position_world is None

    n = store.backfill_positions({"1": [{"position_world": [1.0, 2.0, 3.0], "frame_id": 5, "timestamp": 1.0}]})
    assert n == 1
    assert store.entities[1].position_world == [1.0, 2.0, 3.0]


def test_backfill_positions_does_not_overwrite_existing_position():
    store = make_store()
    obs = Observation(entity_id=1, category="chair", description="a chair", embedding=None, confidence=1.0, room_label=None, timestamp=0.0, position_world=[9.0, 9.0, 9.0])
    store.update([obs], [])

    n = store.backfill_positions({"1": [{"position_world": [1.0, 2.0, 3.0], "frame_id": 5, "timestamp": 1.0}]})
    assert n == 0
    assert store.entities[1].position_world == [9.0, 9.0, 9.0]


def test_backfill_positions_skips_entities_with_no_match():
    store = make_store()
    obs = Observation(entity_id=1, category="chair", description="a chair", embedding=None, confidence=1.0, room_label=None, timestamp=0.0, position_world=None)
    store.update([obs], [])

    n = store.backfill_positions({"999": [{"position_world": [1.0, 2.0, 3.0], "frame_id": 5, "timestamp": 1.0}]})
    assert n == 0
    assert store.entities[1].position_world is None


def test_diff_summary_tracks_added_and_updated():
    store = make_store()
    before = dict(store.entities)
    obs = Observation(entity_id=1, category="phone", description="a phone", embedding=None, confidence=1.0, room_label=None, timestamp=0.0)
    store.update([obs], [])
    diff = store.diff_summary(before)
    assert diff["added"] == [1]
    assert diff["updated"] == []
    assert diff["total_entities"] == 1


def test_new_entity_uses_true_observation_time_not_arrival_time():
    """Regression test for the 2026-08-24 bug: a caller must be able to pass the
    TRUE last-observed time (distinct from when the Observation was constructed),
    and it must be what ends up in last_observed -- since that's what the
    reliability decay term (delta_t_i = now - last_observed) actually uses."""
    store = make_store()
    true_last_observed = 5.0   # object was actually last seen at t=5
    arrival_time = 15.0        # but grounding didn't complete/arrive until t=15
    obs = Observation(
        entity_id=1, category="chair", description="a chair", embedding=None,
        confidence=1.0, room_label=None,
        timestamp=true_last_observed,  # caller passes the TRUE time here, not arrival_time
        first_observed=2.0,
    )
    store.update([obs], [])
    entity = store.entities[1]
    assert entity.last_observed == true_last_observed
    assert entity.last_observed != arrival_time
    assert entity.first_observed == 2.0


def test_first_observed_falls_back_to_timestamp_when_not_provided():
    store = make_store()
    obs = Observation(
        entity_id=1, category="chair", description="a chair", embedding=None,
        confidence=1.0, room_label=None, timestamp=7.0,
        # first_observed omitted -- simulates DAAAM not reporting temporal data
    )
    store.update([obs], [])
    entity = store.entities[1]
    assert entity.first_observed == 7.0
    assert entity.last_observed == 7.0
