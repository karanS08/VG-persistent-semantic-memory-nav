"""pipeline-v8: geometry, position fusion and revisit evidence."""
import math

import numpy as np

from memory_layer.geometry import crop_around, expected_visible, project_point
from memory_layer.models import MemoryEntity
from memory_layer.revisit import RevisitEvidence
from memory_layer.store import MemoryStore, Observation

INTR = {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0}
CAM_H = 1.5


def pose(x=0.0, z=0.0, phi=0.0):
    """Agent at (x, 0, z), rotated phi about +y (camera looks along (-sin phi, -cos phi))."""
    return {"position": [x, 0.0, z], "rotation": [math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0]}


def flat_depth(d=10.0):
    return np.full((480, 640), d, dtype=np.float32)


def backproject(u, v, depth, p):
    """Inverse of the forward model, written exactly as scripts/thor/validate_thor_env.py does."""
    phi = 2 * math.atan2(p["rotation"][2], p["rotation"][0])
    xc, zc = (u - INTR["cx"]) / INTR["fx"] * depth, -depth
    yc = -(v - INTR["cy"]) / INTR["fy"] * depth
    pos = p["position"]
    return [pos[0] + xc * math.cos(phi) + zc * math.sin(phi), pos[1] + CAM_H + yc,
            pos[2] - xc * math.sin(phi) + zc * math.cos(phi)]


def test_projection_round_trip_any_heading():
    for phi in (0.0, 0.7, -2.1, math.pi):
        p = pose(1.0, -2.0, phi)
        for pt in ([1.3, 0.4, -4.5], [0.2, 1.1, -3.0], [2.5, 0.0, -2.5]):
            proj = project_point(pt, p, INTR, CAM_H)
            if proj is None:
                continue
            back = backproject(*proj, p)
            assert np.allclose(back, pt, atol=1e-6)


def test_point_straight_ahead_projects_to_centre_column():
    u, v, d = project_point([0.0, CAM_H, -2.0], pose(), INTR, CAM_H)
    assert abs(u - 320) < 1e-6 and abs(v - 240) < 1e-6 and abs(d - 2.0) < 1e-9
    assert project_point([0.0, 1.0, 2.0], pose(), INTR, CAM_H) is None  # behind


def test_expected_visible_range_fov_occlusion():
    p = pose()
    assert expected_visible([0.0, 0.5, -2.0], p, flat_depth(), INTR, CAM_H)
    assert not expected_visible([0.0, 0.5, -5.0], p, flat_depth(), INTR, CAM_H)   # beyond 3 m
    assert not expected_visible([3.0, 0.5, -0.5], p, flat_depth(), INTR, CAM_H)   # outside the image
    assert not expected_visible([0.0, 0.5, -2.0], p, flat_depth(1.0), INTR, CAM_H)  # wall at 1 m hides it


def test_crop_clipped_to_image():
    rgb = np.zeros((480, 640, 3), np.uint8)
    c = crop_around(rgb, 10, 10, 1.0, 320.0)
    assert c is not None and c.shape[0] <= 480 and c.shape[1] <= 640
    assert crop_around(rgb, -500, -500, 1.0, 320.0) is None


def test_fuse_positions_uses_median_of_history():
    store = MemoryStore(initial_v=1.0, verification_failure_discount=0.5)
    store.update([Observation(entity_id=7, category="sofa", description="a sofa", embedding=[1.0], confidence=1.0,
                              room_label=None, timestamp=0.0, position_world=[9.0, 9.0, 9.0])], [])
    hist = [{"position_world": p} for p in ([1.0, 0.4, 1.0], [1.2, 0.5, 1.1], [5.0, 0.0, 5.0], [1.1, 0.45, 0.9])]
    assert store.fuse_positions({"7": hist, "99": hist}) == 1   # 99 is not in memory: ignored
    assert np.allclose(store.entities[7].position_world, [1.15, 0.425, 1.05])
    store.entities[8] = MemoryEntity(entity_id=1000008, category="x", description="x", position_world=[3, 0, 3])
    store.fuse_positions({"8": hist}, id_offset=1_000_000)  # id offset maps DAAAM id -> store id
    assert store.entities[8].position_world == [3, 0, 3]


def _ent(eid, desc, pos, t):
    return MemoryEntity(entity_id=eid, category=desc, description=desc, embedding=[1.0, 0.0],
                        position_world=pos, last_observed=t, first_observed=t)


def _same(a, b):
    return a.description.split()[-1] == b.description.split()[-1]


def test_revisit_discounts_after_grace_when_location_empty():
    rv = RevisitEvidence(discount=0.5, same_kind=_same, grace_s=10.0)
    rv.begin_query(100.0)
    ents = {1: _ent(1, "a sofa", [0.0, 0.4, -2.0], 50.0)}
    assert rv.step(ents, 101.0, pose(), flat_depth(), INTR, CAM_H) == []      # inspected, pending
    assert rv.step(ents, 105.0, pose(phi=math.pi), flat_depth(), INTR, CAM_H) == []  # turned away, still waiting
    assert rv.step(ents, 112.0, pose(phi=math.pi), flat_depth(), INTR, CAM_H) == [1]
    assert ents[1].v_i == 0.5
    assert rv.step(ents, 130.0, pose(), flat_depth(), INTR, CAM_H) == []      # once per query
    assert ents[1].v_i == 0.5


def test_revisit_no_discount_when_same_kind_seen_nearby():
    rv = RevisitEvidence(discount=0.5, same_kind=_same, grace_s=10.0)
    rv.begin_query(100.0)
    ents = {1: _ent(1, "a sofa", [0.0, 0.4, -2.0], 50.0), 2: _ent(2, "a grey sofa", [0.3, 0.4, -2.4], 104.0)}
    rv.step(ents, 101.0, pose(), flat_depth(), INTR, CAM_H)
    assert rv.step(ents, 120.0, pose(), flat_depth(), INTR, CAM_H) == []
    assert ents[1].v_i == 1.0 and ents[1].last_observed == 50.0  # support is not written back


def test_revisit_other_kind_nearby_does_not_support():
    rv = RevisitEvidence(discount=0.5, same_kind=_same, grace_s=10.0)
    rv.begin_query(100.0)
    ents = {1: _ent(1, "a sofa", [0.0, 0.4, -2.0], 50.0), 2: _ent(2, "a wooden chair", [0.2, 0.4, -2.0], 104.0)}
    rv.step(ents, 101.0, pose(), flat_depth(), INTR, CAM_H)
    assert rv.step(ents, 112.0, pose(), flat_depth(), INTR, CAM_H) == [1]


def test_revisit_occupied_spot_is_never_discounted_even_without_a_caption():
    # depth along the ray matches the remembered position: something is still there, whatever DAAAM captions
    rv = RevisitEvidence(discount=0.5, same_kind=_same, grace_s=10.0)
    rv.begin_query(100.0)
    ents = {1: _ent(1, "a window", [0.0, 0.4, -2.0], 50.0)}
    for t in (101.0, 120.0, 140.0):
        assert rv.step(ents, t, pose(), flat_depth(2.1), INTR, CAM_H) == []
    assert ents[1].v_i == 1.0 and rv.n_occupied_looks == 3


def test_revisit_occupied_look_clears_an_earlier_empty_look():
    rv = RevisitEvidence(discount=0.5, same_kind=_same, grace_s=10.0)
    rv.begin_query(100.0)
    ents = {1: _ent(1, "a sofa", [0.0, 0.4, -2.0], 50.0)}
    rv.step(ents, 101.0, pose(), flat_depth(), INTR, CAM_H)        # looked empty (e.g. depth noise)
    rv.step(ents, 105.0, pose(), flat_depth(2.0), INTR, CAM_H)     # then clearly occupied
    assert rv.step(ents, 120.0, pose(phi=math.pi), flat_depth(), INTR, CAM_H) == []
    assert ents[1].v_i == 1.0


def test_revisit_ignores_occluded_and_live_entries():
    rv = RevisitEvidence(discount=0.5, same_kind=_same, grace_s=0.0)
    rv.begin_query(100.0)
    ents = {1: _ent(1, "a sofa", [0.0, 0.4, -2.0], 50.0), 2: _ent(2, "a bed", [0.5, 0.4, -2.0], 101.0)}
    assert rv.step(ents, 102.0, pose(), flat_depth(1.0), INTR, CAM_H) == []  # wall in front: not inspected
    assert rv.step(ents, 103.0, pose(), flat_depth(), INTR, CAM_H) == [1]    # entry 2 is live: never a candidate


def test_crop_object_frames_upper_body_when_centre_is_below_image():
    from memory_layer.geometry import crop_object
    rgb = np.zeros((480, 640, 3), np.uint8)
    p = pose()
    pt = [0.0, 0.4, -1.0]  # 1 m ahead, low: centre projects below the bottom edge
    assert project_point(pt, p, INTR, CAM_H)[1] > 480
    c = crop_object(rgb, pt, p, INTR, CAM_H)
    assert c is not None and c.shape[0] >= 24
    assert not expected_visible(pt, p, flat_depth(), INTR, CAM_H, min_range=0.0, margin_px=0)
    assert expected_visible(pt, p, flat_depth(), INTR, CAM_H, min_range=0.0, margin_px=0, clamp_vertical=True)
