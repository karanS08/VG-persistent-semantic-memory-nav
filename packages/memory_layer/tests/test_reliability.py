import math

import pytest

from memory_layer.models import MemoryEntity
from memory_layer.reliability import clip01, compute_reliability, compute_lambda_i


def test_clip01():
    assert clip01(-0.5) == 0.0
    assert clip01(1.5) == 1.0
    assert clip01(0.3) == 0.3


def test_compute_lambda_i():
    assert compute_lambda_i(mobility=1.0, lambda_base=0.001) == pytest.approx(0.001)
    assert compute_lambda_i(mobility=0.0, lambda_base=0.001) == 0.0
    assert compute_lambda_i(mobility=0.5, lambda_base=0.002) == pytest.approx(0.001)


def test_compute_reliability_hand_computed():
    """Independently reimplements the formula from CLAUDE.md Section 6 (not just
    re-calling compute_reliability) so this test can actually catch a bug in the
    implementation, not just confirm it agrees with itself.
    """
    entity = MemoryEntity(
        entity_id=1, category="phone", description="a phone",
        c_i=1.0, n_i=3, v_i=1.0, first_observed=0.0, last_observed=0.0,
    )
    now = 100.0
    mobility = 1.0
    alpha = 0.5
    lambda_base = 0.001

    # hand-computed independently:
    diminishing_returns = 1 - math.exp(-alpha * 3)          # 1 - exp(-1.5) = 0.7768698...
    confirmed_quality = min(1.0, max(0.0, 1.0 * diminishing_returns))
    lambda_i = lambda_base * mobility                        # 0.001
    delta_t = now - 0.0                                      # 100.0
    temporal_decay = math.exp(-lambda_i * delta_t)            # exp(-0.1) = 0.9048374...
    expected = 1.0 * confirmed_quality * temporal_decay

    result = compute_reliability(entity, now, mobility, alpha, lambda_base)
    assert result == pytest.approx(expected, rel=1e-9)
    assert result == pytest.approx(0.702941, abs=1e-5)


def test_compute_reliability_zero_mobility_never_decays():
    """A static-category entity (mobility=0) should have full reliability even
    after a huge elapsed time, since lambda_i=0 -> exp(0)=1 regardless of delta_t.
    """
    entity = MemoryEntity(
        entity_id=2, category="wall", description="a wall",
        c_i=1.0, n_i=5, v_i=1.0, first_observed=0.0, last_observed=0.0,
    )
    result_soon = compute_reliability(entity, now=1.0, mobility=0.0, alpha=0.5, lambda_base=0.001)
    result_later = compute_reliability(entity, now=1e9, mobility=0.0, alpha=0.5, lambda_base=0.001)
    assert result_soon == pytest.approx(result_later, rel=1e-9)


def test_compute_reliability_v_i_scales_linearly():
    entity_full = MemoryEntity(entity_id=3, category="x", description="x", c_i=1.0, n_i=3, v_i=1.0, last_observed=0.0)
    entity_half = MemoryEntity(entity_id=4, category="x", description="x", c_i=1.0, n_i=3, v_i=0.5, last_observed=0.0)
    full = compute_reliability(entity_full, now=0.0, mobility=0.5, alpha=0.5, lambda_base=0.001)
    half = compute_reliability(entity_half, now=0.0, mobility=0.5, alpha=0.5, lambda_base=0.001)
    assert half == pytest.approx(full * 0.5, rel=1e-9)


def test_compute_reliability_single_observation():
    """n_i=1 (a brand-new entity, one observation): diminishing_returns should be
    the smallest nonzero value, not near max even at moderate alpha.
    """
    entity = MemoryEntity(entity_id=5, category="x", description="x", c_i=1.0, n_i=1, v_i=1.0, last_observed=0.0)
    result = compute_reliability(entity, now=0.0, mobility=0.0, alpha=0.5, lambda_base=0.001)
    expected_diminishing = 1 - math.exp(-0.5)  # 0.3934693...
    assert result == pytest.approx(expected_diminishing, rel=1e-9)
