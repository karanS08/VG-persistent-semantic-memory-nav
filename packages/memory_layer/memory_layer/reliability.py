"""Reliability score C_i, exactly per CLAUDE.md Section 6:

    C_i = v_i * clip[0,1]( c_i * (1 - exp(-alpha * n_i)) ) * exp(-lambda_i * delta_t_i)
    lambda_i = lambda_base * mobility(category_i)
"""
from __future__ import annotations

import math

from .models import MemoryEntity


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_lambda_i(mobility: float, lambda_base: float) -> float:
    return lambda_base * mobility


def compute_reliability(
    entity: MemoryEntity,
    now: float,
    mobility: float,
    alpha: float,
    lambda_base: float,
) -> float:
    diminishing_returns = 1.0 - math.exp(-alpha * entity.n_i)
    confirmed_quality = clip01(entity.c_i * diminishing_returns)

    lambda_i = compute_lambda_i(mobility, lambda_base)
    delta_t_i = max(0.0, now - entity.last_observed)
    temporal_decay = math.exp(-lambda_i * delta_t_i)

    return entity.v_i * confirmed_quality * temporal_decay
