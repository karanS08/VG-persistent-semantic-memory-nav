import math

import pytest

from memory_layer.relevance import cosine_similarity, compute_relevance


def test_cosine_similarity_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_hand_computed():
    a = [1.0, 2.0]
    b = [2.0, 1.0]
    # hand-computed: dot=1*2+2*1=4, |a|=sqrt(5), |b|=sqrt(5) -> 4/5 = 0.8
    expected = 4.0 / (math.sqrt(5) * math.sqrt(5))
    assert cosine_similarity(a, b) == pytest.approx(expected)
    assert cosine_similarity(a, b) == pytest.approx(0.8)


def test_cosine_similarity_zero_vector_is_safe():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_compute_relevance_missing_embedding_returns_zero():
    assert compute_relevance([1.0, 0.0], None) == 0.0


def test_compute_relevance_matches_cosine_similarity():
    q = [1.0, 2.0, 3.0]
    d = [3.0, 2.0, 1.0]
    assert compute_relevance(q, d) == pytest.approx(cosine_similarity(q, d))
