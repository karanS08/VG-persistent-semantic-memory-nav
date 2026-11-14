import pytest

from memory_layer.mobility import MobilityPriors, update_mobility_from_verification


def make_priors():
    return MobilityPriors({"phone": 0.85, "wall": 0.0, "kitchen island": 0.02}, fallback=0.5)


def test_exact_match():
    p = make_priors()
    assert p.mobility_for("phone") == 0.85
    assert p.mobility_for("wall") == 0.0


def test_substring_match_from_description():
    p = make_priors()
    assert p.mobility_for("A black phone on the table") == 0.85


def test_longest_match_wins():
    p = make_priors()
    # "kitchen island" should win over any shorter contained term
    assert p.mobility_for("a large kitchen island in the center") == 0.02


def test_unmatched_falls_back():
    p = make_priors()
    assert p.mobility_for("a mysterious unrecognizable object") == 0.5


def test_real_mobility_priors_file_loads_and_has_sane_bounds():
    priors = MobilityPriors.load("configs/mobility_priors.json")
    assert priors.mobility_for("wall") <= 0.1  # should be near-static
    assert priors.mobility_for("phone") >= 0.6  # should be highly mobile
    for name, value in priors._categories.items():
        assert 0.0 <= value <= 1.0, f"{name} out of [0,1] bounds: {value}"


def test_update_hook_is_explicitly_not_implemented():
    p = make_priors()
    with pytest.raises(NotImplementedError):
        update_mobility_from_verification(p, "phone", verification_succeeded=False)
