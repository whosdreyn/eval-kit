"""Tests of the gate's power: which drop can it prove, and how many scenarios does that take?"""

import pytest

from eval_kit.stats import bootstrap_diff_ci, detectable_drop, scenarios_needed


def test_ten_scenarios_prove_a_drop_only_from_four_broken_ones():
    # (1 - 3/10)^10 = 2.8% >= 2.5%: 3 broken scenarios are not enough.
    # (1 - 4/10)^10 = 0.6% <  2.5%: 4 are.
    assert detectable_drop(10) == 4


def test_more_scenarios_detect_smaller_drops():
    assert detectable_drop(40) == 4          # 4 out of 40 = 10 points, instead of 40 points


def test_no_scenario_detects_nothing():
    assert detectable_drop(0) is None


def test_the_formula_agrees_with_the_real_bootstrap():
    # The rule is not a guess: the gate's own bootstrap blocks from exactly that point.
    n = 10
    m = detectable_drop(n)
    baseline = {f"s{i}": [1] for i in range(n)}

    def blocked(broken):
        candidate = {f"s{i}": [0] if i < broken else [1] for i in range(n)}
        return bootstrap_diff_ci(baseline, candidate)[2] < 0

    assert not blocked(m - 1)
    assert blocked(m)


def test_scenarios_needed_for_a_given_drop():
    assert scenarios_needed(0.10) == 36      # to prove a 10-point drop
    assert scenarios_needed(0.20) == 17      # to prove a 20-point drop


@pytest.mark.parametrize("drop", [0, 1, -0.1, 1.5])
def test_drop_must_be_strictly_between_0_and_1(drop):
    with pytest.raises(ValueError):
        scenarios_needed(drop)
