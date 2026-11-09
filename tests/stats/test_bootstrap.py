"""Tests of the bootstrap intervals: known answers, reproducibility, errors."""

import pytest

from eval_kit.stats import bootstrap_diff_ci, bootstrap_mean_ci


def test_constant_scores_have_a_zero_width_interval():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)


def test_mean_interval_contains_the_mean():
    mean, low, high = bootstrap_mean_ci([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    assert low <= mean <= high


def test_empty_scores_are_refused():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])


def test_same_seed_gives_the_same_interval():
    a = {f"ex{i}": [1, 0, 1] for i in range(10)}
    b = {f"ex{i}": [1, 1, 1] for i in range(10)}
    assert bootstrap_diff_ci(a, b, seed=7) == bootstrap_diff_ci(a, b, seed=7)


def test_diff_is_candidate_minus_baseline():
    baseline = {f"ex{i}": [1, 1, 1] for i in range(10)}
    candidate = {f"ex{i}": [0, 0, 0] if i < 4 else [1, 1, 1] for i in range(10)}
    diff, low, high = bootstrap_diff_ci(baseline, candidate)
    assert diff == pytest.approx(-0.4)
    assert low <= diff <= high < 0


def test_no_common_scenario_is_refused():
    with pytest.raises(ValueError, match="no scenario in common"):
        bootstrap_diff_ci({"ex1": [1]}, {"ex2": [1]})
