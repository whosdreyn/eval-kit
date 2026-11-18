"""Tests of the consistency measure: does the agent pass EVERY repetition of a scenario?"""

from eval_kit.stats import consistency


def test_scenarios_are_sorted_into_three_groups():
    by_scenario = {
        "ex1": [1, 1, 1],   # always passes
        "ex2": [1, 0, 1],   # flaky: the v6 "casque remboursable ?" case
        "ex3": [0, 0, 0],   # always fails
    }
    result = consistency(by_scenario)
    assert result["always"] == ["ex1"]
    assert result["flaky"] == ["ex2"]
    assert result["never"] == ["ex3"]


def test_pass_hat_k_is_the_share_of_scenarios_passed_on_every_run():
    by_scenario = {"ex1": [1, 1, 1], "ex2": [1, 0, 1], "ex3": [1, 1, 1], "ex4": [0, 0, 0]}
    assert consistency(by_scenario)["pass_hat_k"] == 0.5     # 2 scenarios out of 4


def test_mean_can_hide_what_pass_hat_k_shows():
    # 29 runs out of 30 pass: the mean looks excellent (96.7%)...
    by_scenario = {f"ex{i}": [1, 1, 1] for i in range(9)}
    by_scenario["ex9"] = [1, 0, 1]
    mean = sum(s for scores in by_scenario.values() for s in scores) / 30
    assert round(mean, 3) == 0.967
    # ...but one scenario in ten fails a real customer one time in three.
    assert consistency(by_scenario)["pass_hat_k"] == 0.9
    assert consistency(by_scenario)["flaky"] == ["ex9"]


def test_only_a_full_score_counts_as_a_pass():
    # trajectoire_dag gives 0.5 for a non-critical fault: not a clean pass.
    assert consistency({"ex1": [1, 0.5, 1]})["flaky"] == ["ex1"]


def test_groups_are_sorted_for_a_stable_report():
    result = consistency({"b": [1], "a": [1], "c": [0]})
    assert result["always"] == ["a", "b"]


def test_no_scenario_gives_no_measure():
    assert consistency({})["pass_hat_k"] is None
