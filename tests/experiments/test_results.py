"""Tests of score grouping: no network, fake runs only."""

from types import SimpleNamespace as NS

from eval_kit.experiments.results import group_scores


def run(example, **scores):
    """A fake run: an example id and one score per evaluator."""
    return NS(reference_example_id=example,
              feedback_stats={key: NS(avg=value) for key, value in scores.items()})


def test_scores_are_grouped_by_evaluator_then_scenario():
    runs = [run("ex1", decision=1.0, dag=1.0),
            run("ex1", decision=0.0, dag=1.0),
            run("ex2", decision=1.0)]
    assert group_scores(runs) == {
        "decision": {"ex1": [1.0, 0.0], "ex2": [1.0]},
        "dag": {"ex1": [1.0, 1.0]},
    }

def test_run_without_example_is_ignored():
    assert group_scores([run(None, decision=1.0)]) == {}


def test_run_without_feedback_is_ignored():
    assert group_scores([NS(reference_example_id="ex1", feedback_stats=None)]) == {}


def test_evaluator_error_is_not_counted_as_zero():
    # An evaluator that crashed leaves no score: the scenario must look MISSING
    # (the gate then fails closed), never scored 0 (that would blame the agent).
    assert group_scores([run("ex1", decision=None)]) == {}
