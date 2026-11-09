"""Tests of score grouping: no network, fake runs only."""

from types import SimpleNamespace as NS

import pytest

from eval_kit.experiments.errors import RunError
from eval_kit.experiments.results import group_scores, incomplete, wait_for_complete_scores


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



# ── completeness: never decide on a partial read ───────────────────────────

FULL = {"dag": {"ex1": [1, 1, 1], "ex2": [1, 1, 1]}}
PARTIAL = {"dag": {"ex1": [1, 1, 1], "ex2": [1]}}   # 2 scores of ex2 not stored yet


def test_complete_scores_have_no_problem():
    assert incomplete(FULL, ["dag"], ["ex1", "ex2"], 3) == []


def test_missing_runs_are_reported():
    assert incomplete(PARTIAL, ["dag"], ["ex1", "ex2"], 3) == ["dag: 1 scenario(s) with fewer than 3 scores"]


def test_missing_evaluator_is_reported_as_incomplete():
    assert incomplete({}, ["dag"], ["ex1"], 3) == ["dag: 1 scenario(s) with fewer than 3 scores"]


class FakeTime:
    """A clock that only moves when sleep() is called."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.sleeps += 1


def test_wait_returns_once_the_scores_are_complete():
    reads = iter([PARTIAL, PARTIAL, FULL])
    fake = FakeTime()
    scores = wait_for_complete_scores("exp", ["dag"], ["ex1", "ex2"], 3,
                                      fetch=lambda name: next(reads),
                                      sleep=fake.sleep, clock=fake.clock)
    assert scores == FULL
    assert fake.sleeps == 2


def test_wait_gives_up_after_the_timeout():
    fake = FakeTime()
    with pytest.raises(RunError, match="still incomplete after 20s"):
        wait_for_complete_scores("exp", ["dag"], ["ex1", "ex2"], 3, timeout=20, interval=5,
                                 fetch=lambda name: PARTIAL, sleep=fake.sleep, clock=fake.clock)
