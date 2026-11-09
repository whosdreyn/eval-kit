"""Tests of the runner: no network, no agent."""

import pytest

from eval_kit import runner
from eval_kit.runner import RunError, resolve


def test_resolve_returns_the_named_object():
    from eval_kit.config import load_config
    assert resolve("eval_kit.config:load_config") is load_config


def test_unknown_module_is_reported():
    with pytest.raises(RunError, match="Cannot import 'eval_kit.unknown'"):
        resolve("eval_kit.unknown:load_config")

def test_unknown_name_is_reported():
    with pytest.raises(RunError, match="has no attribute 'absent'"):
        resolve("eval_kit.config:absent")


from types import SimpleNamespace as NS

from eval_kit.config import EvalConfig
from eval_kit.runner import load_examples

CONFIG = EvalConfig(
    project={"profile": "rag", "target": "app:target", "evaluators": "app:EVALUATORS"},
    dataset={"name": "refund-scenarios", "tag": "v3"},
    judge={"model": "gpt-4.1-mini-2025-04-14"},
    run={"budget_usd": 2},
    rules={"safety": {"trajectoire_dag": "x"}},
)


class FakeClient:
    """Stands in for langsmith.Client: returns fixed examples, remembers the call."""

    def __init__(self, examples):
        self.examples = examples
        self.call = None

    def list_examples(self, **kwargs):
        self.call = kwargs
        return iter(self.examples)


def test_frozen_version_and_split_are_requested():
    client = FakeClient([NS(id="ex1")])
    load_examples(CONFIG, client)
    assert client.call == {"dataset_name": "refund-scenarios", "as_of": "v3", "splits": ["holdout"]}

def test_examples_are_returned():
    client = FakeClient([NS(id="ex1"), NS(id="ex2")])
    examples = load_examples(CONFIG, client)
    assert [ex.id for ex in examples] == ["ex1", "ex2"]

def test_empty_version_is_refused():
    client = FakeClient([])
    with pytest.raises(RunError, match="No example in dataset 'refund-scenarios' version 'v3' split 'holdout'"):
        load_examples(CONFIG, client)

from eval_kit.runner import check_evaluators


def trajectoire_dag(outputs):
    return True


def test_every_rule_has_its_evaluator():
    check_evaluators(CONFIG, [trajectoire_dag])      # ne doit rien lever

def test_missing_evaluator_is_reported():
    with pytest.raises(RunError, match="Rules with no matching evaluator: trajectoire_dag"):
        check_evaluators(CONFIG, [])



class FakeEvaluateClient(FakeClient):
    """A fake client that also accepts evaluate() and remembers its arguments."""

    def evaluate(self, target, **kwargs):
        self.evaluate_call = kwargs
        return NS(experiment_name="evalkit-123")


def test_run_experiment_passes_the_config_to_langsmith(monkeypatch):
    fake = {"app:target": lambda inputs: {}, "app:EVALUATORS": [trajectoire_dag]}
    monkeypatch.setattr(runner, "resolve", lambda spec: fake[spec])
    client = FakeEvaluateClient([NS(id="ex1")])

    name = runner.run_experiment(CONFIG, client)

    assert name == "evalkit-123"
    call = client.evaluate_call
    assert call["num_repetitions"] == 3
    assert call["max_concurrency"] == 4
    assert call["metadata"]["judge_model"] == "gpt-4.1-mini-2025-04-14"
    assert call["metadata"]["dataset_tag"] == "v3"