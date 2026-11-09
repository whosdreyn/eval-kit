"""Tests of the runner: no network, no agent (fake LangSmith client, monkeypatch)."""

from types import SimpleNamespace as NS

import pytest

from eval_kit.config import EvalConfig, load_config
from eval_kit.experiments import runner
from eval_kit.experiments.runner import RunError, check_evaluators, load_examples, resolve

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


class FakeEvaluateClient(FakeClient):
    """A fake client that also accepts evaluate() and remembers its arguments."""

    def evaluate(self, target, **kwargs):
        self.evaluate_call = kwargs
        return NS(experiment_name="evalkit-123")


def trajectoire_dag(outputs):
    """A fake evaluator: only its name matters to check_evaluators."""
    return True


# ── resolve ────────────────────────────────────────────────────────────────

def test_resolve_returns_the_named_object():
    assert resolve("eval_kit.config:load_config") is load_config


def test_reference_without_colon_is_reported():
    with pytest.raises(RunError, match="must be module.path:name"):
        resolve("eval_kit.config")


def test_unknown_module_is_reported():
    with pytest.raises(RunError, match="cannot import 'eval_kit.unknown'"):
        resolve("eval_kit.unknown:load_config")


def test_unknown_name_is_reported():
    with pytest.raises(RunError, match="has no attribute 'absent'"):
        resolve("eval_kit.config:absent")


# ── load_examples ──────────────────────────────────────────────────────────

def test_frozen_version_and_split_are_requested():
    client = FakeClient([NS(id="ex1")])
    load_examples(CONFIG, client)
    assert client.call == {"dataset_name": "refund-scenarios", "as_of": "v3", "splits": ["holdout"]}


def test_examples_are_returned():
    client = FakeClient([NS(id="ex1"), NS(id="ex2")])
    examples = load_examples(CONFIG, client)
    assert [ex.id for ex in examples] == ["ex1", "ex2"]


def test_empty_version_is_refused():
    with pytest.raises(RunError, match="no example in dataset 'refund-scenarios' at tag 'v3'"):
        load_examples(CONFIG, FakeClient([]))


# ── check_evaluators ───────────────────────────────────────────────────────

def test_every_rule_has_its_evaluator():
    check_evaluators(CONFIG, [trajectoire_dag])  # must not raise


def test_missing_evaluator_is_reported():
    with pytest.raises(RunError, match="rules with no matching evaluator: trajectoire_dag"):
        check_evaluators(CONFIG, [])


# ── run_experiment ─────────────────────────────────────────────────────────

def test_run_experiment_passes_the_config_to_langsmith(monkeypatch):
    fake = {"app:target": lambda inputs: {}, "app:EVALUATORS": [trajectoire_dag]}
    monkeypatch.setattr(runner, "resolve", lambda spec: fake[spec])
    waited = {}
    monkeypatch.setattr(runner, "wait_for_complete_scores",
                        lambda experiment, **kwargs: waited.update(experiment=experiment, **kwargs))
    client = FakeEvaluateClient([NS(id="ex1")])

    name = runner.run_experiment(CONFIG, client)

    assert name == "evalkit-123"
    call = client.evaluate_call
    assert call["num_repetitions"] == 3
    assert call["max_concurrency"] == 4
    assert call["metadata"]["judge_model"] == "gpt-4.1-mini-2025-04-14"
    assert call["metadata"]["dataset_tag"] == "v3"
    # The gate must never read a partial experiment: run_experiment waits first.
    assert waited == {"experiment": "evalkit-123", "evaluators": {"trajectoire_dag"},
                      "scenarios": ["ex1"], "repetitions": 3}
