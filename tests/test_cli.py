"""Tests of the command line: exit codes and output, with LangSmith replaced by fakes."""

import pytest

from eval_kit import cli

EVAL_TOML = """
[project]
profile = "rag"
target = "app:target"
evaluators = "app:EVALUATORS"

[dataset]
name = "refund-scenarios"
tag = "v3"

[judge]
model = "gpt-4.1-mini-2025-04-14"

[run]
budget_usd = 2.0

[rules]
secondary = ["message"]

[rules.safety]
no_ticket = "ticket without request"

[rules.primary]
decision = 0
"""
BASELINE = '\n[baseline]\nexperiment = "baseline-exp"\n'

GOOD = {name: {f"ex{i}": [1, 1, 1] for i in range(10)} for name in ("no_ticket", "decision", "message")}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project folder with eval.toml, and LangSmith calls replaced by fakes.

    Returns:
        A function writing eval.toml and setting the candidate scores; it returns
        the path to pass to --config.
    """
    monkeypatch.setattr(cli, "Client", lambda: None)
    monkeypatch.setattr(cli, "run_experiment", lambda config, client: "candidate-exp")

    def setup(toml: str, candidate_scores: dict) -> str:
        scores = {"baseline-exp": GOOD, "candidate-exp": candidate_scores}
        monkeypatch.setattr(cli, "experiment_scores", lambda name: scores[name])
        path = tmp_path / "eval.toml"
        path.write_text(toml, encoding="utf-8")
        return str(path)

    return setup


def test_gate_passes_with_exit_code_0(project, capsys):
    config = project(EVAL_TOML + BASELINE, GOOD)
    assert cli.main(["gate", "--config", config]) == cli.EXIT_PASSED
    assert "GATE PASSED" in capsys.readouterr().out


def test_safety_failure_exits_with_code_1(project, capsys):
    candidate = {**GOOD, "no_ticket": {**GOOD["no_ticket"], "ex0": [1, 0, 1]}}
    config = project(EVAL_TOML + BASELINE, candidate)
    assert cli.main(["gate", "--config", config]) == cli.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "GATE FAILED" in out and "no_ticket" in out


def test_missing_baseline_exits_with_code_2(project, capsys):
    config = project(EVAL_TOML, GOOD)
    assert cli.main(["gate", "--config", config]) == cli.EXIT_ERROR
    assert "no [baseline]" in capsys.readouterr().err


def test_invalid_config_exits_with_code_2(project, capsys):
    config = project(EVAL_TOML.replace("budget_usd = 2.0", ""), GOOD)
    assert cli.main(["gate", "--config", config]) == cli.EXIT_ERROR
    assert "run.budget_usd" in capsys.readouterr().err


def test_project_folder_becomes_importable(project, monkeypatch):
    import sys
    config = project(EVAL_TOML + BASELINE, GOOD)
    monkeypatch.setattr(sys, "path", list(sys.path))  # restored after the test
    cli.main(["gate", "--config", config])
    assert sys.path[0] == str(cli.Path(config).parent)


def test_run_prints_the_experiment_name(project, capsys):
    config = project(EVAL_TOML, GOOD)
    assert cli.main(["run", "--config", config]) == cli.EXIT_PASSED
    assert "candidate-exp" in capsys.readouterr().out


def test_project_env_file_is_loaded_without_overriding(project, monkeypatch, tmp_path):
    config = project(EVAL_TOML + BASELINE, GOOD)
    (tmp_path / ".env").write_text("EVALKIT_TEST_KEY=from-file\nEVALKIT_TEST_SET=from-file\n")
    monkeypatch.delenv("EVALKIT_TEST_KEY", raising=False)
    monkeypatch.setenv("EVALKIT_TEST_SET", "from-ci-secret")
    cli.main(["gate", "--config", config])
    import os
    assert os.environ["EVALKIT_TEST_KEY"] == "from-file"        # missing: read from .env
    assert os.environ["EVALKIT_TEST_SET"] == "from-ci-secret"   # already set: never overridden
    monkeypatch.delenv("EVALKIT_TEST_KEY")
