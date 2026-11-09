"""Tests of eval.toml loading: every invalid config must stop the tool with a clear message."""

import pytest
from pydantic import ValidationError

from eval_kit.config import ConfigError, load_config

# One valid config; each test changes exactly one thing in it.
VALID = """
[project]
profile = "agent"
target  = "refund_agent.eval:target"
evaluators = "refund_agent.eval:EVALUATORS"

[dataset]
name = "refund-scenarios"
tag  = "v3"

[judge]
model = "gpt-4.1-mini-2025-04-14"

[run]
budget_usd = 2.0

[tools]
create_refund_ticket = "simulated"

[rules.safety]
trajectoire_dag = "argument inventé"
"""

TOOLS = '[tools]\ncreate_refund_ticket = "simulated"\n'


def write(tmp_path, text):
    """Write a config file in pytest's temporary folder and return its path."""
    path = tmp_path / "eval.toml"
    path.write_text(text, encoding="utf-8")
    return path


def error_message(path) -> str:
    """Load a config that must be refused and return the ConfigError message."""
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    return str(excinfo.value)


def test_valid_config_loads_with_defaults(tmp_path):
    config = load_config(write(tmp_path, VALID))
    assert config.run.repetitions == 3
    assert config.judge.temperature == 0.0
    assert config.dataset.split == "holdout"


def test_every_invalid_field_is_reported(tmp_path):
    text = VALID.replace("budget_usd = 2.0", "").replace(
        'model = "gpt-4.1-mini-2025-04-14"',
        'model = "gpt-4.1-mini-2025-04-14"\ntemperature = 0.5',
    )
    message = error_message(write(tmp_path, text))
    assert "run.budget_usd" in message
    assert "judge.temperature" in message


def test_typo_in_a_key_is_refused(tmp_path):
    text = VALID.replace("budget_usd", "budjet_usd")
    message = error_message(write(tmp_path, text))
    assert "run.budjet_usd: Extra inputs are not permitted" in message
    assert "run.budget_usd: Field required" in message


def test_agent_without_tools_is_refused(tmp_path):
    text = VALID.replace(TOOLS, "")
    message = error_message(write(tmp_path, text))
    assert "profile 'agent' requires a [tools] whitelist" in message


def test_rag_without_tools_is_accepted(tmp_path):
    text = VALID.replace('profile = "agent"', 'profile = "rag"').replace(TOOLS, "")
    config = load_config(write(tmp_path, text))
    assert config.project.profile == "rag"
    assert config.tools == {}


def test_evaluator_in_two_categories_is_refused(tmp_path):
    text = VALID + "\n[rules.primary]\ntrajectoire_dag = 0\n"
    message = error_message(write(tmp_path, text))
    assert "safety/primary: ['trajectoire_dag']" in message


def test_broken_toml_is_reported(tmp_path):
    message = error_message(write(tmp_path, "[project\n"))
    assert "is not valid TOML" in message


def test_missing_file_is_reported(tmp_path):
    message = error_message(tmp_path / "absent.toml")
    assert "config file not found" in message


def test_config_is_read_only(tmp_path):
    config = load_config(write(tmp_path, VALID))
    with pytest.raises(ValidationError):
        config.run.repetitions = 10

def test_baseline_is_optional(tmp_path):
    config = load_config(write(tmp_path, VALID))
    assert config.baseline is None


def test_baseline_is_read(tmp_path):
    text = VALID + '\n[baseline]\nexperiment = "prompt-v6-7521d4ba"\n'
    config = load_config(write(tmp_path, text))
    assert config.baseline.experiment == "prompt-v6-7521d4ba"
