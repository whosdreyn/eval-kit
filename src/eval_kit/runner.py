"""Run the project's agent on its frozen dataset and record a LangSmith experiment."""

import importlib
import os
import subprocess

from langsmith import Client

from eval_kit.config import EvalConfig


class RunError(Exception):
    """Raised when the evaluation cannot start: nothing has been spent yet."""


def resolve(spec: str):
    """Return the Python object named by "module.path:name".

    Args:
        spec: A reference such as "evaluation.ls_evals:target".

    Returns:
        The object: a function for target, a list for evaluators.

    Raises:
        RunError: if the module cannot be imported or has no such name.
    """
    try:
        module_path, name = spec.split(":", 1)
    except ValueError as e:
        raise RunError(f"Invalid spec {spec!r}: must be module.path:name") from e
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise RunError(f"Cannot import {module_path!r}: {e}") from e
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise RunError(f"Module {module_path!r} has no attribute {name!r}") from e


def load_examples(config: EvalConfig, client: Client) -> list:
    """Return the examples of the frozen dataset version and split.

    Raises:
        RunError: if the version or split holds no example.
    """
    examples=client.list_examples(
    dataset_name=config.dataset.name,
    as_of=config.dataset.tag,              # la version taguée "v3", pas la dernière
    splits=[config.dataset.split],      # seulement les exemples de ce split
    )
    examples = list(examples)
    if not examples:
        raise RunError(
            f"No example in dataset {config.dataset.name!r} "
            f"version {config.dataset.tag!r} split {config.dataset.split!r}"
        )
    return examples


def check_evaluators(config: EvalConfig, evaluators: list) -> None:
    """Fail before spending anything if a rule has no evaluator function.

    Raises:
        RunError: listing every rule name that matches no evaluator.
    """
    evaluator_names = {evaluator.__name__ for evaluator in evaluators}
    rule_names = set(config.rules.safety) | set(config.rules.primary) | set(config.rules.secondary)
    missing = set(rule_names) - evaluator_names
    if missing:
        raise RunError(
            f"Rules with no matching evaluator: {', '.join(sorted(missing))}"
        )




def git_commit() -> str:
    """Return the current git commit of the evaluated project, or "unknown"."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"




def run_experiment(config: EvalConfig, client: Client) -> str:
    """Run the target on the frozen dataset and return the experiment name."""
    target = resolve(config.project.target)
    evaluators = resolve(config.project.evaluators)
    check_evaluators(config, evaluators)
    examples = load_examples(config, client)

    os.environ["EVAL_MODE"] = "1"      # section 3.1: the agent must not act for real

    results = client.evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        num_repetitions=config.run.repetitions,
        max_concurrency=config.run.max_concurrency,
        experiment_prefix='evalkit',
        metadata={
            "git_commit": git_commit(),
            "dataset": config.dataset.name,
            "dataset_tag": config.dataset.tag,
            "dataset_split": config.dataset.split,
            "judge_model": config.judge.model,
            "repetitions": config.run.repetitions,
        },
    )
    return results.experiment_name