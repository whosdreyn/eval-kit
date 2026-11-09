"""Run the project's agent on its frozen dataset and record a LangSmith experiment.

Everything that can fail for free (a wrong reference, a missing evaluator, an
empty dataset version) is checked before the paid evaluate() call.
"""

import importlib
import os
import subprocess

from langsmith import Client

from eval_kit.config import EvalConfig
from eval_kit.experiments.errors import RunError
from eval_kit.experiments.results import wait_for_complete_scores


def resolve(spec: str):
    """Return the Python object named by "module.path:name".

    Args:
        spec: A reference such as "evaluation.ls_evals:target".

    Returns:
        The object: a function for target, a list for evaluators.

    Raises:
        RunError: if the reference is malformed, the module cannot be imported,
            or the module has no such name.
    """
    try:
        module_path, name = spec.split(":", 1)
    except ValueError as e:
        raise RunError(f"invalid reference {spec!r}: must be module.path:name") from e
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise RunError(f"cannot import {module_path!r}: {e}") from e
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise RunError(f"module {module_path!r} has no attribute {name!r}") from e


def load_examples(config: EvalConfig, client: Client) -> list:
    """Return the examples of the frozen dataset version and split.

    Args:
        config: The validated eval.toml.
        client: LangSmith client (a fake one in tests).

    Returns:
        The examples of the dataset at its tag, in its split.

    Raises:
        RunError: if that version and split hold no example.
    """
    examples = list(client.list_examples(
        dataset_name=config.dataset.name,
        as_of=config.dataset.tag,
        splits=[config.dataset.split],
    ))
    if not examples:
        raise RunError(
            f"no example in dataset {config.dataset.name!r} "
            f"at tag {config.dataset.tag!r} in split {config.dataset.split!r}"
        )
    return examples


def rule_names(config: EvalConfig) -> set[str]:
    """Every evaluator named in [rules]: safety, primary and secondary.

    Args:
        config: The validated eval.toml.

    Returns:
        The evaluator names the gate needs scores for.
    """
    rules = config.rules
    return set(rules.safety) | set(rules.primary) | set(rules.secondary)


def check_evaluators(config: EvalConfig, evaluators: list) -> None:
    """Fail before spending anything if a rule has no evaluator function.

    Args:
        config: The validated eval.toml.
        evaluators: The evaluator functions of the project.

    Raises:
        RunError: listing every rule name that matches no evaluator function.
    """
    evaluator_names = {evaluator.__name__ for evaluator in evaluators}
    missing = rule_names(config) - evaluator_names
    if missing:
        raise RunError(f"rules with no matching evaluator: {', '.join(sorted(missing))}")


def judges_of(evaluators: list) -> dict:
    """Find the LLM judges used by the evaluators.

    Convention: an evaluator that calls a Judge exposes it as an attribute,
    e.g. `message_conforme.judge = POLITE`.

    Args:
        evaluators: The evaluator functions of the project.

    Returns:
        {evaluator name: its Judge}.
    """
    return {e.__name__: e.judge for e in evaluators if getattr(e, "judge", None) is not None}


def check_judges(config: EvalConfig, evaluators: list) -> None:
    """Fail before spending anything if a judge is not the model declared in eval.toml.

    [judge] model is what eval.toml promises (dated, not the agent's model): a judge
    built with another model would silently break that promise.

    Args:
        config: The validated eval.toml.
        evaluators: The evaluator functions of the project.

    Raises:
        RunError: listing every evaluator whose judge uses another model.
    """
    wrong = {name: judge.model for name, judge in judges_of(evaluators).items()
             if judge.model != config.judge.model}
    if wrong:
        raise RunError(f"judges not using [judge] model {config.judge.model!r}: "
                       + ", ".join(f"{name} uses {model!r}" for name, model in sorted(wrong.items())))


def git_commit() -> str:
    """Return the current git commit of the evaluated project.

    Returns:
        The commit hash, or "unknown" outside a git repository.
    """
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_experiment(config: EvalConfig, client: Client) -> str:
    """Run the target on the frozen dataset and record the experiment in LangSmith.

    Sets EVAL_MODE=1 so the agent knows it must not act for real (design doc,
    section 3.1), stores what reproduces the run as experiment metadata
    (principle P7), then waits until LangSmith has stored every score the
    rules need, so the gate never decides on a partial read. Evaluators are not
    traced, only the agent runs are: their scores are recorded all the same.

    Args:
        config: The validated eval.toml.
        client: LangSmith client.

    Returns:
        The name of the new experiment.

    Raises:
        RunError: if anything is wrong before the run starts, or if scores are
            still missing after the run.
    """
    target = resolve(config.project.target)
    evaluators = resolve(config.project.evaluators)
    check_evaluators(config, evaluators)
    check_judges(config, evaluators)
    examples = load_examples(config, client)

    os.environ["EVAL_MODE"] = "1"

    results = client.evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        num_repetitions=config.run.repetitions,
        max_concurrency=config.run.max_concurrency,
        experiment_prefix="evalkit",
        # Scores are still recorded, but evaluators (and their LLM judge calls)
        # create no trace of their own: about 30 traces per run instead of ~240,
        # which keeps the gate within LangSmith's monthly trace quota.
        disable_evaluator_tracing=True,
        metadata={
            "git_commit": git_commit(),
            "dataset": config.dataset.name,
            "dataset_tag": config.dataset.tag,
            "dataset_split": config.dataset.split,
            "judge_model": config.judge.model,
            # Same hash in two experiments = same judge = comparable judge scores (P5).
            "judge_hashes": {name: judge.hash for name, judge in judges_of(evaluators).items()},
            "repetitions": config.run.repetitions,
        },
    )
    wait_for_complete_scores(
        results.experiment_name,
        evaluators=rule_names(config),
        scenarios=[str(example.id) for example in examples],
        repetitions=config.run.repetitions,
    )
    return results.experiment_name
