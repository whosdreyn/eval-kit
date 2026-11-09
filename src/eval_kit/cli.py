"""Command line: `evalkit run` and `evalkit gate`, driven by eval.toml.

Exit codes: 0 passed, 1 blocked by the gate (the agent regressed),
2 error (the evaluation could not be done: config, baseline, dataset...).
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from eval_kit.config import ConfigError, EvalConfig, load_config
from eval_kit.experiments import RunError, experiment_scores, run_experiment
from eval_kit.gate import decide

EXIT_PASSED, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2


def _prepare(config_path: str) -> EvalConfig:
    """Load eval.toml, the project's .env and make the evaluated project importable.

    The .env next to eval.toml provides the API keys on a developer machine. It
    never overrides a variable that is already set, so CI secrets always win.
    Key values are never printed.

    Args:
        config_path: Path to eval.toml; its folder is the project root.

    Returns:
        The validated configuration.
    """
    path = Path(config_path).resolve()
    config = load_config(path)
    load_dotenv(path.parent / ".env", override=False)
    sys.path.insert(0, str(path.parent))
    return config


def cmd_run(args: argparse.Namespace) -> int:
    """Run an experiment and print its name (used to create a first baseline).

    Args:
        args: Parsed command line (args.config).

    Returns:
        EXIT_PASSED once the experiment is recorded.
    """
    config = _prepare(args.config)
    print(f"experiment: {run_experiment(config, Client())}")
    return EXIT_PASSED


def cmd_gate(args: argparse.Namespace) -> int:
    """Run the candidate, compare it to the baseline and print the verdict.

    Args:
        args: Parsed command line (args.config).

    Returns:
        EXIT_PASSED if the gate passes, EXIT_BLOCKED otherwise.

    Raises:
        RunError: if eval.toml has no [baseline] (fail closed).
    """
    config = _prepare(args.config)
    if config.baseline is None:
        raise RunError("eval.toml has no [baseline]: run `evalkit run` once, then put "
                       "the experiment name in [baseline] experiment = \"...\"")

    candidate = run_experiment(config, Client())
    result = decide(config.rules,
                    experiment_scores(config.baseline.experiment),
                    experiment_scores(candidate))

    for line in result.report:
        print(line)
    if result.passed:
        print("✅ GATE PASSED")
    else:
        print("❌ GATE FAILED")
        for problem in result.problems:
            print(f"  - {problem}")
    return EXIT_PASSED if result.passed else EXIT_BLOCKED


def build_parser() -> argparse.ArgumentParser:
    """Describe the commands and their options for argparse.

    Returns:
        The parser of the `evalkit` command line.
    """
    parser = argparse.ArgumentParser(prog="evalkit", description="Evaluate an LLM agent with LangSmith.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in [
        ("run", cmd_run, "run an experiment on the frozen dataset"),
        ("gate", cmd_gate, "run the candidate and block it if it regressed"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default="eval.toml", help="path to eval.toml")
        command.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `evalkit` command.

    Args:
        argv: Arguments without the program name; None reads the real command line.

    Returns:
        The exit code: 0 passed, 1 blocked, 2 error.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, RunError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR