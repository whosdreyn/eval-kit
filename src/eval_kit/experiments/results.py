"""Read the scores of a LangSmith experiment, grouped by evaluator and by scenario."""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable, Iterable

from langsmith import Client

from eval_kit.experiments.errors import RunError

# {evaluator: {scenario id: [score of each repetition]}}
Scores = dict[str, dict[str, list[float]]]


async def _fetch_runs(experiment: str) -> list:
    """Query every root run of an experiment with the runs.query API.

    Args:
        experiment: Name of the LangSmith experiment.

    Returns:
        The root runs, with their example id and feedback stats.
    """
    client = Client()
    project = client.read_project(project_name=experiment)
    return [
        run
        async for run in client.runs.query(
            project_ids=[str(project.id)],
            is_root=True,
            # Without min_start_time, runs.query only returns the last 24 hours:
            # an experiment older than a day would look empty.
            min_start_time=project.start_time,
            selects=["ID", "REFERENCE_EXAMPLE_ID", "FEEDBACK_STATS"],
        )
    ]


def fetch_runs(experiment: str) -> list:
    """Return the root runs of an experiment, with their example id and feedback stats.

    Args:
        experiment: Name of the LangSmith experiment.

    Returns:
        The root runs of the experiment (one per example and repetition).
    """
    return asyncio.run(_fetch_runs(experiment))


def group_scores(runs) -> Scores:
    """Group the scores of an experiment by evaluator, then by scenario.

    Args:
        runs: Root runs returned by fetch_runs (reference_example_id, feedback_stats).

    Returns:
        {evaluator: {example_id: [score of each repetition]}}. Runs without a
        scenario or without feedback are skipped, and an evaluator error leaves
        the score missing (the gate then fails closed) instead of counting it as 0.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run.reference_example_id is None:
            continue
        example = str(run.reference_example_id)
        for evaluator, stats in (run.feedback_stats or {}).items():
            if stats.avg is None:  # evaluator crashed: missing score, not a 0
                continue
            grouped[evaluator][example].append(stats.avg)
    # Plain dicts: a defaultdict would silently create empty entries when read.
    return {evaluator: dict(by_example) for evaluator, by_example in grouped.items()}


def experiment_scores(experiment: str) -> Scores:
    """Read and group the scores of an experiment in one call.

    Args:
        experiment: Name of the LangSmith experiment.

    Returns:
        {evaluator: {example_id: [scores]}}, see group_scores.
    """
    return group_scores(fetch_runs(experiment))


def incomplete(scores: Scores, evaluators: Iterable[str], scenarios: Iterable[str],
               repetitions: int) -> list[str]:
    """List the evaluators that do not have every expected score yet.

    LangSmith stores evaluator scores a few seconds after evaluate() returns, so
    an experiment read too early looks partial. A partial read could hide a
    failed run, so nothing is decided until every score is there.

    Args:
        scores: Scores read so far, see group_scores.
        evaluators: Evaluator names that must be complete (the rules of eval.toml).
        scenarios: Ids of every example of the run.
        repetitions: Expected number of scores per example.

    Returns:
        One line per incomplete evaluator; an empty list means complete.
    """
    scenarios = list(scenarios)
    problems = []
    for evaluator in sorted(evaluators):
        by_scenario = scores.get(evaluator, {})
        short = [s for s in scenarios if len(by_scenario.get(s, [])) < repetitions]
        if short:
            problems.append(f"{evaluator}: {len(short)} scenario(s) with fewer than "
                            f"{repetitions} scores")
    return problems


def wait_for_complete_scores(
    experiment: str,
    evaluators: Iterable[str],
    scenarios: Iterable[str],
    repetitions: int,
    *,
    timeout: float = 180.0,
    interval: float = 5.0,
    fetch: Callable[[str], Scores] = experiment_scores,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Scores:
    """Read an experiment again and again until every expected score is stored.

    Args:
        experiment: Name of the LangSmith experiment.
        evaluators: Evaluator names that must be complete.
        scenarios: Ids of every example of the run.
        repetitions: Expected number of scores per example.
        timeout: Seconds to wait before giving up.
        interval: Seconds between two reads.
        fetch: Reads the scores (replaced by a fake in tests).
        sleep: Waits between reads (replaced in tests).
        clock: Measures the elapsed time (replaced in tests).

    Returns:
        The complete scores.

    Raises:
        RunError: if scores are still missing after `timeout` seconds, e.g. an
            evaluator that crashed on some runs (fail closed).
    """
    evaluators, scenarios = list(evaluators), list(scenarios)
    deadline = clock() + timeout
    while True:
        scores = fetch(experiment)
        problems = incomplete(scores, evaluators, scenarios, repetitions)
        if not problems:
            return scores
        if clock() >= deadline:
            raise RunError(f"experiment {experiment!r} still incomplete after {timeout:g}s: "
                           + "; ".join(problems))
        sleep(interval)
