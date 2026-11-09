"""Read the scores of a LangSmith experiment, grouped by evaluator and by scenario."""

import asyncio
from collections import defaultdict

from langsmith import Client

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
