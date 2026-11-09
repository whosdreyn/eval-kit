"""Bootstrap confidence intervals: how sure are we that a score, or a difference, is real?

The unit of resampling is the scenario, never the single run: the repetitions of
one scenario are not independent, so resampling runs would make the intervals
look narrower than they really are.
"""

import random


def _interval(values: list[float], confidence: float) -> tuple[float, float]:
    """Return the central `confidence` interval of resampled values.

    Args:
        values: Resampled statistics (means or differences).
        confidence: Coverage of the interval, e.g. 0.95.

    Returns:
        (low, high) bounds of the interval.
    """
    values = sorted(values)
    alpha = (1 - confidence) / 2
    low = values[int(alpha * len(values))]
    high = values[int((1 - alpha) * len(values)) - 1]
    return low, high


def bootstrap_mean_ci(scores: list[float], n_resamples: int = 10_000,
                      confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a list of scores and its bootstrap confidence interval.

    Args:
        scores: One score per scenario (average the repetitions first).
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the interval.
        seed: Random seed, so the same scores always give the same interval.

    Returns:
        (mean, low, high).

    Raises:
        ValueError: if `scores` is empty.
    """
    if not scores:
        raise ValueError("no score: the mean is undefined")
    rng = random.Random(seed)
    n = len(scores)
    means = [sum(rng.choices(scores, k=n)) / n for _ in range(n_resamples)]
    low, high = _interval(means, confidence)
    return sum(scores) / n, low, high


def _pooled_mean(scores: dict[str, list[float]], scenarios: list[str]) -> float:
    """Mean of every run of the given scenarios (a scenario drawn twice counts twice).

    Args:
        scores: {scenario: [score of each repetition]}.
        scenarios: The scenarios to include, possibly with repeats.

    Returns:
        The mean of all their scores.
    """
    values = [s for scenario in scenarios for s in scores[scenario]]
    return sum(values) / len(values)


def bootstrap_diff_ci(scores_a: dict, scores_b: dict, n_resamples: int = 10_000,
                      confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of mean(B) - mean(A), resampling whole scenarios.

    "Paired" means each resample draws the same scenarios for A and B, so the
    difference compares the two versions on identical cases.

    Args:
        scores_a: {scenario: [scores]} of version A (the baseline).
        scores_b: {scenario: [scores]} of version B (the candidate).
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the interval.
        seed: Random seed, so the same scores always give the same interval.

    Returns:
        (diff, low, high), as fractions between -1 and 1: diff > 0 means B is better.

    Raises:
        ValueError: if A and B have no scenario with scores in common.
    """
    rng = random.Random(seed)
    scenarios = [s for s in scores_a if s in scores_b and scores_a[s] and scores_b[s]]
    if not scenarios:
        raise ValueError("no scenario in common between the two versions: nothing to compare")

    diff = _pooled_mean(scores_b, scenarios) - _pooled_mean(scores_a, scenarios)
    diffs = []
    for _ in range(n_resamples):
        drawn = rng.choices(scenarios, k=len(scenarios))
        diffs.append(_pooled_mean(scores_b, drawn) - _pooled_mean(scores_a, drawn))
    low, high = _interval(diffs, confidence)
    return diff, low, high
