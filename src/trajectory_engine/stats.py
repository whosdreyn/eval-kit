import random


def bootstrap_mean_ci(scores: list[float], n_resamples: int = 10_000,
                      confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, low, high): the mean of `scores` and its bootstrap confidence interval."""
    rng = random.Random(seed)
    n = len(scores)
    mean = sum(scores) / n
    means = []

    for _ in range(n_resamples):
        tirage = rng.choices(scores, k=n)
        mean_tirage = sum(tirage) / n
        means.append(mean_tirage)

    means.sort()
    alpha = (1 - confidence) / 2
    low = means[int(alpha * n_resamples)]
    high = means[int((1 - alpha) * n_resamples) - 1]

    return mean, low, high


def bootstrap_diff_ci(scores_a: dict, scores_b: dict, n_resamples: int = 10_000,
                      confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of mean(B) - mean(A), resampling whole scenarios.

    scores_a and scores_b map each scenario id to the list of its scores
    (one per repetition), for versions A and B on the SAME scenarios.
    """
    rng = random.Random(seed)
    scenarios = [s for s in scores_a if s in scores_b and scores_a[s] and scores_b[s]]
    if not scenarios:
        raise ValueError("no scenario in common between the two versions: nothing to compare")

    def moyenne(scores, ids):
        valeurs = [s for i in ids for s in scores[i]] 
        return sum(valeurs) / len(valeurs)

    diff = moyenne(scores_b, scenarios) - moyenne(scores_a, scenarios)
    diffs = []
 
    for _ in range(n_resamples):
        ids = rng.choices(scenarios, k=len(scenarios))
        diffs.append(moyenne(scores_b, ids) - moyenne(scores_a, ids))

    diffs.sort()
    alpha = (1 - confidence) / 2
    low = diffs[int(alpha * n_resamples)]
    high = diffs[int((1 - alpha) * n_resamples) - 1]

    return diff, low, high
