"""Power of the gate: which drop can it prove, and how many scenarios does that take?

Model: a regression breaks whole scenarios (every repetition fails) and the
baseline passes everything. A bootstrap resample then shows no drop only when it
draws none of the m broken scenarios out of n, with probability (1 - m/n)^n.
The gate proves the drop when that probability is below 2.5% (the 97.5th
percentile is then below 0).
"""

import math

ALPHA = 0.025   # one side of the 95% interval


def detectable_drop(n_scenarios: int, alpha: float = ALPHA) -> int | None:
    """Smallest number of broken scenarios, out of n, that the gate can prove.

    Args:
        n_scenarios: Scenarios in the frozen dataset version.
        alpha: One-sided error level of the gate's interval.

    Returns:
        The number of scenarios that must break for the drop to be proven,
        or None when there is no scenario.
    """
    for m in range(1,n_scenarios +1):
        if (1 - m / n_scenarios) ** n_scenarios < alpha:
            return m
    return None


def scenarios_needed(drop: float, alpha: float = ALPHA) -> int:
    """How many scenarios the dataset needs to prove a drop of that size.

    Args:
        drop: Share of scenarios broken by the regression, between 0 and 1
            (0.10 means 10 points).
        alpha: One-sided error level of the gate's interval.

    Returns:
        The smallest number of scenarios n with (1 - drop)^n < alpha.

    Raises:
        ValueError: if drop is not strictly between 0 and 1.
    """
    if not 0 < drop < 1:
            raise ValueError(f"drop must be strictly between 0 and 1 (0.10 = 10 points), got {drop}")
    return math.floor(math.log(alpha) / math.log(1 - drop)) + 1