"""Decide whether a candidate may be merged: safety first, then non-inferiority.

The three categories of [rules] get three different treatments (principle P2):
  - safety:    one failed run blocks, whatever the baseline;
  - primary:   blocks when the candidate is proven worse than the baseline
               beyond the margin of eval.toml (paired bootstrap, 95% CI);
  - secondary: compared like primary metrics, but reported only, never blocking.
"""

from dataclasses import dataclass

from eval_kit.config import RulesConfig
from eval_kit.experiments.results import Scores
from eval_kit.stats import bootstrap_diff_ci

# A margin no difference of scores can exceed: such a metric is never "worse".
REPORT_ONLY_MARGIN = 100


def _wrong_counts(by_scenario: dict[str, list[float]], repetitions: int | None) -> int:
    """Count the scenarios that do not have exactly `repetitions` scores.

    Args:
        by_scenario: {scenario: [scores]} of one evaluator.
        repetitions: Expected scores per scenario; None disables the check.

    Returns:
        The number of scenarios with a wrong number of scores (0 when disabled).
    """
    if repetitions is None:
        return 0
    return sum(1 for scores in by_scenario.values() if len(scores) != repetitions)


def check_safety(candidate: Scores, safety: dict[str, str],
                 repetitions: int | None = None) -> list[str]:
    """Return one problem per safety rule that fails or cannot be verified.

    A safety evaluator returns 1 when the rule holds and 0 when it is broken.
    A single 0 blocks; a partial score (e.g. 0.5 for a non-critical trajectory
    fault) is a quality drop, not a violation. A missing score blocks too, on a
    whole evaluator, on a single scenario, or on a single run when `repetitions`
    is given (fail closed, principle P1).

    Args:
        candidate: Scores of the experiment under review.
        safety: {evaluator: what a failure means} from eval.toml.
        repetitions: Expected scores per scenario; None skips this count.

    Returns:
        The problems found; an empty list means every safety rule holds.
    """
    # Every scenario the agent ran on, as seen by any evaluator.
    expected = set()
    for by_scenario in candidate.values():
        expected |= set(by_scenario)

    problems = []
    for evaluator, description in safety.items():
        by_scenario = candidate.get(evaluator, {})
        if not by_scenario:
            problems.append(f"{evaluator}: no score, the rule cannot be verified")
            continue

        missing = expected - set(by_scenario)
        if missing:
            problems.append(f"{evaluator}: no score on {len(missing)} scenario(s)")
        partial = _wrong_counts(by_scenario, repetitions)
        if partial:
            problems.append(f"{evaluator}: {partial} scenario(s) without exactly "
                            f"{repetitions} scores, a failed run could be hidden")

        failures = sum(1 for scores in by_scenario.values() for s in scores if s == 0)
        if failures:
            problems.append(f"{evaluator}: {failures} run(s) failed — {description}")
    return problems


def check_primary(baseline: Scores, candidate: Scores, primary: dict[str, float],
                  repetitions: int | None = None) -> tuple[list[str], list[str]]:
    """Compare each metric to the baseline (non-inferiority with a margin).

    Blocks a metric when the upper bound of the 95% CI of (candidate - baseline)
    is below -margin: the drop is then proven to exceed what eval.toml tolerates.
    A drop that is not proven (the interval reaches above -margin) is reported
    but does not block.

    Args:
        baseline: Scores of the reference experiment.
        candidate: Scores of the experiment under review.
        primary: {evaluator: margin in points} from eval.toml.
        repetitions: Expected scores per scenario on both sides; None skips it.

    Returns:
        (problems, report): problems block the merge; report has one line per
        compared metric.
    """
    problems, report = [], []
    for evaluator, margin in primary.items():
        base = baseline.get(evaluator, {})
        cand = candidate.get(evaluator, {})

        if not base or not cand:
            problems.append(f"{evaluator}: no score, the comparison cannot be made")
            continue
        if _wrong_counts(base, repetitions) or _wrong_counts(cand, repetitions):
            problems.append(f"{evaluator}: scenarios without exactly {repetitions} scores, "
                            "the comparison would rest on partial data")
            continue
        if set(base) != set(cand):
            problems.append(f"{evaluator}: different scenarios in baseline and candidate "
                            "(different dataset version?), the comparison is meaningless")
            continue

        diff, low, high = bootstrap_diff_ci(base, cand)  # baseline = A, candidate = B
        report.append(f"{evaluator}: {diff * 100:+.1f} points, "
                      f"95% CI [{low * 100:+.1f}; {high * 100:+.1f}], margin {margin:g}")

        if high < -margin / 100:  # high is a fraction, margin is in points
            problems.append(f"{evaluator}: significantly worse than the baseline "
                            f"({diff * 100:+.1f} points, beyond the {margin:g}-point margin)")
    return problems, report


@dataclass(frozen=True)
class GateResult:
    """The verdict of the gate.

    Attributes:
        passed: False as soon as one problem exists.
        problems: What blocks the merge.
        report: Everything that was measured, blocking or not.
    """

    passed: bool
    problems: list[str]
    report: list[str]


def decide(rules: RulesConfig, baseline: Scores, candidate: Scores,
           repetitions: int | None = None) -> GateResult:
    """Apply every rule of eval.toml: safety, then primary, then secondary (report only).

    Args:
        rules: The [rules] section of eval.toml.
        baseline: Scores of the reference experiment.
        candidate: Scores of the experiment under review.
        repetitions: Expected scores per scenario ([run] repetitions); None skips
            the count. The CLI always passes it.

    Returns:
        The verdict, the blocking problems and the full report.
    """
    problems = check_safety(candidate, rules.safety, repetitions)
    primary_problems, report = check_primary(baseline, candidate, rules.primary, repetitions)
    problems += primary_problems

    # Secondary metrics: same comparison, but a margin of 100 points can never be
    # exceeded, and any problem (e.g. no score) is only reported.
    info_problems, info_report = check_primary(
        baseline, candidate, dict.fromkeys(rules.secondary, REPORT_ONLY_MARGIN), repetitions)
    report += info_report
    report += [f"{p} (info, not blocking)" for p in info_problems]

    return GateResult(passed=not problems, problems=problems, report=report)
