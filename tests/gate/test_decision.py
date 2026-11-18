"""Tests of the gate decisions: fake scores, no network."""

from eval_kit.config import RulesConfig
from eval_kit.gate import check_primary, check_safety, consistency_report, decide

SAFETY = {"no_ticket": "ticket without request"}


def test_safety_passes_when_every_run_holds():
    candidate = {"no_ticket": {"ex1": [1, 1, 1], "ex2": [1, 1, 1]}}
    assert check_safety(candidate, SAFETY) == []


def test_one_failed_run_blocks():
    candidate = {"no_ticket": {"ex1": [1, 0, 1]}}
    problems = check_safety(candidate, SAFETY)
    assert len(problems) == 1
    assert "1 run(s) failed" in problems[0]
    assert "ticket without request" in problems[0]


def test_failed_runs_are_counted_per_run_not_per_scenario():
    # Two failures in the same scenario are two failed runs.
    candidate = {"no_ticket": {"ex1": [0, 0, 1], "ex2": [1, 1, 1]}}
    assert "2 run(s) failed" in check_safety(candidate, SAFETY)[0]


def test_missing_evaluator_blocks():
    candidate = {"decision": {"ex1": [1]}}
    problems = check_safety(candidate, SAFETY)
    assert len(problems) == 1
    assert "no score" in problems[0]


def test_missing_scenario_blocks():
    # The agent ran on ex2 (decision has a score) but the safety rule never checked it.
    candidate = {"decision": {"ex1": [1], "ex2": [1]}, "no_ticket": {"ex1": [1]}}
    problems = check_safety(candidate, SAFETY)
    assert len(problems) == 1
    assert "1 scenario" in problems[0]


def test_partial_score_is_not_a_failure():
    # trajectoire_dag gives 0.5 for an unexpected but non-critical transition:
    # a quality drop, not a safety violation. Only 0 is a violation.
    candidate = {"no_ticket": {"ex1": [0.5]}}
    assert check_safety(candidate, SAFETY) == []


# ── check_primary: non-inferiority against the baseline ────────────────────

BASE = {"decision": {f"ex{i}": [1, 1, 1] for i in range(10)}}


def candidate_with(zero_scenarios: int, one_bad_run: bool = False) -> dict:
    """Candidate scores: the first `zero_scenarios` scenarios fail on every run."""
    scores = {f"ex{i}": [0, 0, 0] if i < zero_scenarios else [1, 1, 1] for i in range(10)}
    if one_bad_run:
        scores["ex0"] = [0, 1, 1]
    return {"decision": scores}


def test_equal_scores_pass_and_are_reported():
    problems, report = check_primary(BASE, BASE, {"decision": 0})
    assert problems == []
    assert len(report) == 1 and "decision" in report[0]


def test_one_bad_run_is_noise_not_a_regression():
    # -3.3 points, 95% CI [-10.0; +0.0]: the interval touches 0, nothing is proven.
    problems, _ = check_primary(BASE, candidate_with(0, one_bad_run=True), {"decision": 0})
    assert problems == []


def test_significant_drop_blocks():
    # -40 points, 95% CI [-70.0; -10.0]: the whole interval is below 0.
    problems, _ = check_primary(BASE, candidate_with(4), {"decision": 0})
    assert len(problems) == 1
    assert "significantly worse" in problems[0]


def test_drop_within_the_margin_passes():
    # Same -40 points, but eval.toml tolerates up to 20 points: the CI upper
    # bound (-10) is not below -20, so the drop is not proven beyond the margin.
    problems, _ = check_primary(BASE, candidate_with(4), {"decision": 20})
    assert problems == []


def test_missing_candidate_scores_block():
    problems, _ = check_primary(BASE, {}, {"decision": 0})
    assert len(problems) == 1
    assert "no score" in problems[0]


def test_different_scenarios_block():
    # The baseline was run on another dataset version: the comparison is meaningless.
    candidate = {"decision": {f"ex{i}": [1, 1, 1] for i in range(5)}}
    problems, _ = check_primary(BASE, candidate, {"decision": 0})
    assert len(problems) == 1
    assert "different scenarios" in problems[0]


def test_drop_beyond_a_small_margin_blocks():
    # -40 points, CI upper bound -10 points, margin 5 points: -10 < -5, the drop is
    # proven beyond the margin. Catches a margin compared in the wrong unit
    # (fraction -0.10 vs points -5 would wrongly pass).
    problems, _ = check_primary(BASE, candidate_with(4), {"decision": 5})
    assert len(problems) == 1
    assert "significantly worse" in problems[0]



# ── decide: the whole verdict ──────────────────────────────────────────────

RULES = RulesConfig(safety={"no_ticket": "ticket without request"},
                    primary={"decision": 0},
                    secondary=["message"])
GOOD = {
    "no_ticket": {f"ex{i}": [1, 1, 1] for i in range(10)},
    "decision": {f"ex{i}": [1, 1, 1] for i in range(10)},
    "message": {f"ex{i}": [1, 1, 1] for i in range(10)},
}


def with_scores(evaluator: str, scores: dict) -> dict:
    """A copy of GOOD where one evaluator gets other scores."""
    return {**GOOD, evaluator: scores}


def test_good_candidate_passes():
    result = decide(RULES, GOOD, GOOD)
    assert result.passed
    assert result.problems == []


def test_safety_failure_blocks():
    candidate = with_scores("no_ticket", {**GOOD["no_ticket"], "ex3": [1, 0, 1]})
    result = decide(RULES, GOOD, candidate)
    assert not result.passed
    assert any("no_ticket" in p for p in result.problems)


def test_primary_regression_blocks():
    candidate = with_scores("decision", candidate_with(4)["decision"])
    result = decide(RULES, GOOD, candidate)
    assert not result.passed
    assert any("significantly worse" in p for p in result.problems)


def test_secondary_drop_never_blocks_but_is_reported():
    candidate = with_scores("message", {f"ex{i}": [0, 0, 0] for i in range(10)})
    result = decide(RULES, GOOD, candidate)
    assert result.passed
    assert any(line.startswith("message") for line in result.report)


def test_secondary_without_scores_never_blocks():
    candidate = {k: v for k, v in GOOD.items() if k != "message"}
    result = decide(RULES, GOOD, candidate)
    assert result.passed
    assert any("message" in line and "no score" in line for line in result.report)



# ── repetitions: a missing run must never hide a failure ───────────────────

def test_missing_run_blocks_safety():
    # ex2 has 2 scores out of 3: the third run could have been a failure.
    candidate = {"no_ticket": {"ex1": [1, 1, 1], "ex2": [1, 1]}}
    problems = check_safety(candidate, SAFETY, repetitions=3)
    assert len(problems) == 1
    assert "without exactly 3 scores" in problems[0]


def test_missing_run_blocks_primary():
    candidate = {"decision": {**BASE["decision"], "ex9": [1, 1]}}
    problems, _ = check_primary(BASE, candidate, {"decision": 0}, repetitions=3)
    assert len(problems) == 1
    assert "partial data" in problems[0]


def test_decide_checks_repetitions():
    candidate = with_scores("no_ticket", {**GOOD["no_ticket"], "ex0": [1, 1]})
    assert decide(RULES, GOOD, candidate).passed                    # count not checked
    assert not decide(RULES, GOOD, candidate, repetitions=3).passed  # count checked



# ── consistency in the report (pass^k), never blocking ─────────────────────

LONG_ID = "d2c300e5-c2ed-4b63-be75-a9fd12af1f7b"   # real LangSmith example ids are long


def test_stable_evaluator_reports_full_pass_hat_k():
    candidate = {"decision": {f"ex{i}": [1, 1, 1] for i in range(10)}}
    lines = consistency_report(candidate, ["decision"], repetitions=3)
    assert lines == ["decision: pass^3 10/10"]


def test_flaky_and_never_scenarios_are_named_with_short_ids():
    candidate = {"no_ticket": {LONG_ID: [1, 0, 1], "ex2": [0, 0, 0], "ex3": [1, 1, 1]}}
    [line] = consistency_report(candidate, ["no_ticket"], repetitions=3)
    assert line.startswith("no_ticket: pass^3 1/3")
    assert "flaky: d2c300e5" in line and LONG_ID not in line
    assert "never: ex2" in line


def test_evaluator_without_scores_is_reported():
    assert consistency_report({}, ["decision"], repetitions=3) == ["decision: pass^3 no score"]


def test_unknown_repetitions_reads_pass_hat_k():
    candidate = {"decision": {"ex1": [1, 1]}}
    assert consistency_report(candidate, ["decision"]) == ["decision: pass^k 1/1"]


def test_decide_reports_consistency_without_blocking():
    # A flaky primary scenario: not a proven regression, so the gate passes,
    # but the instability must be visible in the report.
    candidate = with_scores("decision", {**GOOD["decision"], "ex3": [1, 0, 1]})
    result = decide(RULES, GOOD, candidate, repetitions=3)
    assert result.passed
    assert any(line.startswith("decision: pass^3 9/10") and "flaky: ex3" in line
               for line in result.report)
