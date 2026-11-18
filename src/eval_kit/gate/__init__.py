"""CI gate: turn the scores of a candidate and its baseline into a merge verdict."""

from eval_kit.gate.decision import (
    GateResult,
    check_primary,
    check_safety,
    consistency_report,
    decide,
    power_report,
)

__all__ = ["GateResult", "check_primary", "check_safety", "consistency_report", "decide",
           "power_report"]