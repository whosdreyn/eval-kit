"""Consistency: does the agent pass EVERY repetition of a scenario, not just on average?"""

def consistency(by_scenario: dict[str, list[float]]) -> dict:
    """Sort scenarios by how consistently they pass, and compute pass^k.

    A run passes only with a score of exactly 1. pass^k is the share of scenarios
    passed on every repetition: a real customer gets one chance, not k.

    Args:
        by_scenario: {scenario: [score of each repetition]} of one evaluator.

    Returns:
        {"always": [...], "flaky": [...], "never": [...], "pass_hat_k": float or None},
        scenario lists sorted; pass_hat_k is None when there is no scenario.
    """
    always, flaky, never = [], [], [] 
    for scenario, scores in by_scenario.items():
        passed = sum(1 for i in scores if i ==1)
        if passed == len(scores):
            always.append(scenario)
        elif passed ==0:
            never.append(scenario)
        else:
            flaky.append(scenario)
    pass_hat_k = len(always) / len(by_scenario) if by_scenario else None
    return{"always" : sorted(always),"flaky": sorted(flaky), "never" : sorted(never), "pass_hat_k" : pass_hat_k}