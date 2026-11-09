"""Safety check: run the business gates that must pass before a risky tool call."""

from eval_kit.trajectory.step import Step


def check_gates(steps: list[Step], gates: dict) -> list[dict]:
    """Run the safety gates registered for each tool of a trajectory.

    The engine knows nothing about the business: it only calls the gates.
    Each gate is a function written in the agent's configuration, which
    receives the step being checked and the steps before it, and returns
    (valid, reason). Example: "a refund ticket requires an earlier
    eligibility check that answered eligible: True".

    Args:
        steps: The trajectory to check.
        gates: Maps a tool name to the list of gate functions to run before it.

    Returns:
        One verdict per gate run: a dict with "step", "tool", "gate" (the gate
        function name), "valid" and "reason".
    """
    verdicts = []
    for i, step in enumerate(steps, start=1):
        previous_steps = steps[: i - 1]
        for gate in gates.get(step.tool, []):
            valid, reason = gate(step, previous_steps)
            verdicts.append(
                {"step": i, "tool": step.tool, "gate": gate.__name__, "valid": valid, "reason": reason}
            )
    return verdicts
