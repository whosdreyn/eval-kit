"""Order check: is every move from one tool to the next allowed by the agent's DAG?"""

from eval_kit.trajectory.step import Step

START = "START"
END = "END"


def check_transitions(steps: list[Step], transitions: dict) -> list[dict]:
    """Check every transition of a trajectory against an allowed-transitions graph.

    `transitions` maps each node to the list of nodes allowed right after it.
    The special node START is where every trajectory begins, and END must be
    allowed after the last tool for the agent to be allowed to stop there.
    A tool missing from `transitions` has no allowed successor, so an unknown
    (invented) tool is rejected without any special code.

    Example:
        {"START": ["lookup", "END"], "lookup": ["check", "END"], ...}

    Args:
        steps: The trajectory to check.
        transitions: The allowed-transitions graph of the agent.

    Returns:
        One verdict per transition, plus one for the final move to END if it
        is not allowed. Each verdict is a dict with "step", "tool", "valid"
        and, when invalid, "reason".
    """
    verdicts = []
    previous = START
    for i, step in enumerate(steps, start=1):
        if step.tool in transitions.get(previous, []):
            verdicts.append({"step": i, "tool": step.tool, "valid": True})
        else:
            verdicts.append(
                {
                    "step": i,
                    "tool": step.tool,
                    "valid": False,
                    "reason": f"forbidden transition: {previous} → {step.tool}",
                }
            )
        previous = step.tool

    if END not in transitions.get(previous, []):
        verdicts.append(
            {
                "step": len(steps) + 1,
                "tool": END,
                "valid": False,
                "reason": f"forbidden transition: {previous} → {END}",
            }
        )
    return verdicts
