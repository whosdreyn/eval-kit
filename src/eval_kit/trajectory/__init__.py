"""Trajectory engine: step-level checks of an agent's tool calls, without any LLM.

Checks a trajectory against an agent configuration made of:
  - a DAG of allowed transitions (order of tool calls);
  - argument provenance rules (no invented API parameter);
  - safety gates (no risky action without its prerequisite checks).

Typical use:
    from eval_kit.trajectory import from_langchain_messages, evaluate_trajectory
    steps = from_langchain_messages(result["messages"])
    report = evaluate_trajectory(steps, TRANSITIONS, ARGUMENTS, GATES)
"""

from eval_kit.trajectory.adapters import from_langchain_messages
from eval_kit.trajectory.arguments import check_arguments, field_values
from eval_kit.trajectory.engine import evaluate_trajectory
from eval_kit.trajectory.safety_gates import check_gates
from eval_kit.trajectory.step import Step
from eval_kit.trajectory.transitions import END, START, check_transitions

__all__ = [
    "END",
    "START",
    "Step",
    "check_arguments",
    "check_gates",
    "check_transitions",
    "evaluate_trajectory",
    "field_values",
    "from_langchain_messages",
]
