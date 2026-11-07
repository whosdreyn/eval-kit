"""Trajectory engine: step-level evaluation of agent tool calls.

Checks a trajectory against an agent configuration made of:
  - a DAG of allowed transitions (order of tool calls);
  - argument provenance rules (no invented API parameter);
  - safety gates (no risky action without its prerequisite checks).

Typical use:
    from trajectory_engine import from_langchain_messages, evaluate_trajectory
    steps = from_langchain_messages(result["messages"])
    report = evaluate_trajectory(steps, TRANSITIONS, ARGUMENTS, GATES)
"""

from trajectory_engine.adapters import from_langchain_messages
from trajectory_engine.arguments import check_arguments, field_values
from trajectory_engine.engine import evaluate_trajectory
from trajectory_engine.gates import check_gates
from trajectory_engine.trajectory import Step
from trajectory_engine.transitions import END, START, check_transitions

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
