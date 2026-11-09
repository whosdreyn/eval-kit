"""eval-kit: automated evaluation of LLM agents, graphs, chatbots and RAG with LangSmith.

Subpackages:
    config        eval.toml schema and loader
    trajectory    step-level checks of tool calls (DAG, argument provenance, safety gates)
    stats         bootstrap confidence intervals
    experiments   run an experiment on the frozen dataset and read its scores (LangSmith)
    gate          the CI merge verdict (safety, non-inferiority, report)

The trajectory engine is also exported at the top level, so existing code such as
`from eval_kit import Step, evaluate_trajectory` keeps working.
"""

from eval_kit.trajectory import (
    END,
    START,
    Step,
    check_arguments,
    check_gates,
    check_transitions,
    evaluate_trajectory,
    field_values,
    from_langchain_messages,
)

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
