"""Hallucination check: does every controlled argument come from an earlier tool result?"""

from eval_kit.trajectory import Step


def field_values(obj, field: str) -> list:
    """Collect every value of `field`, at any depth of nested dicts and lists.

    The function calls itself on each value of a dict and each item of a list,
    so it finds the field however deep it is, without knowing the structure.

    Example:
        >>> field_values({"purchases": [{"id": "A1"}, {"id": "A2"}]}, "id")
        ['A1', 'A2']

    Args:
        obj: Any value: dict, list, number, text...
        field: Name of the key to look for.

    Returns:
        All the values found for `field`, in reading order (empty if none).
    """
    values = []
    if isinstance(obj, dict):
        if field in obj:
            values.append(obj[field])
        for value in obj.values():
            values.extend(field_values(value, field))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(field_values(item, field))
    return values


def check_arguments(steps: list[Step], rules: dict) -> list[dict]:
    """Check that each controlled argument comes from the result of an earlier tool.

    `rules` maps a tool to its controlled arguments, and each argument to the
    (source tool, field) it must come from. An argument is valid when its value
    appears in that field of at least one result of the source tool called
    *before* it. All allowed values are gathered first, then a single decision
    is made, so each controlled argument gets exactly one verdict.

    Example:
        {"create_refund_ticket": {"amount": ("calculate_refund_amount", "refund_amount")}}

    Args:
        steps: The trajectory to check.
        rules: The provenance rules of the agent.

    Returns:
        One verdict per controlled argument: a dict with "step", "tool",
        "argument", "valid" and, when invalid, "reason".
    """
    verdicts = []
    for i, step in enumerate(steps, start=1):
        for argument, (source_tool, field) in rules.get(step.tool, {}).items():
            value = step.arguments.get(argument)

            allowed = []
            for previous in steps[: i - 1]:
                if previous.tool == source_tool:
                    allowed.extend(field_values(previous.result, field))

            verdict = {"step": i, "tool": step.tool, "argument": argument, "valid": value in allowed}
            if not verdict["valid"]:
                verdict["reason"] = (
                    f"invented argument: {argument}={value!r} appears in no result of {source_tool}"
                )
            verdicts.append(verdict)
    return verdicts
