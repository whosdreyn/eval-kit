"""Adapters that turn an agent's raw output into the common trajectory format."""

import json

from eval_kit.trajectory.step import Step


def _parse_result(content):
    """Decode a tool result sent as JSON text.

    Args:
        content: The raw content of a tool message.

    Returns:
        The decoded value (usually a dict) if `content` is valid JSON,
        otherwise `content` unchanged.
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def from_langchain_messages(messages) -> list[Step]:
    """Rebuild the trajectory of a LangChain agent from its messages.

    A tool call and its result live in two different messages: the call in an
    AIMessage (`tool_calls`, each with a "name", "args" and "id"), the result in
    a later ToolMessage (`content`, linked to its call by `tool_call_id`).
    The function first indexes every result by call id, then walks the calls
    in order and pairs each one with its result.

    LangChain is not imported: messages are read through their attributes, so
    the engine stays independent from the framework.

    Args:
        messages: The `messages` list returned by a LangChain agent.

    Returns:
        One Step per tool call, in the order the agent made them. A call with
        no matching result gets `result=None`.
    """
    results = {}
    for message in messages:
        if hasattr(message, "tool_call_id"):
            results[message.tool_call_id] = _parse_result(message.content)

    steps = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            steps.append(
                Step(
                    tool=call["name"],
                    arguments=call["args"],
                    result=results.get(call["id"]),
                )
            )
    return steps
