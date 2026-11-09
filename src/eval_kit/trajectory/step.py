"""Common trajectory format: the list of tool calls an agent made, as Steps."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    """One tool call made by an agent, with what it sent and what it got back.

    A trajectory is simply a list of steps, in the order the agent made them.

    Attributes:
        tool: Name of the tool that was called, e.g. "calculate_refund_amount".
        arguments: Arguments the agent passed to the tool.
        result: What the tool returned (usually a dict, sometimes plain text or None).
    """

    tool: str
    arguments: dict
    result: Any
