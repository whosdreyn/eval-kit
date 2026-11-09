"""A frozen LLM judge: dated model, temperature 0, protected prompt, structured verdict.

A judge never guesses: an answer that is not exactly {"reason": str, "verdict": bool}
raises JudgeError. The evaluator then has no score, and the gate fails closed
instead of reading "Oui... mais non" as a pass.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from eval_kit.config.models import DATED_MODEL_PATTERN
from eval_kit.judges.prompt import build_messages, prompt_hash

# Sends the chat messages to a model and returns its raw text answer.
Completer = Callable[[str, list[dict[str, str]]], str]

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},   # first: the judge explains, then decides
        "verdict": {"type": "boolean"},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}


class JudgeError(Exception):
    """The judge gave no usable verdict: no score must be recorded."""


@dataclass(frozen=True)
class Verdict:
    """What a judge decided.

    Attributes:
        passed: True only if every criterion holds.
        reason: The judge's justification, stored as the feedback comment.
    """

    passed: bool
    reason: str


def parse_verdict(text: str) -> Verdict:
    """Read a judge's answer; anything but the exact expected shape is an error.

    Args:
        text: The raw answer of the model.

    Returns:
        The verdict.

    Raises:
        JudgeError: if the answer is not JSON, or verdict is not a real boolean,
            or reason is missing. "true" as text is refused too: no guessing.
    """
    try:
        answer = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise JudgeError(f"judge answer is not JSON: {text!r:.200}") from e
    if not isinstance(answer, dict):
        raise JudgeError(f"judge answer is not an object: {text!r:.200}")
    verdict, reason = answer.get("verdict"), answer.get("reason")
    if not isinstance(verdict, bool) or not isinstance(reason, str):
        raise JudgeError(f"judge answer needs a boolean verdict and a reason: {text!r:.200}")
    return Verdict(passed=verdict, reason=reason)


def openai_completer(model: str, messages: list[dict[str, str]]) -> str:
    """Ask an OpenAI model, at temperature 0, for an answer that follows VERDICT_SCHEMA.

    Args:
        model: Dated OpenAI model name.
        messages: The chat messages built by build_messages.

    Returns:
        The raw JSON text of the answer.
    """
    from openai import OpenAI  # imported here: only needed when a real judge runs

    response = OpenAI().chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "strict": True, "schema": VERDICT_SCHEMA},
        },
    )
    return response.choices[0].message.content


class Judge:
    """A reusable LLM judge for one question, e.g. "is this message polite and clear?".

    Example:
        polite = Judge("gpt-4.1-mini-2025-04-14", "The message must be polite and clear.")
        verdict = polite(message=outputs["message"])
        return {"key": "message_conforme", "score": verdict.passed, "comment": verdict.reason}
    """

    def __init__(self, model: str, instructions: str, complete: Completer = openai_completer):
        """Create a judge.

        Args:
            model: Dated model name (principle P5); an alias is refused.
            instructions: The criteria, written as plain rules.
            complete: Sends messages to the model (a fake one in tests).

        Raises:
            ValueError: if the model name is not dated or the instructions are empty.
        """
        if not re.fullmatch(DATED_MODEL_PATTERN, model):
            raise ValueError(f"judge model {model!r} is not dated: use e.g. gpt-4.1-mini-2025-04-14")
        if not instructions.strip():
            raise ValueError("a judge needs instructions")
        self.model = model
        self.instructions = instructions
        self._complete = complete

    @property
    def hash(self) -> str:
        """Fingerprint of the model and prompt: equal hashes mean comparable scores."""
        return prompt_hash(self.model, self.instructions)

    def __call__(self, **data: object) -> Verdict:
        """Judge the given data.

        Args:
            **data: Everything to evaluate, by name, e.g. question=..., answer=...

        Returns:
            The verdict.

        Raises:
            JudgeError: if the model's answer is not a valid verdict.
        """
        return parse_verdict(self._complete(self.model, build_messages(self.instructions, data)))
