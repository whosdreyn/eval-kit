"""Tests of the trajectory engine on a tiny generic agent: search, then order."""

from types import SimpleNamespace as NS

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

TRANSITIONS = {START: ["search", END], "search": ["order", END], "order": [END]}
ARGUMENTS = {"order": {"product_id": ("search", "id")}}


def stock_checked(step, previous_steps):
    """Gate: an order needs an earlier search that found the product in stock."""
    ok = any(s.tool == "search" and s.result.get("in_stock") is True for s in previous_steps)
    return ok, "stock checked" if ok else "order without a stock check"


GATES = {"order": [stock_checked]}

SEARCH = Step("search", {"query": "lamp"}, {"items": [{"id": "P1"}], "in_stock": True})
ORDER = Step("order", {"product_id": "P1"}, {"status": "ok"})


def test_valid_trajectory_scores_1():
    report = evaluate_trajectory([SEARCH, ORDER], TRANSITIONS, ARGUMENTS, GATES)
    assert report == {"score": 1.0, "failed": False, "errors": []}


def test_forbidden_transition_lowers_the_score_without_failing():
    verdicts = check_transitions([ORDER], TRANSITIONS)
    assert not verdicts[0]["valid"]
    # search -> search is not in the graph, every other check passes.
    report = evaluate_trajectory([SEARCH, SEARCH, ORDER], TRANSITIONS, ARGUMENTS, GATES)
    assert 0 < report["score"] < 1
    assert not report["failed"]


def test_unknown_tool_is_a_forbidden_transition():
    verdicts = check_transitions([Step("delete_all", {}, None)], TRANSITIONS)
    assert "forbidden transition" in verdicts[0]["reason"]


def test_invented_argument_is_critical():
    invented = Step("order", {"product_id": "P999"}, {"status": "ok"})
    assert not check_arguments([SEARCH, invented], ARGUMENTS)[0]["valid"]
    report = evaluate_trajectory([SEARCH, invented], TRANSITIONS, ARGUMENTS, GATES)
    assert report["score"] == 0.0 and report["failed"]


def test_failed_gate_is_critical():
    out_of_stock = Step("search", {"query": "lamp"}, {"items": [{"id": "P1"}], "in_stock": False})
    assert not check_gates([out_of_stock, ORDER], GATES)[0]["valid"]
    report = evaluate_trajectory([out_of_stock, ORDER], TRANSITIONS, ARGUMENTS, GATES)
    assert report["score"] == 0.0 and report["failed"]


def test_field_values_finds_nested_fields():
    assert field_values({"a": [{"id": 1}, {"b": {"id": 2}}]}, "id") == [1, 2]


def test_langchain_messages_become_steps():
    messages = [
        NS(tool_calls=[{"name": "search", "args": {"query": "lamp"}, "id": "c1"}]),
        NS(tool_call_id="c1", content='{"in_stock": true}'),
    ]
    assert from_langchain_messages(messages) == [Step("search", {"query": "lamp"}, {"in_stock": True})]
