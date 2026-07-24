"""The universal declarative-expression evaluator: operators only, no question families."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sworldmodel.expressions import evaluate
from sworldmodel.worldspec import parse_expr

T0 = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
T1 = datetime.fromisoformat("2024-02-01T00:00:00+00:00")


class _Ctx:
    """A minimal ExprContext backed by plain dicts."""

    def __init__(self, **kw: Any) -> None:
        self.fields = kw.get("fields", {})
        self.records = kw.get("records", {})
        self.events = kw.get("events", {})
        self.resources = kw.get("resources", {})
        self.documents = kw.get("documents", {})
        self.stage = kw.get("stage", "")

    def get_field(self, name: str) -> Any:
        return self.fields.get(name)

    def get_records(self, collection: str) -> list[dict[str, Any]]:
        return self.records.get(collection, [])

    def get_events(self, event_type: str) -> list[dict[str, Any]]:
        return self.events.get(event_type, [])

    def get_resource(self, resource_id: str, holder: str) -> float:
        return self.resources.get((resource_id, holder), 0.0)

    def get_document_field(self, document_id: str, field_name: str) -> Any:
        return self.documents.get(document_id, {}).get(field_name)

    def get_stage(self) -> str:
        return self.stage

    def get_now(self) -> datetime:
        return T1

    def get_horizon(self) -> datetime:
        return T1

    def get_as_of(self) -> datetime:
        return T0


def _e(obj: Any) -> Any:
    return parse_expr(obj)


def test_comparison_and_boolean_operators() -> None:
    ctx = _Ctx(fields={"x": 3, "y": "hold"})
    assert evaluate(_e({"op": "greater_than", "args": [{"field": "x"}, 2]}), ctx) is True
    assert evaluate(_e({"op": "less_than", "args": [{"field": "x"}, 2]}), ctx) is False
    assert evaluate(_e({"op": "equals", "args": [{"field": "y"}, "hold"]}), ctx) is True
    assert (
        evaluate(
            _e(
                {
                    "op": "and",
                    "args": [
                        {"op": "greater_or_equal", "args": [{"field": "x"}, 3]},
                        {"op": "not", "args": [{"op": "equals", "args": [{"field": "y"}, "cut"]}]},
                    ],
                }
            ),
            ctx,
        )
        is True
    )


def test_count_and_sum_with_where_over_records() -> None:
    ctx = _Ctx(
        records={
            "votes": [
                {"key": "a", "value": "hold", "weight": 40},
                {"key": "b", "value": "hold", "weight": 35},
                {"key": "c", "value": "cut", "weight": 25},
            ]
        }
    )
    holds = _e(
        {
            "op": "count",
            "args": [
                "votes",
                {"op": "equals", "args": [{"op": "item", "args": ["value"]}, "hold"]},
            ],
        }
    )
    assert evaluate(holds, ctx) == 2
    weight = _e(
        {
            "op": "sum",
            "args": [
                "votes",
                "weight",
                {"op": "equals", "args": [{"op": "item", "args": ["value"]}, "hold"]},
            ],
        }
    )
    assert evaluate(weight, ctx) == 75.0
    assert evaluate(_e({"op": "count", "args": ["votes"]}), ctx) == 3


def test_event_count_and_existence() -> None:
    ctx = _Ctx(events={"reply_delivered": [{"type": "reply_delivered", "by": "d"}]})
    assert evaluate(_e({"op": "event_count", "args": ["reply_delivered"]}), ctx) == 1
    assert evaluate(_e({"op": "event_count", "args": ["never"]}), ctx) == 0
    assert evaluate(_e({"op": "exists", "args": ["missing"]}), _Ctx(records={})) is False


def test_temporal_operators() -> None:
    ctx = _Ctx()
    assert (
        evaluate(
            _e(
                {
                    "op": "before",
                    "args": [{"op": "as_of", "args": []}, {"op": "horizon", "args": []}],
                }
            ),
            ctx,
        )
        is True
    )
    dur = evaluate(
        _e(
            {"op": "duration", "args": [{"op": "as_of", "args": []}, {"op": "horizon", "args": []}]}
        ),
        ctx,
    )
    assert dur == (T1 - T0).total_seconds()


def test_numeric_string_compares_equal_to_number() -> None:
    ctx = _Ctx(fields={"n": "5"})
    assert evaluate(_e({"op": "equals", "args": [{"field": "n"}, 5]}), ctx) is True
