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


# --------------------------------------------------------------------------- #
# Effect parameters carrying a computed value
# --------------------------------------------------------------------------- #


def test_an_effect_value_that_is_an_expression_is_computed_not_stored() -> None:
    """A live Tesla run compiled a world with no invented executive — deliveries were
    produced by a delivery-cycle process, correctly — and the process set the quarter's
    deliveries to ``Q1_deliveries * demand_multiplier``. The effect wrote the *formula*
    into the field. The terminal then compared a dict against 400,000, could not, and
    the forecast came back 1.0 unresolved on both branches: a hollow answer from a world
    that held every number it needed."""

    from sworldmodel.effects import _resolve

    ctx = _Ctx(fields={"Q1_deliveries": 358023, "demand_multiplier": 1.2})
    computed = _resolve(
        {"op": "multiply", "args": [{"field": "Q1_deliveries"}, {"field": "demand_multiplier"}]},
        {},
        ctx,  # type: ignore[arg-type]
    )
    assert computed == 358023 * 1.2
    assert evaluate(_e({"op": "greater_than", "args": [computed, 400000]}), ctx) is True

    # Nested inside a payload, and reached through a list, on the same rule.
    payload = _resolve(
        {"fields": {"deliveries": {"field": "Q1_deliveries"}}, "seen": [{"field": "unset"}]},
        {},
        ctx,  # type: ignore[arg-type]
    )
    assert payload == {"fields": {"deliveries": 358023}, "seen": [None]}


def test_a_payload_that_merely_looks_like_an_expression_stays_data() -> None:
    """Recognition is closed over the operators the evaluator implements, so a document
    field named ``op`` is still a document field."""

    from sworldmodel.effects import _resolve

    ctx = _Ctx(fields={})
    data = {"op": "sign the agreement", "args": ["EU", "Mercosur"]}
    assert _resolve(data, {}, ctx) == data  # type: ignore[arg-type]
    assert _resolve({"field": "x", "note": "y"}, {}, ctx) == {  # type: ignore[arg-type]
        "field": "x",
        "note": "y",
    }


def test_an_undeterminable_effect_value_is_no_value_never_a_coerced_zero() -> None:
    """Writing zero would state a quantity nobody produced; writing the formula would
    state a dict as the field's value. Neither is the truth, which is that the world has
    not determined it."""

    from sworldmodel.effects import _resolve

    ctx = _Ctx(fields={"known": 4})
    undetermined = {"op": "multiply", "args": [{"field": "known"}, {"field": "never_set"}]}
    assert _resolve(undetermined, {}, ctx) is None  # type: ignore[arg-type]


def test_an_ordinary_payload_key_that_is_also_an_operator_stays_data() -> None:
    """``count``, ``sum``, ``min`` and ``max`` are operators and also ordinary names for
    a thing a document records. Reading ``{"count": 3}`` as the aggregate ``count(3)``
    would quietly turn a recorded number into nothing."""

    from sworldmodel.effects import _resolve

    ctx = _Ctx(fields={"n": 7})
    for key in ("count", "sum", "values", "min", "max", "exists", "contains"):
        assert _resolve({key: 3}, {}, ctx) == {key: 3}  # type: ignore[arg-type]
    # The explicit form is unambiguous and is still computed.
    assert _resolve({"op": "add", "args": [{"field": "n"}, 1]}, {}, ctx) == 8  # type: ignore[arg-type]
