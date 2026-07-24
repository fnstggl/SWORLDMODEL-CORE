"""The universal declarative-expression evaluator.

This module hardcodes only *universal* logical and quantitative operators — the same
ones any question could need — and never a question type. It evaluates an
:class:`~sworldmodel.worldspec.Expr` tree against a read-only view of world state and
returns a value (number / string / bool / datetime / list).

Operators
---------
accessors : ``const``, ``field``, ``stage``, ``now``, ``horizon``, ``as_of``,
            ``resource``, ``document_field``, ``item`` (current record, inside a
            ``where`` predicate)
collections: ``count``, ``sum``, ``values``, ``exists``, ``event_count``
comparison : ``equals``, ``not_equals``, ``greater_than``, ``less_than``,
             ``greater_or_equal``, ``less_or_equal``, ``contains``
quantifier : ``all``, ``any``
temporal   : ``before``, ``after``, ``duration`` (seconds between two times)
boolean    : ``and``, ``or``, ``not``

A ``count``/``sum``/``values``/``exists`` over a record collection may take an optional
trailing ``where`` sub-expression; it is evaluated once per record with ``item(<attr>)``
bound to that record's attributes (``key``, ``value``, ``by``, ``time``, or any payload
field). This is enough to express, without any hardcoded family, predicates such as
"five recorded votes all equal hold", "a reply was delivered before the deadline", or
"summed stratum weight approving exceeds half the total".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from .worldspec import Expr


class ExprContext(Protocol):
    """The read-only world surface the evaluator needs. :class:`WorldState` implements
    it; a per-record context wraps it to bind ``item``."""

    def get_field(self, name: str) -> Any: ...
    def get_records(self, collection: str) -> list[dict[str, Any]]: ...
    def get_events(self, event_type: str) -> list[dict[str, Any]]: ...
    def get_resource(self, resource_id: str, holder: str) -> float: ...
    def get_document_field(self, document_id: str, field_name: str) -> Any: ...
    def get_stage(self) -> str: ...
    def get_now(self) -> datetime: ...
    def get_horizon(self) -> datetime: ...
    def get_as_of(self) -> datetime: ...


class _RecordContext:
    """Wraps a context so ``item(<attr>)`` reads the current record; everything else
    delegates to the base context."""

    def __init__(self, base: ExprContext, record: dict[str, Any]) -> None:
        self._base = base
        self._record = record

    @property
    def record(self) -> dict[str, Any]:
        return self._record

    def __getattr__(self, name: str) -> Any:  # delegate get_* to base
        return getattr(self._base, name)


def evaluate(expr: Any, ctx: ExprContext) -> Any:
    """Evaluate an expression (or bare literal) against ``ctx``."""

    if not isinstance(expr, Expr):
        return expr
    op = expr.op
    args = expr.args

    # -- literals / world accessors --------------------------------------------
    if op == "const":
        return args[0] if args else None
    if op == "field":
        return ctx.get_field(_s(evaluate(args[0], ctx)))
    if op == "stage":
        return ctx.get_stage()
    if op == "now":
        return ctx.get_now()
    if op == "horizon":
        return ctx.get_horizon()
    if op == "as_of":
        return ctx.get_as_of()
    if op == "resource":
        return ctx.get_resource(_s(evaluate(args[0], ctx)), _s(evaluate(args[1], ctx)))
    if op == "document_field":
        return ctx.get_document_field(_s(evaluate(args[0], ctx)), _s(evaluate(args[1], ctx)))
    if op == "item":
        rec = getattr(ctx, "record", {})
        return rec.get(_s(evaluate(args[0], ctx)))

    # -- collection aggregates (optional trailing where-expr) ------------------
    if op in ("count", "values", "sum", "exists"):
        return _aggregate(op, args, ctx)
    if op == "event_count":
        etype = _s(evaluate(args[0], ctx))
        where = args[1] if len(args) > 1 else None
        events = ctx.get_events(etype)
        return sum(1 for e in events if _match(where, e, ctx))

    # -- comparison ------------------------------------------------------------
    if op == "equals":
        return _norm(evaluate(args[0], ctx)) == _norm(evaluate(args[1], ctx))
    if op == "not_equals":
        return _norm(evaluate(args[0], ctx)) != _norm(evaluate(args[1], ctx))
    if op == "greater_than":
        return _num(evaluate(args[0], ctx)) > _num(evaluate(args[1], ctx))
    if op == "less_than":
        return _num(evaluate(args[0], ctx)) < _num(evaluate(args[1], ctx))
    if op == "greater_or_equal":
        return _num(evaluate(args[0], ctx)) >= _num(evaluate(args[1], ctx))
    if op == "less_or_equal":
        return _num(evaluate(args[0], ctx)) <= _num(evaluate(args[1], ctx))
    if op == "contains":
        container = evaluate(args[0], ctx)
        needle = evaluate(args[1], ctx)
        try:
            return needle in container
        except TypeError:
            return False

    # -- quantifiers over an evaluated list ------------------------------------
    if op == "all":
        seq = evaluate(args[0], ctx)
        return all(bool(x) for x in _as_list(seq))
    if op == "any":
        seq = evaluate(args[0], ctx)
        return any(bool(x) for x in _as_list(seq))

    # -- temporal --------------------------------------------------------------
    if op == "before":
        return _time(evaluate(args[0], ctx)) < _time(evaluate(args[1], ctx))
    if op == "after":
        return _time(evaluate(args[0], ctx)) > _time(evaluate(args[1], ctx))
    if op == "duration":  # seconds from args[0] to args[1]
        return (_time(evaluate(args[1], ctx)) - _time(evaluate(args[0], ctx))).total_seconds()

    # -- boolean ---------------------------------------------------------------
    if op == "and":
        return all(bool(evaluate(a, ctx)) for a in args)
    if op == "or":
        return any(bool(evaluate(a, ctx)) for a in args)
    if op == "not":
        return not bool(evaluate(args[0], ctx))

    raise ValueError(f"unknown expression operator {op!r}")


def _aggregate(op: str, args: tuple[Any, ...], ctx: ExprContext) -> Any:
    collection = _s(evaluate(args[0], ctx))
    records = ctx.get_records(collection)
    if op == "sum":
        value_key = _s(evaluate(args[1], ctx)) if len(args) > 1 else "value"
        where = args[2] if len(args) > 2 else None
        total = 0.0
        for r in records:
            if _match(where, r, ctx):
                total += _num(r.get(value_key, r.get("value")))
        return total
    where = args[1] if len(args) > 1 else None
    matched = [r for r in records if _match(where, r, ctx)]
    if op == "count":
        return len(matched)
    if op == "exists":
        return len(matched) > 0
    return [r.get("value") for r in matched]  # values


def _match(where: Any, record: dict[str, Any], ctx: ExprContext) -> bool:
    if where is None:
        return True
    return bool(evaluate(where, _RecordContext(ctx, record)))


# ---------------------------------------------------------------------------
# Coercions (total; never raise on a missing/None value)
# ---------------------------------------------------------------------------


def _s(v: Any) -> str:
    return v if isinstance(v, str) else ("" if v is None else str(v))


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)  # numeric string
    except (TypeError, ValueError):
        return 0.0


def _norm(v: Any) -> Any:
    # Numeric strings compare equal to numbers; everything else compares as-is.
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)  # a numeric string compares equal to the number
        except ValueError:
            return v
    return v


def _as_list(v: Any) -> list[Any]:
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return [v]


def _time(v: Any) -> datetime:
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    if isinstance(v, datetime):
        # Normalize to timezone-aware UTC so a model-emitted naive datetime never
        # crashes a comparison against the (timezone-aware) as_of / horizon.
        return v if v.tzinfo is not None else v.replace(tzinfo=UTC)
    raise ValueError(f"not a time value: {v!r}")
