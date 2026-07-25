"""The universal effect language — the ONLY hardcoded action machinery.

A compiled :class:`~sworldmodel.worldspec.Effect` names one of a small, fixed set of
*universal world operations*. These are the execution language, analogous to the
primitives of a programming language: the LLM composes them into scenario-specific
actions, and the runtime executes them safely. The set never grows when a new kind of
question appears.

Operations
----------
``create_event``       a generic happening exposed to observers
``schedule_event``     a create_event stamped at a future time
``deliver_information``deliver text / field levels to specific actors
``release_data``       release external data (sets public world fields)
``set_field`` / ``adjust_field``   set / increment a typed world field
``append_record``      append an entry to a named collection (vote/offer/decision/…)
``update_commitment``  record a commitment by an actor
``transfer_resource``  move a resource between holders (checked: never negative)
``consume_resource``   consume a resource from a holder (checked)
``create_or_update_document``  create / modify a document or object
``advance_time``       move world time forward

Effect parameters may be literals or *binding strings* resolved at execution time:
``$actor`` (the acting actor), ``$target`` (its chosen target), ``$param.<name>``
(an action parameter), ``$self.<attr>`` (an attribute/role/name of the actor), and
``$now``. Nested dict/list params are resolved recursively.

An action whose effect cannot be applied safely (a resource would go negative, an
unknown effect op) is **rejected** — never partially applied, never silently coerced.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .errors import UndeterminedExpressionError
from .expressions import evaluate, looks_like_expression
from .ids import content_id
from .models import Event, Visibility, make_payload
from .world import WorldState
from .worldspec import Effect, parse_expr

# The closed set of universal effect operations. This is the whole execution language.
UNIVERSAL_OPS = frozenset(
    {
        "create_event",
        "schedule_event",
        "deliver_information",
        "release_data",
        "set_field",
        "adjust_field",
        "append_record",
        "update_commitment",
        "transfer_resource",
        "consume_resource",
        "create_or_update_document",
        "advance_time",
    }
)

# Default visibility per op (overridable via an effect ``visibility`` param).
_DEFAULT_VISIBILITY = {
    "create_event": Visibility.PUBLIC,
    "schedule_event": Visibility.PUBLIC,
    "deliver_information": Visibility.PRIVATE,
    "release_data": Visibility.PUBLIC,
    "set_field": Visibility.PRIVATE,
    "adjust_field": Visibility.PRIVATE,
    "append_record": Visibility.PUBLIC,
    "update_commitment": Visibility.PUBLIC,
    "transfer_resource": Visibility.PRIVATE,
    "consume_resource": Visibility.PRIVATE,
    "create_or_update_document": Visibility.PUBLIC,
    "advance_time": Visibility.PRIVATE,
}


class EffectExecutor:
    """Builds ledger events from compiled effects, with a monotonic sequence so ids
    are unique and deterministic under the (deterministic) rollout order."""

    def __init__(self) -> None:
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- feasibility (dry run; no partial application) --------------------------

    def can_apply(
        self, world: WorldState, effects: tuple[Effect, ...], binding: dict[str, Any]
    ) -> tuple[bool, str]:
        """Return ``(ok, reason)``. Checks that every op is universal, that every value
        an effect will treat as a quantity really is one, and that no resource would go
        negative when the effects are applied in order."""

        resources = dict(world.resources)
        for eff in effects:
            if eff.op not in UNIVERSAL_OPS:
                return False, f"effect op {eff.op!r} is not a universal world operation"
            p = _resolve_params(eff.params_dict, binding, world)
            bad = _unusable_quantity(eff.op, p)
            if bad:
                # Coercing an unreadable amount to zero would turn "transfer what I
                # said" into "transfer nothing" and record it as done.
                return False, bad
            if eff.op == "transfer_resource":
                res, frm, amt = str(p.get("resource")), str(p.get("from")), _num(p.get("amount"))
                if amt < 0:
                    return False, "transfer amount is negative"
                key = f"{res}@{frm}"
                if _num(resources.get(key, 0.0)) < amt:
                    return False, f"holder {frm!r} lacks {amt} of resource {res!r}"
                resources[key] = _num(resources.get(key, 0.0)) - amt
                to_key = f"{res}@{p.get('to')}"
                resources[to_key] = _num(resources.get(to_key, 0.0)) + amt
            elif eff.op == "consume_resource":
                res, holder = str(p.get("resource")), str(p.get("holder"))
                amt = _num(p.get("amount"))
                key = f"{res}@{holder}"
                if amt < 0:
                    return False, "consume amount is negative"
                if _num(resources.get(key, 0.0)) < amt:
                    return False, f"holder {holder!r} lacks {amt} of resource {res!r}"
                resources[key] = _num(resources.get(key, 0.0)) - amt
        return True, "ok"

    # -- event construction -----------------------------------------------------

    def build_events(
        self, world: WorldState, effects: tuple[Effect, ...], binding: dict[str, Any]
    ) -> tuple[list[Event], list[tuple[datetime, Effect]]]:
        """Split compiled effects into what happens **now** and what is *scheduled*.

        An effect stamped with a future time is not something that has happened; it is
        something that will. Applying it immediately — which is what a single event list
        forces — both makes the future arrive early and drags the branch clock forward
        over everything legitimately scheduled in between. ``schedule_event`` in
        particular has to actually schedule.
        """

        actor_id = binding.get("actor")
        events: list[Event] = []
        deferred: list[tuple[datetime, Effect]] = []
        for eff in effects:
            p = _resolve_params(eff.params_dict, binding, world)
            when = _event_time(world, eff.op, p)
            if when > world.time:
                deferred.append((when, eff))
                continue
            events.append(self._event(world, eff, p, actor_id))
        return events, deferred

    def raw_event(
        self,
        world: WorldState,
        *,
        kind: str,
        actor_id: str | None,
        payload: dict[str, Any],
        visibility: Visibility = Visibility.PRIVATE,
        audience: tuple[str, ...] = (),
    ) -> Event:
        """Build a bookkeeping event (e.g. a recorded wait or an action rejection)
        that is not itself a world-mutating effect op."""

        seq = self._next_seq()
        eid = content_id("ev", world.branch_id, kind, actor_id, seq)
        return Event(
            event_id=eid,
            branch_id=world.branch_id,
            time=world.time,
            kind=kind,
            actor_id=actor_id,
            target_ids=audience,
            payload=make_payload(payload),
            visibility=visibility,
            audience=audience,
        )

    def _event(self, world: WorldState, eff: Effect, p: dict[str, Any], actor_id: Any) -> Event:
        visibility = _visibility(eff.op, p)
        audience = _audience(eff.op, p)
        # An applied event happens now. Anything later was split off as deferred.
        time = min(_event_time(world, eff.op, p), world.time)
        payload = _payload_for(eff.op, p)
        ev_ids = tuple(str(x) for x in (p.get("evidence_claim_ids") or []))
        seq = self._next_seq()
        eid = content_id("ev", world.branch_id, eff.op, actor_id, seq)
        return Event(
            event_id=eid,
            branch_id=world.branch_id,
            time=time,
            kind=eff.op,
            actor_id=actor_id if isinstance(actor_id, str) else None,
            target_ids=audience,
            payload=make_payload(payload),
            visibility=visibility,
            audience=audience,
            evidence_claim_ids=ev_ids,
        )


# ---------------------------------------------------------------------------
# Parameter binding resolution
# ---------------------------------------------------------------------------


def _resolve_params(
    params: dict[str, Any], binding: dict[str, Any], world: WorldState
) -> dict[str, Any]:
    return {k: _resolve(v, binding, world) for k, v in params.items()}


def _resolve(value: Any, binding: dict[str, Any], world: WorldState) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return _resolve_ref(value, binding, world)
    if isinstance(value, dict):
        if looks_like_expression(value):
            return _evaluate_param(value, world)
        return {k: _resolve(v, binding, world) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, binding, world) for v in value]
    return value


def _evaluate_param(value: dict[str, Any], world: WorldState) -> Any:
    """Compute the expression against the world the effect is being applied to.

    A live Tesla run set the quarter's deliveries to ``Q1_deliveries *
    demand_multiplier`` — produced by a delivery-cycle process rather than by an
    invented executive, which is the right shape — and the effect wrote the *formula*
    into the field. The terminal then compared a dict against 400,000, could not, and
    the forecast came back 1.0 unresolved on every branch: a hollow answer from a world
    that held every number it needed.

    An expression the world cannot yet determine resolves to *no value*, never to the
    expression object and never to a coerced zero: writing the formula would state a
    dict as the field's value, and writing zero would state a quantity nobody produced.
    """

    try:
        return evaluate(parse_expr(value), world)
    except UndeterminedExpressionError:
        return None
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def _resolve_ref(ref: str, binding: dict[str, Any], world: WorldState) -> Any:
    if ref == "$actor":
        return binding.get("actor")
    if ref == "$target":
        return binding.get("target", "")
    if ref == "$now":
        return world.time.isoformat()
    if ref.startswith("$param."):
        return binding.get("params", {}).get(ref[len("$param.") :])
    if ref.startswith("$self."):
        entity = binding.get("self")
        attr = ref[len("$self.") :]
        if entity is None:
            return None
        if attr == "role":
            return entity.role
        if attr == "name":
            return entity.name
        if attr == "id":
            return entity.entity_id
        return entity.attributes_dict.get(attr)
    return ref  # an unknown $-string is treated as a literal


# ---------------------------------------------------------------------------
# Per-op payload / visibility / time
# ---------------------------------------------------------------------------


def _payload_for(op: str, p: dict[str, Any]) -> dict[str, Any]:
    if op == "create_event" or op == "schedule_event":
        return {
            "event_type": str(p.get("event_type", "happening")),
            "text": str(p.get("text", "")),
            "data": _as_dict(p.get("data")),
        }
    if op == "deliver_information":
        return {"text": str(p.get("text", "")), "info_fields": _as_dict(p.get("info_fields"))}
    if op == "release_data":
        return {"fields": _as_dict(p.get("fields"))}
    if op == "set_field":
        return {"field": str(p.get("field", "")), "value": p.get("value")}
    if op == "adjust_field":
        return {"field": str(p.get("field", "")), "delta": _num(p.get("delta"))}
    if op == "append_record":
        return {
            "collection": str(p.get("collection", "")),
            "key": str(p.get("key", "")) if p.get("key") is not None else "",
            "value": p.get("value"),
            "extra": _as_dict(p.get("extra")),
        }
    if op == "update_commitment":
        return {"text": str(p.get("text", "")), "tag": str(p.get("tag", ""))}
    if op == "transfer_resource":
        return {
            "resource": str(p.get("resource", "")),
            "from": str(p.get("from", "")),
            "to": str(p.get("to", "")),
            "amount": _num(p.get("amount")),
        }
    if op == "consume_resource":
        return {
            "resource": str(p.get("resource", "")),
            "holder": str(p.get("holder", "")),
            "amount": _num(p.get("amount")),
        }
    if op == "create_or_update_document":
        return {"document": str(p.get("document", "")), "fields": _as_dict(p.get("fields"))}
    if op == "advance_time":
        return {"reason": str(p.get("reason", "time advanced"))}
    return dict(p)


def _visibility(op: str, p: dict[str, Any]) -> Visibility:
    raw = p.get("visibility")
    if isinstance(raw, str):
        try:
            return Visibility(raw)
        except ValueError:
            pass
    # deliver_information with no audience is effectively public.
    if op == "deliver_information" and not _audience(op, p):
        return Visibility.PUBLIC
    return _DEFAULT_VISIBILITY.get(op, Visibility.PUBLIC)


def _audience(op: str, p: dict[str, Any]) -> tuple[str, ...]:
    to = p.get("to") if p.get("to") is not None else p.get("audience")
    if isinstance(to, str):
        return (to,)
    if isinstance(to, (list, tuple)):
        return tuple(str(x) for x in to)
    return ()


def _event_time(world: WorldState, op: str, p: dict[str, Any]) -> datetime:
    at = p.get("at")
    if isinstance(at, str) and at:
        try:
            return datetime.fromisoformat(at)
        except ValueError:
            pass
    if op == "advance_time":
        by = p.get("by_seconds")
        if by is not None:
            return world.time + timedelta(seconds=_num(by))
    if op == "schedule_event":
        after = p.get("after_seconds")
        if after is not None:
            return world.time + timedelta(seconds=_num(after))
    return world.time


# Effect parameters that are genuinely quantities: if one of these cannot be read as a
# number, the action is refused rather than applied with a value nobody chose.
_QUANTITY_PARAMS: dict[str, tuple[str, ...]] = {
    "transfer_resource": ("amount",),
    "consume_resource": ("amount",),
    "adjust_field": ("delta",),
    "advance_time": ("by_seconds",),
    "schedule_event": ("after_seconds",),
}


def _unusable_quantity(op: str, p: dict[str, Any]) -> str:
    for name in _QUANTITY_PARAMS.get(op, ()):
        if name not in p or p[name] is None:
            continue
        if not _is_number(p[name]):
            return (
                f"{op} needs {name!r} to be a number; got {p[name]!r}, which cannot be read as one"
            )
    return ""


def _is_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        float(v)
    except (TypeError, ValueError):
        return False
    return True


def _as_dict(v: Any) -> dict[str, Any]:
    return dict(v) if isinstance(v, dict) else {}


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
