"""The single ledger-replay core (standing decision D7).

Every reconstruction of a run from its own record goes through this module: the
production trace writer derives the persisted observability artifacts from it at
trace-write time, the publication gate replays deletion counterfactuals through it,
and ``scripts/forensics.py`` rebuilds published numbers with it. No second
implementation of ledger replay may exist anywhere — a forensic tool that replays the
ledger one way while the trace writer replays it another is two opinions about what
happened, and a reviewer cannot audit an opinion.

What lives here:

* **branch-key joining** — branch ids appear with and without the structure namespace
  prefix across artifacts (``primary/sc_x`` in ``forecast.json`` vs ``sc_x`` in older
  ledgers); :func:`branch_key` is the one joined view every reader uses.
* **event replay** — the exact, total mapping from recorded events to field state,
  event-type counts, record collections and event history (``release_data``,
  ``set_field``, ``adjust_field``, ``create_event``, ``append_record``). Nothing else
  writes state. Record CONTENT is reconstructed, not just cardinality: the terminals
  these worlds actually use read a record's ``value``, and a replay that could only
  count records answered every such question NO (FD-42).
* **initial state** — per-branch complete field state after scenario conditions and
  before any simulated effect, from the executed compilation's field initials plus the
  branch's own scenario ``release_data`` conditions.
* **state-diff derivation** — ordered minimal diffs such that initial state plus the
  diffs reconstructs every branch's final fields exactly, with zero LLM calls.
* **communications / process-transition extraction** — including the honest
  delivery→notice join against the recorded actor decisions: a communication no
  decision ever consumed is ``noticed="unknown"``, never guessed.
* **keep-predicate counterfactual replay** — remove selected recorded events, replay,
  re-evaluate the terminal with the engine's own evaluator, and report what the number
  rested on.
* **full reconstruction** (:func:`reconstruct_run`) and the self-contained HTML
  dossier renderer, shared by the production trace writer and the forensic tool.

Everything is total over both artifact dicts (the on-disk JSON schema) and live
runtime objects (:class:`~sworldmodel.models.Event`,
:class:`~sworldmodel.engine.ActorDecisionRecord`, gateway call records): the same
replay runs at trace-write time and years later from the files alone. A field the
ledger never wrote reads ``None``; a value the run did not record is reported as not
recorded. Nothing here invents a value.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import UndeterminedExpressionError
from .expressions import evaluate
from .worldspec import Expr, parse_expr

# Provider list price for the model these runs used, in dollars per million tokens.
# Recorded here as an explicit, labeled ESTIMATE: the trace records tokens, which are a
# fact, and cost is a rate applied to them. It is never written as if it were measured.
RATE_IN_PER_MTOK = 0.28
RATE_OUT_PER_MTOK = 0.42

# One recorded event / decision / call, as an artifact dict or a live runtime object.
EventLike = Any
KeepPredicate = Callable[[EventLike], bool]

# The event kinds that are a communication: something said, sent, delivered, or
# announced — as opposed to the world silently changing.
COMMUNICATION_EVENT_KINDS = ("send_information", "deliver_information", "create_event")


# --------------------------------------------------------------------------- #
# Total accessors over artifact dicts and live objects
# --------------------------------------------------------------------------- #


def record_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an artifact dict or a live object attribute — never raises."""

    if isinstance(obj, Mapping):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return default if value is None and default is not None else value


def event_kind(e: EventLike) -> str:
    return str(record_get(e, "kind") or "")


def event_payload(e: EventLike) -> dict[str, Any]:
    if isinstance(e, Mapping):
        payload = e.get("payload")
        return dict(payload) if isinstance(payload, Mapping) else {}
    payload_dict = getattr(e, "payload_dict", None)
    if isinstance(payload_dict, Mapping):
        return dict(payload_dict)
    payload = getattr(e, "payload", None)
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, (list, tuple)):
        try:
            return dict(payload)
        except (TypeError, ValueError):
            return {}
    return {}


def event_actor_id(e: EventLike) -> str | None:
    actor = record_get(e, "actor_id")
    return str(actor) if actor else None


def event_branch_id(e: EventLike) -> str:
    return str(record_get(e, "branch_id") or "")


def event_id_of(e: EventLike) -> str:
    return str(record_get(e, "event_id") or "")


def event_time(e: EventLike) -> str:
    t = record_get(e, "time")
    if isinstance(t, datetime):
        return t.isoformat()
    return str(t) if t is not None else ""


def event_visibility(e: EventLike) -> str:
    v = record_get(e, "visibility")
    value = getattr(v, "value", v)
    return str(value) if value is not None else ""


def event_evidence_ids(e: EventLike) -> list[str]:
    ids = record_get(e, "evidence_claim_ids") or []
    return [str(i) for i in ids]


# --------------------------------------------------------------------------- #
# Branch-key joining (FD-16)
# --------------------------------------------------------------------------- #


def branch_key(branch_id: str) -> str:
    """The joined view of a branch id.

    Branch ids are namespaced with their structure prefix (``primary/sc_x``) since
    commit b2c10ed, but appear both with and without the prefix across a run's own
    artifacts (and across runs recorded before the namespacing). Every reader joins
    on this key; no reader compares raw branch ids across artifacts.
    """

    return branch_id.split("/", 1)[1] if "/" in branch_id else branch_id


def group_events_by_branch(events: Iterable[EventLike]) -> dict[str, list[EventLike]]:
    """Events grouped by joined branch key, preserving ledger order within a branch."""

    grouped: dict[str, list[EventLike]] = defaultdict(list)
    for e in events:
        grouped[branch_key(event_branch_id(e))].append(e)
    return dict(grouped)


# --------------------------------------------------------------------------- #
# Event -> state replay (the one implementation)
# --------------------------------------------------------------------------- #


def _apply_field_op(fields: dict[str, Any], e: EventLike) -> None:
    """Apply one recorded event's field effect. The ONLY event→field mapping."""

    kind = event_kind(e)
    payload = event_payload(e)
    if kind == "release_data":
        released = payload.get("fields") or {}
        if isinstance(released, Mapping):
            for k, v in released.items():
                fields[str(k)] = v
    elif kind == "set_field":
        if "field" in payload:
            fields[str(payload["field"])] = payload.get("value")
    elif kind == "adjust_field":
        name = str(payload.get("field", ""))
        delta = payload.get("delta")
        if name and delta is not None:
            fields[name] = float(fields.get(name) or 0) + float(delta)


def _apply_count_op(counts: dict[str, int], e: EventLike) -> None:
    """Apply one recorded event's count effect. The ONLY event→count mapping."""

    kind = event_kind(e)
    payload = event_payload(e)
    if kind == "create_event":
        event_type = payload.get("event_type")
        if event_type:
            counts[str(event_type)] += 1
    elif kind == "append_record":
        collection = payload.get("collection")
        if collection:
            counts[str(collection)] += 1


def _apply_record_op(records: dict[str, list[dict[str, Any]]], e: EventLike) -> None:
    """Apply one recorded event's collection effect. The ONLY event→record mapping.

    Mirrors :class:`sworldmodel.world.Record` and its ``as_item()`` view exactly, so a
    record the evaluator reads on replay carries the same attributes it carried live —
    ``key``, ``value``, ``by``, ``time``, plus whatever the effect put in ``extra``.
    Every attribute comes from the recorded payload or from the event's own envelope;
    a payload that recorded no value yields ``value: None``, never a substitute.
    """

    if event_kind(e) != "append_record":
        return
    payload = event_payload(e)
    collection = payload.get("collection")
    if not collection:
        return
    actor = event_actor_id(e)
    item: dict[str, Any] = {
        "key": str(payload.get("key", actor or "")),
        "value": payload.get("value"),
        "by": str(actor or payload.get("by", "environment")),
        "time": event_time(e),
    }
    extra = payload.get("extra")
    if isinstance(extra, Mapping):
        item.update({str(k): v for k, v in extra.items()})
    records.setdefault(str(collection), []).append(item)


def _apply_event_item_op(items: dict[str, list[dict[str, Any]]], e: EventLike) -> None:
    """Apply one recorded event's event-history effect. The ONLY event→event-item mapping.

    Mirrors :meth:`sworldmodel.world.WorldState.get_events`, which matches an event
    either by its runtime kind or by the ``event_type`` its payload declares, and
    presents it as ``{type, by, time}`` overlaid with the recorded payload. Indexing
    under both names is what makes ``event_count('x', <where>)`` replay the way the
    engine evaluated it.
    """

    payload = event_payload(e)
    names = {event_kind(e)}
    declared = payload.get("event_type")
    if declared:
        names.add(str(declared))
    for name in names:
        if not name:
            continue
        item: dict[str, Any] = {
            "type": name,
            "by": event_actor_id(e) or "environment",
            "time": event_time(e),
        }
        item.update(payload)
        items.setdefault(name, []).append(item)


@dataclass(frozen=True)
class ReplayState:
    """Everything one replay of the recorded ledger produced.

    ``fields`` and ``counts`` are the state and cardinality views the replay has always
    had. ``records`` and ``events`` are the CONTENT views (FD-42): the terminals these
    worlds actually compile — ``count('positions', equals(item('value'), 'hold')) >= 5``
    — read a record's value, not how many records there are. A replay that could only
    count them evaluated the predicate over blanks and answered NO to every such
    question, confidently and wrongly.

    A collection or event type absent from the mapping is one the replayed ledger never
    wrote. It is empty, not unknown.
    """

    fields: dict[str, Any]
    counts: dict[str, int]
    records: dict[str, list[dict[str, Any]]]
    events: dict[str, list[dict[str, Any]]]


def replay(events: Sequence[EventLike], keep: KeepPredicate | None = None) -> ReplayState:
    """Replay the recorded ledger into complete state. The one replay implementation.

    This is the whole counterfactual machine: ``keep`` selects which recorded events
    are allowed to have happened, and the terminal is then re-evaluated over the
    result. Nothing is simulated and no model is called. Every view is filtered by the
    same ``keep``, so a deleted event takes its record and its event item with it —
    a reconstruction that returned content regardless of ``keep`` would make every
    deletion counterfactual answer YES and silently void the responsibility finding.
    """

    fields: dict[str, Any] = {}
    counts: dict[str, int] = defaultdict(int)
    records: dict[str, list[dict[str, Any]]] = {}
    items: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        if keep is not None and not keep(e):
            continue
        _apply_field_op(fields, e)
        _apply_count_op(counts, e)
        _apply_record_op(records, e)
        _apply_event_item_op(items, e)
    return ReplayState(fields, dict(counts), records, items)


def replay_fields(
    events: Sequence[EventLike], keep: KeepPredicate | None = None
) -> tuple[dict[str, Any], dict[str, int]]:
    """The (fields, event-type counts) projection of :func:`replay`.

    Kept for readers that genuinely only need field state and cardinality. A caller
    that re-evaluates a terminal wants :func:`replay`: dropping the record and event
    content is what made a content-predicated terminal replay as NO.
    """

    state = replay(events, keep)
    return state.fields, state.counts


def fields_written_by_events(events: Sequence[EventLike]) -> set[str]:
    """Every field name some recorded event actually wrote."""

    written: set[str] = set()
    for e in events:
        payload = event_payload(e)
        kind = event_kind(e)
        if kind == "release_data":
            released = payload.get("fields") or {}
            if isinstance(released, Mapping):
                written |= {str(k) for k in released}
        elif kind in ("set_field", "adjust_field") and "field" in payload:
            written.add(str(payload["field"]))
    return written


# --------------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------------- #


def _spec_field_entries(world: Any) -> list[Any] | None:
    """The field declarations of an executable world — artifact dict or live spec."""

    if world is None:
        return None
    if isinstance(world, Mapping):
        spec = world.get("world_spec")
        if isinstance(spec, Mapping):
            entries = spec.get("fields")
            if isinstance(entries, list) and entries:
                return entries
        return None
    entries_attr = getattr(world, "fields", None)
    if entries_attr:
        return list(entries_attr)
    return None


def initial_fields_from_world(world: Any) -> dict[str, Any] | None:
    """Every declared field's initial value, from the run's own executable world.

    Accepts the persisted ``compiled_world.json`` / ``executed_compilation`` dict or a
    live :class:`~sworldmodel.worldspec.WorldSpec`. Returns ``None`` when the source
    declares no fields — the caller falls back to the carried-in inference, it never
    invents initials.
    """

    entries = _spec_field_entries(world)
    if not entries:
        return None
    out: dict[str, Any] = {}
    for f in entries:
        fid = record_get(f, "field_id")
        if fid is not None:
            out[str(fid)] = record_get(f, "initial")
    return out or None


def coerce_recorded_value(value: Any) -> Any:
    """Recorded state values are stringified in the branch table; restore their type."""

    if not isinstance(value, str):
        return value
    if value in ("True", "False"):
        return value == "True"
    if value == "None":
        return None
    try:
        return float(value) if ("." in value or "e" in value.lower()) else int(value)
    except ValueError:
        return value


def recorded_final_fields(branch_entry: Mapping[str, Any]) -> dict[str, Any]:
    """The branch's final field values as the run itself recorded them."""

    out: dict[str, Any] = {}
    for key, value in (branch_entry.get("world_state") or {}).items():
        if str(key).startswith("field:"):
            out[str(key)[len("field:") :]] = coerce_recorded_value(value)
    return out


def replay_initial_fields(
    world_initials: Mapping[str, Any] | None,
    final_fields: Mapping[str, Any],
    events: Sequence[EventLike],
) -> dict[str, Any]:
    """The branch's field state before anything ran, for replay purposes.

    Preferred source is the executable world the run persisted. Runs written before
    that artifact existed are reconstructed the only sound way available: a field that
    appears in the recorded FINAL state while no recorded event ever wrote it cannot
    have been produced by the trajectory, so its final value is also its initial value.
    That inference is itself the forensic finding for such a field — it was carried in,
    not produced — and it is reported as such rather than hidden.
    """

    if world_initials:
        return dict(world_initials)
    written = fields_written_by_events(events)
    return {k: v for k, v in final_fields.items() if k not in written}


def scenario_conditions(events: Sequence[EventLike]) -> dict[str, Any]:
    """This branch's own scenario conditions, from its seed ``release_data`` event."""

    conditions: dict[str, Any] = {}
    for e in events:
        payload = event_payload(e)
        if event_kind(e) == "release_data" and "branch_conditions" in payload:
            declared = payload.get("branch_conditions") or {}
            if isinstance(declared, Mapping):
                conditions.update({str(k): v for k, v in declared.items()})
    return conditions


def branch_initial_state(
    world_initials: Mapping[str, Any] | None, events: Sequence[EventLike]
) -> dict[str, Any]:
    """Complete per-branch field state after scenario conditions, before any effect.

    The executed compilation's field initials, overlaid with the field levels of this
    branch's own scenario ``release_data`` event (the one carrying
    ``branch_conditions``). A declared field neither initialized nor conditioned reads
    ``None`` — it is present, honestly unset, never invented.

    A branch hypothesis whose release is *dated* is stamped ``deferred_release`` and is
    excluded: the branch did not start out knowing that value, it learned it when the
    release fired, and folding it in here would restate the initial state as if the
    future had already happened. Records written before that stamp existed carry no
    such key and are read exactly as before.
    """

    state: dict[str, Any] = dict(world_initials or {})
    for e in events:
        payload = event_payload(e)
        if event_kind(e) == "release_data" and "branch_conditions" in payload:
            if payload.get("deferred_release"):
                continue
            released = payload.get("fields") or {}
            if isinstance(released, Mapping):
                for k, v in released.items():
                    state[str(k)] = v
    return state


# --------------------------------------------------------------------------- #
# Terminal evaluation (the engine's own evaluator over replayed state)
# --------------------------------------------------------------------------- #


class Unreconstructable(Exception):
    """The terminal cannot be re-evaluated from what this run recorded."""


def _copy_content(
    source: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]] | None:
    """A private copy of a reconstructed content mapping — ``None`` stays ``None``.

    ``None`` and ``{}`` mean different things here and must not collapse: ``{}`` is a
    replay that reconstructed content and found none, ``None`` is a caller that supplied
    no content at all.
    """

    if source is None:
        return None
    return {str(k): [dict(r) for r in v] for k, v in source.items()}


class ReplayWorld:
    """Replayed state, presented as the read-only surface the evaluator expects.

    The replay re-evaluates a run's terminal against states replayed from the recorded
    event ledger — including counterfactual states with some events removed. To do that
    with the engine's OWN evaluator rather than a reimplementation of it, the replayed
    state has to satisfy :class:`~sworldmodel.expressions.ExprContext`.

    This is that adapter and nothing more. It holds replayed fields, event-type counts,
    record collections and event history, and answers the evaluator's questions from
    them. It never invents a value: a field the ledger never set reads as ``None``,
    which is exactly what the runtime's unresolved guards test for.

    ``records`` / ``events`` come from :func:`replay`, which reconstructs each record's
    ``key``/``value``/``by``/``time`` from the recorded ``append_record`` payload. Omit
    them and this world knows only cardinality: a ``count('positions') >= 5`` terminal
    still replays exactly, while ``count('positions', equals(item('value'), 'hold'))``
    reads blank placeholders and counts zero. That is FD-42, and it is why every caller
    in this package supplies them.
    """

    def __init__(
        self,
        fields: dict[str, Any],
        counts: dict[str, int],
        *,
        now: datetime | None = None,
        horizon: datetime | None = None,
        as_of: datetime | None = None,
        documents: dict[str, dict[str, Any]] | None = None,
        resources: dict[tuple[str, str], float] | None = None,
        stage: str = "final",
        records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        events: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self._fields = dict(fields)
        self._counts = dict(counts)
        self._records = _copy_content(records)
        self._events = _copy_content(events)
        self._documents = documents or {}
        self._resources = resources or {}
        self._stage = stage
        default = datetime(1970, 1, 1, tzinfo=UTC)
        self._now = now or default
        self._horizon = horizon or default
        self._as_of = as_of or default

    def get_field(self, name: str) -> Any:
        return self._fields.get(name)

    def get_records(self, collection: str) -> list[dict[str, Any]]:
        if self._records is not None:
            # Reconstructed content: exactly the records the ledger recorded, in the
            # order it recorded them. A collection nothing appended to is empty.
            return [dict(r) for r in self._records.get(collection, ())]
        # No content was supplied, so cardinality is all this world knows. The
        # placeholders carry no fabricated content — and no readable content either.
        return [{} for _ in range(self._counts.get(collection, 0))]

    def get_events(self, event_type: str) -> list[dict[str, Any]]:
        if self._events is not None:
            return [dict(r) for r in self._events.get(event_type, ())]
        return [{} for _ in range(self._counts.get(event_type, 0))]

    def get_resource(self, resource_id: str, holder: str) -> float:
        return float(self._resources.get((resource_id, holder), 0.0))

    def get_document_field(self, document_id: str, field_name: str) -> Any:
        return (self._documents.get(document_id) or {}).get(field_name)

    def get_stage(self) -> str:
        return self._stage

    def get_now(self) -> datetime:
        return self._now

    def get_horizon(self) -> datetime:
        return self._horizon

    def get_as_of(self) -> datetime:
        return self._as_of


def terminal_ast_from_world(world: Any) -> dict[str, Any] | None:
    """The executable terminal, when the run's executable world is available.

    Accepts the persisted ``compiled_world.json`` dict (JSON expression trees) or a
    live spec whose ``terminal`` carries :class:`~sworldmodel.worldspec.Expr` nodes;
    ``parse_expr`` consumes both identically.
    """

    if world is None:
        return None
    if isinstance(world, Mapping):
        spec = world.get("world_spec")
        if isinstance(spec, Mapping) and isinstance(spec.get("terminal"), Mapping):
            return dict(spec["terminal"])
        return None
    terminal = getattr(world, "terminal", None)
    if terminal is None:
        return None
    return {
        "yes_when": getattr(terminal, "yes_when", None),
        "unresolved_when": getattr(terminal, "unresolved_when", None),
        "description": getattr(terminal, "description", ""),
    }


_CALL = re.compile(r"^(\w+)\((.*)\)$", re.DOTALL)


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _eval_rendered(
    expr: str,
    fields: dict[str, Any],
    counts: dict[str, int],
    records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    events: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    item: Mapping[str, Any] | None = None,
) -> Any:
    """Evaluate the RENDERED terminal string a pre-compiled-world run recorded.

    Deliberately tiny and total: it supports exactly the forms these runs used, and
    raises :class:`Unreconstructable` for anything else. Guessing at an unsupported
    operator would manufacture the very confidence the replay exists to check.

    ``count``/``event_count`` may carry a trailing where-predicate over ``item(<attr>)``,
    which this evaluated by silently DROPPING it and returning the raw cardinality —
    reporting five recorded positions as five holds. It now evaluates the predicate
    against the reconstructed content, and refuses when no content was supplied.
    """

    def sub(text: str, current: Mapping[str, Any] | None = None) -> Any:
        return _eval_rendered(
            text, fields, counts, records, events, item if current is None else current
        )

    expr = expr.strip()
    if expr in ("True", "False"):
        return expr == "True"
    if expr == "None":
        return None
    m = _CALL.match(expr)
    if not m:
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        try:
            return float(expr) if "." in expr else int(expr)
        except ValueError:
            raise Unreconstructable(f"unparsable terminal atom: {expr!r}") from None
    op, arg_text = m.group(1), m.group(2)
    args = _split_args(arg_text)
    if op == "field":
        return fields.get(str(sub(args[0])))
    if op == "item":
        return dict(item or {}).get(str(sub(args[0])))
    if op in ("count", "event_count"):
        name = str(sub(args[0]))
        if len(args) == 1:
            return counts.get(name, 0)
        pool = records if op == "count" else events
        if pool is None:
            raise Unreconstructable(
                f"{op}({name!r}, <where>) reads record content and none was "
                "reconstructed for this replay"
            )
        return sum(1 for r in pool.get(name, ()) if bool(sub(args[1], r)))
    vals = [sub(a) for a in args]
    if op == "equals":
        return vals[0] == vals[1]
    if op == "not":
        return not vals[0]
    if op == "and":
        return all(bool(v) for v in vals)
    if op == "or":
        return any(bool(v) for v in vals)
    if op in ("greater_than", "greater_or_equal", "less_than", "less_or_equal"):
        a, b = vals[0], vals[1]
        if a is None or b is None:
            return False
        a, b = float(a), float(b)
        return {
            "greater_than": a > b,
            "greater_or_equal": a >= b,
            "less_than": a < b,
            "less_or_equal": a <= b,
        }[op]
    if op in ("multiply", "add", "subtract"):
        nums = [0.0 if v is None else float(v) for v in vals]
        if op == "multiply":
            out = 1.0
            for n in nums:
                out *= n
            return out
        if op == "add":
            return sum(nums)
        return nums[0] - nums[1]
    raise Unreconstructable(f"unsupported terminal operator {op!r}")


def _eval_ast(
    node: Any,
    fields: dict[str, Any],
    counts: dict[str, int],
    records: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    events: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> Any:
    """Evaluate the executable terminal AST with the engine's own evaluator."""

    return evaluate(parse_expr(node), ReplayWorld(fields, counts, records=records, events=events))


def reevaluate_terminal(
    terminal: Mapping[str, Any] | None,
    rendered: Mapping[str, Any],
    fields: dict[str, Any],
    counts: dict[str, int],
    *,
    records: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    events: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[bool, str | None, str]:
    """(resolved, outcome, how) for one REPLAYED branch state. Never guesses.

    This is re-evaluation of the *recorded* terminal over replayed ledger state — it
    prefers the run's own executable AST (run through the engine's evaluator via
    :class:`ReplayWorld`) and falls back to the declared-support rendered-string
    grammar for runs that persisted no executable world. The live terminal evaluator
    stays in :mod:`sworldmodel.engine`; this one exists so a reviewer can re-run the
    recorded question without the engine or a model.

    ``records`` / ``events`` are the reconstructed content from :func:`replay`. They
    are what a content-predicated terminal reads; without them such a terminal
    evaluates over blanks and answers NO (FD-42), so pass the whole
    :class:`ReplayState`.
    """

    if terminal is not None:
        how = "executable AST (compiled_world.json)"
        try:
            yes = bool(_eval_ast(terminal.get("yes_when"), fields, counts, records, events))
            unres_node = terminal.get("unresolved_when")
            unres = (
                bool(_eval_ast(unres_node, fields, counts, records, events))
                if unres_node is not None
                else False
            )
        except UndeterminedExpressionError:
            # The terminal reads something this replayed state never determined. The
            # engine reports that as an honest unresolved outcome, and the replay must
            # say exactly what the engine would say — not a NO, and not a crash that
            # destroys a completed run's trace write.
            return False, None, how
    else:
        yes_src = str(rendered.get("yes_when") or "")
        unres_src = str(rendered.get("unresolved_when") or "False")
        yes = bool(_eval_rendered(yes_src, fields, counts, records, events))
        unres = bool(_eval_rendered(unres_src, fields, counts, records, events))
        how = "rendered terminal string (world_manifest.json) — no executable world persisted"
    if unres:
        return False, None, how
    return True, ("YES" if yes else "NO"), how


_RENDERED_FIELD = re.compile(r"field\('([^']*)'\)")


def _walk_field_reads(expr: Any, out: set[str]) -> None:
    if not isinstance(expr, Expr):
        return
    if expr.op == "field" and expr.args:
        arg = expr.args[0]
        if isinstance(arg, Expr) and arg.op == "const" and arg.args:
            arg = arg.args[0]
        if isinstance(arg, str):
            out.add(arg)
    for a in expr.args:
        _walk_field_reads(a, out)


def terminal_field_reads(
    terminal: Mapping[str, Any] | None, rendered: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Every field name the terminal condition reads, derived from the spec itself.

    Walks the executable AST when the run has one; otherwise reads the rendered
    terminal strings the run recorded. A run that recorded neither yields () — the
    diffs then say nothing about terminal relevance rather than guessing it.
    """

    out: set[str] = set()
    if terminal is not None:
        for key in ("yes_when", "unresolved_when"):
            node = terminal.get(key)
            if node is None:
                continue
            try:
                _walk_field_reads(parse_expr(node), out)
            except ValueError:
                continue
        return tuple(sorted(out))
    for key in ("yes_when", "unresolved_when"):
        text = str((rendered or {}).get(key) or "")
        out |= set(_RENDERED_FIELD.findall(text))
    return tuple(sorted(out))


# --------------------------------------------------------------------------- #
# Derived views: state diffs, communications, process transitions, timeline
# --------------------------------------------------------------------------- #


def derive_state_diffs(
    events: Sequence[EventLike],
    *,
    initial_by_branch: Mapping[str, Mapping[str, Any]] | None = None,
    terminal_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Ordered minimal state diffs, replayed from the ledger — branch by branch.

    ``initial_by_branch`` maps joined branch keys to that branch's initial field
    state; the first diff's ``state_before`` is exactly that state, and applying every
    ``diff`` in order onto it reconstructs the branch's final fields exactly. Events
    that change no field produce no diff.
    """

    initial_by_branch = initial_by_branch or {}
    terminal_set = set(terminal_fields)
    diffs: list[dict[str, Any]] = []
    grouped = group_events_by_branch(events)
    for key in sorted(grouped):
        fields: dict[str, Any] = dict(initial_by_branch.get(key, {}))
        for e in grouped[key]:
            before = dict(fields)
            _apply_field_op(fields, e)
            after = dict(fields)
            if before == after:
                continue
            changed = {k for k in after if before.get(k) != after.get(k)}
            actor = event_actor_id(e)
            diffs.append(
                {
                    "branch_id": event_branch_id(e),
                    "branch_key": key,
                    "event_id": event_id_of(e),
                    "simulation_time": event_time(e),
                    "triggered_by": actor or f"process:{event_kind(e)}",
                    "state_before": before,
                    "state_after": after,
                    "diff": {k: {"from": before.get(k), "to": after.get(k)} for k in changed},
                    "runtime_operation": event_kind(e),
                    "evidence_claim_ids": event_evidence_ids(e),
                    "terminal_read_fields_affected": sorted(changed & terminal_set),
                }
            )
    return diffs


def _decision_branch_key(d: Any) -> str:
    return branch_key(str(record_get(d, "branch_id") or ""))


def extract_communications(
    events: Sequence[EventLike], decisions: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    """Every send/deliver/create_event, with the honest delivery→notice join.

    Delivery and notice are joined against the recorded actor decisions: a decision
    whose ``noticed_observation_ids`` contains this event's id noticed it; one whose
    ``delivered_observation_ids`` contains it had it delivered without noticing it. A
    communication that no recorded decision ever consumed is ``noticed="unknown"`` —
    the run did not record whether anyone took it in, and the replay never guesses.
    """

    comms: list[dict[str, Any]] = []
    for e in events:
        kind = event_kind(e)
        if kind not in COMMUNICATION_EVENT_KINDS:
            continue
        payload = event_payload(e)
        eid = event_id_of(e)
        key = branch_key(event_branch_id(e))

        deliveries: dict[str, dict[str, Any]] = {}
        for d in decisions:
            if _decision_branch_key(d) != key:
                continue
            actor = str(record_get(d, "actor_id") or "")
            if not actor:
                continue
            noticed_ids = [str(i) for i in (record_get(d, "noticed_observation_ids") or [])]
            delivered_ids = [str(i) for i in (record_get(d, "delivered_observation_ids") or [])]
            if eid and eid in noticed_ids:
                entry = deliveries.setdefault(actor, {"actor_id": actor})
                entry["noticed"] = True
                entry.setdefault("noticed_at", record_get(d, "branch_time"))
            elif eid and eid in delivered_ids:
                deliveries.setdefault(actor, {"actor_id": actor}).setdefault("noticed", False)
        for entry in deliveries.values():
            entry.setdefault("noticed_at", None)
        noticed: bool | str
        if not deliveries:
            noticed = "unknown"
        else:
            noticed = any(bool(v.get("noticed")) for v in deliveries.values())

        comms.append(
            {
                "branch_id": event_branch_id(e),
                "branch_key": key,
                "kind": kind,
                "time": event_time(e),
                "sender": event_actor_id(e),
                "recipients": payload.get("to") or payload.get("participants") or [],
                "visibility": event_visibility(e),
                "channel": payload.get("channel"),
                "content": payload.get("information_created")
                or payload.get("text")
                or payload.get("event_type")
                or payload.get("description"),
                "event_id": eid,
                "deliveries": sorted(deliveries.values(), key=lambda v: str(v["actor_id"])),
                "noticed": noticed,
            }
        )
    return comms


def extract_process_transitions(events: Sequence[EventLike]) -> list[dict[str, Any]]:
    """Every non-actor event as a process transition: inputs, outputs, evidence."""

    procs: list[dict[str, Any]] = []
    for e in events:
        if event_actor_id(e):
            continue
        payload = event_payload(e)
        procs.append(
            {
                "branch_id": event_branch_id(e),
                "branch_key": branch_key(event_branch_id(e)),
                "time": event_time(e),
                "operation": event_kind(e),
                "inputs": payload.get("fields") or payload.get("branch_conditions") or {},
                "outputs": {
                    k: payload.get(k) for k in ("field", "value", "event_type") if k in payload
                },
                "event_id": event_id_of(e),
                "evidence_claim_ids": event_evidence_ids(e),
            }
        )
    return procs


def build_timeline(events: Sequence[EventLike], decisions: Sequence[Any]) -> list[dict[str, Any]]:
    """The chronological timeline: every event and actor invocation, in branch order."""

    timeline: list[dict[str, Any]] = []
    seq = 0
    for e in sorted(events, key=lambda e: (event_branch_id(e), event_time(e))):
        seq += 1
        payload = event_payload(e)
        timeline.append(
            {
                "seq": seq,
                "branch_id": event_branch_id(e),
                "simulation_time": event_time(e),
                "stage": "simulation",
                "occurrence": event_kind(e),
                "actor_id": event_actor_id(e),
                "event_id": event_id_of(e),
                "detail": {
                    k: payload.get(k)
                    for k in ("field", "value", "event_type", "collection", "fields", "to")
                    if k in payload
                },
                "evidence_claim_ids": event_evidence_ids(e),
            }
        )
    for d in decisions:
        seq += 1
        intent = record_get(d, "intent") or {}
        timeline.append(
            {
                "seq": seq,
                "branch_id": record_get(d, "branch_id"),
                "simulation_time": record_get(d, "branch_time"),
                "stage": "actor_invocation",
                "occurrence": f"actor_awakened:{record_get(d, 'wake_reason')}",
                "actor_id": record_get(d, "actor_id"),
                "detail": {
                    "intent_mode": record_get(intent, "mode"),
                    "action_id": record_get(intent, "action_id"),
                    "validation_status": record_get(d, "validation_status"),
                },
            }
        )
    timeline.sort(key=lambda r: (str(r["branch_id"]), str(r["simulation_time"]), int(r["seq"])))
    return timeline


# --------------------------------------------------------------------------- #
# Keep-predicate counterfactual replay
# --------------------------------------------------------------------------- #


def counterfactual_outcome(
    events: Sequence[EventLike],
    keep: KeepPredicate,
    *,
    initial: Mapping[str, Any],
    terminal: Mapping[str, Any] | None,
    rendered: Mapping[str, Any],
) -> str | None:
    """The branch's terminal answer with only the ``keep``-selected events replayed.

    Returns ``"YES"`` / ``"NO"``, ``"UNRESOLVED"`` when the counterfactual state does
    not determine the answer, or ``"UNRECONSTRUCTABLE"`` when the recorded terminal
    cannot be re-evaluated. Deterministic replay of the recorded ledger; no
    simulation, no model call.
    """

    state = replay(events, keep)
    fields = {**dict(initial), **state.fields}
    try:
        resolved, outcome, _ = reevaluate_terminal(
            terminal,
            rendered,
            fields,
            state.counts,
            records=state.records,
            events=state.events,
        )
    except Unreconstructable:
        return "UNRECONSTRUCTABLE"
    return outcome if resolved else "UNRESOLVED"


# --------------------------------------------------------------------------- #
# Full run reconstruction
# --------------------------------------------------------------------------- #


@dataclass
class RunRecord:
    """One completed run, as the replay core reads it — from disk or live objects.

    ``events`` / ``decisions`` / ``calls`` may be artifact dicts or the live runtime
    objects; the accessors are total over both. ``world`` is the executable world (the
    persisted ``compiled_world.json`` dict, the live ``executed_compilation`` dict, or
    a live ``WorldSpec``); ``None`` when the run persisted none.
    """

    label: str
    run_dir: str
    forecast: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    stamp: dict[str, Any] = field(default_factory=dict)
    events: list[EventLike] = field(default_factory=list)
    decisions: list[Any] = field(default_factory=list)
    calls: list[Any] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    world: Any = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_run_record(run: Path, label: str) -> RunRecord:
    """Read a completed run's own artifacts into a :class:`RunRecord`."""

    return RunRecord(
        label=label,
        run_dir=str(run),
        forecast=_read_json(run / "forecast.json", {}) or {},
        manifest=_read_json(run / "world_manifest.json", {}) or {},
        schedule=_read_json(run / "branch_schedule.json", {}) or {},
        stamp=_read_json(run / "run_stamp.json", {}) or {},
        events=_read_jsonl(run / "event_ledger.jsonl"),
        decisions=_read_jsonl(run / "actor_decisions.jsonl"),
        calls=_read_jsonl(run / "llm_calls.jsonl"),
        audit=_read_json(run / "trajectory_audit.json", {}) or {},
        review=_read_json(run / "world_review.json", {}) or {},
        world=_read_json(run / "compiled_world.json"),
    )


def reconstruct_run(record: RunRecord) -> dict[str, Any]:
    """Rebuild the published result from the run's own record, and say what it rested on.

    Replays the ledger, re-evaluates every branch terminal, re-derives the published
    probability and compares it, runs the deletion counterfactuals, and classifies
    what produced the number. A disagreement about the ANSWER makes the run
    FORENSICALLY_INVALID; everything else is reported beside it. Computed, never
    asserted; no model call anywhere.
    """

    forecast = record.forecast
    manifest = record.manifest
    schedule = record.schedule
    events = record.events
    decisions = record.decisions
    calls = record.calls
    terminal_ast = terminal_ast_from_world(record.world)
    world_initials = initial_fields_from_world(record.world)
    rendered_obj = manifest.get("terminal") or {}
    rendered: dict[str, Any] = dict(rendered_obj) if isinstance(rendered_obj, Mapping) else {}

    by_branch = group_events_by_branch(events)

    branches = forecast.get("branches") or []
    per_branch: list[dict[str, Any]] = []
    notes: list[str] = []

    for b in branches:
        bid = str(b["branch_id"])
        key = branch_key(bid)
        evs = by_branch.get(key, [])
        actor_events = [e for e in evs if event_actor_id(e)]
        process_events = [e for e in evs if not event_actor_id(e)]
        initial = replay_initial_fields(world_initials, recorded_final_fields(b), evs)
        carried_in = sorted(set(initial) - fields_written_by_events(evs))

        def state(
            keep: KeepPredicate,
            evs: list[EventLike] = evs,
            initial: dict[str, Any] = initial,
        ) -> ReplayState:
            # Loop variables bound as defaults: a closure capturing `evs` by
            # reference evaluates against whichever branch the loop reached LAST if
            # ever called late. Called-in-iteration today, but a replay may not
            # depend on that.
            replayed = replay(evs, keep)
            return ReplayState(
                {**initial, **replayed.fields},
                replayed.counts,
                replayed.records,
                replayed.events,
            )

        full = state(lambda e: True)
        try:
            resolved, outcome, how = reevaluate_terminal(
                terminal_ast,
                rendered,
                full.fields,
                full.counts,
                records=full.records,
                events=full.events,
            )
        except Unreconstructable as exc:
            notes.append(f"{bid}: {exc}")
            resolved, outcome, how = False, None, f"UNRECONSTRUCTABLE: {exc}"

        entry: dict[str, Any] = {
            "branch_id": bid,
            "weight": b.get("weight"),
            "conditions": b.get("conditions"),
            "published_outcome": b.get("outcome"),
            "published_resolved": b.get("resolved"),
            "recomputed_outcome": outcome,
            "recomputed_resolved": resolved,
            "terminal_evaluated_via": how,
            "event_count": len(evs),
            "actor_event_count": len(actor_events),
            "process_event_count": len(process_events),
            "actor_invocations": (schedule.get(bid, {}) or {}).get("actor_invocations", {}),
            "final_fields": full.fields,
            "final_event_type_counts": full.counts,
            # What the collections actually CONTAIN, not merely how many entries
            # they have. A terminal predicated on record values is decided by this,
            # and a reviewer who cannot see it cannot check the answer (FD-42).
            "final_records": full.records,
            "initial_fields": initial,
            "fields_carried_in_never_written_by_any_event": carried_in,
            "matches_published": (outcome == b.get("outcome")) if resolved else None,
        }

        # -- counterfactual replays ------------------------------------------
        cf: dict[str, Any] = {}

        def outcome_of(
            keep: KeepPredicate,
            evs: list[EventLike] = evs,
            initial: dict[str, Any] = initial,
        ) -> str | None:
            return counterfactual_outcome(
                evs, keep, initial=initial, terminal=terminal_ast, rendered=rendered
            )

        cf["all_actor_output_removed"] = outcome_of(lambda e: not event_actor_id(e))
        cf["all_process_output_removed"] = outcome_of(lambda e: bool(event_actor_id(e)))
        cf["everything_removed"] = outcome_of(lambda e: False)
        actors = sorted({a for e in actor_events if (a := event_actor_id(e))})
        cf["per_actor_removed"] = {
            a: outcome_of(lambda e, a=a: event_actor_id(e) != a)  # type: ignore[misc]
            for a in actors
        }
        kinds = sorted({event_kind(e) for e in process_events})
        cf["per_process_kind_removed"] = {
            k: outcome_of(
                lambda e, k=k: bool(event_actor_id(e)) or event_kind(e) != k  # type: ignore[misc]
            )
            for k in kinds
        }
        cf["terminal_relevant_actions_removed"] = outcome_of(
            lambda e: event_kind(e) not in ("create_event", "append_record", "set_field")
        )
        entry["counterfactuals"] = cf
        per_branch.append(entry)

    # -- probability reconstruction -----------------------------------------
    yes = sum(float(b["weight"]) for b in per_branch if b["recomputed_outcome"] == "YES")
    no = sum(float(b["weight"]) for b in per_branch if b["recomputed_outcome"] == "NO")
    unres = sum(
        float(b["weight"])
        for b in per_branch
        if not b["recomputed_resolved"] or b["recomputed_outcome"] is None
    )
    resolved_mass = yes + no
    recomputed_p = (yes / resolved_mass) if resolved_mass > 1e-12 else None
    published_p = forecast.get("simulation_probability")

    terms = [
        f"{float(b['weight']):.4f}x{1 if b['recomputed_outcome'] == 'YES' else 0}"
        for b in per_branch
    ]
    substitution = (
        f"({' + '.join(terms)}) / {resolved_mass:.4f} = {yes:.4f} / {resolved_mass:.4f} = "
        + (f"{recomputed_p:.6f}" if recomputed_p is not None else "undefined")
    )

    weights = [float(b["weight"]) for b in per_branch]
    uniform = len({round(w, 9) for w in weights}) == 1 and len(weights) > 1
    perfect_fraction = uniform and recomputed_p is not None

    # weight provenance, from the run's own uncertainty record
    uncertainties = manifest.get("uncertainty") or []
    provenances = sorted(
        {
            str(o.get("provenance"))
            for u in uncertainties
            for o in (u.get("outcomes") or [])
            if o.get("provenance")
        }
    )
    all_ungrounded = bool(provenances) and all(
        p in ("symmetric_ignorance_assumption", "sensitivity_only") for p in provenances
    )

    # -- responsibility classification --------------------------------------
    actor_calls_total = len(decisions)
    any_actor_changed = any(
        b["counterfactuals"]["all_actor_output_removed"] != b["recomputed_outcome"]
        for b in per_branch
    )
    any_process_changed = any(
        b["counterfactuals"]["all_process_output_removed"] != b["recomputed_outcome"]
        for b in per_branch
    )
    pre_resolved_all_yes = all(
        b["counterfactuals"]["everything_removed"] == "YES" for b in per_branch
    )

    # Is each branch's answer already fixed by its own condition tuple? If no two
    # branches share a tuple and none disagrees, the answer is decided the moment the
    # branch is chosen, and the probability is just the weight of the winning cells —
    # whatever happened in between.
    tuples: dict[str, set[str]] = defaultdict(set)
    for b in per_branch:
        tuples[json.dumps(b["conditions"], sort_keys=True)].add(str(b["recomputed_outcome"]))
    answer_is_a_function_of_the_conditions = (
        len(tuples) == len(per_branch)
        and all(len(v) == 1 for v in tuples.values())
        and len({next(iter(v)) for v in tuples.values()}) > 1
    )

    if any(b["matches_published"] is False for b in per_branch) or (
        published_p is not None
        and recomputed_p is not None
        and abs(published_p - recomputed_p) > 1e-9
    ):
        classification = "INVALID_TRACE"
    elif recomputed_p is None:
        classification = "UNRESOLVED"
    elif pre_resolved_all_yes:
        classification = "FACTUALLY_RESOLVED"
    elif not any_actor_changed and not any_process_changed:
        classification = "INITIAL_ASSUMPTIONS_DOMINATED"
    elif uniform and all_ungrounded and answer_is_a_function_of_the_conditions:
        # The trajectory may well have been necessary to produce any YES at all — that
        # is recorded separately — but it did not fix the NUMBER. Equal, ungrounded
        # weights over cells whose answers are decided by their own conditions make the
        # magnitude an enumeration artifact: re-weight the same cells to any
        # evidence-plausible asymmetry and the probability moves with nothing else
        # changing. That is what dominated means here.
        classification = "BRANCH_WEIGHTS_DOMINATED"
    elif any_actor_changed and uniform and all_ungrounded:
        classification = "PARTLY_TRAJECTORY_CAUSED"
    elif any_actor_changed or any_process_changed:
        classification = "TRAJECTORY_CAUSED"
    else:
        classification = "INITIAL_ASSUMPTIONS_DOMINATED"

    tokens_in = sum(int(record_get(c, "tokens_in") or 0) for c in calls)
    tokens_out = sum(int(record_get(c, "tokens_out") or 0) for c in calls)
    est_cost = tokens_in / 1e6 * RATE_IN_PER_MTOK + tokens_out / 1e6 * RATE_OUT_PER_MTOK

    mismatches: list[dict[str, Any]] = []

    # A disagreement about the ANSWER invalidates the run. A disagreement about what
    # the run COST is a real reporting defect that leaves the answer standing, and the
    # two must not be graded the same or a stale counter would void a sound forecast.
    result_fields = {
        "simulation_probability",
        "resolved_yes_mass",
        "resolved_no_mass",
        "unresolved_mass",
    }

    def check(name: str, published: Any, recomputed: Any) -> None:
        if published is None:
            return
        ok = (
            abs(float(published) - float(recomputed)) < 1e-6
            if isinstance(published, (int, float)) and isinstance(recomputed, (int, float))
            else bool(published == recomputed)
        )
        if not ok:
            mismatches.append(
                {
                    "field": name,
                    "published": published,
                    "recomputed": recomputed,
                    "severity": "CRITICAL" if name in result_fields else "HIGH",
                    "invalidates_result": name in result_fields,
                    "note": (
                        ""
                        if name in result_fields
                        else "cost counters were snapshotted before the post-run "
                        "trajectory-audit call; fixed at commit 189c88d, so runs "
                        "recorded after it will not show this"
                    ),
                }
            )

    check("simulation_probability", published_p, recomputed_p)
    check("resolved_yes_mass", forecast.get("resolved_yes_mass"), yes)
    check("resolved_no_mass", forecast.get("resolved_no_mass"), no)
    check("unresolved_mass", forecast.get("unresolved_mass"), unres)
    check("model_call_count", forecast.get("model_call_count"), len(calls))
    check("token_usage", forecast.get("token_usage"), tokens_in + tokens_out)

    return {
        "label": record.label,
        "run_dir": record.run_dir,
        "commit": record.stamp.get("commit"),
        "question": forecast.get("question"),
        "published": {
            "probability": published_p,
            "probability_source": forecast.get("probability_source"),
            "status": forecast.get("status"),
            "yes_mass": forecast.get("resolved_yes_mass"),
            "no_mass": forecast.get("resolved_no_mass"),
            "unresolved_mass": forecast.get("unresolved_mass"),
            "lower_bound": forecast.get("lower_bound"),
            "upper_bound": forecast.get("upper_bound"),
            "model_calls": forecast.get("model_call_count"),
            "tokens": forecast.get("token_usage"),
        },
        "recomputed": {
            "probability": recomputed_p,
            "yes_mass": yes,
            "no_mass": no,
            "unresolved_mass": unres,
            "substitution": substitution,
            "model_calls": len(calls),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "estimated_cost_usd": round(est_cost, 4),
            "estimated_cost_note": (
                f"ESTIMATE: recorded tokens x list rate (${RATE_IN_PER_MTOK}/Mtok in, "
                f"${RATE_OUT_PER_MTOK}/Mtok out); the trace records tokens, not price"
            ),
            "events": len(events),
            "actor_decisions": len(decisions),
        },
        "weights": {
            "branch_weights": weights,
            "all_equal": uniform,
            "perfect_fraction_from_branch_weights": perfect_fraction,
            "outcome_provenances": provenances,
            "all_weights_ungrounded": all_ungrounded,
            "flag": (
                "PERFECT_FRACTION_FROM_BRANCH_WEIGHTS"
                if perfect_fraction
                else "no equal-weight artifact"
            ),
        },
        "branches": per_branch,
        "responsibility": {
            "classification": classification,
            "any_actor_output_changed_an_outcome": any_actor_changed,
            "any_process_output_changed_an_outcome": any_process_changed,
            "terminal_already_satisfied_with_everything_removed": pre_resolved_all_yes,
            "actor_invocations_total": actor_calls_total,
        },
        "audit_classification": record.audit.get("classification"),
        "world_review_disposition": record.review.get("disposition"),
        "world_review_blocking": record.review.get("blocking_failures"),
        "mismatches": mismatches,
        "reconstruction_notes": notes,
        "verdict": (
            "FORENSICALLY_INVALID"
            if any(m["invalidates_result"] for m in mismatches) or classification == "INVALID_TRACE"
            else "RECONSTRUCTED"
        ),
    }


# --------------------------------------------------------------------------- #
# The dossier (one openable page per run)
# --------------------------------------------------------------------------- #


FORENSIC_ARTIFACTS_NOTE = (
    "forensic_timeline · llm_calls_full · actor_invocations · communications ·\n"
    "process_transitions · state_diffs · branch_weight_history · semantic_runtime_lineage ·\n"
    "terminal_evaluations · probability_reconstruction · trajectory_responsibility ·\n"
    "forensic_verdict"
)

RUN_ARTIFACTS_NOTE = (
    "branch_initial_state · state_diffs · communications · process_transitions ·\n"
    "event_ledger · actor_decisions · llm_calls · branch_schedule · forecast ·\n"
    "world_manifest · evidence_manifest · compiled_world · trajectory_audit · world_review"
)


def _esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_dossier(
    result: dict[str, Any],
    timeline: list[dict[str, Any]],
    decisions: Sequence[Any],
    calls: Sequence[Any],
    *,
    title_prefix: str = "Forensic dossier",
    artifacts_note: str = FORENSIC_ARTIFACTS_NOTE,
) -> str:
    """One openable page per run: the number, how it was produced, and whether it holds.

    Deliberately self-contained and dependency-free — a dossier that needs a server to
    read is a dossier nobody reads. It renders the reconstruction, not a retelling:
    every figure on the page comes from the artifacts emitted beside it.
    """

    rec, pub, w = result["recomputed"], result["published"], result["weights"]
    resp = result["responsibility"]

    def rows(headers: list[str], data: list[list[Any]]) -> str:
        head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in data)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    branch_rows = [
        [
            b["branch_id"],
            b["weight"],
            json.dumps(b["conditions"]),
            b["published_outcome"],
            b["recomputed_outcome"],
            "yes" if b["matches_published"] else "NO",
            b["counterfactuals"]["all_actor_output_removed"],
            b["counterfactuals"]["everything_removed"],
            b["actor_event_count"],
        ]
        for b in result["branches"]
    ]
    call_rows = [
        [
            record_get(c, "call_number") or i,
            record_get(c, "task_kind"),
            record_get(c, "tokens_in"),
            record_get(c, "tokens_out"),
            record_get(c, "latency_ms", "not recorded"),
            record_get(c, "started_at", "not recorded"),
        ]
        for i, c in enumerate(calls, start=1)
    ]
    decision_rows = [
        [
            str(record_get(d, "branch_id") or "")[:44],
            record_get(d, "branch_time"),
            record_get(d, "actor_id"),
            record_get(d, "wake_reason"),
            record_get(record_get(d, "intent") or {}, "mode"),
            record_get(record_get(d, "intent") or {}, "action_id") or "",
            record_get(d, "validation_status"),
        ]
        for d in decisions
    ]
    mism = [
        [m["field"], m["published"], m["recomputed"], m["severity"], m.get("note", "")]
        for m in result["mismatches"]
    ]

    verdict_class = "ok" if result["verdict"] == "RECONSTRUCTED" else "bad"
    flag = w["flag"]
    return f"""<!doctype html>
<meta charset="utf-8"><title>{_esc(title_prefix)} — {_esc(result["label"])}</title>
<style>
 body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
   margin:0 auto;max-width:1100px;padding:2rem;color:#111}}
 h1{{font-size:1.5rem;margin-bottom:.2rem}} h2{{font-size:1.05rem;margin-top:2rem}}
 .q{{color:#555;margin-top:0}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:12.5px}}
 th,td{{border:1px solid #ddd;padding:.35rem .5rem;text-align:left;vertical-align:top}}
 th{{background:#f6f6f6}}
 .big{{font-size:2rem;font-weight:600}}
 .ok{{color:#0a7d33}} .bad{{color:#b00}} .warn{{color:#a60}}
 code,pre{{background:#f6f6f6;padding:.15rem .3rem;border-radius:3px;
   font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
 pre{{padding:.6rem;overflow-x:auto}}
 .k{{color:#666}}
 @media (prefers-color-scheme:dark){{
   body{{background:#111;color:#eee}} th{{background:#1c1c1c}}
   th,td{{border-color:#333}} code,pre{{background:#1c1c1c}} .k{{color:#999}}
   .ok{{color:#4ade80}} .bad{{color:#f87171}} .warn{{color:#fbbf24}}}}
</style>
<h1>{_esc(title_prefix)} — {_esc(result["label"])}</h1>
<p class="q">{_esc(result["question"])}</p>
<p class="k">run {_esc(result["run_dir"])} · commit {_esc(result["commit"])}</p>

<h2>1. Verdict</h2>
<p class="big {verdict_class}">{_esc(result["verdict"])}</p>
<p>Responsibility: <strong>{_esc(resp["classification"])}</strong>.
Weights: <strong>{_esc(flag)}</strong>.
Published <code>probability_source</code>: <code>{_esc(pub["probability_source"])}</code>.
Post-run audit classification: <code>{_esc(result["audit_classification"])}</code>.</p>

<h2>2. Exact probability reconstruction</h2>
<p>P(YES) = sum(branch_weight x yes_indicator) / sum(resolved branch_weight)</p>
<pre>{_esc(rec["substitution"])}</pre>
<p>published <strong>{_esc(pub["probability"])}</strong> ·
recomputed <strong>{_esc(rec["probability"])}</strong> ·
bounds [{_esc(pub["lower_bound"])}, {_esc(pub["upper_bound"])}]</p>

<h2>3. Branches, weights and counterfactuals</h2>
{
        rows(
            [
                "branch",
                "weight",
                "conditions",
                "published",
                "recomputed",
                "match",
                "actors removed",
                "everything removed",
                "actor events",
            ],
            branch_rows,
        )
    }
<p class="k">Weight provenance: {_esc(", ".join(w["outcome_provenances"]) or "none recorded")}
· all weights ungrounded: {_esc(w["all_weights_ungrounded"])}
· all weights equal: {_esc(w["all_equal"])}</p>

<h2>4. Did the trajectory matter?</h2>
{rows(["signal", "value"], [[k, v] for k, v in resp.items()])}

<h2>5. Actor invocations ({len(decisions)})</h2>
{
        rows(
            ["branch", "sim time", "actor", "woken by", "intent", "action", "validation"],
            decision_rows,
        )
        if decision_rows
        else "<p>None. No actor was invoked in this run.</p>"
    }

<h2>6. Provider calls ({len(calls)})</h2>
{rows(["#", "task", "tokens in", "tokens out", "latency ms", "started"], call_rows)}
<p class="k">Estimated cost {_esc(rec["estimated_cost_usd"])} USD —
{_esc(rec["estimated_cost_note"])}</p>

<h2>7. Recomputation vs published</h2>
{
        rows(["field", "published", "recomputed", "severity", "note"], mism)
        if mism
        else '<p class="ok">Every published figure recomputes exactly.</p>'
    }

<h2>8. Timeline ({len(timeline)} entries)</h2>
{
        rows(
            ["#", "branch", "sim time", "stage", "occurrence", "actor"],
            [
                [
                    t["seq"],
                    str(t["branch_id"])[:40],
                    t["simulation_time"],
                    t["stage"],
                    t["occurrence"],
                    t.get("actor_id") or "",
                ]
                for t in timeline
            ],
        )
    }

<h2>9. Artifacts beside this page</h2>
<p class="k">{artifacts_note}</p>
"""
