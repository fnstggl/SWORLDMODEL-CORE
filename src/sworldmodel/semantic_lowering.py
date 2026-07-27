"""Deterministic lowering: semantic plan in, executable WorldSpec JSON out.

Two passes, no model calls, no randomness. Pass 1 walks the plan and mints every
runtime symbol — entity ids, field ids, action ids, node ids, event types, record
collections, authority tokens — into a symbol table keyed by (namespace, semantic
name). Pass 2 resolves every reference through that table and emits the exact JSON
shape the direct compiler's model emits, so `assemble_bundle`, every compile gate and
the runtime are byte-for-byte the same executor for both compiler modes.

The model never authors an identifier and never chooses between `field` and
`field_id`: those keys exist only below this line. A meaning this layer cannot
represent is a LOWERING_GAP — named, recorded and refused — never approximated and
never silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import WorldIntegrityError
from .semantic_plan import (
    UNKNOWN,
    Period,
    SemanticChange,
    SemanticPlan,
    SemanticProcess,
    SemanticState,
    SemanticValue,
    TerminalQuery,
    duration_scalars,
    flow_periods,
    terminal_relevant_states,
)


class LoweringGap(WorldIntegrityError):
    """A semantic construct the universal change mapping cannot yet represent.

    Raised instead of approximating, substituting a nearby operation, or silently
    discarding meaning. Carries everything PART 9 requires so the gap can be judged:
    what construct, why existing changes cannot express it, whether it composes from
    current primitives, and the smallest genuinely universal capability that is
    missing.

    ``must_refuse`` is the revision policy: True (the default) means a differently
    phrased plan might avoid the gap, so the compiler may spend its one revision round
    before refusing. False means no rephrasing can help — an unresolved reference is a
    defect of the validator, not of the plan's wording — so the compiler refuses
    immediately instead of burning a model call reproducing the same failure.
    """

    def __init__(
        self,
        construct: str,
        *,
        why: str,
        composable: bool,
        smallest_missing: str,
        must_refuse: bool = True,
    ) -> None:
        self.must_refuse = must_refuse
        super().__init__(
            f"LOWERING_GAP: {construct} — {why}",
            details={
                "failure": "lowering_gap",
                "recompilable": True,
                "unsupported construct": construct,
                "why existing changes cannot represent it": why,
                "composable from current primitives": composable,
                "smallest missing universal capability": smallest_missing,
                "must refuse": must_refuse,
            },
        )


# ---------------------------------------------------------------------------
# Pass 1 — symbol construction
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "x"


@dataclass
class SymbolTable:
    """Every runtime identifier, minted deterministically from semantic names.

    Within one namespace, first-come order (which is plan order) resolves slug
    collisions with a numeric suffix — so the same plan always lowers to the same
    symbols, and two objects can never share an id.
    """

    by_key: dict[tuple[str, str], str] = field(default_factory=dict)
    taken: dict[str, set[str]] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def mint(self, namespace: str, name: str, *, rule: str, evidence: tuple[str, ...]) -> str:
        key = (namespace, name)
        if key in self.by_key:
            return self.by_key[key]
        base = _slug(name)
        used = self.taken.setdefault(namespace, set())
        candidate = base
        n = 2
        while candidate in used:
            candidate = f"{base}_{n}"
            n += 1
        used.add(candidate)
        self.by_key[key] = candidate
        self.records.append(
            {
                "semantic": name,
                "namespace": namespace,
                "runtime_id": candidate,
                "lowering_rule": rule,
                "evidence_claim_ids": list(evidence),
            }
        )
        return candidate

    def resolve(self, namespace: str, name: str) -> str:
        key = (namespace, name)
        if key not in self.by_key:
            raise LoweringGap(
                f"reference to {namespace} {name!r}",
                why="the reference does not resolve to a declared semantic object; the "
                "validator should have refused this plan before lowering",
                composable=False,
                smallest_missing="nothing — this is an unresolved reference, not a "
                "missing capability",
                must_refuse=False,
            )
        return self.by_key[key]


def build_symbols(plan: SemanticPlan) -> SymbolTable:
    t = SymbolTable()
    changes = [c for a in plan.affordances for c in a.changes] + [
        c for p in plan.processes for o in p.occurrences for c in o.changes
    ]
    drawn = {c.target for c in changes if c.op == "decrease"}
    filled = {c.target for c in changes if c.op == "increase"}
    for e in plan.entities:
        t.mint(
            "entity", e.name, rule="entity → entities[].entity_id", evidence=e.evidence_claim_ids
        )
    for s in plan.states:
        t.mint("field", s.name, rule="state → fields[].field_id", evidence=s.evidence_claim_ids)
        if s.kind != "stock":
            continue
        # What a bounded stock could not absorb is causal information, not an error: the
        # draw a depleted stock could not pay for IS the backlog, the unmet demand, the
        # queue that grows. It gets its own field so downstream conditions, the terminal,
        # the state diffs and the replay core can all read it — never a log line.
        if s.name in drawn:
            t.mint(
                "field",
                f"unmet draw on {s.name}",
                rule="stock floor → fields[].field_id recording the quantity a draw could "
                "not take (the shortfall: unmet demand, backlog, queue)",
                evidence=s.evidence_claim_ids,
            )
        if s.capacity is not None and s.name in filled:
            t.mint(
                "field",
                f"overflow above {s.name}",
                rule="stock capacity → fields[].field_id recording the quantity offered "
                "above the ceiling (the spill)",
                evidence=s.evidence_claim_ids,
            )
    for ev in plan.events:
        t.mint(
            "event",
            ev.name,
            rule="event → create_event.event_type + append_record.collection",
            evidence=ev.evidence_claim_ids,
        )
    for a in plan.affordances:
        t.mint(
            "action", a.name, rule="affordance → actions[].action_id", evidence=a.evidence_claim_ids
        )
        t.mint(
            "authority",
            a.name,
            rule="affordance → required_authority token, granted to its actor",
            evidence=(),
        )
    node_processes = _node_process_names(plan)
    for p in plan.processes:
        ns = "node" if p.name in node_processes else "external"
        t.mint(
            ns,
            p.name,
            rule=f"process ({p.kind}) → "
            + ("process.nodes[].node_id" if ns == "node" else "external_processes[].process_id"),
            evidence=p.evidence_claim_ids,
        )
        if ns == "node" and p.kind != "actor_moment":
            # Every occurrence beyond the first is its own node, and its id is minted
            # HERE — through the same table, against the same taken-set — never by
            # string concatenation at emission time, where it could collide with a
            # legitimately suffixed sibling and leave two objects sharing one id.
            for j in range(1, len(p.occurrences)):
                t.mint(
                    "node",
                    f"{p.name} occurrence {j + 1}",
                    rule="process occurrence → process.nodes[].node_id",
                    evidence=p.evidence_claim_ids,
                )
    return t


def _node_process_names(plan: SemanticPlan) -> set[str]:
    """Which processes must lower to process nodes rather than external processes.

    A process needs nodes when it is an actor_moment, when any of its own occurrences
    chains on another process, or when ANY other occurrence in the plan chains on it —
    only nodes have ids that ``next_nodes``/``after_node`` can name, so a dated-only
    process that something follows must still be a node. Judged as a closure over the
    whole plan, not per process, so the two passes can never disagree about a
    reference's namespace.
    """

    process_names = {p.name for p in plan.processes}
    names = {p.name for p in plan.processes if p.kind == "actor_moment"}
    names |= {
        p.name for p in plan.processes if any(o.after_process is not None for o in p.occurrences)
    }
    names |= {
        o.after_process
        for p in plan.processes
        for o in p.occurrences
        if o.after_process is not None
    }
    return names & process_names


# ---------------------------------------------------------------------------
# Pass 2 — reference resolution and executable generation
# ---------------------------------------------------------------------------


def _lower_value(v: SemanticValue, t: SymbolTable, flows: dict[str, Period]) -> Any:
    """A semantic value → a runtime expression, with rate × duration made arithmetic.

    A duration is not a number until it meets the rate it multiplies: "P1W" against a
    per-week rate is the scalar 1, against a per-day rate it is 7. Code owns that
    conversion — the planner writes the length of time the firing covers and never a
    conversion factor, which is the whole reason the total can be checked against the
    window instead of against how many dates were typed.
    """

    if v.kind == "literal":
        return v.literal
    if v.kind == "state":
        assert v.state is not None
        return {"op": "field", "args": [t.resolve("field", v.state)]}
    if v.kind == "duration":
        raise LoweringGap(
            f"duration {v.literal!r} standing alone",
            why="a length of time is a quantity of nothing until it multiplies a rate; "
            "the validator should have refused this plan before lowering",
            composable=False,
            smallest_missing="nothing — put the duration in a product with its rate",
            must_refuse=False,
        )
    op = "add" if v.kind == "sum" else "multiply"
    scalars: dict[int, float] = {}
    if v.kind == "product":
        scalars, why = duration_scalars(v.parts, flows)
        if why:
            raise LoweringGap(
                f"product containing a duration: {why}",
                why="the duration cannot be resolved against a single rate; the "
                "validator should have refused this plan before lowering",
                composable=False,
                smallest_missing="nothing — this is an unresolvable duration, not a "
                "missing capability",
                must_refuse=False,
            )
    parts = [
        scalars[i] if i in scalars else _lower_value(p, t, flows) for i, p in enumerate(v.parts)
    ]
    # The runtime's arithmetic is binary; fold left so any arity lowers.
    out = {"op": op, "args": [parts[0], parts[1]]}
    for extra in parts[2:]:
        out = {"op": op, "args": [out, extra]}
    return out


def _negate(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return -value
    return {"op": "multiply", "args": [-1, value]}


@dataclass(frozen=True)
class _Stock:
    """A declared stock and the runtime symbols its bounds are written in."""

    state: SemanticState
    field_id: str
    unmet_id: str  # where a draw the stock could not pay for is recorded
    overflow_id: str  # where quantity offered above the capacity is recorded


def _stock_table(plan: SemanticPlan, t: SymbolTable) -> dict[str, _Stock]:
    """Every stock in the plan, keyed by its semantic name."""

    out: dict[str, _Stock] = {}
    for s in plan.states:
        if s.kind != "stock":
            continue
        out[s.name] = _Stock(
            state=s,
            field_id=t.resolve("field", s.name),
            unmet_id=t.by_key.get(("field", f"unmet draw on {s.name}"), ""),
            overflow_id=t.by_key.get(("field", f"overflow above {s.name}"), ""),
        )
    return out


def _available(stock: _Stock) -> dict[str, Any]:
    """How much of the stock is above its floor right now, never below nothing."""

    return {
        "op": "max",
        "args": [
            0,
            {
                "op": "subtract",
                "args": [
                    {"op": "field", "args": [stock.field_id]},
                    stock.state.conserved_floor,
                ],
            },
        ],
    }


def _room(stock: _Stock) -> dict[str, Any]:
    """How much more the stock can hold before its capacity, never below nothing."""

    assert stock.state.capacity is not None
    return {
        "op": "max",
        "args": [
            0,
            {
                "op": "subtract",
                "args": [stock.state.capacity, {"op": "field", "args": [stock.field_id]}],
            },
        ],
    }


def _lower_stock_move(op: str, demand: Any, stock: _Stock) -> list[dict[str, Any]]:
    """A change to a stock, CLAMPED at its bounds with the remainder recorded.

    The runtime does not conserve anything on its own. ``effects.can_apply`` holds the
    only non-negativity check in the system and it is reached from the actor paths alone
    (``executor.py`` and ``novel.py``); an operational process applies its effects at
    ``engine.py`` with no feasibility check at all, and ``world.py`` subtracts with no
    floor. That is precisely how a live world drove an inventory forty-eight thousand
    units negative: the drawdown was issued by a process, so nothing was ever consulted.

    So the bound is compiled into the write itself, which is the one place every issuer
    passes through — actor, node and external occurrence alike lower through this
    function. And it CLAMPS rather than refuses, because refusing the firing would throw
    away the causal fact: a delivery run that could only half-load did happen, and the
    half it could not load is the backlog. What moves is the demand or what is there,
    whichever is smaller; the remainder is written to its own field, where downstream
    conditions, the terminal, the state diffs and the replay core can all read it as the
    unmet demand it is.
    """

    field_id = stock.field_id
    if op == "decrease":
        moved = {"op": "min", "args": [demand, _available(stock)]}
        out = [{"op": "adjust_field", "field": field_id, "delta": _negate(moved)}]
        if stock.unmet_id:
            out.append(
                {
                    "op": "adjust_field",
                    "field": stock.unmet_id,
                    "delta": {
                        "op": "max",
                        "args": [0, {"op": "subtract", "args": [demand, _available(stock)]}],
                    },
                }
            )
        return out
    if stock.state.capacity is None:
        # Nothing declared a ceiling, so there is nothing to clamp against: an inflow
        # into an unbounded stock is exactly today's adjustment.
        return [{"op": "adjust_field", "field": field_id, "delta": demand}]
    moved = {"op": "min", "args": [demand, _room(stock)]}
    out = [{"op": "adjust_field", "field": field_id, "delta": moved}]
    if stock.overflow_id:
        out.append(
            {
                "op": "adjust_field",
                "field": stock.overflow_id,
                "delta": {
                    "op": "max",
                    "args": [0, {"op": "subtract", "args": [demand, _room(stock)]}],
                },
            }
        )
    return out


def _lower_change(
    c: SemanticChange,
    t: SymbolTable,
    plan: SemanticPlan,
    stocks: dict[str, _Stock],
    flows: dict[str, Period],
) -> list[dict[str, Any]]:
    """One universal semantic change → the existing effect operations."""

    if c.op == "set":
        assert c.value is not None  # validator guarantees
        return [
            {
                "op": "set_field",
                "field": t.resolve("field", c.target),
                "value": _lower_value(c.value, t, flows),
            }
        ]
    if c.op in ("increase", "decrease"):
        assert c.amount is not None
        delta = _lower_value(c.amount, t, flows)
        stock = stocks.get(c.target)
        if stock is not None:
            return _lower_stock_move(c.op, delta, stock)
        source = stocks.get(c.drawn_from) if c.drawn_from else None
        if source is not None and c.op == "increase":
            # The receiving side of one movement: it takes what the source could give,
            # computed from the same pre-firing world the source's own clamp reads, so
            # the two halves can never disagree about how much moved.
            delta = {"op": "min", "args": [delta, _available(source)]}
        return [
            {
                "op": "adjust_field",
                "field": t.resolve("field", c.target),
                "delta": _negate(delta) if c.op == "decrease" else delta,
            }
        ]
    if c.op == "record_event":
        sym = t.resolve("event", c.target)
        ev = next((e for e in plan.events if e.name == c.target), None)
        if ev is None:
            raise LoweringGap(
                f"record_event target {c.target!r}",
                why="the change records an event the plan never declared; the validator "
                "should have refused this plan before lowering",
                composable=False,
                smallest_missing="nothing — this is an unresolved reference",
                must_refuse=False,
            )
        # The event's declared meaning survives whole: its visibility governs who the
        # runtime delivers it to, and its participants and created information ride in
        # the payload — a private briefing must not become a public broadcast because
        # lowering forgot to say otherwise.
        data: dict[str, Any] = {"detail": c.detail}
        create: dict[str, Any] = {
            "op": "create_event",
            "event_type": sym,
            "text": ev.meaning,
            "visibility": ev.visibility,
            "data": data,
        }
        if ev.participants:
            data["participants"] = {role: t.resolve("entity", who) for role, who in ev.participants}
            # The runtime's audience resolver reads only the effect's "to"/"audience"
            # keys — participants riding in the payload are invisible to delivery. A
            # private event whose participants lived only in `data` had an empty
            # audience, so visible_to returned False for everyone and the briefing
            # reached nobody, including its own participants. The participants ARE the
            # audience, so they are emitted where delivery actually looks.
            create["to"] = sorted({t.resolve("entity", who) for _, who in ev.participants})
        if ev.information_created:
            data["information_created"] = ev.information_created
        return [
            create,
            {
                "op": "append_record",
                "collection": sym,
                "key": "$actor",
                "value": c.detail or ev.meaning,
            },
        ]
    if c.op == "send":
        return [
            {
                "op": "deliver_information",
                "text": c.detail or c.target,
                "info_fields": {},
                "to": [t.resolve("entity", r) for r in c.recipients],
            }
        ]
    if c.op == "schedule":
        raise LoweringGap(
            f"change op 'schedule' targeting process {c.target!r}",
            why="dynamic scheduling of a declared process from inside another change is "
            "not yet mapped; scheduling is expressed by the process's own occurrences "
            "(at / after_process), which lower to dated nodes and external occurrences",
            composable=True,
            smallest_missing="a universal 'enqueue occurrence of process P at time T' "
            "runtime binding",
        )
    raise LoweringGap(
        f"change op {c.op!r}",
        why="no universal mapping exists for this operation",
        composable=False,
        smallest_missing=f"a universal runtime meaning for {c.op!r}",
    )


_KIND_BY_TYPE = {
    "person": "person",
    "object": "object",
    # Everything institutional or aggregate is an organization at the entity-kind level;
    # representation_scale carries the finer truth (subunit, population_stratum, ...).
    "organization": "organization",
    "coalition": "organization",
    "institution": "organization",
    "population": "organization",
    "market": "organization",
    "system": "organization",
}

_CMP_OPS = {
    "greater_than": "greater_than",
    "greater_or_equal": "greater_or_equal",
    "less_than": "less_than",
    "less_or_equal": "less_or_equal",
    "equals": "equals",
}


def _aware_iso(value: str | None) -> str | None:
    """Every timestamp the executable carries is timezone-aware, or absent.

    The validator coerces naive datetimes to UTC for its own comparisons, but the
    lowered spec used to carry the planner's naive ISO string verbatim — and a live run
    crashed 354 seconds in when the ENGINE compared that naive time against the aware
    contract clock. Normalized once here, at the only place timestamps enter the
    executable, so no downstream consumer can ever see a naive one.
    """

    from .semantic_plan import _parse_when

    when = _parse_when(value)
    return when.isoformat() if when is not None else None


def _recurrence_keys(p: SemanticProcess) -> dict[str, Any]:
    """The declared cadence, carried onto the compiled process it produced.

    The expansion is what runs; the declaration is what gets audited. A reviewer handed
    a flat list of dated firings cannot tell a cadence declared over a window from a
    calendar somebody typed — which is the FD-25 hole itself — so both travel together
    and a compiled review can check one against the other without the semantic plan.

    ISO strings rather than seconds or counts, because that is what every other timestamp
    in the executable already is and because a period is not always a fixed number of
    seconds (P1M is a calendar step). ``recurrence_firings`` is the count code generated,
    so a reviewer can compare it against its own reading of period and window.
    """

    rec = p.recurrence
    if rec is None or not p.occurrences_generated:
        return {
            "recurrence_period": "",
            "recurrence_start": "",
            "recurrence_end": "",
            "recurrence_firings": 0,
        }
    return {
        "recurrence_period": rec.period,
        "recurrence_start": _aware_iso(rec.start) or "",
        "recurrence_end": _aware_iso(rec.end) or "",
        "recurrence_firings": len(p.occurrences),
    }


def _lookup(table: dict[str, str], key: Any, what: str) -> str:
    """A table miss is a named gap, never a KeyError and never a silent default.

    A live plan with a missing ``state_type`` validated cleanly (the empty string
    slipped the truthiness guard) and then died here as a bare ``KeyError('')`` — no
    diagnosis, no gate, an uncaught traceback. Every fixed-table lookup in this module
    goes through this helper so an unmapped value refuses with its own name.
    """

    if isinstance(key, str) and key in table:
        return table[key]
    raise LoweringGap(
        f"{what} {key!r}",
        why=f"no universal mapping exists for this {what}; the legal values are {sorted(table)}",
        composable=False,
        smallest_missing=f"a universal runtime meaning for {what} {key!r}",
    )


def _lower_terminal(q: TerminalQuery, t: SymbolTable, flows: dict[str, Period]) -> dict[str, Any]:
    if q.form == "all_of":
        return {"op": "and", "args": [_lower_terminal(p, t, flows) for p in q.parts]}
    if q.form == "any_of":
        return {"op": "or", "args": [_lower_terminal(p, t, flows) for p in q.parts]}
    if q.form == "not":
        return {"op": "not", "args": [_lower_terminal(q.parts[0], t, flows)]}
    if q.form == "event_exists":
        assert q.event is not None
        return {
            "op": "greater_than",
            "args": [{"op": "event_count", "args": [t.resolve("event", q.event)]}, 0],
        }
    if q.form == "record_count":
        assert q.record_event is not None and q.comparison is not None
        assert q.threshold is not None
        return {
            "op": _lookup(_CMP_OPS, q.comparison, "comparison"),
            "args": [
                {"op": "count", "args": [t.resolve("event", q.record_event)]},
                _lower_value(q.threshold, t, flows),
            ],
        }
    if q.form == "state_equals":
        assert q.state is not None
        if q.value is None:
            raise LoweringGap(
                f"state_equals over {q.state!r} with no value",
                why="a comparison against a missing value would resolve NO forever (or "
                "YES exactly while the state is unset) — an answer manufactured by an "
                "absent JSON key",
                composable=False,
                smallest_missing="nothing — the terminal must state the value it asks about",
            )
        return {
            "op": "equals",
            "args": [{"op": "field", "args": [t.resolve("field", q.state)]}, q.value],
        }
    if q.form == "quantity_comparison":
        assert q.state is not None and q.comparison is not None and q.threshold is not None
        return {
            "op": _lookup(_CMP_OPS, q.comparison, "comparison"),
            "args": [
                {"op": "field", "args": [t.resolve("field", q.state)]},
                _lower_value(q.threshold, t, flows),
            ],
        }
    raise LoweringGap(
        f"terminal form {q.form!r}",
        why="no deterministic lowering exists for this form",
        composable=False,
        smallest_missing=f"a universal terminal lowering for {q.form!r}",
    )


_VALUE_TYPES = {"quantity": "number", "boolean": "bool", "category": "string", "text": "string"}


def _produced(base: str, p: SemanticProcess) -> str:
    """information_produced is meaning, not metadata: it rides in the description the
    actors, auditors and replay viewer actually read — validated content must lower
    into something a consumer reads, or be recorded as dropped, never vanish."""

    if p.information_produced:
        return f"{base} [produces: {p.information_produced}]"
    return base


def _terminal_counted_events(q: TerminalQuery) -> set[str]:
    """Event names whose records the terminal counts (event_exists / record_count)."""

    out: set[str] = set()
    if q.form == "event_exists" and q.event:
        out.add(q.event)
    if q.form == "record_count" and q.record_event:
        out.add(q.record_event)
    for part in q.parts:
        out |= _terminal_counted_events(part)
    return out


def _wake_rules(plan: SemanticPlan, t: SymbolTable) -> list[dict[str, Any]]:
    """Deterministic wake rules — who is brought back when the world moves.

    The engine consults ``spec.wake_rules`` only after the directed / asked / revisit /
    communication checks, and field writes, data releases and appended records stay
    ambient unless a compiled rule says otherwise. A lowering that emits no rules
    therefore produces a world that is inert between dated actor moments: nothing any
    mechanism does ever wakes a decider. Three universal derivations, every reason
    quoting the plan's own meaning text:

      (a) a state named in a process's ``inputs`` wakes every deciding entity when it
          changes — a mechanism input changing is exactly what its readers react to;
      (b) a declared event wakes its deciding participants (all deciding entities when
          it is public with no participants);
      (c) a record collection the terminal counts wakes every deciding entity — the
          resolving tally moving is material to anyone who can still act.

    A derivation whose wake set is empty (a world with no deciding entities, or an
    event none of whose participants decide) is recorded as a "dropped" mapping entry,
    never silently discarded.
    """

    decider_names = {e.name for e in plan.entities if e.decides}
    deciders = sorted(t.resolve("entity", n) for n in decider_names)
    rules: list[dict[str, Any]] = []

    def emit(
        name: str,
        *,
        wakes: list[str],
        reason: str,
        trigger_key: str,
        trigger: str,
        rule: str,
        evidence: tuple[str, ...],
        dropped_why: str,
    ) -> None:
        if not wakes:
            t.records.append(
                {
                    "semantic": name,
                    "namespace": "dropped",
                    "runtime_id": "",
                    "lowering_rule": f"carried nowhere: {dropped_why}",
                    "evidence_claim_ids": list(evidence),
                }
            )
            return
        rule_id = t.mint("wake_rule", name, rule=rule, evidence=evidence)
        entry: dict[str, Any] = {
            "rule_id": rule_id,
            "wakes": wakes,
            "reason": reason,
            "on_record_in": "",
            "on_field_change": "",
            "on_event_type": "",
            "evidence_claim_ids": list(evidence),
        }
        entry[trigger_key] = trigger
        rules.append(entry)

    for p in plan.processes:
        for inp in p.inputs:
            emit(
                f"wake when {inp} changes (input to {p.name})",
                wakes=deciders,
                reason=f"{inp!r} is an input to {p.name!r}: {p.meaning}",
                trigger_key="on_field_change",
                trigger=t.resolve("field", inp),
                rule="process input → wake_rules[].on_field_change waking every deciding entity",
                evidence=p.evidence_claim_ids,
                dropped_why="no deciding entity exists to wake when this input changes",
            )

    for ev in plan.events:
        participant_ids = sorted(
            {t.resolve("entity", who) for _, who in ev.participants if who in decider_names}
        )
        wakes = participant_ids
        if not wakes and ev.visibility == "public" and not ev.participants:
            wakes = deciders
        emit(
            f"wake on event {ev.name}",
            wakes=wakes,
            reason=f"{ev.name!r}: {ev.meaning}",
            trigger_key="on_event_type",
            trigger=t.resolve("event", ev.name),
            rule="event → wake_rules[].on_event_type waking its deciding participants "
            "(all deciding entities when public with no participants)",
            evidence=ev.evidence_claim_ids,
            dropped_why="no deciding entity participates in or is woken by this event",
        )

    for name in sorted(_terminal_counted_events(plan.terminal)):
        ev2 = next((e for e in plan.events if e.name == name), None)
        meaning = ev2.meaning if ev2 is not None else name
        evidence = ev2.evidence_claim_ids if ev2 is not None else ()
        emit(
            f"wake on recorded {name} (terminal)",
            wakes=deciders,
            reason=f"the terminal counts records of {name!r}: {meaning}",
            trigger_key="on_record_in",
            trigger=t.resolve("event", name),
            rule="terminal-counted event → wake_rules[].on_record_in waking every deciding entity",
            evidence=evidence,
            dropped_why="no deciding entity exists to wake when this record is appended",
        )

    return rules


def representation_record(plan: SemanticPlan, t: SymbolTable) -> dict[str, Any]:
    """The representation-scale record (§7 / CWF-1), as an artifact rather than prose.

    Who is in this world and why, who was left out and why that cannot matter, at what
    scale each thing is represented, and — where the plan claimed one — the standing
    exemptions it is relying on. It rides in the compilation, so it lands in
    ``compiled_world.json`` and can be read beside the world it justifies instead of
    living only in the planner's reasoning, which nobody keeps.

    Every entry carries the runtime id its semantic name minted, so a reviewer reading
    the executable can join a compiled entity back to the argument for its existence.
    """

    relevant = sorted(terminal_relevant_states(plan))
    included = [
        {
            "entity": e.name,
            "runtime_id": t.resolve("entity", e.name),
            "structural_type": e.structural_type,
            "representation_scale": e.representation_scale,
            "represents_count": e.represents_count,
            "decides": e.decides,
            "why_it_can_change_the_answer": e.why_material,
            "terminal_relevant_state_it_can_alter": e.terminal_state_it_can_change,
            "information_it_receives": e.information_received,
            "authority_it_holds": e.authority,
            "if_removed": e.if_removed,
            "affordances": sorted(a.name for a in plan.affordances if a.actor == e.name),
            "evidence_claim_ids": list(e.evidence_claim_ids),
        }
        for e in plan.entities
    ]
    record: dict[str, Any] = {
        "terminal_relevant_states": relevant,
        "included": included,
        "excluded": [
            {
                "candidate": x.name,
                "why_removal_cannot_change_the_answer": x.why_immaterial,
                "evidence_claim_ids": list(x.evidence_claim_ids),
            }
            for x in plan.excluded_candidates
        ],
        "deciding_entities": sorted(e.name for e in plan.entities if e.decides),
        "zero_actor_justification": None,
        "single_multiplier_exemption": None,
    }
    if plan.zero_actor_claim is not None:
        record["zero_actor_justification"] = {
            "no_material_decision": plan.zero_actor_claim.no_material_decision,
            "process_sufficiency": plan.zero_actor_claim.process_sufficiency,
            "evidence_claim_ids": list(plan.zero_actor_claim.evidence_claim_ids),
        }
    if plan.single_driver_exemption is not None:
        record["single_multiplier_exemption"] = {
            "empirical_model": plan.single_driver_exemption.empirical_model,
            "parameter_uncertainty": plan.single_driver_exemption.parameter_uncertainty,
            "evidence_claim_ids": list(plan.single_driver_exemption.evidence_claim_ids),
        }
    return record


def lower_plan(
    plan: SemanticPlan, *, structure_id: str = "primary"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The approved plan → (compilation dict, mapping artifact records).

    The compilation dict has exactly the shape the direct compiler's model returns —
    world_spec / uncertainties / world_facts / required_reality_facts plus the contract
    fields — so both compiler modes feed the identical downstream path. ``structure_id``
    is stamped before the digest is computed, so the recorded hash is the hash of the
    artifact that is actually emitted.
    """

    t = build_symbols(plan)
    node_processes = _node_process_names(plan)
    stocks = _stock_table(plan, t)
    flows = flow_periods(plan)

    entities: list[dict[str, Any]] = []
    for e in plan.entities:
        auth = sorted(t.resolve("authority", a.name) for a in plan.affordances if a.actor == e.name)
        attributes: dict[str, Any] = {}
        if e.authority:
            # The minted tokens are what the executor checks; the ordinary-language
            # authority is what the actor was described as holding. Both survive.
            attributes["authority_description"] = e.authority
        entities.append(
            {
                "entity_id": t.resolve("entity", e.name),
                "name": e.name,
                "kind": _lookup(_KIND_BY_TYPE, e.structural_type, "structural_type"),
                "is_actor": e.decides,
                "role": e.role,
                "authority": auth,
                "representation_scale": e.representation_scale,
                "represents_count": e.represents_count,
                "attributes": attributes,
                "evidence_claim_ids": list(e.evidence_claim_ids),
            }
        )

    actors = [
        {
            "entity_id": t.resolve("entity", e.name),
            "reasoning": e.why_material,
            "goals": [],
            "memory_seeds": [],
            "commitments": [],
        }
        for e in plan.entities
        if e.decides
    ]

    fields = []
    for s in plan.states:
        # why_material is meaning, not metadata: it survives in the field description
        # every consumer of the field reads, never validated-then-vanished. So does the
        # kind: a reader of the executable can see that this number is a held quantity
        # with a floor, or a rate quoted over a period, without going back to the plan.
        desc = f"{s.name} ({s.owner})" + (f" [{s.unit}]" if s.unit else "")
        if s.why_material:
            desc = f"{desc} — {s.why_material}"
        if s.kind == "stock":
            bounds = f"floor {s.conserved_floor:g}"
            if s.capacity is not None:
                bounds = f"{bounds}, capacity {s.capacity:g}"
            desc = f"{desc} [stock held by {s.owner}; {bounds}]"
        elif s.kind == "flow":
            desc = f"{desc} [rate per {s.period}]"
        f: dict[str, Any] = {
            "field_id": t.resolve("field", s.name),
            "value_type": _lookup(_VALUE_TYPES, s.state_type, "state_type"),
            "description": desc,
            "evidence_claim_ids": list(s.evidence_claim_ids),
        }
        if s.initial != UNKNOWN:
            f["initial"] = s.initial
        fields.append(f)

    # The remainder fields. A draw a stock could not pay for, and quantity offered above
    # a ceiling, are causal facts the world produced — the backlog and the spill — so
    # they are world state a condition or a terminal can read, not a note in a log.
    for name in sorted(stocks):
        stock = stocks[name]
        s = stock.state
        if stock.unmet_id:
            fields.append(
                {
                    "field_id": stock.unmet_id,
                    "value_type": "number",
                    "initial": 0,
                    "description": f"cumulative quantity of {s.name} drawn for but not "
                    f"there [{s.unit}] — the shortfall this world's draws left unmet "
                    f"once {s.name} reached its floor of {s.conserved_floor:g}",
                    "evidence_claim_ids": list(s.evidence_claim_ids),
                }
            )
        if stock.overflow_id and s.capacity is not None:
            fields.append(
                {
                    "field_id": stock.overflow_id,
                    "value_type": "number",
                    "initial": 0,
                    "description": f"cumulative quantity offered to {s.name} above its "
                    f"capacity of {s.capacity:g} [{s.unit}] — what the world produced "
                    "and this stock could not hold",
                    "evidence_claim_ids": list(s.evidence_claim_ids),
                }
            )

    actions = []
    for a in plan.affordances:
        effects: list[dict[str, Any]] = []
        for c in a.changes:
            effects.extend(_lower_change(c, t, plan, stocks, flows))
        meaning = a.meaning
        if a.preconditions:
            # Free-text preconditions are not mechanically enforceable in this slice;
            # they are carried into the action's meaning — which the actor reads when
            # deciding — and recorded in the mapping, never silently dropped.
            meaning = f"{meaning} [precondition: {a.preconditions}]"
        if a.authority_required:
            meaning = f"{meaning} [requires: {a.authority_required}]"
        actions.append(
            {
                "action_id": t.resolve("action", a.name),
                "meaning": meaning,
                "eligible_actors": [t.resolve("entity", a.actor)],
                "required_authority": [t.resolve("authority", a.name)],
                "parameters": [],
                # An affordance with no semantic target takes no target: [] is the
                # runtime's "no target required". ["*"] means "any target, but one is
                # REQUIRED" — emitting it for a targetless act made the executor reject
                # every attempt whose actor did not invent a target string.
                "valid_targets": [t.resolve("entity", a.target)] if a.target else [],
                "visibility": a.visibility,
                "duration_seconds": a.duration_seconds,
                "effects": effects,
                "evidence_claim_ids": list(a.evidence_claim_ids),
            }
        )

    # -- processes: nodes chained by the edge the runtime actually executes ---------
    #
    # The engine schedules successors from `next_nodes` alone; `after_node` is only a
    # seeding suppressor plus timing metadata. So dependency chains are built by
    # appending each dependent node's id to its PREDECESSOR's next_nodes — a node
    # reachable only through `after_node` would never fire while still counting as a
    # terminal producer, the exact inert-but-gate-passing world this mode exists to
    # prevent.
    nodes: list[dict[str, Any]] = []
    externals: list[dict[str, Any]] = []
    node_index: dict[str, dict[str, Any]] = {}
    last_node_of_process: dict[str, str] = {}

    def emit_node(node: dict[str, Any]) -> None:
        nodes.append(node)
        node_index[str(node["node_id"])] = node

    for p in plan.processes:
        if p.kind == "actor_moment":
            node_id = t.resolve("node", p.name)
            allowed = (
                [t.resolve("action", n) for n in p.allowed_affordances]
                if p.allowed_affordances
                else ["*"]
            )
            # An actor_moment's occurrence changes are environment effects of the
            # moment itself — the gavel that seats the quorum — and land on the node,
            # not on the floor.
            effects = []
            for o in p.occurrences:
                if o.after_process is not None:
                    raise LoweringGap(
                        f"actor_moment {p.name!r} occurrence chained on {o.after_process!r}",
                        why="an actor moment is one dated occasion; a dependent "
                        "occurrence inside it has no universal meaning yet",
                        composable=True,
                        smallest_missing="a follow-up actor_moment process declared "
                        "separately and chained with after_process",
                    )
                for c in o.changes:
                    effects.extend(_lower_change(c, t, plan, stocks, flows))
            emit_node(
                {
                    "node_id": node_id,
                    "stage": node_id,
                    "description": _produced(p.meaning, p),
                    "at": _aware_iso(p.at),
                    "after_node": "",
                    "delay_seconds": 0,
                    "participants": [t.resolve("entity", who) for who in p.participants],
                    "action_ids": allowed,
                    "allow_novel": False,
                    "deadline": _aware_iso(p.deadline),
                    "effects": effects,
                    "next_nodes": [],
                    "evidence_claim_ids": list(p.evidence_claim_ids),
                }
            )
            last_node_of_process[p.name] = node_id
        elif p.name in node_processes:
            for j, o in enumerate(p.occurrences):
                effects = []
                for c in o.changes:
                    effects.extend(_lower_change(c, t, plan, stocks, flows))
                node_id = (
                    t.resolve("node", p.name)
                    if j == 0
                    else t.resolve("node", f"{p.name} occurrence {j + 1}")
                )
                emit_node(
                    {
                        "node_id": node_id,
                        "stage": t.resolve("node", p.name),
                        "description": _produced(o.description or p.meaning, p),
                        "at": _aware_iso(o.at),
                        "after_node": "",
                        "delay_seconds": o.delay_seconds,
                        "participants": [],
                        "action_ids": [],
                        "allow_novel": False,
                        # The engine schedules a deadline entry only for a node's
                        # participants; on a participant-less operational node the field
                        # is inert, and an inert field is a silent drop wearing a
                        # carried one's shape. Recorded as dropped below instead.
                        "effects": effects,
                        "next_nodes": [],
                        "evidence_claim_ids": list(p.evidence_claim_ids),
                        **_recurrence_keys(p),
                    }
                )
                last_node_of_process[p.name] = node_id
            if p.deadline:
                t.records.append(
                    {
                        "semantic": f"{p.name} deadline",
                        "namespace": "dropped",
                        "runtime_id": "",
                        "lowering_rule": "carried nowhere: the engine schedules deadline "
                        "entries only for a node's participants, and an operational node "
                        "has none",
                        "evidence_claim_ids": [],
                    }
                )
        else:
            occurrences = []
            for o in p.occurrences:
                effects = []
                for c in o.changes:
                    effects.extend(_lower_change(c, t, plan, stocks, flows))
                occurrences.append(
                    {
                        "at": _aware_iso(o.at),
                        "description": o.description or p.meaning,
                        "effects": effects,
                    }
                )
            externals.append(
                {
                    "process_id": t.resolve("external", p.name),
                    "description": _produced(p.meaning, p),
                    "occurrences": occurrences,
                    "evidence_claim_ids": list(p.evidence_claim_ids),
                    **_recurrence_keys(p),
                }
            )
            if p.deadline is not None:
                # An external process has no node to carry a deadline — the runtime
                # enforces deadlines on process nodes only. The meaning cannot lower,
                # so its loss is recorded, never silent.
                t.records.append(
                    {
                        "semantic": f"deadline of process {p.name}",
                        "namespace": "dropped",
                        "runtime_id": "",
                        "lowering_rule": "carried nowhere: an external process has no "
                        "node to carry a deadline; only process nodes enforce one",
                        "evidence_claim_ids": list(p.evidence_claim_ids),
                    }
                )

    # Second pass over dependencies: wire each dependent node into its predecessor's
    # next_nodes. Within a process, occurrence j follows occurrence j-1 when it names
    # its own process; across processes it follows the other process's last node.
    for p in plan.processes:
        if p.kind == "actor_moment" or p.name not in node_processes:
            continue
        prev_in_process: str | None = None
        for j, o in enumerate(p.occurrences):
            node_id = (
                t.resolve("node", p.name)
                if j == 0
                else t.resolve("node", f"{p.name} occurrence {j + 1}")
            )
            if o.after_process is not None:
                if o.after_process == p.name:
                    predecessor = prev_in_process
                else:
                    predecessor = last_node_of_process.get(o.after_process)
                if predecessor is None or predecessor == node_id:
                    raise LoweringGap(
                        f"occurrence of {p.name!r} chained on {o.after_process!r}",
                        why="the dependency has no predecessor node to fire from — a "
                        "first occurrence cannot follow its own process, and the "
                        "referenced process produced no node",
                        composable=False,
                        smallest_missing="nothing — the dependency must name a real "
                        "prior occurrence",
                    )
                node_index[predecessor]["next_nodes"] = list(
                    node_index[predecessor].get("next_nodes") or []
                ) + [node_id]
                node_index[node_id]["after_node"] = predecessor
            prev_in_process = node_id

    # A declared cadence is a claim about the world, and the calendar it produced is
    # arithmetic on that claim. Both are recorded, so a reader of the executable can see
    # that thirteen firings came from "every P1W from … to …" rather than from a list
    # somebody typed — and can check the count without re-deriving it.
    for p in plan.processes:
        if p.recurrence is None or not p.occurrences_generated:
            continue
        rec = p.recurrence
        t.records.append(
            {
                "semantic": f"{p.name} every {rec.period} from {rec.start} to {rec.end}",
                "namespace": "recurrence",
                "runtime_id": t.by_key.get(("node", p.name), "")
                or t.by_key.get(("external", p.name), ""),
                "lowering_rule": f"process recurrence → {len(p.occurrences)} enumerated "
                "occurrence(s), generated deterministically by code; the planner declared "
                "the cadence and wrote no dates",
                "evidence_claim_ids": list(p.evidence_claim_ids),
            }
        )

    uncertainties = []
    for u in plan.uncertainties:
        field_id = t.resolve("field", u.affects_state)
        n = len(u.alternatives)
        outcomes = []
        for alt in u.alternatives:
            if alt.weight is None:
                # The planner declared no split, so code mints the uniform one — and a
                # weight the code minted cannot inherit a grounded label the code did
                # not earn. Symmetric ignorance is what it is, and downstream that is
                # exactly what keeps the point estimate honest (scenario bounds, not a
                # calibrated number).
                weight = 1.0 / n
                provenance = "symmetric_ignorance_assumption"
            else:
                weight = alt.weight
                provenance = alt.provenance
            outcomes.append(
                {
                    "value": str(alt.value),
                    "weight": weight,
                    "provenance": provenance,
                    "field_effects": [[field_id, alt.value]],
                    "description": alt.grounding,
                }
            )
        uncertainties.append(
            {
                "variable": field_id,
                "why_unknown": f"{u.what_unknown} — {u.why_unknown}",
                "reversal_capable": True,
                "release_at": _aware_iso(u.release_at),
                "constraining_evidence_ids": sorted(
                    {i for alt in u.alternatives for i in alt.evidence_claim_ids}
                ),
                "outcomes": outcomes,
            }
        )
        t.records.append(
            {
                "semantic": u.name,
                "namespace": "uncertainty",
                "runtime_id": field_id,
                "lowering_rule": "uncertainty → uncertainties[].variable (the affected "
                "state's field id)",
                "evidence_claim_ids": sorted(
                    {i for alt in u.alternatives for i in alt.evidence_claim_ids}
                ),
            }
        )

    world_spec = {
        "title": plan.target_outcome or plan.question,
        "structure_rationale": plan.terminal_producer_note,
        "structure_id": structure_id,
        "entities": entities,
        "actors": actors,
        "fields": fields,
        "resources": [],
        "channels": [],
        "documents": [],
        "actions": actions,
        "process": {"nodes": nodes},
        "external_processes": externals,
        "wake_rules": _wake_rules(plan, t),
        "terminal": {
            "yes_when": _lower_terminal(plan.terminal, t, flows),
            "unresolved_when": _unresolved_when(plan, t),
            "description": plan.yes_condition,
        },
    }

    # Required reality facts, derived from what the plan itself claims about reality.
    # Emitting [] here made reality-gate check 4 vacuous in semantic mode only: with
    # no required facts, nothing was ever checked against the evidence and
    # evidence_coverage reported a meaningless 1.0. What the plan states as established
    # — a cited world_fact, a state whose initial value citations establish — is
    # exactly what rollout depends on being true, so each becomes a fact the gate must
    # find satisfied by available evidence.
    required_facts: list[dict[str, Any]] = []
    for text, ids in plan.world_facts:
        if ids:
            required_facts.append(
                {
                    "key": t.mint(
                        "reality_fact",
                        text,
                        rule="cited world_fact → required_reality_facts[].key",
                        evidence=ids,
                    ),
                    "description": text,
                    "evidence_claim_ids": list(ids),
                }
            )
    for s in plan.states:
        if s.initial is not None and s.initial != UNKNOWN and s.evidence_claim_ids:
            required_facts.append(
                {
                    "key": t.mint(
                        "reality_fact",
                        f"initial {s.name}",
                        rule="citation-established initial state → required_reality_facts[].key",
                        evidence=s.evidence_claim_ids,
                    ),
                    "description": f"initial value of {s.name} is {s.initial!r}, "
                    "established by the cited evidence",
                    "evidence_claim_ids": list(s.evidence_claim_ids),
                }
            )

    record = representation_record(plan, t)
    compilation = {
        "subject_entity": plan.subject_entity,
        "resolution_units": plan.resolution_units,
        "target_outcome": plan.target_outcome,
        "expected_participants": plan.expected_participants,
        # A first-class section of the compilation, not a note: it is written with the
        # executable, so every run's own artifacts carry the argument for who is in its
        # world and who is not.
        "representation_record": record,
        # FD-26. The standing claims a reviewer is required to audit are lifted to the
        # top of the compilation as well as living inside the record, because the
        # artifact a reviewer opens is `compiled_world.json` and a claim they have to
        # go looking for is a claim that does not get audited. A live actor-free run
        # supplied a zero-actor justification with four cited claims and the executable
        # a reviewer read reported it as null. Any claim a reviewer must audit reaches
        # the artifact the reviewer reads, at the level they read it.
        "zero_actor_justification": record["zero_actor_justification"],
        "single_multiplier_exemption": record["single_multiplier_exemption"],
        "world_spec": world_spec,
        "uncertainties": uncertainties,
        "world_facts": [
            {
                "text": text,
                "evidence_claim_ids": list(ids),
                # An uncited statement is a hypothesis regardless of its label; only a
                # cited one may enter the world as an observation.
                "epistemic_type": "observation" if ids else "hypothesis",
            }
            for text, ids in plan.world_facts
        ],
        "required_reality_facts": required_facts,
    }

    exe_hash = hashlib.sha256(
        json.dumps(compilation, sort_keys=True, default=str).encode()
    ).hexdigest()
    plan_hash = hashlib.sha256(repr(plan).encode()).hexdigest()
    mapping = list(t.records)
    mapping.append(
        {
            "semantic": "(whole plan)",
            "namespace": "lowering",
            "runtime_id": exe_hash[:16],
            "lowering_rule": "sha256 of the emitted compilation (post structure_id) — "
            f"identical plans lower to identical executables; plan sha256 {plan_hash[:16]}",
            "evidence_claim_ids": [],
        }
    )
    return compilation, mapping


def _unresolved_when(plan: SemanticPlan, t: SymbolTable) -> dict[str, Any]:
    """Honest unresolved, derived from the terminal's own unknown terms.

    The runtime's comparison operators are total — an absent quantity coerces to zero —
    so a terminal over a state the world never produced would confidently resolve NO.
    UNKNOWN must stay unresolved instead: the condition is the OR of an is-unset test
    for every UNKNOWN-initial state the terminal reads (directly or through a
    threshold). When the world later writes the state, the test turns false and the
    terminal resolves on the produced value; when nothing ever writes it, the branch
    reports unresolved, never a manufactured NO.
    """

    unknown_states = {s.name for s in plan.states if s.initial == UNKNOWN}

    read: set[str] = set()

    def walk(q: TerminalQuery) -> None:
        if q.state:
            read.add(q.state)
        if q.threshold is not None:
            read.update(q.threshold.states_read())
        for part in q.parts:
            walk(part)

    walk(plan.terminal)

    # Unknowns propagate through production: a terminal total computed from an UNKNOWN
    # driver is itself undetermined, even though the terminal never reads the driver by
    # name — a live world scaled its second production stage by an UNKNOWN rate, the
    # effect could not evaluate, and the branch resolved a confident NO off the partial
    # total. Chase the reads of every change that writes a relevant state, to fixpoint.
    all_changes = [c for a in plan.affordances for c in a.changes] + [
        c for p in plan.processes for o in p.occurrences for c in o.changes
    ]
    relevant = set(read)
    while True:
        grown = set(relevant)
        for c in all_changes:
            if c.op in ("set", "increase", "decrease") and c.target in relevant:
                if c.value is not None:
                    grown |= c.value.states_read()
                if c.amount is not None:
                    grown |= c.amount.states_read()
        if grown == relevant:
            break
        relevant = grown

    unset_tests = [
        {"op": "equals", "args": [{"op": "field", "args": [t.resolve("field", name)]}, None]}
        for name in sorted(relevant & unknown_states)
    ]
    if not unset_tests:
        return {"op": "const", "args": [False]}
    if len(unset_tests) == 1:
        return unset_tests[0]
    return {"op": "or", "args": unset_tests}
