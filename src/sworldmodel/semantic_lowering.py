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
    SemanticChange,
    SemanticPlan,
    SemanticValue,
    TerminalQuery,
)


class LoweringGap(WorldIntegrityError):
    """A semantic construct the universal change mapping cannot yet represent.

    Raised instead of approximating, substituting a nearby operation, or silently
    discarding meaning. Carries everything PART 9 requires so the gap can be judged:
    what construct, why existing changes cannot express it, whether it composes from
    current primitives, and the smallest genuinely universal capability that is
    missing.
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
            )
        return self.by_key[key]


def build_symbols(plan: SemanticPlan) -> SymbolTable:
    t = SymbolTable()
    for e in plan.entities:
        t.mint("entity", e.name, rule="entity → entities[].entity_id", evidence=e.evidence_claim_ids)
    for s in plan.states:
        t.mint("field", s.name, rule="state → fields[].field_id", evidence=s.evidence_claim_ids)
    for ev in plan.events:
        t.mint(
            "event",
            ev.name,
            rule="event → create_event.event_type + append_record.collection",
            evidence=ev.evidence_claim_ids,
        )
    for a in plan.affordances:
        t.mint("action", a.name, rule="affordance → actions[].action_id", evidence=a.evidence_claim_ids)
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


def _lower_value(v: SemanticValue, t: SymbolTable) -> Any:
    if v.kind == "literal":
        return v.literal
    if v.kind == "state":
        assert v.state is not None
        return {"op": "field", "args": [t.resolve("field", v.state)]}
    op = "add" if v.kind == "sum" else "multiply"
    parts = [_lower_value(p, t) for p in v.parts]
    # The runtime's arithmetic is binary; fold left so any arity lowers.
    out = {"op": op, "args": [parts[0], parts[1]]}
    for extra in parts[2:]:
        out = {"op": op, "args": [out, extra]}
    return out


def _negate(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return -value
    return {"op": "multiply", "args": [-1, value]}


def _lower_change(c: SemanticChange, t: SymbolTable, plan: SemanticPlan) -> list[dict[str, Any]]:
    """One universal semantic change → the existing effect operations."""

    if c.op == "set":
        assert c.value is not None  # validator guarantees
        return [
            {"op": "set_field", "field": t.resolve("field", c.target), "value": _lower_value(c.value, t)}
        ]
    if c.op in ("increase", "decrease"):
        assert c.amount is not None
        delta = _lower_value(c.amount, t)
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
            )
        # The event's declared meaning survives whole: its visibility governs who the
        # runtime delivers it to, and its participants and created information ride in
        # the payload — a private briefing must not become a public broadcast because
        # lowering forgot to say otherwise.
        data: dict[str, Any] = {"detail": c.detail}
        if ev.participants:
            data["participants"] = {
                role: t.resolve("entity", who) for role, who in ev.participants
            }
        if ev.information_created:
            data["information_created"] = ev.information_created
        return [
            {
                "op": "create_event",
                "event_type": sym,
                "text": ev.meaning,
                "visibility": ev.visibility,
                "data": data,
            },
            {"op": "append_record", "collection": sym, "key": "$actor", "value": c.detail or ev.meaning},
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
        why=f"no universal mapping exists for this {what}; the legal values are "
        f"{sorted(table)}",
        composable=False,
        smallest_missing=f"a universal runtime meaning for {what} {key!r}",
    )


def _lower_terminal(q: TerminalQuery, t: SymbolTable) -> dict[str, Any]:
    if q.form == "all_of":
        return {"op": "and", "args": [_lower_terminal(p, t) for p in q.parts]}
    if q.form == "any_of":
        return {"op": "or", "args": [_lower_terminal(p, t) for p in q.parts]}
    if q.form == "not":
        return {"op": "not", "args": [_lower_terminal(q.parts[0], t)]}
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
                _lower_value(q.threshold, t),
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
                _lower_value(q.threshold, t),
            ],
        }
    raise LoweringGap(
        f"terminal form {q.form!r}",
        why="no deterministic lowering exists for this form",
        composable=False,
        smallest_missing=f"a universal terminal lowering for {q.form!r}",
    )


_VALUE_TYPES = {"quantity": "number", "boolean": "bool", "category": "string", "text": "string"}


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
                rule="process input → wake_rules[].on_field_change waking every "
                "deciding entity",
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
            rule="terminal-counted event → wake_rules[].on_record_in waking every "
            "deciding entity",
            evidence=evidence,
            dropped_why="no deciding entity exists to wake when this record is appended",
        )

    return rules


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

    entities: list[dict[str, Any]] = []
    for e in plan.entities:
        auth = sorted(
            t.resolve("authority", a.name) for a in plan.affordances if a.actor == e.name
        )
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
        f: dict[str, Any] = {
            "field_id": t.resolve("field", s.name),
            "value_type": _lookup(_VALUE_TYPES, s.state_type, "state_type"),
            "description": f"{s.name} ({s.owner})" + (f" [{s.unit}]" if s.unit else ""),
            "evidence_claim_ids": list(s.evidence_claim_ids),
        }
        if s.initial != UNKNOWN:
            f["initial"] = s.initial
        fields.append(f)

    actions = []
    for a in plan.affordances:
        effects: list[dict[str, Any]] = []
        for c in a.changes:
            effects.extend(_lower_change(c, t, plan))
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
                        f"actor_moment {p.name!r} occurrence chained on "
                        f"{o.after_process!r}",
                        why="an actor moment is one dated occasion; a dependent "
                        "occurrence inside it has no universal meaning yet",
                        composable=True,
                        smallest_missing="a follow-up actor_moment process declared "
                        "separately and chained with after_process",
                    )
                for c in o.changes:
                    effects.extend(_lower_change(c, t, plan))
            emit_node(
                {
                    "node_id": node_id,
                    "stage": node_id,
                    "description": p.meaning,
                    "at": p.at,
                    "after_node": "",
                    "delay_seconds": 0,
                    "participants": [t.resolve("entity", who) for who in p.participants],
                    "action_ids": allowed,
                    "allow_novel": False,
                    "deadline": p.deadline,
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
                    effects.extend(_lower_change(c, t, plan))
                node_id = (
                    t.resolve("node", p.name)
                    if j == 0
                    else t.resolve("node", f"{p.name} occurrence {j + 1}")
                )
                emit_node(
                    {
                        "node_id": node_id,
                        "stage": t.resolve("node", p.name),
                        "description": o.description or p.meaning,
                        "at": o.at,
                        "after_node": "",
                        "delay_seconds": o.delay_seconds,
                        "participants": [],
                        "action_ids": [],
                        "allow_novel": False,
                        "effects": effects,
                        "next_nodes": [],
                        "evidence_claim_ids": list(p.evidence_claim_ids),
                    }
                )
                last_node_of_process[p.name] = node_id
        else:
            occurrences = []
            for o in p.occurrences:
                effects = []
                for c in o.changes:
                    effects.extend(_lower_change(c, t, plan))
                occurrences.append(
                    {"at": o.at, "description": o.description or p.meaning, "effects": effects}
                )
            externals.append(
                {
                    "process_id": t.resolve("external", p.name),
                    "description": p.meaning,
                    "occurrences": occurrences,
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
                "release_at": u.release_at,
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
            "yes_when": _lower_terminal(plan.terminal, t),
            "unresolved_when": _unresolved_when(plan, t),
            "description": plan.yes_condition,
        },
    }

    compilation = {
        "subject_entity": plan.subject_entity,
        "resolution_units": plan.resolution_units,
        "target_outcome": plan.target_outcome,
        "expected_participants": plan.expected_participants,
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
        "required_reality_facts": [],
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
