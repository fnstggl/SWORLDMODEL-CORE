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
    for p in plan.processes:
        ns = "node" if p.kind == "actor_moment" or _needs_nodes(p) else "external"
        t.mint(
            ns,
            p.name,
            rule=f"process ({p.kind}) → "
            + ("process.nodes[].node_id" if ns == "node" else "external_processes[].process_id"),
            evidence=p.evidence_claim_ids,
        )
    for u in plan.uncertainties:
        t.mint("uncertainty", u.name, rule="uncertainty → uncertainties[].variable", evidence=())
    return t


def _needs_nodes(p: Any) -> bool:
    """An operational process chained by dependency lowers to process nodes, because
    only nodes carry `after_node`; one whose occurrences are all absolutely dated
    lowers to an external process."""

    return any(o.after_process is not None for o in p.occurrences)


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
        ev = next(e for e in plan.events if e.name == c.target)
        return [
            {"op": "create_event", "event_type": sym, "text": ev.meaning, "data": {"detail": c.detail}},
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
            "op": _CMP_OPS[q.comparison],
            "args": [
                {"op": "count", "args": [t.resolve("event", q.record_event)]},
                _lower_value(q.threshold, t),
            ],
        }
    if q.form == "state_equals":
        assert q.state is not None
        return {
            "op": "equals",
            "args": [{"op": "field", "args": [t.resolve("field", q.state)]}, q.value],
        }
    if q.form == "quantity_comparison":
        assert q.state is not None and q.comparison is not None and q.threshold is not None
        return {
            "op": _CMP_OPS[q.comparison],
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


def lower_plan(plan: SemanticPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The approved plan → (compilation dict, mapping artifact records).

    The compilation dict has exactly the shape the direct compiler's model returns —
    world_spec / uncertainties / world_facts / required_reality_facts plus the contract
    fields — so both compiler modes feed the identical downstream path.
    """

    t = build_symbols(plan)

    entities: list[dict[str, Any]] = []
    for e in plan.entities:
        auth = sorted(
            t.resolve("authority", a.name) for a in plan.affordances if a.actor == e.name
        )
        entities.append(
            {
                "entity_id": t.resolve("entity", e.name),
                "name": e.name,
                "kind": _KIND_BY_TYPE.get(e.structural_type, "organization"),
                "is_actor": e.decides,
                "role": e.role,
                "authority": auth,
                "representation_scale": e.representation_scale,
                "represents_count": e.represents_count,
                "attributes": {},
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
            "value_type": _VALUE_TYPES[s.state_type],
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
        actions.append(
            {
                "action_id": t.resolve("action", a.name),
                "meaning": a.meaning,
                "eligible_actors": [t.resolve("entity", a.actor)],
                "required_authority": [t.resolve("authority", a.name)],
                "parameters": [],
                "valid_targets": [t.resolve("entity", a.target)] if a.target else ["*"],
                "visibility": a.visibility,
                "duration_seconds": a.duration_seconds,
                "effects": effects,
                "evidence_claim_ids": list(a.evidence_claim_ids),
            }
        )

    nodes: list[dict[str, Any]] = []
    externals: list[dict[str, Any]] = []
    for p in plan.processes:
        if p.kind == "actor_moment":
            node_id = t.resolve("node", p.name)
            allowed = (
                [t.resolve("action", n) for n in p.allowed_affordances]
                if p.allowed_affordances
                else ["*"]
            )
            nodes.append(
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
                    "next_nodes": [],
                    "evidence_claim_ids": list(p.evidence_claim_ids),
                }
            )
        elif _needs_nodes(p):
            base = t.resolve("node", p.name)
            for j, o in enumerate(p.occurrences):
                effects = []
                for c in o.changes:
                    effects.extend(_lower_change(c, t, plan))
                after = ""
                if o.after_process is not None:
                    ns = "node" if o.after_process != p.name else "node"
                    after = t.resolve(ns, o.after_process)
                nodes.append(
                    {
                        "node_id": base if j == 0 else f"{base}_{j + 1}",
                        "stage": base,
                        "description": o.description or p.meaning,
                        "at": o.at,
                        "after_node": after,
                        "delay_seconds": o.delay_seconds,
                        "participants": [],
                        "action_ids": [],
                        "allow_novel": False,
                        "effects": effects,
                        "next_nodes": [],
                        "evidence_claim_ids": list(p.evidence_claim_ids),
                    }
                )
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

    uncertainties = []
    for u in plan.uncertainties:
        field_id = t.resolve("field", u.affects_state)
        n = len(u.alternatives)
        outcomes = []
        for alt in u.alternatives:
            weight = alt.weight if alt.weight is not None else 1.0 / n
            outcomes.append(
                {
                    "value": str(alt.value),
                    "weight": weight,
                    "provenance": alt.provenance,
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

    world_spec = {
        "title": plan.target_outcome or plan.question,
        "structure_rationale": plan.terminal_producer_note,
        "entities": entities,
        "actors": actors,
        "fields": fields,
        "resources": [],
        "channels": [],
        "documents": [],
        "actions": actions,
        "process": {"nodes": nodes},
        "external_processes": externals,
        "wake_rules": [],
        "terminal": {
            "yes_when": _lower_terminal(plan.terminal, t),
            "unresolved_when": {"op": "const", "args": [False]},
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
            {"text": text, "evidence_claim_ids": list(ids), "epistemic_type": "observation"}
            for text, ids in plan.world_facts
        ],
        "required_reality_facts": [],
    }

    plan_hash = hashlib.sha256(
        json.dumps(compilation, sort_keys=True, default=str).encode()
    ).hexdigest()
    mapping = list(t.records)
    mapping.append(
        {
            "semantic": "(whole plan)",
            "namespace": "lowering",
            "runtime_id": plan_hash[:16],
            "lowering_rule": "sha256 of the lowered compilation — identical plans lower "
            "to identical executables",
            "evidence_claim_ids": [],
        }
    )
    return compilation, mapping
