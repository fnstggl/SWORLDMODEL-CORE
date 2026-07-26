"""The semantic causal-world plan: meaning without runtime syntax.

This is the intermediate representation between research and the executable WorldSpec.
It exists to split the compiler's two jobs — understanding the causal world, and
authoring an internally consistent program — so the model does only the first and code
does all of the second. Nothing here contains a runtime ID, a field namespace, an
effect-operation name, an expression AST, or a trace identifier: every object is named
in ordinary language, and the deterministic lowerer owns every symbol the runtime sees.

The object types are universal world structure, not question types: entity, state,
event, process, action affordance, uncertainty, terminal query. Real-world meanings
stay open-ended natural language inside them — "publicly communicates conditional
support for a further cut" is an event *meaning*, never a hardcoded mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# The single sentinel for a value the evidence does not establish. It survives to the
# runtime as an absent initial, where reading it is an honest unresolved — never zero,
# never False.
UNKNOWN = "UNKNOWN"

STRUCTURAL_TYPES = (
    "person",
    "organization",
    "coalition",
    "institution",
    "population",
    "market",
    "system",
    "object",
)
REPRESENTATION_SCALES = (
    "individual",
    "organization",
    "subunit",
    "population_stratum",
    "network",
    "external_process",
)
STATE_TYPES = ("quantity", "category", "boolean", "text")
PROCESS_KINDS = ("actor_moment", "operational", "scheduled_release")
VISIBILITIES = ("public", "private")

# Universal ways a world can change. These are operations, not domain events: "increase"
# can raise a delivery total, a reservoir level or a vote count without any of those
# being named here.
CHANGE_OPS = (
    "set",  # a state takes a value (literal, another state, or arithmetic over states)
    "increase",  # a quantity goes up by an amount
    "decrease",  # a quantity goes down by an amount
    "record_event",  # a declared event occurs (creates its record and its occurrence)
    "send",  # information reaches named recipients
    "schedule",  # a declared process occurrence is placed on the calendar
)

# Universal terminal forms. The meanings inside them stay open-ended; the form is what
# lowers deterministically to the runtime's expression operators.
TERMINAL_FORMS = (
    "event_exists",
    "state_equals",
    "quantity_comparison",
    "record_count",
    "all_of",
    "any_of",
    "not",
)
COMPARISONS = ("greater_than", "greater_or_equal", "less_than", "less_or_equal", "equals")

WEIGHT_PROVENANCES = (
    "symmetric_ignorance_assumption",
    "explicit_model_distribution",
    "calibrated_behavior_model",
    "market_or_survey_distribution",
    "direct_empirical_distribution",
    "sensitivity_only_branch",
)


class SemanticPlanError(ValueError):
    """A semantic plan that cannot be read or does not hold together.

    Carries precise, per-object messages so a revision round can name exactly what to
    fix — the semantic analogue of a compiler diagnostic, never a stack trace.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors) if errors else "invalid semantic plan")


@dataclass(frozen=True)
class SemanticValue:
    """A value in semantic form: a literal, a named state, or arithmetic over them.

    Arithmetic exists so production can be *computed* — a total as a grounded base
    scaled by an uncertain rate — without the model writing a runtime expression tree.
    """

    kind: str  # "literal" | "state" | "sum" | "product"
    literal: Any = None
    state: str | None = None
    parts: tuple[SemanticValue, ...] = ()

    def states_read(self) -> set[str]:
        if self.kind == "state" and self.state:
            return {self.state}
        out: set[str] = set()
        for p in self.parts:
            out |= p.states_read()
        return out


@dataclass(frozen=True)
class SemanticChange:
    """One universal change to the world, in semantic terms."""

    op: str
    target: str  # state name, event name, or information description
    value: SemanticValue | None = None  # for set
    amount: SemanticValue | None = None  # for increase/decrease
    recipients: tuple[str, ...] = ()  # for send: entity names
    detail: str = ""  # open-ended meaning (e.g. what the recorded event's value is)


@dataclass(frozen=True)
class SemanticEntity:
    name: str  # canonical real-world name
    structural_type: str
    role: str
    representation_scale: str
    decides: bool  # its own decisions can move the outcome (it is an actor)
    authority: str  # ordinary language description of what it may do
    why_material: str  # why it could change the answer
    represents_count: int | None = None
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticState:
    name: str  # ordinary-language name, unique among states
    owner: str  # entity name, or "world"
    state_type: str
    initial: Any  # literal, or UNKNOWN
    why_material: str
    unit: str = ""
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticEvent:
    """A material event *meaning* the world can produce — dynamically declared, never
    from a closed list."""

    name: str
    meaning: str
    participants: tuple[tuple[str, str], ...] = ()  # (semantic role, entity name)
    visibility: str = "public"
    information_created: str = ""
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticAffordance:
    """An action an actor may attempt, in semantic terms."""

    name: str
    meaning: str
    actor: str  # entity name; must have decides=True
    changes: tuple[SemanticChange, ...]
    target: str = ""  # optional entity name the action is directed at
    authority_required: str = ""  # ordinary language; lowering mints the token
    preconditions: str = ""  # ordinary language (advisory in the vertical slice)
    visibility: str = "public"
    duration_seconds: int = 0
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticOccurrence:
    """One dated or dependent firing of a process."""

    description: str
    changes: tuple[SemanticChange, ...]
    at: str | None = None  # ISO datetime
    after_process: str | None = None  # name of the process this follows
    delay_seconds: int = 0


@dataclass(frozen=True)
class SemanticProcess:
    """A moment or mechanism through which the world advances.

    actor_moment: a dated occasion at which named actors gain the opportunity to act.
    operational: a non-agent mechanism that produces state (throughput, accumulation).
    scheduled_release: information or data arriving on a calendar.
    """

    name: str
    meaning: str
    kind: str
    participants: tuple[str, ...] = ()  # entity names (actor_moment)
    allowed_affordances: tuple[str, ...] = ()  # affordance names available at the moment
    inputs: tuple[str, ...] = ()  # state names read
    occurrences: tuple[SemanticOccurrence, ...] = ()
    at: str | None = None  # actor_moment: the dated occasion
    deadline: str | None = None
    information_produced: str = ""
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticAlternative:
    value: Any
    provenance: str
    weight: float | None = None  # None with symmetric_ignorance — never invented
    grounding: str = ""  # what supports this alternative
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticUncertainty:
    name: str
    what_unknown: str
    why_unknown: str
    affects_state: str  # the state whose value the draw sets
    alternatives: tuple[SemanticAlternative, ...]
    release_at: str | None = None


@dataclass(frozen=True)
class TerminalQuery:
    """The resolution rule in one of the universal forms."""

    form: str
    event: str | None = None  # event_exists
    state: str | None = None  # state_equals / quantity_comparison
    value: Any = None  # state_equals
    comparison: str | None = None  # quantity_comparison / record_count
    threshold: SemanticValue | None = None  # quantity_comparison / record_count
    record_event: str | None = None  # record_count: counts records of this event
    parts: tuple[TerminalQuery, ...] = ()  # all_of / any_of / not


@dataclass(frozen=True)
class SemanticPlan:
    question: str
    yes_condition: str  # ordinary-language YES rule
    subject_entity: str
    resolution_units: str
    target_outcome: str
    entities: tuple[SemanticEntity, ...]
    states: tuple[SemanticState, ...]
    events: tuple[SemanticEvent, ...]
    affordances: tuple[SemanticAffordance, ...]
    processes: tuple[SemanticProcess, ...]
    uncertainties: tuple[SemanticUncertainty, ...]
    terminal: TerminalQuery
    terminal_producer_note: str  # who/what produces the resolving state and how
    expected_participants: int | None = None
    resolution_evidence_ids: tuple[str, ...] = ()
    world_facts: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (text, claim ids)


# ---------------------------------------------------------------------------
# Parsing — defensive, with per-field messages the revision round can act on
# ---------------------------------------------------------------------------


def _s(d: dict[str, Any], key: str, where: str, errors: list[str], default: str = "") -> str:
    v = d.get(key, default)
    if not isinstance(v, str):
        errors.append(f"{where}: '{key}' must be a string")
        return default
    return v


def _ids(d: dict[str, Any], key: str = "evidence_claim_ids") -> tuple[str, ...]:
    v = d.get(key) or []
    return tuple(str(x) for x in v) if isinstance(v, list) else ()


def _value(obj: Any, where: str, errors: list[str]) -> SemanticValue | None:
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool)) or (
        isinstance(obj, str) and not obj.startswith("state:")
    ):
        return SemanticValue(kind="literal", literal=obj)
    if isinstance(obj, str):
        return SemanticValue(kind="state", state=obj.removeprefix("state:").strip())
    if isinstance(obj, dict):
        kind = str(obj.get("kind", ""))
        if kind == "literal":
            return SemanticValue(kind="literal", literal=obj.get("value"))
        if kind == "state":
            name = obj.get("state") or obj.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"{where}: state reference needs a state name")
                return None
            return SemanticValue(kind="state", state=name)
        if kind in ("sum", "product"):
            parts = []
            for i, p in enumerate(obj.get("parts") or []):
                pv = _value(p, f"{where}.parts[{i}]", errors)
                if pv is not None:
                    parts.append(pv)
            if len(parts) < 2:
                errors.append(f"{where}: {kind} needs at least two parts")
                return None
            return SemanticValue(kind=kind, parts=tuple(parts))
        errors.append(f"{where}: unknown value kind {kind!r}")
        return None
    errors.append(f"{where}: unreadable value {type(obj).__name__}")
    return None


def _change(obj: Any, where: str, errors: list[str]) -> SemanticChange | None:
    if not isinstance(obj, dict):
        errors.append(f"{where}: a change must be an object")
        return None
    op = _s(obj, "op", where, errors)
    target = _s(obj, "target", where, errors)
    if op not in CHANGE_OPS:
        errors.append(f"{where}: unknown change op {op!r} (universal ops: {list(CHANGE_OPS)})")
        return None
    if not target:
        errors.append(f"{where}: change op {op!r} needs a target")
        return None
    recipients = obj.get("recipients") or []
    return SemanticChange(
        op=op,
        target=target,
        value=_value(obj.get("value"), f"{where}.value", errors),
        amount=_value(obj.get("amount"), f"{where}.amount", errors),
        recipients=tuple(str(r) for r in recipients) if isinstance(recipients, list) else (),
        detail=_s(obj, "detail", where, errors),
    )


def _terminal(obj: Any, where: str, errors: list[str]) -> TerminalQuery:
    bad = TerminalQuery(form="state_equals", state=None)
    if not isinstance(obj, dict):
        errors.append(f"{where}: terminal must be an object")
        return bad
    form = _s(obj, "form", where, errors)
    if form not in TERMINAL_FORMS:
        errors.append(f"{where}: unknown terminal form {form!r} (forms: {list(TERMINAL_FORMS)})")
        return bad
    if form in ("all_of", "any_of", "not"):
        parts = [
            _terminal(p, f"{where}.parts[{i}]", errors)
            for i, p in enumerate(obj.get("parts") or [])
        ]
        need = 1 if form == "not" else 2
        if len(parts) < need:
            errors.append(f"{where}: {form} needs at least {need} part(s)")
        return TerminalQuery(form=form, parts=tuple(parts))
    comparison = obj.get("comparison")
    if form in ("quantity_comparison", "record_count") and comparison not in COMPARISONS:
        errors.append(f"{where}: {form} needs comparison from {list(COMPARISONS)}")
    return TerminalQuery(
        form=form,
        event=obj.get("event"),
        state=obj.get("state"),
        value=obj.get("value"),
        comparison=comparison if isinstance(comparison, str) else None,
        threshold=_value(obj.get("threshold"), f"{where}.threshold", errors),
        record_event=obj.get("record_event"),
    )


def parse_semantic_plan(data: dict[str, Any]) -> SemanticPlan:
    """Read the planner's JSON into a SemanticPlan, or raise with per-field messages."""

    errors: list[str] = []
    if not isinstance(data, dict):
        raise SemanticPlanError(["the semantic plan must be a JSON object"])

    res = data.get("resolution") if isinstance(data.get("resolution"), dict) else {}
    assert isinstance(res, dict)

    entities: list[SemanticEntity] = []
    for i, e in enumerate(data.get("entities") or []):
        w = f"entities[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{w}: must be an object")
            continue
        st = _s(e, "structural_type", w, errors)
        scale = _s(e, "representation_scale", w, errors)
        if st not in STRUCTURAL_TYPES:
            errors.append(f"{w}: structural_type {st!r} not in {list(STRUCTURAL_TYPES)}")
        if scale not in REPRESENTATION_SCALES:
            errors.append(
                f"{w}: representation_scale {scale!r} not in {list(REPRESENTATION_SCALES)}"
            )
        rc = e.get("represents_count")
        entities.append(
            SemanticEntity(
                name=_s(e, "name", w, errors),
                structural_type=st,
                role=_s(e, "role", w, errors),
                representation_scale=scale,
                decides=bool(e.get("decides", False)),
                authority=_s(e, "authority", w, errors),
                why_material=_s(e, "why_material", w, errors),
                represents_count=int(rc) if isinstance(rc, (int, float)) and rc else None,
                evidence_claim_ids=_ids(e),
            )
        )

    states: list[SemanticState] = []
    for i, s in enumerate(data.get("states") or []):
        w = f"states[{i}]"
        if not isinstance(s, dict):
            errors.append(f"{w}: must be an object")
            continue
        stype = _s(s, "state_type", w, errors)
        if stype not in STATE_TYPES:
            errors.append(f"{w}: state_type {stype!r} not in {list(STATE_TYPES)}")
        states.append(
            SemanticState(
                name=_s(s, "name", w, errors),
                owner=_s(s, "owner", w, errors, default="world") or "world",
                state_type=stype,
                initial=s.get("initial", UNKNOWN),
                why_material=_s(s, "why_material", w, errors),
                unit=_s(s, "unit", w, errors),
                evidence_claim_ids=_ids(s),
            )
        )

    events: list[SemanticEvent] = []
    for i, ev in enumerate(data.get("events") or []):
        w = f"events[{i}]"
        if not isinstance(ev, dict):
            errors.append(f"{w}: must be an object")
            continue
        raw = ev.get("participants") or {}
        participants: tuple[tuple[str, str], ...] = ()
        if isinstance(raw, dict):
            participants = tuple((str(k), str(v)) for k, v in sorted(raw.items()))
        vis = _s(ev, "visibility", w, errors, default="public") or "public"
        if vis not in VISIBILITIES:
            errors.append(f"{w}: visibility {vis!r} not in {list(VISIBILITIES)}")
        events.append(
            SemanticEvent(
                name=_s(ev, "name", w, errors),
                meaning=_s(ev, "meaning", w, errors),
                participants=participants,
                visibility=vis,
                information_created=_s(ev, "information_created", w, errors),
                evidence_claim_ids=_ids(ev),
            )
        )

    affordances: list[SemanticAffordance] = []
    for i, a in enumerate(data.get("affordances") or data.get("action_affordances") or []):
        w = f"affordances[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{w}: must be an object")
            continue
        changes = [
            c
            for j, ch in enumerate(a.get("changes") or [])
            if (c := _change(ch, f"{w}.changes[{j}]", errors)) is not None
        ]
        vis = _s(a, "visibility", w, errors, default="public") or "public"
        dur = a.get("duration_seconds", 0)
        affordances.append(
            SemanticAffordance(
                name=_s(a, "name", w, errors),
                meaning=_s(a, "meaning", w, errors),
                actor=_s(a, "actor", w, errors),
                changes=tuple(changes),
                target=_s(a, "target", w, errors),
                authority_required=_s(a, "authority_required", w, errors),
                preconditions=_s(a, "preconditions", w, errors),
                visibility=vis if vis in VISIBILITIES else "public",
                duration_seconds=int(dur) if isinstance(dur, (int, float)) else 0,
                evidence_claim_ids=_ids(a),
            )
        )

    processes: list[SemanticProcess] = []
    for i, p in enumerate(data.get("processes") or []):
        w = f"processes[{i}]"
        if not isinstance(p, dict):
            errors.append(f"{w}: must be an object")
            continue
        kind = _s(p, "kind", w, errors)
        if kind not in PROCESS_KINDS:
            errors.append(f"{w}: kind {kind!r} not in {list(PROCESS_KINDS)}")
        occurrences: list[SemanticOccurrence] = []
        for j, o in enumerate(p.get("occurrences") or []):
            wo = f"{w}.occurrences[{j}]"
            if not isinstance(o, dict):
                errors.append(f"{wo}: must be an object")
                continue
            ochanges = [
                c
                for k, ch in enumerate(o.get("changes") or [])
                if (c := _change(ch, f"{wo}.changes[{k}]", errors)) is not None
            ]
            delay = o.get("delay_seconds", 0)
            occurrences.append(
                SemanticOccurrence(
                    description=_s(o, "description", wo, errors),
                    changes=tuple(ochanges),
                    at=o.get("at"),
                    after_process=o.get("after_process"),
                    delay_seconds=int(delay) if isinstance(delay, (int, float)) else 0,
                )
            )
        parts = p.get("participants") or []
        allowed = p.get("allowed_affordances") or []
        inputs = p.get("inputs") or []
        processes.append(
            SemanticProcess(
                name=_s(p, "name", w, errors),
                meaning=_s(p, "meaning", w, errors),
                kind=kind,
                participants=tuple(str(x) for x in parts) if isinstance(parts, list) else (),
                allowed_affordances=tuple(str(x) for x in allowed)
                if isinstance(allowed, list)
                else (),
                inputs=tuple(str(x) for x in inputs) if isinstance(inputs, list) else (),
                occurrences=tuple(occurrences),
                at=p.get("at"),
                deadline=p.get("deadline"),
                information_produced=_s(p, "information_produced", w, errors),
                evidence_claim_ids=_ids(p),
            )
        )

    uncertainties: list[SemanticUncertainty] = []
    for i, u in enumerate(data.get("uncertainties") or []):
        w = f"uncertainties[{i}]"
        if not isinstance(u, dict):
            errors.append(f"{w}: must be an object")
            continue
        alts: list[SemanticAlternative] = []
        for j, alt in enumerate(u.get("alternatives") or []):
            wa = f"{w}.alternatives[{j}]"
            if not isinstance(alt, dict):
                errors.append(f"{wa}: must be an object")
                continue
            prov = _s(alt, "provenance", wa, errors)
            if prov not in WEIGHT_PROVENANCES:
                errors.append(f"{wa}: provenance {prov!r} not in {list(WEIGHT_PROVENANCES)}")
            wt = alt.get("weight")
            alts.append(
                SemanticAlternative(
                    value=alt.get("value"),
                    provenance=prov,
                    weight=float(wt) if isinstance(wt, (int, float)) else None,
                    grounding=_s(alt, "grounding", wa, errors),
                    evidence_claim_ids=_ids(alt),
                )
            )
        uncertainties.append(
            SemanticUncertainty(
                name=_s(u, "name", w, errors),
                what_unknown=_s(u, "what_unknown", w, errors),
                why_unknown=_s(u, "why_unknown", w, errors),
                affects_state=_s(u, "affects_state", w, errors),
                alternatives=tuple(alts),
                release_at=u.get("release_at"),
            )
        )

    ep = res.get("expected_participants")
    facts: list[tuple[str, tuple[str, ...]]] = []
    for f in data.get("world_facts") or []:
        if isinstance(f, dict) and isinstance(f.get("text"), str):
            facts.append((f["text"], _ids(f)))

    plan = SemanticPlan(
        question=_s(res, "question", "resolution", errors),
        yes_condition=_s(res, "yes_condition", "resolution", errors),
        subject_entity=_s(res, "subject_entity", "resolution", errors),
        resolution_units=_s(res, "resolution_units", "resolution", errors),
        target_outcome=_s(res, "target_outcome", "resolution", errors),
        entities=tuple(entities),
        states=tuple(states),
        events=tuple(events),
        affordances=tuple(affordances),
        processes=tuple(processes),
        uncertainties=tuple(uncertainties),
        terminal=_terminal(data.get("terminal"), "terminal", errors),
        terminal_producer_note=_s(data, "terminal_producer_note", "plan", errors),
        expected_participants=int(ep) if isinstance(ep, (int, float)) else None,
        resolution_evidence_ids=_ids(res),
        world_facts=tuple(facts),
    )
    if errors:
        raise SemanticPlanError(errors)
    return plan


# ---------------------------------------------------------------------------
# Static validation — mechanical, zero LLM calls (Slice C)
# ---------------------------------------------------------------------------


def _parse_when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return None
    if when.tzinfo is None:
        # A planner that omits the offset means the moment, not a different one per
        # server timezone; naive reads as UTC so it can be compared with the contract's
        # aware cutoff and horizon instead of raising mid-validation.
        when = when.replace(tzinfo=UTC)
    return when


def validate_semantic_plan(
    plan: SemanticPlan,
    *,
    as_of: datetime | None = None,
    horizon: datetime | None = None,
    known_claim_ids: frozenset[str] | None = None,
) -> list[str]:
    """Every reference resolves; every producer chain holds; nothing writes the answer.

    Returns precise semantic errors (empty when valid). Mechanical only — the reality
    review is a separate, independent judgement.
    """

    errors: list[str] = []
    entity_names = {e.name for e in plan.entities}
    state_names = {s.name for s in plan.states}
    event_names = {ev.name for ev in plan.events}
    affordance_names = {a.name for a in plan.affordances}
    process_names = {p.name for p in plan.processes}

    def dup(names: list[str], kind: str) -> None:
        seen: set[str] = set()
        for n in names:
            if n in seen:
                errors.append(f"duplicate {kind} name {n!r} — every name must be unique")
            seen.add(n)

    dup([e.name for e in plan.entities], "entity")
    dup([s.name for s in plan.states], "state")
    dup([ev.name for ev in plan.events], "event")
    dup([a.name for a in plan.affordances], "affordance")
    dup([p.name for p in plan.processes], "process")
    dup([u.name for u in plan.uncertainties], "uncertainty")

    for e in plan.entities:
        if not e.name:
            errors.append("an entity has no name")
        if e.represents_count is not None and e.represents_count < 1:
            errors.append(f"entity {e.name!r}: represents_count must be >= 1")
        if e.decides and not e.evidence_claim_ids:
            errors.append(
                f"entity {e.name!r} decides but cites no evidence — a name with nothing "
                "behind it cannot be an actor"
            )

    for s in plan.states:
        if s.owner != "world" and s.owner not in entity_names:
            errors.append(f"state {s.name!r}: owner {s.owner!r} is not a declared entity")
        if s.state_type == "quantity" and s.initial is not UNKNOWN and s.initial != UNKNOWN:
            if isinstance(s.initial, (int, float)) and not isinstance(s.initial, bool):
                if not s.evidence_claim_ids:
                    errors.append(
                        f"state {s.name!r}: a precise initial quantity ({s.initial}) needs "
                        "evidence citations, or must be declared UNKNOWN"
                    )
            elif s.initial is not None:
                errors.append(f"state {s.name!r}: quantity initial must be a number or UNKNOWN")

    for ev in plan.events:
        for role, who in ev.participants:
            if who not in entity_names:
                errors.append(f"event {ev.name!r}: participant {who!r} ({role}) is not declared")
        if ev.visibility == "private" and not ev.participants:
            # A private occurrence is defined by who it reaches: with nobody named,
            # its audience is empty and the runtime delivers it to no one — an event
            # that happens to nobody is not a meaning the world can carry.
            errors.append(
                f"event {ev.name!r}: a private event with no participants cannot mean "
                "anything — name the participants it reaches, or make it public"
            )

    def check_value(v: SemanticValue | None, where: str) -> None:
        if v is None:
            return
        for ref in v.states_read():
            if ref not in state_names:
                errors.append(f"{where}: references undeclared state {ref!r}")

    def check_changes(changes: tuple[SemanticChange, ...], where: str) -> None:
        for c in changes:
            if c.op in ("set", "increase", "decrease"):
                if c.target not in state_names:
                    errors.append(f"{where}: change targets undeclared state {c.target!r}")
                if c.op == "set" and c.value is None:
                    errors.append(f"{where}: set {c.target!r} has no value")
                if c.op in ("increase", "decrease") and c.amount is None:
                    errors.append(f"{where}: {c.op} {c.target!r} has no amount")
            elif c.op == "record_event":
                if c.target not in event_names:
                    errors.append(f"{where}: records undeclared event {c.target!r}")
            elif c.op == "send":
                if not c.recipients:
                    # An audience-less delivery lowers to a public broadcast — the
                    # opposite of what "send to named recipients" means.
                    errors.append(f"{where}: send needs at least one recipient")
                for r in c.recipients:
                    if r not in entity_names:
                        errors.append(f"{where}: sends to undeclared entity {r!r}")
            elif c.op == "schedule":
                if c.target not in process_names:
                    errors.append(f"{where}: schedules undeclared process {c.target!r}")
            check_value(c.value, where)
            check_value(c.amount, where)

    for a in plan.affordances:
        if a.actor not in entity_names:
            errors.append(f"affordance {a.name!r}: actor {a.actor!r} is not declared")
        else:
            actor = next(e for e in plan.entities if e.name == a.actor)
            if not actor.decides:
                errors.append(
                    f"affordance {a.name!r}: {a.actor!r} is not a deciding entity — mark "
                    "decides=true or remove the affordance"
                )
        if a.target and a.target not in entity_names:
            errors.append(f"affordance {a.name!r}: target {a.target!r} is not declared")
        if not a.changes:
            errors.append(
                f"affordance {a.name!r} changes nothing — an action's meaning is its "
                "changes, so add them or remove it"
            )
        check_changes(a.changes, f"affordance {a.name!r}")

    deciders = {e.name for e in plan.entities if e.decides}
    for d in sorted(deciders):
        if not any(a.actor == d for a in plan.affordances):
            errors.append(
                f"entity {d!r} decides but has no affordance — give it the actions its "
                "role affords, or mark it decides=false"
            )

    # Every actor with an affordance needs a moment that can actually invoke it. An
    # affordance nobody is ever woken to use is an inert world wearing a live one's
    # shape: the schedule-viability question, answered before anything runs.
    moment_participants = {
        who for p in plan.processes if p.kind == "actor_moment" for who in p.participants
    }
    for a in plan.affordances:
        if a.actor in deciders and a.actor not in moment_participants:
            errors.append(
                f"affordance {a.name!r}: actor {a.actor!r} is never a participant of any "
                "actor_moment process, so no moment ever gives them the opportunity to "
                "act — declare their real dated occasion(s) as actor_moment processes"
            )
            break

    for p in plan.processes:
        for who in p.participants:
            if who not in entity_names:
                errors.append(f"process {p.name!r}: participant {who!r} is not declared")
        for aff in p.allowed_affordances:
            if aff not in affordance_names:
                errors.append(f"process {p.name!r}: allows undeclared affordance {aff!r}")
        for inp in p.inputs:
            if inp not in state_names:
                errors.append(f"process {p.name!r}: input {inp!r} is not a declared state")
        if p.kind == "actor_moment":
            if not p.participants:
                errors.append(f"process {p.name!r}: an actor_moment needs participants")
            if not p.at and not p.deadline:
                errors.append(
                    f"process {p.name!r}: an actor_moment needs a dated occasion (at) or "
                    "a deadline — nobody can act at a moment that never comes"
                )
            for who in p.participants:
                if who in entity_names and who not in deciders:
                    errors.append(
                        f"process {p.name!r}: participant {who!r} does not decide — an "
                        "actor_moment's participants must be deciding entities"
                    )
        else:
            if not p.occurrences:
                errors.append(f"process {p.name!r}: {p.kind} needs at least one occurrence")
            if p.participants:
                # A live slice run declared an actor's "decision window" as an
                # operational process with participants: it lowered to a non-agent
                # process, no moment ever woke the actor, and the world would have
                # resolved NO with zero actor invocations — inert but gate-passing.
                # Participants mean opportunities to act, and only an actor_moment
                # grants those.
                errors.append(
                    f"process {p.name!r}: a {p.kind} process cannot have participants — "
                    "if these actors gain an opportunity to act here, declare it as "
                    "kind=actor_moment with its dated occasion"
                )
        for j, o in enumerate(p.occurrences):
            where = f"process {p.name!r} occurrence[{j}]"
            if o.at is None and o.after_process is None:
                errors.append(f"{where}: needs 'at' or 'after_process'")
            if o.after_process is not None and o.after_process not in process_names:
                errors.append(f"{where}: after_process {o.after_process!r} is not declared")
            o_when = _parse_when(o.at)
            if o.at is not None and o_when is None:
                errors.append(f"{where}: 'at' is not an ISO datetime: {o.at!r}")
            if o_when and as_of and o_when <= as_of:
                # The simulation window opens at the cutoff; an occurrence dated at or
                # before it is the world re-performing history. A live run scheduled a
                # t0 occurrence that recorded the very event the question asks about —
                # the record already established it — and the branch resolved YES off a
                # simulated re-enactment nobody in the world produced.
                errors.append(
                    f"{where}: dated {o.at}, at or before the cutoff "
                    f"{as_of.isoformat()} — the simulation cannot re-perform history. "
                    "If the record establishes this outcome, put it in a cited initial "
                    "state value (or world_facts); otherwise date the occurrence "
                    "strictly after the cutoff"
                )
            check_changes(o.changes, where)
        when = _parse_when(p.at)
        if p.at is not None and when is None:
            errors.append(f"process {p.name!r}: 'at' is not an ISO datetime: {p.at!r}")
        if when and as_of and when <= as_of and p.kind == "actor_moment":
            errors.append(
                f"process {p.name!r}: actor_moment dated {p.at}, at or before the "
                f"cutoff {as_of.isoformat()} — a moment to act must lie inside the "
                "question's open window"
            )
        if when and horizon and when > horizon:
            errors.append(
                f"process {p.name!r}: occurs at {p.at} which is after the horizon "
                f"{horizon.isoformat()} — it cannot affect this question"
            )

    uncertain_states: set[str] = set()
    for u in plan.uncertainties:
        if u.affects_state not in state_names:
            errors.append(f"uncertainty {u.name!r}: affects_state {u.affects_state!r} not declared")
        else:
            uncertain_states.add(u.affects_state)
        if len(u.alternatives) < 2:
            errors.append(f"uncertainty {u.name!r}: needs at least two alternatives")
        weights = [a.weight for a in u.alternatives]
        if any(w is not None for w in weights):
            if any(w is None for w in weights):
                errors.append(
                    f"uncertainty {u.name!r}: give every alternative a weight or none — "
                    "a partial split is an invented distribution"
                )
            else:
                total = sum(w for w in weights if w is not None)
                if abs(total - 1.0) > 1e-6:
                    errors.append(f"uncertainty {u.name!r}: weights sum to {total}, not 1.0")
        for alt in u.alternatives:
            if alt.weight is None and alt.provenance != "symmetric_ignorance_assumption":
                # A null weight means "nothing supports a split" — code will mint the
                # uniform one, and a code-minted weight cannot wear a grounded label.
                errors.append(
                    f"uncertainty {u.name!r}: an alternative with no weight must carry "
                    "provenance symmetric_ignorance_assumption — a split nothing "
                    f"supports cannot be labeled {alt.provenance!r}"
                )
            if alt.weight is not None and alt.provenance == "symmetric_ignorance_assumption":
                continue
            if alt.weight is not None and not (alt.grounding or alt.evidence_claim_ids):
                errors.append(
                    f"uncertainty {u.name!r}: a weighted alternative needs grounding or "
                    "claim ids — weights are evidence or they are nothing"
                )

    # -- the terminal: every term resolves, and something produces it -------------
    written_states: set[str] = set()
    recorded_events: set[str] = set()
    for a in plan.affordances:
        for c in a.changes:
            if c.op in ("set", "increase", "decrease"):
                written_states.add(c.target)
            elif c.op == "record_event":
                recorded_events.add(c.target)
    for p in plan.processes:
        for o in p.occurrences:
            for c in o.changes:
                if c.op in ("set", "increase", "decrease"):
                    written_states.add(c.target)
                elif c.op == "record_event":
                    recorded_events.add(c.target)

    known = known_claim_ids if known_claim_ids is not None else None

    def cited(ids: tuple[str, ...]) -> bool:
        if not ids:
            return False
        return True if known is None else all(i in known for i in ids)

    def check_terminal(t: TerminalQuery, where: str) -> None:
        if t.form in ("all_of", "any_of", "not"):
            for i, p_ in enumerate(t.parts):
                check_terminal(p_, f"{where}.parts[{i}]")
            return
        if t.form == "event_exists":
            if not t.event or t.event not in event_names:
                errors.append(f"{where}: event_exists names undeclared event {t.event!r}")
            elif t.event not in recorded_events:
                errors.append(
                    f"{where}: nothing records event {t.event!r} — no affordance or "
                    "process has a record_event change for it, so the terminal has no "
                    "producer"
                )
            return
        if t.form == "record_count":
            if not t.record_event or t.record_event not in event_names:
                errors.append(f"{where}: record_count names undeclared event {t.record_event!r}")
            elif t.record_event not in recorded_events:
                errors.append(f"{where}: nothing records event {t.record_event!r}")
            if t.threshold is None:
                errors.append(f"{where}: record_count needs a threshold")
            check_value(t.threshold, where)
            return
        # state_equals / quantity_comparison
        if not t.state or t.state not in state_names:
            errors.append(f"{where}: names undeclared state {t.state!r}")
            return
        if t.form == "state_equals" and t.value is None:
            errors.append(
                f"{where}: state_equals over {t.state!r} needs a value — without one "
                "the comparison is against nothing and the answer is manufactured by "
                "an absent key"
            )
        state = next(s for s in plan.states if s.name == t.state)
        established = state.initial not in (None, UNKNOWN) and cited(state.evidence_claim_ids)
        if t.state not in written_states and not established:
            errors.append(
                f"{where}: state {t.state!r} has no producer — nothing sets or adjusts "
                "it, and its initial value is not established by cited evidence"
            )
        if t.state in uncertain_states:
            errors.append(
                f"{where}: state {t.state!r} is set directly by an uncertainty — the "
                "branch weights would be the answer. Put the uncertainty on a driver "
                "and let the world produce the resolving state"
            )

        # The launder, at the semantic level: a producer whose value for the terminal
        # state reads nothing but uncertainty draws. A bare copy is the one-hop form; a
        # product of two draws is the same defect in arithmetic clothing — a live Tesla
        # slice computed deliveries as battery_packs × efficiency with both factors
        # uncertain, so the "production model" was a pure function of the branch draw
        # and the weights were still the whole answer. Production needs at least one
        # grounded input.
        def launders(value: SemanticValue | None) -> bool:
            if value is None or value.kind == "literal":
                return False
            reads = value.states_read()
            return bool(reads) and reads <= uncertain_states

        for a in plan.affordances:
            for c in a.changes:
                if c.op == "set" and c.target == t.state and launders(c.value):
                    errors.append(
                        f"{where}: affordance {a.name!r} sets {t.state!r} from "
                        "uncertainty draws alone — model what produces the value with "
                        "at least one evidence-grounded input (a cited base, rate or "
                        "level), and keep the uncertainty on a driver"
                    )
        for p in plan.processes:
            for o in p.occurrences:
                for c in o.changes:
                    if c.op == "set" and c.target == t.state and launders(c.value):
                        errors.append(
                            f"{where}: process {p.name!r} sets {t.state!r} from "
                            "uncertainty draws alone — a value computed only from draws "
                            "is the branch weights wearing arithmetic; ground at least "
                            "one input in cited evidence"
                        )
        if t.form == "quantity_comparison":
            if state.state_type != "quantity":
                errors.append(f"{where}: quantity_comparison over non-quantity {t.state!r}")
            if t.threshold is None:
                errors.append(f"{where}: quantity_comparison needs a threshold")
            check_value(t.threshold, where)

    check_terminal(plan.terminal, "terminal")

    if not plan.terminal_producer_note:
        errors.append(
            "terminal_producer_note is required: say what produces the resolving state "
            "and why initialization or uncertainty does not already write the answer"
        )
    if plan.expected_participants is not None:
        represented = sum(max(1, e.represents_count or 1) for e in plan.entities if e.decides)
        if represented < plan.expected_participants:
            errors.append(
                f"the plan declares {plan.expected_participants} decision-relevant "
                f"participants but its deciding entities represent only {represented} — "
                "either add the missing participants, or put represents_count on the "
                "aggregate that stands for them (a coalition of seven deciding as one "
                "unit is one entity with represents_count 7), or lower "
                "expected_participants to what the evidence actually names"
            )
    if known is not None:
        for e in plan.entities:
            missing = [i for i in e.evidence_claim_ids if i not in known]
            if missing:
                errors.append(f"entity {e.name!r} cites unknown claim ids {missing}")

    return errors
