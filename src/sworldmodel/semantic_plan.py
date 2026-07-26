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

# The provenances that name no distribution at all. Mirrors
# ``uncertainty.UNGROUNDED_PROVENANCES`` deliberately in string form: this module has no
# package imports, so the semantic layer stays readable without dragging the runtime
# model types into it.
UNGROUNDED_WEIGHT_PROVENANCES = ("symmetric_ignorance_assumption", "sensitivity_only_branch")

# What an alternative changes about the world. An alternative that changes nothing under
# any of these headings is a probability sink, not a possibility.
ALTERNATIVE_CHANGE_KINDS = ("structure", "actor_state", "process_state")

# How the terminal responds if this alternative is the one that holds. A closed
# vocabulary so the declaration can be checked against the plan's own arithmetic instead
# of being read as prose.
TERMINAL_SENSITIVITIES = (
    "decides_the_terminal",
    "moves_the_terminal",
    "immaterial_to_the_terminal",
)

# The state-writing operations. A change with one of these ops is production; anything
# else moves information rather than state.
WRITE_OPS = ("set", "increase", "decrease")

# Every defect this module can refuse by name. A refusal must name its defect so the
# repair instruction can state the correction boundary, and so a reader of a refused run
# can tell which rule fired without parsing prose.
SEMANTIC_DEFECTS = (
    "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS",
    "ONE_STEP_OPERATIONAL_WORLD",
    "SINGLE_DRIVER_EXEMPTION_UNGROUNDED",
    "ZERO_ACTOR_WORLD_UNJUSTIFIED",
    "ZERO_ACTOR_CLAIM_CONTRADICTED",
    "DECORATIVE_ACTOR",
    "REPRESENTATION_RECORD_INCOMPLETE",
    "UNCERTAINTY_ALTERNATIVE_UNDESCRIBED",
    "DEGENERATE_FILLER_ALTERNATIVE",
    "TERMINAL_SENSITIVITY_MISDECLARED",
)


def defects_in(errors: list[str]) -> list[str]:
    """Which named defects a validation result contains, for the refusal's details."""

    return sorted({d for d in SEMANTIC_DEFECTS if any(e.startswith(d) for e in errors)})


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
    """One occupant of the world, with the record of why it is represented at all.

    The five representation-scale questions (§7 / CWF-1) are fields rather than prose:
    why it can change the answer, what terminal-relevant state it can alter, what
    information it receives, what authority it holds, and what happens if it is removed.
    A world that cannot answer them for an entity has not decided what that entity is
    doing there.
    """

    name: str  # canonical real-world name
    structural_type: str
    role: str
    representation_scale: str
    decides: bool  # its own decisions can move the outcome (it is an actor)
    authority: str  # ordinary language description of what it may do
    why_material: str  # why it could change the answer
    represents_count: int | None = None
    evidence_claim_ids: tuple[str, ...] = ()
    terminal_state_it_can_change: str = ""  # which terminal-relevant state it can move
    information_received: str = ""  # what this occupant learns, and through what
    if_removed: str = ""  # what the world loses if it is deleted


@dataclass(frozen=True)
class ExcludedCandidate:
    """Someone or something the evidence names that the world deliberately leaves out.

    The exclusion half of the representation-scale record: an omission nobody had to
    justify is indistinguishable from an omission nobody noticed.
    """

    name: str
    why_immaterial: str  # why removing it cannot materially change the answer
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ZeroActorClaim:
    """The D5 admissibility claim for a world with no deciding entity.

    Both halves are required and both are evidence-bearing: that no material human or
    population decision can change the answer, and that the non-agent process is
    causally sufficient on its own.
    """

    no_material_decision: str
    process_sufficiency: str
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SingleDriverExemption:
    """The D4 exemption a plan must claim explicitly to stand on one multiplier.

    A single multiplier may stand in for a whole operating system only when a documented
    empirical model says it may, the parameter's uncertainty is grounded, and the
    threshold-straddling gate is clear. Recording the claim in the plan is the point: a
    reviewer can see that the exemption was taken, and on what basis.
    """

    empirical_model: str
    parameter_uncertainty: str
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
    """One way an unknown could turn out — with what it means, not only what it is.

    A live run's decisive alternative was the filler label ``"other"`` whose own
    grounding read "No specific alternative in evidence", and it carried half the branch
    mass and all of the YES mass. An alternative therefore has to say what state of the
    world it is, why the record does not settle it, what it changes, and how the terminal
    responds under it. Mass follows meaning; an alternative that cannot state its meaning
    is not a possibility the world contains.
    """

    value: Any
    provenance: str
    weight: float | None = None  # None with symmetric_ignorance — never invented
    grounding: str = ""  # what supports this alternative
    evidence_claim_ids: tuple[str, ...] = ()
    meaning: str = ""  # what is true about the world if this alternative holds
    why_unresolved: str = ""  # why the record does not settle whether it holds
    changes: tuple[str, ...] = ()  # subset of ALTERNATIVE_CHANGE_KINDS
    terminal_sensitivity: str = ""  # one of TERMINAL_SENSITIVITIES

    def weight_is_ungrounded(self) -> bool:
        """True when nothing supports this alternative's probability."""

        return self.weight is None or self.provenance in UNGROUNDED_WEIGHT_PROVENANCES


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
    excluded_candidates: tuple[ExcludedCandidate, ...] = ()
    zero_actor_claim: ZeroActorClaim | None = None
    single_driver_exemption: SingleDriverExemption | None = None


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


def _any_content(d: dict[str, Any]) -> bool:
    """Whether an optional block was actually filled in, or merely echoed empty."""

    return any(v for v in d.values() if not isinstance(v, (dict, list)) or v)


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
                terminal_state_it_can_change=_s(e, "terminal_state_it_can_change", w, errors),
                information_received=_s(e, "information_received", w, errors),
                if_removed=_s(e, "if_removed", w, errors),
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
            raw_changes = alt.get("changes") or []
            change_kinds = (
                tuple(str(x) for x in raw_changes) if isinstance(raw_changes, list) else ()
            )
            for kind in change_kinds:
                if kind not in ALTERNATIVE_CHANGE_KINDS:
                    errors.append(f"{wa}: changes {kind!r} not in {list(ALTERNATIVE_CHANGE_KINDS)}")
            sensitivity = _s(alt, "terminal_sensitivity", wa, errors)
            if sensitivity and sensitivity not in TERMINAL_SENSITIVITIES:
                errors.append(
                    f"{wa}: terminal_sensitivity {sensitivity!r} not in "
                    f"{list(TERMINAL_SENSITIVITIES)}"
                )
            alts.append(
                SemanticAlternative(
                    value=alt.get("value"),
                    provenance=prov,
                    weight=float(wt) if isinstance(wt, (int, float)) else None,
                    grounding=_s(alt, "grounding", wa, errors),
                    evidence_claim_ids=_ids(alt),
                    meaning=_s(alt, "meaning", wa, errors),
                    why_unresolved=_s(alt, "why_unresolved", wa, errors),
                    changes=change_kinds,
                    terminal_sensitivity=sensitivity,
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

    excluded: list[ExcludedCandidate] = []
    for i, x in enumerate(data.get("excluded_candidates") or []):
        w = f"excluded_candidates[{i}]"
        if not isinstance(x, dict):
            errors.append(f"{w}: must be an object")
            continue
        excluded.append(
            ExcludedCandidate(
                name=_s(x, "name", w, errors),
                why_immaterial=_s(x, "why_immaterial", w, errors),
                evidence_claim_ids=_ids(x),
            )
        )

    # Both of these blocks are conditionally required, so the schema shows them to every
    # planner and most plans should leave them empty. An empty object is therefore "not
    # claimed", never "claimed with nothing in it" — otherwise a model that echoes the
    # schema's own keys would have every actor-bearing world refused for contradicting a
    # claim it never made.
    zac_raw = data.get("zero_actor_justification")
    zero_actor = None
    if isinstance(zac_raw, dict) and _any_content(zac_raw):
        zero_actor = ZeroActorClaim(
            no_material_decision=_s(
                zac_raw, "no_material_decision", "zero_actor_justification", errors
            ),
            process_sufficiency=_s(
                zac_raw, "process_sufficiency", "zero_actor_justification", errors
            ),
            evidence_claim_ids=_ids(zac_raw),
        )

    ex_raw = data.get("single_multiplier_exemption")
    exemption = None
    if isinstance(ex_raw, dict) and _any_content(ex_raw):
        exemption = SingleDriverExemption(
            empirical_model=_s(ex_raw, "empirical_model", "single_multiplier_exemption", errors),
            parameter_uncertainty=_s(
                ex_raw, "parameter_uncertainty", "single_multiplier_exemption", errors
            ),
            evidence_claim_ids=_ids(ex_raw),
        )

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
        excluded_candidates=tuple(excluded),
        zero_actor_claim=zero_actor,
        single_driver_exemption=exemption,
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


# ---------------------------------------------------------------------------
# The plan's own arithmetic, evaluated statically
#
# Every gate below that talks about numbers derives them from the plan itself: the
# declared initial values, the declared changes, and the declared terminal threshold.
# No number, threshold, entity or domain is named in this file. The point of computing
# rather than pattern-matching is that a defect like "two invented factors straddle the
# break-even" has no textual signature at all — it exists only in the arithmetic.
# ---------------------------------------------------------------------------


def _numeric(value: Any) -> float | None:
    """A value read as a number, or None. Booleans are not quantities."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _eval_value(v: SemanticValue | None, env: dict[str, float]) -> float | None:
    """A SemanticValue evaluated against known state values, or None if undetermined."""

    if v is None:
        return None
    if v.kind == "literal":
        return _numeric(v.literal)
    if v.kind == "state":
        return env.get(v.state) if v.state else None
    parts = [_eval_value(p, env) for p in v.parts]
    if any(p is None for p in parts):
        return None
    numbers = [p for p in parts if p is not None]
    if v.kind == "sum":
        return sum(numbers)
    if v.kind == "product":
        out = 1.0
        for n in numbers:
            out *= n
        return out
    return None


def _mechanism_changes(plan: SemanticPlan) -> list[SemanticChange]:
    """Every non-agent change, in the order the world would apply it.

    Dated occurrences first in chronological order, then dependency-chained ones in
    declaration order (a chained occurrence follows its predecessor by construction).
    Actor affordances are excluded on purpose: whether an actor acts is not statically
    known, and assuming it does would let a gate refuse a world for a decision nobody
    has made.
    """

    items: list[tuple[tuple[int, str, int], SemanticChange]] = []
    for i, p in enumerate(plan.processes):
        if p.kind == "actor_moment":
            continue
        for j, o in enumerate(p.occurrences):
            dated = isinstance(o.at, str) and bool(o.at)
            key = (0 if dated else 1, o.at if dated and o.at else "", i * 1000 + j)
            for c in o.changes:
                items.append((key, c))
    items.sort(key=lambda kv: kv[0])
    return [c for _, c in items]


def _forward_state_values(plan: SemanticPlan, draw: dict[str, float]) -> dict[str, float]:
    """Run the plan's own non-agent arithmetic forward from a given uncertainty draw.

    A state with no numeric initial and no computable writer simply never appears in the
    result — undetermined stays undetermined, so a gate that needs a number declines to
    fire rather than inventing a zero.
    """

    env = {s.name: n for s in plan.states if (n := _numeric(s.initial)) is not None}
    env.update(draw)
    for c in _mechanism_changes(plan):
        if c.op == "set":
            value = _eval_value(c.value, env)
            if value is not None:
                env[c.target] = value
        elif c.op in ("increase", "decrease"):
            amount = _eval_value(c.amount, env)
            if amount is None or c.target not in env:
                continue
            env[c.target] = env[c.target] + (amount if c.op == "increase" else -amount)
    return env


def _satisfies(comparison: str, value: float, threshold: float) -> bool:
    if comparison == "greater_than":
        return value > threshold
    if comparison == "greater_or_equal":
        return value >= threshold
    if comparison == "less_than":
        return value < threshold
    if comparison == "less_or_equal":
        return value <= threshold
    return value == threshold


def _quantity_comparisons(q: TerminalQuery) -> list[tuple[str, str, SemanticValue | None]]:
    """Every (state, comparison, threshold) the terminal holds a quantity against."""

    out: list[tuple[str, str, SemanticValue | None]] = []
    if q.form == "quantity_comparison" and q.state and q.comparison:
        out.append((q.state, q.comparison, q.threshold))
    for part in q.parts:
        out.extend(_quantity_comparisons(part))
    return out


def _terminal_events(q: TerminalQuery) -> set[str]:
    """Event names the terminal's own resolution counts."""

    out: set[str] = set()
    if q.form == "event_exists" and q.event:
        out.add(q.event)
    if q.form == "record_count" and q.record_event:
        out.add(q.record_event)
    for part in q.parts:
        out |= _terminal_events(part)
    return out


def _terminal_states(q: TerminalQuery) -> set[str]:
    """Every state the terminal reads, directly or through a threshold."""

    out: set[str] = set()
    if q.state:
        out.add(q.state)
    if q.threshold is not None:
        out |= q.threshold.states_read()
    for part in q.parts:
        out |= _terminal_states(part)
    return out


def _all_changes(plan: SemanticPlan) -> list[SemanticChange]:
    return [c for a in plan.affordances for c in a.changes] + [
        c for p in plan.processes for o in p.occurrences for c in o.changes
    ]


def terminal_relevant_states(plan: SemanticPlan) -> set[str]:
    """Every state the terminal reads, plus every state those are computed from.

    The public name for "terminal-relevant": the representation record and the
    decorative-actor rule mean the same thing by it, and so should any reader.
    """

    return _lineage_states(plan, _terminal_states(plan.terminal))


def _lineage_states(plan: SemanticPlan, seeds: set[str]) -> set[str]:
    """Every state whose value can reach ``seeds`` through the plan's own changes.

    The same closure the lowerer uses to decide what an UNKNOWN can undetermine, reused
    here because "terminal-relevant" means exactly this: a state that something the
    terminal reads is computed from, however many hops away.
    """

    changes = _all_changes(plan)
    relevant = set(seeds)
    while True:
        grown = set(relevant)
        for c in changes:
            if c.op in WRITE_OPS and c.target in relevant:
                if c.value is not None:
                    grown |= c.value.states_read()
                if c.amount is not None:
                    grown |= c.amount.states_read()
        if grown == relevant:
            return relevant
        relevant = grown


def _draw_combinations(
    uncertainties: list[SemanticUncertainty], *, cap: int = 16
) -> list[dict[str, float]]:
    """Assignments of the other uncertainties, so one variable can be varied against them.

    Bounded: beyond the cap only the first numeric alternative of each is used, which
    keeps a wide plan from turning a static check into an exponential one. Under-, never
    over-refusing: fewer background assignments can only mean fewer straddles found.
    """

    per_variable: list[list[tuple[str, float]]] = []
    for u in uncertainties:
        values = [
            (u.affects_state, n) for a in u.alternatives if (n := _numeric(a.value)) is not None
        ]
        if values:
            per_variable.append(values)
    if not per_variable:
        return [{}]
    total = 1
    for values in per_variable:
        total *= len(values)
    if total > cap:
        return [{name: value for values in per_variable for name, value in values[:1]}]
    combos: list[dict[str, float]] = [{}]
    for values in per_variable:
        combos = [{**combo, name: value} for combo in combos for name, value in values]
    return combos


def _all_citations(plan: SemanticPlan) -> list[tuple[str, tuple[str, ...]]]:
    """Every (location, claim ids) pair the plan carries.

    Enumerated exhaustively rather than by rule, because the point is to leave no
    citation unchecked: entities, states, events, affordances, processes and their
    occurrences, uncertainty alternatives, the resolution rule, and world facts.
    """

    out: list[tuple[str, tuple[str, ...]]] = [
        ("resolution", plan.resolution_evidence_ids),
    ]
    out += [(f"entity {e.name!r}", e.evidence_claim_ids) for e in plan.entities]
    out += [(f"state {s.name!r}", s.evidence_claim_ids) for s in plan.states]
    out += [(f"event {e.name!r}", e.evidence_claim_ids) for e in plan.events]
    out += [(f"affordance {a.name!r}", a.evidence_claim_ids) for a in plan.affordances]
    for p in plan.processes:
        out.append((f"process {p.name!r}", p.evidence_claim_ids))
    for u in plan.uncertainties:
        for i, alt in enumerate(u.alternatives):
            out.append((f"uncertainty {u.name!r} alternative[{i}]", alt.evidence_claim_ids))
    for i, (_text, ids) in enumerate(plan.world_facts):
        out.append((f"world_fact[{i}]", ids))
    for x in plan.excluded_candidates:
        out.append((f"excluded candidate {x.name!r}", x.evidence_claim_ids))
    if plan.zero_actor_claim is not None:
        out.append(("zero_actor_justification", plan.zero_actor_claim.evidence_claim_ids))
    if plan.single_driver_exemption is not None:
        out.append(("single_multiplier_exemption", plan.single_driver_exemption.evidence_claim_ids))
    return [(where, ids) for where, ids in out if ids]


# ---------------------------------------------------------------------------
# Causal-world fidelity gates (Phase 3). Each names its defect and its correction
# boundary, because a refusal a revision round cannot act on is a dead end.
# ---------------------------------------------------------------------------


def _representation_record_errors(plan: SemanticPlan) -> list[str]:
    """CWF-1: every occupant answers the five representation questions, in fields.

    A world is a claim about who matters. The claim is auditable only if each included
    entity says why it can change the answer, which terminal-relevant state it can move,
    what it learns, what it may do, and what is lost by deleting it — and if each
    deliberately excluded candidate says why its removal cannot matter.
    """

    errors: list[str] = []
    for e in plan.entities:
        missing = [
            label
            for label, value in (
                ("why_material", e.why_material),
                ("authority", e.authority),
                ("terminal_state_it_can_change", e.terminal_state_it_can_change),
                ("information_received", e.information_received),
                ("if_removed", e.if_removed),
            )
            if not value.strip()
        ]
        if missing:
            errors.append(
                f"REPRESENTATION_RECORD_INCOMPLETE: entity {e.name!r} is in the world "
                f"without answering {missing} — state, for this entity, why it can "
                "change the answer, which terminal-relevant state it can alter, what "
                "information it receives, what authority it holds, and what the world "
                "loses if it is removed. Correction boundary: these fields on this "
                "entity, or remove the entity"
            )
    for x in plan.excluded_candidates:
        if not x.name.strip() or not x.why_immaterial.strip():
            errors.append(
                "REPRESENTATION_RECORD_INCOMPLETE: an excluded_candidates entry names "
                f"{x.name!r} without saying why its removal cannot materially change "
                "the answer. Correction boundary: why_immaterial on that entry"
            )
    return errors


def _alternative_quality_errors(plan: SemanticPlan, cited: Any) -> list[str]:
    """FD-10 / FD-11: an alternative carries meaning, or it carries nothing.

    FD-11's shape was a filler alternative — value ``"other"``, grounding "No specific
    alternative in evidence" — carrying half the mass and all of the YES mass. The
    universal rule is not a word list: an alternative must say what it means, why the
    record leaves it open, what it changes about structure / actor state / process
    state, and how the terminal responds under it; and an alternative that nothing
    supports may not be the one the planner itself declares decisive.
    """

    errors: list[str] = []
    for u in plan.uncertainties:
        for i, alt in enumerate(u.alternatives):
            where = f"uncertainty {u.name!r} alternative[{i}] ({alt.value!r})"
            missing = [
                label
                for label, present in (
                    ("meaning", bool(alt.meaning.strip())),
                    ("why_unresolved", bool(alt.why_unresolved.strip())),
                    ("changes", bool(alt.changes)),
                    ("terminal_sensitivity", bool(alt.terminal_sensitivity.strip())),
                )
                if not present
            ]
            if missing:
                errors.append(
                    f"UNCERTAINTY_ALTERNATIVE_UNDESCRIBED: {where} carries branch mass "
                    f"without {missing} — say what is true about the world under this "
                    "alternative, why the record does not settle it, which of "
                    f"{list(ALTERNATIVE_CHANGE_KINDS)} it changes, and its "
                    f"terminal_sensitivity from {list(TERMINAL_SENSITIVITIES)}. "
                    "Correction boundary: these fields on this alternative, or drop the "
                    "alternative if there is nothing to say"
                )
            if (
                alt.terminal_sensitivity == "decides_the_terminal"
                and not cited(alt.evidence_claim_ids)
                and alt.weight_is_ungrounded()
            ):
                errors.append(
                    f"DEGENERATE_FILLER_ALTERNATIVE: {where} is declared to decide the "
                    "terminal while citing no evidence and carrying a weight nothing "
                    "supports — the answer would be this invented alternative and its "
                    "invented share of the mass. Correction boundary: cite the claims "
                    "that establish this alternative as a real possibility, or give its "
                    "weight a grounded provenance, or remove it and let the mass sit "
                    "with the alternatives the record does support"
                )
    return errors


def _actor_admissibility_errors(plan: SemanticPlan, cited: Any) -> list[str]:
    """CWF-5 / D5: no deciding entity is admissible only on stated, cited grounds; and
    an entity whose decisions cannot move anything terminal-relevant is decoration.

    Both directions are the same rule read from either end. A world may contain no
    actors only when the evidence says no material human or population decision can
    change the answer AND the non-agent process is causally sufficient; and an actor may
    be present only when its decisions can reach the terminal — directly, or by reaching
    someone whose decisions can.
    """

    errors: list[str] = []
    deciders = {e.name for e in plan.entities if e.decides}
    claim = plan.zero_actor_claim

    if not deciders:
        if (
            claim is None
            or not claim.no_material_decision.strip()
            or not claim.process_sufficiency.strip()
            or not cited(claim.evidence_claim_ids)
        ):
            errors.append(
                "ZERO_ACTOR_WORLD_UNJUSTIFIED: this world contains no deciding entity, "
                "so nobody's choice can change the answer — that is admissible only "
                "when the plan says so with evidence. Declare "
                "zero_actor_justification {no_material_decision, process_sufficiency, "
                "evidence_claim_ids}: which human or population decisions could bear on "
                "this outcome and why the record shows none of them can move it, and "
                "why the non-agent process alone is causally sufficient. Correction "
                "boundary: that justification with cited claims, or the deciding "
                "entities the world is missing"
            )
        if plan.expected_participants:
            errors.append(
                "ZERO_ACTOR_CLAIM_CONTRADICTED: the plan declares "
                f"{plan.expected_participants} decision-relevant participants the "
                "evidence names, and then represents none of them as a deciding entity "
                "— a world cannot both need those decisions and have none. Correction "
                "boundary: model those participants as deciding entities with their "
                "affordances and dated occasions, or lower expected_participants to "
                "what the evidence actually requires"
            )
        return errors

    if claim is not None:
        errors.append(
            "ZERO_ACTOR_CLAIM_CONTRADICTED: the plan claims no material decision can "
            f"change the answer while modelling {sorted(deciders)} as deciding "
            "entities. Correction boundary: drop zero_actor_justification, or drop the "
            "deciding entities and justify the actor-free world"
        )

    relevant = _lineage_states(plan, _terminal_states(plan.terminal))
    counted = _terminal_events(plan.terminal)
    events_by_name = {ev.name: ev for ev in plan.events}
    direct: set[str] = set()
    informs: dict[str, set[str]] = {}
    for a in plan.affordances:
        if a.actor not in deciders:
            continue
        for c in a.changes:
            writes_terminal_state = c.op in WRITE_OPS and c.target in relevant
            records_counted_event = c.op == "record_event" and c.target in counted
            if writes_terminal_state or records_counted_event:
                direct.add(a.actor)
            elif c.op == "record_event":
                ev = events_by_name.get(c.target)
                if ev is None:
                    continue
                audience = {who for _role, who in ev.participants}
                if not audience and ev.visibility == "public":
                    audience = set(deciders)
                informs.setdefault(a.actor, set()).update(audience - {a.actor})
            elif c.op == "send":
                informs.setdefault(a.actor, set()).update(set(c.recipients) - {a.actor})

    # An actor who informs an actor who matters, matters. Closed to a fixpoint so a
    # genuine chain of influence is never called decoration.
    material = set(direct)
    changed = True
    while changed:
        changed = False
        for actor, reach in informs.items():
            if actor not in material and reach & material:
                material.add(actor)
                changed = True

    for name in sorted(deciders - material):
        errors.append(
            f"DECORATIVE_ACTOR: {name!r} decides nothing that matters here — none of its "
            "affordances writes a state the terminal reads or is computed from, records "
            "an event the terminal counts, or informs an actor who can. An actor present "
            "without a causal path is decoration that makes the world look alive. "
            "Correction boundary: give this entity the affordance through which its real "
            "authority reaches the outcome, or remove it and record it under "
            "excluded_candidates with why its removal cannot change the answer"
        )
    return errors


def _one_step_operational_errors(plan: SemanticPlan, cited: Any) -> list[str]:
    """CWF-3 / D4: a terminal quantity produced in one non-agent step is not a simulation.

    A live run computed the whole answer as one cited quarter multiplied by one invented
    factor: one change, no intermediate state, nothing happening across the window. The
    universal test is structural, not numeric — exactly one non-agent change writes the
    terminal quantity, and its inputs are things nothing else in the world produces, so
    there is no causal process through time at all. A single multiplier stands in for a
    system only under a documented, cited empirical model, with the parameter's
    uncertainty grounded (and the straddling gate clear on its own).

    Deliberately no wider than D4's two forms: one final set-the-total, or one uncertain
    factor scaling a base. One scheduled accumulation of a fully grounded amount is a
    thin world, not a false one — no branch weight is doing work in it — and refusing it
    would only extract detail the record does not have.
    """

    errors: list[str] = []
    all_changes = _all_changes(plan)
    for state, _comparison, threshold in _quantity_comparisons(plan.terminal):
        actor_writers = [
            a.name
            for a in plan.affordances
            for c in a.changes
            if c.op in WRITE_OPS and c.target == state
        ]
        if actor_writers:
            continue  # an actor-produced quantity is judged by the actor gates
        mechanism = [
            (p, c)
            for p in plan.processes
            if p.kind != "actor_moment"
            for o in p.occurrences
            for c in o.changes
            if c.op in WRITE_OPS and c.target == state
        ]
        if not mechanism:
            continue  # no producer at all, or a cited factual resolution — judged above
        produced_elsewhere = {c.target for c in all_changes if c.op in WRITE_OPS} - {state}
        inputs: set[str] = set()
        for _p, c in mechanism:
            if c.value is not None:
                inputs |= c.value.states_read()
            if c.amount is not None:
                inputs |= c.amount.states_read()
        if len(mechanism) > 1 or (inputs & produced_elsewhere):
            continue  # a real progression: several steps, or a produced intermediate
        process_name, change = mechanism[0]
        # D4 names two forms, and the gate is exactly as wide as they are: one final
        # set-the-total (the environment announcing the answer), or one arbitrary
        # multiplier (a draw scaling a base). A single scheduled accumulation whose whole
        # amount is grounded — a contracted transfer of a cited size — is neither: no
        # branch weight is doing any work in it, and refusing it would demand detail the
        # record does not contain.
        uncertain_states = {u.affects_state for u in plan.uncertainties}
        if change.op != "set" and not (inputs & uncertain_states):
            continue
        exemption = plan.single_driver_exemption
        claimed = (
            exemption is not None
            and bool(exemption.empirical_model.strip())
            and bool(exemption.parameter_uncertainty.strip())
            and cited(exemption.evidence_claim_ids)
        )
        if not claimed:
            errors.append(
                f"ONE_STEP_OPERATIONAL_WORLD: the terminal quantity {state!r} is produced "
                f"by a single non-agent change ({change.op}) in process "
                f"{process_name.name!r}, reading only inputs nothing in this world "
                f"produces ({sorted(inputs)}) — that is a reported figure, not an "
                "operating process. Model what actually makes the quantity: grounded "
                "inputs, at least one intermediate state the mechanism updates inside "
                "the window, and occurrences spread across the real causal period, so "
                "the total is reached rather than announced. If one multiplier genuinely "
                "stands for the system, declare single_multiplier_exemption "
                "{empirical_model, parameter_uncertainty, evidence_claim_ids} citing the "
                "documented model. Correction boundary: the production process for "
                f"{state!r}, or that exemption"
            )
            continue
        lineage = _lineage_states(
            plan, {state} | (threshold.states_read() if threshold is not None else set())
        )
        ungrounded = sorted(
            u.name
            for u in plan.uncertainties
            if u.affects_state in lineage
            and any(
                not cited(a.evidence_claim_ids) and a.weight_is_ungrounded() for a in u.alternatives
            )
        )
        if ungrounded:
            errors.append(
                "SINGLE_DRIVER_EXEMPTION_UNGROUNDED: the plan claims the single-"
                f"multiplier exemption for {state!r}, but the parameter it rests on is "
                f"not grounded — {ungrounded} carries alternatives with no cited "
                "evidence and no supported weight, so the documented model is being used "
                "to license invented numbers. Correction boundary: cite the evidence "
                "that establishes those alternative values or their distribution, or "
                "withdraw the exemption and model the process"
            )
    return errors


def _straddling_errors(plan: SemanticPlan, cited: Any) -> list[str]:
    """CWF-4 / D3: ungrounded alternatives may not sit on both sides of the threshold.

    The failure this exists to end: a terminal comparing a quantity against a threshold,
    where the quantity is a cited anchor times an uncertainty whose two alternatives were
    invented, one on each side of the break-even. The published probability was then the
    count of invented branches and nothing else — had the planner written a different
    number, the answer would have been different with nothing else changed.

    Everything here is computed from the plan's own arithmetic: the declared initial
    values, the declared changes, the declared threshold. The break-even is *found*, by
    bisecting the plan's own production function between the two straddling draws, so the
    refusal can name the exact boundary the invented numbers were placed around without
    any number, domain or question family appearing in this file.
    """

    errors: list[str] = []
    if not plan.uncertainties:
        return errors

    def ungrounded(alt: SemanticAlternative) -> bool:
        # Nothing supports the value, and nothing supports its probability. Either one
        # alone is legal: a cited value under honest symmetric-ignorance weights is the
        # shape of not knowing, and the runtime prices it as bounds.
        return not cited(alt.evidence_claim_ids) and alt.weight_is_ungrounded()

    for state, comparison, threshold in _quantity_comparisons(plan.terminal):
        seeds = {state} | (threshold.states_read() if threshold is not None else set())
        lineage = _lineage_states(plan, seeds)
        feeders = [u for u in plan.uncertainties if u.affects_state in lineage]
        for u in feeders:
            numeric_alts = [(a, n) for a in u.alternatives if (n := _numeric(a.value)) is not None]
            if len(numeric_alts) < 2:
                continue
            others = [o for o in feeders if o.name != u.name]
            for base in _draw_combinations(others):
                evaluated: list[tuple[SemanticAlternative, float, bool]] = []
                for alt, value in numeric_alts:
                    env = _forward_state_values(plan, {**base, u.affects_state: value})
                    quantity = env.get(state)
                    limit = _eval_value(threshold, env) if threshold is not None else None
                    if quantity is None or limit is None:
                        continue
                    evaluated.append((alt, value, _satisfies(comparison, quantity, limit)))
                sides = {side for _a, _v, side in evaluated}
                if len(sides) < 2:
                    continue
                pair = _straddling_pair(evaluated)
                if pair is None:
                    continue
                low, high = pair
                if ungrounded(low[0]) and ungrounded(high[0]):
                    break_even = _break_even(
                        plan, state, comparison, threshold, u.affects_state, low, high, base
                    )
                    errors.append(
                        "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS: uncertainty "
                        f"{u.name!r} offers {low[1]:.6g} and {high[1]:.6g} — neither "
                        "value cited and neither weight supported — and the plan's own "
                        f"arithmetic puts them on opposite sides of the terminal on "
                        f"{state!r}"
                        + (
                            f" (break-even draw ≈ {break_even:.6g})"
                            if break_even is not None
                            else ""
                        )
                        + ". The answer would be the choice of those two numbers and the "
                        "fact that there are two of them, not anything the world does. "
                        "Correction boundary: the alternatives' VALUES and their "
                        "citations — anchor each in cited evidence (a published range, a "
                        "recorded distribution, a stated forecast), or replace the "
                        "invented split with the mechanism that produces the quantity so "
                        "the crossing follows from the process. Do not move the "
                        "threshold and do not restate the terminal"
                    )
                    break
                misdeclared = sorted(
                    {
                        f"{a.value!r}"
                        for a, _v, _s in evaluated
                        if a.terminal_sensitivity == "immaterial_to_the_terminal"
                    }
                )
                if misdeclared:
                    errors.append(
                        f"TERMINAL_SENSITIVITY_MISDECLARED: uncertainty {u.name!r} "
                        f"declares {misdeclared} immaterial to the terminal, but the "
                        "plan's own arithmetic has its alternatives resolving "
                        f"{state!r} on both sides of the threshold. Correction "
                        "boundary: the terminal_sensitivity declaration on those "
                        "alternatives"
                    )
                break
    return errors


def _straddling_pair(
    evaluated: list[tuple[SemanticAlternative, float, bool]],
) -> tuple[tuple[SemanticAlternative, float, bool], tuple[SemanticAlternative, float, bool]] | None:
    """The adjacent pair of draws whose outcomes differ — the crossing itself."""

    ordered = sorted(evaluated, key=lambda item: item[1])
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left[2] != right[2]:
            return left, right
    return None


def _break_even(
    plan: SemanticPlan,
    state: str,
    comparison: str,
    threshold: SemanticValue | None,
    variable: str,
    low: tuple[SemanticAlternative, float, bool],
    high: tuple[SemanticAlternative, float, bool],
    base: dict[str, float],
) -> float | None:
    """The draw at which the plan's own production function crosses its own threshold.

    Found by bisection on the plan's arithmetic rather than by solving a form we assumed
    — so it is right for a product, a sum, a chain of accumulations, or anything else the
    universal value forms can express.
    """

    lo, hi, side_at_lo = low[1], high[1], low[2]
    for _ in range(64):
        mid = (lo + hi) / 2.0
        env = _forward_state_values(plan, {**base, variable: mid})
        quantity = env.get(state)
        limit = _eval_value(threshold, env) if threshold is not None else None
        if quantity is None or limit is None:
            return None
        if _satisfies(comparison, quantity, limit) == side_at_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


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

    # Every citation in the plan, not just the ones a producer rule happens to read.
    # A live Tesla plan cited 'c-83b62bb5759f' — the real id with two digits transposed
    # — on a world_fact and on the alternative backing its entire NO branch. The id
    # existed in no evidence store, it was lowered into the executable as a cited fact,
    # and coverage still reported complete. A fabricated citation is indistinguishable
    # from a real one downstream, so it must die here.
    if known is not None:
        unknown: dict[str, set[str]] = {}
        for where, ids in _all_citations(plan):
            for cid in ids:
                if cid not in known:
                    unknown.setdefault(cid, set()).add(where)
        for cid in sorted(unknown):
            errors.append(
                f"cites claim {cid!r}, which is not in the evidence store, at "
                f"{sorted(unknown[cid])} — a citation to a claim that does not exist "
                "grounds nothing"
            )

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

    # The causal-world fidelity gates. They run last because they read the plan as a
    # world rather than as a set of references — a plan whose references do not resolve
    # is corrected above first, and these checks are written to decline (never to
    # invent) when a name or a number is missing.
    errors += _representation_record_errors(plan)
    errors += _alternative_quality_errors(plan, cited)
    errors += _actor_admissibility_errors(plan, cited)
    errors += _one_step_operational_errors(plan, cited)
    errors += _straddling_errors(plan, cited)
    return errors
