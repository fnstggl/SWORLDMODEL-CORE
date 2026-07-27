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

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

# What KIND of thing a state is, independent of the datatype it is stored in. The
# vocabulary had only datatypes, so a physical quantity and a thermometer reading were
# the same object and the only way to change either was bare arithmetic on a field — no
# conservation anywhere. A live run modelled vehicle inventory that way and ended the
# quarter at -48,000 vehicles with an order backlog of -73,866: the arithmetic was
# impossible and nothing in the language could say so.
#
#   stock — a conserved physical quantity held somewhere: water in a reservoir, grain in
#           an elevator, berths at a quay, beds on a ward, barrels in a tank, ballots in
#           a box. It cannot go below zero, and where it can grow it grows into a
#           declared capacity. Lowering puts it on the runtime's resource machinery, so
#           an overdrawing move is refused by the executor rather than recorded.
#   flow  — a per-period rate a recurring process applies to stocks (declares its
#           period, e.g. P1W). A rate is not a total: what it produces depends on how
#           often it is applied, which is why a flow must say how often that is.
#   level — a reading, indicator, boolean, category or running tally: exactly today's
#           unconstrained field. The default, because most states are readings.
STATE_KINDS = ("level", "stock", "flow")

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
    "UNCONSERVED_PHYSICAL_STOCK",
    "STOCK_DECLARATION_INCOMPLETE",
    "STOCK_DRAINED_WITHOUT_INFLOW",
    "ACCUMULATING_QUANTITY_UNDECLARED",
    "ACCUMULATOR_WITHOUT_AN_ORIGIN",
    "FLOW_PERIOD_UNDECLARED",
    "UNIT_UNDECLARED",
    "DIMENSIONAL_MISMATCH",
    "DURATION_MISUSED",
    "RECURRENCE_DECLARATION_INVALID",
    "UNDER_ENUMERATED_CADENCE",
    "OVER_ENUMERATED_CADENCE",
)


def defects_in(errors: list[str]) -> list[str]:
    """Which named defects a validation result contains, for the refusal's details."""

    return sorted({d for d in SEMANTIC_DEFECTS if any(e.startswith(d) for e in errors)})


# ---------------------------------------------------------------------------
# Cadence: ISO-8601 periods, and the deterministic enumeration they stand for
#
# A cadence is a fact about the world ("deliveries go out weekly"), and the list of
# dates it implies is arithmetic. Splitting them is the whole point: a planner that
# hand-writes the dates writes two of them for a ten-week quarter and the published
# number becomes an artifact of its typing. Here the planner declares the period and
# code enumerates every firing — bounded, ordered and identical on every run.
# ---------------------------------------------------------------------------

# Deliberately the fixed-arity subset plus calendar months/years. Months are stepped on
# the calendar (with day clamping) rather than approximated, so enumeration stays exact;
# only the *ordering* of two periods uses an average month, and that is used solely to
# pick the tightest cadence among several.
_DURATION = re.compile(
    r"^P(?=\d|T\d)(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?"
    r"(?:T(?=\d)(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
)

_AVERAGE_MONTH_SECONDS = 30.436875 * 86400.0

# The most firings one declared cadence may generate. A period of PT1S across a quarter
# is not a cadence, it is an accident, and generating it would build a spec no reviewer
# could read and no engine should schedule. Refused by name rather than truncated.
RECURRENCE_OCCURRENCE_CAP = 500


@dataclass(frozen=True)
class Period:
    """A calendar period: whole months (stepped on the calendar) plus a fixed offset."""

    months: int
    seconds: float
    text: str

    def is_positive(self) -> bool:
        return self.months > 0 or self.seconds > 0

    def step(self, when: datetime) -> datetime:
        """The next firing after ``when``. Months move on the calendar and clamp the day
        (31 January + P1M is 28/29 February), so enumeration never invents a date."""

        out = when
        if self.months:
            total = (out.year * 12 + (out.month - 1)) + self.months
            year, month = divmod(total, 12)
            month += 1
            day = min(out.day, _days_in_month(year, month))
            out = out.replace(year=year, month=month, day=day)
        if self.seconds:
            out = out + timedelta(seconds=self.seconds)
        return out

    def approx_seconds(self) -> float:
        """For ORDERING two periods only — never for enumerating one."""

        return self.months * _AVERAGE_MONTH_SECONDS + self.seconds


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (datetime(year, month + 1, 1) - datetime(year, month, 1)).days


def parse_period(text: str) -> Period | None:
    """An ISO-8601 duration, or None when it is not one."""

    if not isinstance(text, str) or not text.strip():
        return None
    m = _DURATION.match(text.strip())
    if m is None:
        return None
    years, months, weeks, days, hours, minutes, seconds = m.groups()
    total_months = int(years or 0) * 12 + int(months or 0)
    total_seconds = (
        float(weeks or 0) * 7 * 86400.0
        + float(days or 0) * 86400.0
        + float(hours or 0) * 3600.0
        + float(minutes or 0) * 60.0
        + float(seconds or 0)
    )
    return Period(months=total_months, seconds=total_seconds, text=text.strip())


def period_ratio(duration: Period, per: Period) -> float:
    """How many of ``per`` fit in ``duration`` — the scalar a rate is multiplied by.

    Exact when the two are the same sort of period (both calendar months, or both fixed
    offsets), which is the case every real cadence produces. The average-month fallback
    exists only for the mixed case (a P1M rate applied over P10D) and is documented in
    the emitted mapping so nobody has to guess whether a number was exact.
    """

    if duration.months and per.months and not duration.seconds and not per.seconds:
        return duration.months / per.months
    if duration.seconds and per.seconds and not duration.months and not per.months:
        return duration.seconds / per.seconds
    base = per.approx_seconds()
    return duration.approx_seconds() / base if base else 0.0


def duration_scalars(
    parts: tuple[SemanticValue, ...], flows: dict[str, Period]
) -> tuple[dict[int, float], str]:
    """For a product, what scalar each duration part stands for — or why it cannot.

    A duration is only meaningful against a rate: "two weeks" is a number of things only
    once you say a number of *what per week*. So a duration lives inside a product with
    exactly one flow, and its scalar is how many of that flow's periods it contains.
    """

    duration_at = [i for i, p in enumerate(parts) if p.kind == "duration"]
    if not duration_at:
        return {}, ""
    rate_parts = [p for p in parts if p.kind == "state" and p.state in flows]
    if len(rate_parts) != 1:
        return {}, (
            "a duration multiplies exactly one rate, and this product names "
            f"{len(rate_parts)} — a length of time on its own is not a quantity of "
            "anything"
        )
    per = flows[str(rate_parts[0].state)]
    out: dict[int, float] = {}
    for i in duration_at:
        span = parse_period(str(parts[i].literal))
        if span is None or not span.is_positive():
            return {}, f"the duration {parts[i].literal!r} is not a positive ISO-8601 duration"
        out[i] = period_ratio(span, per)
    return out, ""


def firings_between(first: datetime, last: datetime, period: Period) -> int:
    """How many firings of ``period`` a window from ``first`` to ``last`` inclusive holds.

    Counted by stepping, never by dividing, so a monthly cadence over a quarter is three
    and not "3.02". Bounded by the same cap generation uses.
    """

    if not period.is_positive() or last < first:
        return 0
    count = 1
    when = first
    while count <= RECURRENCE_OCCURRENCE_CAP:
        nxt = period.step(when)
        if nxt <= when or nxt > last:
            return count
        when = nxt
        count += 1
    return count


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
    """A value in semantic form: a literal, a named state, a duration, or arithmetic.

    Arithmetic exists so production can be *computed* — a total as a grounded base
    scaled by an uncertain rate — without the model writing a runtime expression tree.

    ``duration`` is the piece the language was missing. Without it a rate and a quantity
    are the same object: "four thousand tonnes a week" could be added straight to a
    tonnage total and nothing could tell that apart from adding one total to another.
    With it, ``rate × duration`` is expressible and therefore checkable, and a quarter's
    output is the rate times the time it ran rather than the rate times however many
    dates somebody typed.
    """

    kind: str  # "literal" | "state" | "duration" | "sum" | "product"
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

    def durations(self) -> list[str]:
        """Every duration literal anywhere in this value, in order."""

        if self.kind == "duration":
            return [str(self.literal)]
        return [d for p in self.parts for d in p.durations()]


@dataclass(frozen=True)
class SemanticChange:
    """One universal change to the world, in semantic terms."""

    op: str
    target: str  # state name, event name, or information description
    value: SemanticValue | None = None  # for set
    amount: SemanticValue | None = None  # for increase/decrease
    recipients: tuple[str, ...] = ()  # for send: entity names
    detail: str = ""  # open-ended meaning (e.g. what the recorded event's value is)
    # For an increase: the stock this quantity comes OUT of. Without it, a tally that
    # counts what a stock gives up counts what the stock was ASKED for rather than what
    # it had — so a depleted elevator still reports a full quarter of shipments and the
    # clamp protects the pile while the answer stays wrong. Naming the source makes the
    # two sides of one movement the same clamped amount.
    drawn_from: str = ""


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
    """One thing about the world that can be true, or hold a value, or hold a quantity.

    ``kind`` is what sort of thing it is (see :data:`STATE_KINDS`) and is the only
    declaration that carries conservation. A ``stock`` names a real quantity somewhere:
    its ``capacity`` is the physical ceiling it grows into, and lowering puts it on the
    runtime's resources so a move that would overdraw it is refused. A ``flow`` is a
    rate and must say over what ``period``. A ``level`` is a reading and is
    unconstrained — which is right for a thermometer and wrong for a warehouse, so a
    level that something draws down has to say, in ``not_a_stock_because``, why this
    particular quantity may legitimately go below zero.
    """

    name: str  # ordinary-language name, unique among states
    owner: str  # entity name, or "world"
    state_type: str
    initial: Any  # literal, or UNKNOWN
    why_material: str
    unit: str = ""
    evidence_claim_ids: tuple[str, ...] = ()
    kind: str = "level"
    capacity: float | None = None  # stock: the physical ceiling it can be filled to
    conserved_floor: float = 0.0  # stock: the level below which it cannot be drawn
    period: str = ""  # flow: ISO-8601 duration the rate is quoted over
    not_a_stock_because: str = ""  # level: why this quantity is not a conserved stock


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
class SemanticRecurrence:
    """A cadence declared once, instead of a calendar typed out by hand.

    ``period`` is an ISO-8601 duration; ``start`` and ``end`` bound the window it runs
    across; ``changes`` are what happens at every firing. Code enumerates the firings —
    the planner never writes the dates, so a weekly process cannot quietly become a
    twice-a-quarter process because two dates were easier to type than thirteen.
    """

    period: str
    start: str
    end: str
    description: str = ""
    changes: tuple[SemanticChange, ...] = ()


@dataclass(frozen=True)
class SemanticProcess:
    """A moment or mechanism through which the world advances.

    actor_moment: a dated occasion at which named actors gain the opportunity to act.
    operational: a non-agent mechanism that produces state (throughput, accumulation).
    scheduled_release: information or data arriving on a calendar.

    A non-agent process states its firings either as ``occurrences`` it enumerates or as
    a ``recurrence`` it declares; ``occurrences_generated`` records that the list below
    was produced from the cadence rather than written by hand, so the two can never be
    confused by a reader or by a gate.
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
    recurrence: SemanticRecurrence | None = None
    occurrences_generated: bool = False


def recurrence_occurrences(rec: SemanticRecurrence) -> tuple[SemanticOccurrence, ...]:
    """Every firing a declared cadence produces, in order.

    Deterministic and bounded: the same declaration always yields the same dates, and a
    cadence that would exceed :data:`RECURRENCE_OCCURRENCE_CAP` yields nothing at all
    rather than a truncated calendar that would silently under-run its own window (the
    validator refuses it by name). An unreadable period or window likewise yields
    nothing — the refusal belongs to the validator, which can say which field is wrong.
    """

    period = parse_period(rec.period)
    start, end = _parse_when(rec.start), _parse_when(rec.end)
    if period is None or not period.is_positive() or start is None or end is None:
        return ()
    if end < start or not rec.changes:
        return ()
    if firings_between(start, end, period) > RECURRENCE_OCCURRENCE_CAP:
        return ()
    out: list[SemanticOccurrence] = []
    when = start
    while when <= end and len(out) < RECURRENCE_OCCURRENCE_CAP:
        label = rec.description or "recurring firing"
        out.append(
            SemanticOccurrence(
                description=f"{label} ({when.isoformat()}, every {rec.period})",
                changes=rec.changes,
                at=when.isoformat(),
            )
        )
        nxt = period.step(when)
        if nxt <= when:
            break
        when = nxt
    return tuple(out)


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
        if kind == "duration":
            text = obj.get("value") or obj.get("duration")
            if not isinstance(text, str) or parse_period(text) is None:
                errors.append(
                    f"{where}: a duration's value must be an ISO-8601 duration "
                    f"(P1D, P1W, PT6H), got {text!r}"
                )
                return None
            return SemanticValue(kind="duration", literal=text)
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
        drawn_from=_s(obj, "drawn_from", where, errors),
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
        skind = _s(s, "kind", w, errors, default="level") or "level"
        if skind not in STATE_KINDS:
            errors.append(f"{w}: kind {skind!r} not in {list(STATE_KINDS)}")
            skind = "level"
        raw_cap = s.get("capacity")
        capacity = _numeric(raw_cap)
        if raw_cap is not None and capacity is None:
            errors.append(f"{w}: capacity must be a number or omitted, got {raw_cap!r}")
        raw_floor = s.get("conserved_floor")
        floor = _numeric(raw_floor)
        if raw_floor is not None and floor is None:
            errors.append(f"{w}: conserved_floor must be a number or omitted, got {raw_floor!r}")
        states.append(
            SemanticState(
                name=_s(s, "name", w, errors),
                owner=_s(s, "owner", w, errors, default="world") or "world",
                state_type=stype,
                initial=s.get("initial", UNKNOWN),
                why_material=_s(s, "why_material", w, errors),
                unit=_s(s, "unit", w, errors),
                evidence_claim_ids=_ids(s),
                kind=skind,
                capacity=capacity,
                conserved_floor=floor if floor is not None else 0.0,
                period=_s(s, "period", w, errors),
                not_a_stock_because=_s(s, "not_a_stock_because", w, errors),
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
        raw_rec = p.get("recurrence")
        recurrence: SemanticRecurrence | None = None
        if isinstance(raw_rec, dict) and _any_content(raw_rec):
            wr = f"{w}.recurrence"
            rchanges = [
                c
                for k, ch in enumerate(raw_rec.get("changes") or [])
                if (c := _change(ch, f"{wr}.changes[{k}]", errors)) is not None
            ]
            recurrence = SemanticRecurrence(
                period=_s(raw_rec, "period", wr, errors),
                start=_s(raw_rec, "start", wr, errors),
                end=_s(raw_rec, "end", wr, errors),
                description=_s(raw_rec, "description", wr, errors),
                changes=tuple(rchanges),
            )
        # The cadence is expanded HERE, once, so every gate and the lowerer read one
        # occurrence list and can never disagree about how often this process fires.
        # Hand-written occurrences are left exactly as written — a plan that supplies
        # both is a contradiction the validator names rather than one the parser
        # silently resolves.
        generated = False
        if recurrence is not None and not occurrences:
            expanded = list(recurrence_occurrences(recurrence))
            if expanded:
                occurrences = expanded
                generated = True
        parts = p.get("participants") or []
        allowed = p.get("allowed_affordances") or []
        inputs = p.get("inputs") or []
        processes.append(
            SemanticProcess(
                name=_s(p, "name", w, errors),
                meaning=_s(p, "meaning", w, errors),
                kind=kind,
                recurrence=recurrence,
                occurrences_generated=generated,
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


def flow_periods(plan: SemanticPlan) -> dict[str, Period]:
    """Every declared rate's period, for the arithmetic that reads rate × duration."""

    out: dict[str, Period] = {}
    for s in plan.states:
        if s.kind == "flow" and (p := parse_period(s.period)) is not None and p.is_positive():
            out[s.name] = p
    return out


def _eval_value(
    v: SemanticValue | None,
    env: dict[str, float],
    flows: dict[str, Period] | None = None,
) -> float | None:
    """A SemanticValue evaluated against known state values, or None if undetermined."""

    if v is None:
        return None
    if v.kind == "literal":
        return _numeric(v.literal)
    if v.kind == "state":
        return env.get(v.state) if v.state else None
    if v.kind == "duration":
        # Alone, a length of time is no quantity at all; inside a product it is resolved
        # against the rate it multiplies, below.
        return None
    scalars: dict[int, float] = {}
    if v.kind == "product" and flows:
        scalars, _why = duration_scalars(v.parts, flows)
    parts = [
        scalars[i] if i in scalars else _eval_value(p, env, flows) for i, p in enumerate(v.parts)
    ]
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

    flows = flow_periods(plan)
    env = {s.name: n for s in plan.states if (n := _numeric(s.initial)) is not None}
    env.update(draw)
    for c in _mechanism_changes(plan):
        if c.op == "set":
            value = _eval_value(c.value, env, flows)
            if value is not None:
                env[c.target] = value
        elif c.op in ("increase", "decrease"):
            amount = _eval_value(c.amount, env, flows)
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
                    limit = (
                        _eval_value(threshold, env, flow_periods(plan))
                        if threshold is not None
                        else None
                    )
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
        limit = _eval_value(threshold, env, flow_periods(plan)) if threshold is not None else None
        if quantity is None or limit is None:
            return None
        if _satisfies(comparison, quantity, limit) == side_at_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _time_exponent(
    v: SemanticValue | None, kinds: dict[str, str], where: str, errors: list[str]
) -> int | None:
    """The power of time this value carries: 0 a quantity, -1 a rate, +1 a length of time.

    This is the whole of the dimensional check, and it is deliberately not a check on
    unit *words*. A unit string is written by a model and "tonnes" / "tonne" / "t" are
    the same thing while "fraction", "ratio", "index" and "factor" are all legitimately
    dimensionless multipliers — equating those strings would refuse true worlds and
    catch nothing this does not. What time exponents make unambiguous is the confusion
    that actually shipped: a per-week rate used where a quantity belongs. Nothing in the
    plan can express that once the exponents have to match, because the only way to turn
    a rate into a quantity is to multiply it by a duration, and then the total is the
    rate times the time it ran.
    """

    if v is None:
        return None
    if v.kind == "literal":
        return 0
    if v.kind == "duration":
        return 1
    if v.kind == "state":
        kind = kinds.get(v.state or "")
        if kind is None:
            return None  # an undeclared reference; the reference gate names it
        return -1 if kind == "flow" else 0
    parts = [_time_exponent(p, kinds, where, errors) for p in v.parts]
    if any(p is None for p in parts):
        return None
    powers = [p for p in parts if p is not None]
    if v.kind == "product":
        return sum(powers)
    if v.kind == "sum":
        if len(set(powers)) > 1:
            errors.append(
                f"DIMENSIONAL_MISMATCH: {where} adds together things of different kinds "
                f"— the parts of this sum carry time exponents {sorted(set(powers))}, so "
                "at least one is a rate and at least one is a quantity. Correction "
                "boundary: multiply each rate by the duration it runs for "
                '({"kind": "duration", "value": "P1W"}) so every part of the sum is the '
                "same sort of thing"
            )
            return None
        return powers[0] if powers else 0
    return None


def _describe_exponent(power: int) -> str:
    if power == 0:
        return "a quantity"
    if power == -1:
        return "a rate (a quantity per unit of time)"
    if power == 1:
        return "a length of time"
    return f"something with time exponent {power}"


def _dimension_errors(plan: SemanticPlan) -> list[str]:
    """A rate is not a quantity, and the plan may not pretend otherwise.

    ``unit`` used to be free text nothing read, and ``SemanticValue`` could not express
    rate × duration at all — so "four thousand tonnes a week" added straight into a
    tonnage total was structurally identical to adding one total to another, and no gate
    could tell them apart. That is precisely what a live world did with a weekly delivery
    rate: the quarter's total became the rate times the number of dates the planner
    happened to write, and the published answer was a fact about the schedule rather than
    about the world.

    Once every quantity declares its kind, the exponents do the work: an ``increase`` of a
    stock by a flow does not typecheck, and the only repair is to say how long the rate
    ran for.
    """

    errors: list[str] = []
    kinds = {s.name: s.kind for s in plan.states}
    units = {s.name: s.unit for s in plan.states}
    flows = flow_periods(plan)

    def check_value(value: SemanticValue | None, target: str, where: str, verb: str) -> None:
        if value is None or target not in kinds:
            return
        want = -1 if kinds[target] == "flow" else 0
        got = _time_exponent(value, kinds, where, errors)
        if got is None or got == want:
            return
        rates = sorted(n for n in value.states_read() if kinds.get(n) == "flow")
        hint = ""
        if rates and got < want:
            rate = rates[0]
            hint = (
                f" {rate!r} is quoted per {units.get(rate) or 'unit'} per "
                f"{next((s.period for s in plan.states if s.name == rate), '')}: multiply it "
                'by the time this firing covers ({"kind": "product", "parts": [{"kind": '
                f'"state", "state": "{rate}"}}, {{"kind": "duration", "value": "P1W"}}]}}) '
                "so the amount is a quantity rather than a rate."
            )
        errors.append(
            f"DIMENSIONAL_MISMATCH: {where} {verb} {target!r}, which is "
            f"{_describe_exponent(want)}, with {_describe_exponent(got)}.{hint} "
            "Correction boundary: the value of this change — never the target's kind, and "
            "never the terminal"
        )

    def check_durations(value: SemanticValue | None, where: str) -> None:
        if value is None or not value.durations():
            return
        if value.kind != "product":
            errors.append(
                f"DURATION_MISUSED: {where} uses a length of time outside a product — a "
                "duration is a quantity of nothing until it multiplies a rate. Correction "
                "boundary: put the duration in a product with the rate it applies to"
            )
            return
        _scalars, why = duration_scalars(value.parts, flows)
        if why:
            errors.append(
                f"DURATION_MISUSED: {where}: {why}. Correction boundary: the parts of this product"
            )
        for part in value.parts:
            check_durations(part, where)

    for a in plan.affordances:
        for i, c in enumerate(a.changes):
            where = f"affordance {a.name!r} change[{i}]"
            check_durations(c.value, where)
            check_durations(c.amount, where)
            if c.op == "set":
                check_value(c.value, c.target, where, "sets")
            elif c.op in ("increase", "decrease"):
                check_value(c.amount, c.target, where, f"{c.op}s")
    for p in plan.processes:
        for j, o in enumerate(p.occurrences):
            for i, c in enumerate(o.changes):
                where = f"process {p.name!r} occurrence[{j}] change[{i}]"
                check_durations(c.value, where)
                check_durations(c.amount, where)
                if c.op == "set":
                    check_value(c.value, c.target, where, "sets")
                elif c.op in ("increase", "decrease"):
                    check_value(c.amount, c.target, where, f"{c.op}s")

    for state, _comparison, threshold in _quantity_comparisons(plan.terminal):
        if threshold is None or state not in kinds:
            continue
        where = f"terminal on {state!r}"
        check_durations(threshold, where)
        want = -1 if kinds[state] == "flow" else 0
        got = _time_exponent(threshold, kinds, where, errors)
        if got is not None and got != want:
            errors.append(
                f"DIMENSIONAL_MISMATCH: {where} compares {_describe_exponent(want)} "
                f"against {_describe_exponent(got)} — the two sides of the question are "
                "not the same sort of thing. Correction boundary: the threshold"
            )
    return errors


def _state_kind_errors(plan: SemanticPlan) -> list[str]:
    """FD-24: a physical quantity must say it is one, and then it has a floor.

    The vocabulary had only datatypes, so a warehouse and a thermometer were the same
    object and the only way to change either was bare arithmetic on a field. A live world
    drained an inventory forty-eight thousand units below zero across a quarter and
    resolved every branch on the result: the arithmetic was impossible and nothing in
    the language could say so.

    The rule is about shape, not subject. A quantity something draws down is a stock — a
    reservoir, an elevator, a berth queue, a ward's beds — unless the plan says why this
    particular quantity may legitimately go below its floor: a net position, a spread, a
    difference. A stock declares where it starts, the floor it cannot be drawn through
    and, if anything adds to it, the capacity it fills to; lowering then clamps every
    move at those bounds and records what could not be moved.
    """

    errors: list[str] = []
    changes = _all_changes(plan)
    decreased = {c.target for c in changes if c.op == "decrease"}
    increased = {c.target for c in changes if c.op == "increase"}
    assigned = {c.target for c in changes if c.op == "set"}
    terminal_read = _lineage_states(plan, _terminal_states(plan.terminal))

    for s in plan.states:
        where = f"state {s.name!r}"
        if s.state_type == "quantity" and not s.unit.strip():
            errors.append(
                f"UNIT_UNDECLARED: {where} is a quantity with no unit — what it is "
                "measured in is what makes it comparable with anything else, and a "
                "number with no unit cannot be checked against a rate, a threshold or "
                "another quantity at all. Correction boundary: unit on this state"
            )
        if s.kind != "flow" and s.period:
            errors.append(
                f"{where}: only a flow declares a period — a {s.kind} is a quantity or a "
                "reading, not a rate. Correction boundary: kind, or drop period"
            )
        if s.kind != "level" and s.not_a_stock_because:
            errors.append(
                f"{where}: not_a_stock_because belongs on a level that has to explain "
                f"itself, not on a {s.kind}. Correction boundary: drop that field"
            )
        if s.kind != "stock" and (s.capacity is not None or s.conserved_floor):
            errors.append(
                f"{where}: capacity and conserved_floor are a stock's bounds, and this "
                f"state is a {s.kind}. Correction boundary: kind, or drop those fields"
            )

        if s.kind == "flow":
            period = parse_period(s.period)
            if s.state_type != "quantity":
                errors.append(
                    f"{where}: a flow is a rate, so its state_type must be 'quantity'. "
                    "Correction boundary: state_type, or kind"
                )
            if period is None or not period.is_positive():
                errors.append(
                    f"FLOW_PERIOD_UNDECLARED: {where} is declared a flow — a rate — but "
                    f"its period is {s.period!r}, which is not a positive ISO-8601 "
                    "duration. A rate with no period is a number pretending to be a "
                    "mechanism: what it produces depends entirely on how long it runs, "
                    "and nothing can check that until the period is stated. Correction "
                    "boundary: period on this state (P1D, P1W, P1M, PT6H …), or "
                    "kind=level if it is a total rather than a rate"
                )

        if s.kind == "stock":
            initial = _numeric(s.initial)
            if s.state_type != "quantity":
                errors.append(
                    f"STOCK_DECLARATION_INCOMPLETE: {where} is a stock but its state_type "
                    f"is {s.state_type!r} — a conserved quantity is a quantity. "
                    "Correction boundary: state_type, or kind"
                )
            if initial is None:
                errors.append(
                    f"STOCK_DECLARATION_INCOMPLETE: {where} is a stock with initial "
                    f"{s.initial!r} — a quantity held somewhere has to start at a known "
                    "amount, because every later move is measured against it and an "
                    "UNKNOWN one is invisible to every check that would catch a "
                    "physically impossible run. Correction boundary: a cited initial "
                    "quantity on this state, or kind=level with not_a_stock_because if "
                    "the record genuinely does not establish the level"
                )
            elif initial < s.conserved_floor:
                errors.append(
                    f"STOCK_DECLARATION_INCOMPLETE: {where} starts at {initial:g}, below "
                    f"its own floor of {s.conserved_floor:g}. Correction boundary: the "
                    "initial value, or conserved_floor"
                )
            if s.name in assigned:
                errors.append(
                    f"UNCONSERVED_PHYSICAL_STOCK: {where} is a stock and something 'set's "
                    "it — assigning a held quantity makes it appear or vanish without "
                    "moving anywhere, which is exactly the arithmetic a stock exists to "
                    "forbid. Correction boundary: express the change as "
                    "increase/decrease of the amount that moves, or declare the state "
                    "kind=level if it is a reading rather than a quantity held somewhere"
                )
            if s.capacity is not None and initial is not None and s.capacity < initial:
                errors.append(
                    f"STOCK_DECLARATION_INCOMPLETE: {where} starts at {initial:g} with "
                    f"capacity {s.capacity:g} — it begins over its own ceiling. "
                    "Correction boundary: capacity or initial"
                )
            if s.capacity is not None and s.capacity <= s.conserved_floor:
                errors.append(
                    f"STOCK_DECLARATION_INCOMPLETE: {where} declares capacity "
                    f"{s.capacity:g} at or below its floor {s.conserved_floor:g} — there "
                    "is no room between them for anything to be held. Correction "
                    "boundary: capacity or conserved_floor"
                )
            if s.name in decreased and s.name not in increased and s.name in terminal_read:
                # Refusable before anything runs: a stock the world only ever draws on,
                # standing between the cutoff and the answer, cannot be what the record
                # says it is unless nothing replenishes it inside the window — and if
                # nothing does, the plan should say what the world does when it empties.
                errors.append(
                    f"STOCK_DRAINED_WITHOUT_INFLOW: {where} is drawn down by this world "
                    "and nothing anywhere puts anything back, while the terminal depends "
                    "on it — so the answer is decided by how long the starting amount "
                    "lasts, and the plan never says whether that is true of the real "
                    "world. Correction boundary: model the inflow that replenishes it "
                    "(with its own rate and cadence), or state in why_material that this "
                    "quantity genuinely is not replenished inside the window so the "
                    "drawdown is the mechanism"
                )

        if s.kind == "level" and s.state_type == "quantity" and not s.not_a_stock_because.strip():
            if s.name in decreased:
                errors.append(
                    f"UNCONSERVED_PHYSICAL_STOCK: {where} is a plain level and something "
                    "decreases it, so nothing stops it going below zero — a live world "
                    "drained one of these for a whole quarter and resolved on the "
                    "impossible arithmetic. If this is a quantity held somewhere, declare "
                    "kind=stock (with its holder, a known initial amount, its floor, and "
                    "a capacity if anything adds to it) and every draw on it is clamped "
                    "at the floor with the shortfall recorded. Correction boundary: "
                    "kind=stock on this state, or not_a_stock_because saying why this "
                    "particular quantity may legitimately go below zero (a net position, "
                    "a balance, a difference)"
                )
            elif s.name in terminal_read and s.name in increased:
                errors.append(
                    f"ACCUMULATING_QUANTITY_UNDECLARED: the terminal depends on "
                    f"{s.name!r}, which this world accumulates, and it is declared a "
                    "plain level — so nothing in the plan says whether it is a quantity "
                    "held somewhere or a tally of what has already happened. The answer "
                    "turns on it, so it has to be said. Correction boundary: kind=stock "
                    "on this state (with its initial amount, floor and capacity), or "
                    "not_a_stock_because saying why it is a running record rather than a "
                    "conserved quantity"
                )

        # CW-F: an accumulator with no starting point is invisible to every forward
        # evaluator this module and the reviewer run — `_forward_state_values` cannot
        # begin the sum, so the straddling gate, the break-even and the pre-simulation
        # probe all silently decline. Requiring a citation for a precise initial made
        # UNKNOWN the cheap way out for exactly the counters this vocabulary is meant to
        # encourage, so the escape is closed where it matters: a quantity the terminal
        # depends on and the world adds to must say where it starts, and "nothing has
        # been recorded yet" is a claim the record can carry.
        if (
            s.state_type == "quantity"
            and s.name in terminal_read
            and s.name in increased
            and _numeric(s.initial) is None
        ):
            errors.append(
                f"ACCUMULATOR_WITHOUT_AN_ORIGIN: {where} is accumulated by this world and "
                f"the terminal depends on it, but it starts at {s.initial!r} — an "
                "accumulation with no starting point has no value at any moment, and "
                "every check that would compare its total against the threshold declines "
                "instead of firing. Correction boundary: the initial value on this state "
                "— the amount recorded so far, cited (0 with the claim that nothing has "
                "been recorded yet is a real and sufficient answer)"
            )
    return errors


def _process_duration(process: SemanticProcess) -> tuple[str, str]:
    """The single length of time this process's firings each cover, or why there isn't one.

    A firing that applies a rate says how long it runs for; two firings of one process
    that claim different lengths are two mechanisms wearing one name, so that is refused
    rather than averaged.
    """

    seen: list[str] = []
    for o in process.occurrences:
        for c in o.changes:
            for v in (c.value, c.amount):
                if v is not None:
                    seen.extend(v.durations())
    unique = sorted(set(seen))
    if not unique:
        return "", ""
    if len(unique) > 1:
        return "", (
            f"its firings claim to cover different lengths of time ({unique}) — one "
            "process is one mechanism, and a mechanism has one cadence"
        )
    return unique[0], ""


def _cadence_errors(plan: SemanticPlan) -> list[str]:
    """FD-25: the number of firings is derived from a window, never typed by hand.

    A live world applied a WEEKLY rate at two dates in a ten-week quarter, so the total
    was the rate times two and every branch resolved NO on an artifact of the schedule.
    The adversary then showed the same hole in the other direction on one invented world:
    the identical mechanism typed eight times totals 840 and typed ten times totals 990
    against a threshold of 900, and both pass every other gate. The count IS the answer,
    so it may not be a free integer.

    Once a firing declares the length of time it covers (see the dimensional gate — a
    rate cannot be applied without one), the count is arithmetic: a window from the first
    firing to the last holds exactly as many firings of that length as it holds, and a
    calendar that disagrees with its own window is refused in whichever direction it
    disagrees. Declaring a recurrence sidesteps the question entirely, which is the
    point: code enumerates, the planner declares.
    """

    errors: list[str] = []
    for p in plan.processes:
        rec = p.recurrence
        if rec is not None:
            where = f"process {p.name!r} recurrence"
            period = parse_period(rec.period)
            start, end = _parse_when(rec.start), _parse_when(rec.end)
            if p.kind == "actor_moment":
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: {where}: an actor_moment is one "
                    "dated occasion, not a cadence — declare each occasion as its own "
                    "actor_moment process. Correction boundary: the recurrence, or the "
                    "process kind"
                )
            if p.occurrences and not p.occurrences_generated:
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: {where} is declared alongside "
                    f"{len(p.occurrences)} hand-written occurrence(s) — the process would "
                    "have two different calendars and no reader could tell which one "
                    "runs. Correction boundary: keep the recurrence and delete the "
                    "occurrences, or keep the occurrences and delete the recurrence"
                )
            if period is None or not period.is_positive():
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: {where}: period {rec.period!r} is "
                    "not a positive ISO-8601 duration. Correction boundary: period"
                )
            if start is None or end is None:
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: {where}: start {rec.start!r} and end "
                    f"{rec.end!r} must both be ISO datetimes bounding the window this "
                    "cadence runs across. Correction boundary: start and end"
                )
            elif end < start:
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: {where}: end {rec.end} is before "
                    f"start {rec.start}. Correction boundary: start and end"
                )
            elif period is not None and period.is_positive():
                total = firings_between(start, end, period)
                if total > RECURRENCE_OCCURRENCE_CAP:
                    errors.append(
                        f"RECURRENCE_DECLARATION_INVALID: {where}: every {rec.period} from "
                        f"{rec.start} to {rec.end} is {total}+ firings, past the "
                        f"{RECURRENCE_OCCURRENCE_CAP} this compiler will build — a "
                        "cadence that fine is not the mechanism, it is its sampling rate. "
                        "Correction boundary: a period at the scale the world actually "
                        "moves, or a shorter window"
                    )
                elif total < 2:
                    errors.append(
                        f"RECURRENCE_DECLARATION_INVALID: {where}: a period of "
                        f"{rec.period} does not fit twice between {rec.start} and "
                        f"{rec.end}, so the declared cadence fires once and means "
                        "nothing. Correction boundary: the period, or the window, or "
                        "state this as a single dated occurrence"
                    )
            if not rec.changes:
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: {where} declares a cadence with no "
                    "changes — a firing that changes nothing is a date, not a mechanism. "
                    "Correction boundary: changes on the recurrence"
                )

        if p.kind == "actor_moment":
            continue
        covered, why = _process_duration(p)
        if why:
            errors.append(
                f"DURATION_MISUSED: process {p.name!r}: {why}. Correction boundary: the "
                "durations inside this process's changes"
            )
            continue
        if not covered:
            continue  # nothing here applies a rate; there is no cadence to check
        span = parse_period(covered)
        if span is None or not span.is_positive():
            continue  # the dimensional gate names it

        if rec is not None:
            rec_period = parse_period(rec.period)
            if rec_period is not None and (rec_period.months, rec_period.seconds) != (
                span.months,
                span.seconds,
            ):
                errors.append(
                    f"RECURRENCE_DECLARATION_INVALID: process {p.name!r} fires every "
                    f"{rec.period} but each firing claims to cover {covered} — the world "
                    f"would advance {covered} of the mechanism every {rec.period} of the "
                    "calendar, counting the same time more than once or skipping it. "
                    "Correction boundary: the duration inside the recurrence's changes, "
                    "or the recurrence period — they are the same length of time"
                )
            continue

        dates = sorted(w for o in p.occurrences if (w := _parse_when(o.at)) is not None)
        if len(dates) < 2:
            continue
        required = firings_between(dates[0], dates[-1], span)
        if len(dates) == required:
            continue
        window = f"{dates[0].isoformat()} → {dates[-1].isoformat()}"
        if len(dates) < required:
            errors.append(
                f"UNDER_ENUMERATED_CADENCE: process {p.name!r} fires {len(dates)} times "
                f"across {window}, each firing covering {covered} — but that window holds "
                f"{required} of them, so {required - len(dates)} periods of the mechanism "
                "never happen and everything it would have produced is missing. The total "
                "this world reaches is a fact about how many dates were written down. "
                f"Correction boundary: declare recurrence {{period: {covered}, start, end, "
                "changes}} on this process and let the calendar be enumerated, or "
                f"enumerate all {required} occurrences"
            )
        else:
            errors.append(
                f"OVER_ENUMERATED_CADENCE: process {p.name!r} fires {len(dates)} times "
                f"across {window}, each firing covering {covered} — but that window holds "
                f"only {required}, so {len(dates) - required} periods of the mechanism are "
                "counted twice and the total is inflated by whatever they produce. "
                f"Correction boundary: declare recurrence {{period: {covered}, start, end, "
                f"changes}} on this process, or enumerate exactly {required} occurrences"
            )
    return errors


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
    errors += _state_kind_errors(plan)
    errors += _dimension_errors(plan)
    errors += _cadence_errors(plan)
    errors += _representation_record_errors(plan)
    errors += _alternative_quality_errors(plan, cited)
    errors += _actor_admissibility_errors(plan, cited)
    errors += _one_step_operational_errors(plan, cited)
    errors += _straddling_errors(plan, cited)
    return errors
