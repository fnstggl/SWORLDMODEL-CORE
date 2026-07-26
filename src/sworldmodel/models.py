"""Core typed domain models — the cross-cutting primitives shared by every module.

These are pure, mostly-immutable records depending only on the standard library
(plus ``ids``/``errors``/``worldspec``). Nothing here imports the evidence store, the
world, or the runtime, which keeps the import graph acyclic.

This module contains **only universal** types: evidence/source enums, the event and
weight primitives, the genuine-uncertainty types, the immutable resolution contract,
the reality manifest, and the forecast outputs. It contains no committee, vote,
proposal, institution, or fixed-action-vocabulary type — those concepts, when a
question needs them, are *compiled data* in :mod:`worldspec`, never types here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any

from .errors import ContractMutationError
from .ids import content_id
from .worldspec import TerminalExpression

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SourceType(StrEnum):
    """Source categories, roughly in descending authority."""

    OFFICIAL_INSTITUTIONAL = "official_institutional"
    LEGAL_REGULATORY = "legal_regulatory"
    OFFICIAL_BIOGRAPHY = "official_biography"
    PRIMARY_RECORD = "primary_record"  # speeches, minutes, filings, votes, transcripts
    CONTEMPORANEOUS_REPORTING = "contemporaneous_reporting"
    SPECIALIST_ANALYSIS = "specialist_analysis"
    LOW_QUALITY = "low_quality"


class AuthorityLevel(IntEnum):
    """Orderable authority levels for load-bearing facts."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    AUTHORITATIVE = 4


class EpistemicType(StrEnum):
    """Hard distinction between what is known, inferred, and merely hypothesized."""

    OBSERVATION = "observation"  # directly supported by evidence
    INFERENCE = "inference"  # reasoned implication from cited observations
    HYPOTHESIS = "hypothesis"  # unobserved possibility represented as uncertainty
    UNSUPPORTED = "unsupported"  # retained only in diagnostics; never enters the world


class WeightProvenance(StrEnum):
    """Every branch weight must identify which of these it came from."""

    DIRECT_EMPIRICAL = "direct_empirical_distribution"
    MARKET_SURVEY = "market_or_survey_distribution"
    CALIBRATED_BEHAVIOR = "calibrated_behavior_model"
    EXPLICIT_MODEL = "explicit_model_distribution"
    SYMMETRIC_IGNORANCE = "symmetric_ignorance_assumption"
    SENSITIVITY_ONLY = "sensitivity_only_branch"


class IntegrityVerdict(StrEnum):
    VERIFIED = "verified"
    REFUSED = "refused"


class ForecastStatus(StrEnum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"
    REFUSED = "refused"


class Visibility(StrEnum):
    PUBLIC = "public"  # visible to everyone once its time arrives
    PRIVATE = "private"  # visible only to target_ids / audience
    ROLE = "role"  # visible to actors holding a named role


class EventStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Reality manifest inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequiredRealityFact:
    """A load-bearing fact that must be verified before behavioral rollout."""

    key: str
    description: str
    evidence_claim_ids: tuple[str, ...] = ()
    satisfied: bool = False

    def satisfy(self, evidence_claim_ids: tuple[str, ...]) -> RequiredRealityFact:
        return replace(self, satisfied=True, evidence_claim_ids=evidence_claim_ids)


@dataclass(frozen=True)
class MemorySeed:
    """An initial durable memory an actor carries into the simulation."""

    content: str
    kind: str  # "episodic" | "semantic"
    importance: float  # 0..1
    valid_time: datetime | None
    evidence_claim_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Resolution contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionContract:
    """The immutable question definition. Created *before* actors; once verified its
    load-bearing fields may not be silently rewritten by any downstream component.

    The terminal is a compiled declarative :class:`TerminalExpression` — never a
    fixed mechanism or family. ``expected_participants`` records the true number of
    actors the compiled world claims must exist (a generic reality check that a nine-
    seat body cannot become five modeled units); it is not committee-specific."""

    question: str
    as_of: datetime
    horizon: datetime
    subject_entity: str
    resolution_units: str
    terminal: TerminalExpression
    target_outcome: str = ""  # human label of the YES condition, for reporting
    authoritative_resolution_sources: tuple[str, ...] = ()
    required_reality_facts: tuple[RequiredRealityFact, ...] = ()
    expected_participants: int | None = None

    @property
    def contract_id(self) -> str:
        return content_id(
            "contract",
            self.question,
            self.as_of.isoformat(),
            self.horizon.isoformat(),
            self.target_outcome,
            self.subject_entity,
        )

    _LOCKED_FIELDS = (
        "question",
        "as_of",
        "horizon",
        "subject_entity",
        "resolution_units",
        "terminal",
        "target_outcome",
        "expected_participants",
    )

    def with_satisfied_facts(self, facts: tuple[RequiredRealityFact, ...]) -> ResolutionContract:
        return replace(self, required_reality_facts=facts)

    def checked_replace(self, **changes: Any) -> ResolutionContract:
        for name in changes:
            if name in self._LOCKED_FIELDS:
                raise ContractMutationError(
                    f"Refusing to modify locked contract field {name!r} after verification"
                )
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Reality manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealityManifest:
    """The determination of whether the world is faithful enough to simulate.

    Fields are generic: entities/roles/authorities/rules verified from evidence, plus
    an ``expected_participants`` vs ``represented_participants`` count that applies to
    any world (a decision body, a negotiation table, a set of strata, a set of orgs)."""

    verified_entities: tuple[str, ...]
    verified_roles: tuple[tuple[str, str], ...]  # (entity_id, role)
    verified_authorities: tuple[tuple[str, tuple[str, ...]], ...]  # (entity, capabilities)
    verified_rules: tuple[str, ...]
    verified_previous_actions: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    missing_required_facts: tuple[str, ...]
    evidence_coverage: float
    integrity_verdict: IntegrityVerdict
    expected_participants: int | None = None
    represented_participants: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_verified(self) -> bool:
        return self.integrity_verdict is IntegrityVerdict.VERIFIED


# ---------------------------------------------------------------------------
# Genuine uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchWeight:
    """A weight that always names its own provenance."""

    value: float
    provenance: WeightProvenance
    source_detail: str

    def scaled(self, factor: float) -> BranchWeight:
        return replace(self, value=self.value * factor)


@dataclass(frozen=True)
class UncertaintyOutcome:
    """One resolution of an uncertainty variable, with its environment effect.

    ``field_effects`` maps a world-field name to the numeric/typed level it takes when
    this outcome is realized — delivered into the world as an observable data release.
    (No notion of "signal" or "vote"; it is any typed field the compiler declared.)"""

    value: str
    weight: BranchWeight
    field_effects: tuple[tuple[str, Any], ...] = ()
    description: str = ""


@dataclass(frozen=True)
class UncertaintySpec:
    """Declares that a world field's *future* value is genuinely unknown, and how its
    branches are weighted (with provenance)."""

    variable: str
    why_unknown: str
    reversal_capable: bool
    outcomes: tuple[UncertaintyOutcome, ...]
    constraining_evidence_ids: tuple[str, ...] = ()
    # Other uncertainty variables this one is NOT independent of. Declaring a
    # dependence is a refusal to be crossed as if independent — the compiler must
    # instead express the dependent set as one uncertainty over joint states.
    depends_on: tuple[str, ...] = ()
    # When the unknown value actually becomes public, if the evidence establishes it.
    release_at: datetime | None = None


@dataclass(frozen=True)
class UncertaintyVariable:
    """A genuine unknown reality that can move the outcome (for the report)."""

    variable_id: str
    description: str
    why_unknown: str
    how_it_affects_outcome: str
    constraining_evidence_ids: tuple[str, ...]
    reversal_capable: bool
    outcomes: tuple[UncertaintyOutcome, ...]
    behavioral_equivalence_note: str = ""

    def total_weight(self) -> float:
        return sum(o.weight.value for o in self.outcomes)


# ---------------------------------------------------------------------------
# Events (the universal ledger entry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One applied happening in a branch. ``kind`` is a free string — a universal
    effect op (``set_field``, ``append_record``, ``deliver_information``, ...) or an
    engine-structural kind. There is no closed committee ontology."""

    event_id: str
    branch_id: str
    time: datetime
    kind: str
    actor_id: str | None
    target_ids: tuple[str, ...]
    payload: tuple[tuple[str, Any], ...]  # sorted key/value pairs (deterministic)
    visibility: Visibility
    audience: tuple[str, ...] = ()  # for PRIVATE (actor ids) or ROLE (role names)
    parent_event_ids: tuple[str, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()
    status: EventStatus = EventStatus.PENDING

    @property
    def payload_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload_dict.get(key, default)


def make_payload(data: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Freeze a payload dict into a deterministic, hashable tuple."""

    return tuple(sorted(data.items(), key=lambda kv: kv[0]))


# ---------------------------------------------------------------------------
# Forecast outputs
# ---------------------------------------------------------------------------

PROBABILITY_SOURCE = "weighted_simulated_trajectories"
# The point estimate materially depends on branch weights that were never grounded in
# any evidence (symmetric ignorance / sensitivity enumeration). The scenario average is
# still reported for reference, but the honest answer is the bounds.
PROBABILITY_SOURCE_UNGROUNDED_WEIGHTS = "scenario_enumeration_ungrounded_weights"
# Every resolved branch already carried its final answer before the first event fired:
# the cited record decided the question and the simulation changed nothing. Reporting
# such a run as "weighted_simulated_trajectories" claims a provenance it does not have —
# a live OPEC+ run published 1.00 under that label with zero scheduling batches, zero
# actor invocations and a terminal that was true at t0.
PROBABILITY_SOURCE_ESTABLISHED = "established_before_simulation_from_cited_record"


@dataclass(frozen=True)
class BranchOutcome:
    branch_id: str
    parent_lineage: tuple[str, ...]
    weight: float
    resolved: bool
    outcome: str | None  # "YES" | "NO" | None
    unresolved_reason: str | None
    truncated: bool
    key_conditions: tuple[tuple[str, str], ...]
    records: tuple[tuple[str, str], ...] = ()  # highlighted (label, value) for the report
    event_count: int = 0
    # What the terminal already said for the *initialized* branch world — its uncertain
    # values applied, but before any actor decision or process/action effect ran. This
    # is the branch's answer with the simulation deleted; comparing it with ``outcome``
    # is how the aggregate knows whether the trajectories added any information.
    pre_outcome: str | None = None  # "YES" | "NO" | None
    pre_resolved: bool = False
    # False when this branch's weight rests on symmetric-ignorance or sensitivity-only
    # enumeration rather than an identified distribution. The engine sets it explicitly
    # from the scenario's provenance; the permissive default exists only so hand-built
    # records (tests, replays) keep constructing.
    weight_grounded: bool = True


@dataclass(frozen=True)
class TrajectorySummary:
    branch_id: str
    weight: float
    outcome: str | None
    narrative: str
    records: tuple[tuple[str, str], ...]
    key_conditions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ForecastIntegrity:
    """Whether the reported number carries more information than its own initialization.

    Built by :func:`sworldmodel.outcomes.aggregate` from the branch table alone. The
    live failure this record exists for: a forecast whose branch weights were an
    arbitrary 0.5/0.5 symmetric-ignorance split reported exactly 0.5000 — the number
    was the prior, repeated, with the simulation as decoration.

    ``probability_before_simulation`` is the same weighted aggregation taken over each
    branch's pre-simulation terminal answer (conditional on pre-resolved mass);
    branches whose initial world did not resolve contribute to ``pre_unresolved_mass``
    instead. ``point_estimate_is_calibrated`` is False when the point estimate depends
    materially on ungrounded weights (or when there is no point estimate at all).
    """

    probability_before_simulation: float | None
    probability_after_simulation: float | None
    simulation_shift: float | None
    pre_resolved_mass: float
    pre_unresolved_mass: float
    weights_grounded_all: bool
    ungrounded_variables: tuple[str, ...]
    point_estimate_is_calibrated: bool
    counterfactual_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability_before_simulation": self.probability_before_simulation,
            "probability_after_simulation": self.probability_after_simulation,
            "simulation_shift": self.simulation_shift,
            "pre_resolved_mass": self.pre_resolved_mass,
            "pre_unresolved_mass": self.pre_unresolved_mass,
            "weights_grounded_all": self.weights_grounded_all,
            "ungrounded_variables": list(self.ungrounded_variables),
            "point_estimate_is_calibrated": self.point_estimate_is_calibrated,
            "counterfactual_note": self.counterfactual_note,
        }


@dataclass(frozen=True)
class ForecastResult:
    question: str
    contract: ResolutionContract
    integrity_manifest: RealityManifest
    status: ForecastStatus
    simulation_probability: float | None
    lower_bound: float | None
    upper_bound: float | None
    resolved_mass: float
    unresolved_mass: float
    resolved_yes_mass: float
    resolved_no_mass: float
    trajectory_summaries: tuple[TrajectorySummary, ...]
    branch_outcomes: tuple[BranchOutcome, ...]
    trace_location: str
    limitations: tuple[str, ...]
    model_call_count: int = 0
    token_usage: int = 0
    diagnostics: tuple[tuple[str, str], ...] = ()
    probability_source: str = PROBABILITY_SOURCE
    integrity: ForecastIntegrity | None = None

    def diagnostics_dict(self) -> dict[str, str]:
        return dict(self.diagnostics)
