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
# D2/FI-2: there is no published point estimate. The run has honest scenario bounds and
# a stated reason, and the scenario average survives in diagnostics only. A label alone
# was demonstrably not enough — two live runs published their scenario average as the
# headline while `probability_source` and bounds of [0, 1] sat beside it saying the
# number constrained nothing — so suppression empties the number itself.
PROBABILITY_SOURCE_SUPPRESSED = "point_estimate_withheld_scenario_bounds_only"
# D6/FI-3: the responsibility gate refused publication outright. Not "here is a number
# we do not trust" but "this run does not get to answer the question".
PROBABILITY_SOURCE_NOT_PUBLISHABLE = "no_answer_published_responsibility_gate"

# Why a point estimate was withheld (D2/FI-2, D3/FI-5). Universal, never per question.
SUPPRESSED_UNGROUNDED_WEIGHTS_DISAGREE = "ungrounded_branch_weights_disagree"
SUPPRESSED_RESOLVED_MASS_IS_A_MINORITY = "resolved_mass_is_a_minority_of_branch_mass"
SUPPRESSED_THRESHOLD_STRADDLING = "threshold_straddling_ungrounded_scenarios"
SUPPRESSED_RESPONSIBILITY_GATE = "responsibility_classification_forbids_publication"
SUPPRESSED_WEIGHTS_UNGROUNDED_FOR_A_CAUSED_RESULT = "caused_result_without_grounded_branch_weights"

# ---------------------------------------------------------------------------
# Responsibility (D6 / FI-3 / FI-4) — what produced the number
# ---------------------------------------------------------------------------

ACTOR_CAUSED = "ACTOR_CAUSED"
PROCESS_CAUSED = "PROCESS_CAUSED"
ACTOR_AND_PROCESS_CAUSED = "ACTOR_AND_PROCESS_CAUSED"
FACTUALLY_RESOLVED = "FACTUALLY_RESOLVED"
INITIAL_ASSUMPTIONS_DOMINATED = "INITIAL_ASSUMPTIONS_DOMINATED"
BRANCH_WEIGHTS_DOMINATED = "BRANCH_WEIGHTS_DOMINATED"
RESPONSIBILITY_UNRESOLVED = "UNRESOLVED"
RESPONSIBILITY_INVALID = "INVALID"

#: The complete D6 vocabulary. Nothing outside it may be written to an artifact.
RESPONSIBILITY_CLASSIFICATIONS: tuple[str, ...] = (
    ACTOR_CAUSED,
    PROCESS_CAUSED,
    ACTOR_AND_PROCESS_CAUSED,
    FACTUALLY_RESOLVED,
    INITIAL_ASSUMPTIONS_DOMINATED,
    BRANCH_WEIGHTS_DOMINATED,
    RESPONSIBILITY_UNRESOLVED,
    RESPONSIBILITY_INVALID,
)

#: The only classifications that may publish an answer at all (D6). The other four
#: describe a run whose number came from its own initialization, its own enumeration,
#: nothing, or a record that does not reconstruct.
PUBLISHING_CLASSIFICATIONS: frozenset[str] = frozenset(
    {ACTOR_CAUSED, PROCESS_CAUSED, ACTOR_AND_PROCESS_CAUSED, FACTUALLY_RESOLVED}
)

#: The counterfactual and sensitivity tests §14 requires before any publication. A
#: report that does not list all of them in ``tests_completed`` has not run the gate.
REQUIRED_RESPONSIBILITY_TESTS: tuple[str, ...] = (
    "all_actor_output_removed",
    "per_actor_removed",
    "per_process_removed",
    "initialization_preserved_terminal_reevaluated",
    "equal_weight_substitution",
    "ungrounded_numeric_alternatives_perturbed",
    "terminal_relevant_actions_removed",
)


@dataclass(frozen=True)
class BranchCounterfactual:
    """One branch's deterministic deletion replays (D6/D7), all zero-LLM.

    Each field holds the terminal answer the branch reaches when the named events are
    removed from its recorded ledger and the terminal is re-evaluated by the engine's
    own evaluator: ``"YES"`` / ``"NO"`` / ``"UNRESOLVED"`` / ``"UNRECONSTRUCTABLE"``.
    """

    branch_id: str
    outcome: str | None  # the branch's published answer, for comparison
    recomputed_outcome: str | None  # the same answer replayed from the ledger
    all_actor_output_removed: str | None
    all_process_output_removed: str | None
    initialization_preserved: str | None  # every recorded event removed
    terminal_relevant_actions_removed: str | None
    per_actor_removed: tuple[tuple[str, str | None], ...] = ()
    per_process_removed: tuple[tuple[str, str | None], ...] = ()
    numeric_perturbations: tuple[tuple[str, str | None], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "outcome": self.outcome,
            "recomputed_outcome": self.recomputed_outcome,
            "all_actor_output_removed": self.all_actor_output_removed,
            "all_process_output_removed": self.all_process_output_removed,
            "initialization_preserved": self.initialization_preserved,
            "terminal_relevant_actions_removed": self.terminal_relevant_actions_removed,
            "per_actor_removed": dict(self.per_actor_removed),
            "per_process_removed": dict(self.per_process_removed),
            "numeric_perturbations": dict(self.numeric_perturbations),
        }


@dataclass(frozen=True)
class ResponsibilityReport:
    """What produced the number, computed before publication and never asserted.

    Built by :func:`sworldmodel.responsibility.classify_responsibility` from the run's
    own ledger through :mod:`sworldmodel.replaycore`. ``classification`` is one of
    :data:`RESPONSIBILITY_CLASSIFICATIONS`; ``may_publish_answer`` is membership in
    :data:`PUBLISHING_CLASSIFICATIONS` **and** a complete test set —
    an incomplete gate publishes nothing, because a gate that did not run is not a
    gate that passed.

    ``point_estimate_permitted`` is the separate, stricter question D6 asks of an
    actor- or process-caused result: it still needs grounded branch weights before its
    magnitude may be presented as a calibrated point estimate. A factual resolution
    does not — the record, not the weights, decided it.
    """

    classification: str
    may_publish_answer: bool
    point_estimate_permitted: bool
    reason: str
    weights_grounded: bool
    trace_reproducible: bool
    trace_reproducible_basis: str
    branch_counterfactuals: tuple[BranchCounterfactual, ...] = ()
    weighted_probability: float | None = None
    equal_weight_probability: float | None = None
    weight_sensitivity_span: float | None = None
    threshold_straddling_variables: tuple[str, ...] = ()
    numeric_perturbation_findings: tuple[str, ...] = ()
    tests_completed: tuple[str, ...] = ()
    tests_missing: tuple[str, ...] = ()
    error: str = ""

    @property
    def gate_complete(self) -> bool:
        return not self.tests_missing and not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "may_publish_answer": self.may_publish_answer,
            "point_estimate_permitted": self.point_estimate_permitted,
            "reason": self.reason,
            "weights_grounded": self.weights_grounded,
            "trace_reproducible": self.trace_reproducible,
            "trace_reproducible_basis": self.trace_reproducible_basis,
            "weighted_probability": self.weighted_probability,
            "equal_weight_probability": self.equal_weight_probability,
            "weight_sensitivity_span": self.weight_sensitivity_span,
            "threshold_straddling_variables": list(self.threshold_straddling_variables),
            "numeric_perturbation_findings": list(self.numeric_perturbation_findings),
            "tests_completed": list(self.tests_completed),
            "tests_missing": list(self.tests_missing),
            "method": (
                "deterministic replay of the recorded event ledger with selected events "
                "removed, then terminal re-evaluation by the engine's own evaluator; no "
                "re-simulation, no model call"
            ),
            "error": self.error,
            "branches": [b.as_dict() for b in self.branch_counterfactuals],
        }


# ---------------------------------------------------------------------------
# The validity triple (D1 / FI-1)
# ---------------------------------------------------------------------------


class ValidityState(StrEnum):
    """Three-valued on purpose. ``NOT_ASSESSED`` is the honest default everywhere: a
    check that did not run must never read as a check that passed."""

    VALID = "valid"
    INVALID = "invalid"
    NOT_ASSESSED = "not_assessed"


#: What a reproducible trace does and does not claim. Carried into every artifact so
#: "RECONSTRUCTED" can never be read as "trustworthy": the forensic audit of three live
#: runs returned RECONSTRUCTED on all three, and two of the three were arithmetic over
#: equal, ungrounded weights that should never have been called forecasts.
RECONSTRUCTED_MEANS = (
    "trace_reproducible states only that the published arithmetic reproduces exactly "
    "from the recorded trace. It is not a claim that the compiled world was right, "
    "that the simulation was causally valid, or that the number is calibrated — read "
    "causal_simulation_valid and point_estimate_calibrated for those, separately."
)


@dataclass(frozen=True)
class ForecastValidity:
    """D1: three separate questions, three separate answers, never collapsed into one.

    A forecast is trustworthy as a forecast only when all three read ``VALID``; any
    other combination is reported as-is, with each leg's basis, and no summary verdict
    is offered that could be quoted in place of the three.
    """

    trace_reproducible: ValidityState
    causal_simulation_valid: ValidityState
    point_estimate_calibrated: ValidityState
    trace_reproducible_basis: str = ""
    causal_simulation_valid_basis: str = ""
    point_estimate_calibrated_basis: str = ""

    @property
    def all_three_valid(self) -> bool:
        return (
            self.trace_reproducible is ValidityState.VALID
            and self.causal_simulation_valid is ValidityState.VALID
            and self.point_estimate_calibrated is ValidityState.VALID
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_reproducible": self.trace_reproducible.value,
            "trace_reproducible_basis": self.trace_reproducible_basis,
            "causal_simulation_valid": self.causal_simulation_valid.value,
            "causal_simulation_valid_basis": self.causal_simulation_valid_basis,
            "point_estimate_calibrated": self.point_estimate_calibrated.value,
            "point_estimate_calibrated_basis": self.point_estimate_calibrated_basis,
            "all_three_valid": self.all_three_valid,
            "reconstructed_means": RECONSTRUCTED_MEANS,
        }


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
    materially on ungrounded weights, when it conditions on a minority of branch mass,
    or when there is no point estimate at all.

    ``probability_after_simulation`` is the scenario average. Under D2 it is a
    **diagnostic**: when ``point_estimate_suppressed`` is True the published answer
    carries no number at all and this record is the only place the average survives.
    It must never be rendered as the answer.
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
    # D2/FI-2 and D3/FI-5. Defaults keep hand-built records (tests, replays) valid.
    point_estimate_suppressed: bool = False
    suppression_reasons: tuple[str, ...] = ()
    resolved_mass_share: float = 1.0
    threshold_straddling_variables: tuple[str, ...] = ()

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
            "point_estimate_suppressed": self.point_estimate_suppressed,
            "suppression_reasons": list(self.suppression_reasons),
            "resolved_mass_share": self.resolved_mass_share,
            "threshold_straddling_variables": list(self.threshold_straddling_variables),
            "scenario_average_is_diagnostic_only": True,
        }


@dataclass(frozen=True)
class ForecastResult:
    """The published result.

    ``simulation_probability`` is **the published point estimate**. It is ``None``
    whenever the run has no honest point estimate to publish — nothing resolved, D2
    suppression, or the D6 responsibility gate. Every reader that renders an answer
    reads this field, which is exactly why suppression empties it rather than merely
    labelling it: two live runs published their scenario average as the headline while
    the labels beside it already said the number constrained nothing.

    ``scenario_average`` is the weighted-YES-over-resolved figure, always computed and
    **always diagnostic**. It is not an answer and must never be rendered as one.
    """

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
    # D2/FI-2: the scenario average, kept for diagnostics, never published as the answer.
    scenario_average: float | None = None
    point_estimate_suppressed: bool = False
    point_estimate_suppression_reason: str = ""
    # D6/FI-3: the gate refused to publish an answer at all (not merely a number).
    answer_withheld: bool = False
    # D1/FI-1 and D6/FI-3.
    validity: ForecastValidity | None = None
    responsibility: ResponsibilityReport | None = None

    def diagnostics_dict(self) -> dict[str, str]:
        return dict(self.diagnostics)
