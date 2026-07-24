"""Core typed domain models.

These are pure, mostly-immutable data records. They depend only on the standard
library (plus ``ids``/``errors``); nothing here imports the evidence store, the
world, or the runtime, which keeps the import graph acyclic. Evidence is always
referenced by *claim id* (a string), never by importing ``EvidenceClaim`` here.

Design rules enforced structurally:
* The ``ResolutionContract`` is frozen; its load-bearing fields cannot be rewritten.
* An ``Intent`` can only carry one of the allowed intent *kinds* — an actor has no
  vocabulary with which to assert a consequence ("persuaded", "passed", ...).
* ``BranchWeight`` always carries its provenance, so no weight can hide where it
  came from.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any

from .errors import ContractMutationError
from .ids import content_id

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


# Generic institutional / social primitives. Not a closed ontology: scenario code
# may introduce further string kinds and register validators/executors for them.
class EventKind:
    BRIEFING_DISTRIBUTED = "briefing_distributed"
    EXTERNAL_DATA_RELEASED = "external_data_released"
    PROPOSAL_INTRODUCED = "proposal_introduced"
    PROPOSAL_REVISED = "proposal_revised"
    STATEMENT_MADE = "statement_made"
    MESSAGE_SENT = "message_sent"
    MESSAGE_DELIVERED = "message_delivered"
    INFORMATION_REQUESTED = "information_requested"
    INFORMATION_PROVIDED = "information_provided"
    DECISION_OPENED = "decision_opened"
    VOTE_CAST = "vote_cast"
    COMMITMENT_MADE = "commitment_made"
    OPERATIONAL_ACTION = "operational_action"
    WAIT_RECORDED = "wait_recorded"
    TALLY_COMPUTED = "tally_computed"
    RESULT_PUBLISHED = "result_published"


class IntentKind:
    """The complete vocabulary an actor may emit. There is deliberately no kind for
    asserting a consequence (persuaded / coalition formed / proposal passed / …)."""

    SEND_MESSAGE = "send_message"
    MAKE_STATEMENT = "make_statement"
    REQUEST_INFORMATION = "request_information"
    INTRODUCE_PROPOSAL = "introduce_proposal"
    REVISE_PROPOSAL = "revise_proposal"
    SUPPORT_PROPOSAL = "support_proposal"
    OPPOSE_PROPOSAL = "oppose_proposal"
    MAKE_COMMITMENT = "make_commitment"
    OPERATIONAL_ACTION = "operational_action"
    CAST_VOTE = "cast_vote"
    WAIT = "wait"
    PRESERVE_PLAN = "preserve_plan"

    ALL: frozenset[str] = frozenset(
        {
            SEND_MESSAGE,
            MAKE_STATEMENT,
            REQUEST_INFORMATION,
            INTRODUCE_PROPOSAL,
            REVISE_PROPOSAL,
            SUPPORT_PROPOSAL,
            OPPOSE_PROPOSAL,
            MAKE_COMMITMENT,
            OPERATIONAL_ACTION,
            CAST_VOTE,
            WAIT,
            PRESERVE_PLAN,
        }
    )


# ---------------------------------------------------------------------------
# Resolution contract
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
class DecisionRule:
    """How an institution mechanically produces a decision (distinct from the
    question predicate). ``total_seats`` and ``threshold`` are verified reality and
    must never be rescaled to fit a smaller modeled roster."""

    kind: str  # "majority" | "supermajority" | "plurality" | "unanimous"
    total_seats: int
    threshold: int  # number of seat-votes required to carry the decision
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalSpec:
    """Maps a completed institutional decision to the question's YES/NO predicate.

    ``mechanism`` names the deterministic evaluator to use. ``yes_condition`` and
    ``target_option`` parameterize it generically (e.g. unanimous-for-"hold").
    """

    mechanism: str  # e.g. "committee_vote"
    yes_condition: str  # "unanimous_for_option" | "at_least_k_for_option" | "majority_for_option"
    target_option: str  # the option that constitutes YES (e.g. "hold")
    k: int | None = None  # threshold for at_least_k_for_option
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolutionContract:
    """The immutable question definition. Created *before* actors; once verified its
    load-bearing fields may not be silently rewritten by any downstream component."""

    question: str
    as_of: datetime
    horizon: datetime
    outcome_space: tuple[str, ...]
    target_outcome: str
    subject_entity: str
    decision_body: str
    resolution_units: str
    terminal_predicate: TerminalSpec
    decision_rule: DecisionRule
    authoritative_resolution_sources: tuple[str, ...] = ()
    required_reality_facts: tuple[RequiredRealityFact, ...] = ()
    expected_voting_seats: int | None = None

    @property
    def contract_id(self) -> str:
        return content_id(
            "contract",
            self.question,
            self.as_of.isoformat(),
            self.horizon.isoformat(),
            self.target_outcome,
            self.decision_body,
        )

    # The set of fields that are load-bearing and may never change after verification.
    _LOCKED_FIELDS = (
        "question",
        "as_of",
        "horizon",
        "outcome_space",
        "target_outcome",
        "subject_entity",
        "decision_body",
        "resolution_units",
        "terminal_predicate",
        "decision_rule",
        "expected_voting_seats",
    )

    def with_satisfied_facts(self, facts: tuple[RequiredRealityFact, ...]) -> ResolutionContract:
        """Return a copy with *only* ``required_reality_facts`` updated.

        This is the sole permitted mutation of a contract: recording which reality
        facts were satisfied by evidence. Any attempt to change a locked field via
        :meth:`checked_replace` raises ``ContractMutationError``.
        """

        return replace(self, required_reality_facts=facts)

    def checked_replace(self, **changes: Any) -> ResolutionContract:
        """Guarded replace that refuses to touch a locked field."""

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
    """The determination of whether the world is faithful enough to simulate."""

    verified_entities: tuple[str, ...]
    verified_roles: tuple[tuple[str, str], ...]  # (actor_id, role)
    verified_institutions: tuple[str, ...]
    verified_memberships: tuple[tuple[str, tuple[str, ...]], ...]  # (inst, members)
    verified_authorities: tuple[tuple[str, tuple[str, ...]], ...]  # (actor, permissions)
    verified_rules: tuple[str, ...]
    verified_previous_actions: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    missing_required_facts: tuple[str, ...]
    evidence_coverage: float
    integrity_verdict: IntegrityVerdict
    expected_voting_seats: int | None = None
    represented_voting_seats: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_verified(self) -> bool:
        return self.integrity_verdict is IntegrityVerdict.VERIFIED


# ---------------------------------------------------------------------------
# Compiled world building-blocks (produced by the compiler, validated by reality)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    kind: str
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReactionRule:
    """A conditional behavioral rule: *if* a signal crosses a threshold, the actor
    would move to a different option. This is causal structure, not a personality
    label."""

    trigger_signal: str  # name of an external signal, e.g. "inflation_surprise"
    direction: str  # "above" | "below"
    threshold: float
    moves_to_option: str
    rationale: str
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConditionalBehavior:
    """How an actor currently leans and what would change it.

    ``current_inclination`` is derived from evidence (prior vote + guidance +
    active proposal), never from an assigned trait. ``reaction_rules`` say what
    observable signal would move the actor, and by how much.
    """

    current_inclination: str
    reaction_rules: tuple[ReactionRule, ...]
    dissent_threshold: float  # 0..1, willingness to hold a lone position
    reasoning: str
    acceptance_tolerance: float = 0.5  # 0..1, readiness to accept a focal proposal
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemorySeed:
    """An initial durable memory an actor carries into the simulation."""

    content: str
    kind: str  # "episodic" | "semantic"
    importance: float  # 0..1
    valid_time: datetime | None
    evidence_claim_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()  # e.g. "commitment:hold"


@dataclass(frozen=True)
class ActorDefinition:
    """The compiled, static definition of an actor. The runtime :class:`ActorState`
    is initialized from this and then evolves; the definition itself is immutable."""

    actor_id: str
    name: str
    role: str
    authority: tuple[str, ...]
    is_voting_seat: bool
    vote_power: int
    conditional_behavior: ConditionalBehavior
    stable_identity: tuple[tuple[str, str], ...] = ()  # (key, value) pairs
    memory_seeds: tuple[MemorySeed, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstitutionSpec:
    institution_id: str
    name: str
    member_actor_ids: tuple[str, ...]
    decision_rule: DecisionRule
    seat_vote_powers: tuple[tuple[str, int], ...]  # (actor_id, power)
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    option: str
    text: str
    introduced_by: str
    introduced_at: datetime
    revision_of: str | None = None
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SignalDef:
    """A named external signal (e.g. an inflation surprise) with a baseline level.

    Signals are the observable levers that can move an actor's preferred option.
    They are evidence-grounded and delivered to actors as observations, never read
    directly by the mechanism.
    """

    name: str
    baseline: float
    description: str
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class UncertaintySpec:
    """Declares that a signal's *future* value is genuinely unknown, and how its
    branches are weighted (with provenance). This is the joint-uncertainty input the
    branch runtime samples — not a Cartesian explosion of decorative variants."""

    signal: str
    why_unknown: str
    reversal_capable: bool
    outcomes: tuple[UncertaintyOutcome, ...]
    constraining_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioFrame:
    """Evidence-grounded scaffolding that constrains world compilation.

    In live operation an LLM generates this from evidence and it is validated
    identically; in the deterministic offline path it is provided by the research
    corpus. Either way, every element cites available evidence and is checked for
    consistency with verified reality before it is used. The frame carries only
    *conditional* structure — it never encodes the terminal answer.
    """

    options: tuple[str, ...]
    signals: tuple[SignalDef, ...]
    reaction_rules: tuple[ReactionRule, ...]
    guidance_option: str | None
    guidance_text: str
    acceptance_tolerance: float  # 0..1 — how readily a seat accepts a focal proposal
    uncertainty: tuple[UncertaintySpec, ...]
    guidance_evidence_ids: tuple[str, ...] = ()

    def reaction_rules_to_option(self, option: str) -> tuple[ReactionRule, ...]:
        return tuple(r for r in self.reaction_rules if r.moves_to_option == option)


@dataclass(frozen=True)
class CausalEdge:
    cause: str
    effect: str
    mechanism: str


@dataclass(frozen=True)
class CausalGraph:
    nodes: tuple[str, ...]
    edges: tuple[CausalEdge, ...]


# ---------------------------------------------------------------------------
# Uncertainty
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
    """One resolution of an uncertainty variable, with an environment effect."""

    value: str
    weight: BranchWeight
    # Effect applied to the world when this outcome is realized: a mapping of
    # external-signal name -> numeric level, delivered as an observable data event.
    signal_effects: tuple[tuple[str, float], ...] = ()
    description: str = ""


@dataclass(frozen=True)
class UncertaintyVariable:
    """A genuine unknown reality. Only decision-relevant, reversal-capable unknowns
    should appear here — not decorative personality variants."""

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
# Events and intents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
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


@dataclass(frozen=True)
class Intent:
    """An actor's *intention*. The environment turns it into consequences; the actor
    cannot. Only kinds in :data:`IntentKind.ALL` are representable."""

    actor_id: str
    kind: str
    payload: tuple[tuple[str, Any], ...]
    rationale: str
    referenced_memory_ids: tuple[str, ...] = ()
    referenced_observation_ids: tuple[str, ...] = ()
    expected_effect: str = ""  # the actor's expectation, NOT the realized consequence

    @property
    def payload_dict(self) -> dict[str, Any]:
        return dict(self.payload)


# ---------------------------------------------------------------------------
# Forecast outputs
# ---------------------------------------------------------------------------

PROBABILITY_SOURCE = "weighted_simulated_trajectories"


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
    votes: tuple[tuple[str, str], ...]  # (actor_id, option)
    final_tally: tuple[tuple[str, int], ...]  # (option, count)
    event_count: int


@dataclass(frozen=True)
class TrajectorySummary:
    branch_id: str
    weight: float
    outcome: str | None
    narrative: str
    votes: tuple[tuple[str, str], ...]
    key_conditions: tuple[tuple[str, str], ...]


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
    # Diagnostics may include a separately-labeled reference-class comparator.
    # It is NEVER allowed to contribute to ``simulation_probability``.
    diagnostics: tuple[tuple[str, str], ...] = ()
    probability_source: str = PROBABILITY_SOURCE

    def diagnostics_dict(self) -> dict[str, str]:
        return dict(self.diagnostics)
