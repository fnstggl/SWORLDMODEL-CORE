"""Evidence-to-world coverage integrity.

Research has no value if the facts it finds are silently lost when the LLM compiles
the world. The motivating failure was concrete: the evidence store held all five
voting members of a committee, but the compiler represented only two, and because the
compiler *also* under-counted the seat total, the seat check passed vacuously. An
incomplete world was simulated as if it were complete.

This module makes that failure structurally impossible, for every kind of world
element, not just named people. The rule it enforces is:

    Every materially relevant item found in verified evidence must either be
    represented AND causally wired into the compiled world, or explicitly excluded
    with a recorded, evidence-grounded reason.

The process is deterministic where it must be (completeness, wiring, blocking) and
model-assisted where judgement helps (materiality of ambiguous items, exclusion
challenge). The LLM may reason; only deterministic code decides whether a run is
allowed to proceed.

Pipeline (canonical production path, not a diagnostic):

    verified evidence
      -> deterministic candidate inventory        (build_candidate_inventory)
      -> materiality review                        (deterministic + optional LLM)
      -> LLM world compilation using the inventory (universal_compiler / corpus)
      -> deterministic coverage comparison         (assess_coverage)
      -> targeted research / compilation repair     (universal_compiler)
      -> world-integrity gate                       (enforce_coverage + verify_reality)
      -> simulation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .errors import WorldIntegrityError
from .evidence import EvidenceClaim, EvidenceView
from .ids import content_id
from .models import ResolutionContract

_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class CandidateKind(StrEnum):
    """The kinds of world element a piece of evidence can imply. Deliberately broad:
    a person, an org, a rule, an event, a document, a resource, a relationship, or a
    resolution requirement are all covered by the same invariant."""

    PERSON = "person"
    ORGANIZATION = "organization"
    POPULATION_GROUP = "population_group"
    AUTHORITY_RELATION = "authority_relation"
    MEMBERSHIP = "membership"
    RULE = "rule"
    DOCUMENT = "document"
    CHANNEL = "channel"
    RESOURCE = "resource"
    LOCATION = "location"
    PRIOR_ACTION = "prior_action"
    SCHEDULED_EVENT = "scheduled_event"
    EXTERNAL_PROCESS = "external_process"
    VARIABLE = "variable"
    CAUSAL_RELATION = "causal_relation"
    RESOLUTION_REQUIREMENT = "resolution_requirement"


class Materiality(StrEnum):
    MATERIAL = "material"
    IMMATERIAL = "immaterial"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


class Disposition(StrEnum):
    INCLUDED = "included"
    EXCLUDED_IRRELEVANT = "excluded_irrelevant"
    MERGED = "merged"
    UNCERTAIN = "uncertain"
    REQUIRED_BUT_UNRESOLVED = "required_but_unresolved"


class CoverageVerdict(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


# ---------------------------------------------------------------------------
# Typed structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceCandidate:
    """A potentially world-relevant item extracted from verified evidence.

    ``claim_ids`` / ``lineage_ids`` preserve full source lineage. A *derived*
    candidate (``is_inference``) is inferred from a combination of claims and cites
    the claims it was inferred from; it is never presented as a directly observed
    fact.
    """

    candidate_id: str
    kind: CandidateKind
    canonical_identity: str
    description: str
    claim_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    relationships: tuple[tuple[str, str], ...] = ()  # (relation, target_identity)
    valid_time: str | None = None
    materiality: Materiality = Materiality.UNKNOWN
    is_inference: bool = False
    inferred_from: tuple[str, ...] = ()

    @property
    def is_material(self) -> bool:
        # CONFLICTED items are treated as material (they must be resolved, not dropped).
        return self.materiality in (Materiality.MATERIAL, Materiality.CONFLICTED)


@dataclass(frozen=True)
class WorldObject:
    """A thing that actually exists in the compiled world, with the claims it stands
    for and — critically — whether it is *wired* into the simulation and where."""

    object_id: str
    kind: str  # "actor" | "institution" | "rule" | "signal" | "terminal" | "world_fact" | ...
    name: str
    claim_ids: tuple[str, ...] = ()
    wired: bool = False
    uses: tuple[str, ...] = ()  # causal-use tags: actor_view / vote / terminal / reaction / ...


@dataclass(frozen=True)
class WorldSpecView:
    """A read-only description of the compiled world, sufficient to check coverage and
    causal use without importing the whole runtime."""

    objects: tuple[WorldObject, ...]
    accessible_claim_ids: frozenset[str] = frozenset()  # claims some actor can perceive
    decision_body: str = ""
    subject_entity: str = ""

    def by_kind(self, kind: str) -> list[WorldObject]:
        return [o for o in self.objects if o.kind == kind]


@dataclass(frozen=True)
class CandidateDisposition:
    candidate_id: str
    disposition: Disposition
    compiled_object_ids: tuple[str, ...] = ()
    reason: str = ""
    reviewer_stage: str = "deterministic_coverage"
    causal_uses: tuple[str, ...] = ()  # where the item is actually used in the sim


@dataclass(frozen=True)
class CompilationCoverageReport:
    total_candidates: int
    material_candidates: int
    included_candidates: int
    excluded_candidates: int
    merged_candidates: int
    uncertain_candidates: int
    unresolved_candidates: int
    missing_material_candidates: tuple[str, ...]
    coverage_verdict: CoverageVerdict
    candidates: tuple[EvidenceCandidate, ...] = ()
    dispositions: tuple[CandidateDisposition, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return self.coverage_verdict is CoverageVerdict.COMPLETE

    def disposition_for(self, candidate_id: str) -> CandidateDisposition | None:
        for d in self.dispositions:
            if d.candidate_id == candidate_id:
                return d
        return None

    def to_dict(self) -> dict[str, object]:
        """Full audit serialization: every candidate, its disposition, and the reason —
        so a live run can report what research found, what entered the world, what was
        merged, excluded, uncertain, or missing, and whether the gate passed."""

        disp = {d.candidate_id: d for d in self.dispositions}
        return {
            "coverage_verdict": self.coverage_verdict.value,
            "totals": {
                "total": self.total_candidates,
                "material": self.material_candidates,
                "included": self.included_candidates,
                "excluded": self.excluded_candidates,
                "merged": self.merged_candidates,
                "uncertain": self.uncertain_candidates,
                "unresolved": self.unresolved_candidates,
            },
            "missing_material_candidates": list(self.missing_material_candidates),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "kind": c.kind.value,
                    "identity": c.canonical_identity,
                    "materiality": c.materiality.value,
                    "is_inference": c.is_inference,
                    "claim_ids": list(c.claim_ids),
                    "lineage_ids": list(c.lineage_ids),
                    "inferred_from": list(c.inferred_from),
                    "disposition": (
                        disp[c.candidate_id].disposition.value if c.candidate_id in disp else None
                    ),
                    "reason": (disp[c.candidate_id].reason if c.candidate_id in disp else ""),
                    "compiled_object_ids": (
                        list(disp[c.candidate_id].compiled_object_ids)
                        if c.candidate_id in disp
                        else []
                    ),
                    "causal_uses": (
                        list(disp[c.candidate_id].causal_uses) if c.candidate_id in disp else []
                    ),
                }
                for c in self.candidates
            ],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Keyword lexicons for deterministic extraction (broad, not scenario-specific)
# ---------------------------------------------------------------------------

def _lex(words: str) -> frozenset[str]:
    return frozenset(words.split())


_ORG_WORDS = _lex(
    "bank board committee commission ministry department court council agency corporation"
    " company party union federation government congress parliament senate assembly authority"
    " bureau office bench tribunal cabinet directorate reserve fund organization association"
    " institute administration coalition alliance panel"
)
_ORG_SUFFIX = ("inc", "ltd", "llc", "plc", "corp", "co", "sa", "ag", "nv", "gmbh")
_POPULATION_WORDS = _lex(
    "voters electorate population households respondents citizens consumers workers residents"
    " members constituents demographic public shareholders taxpayers"
)
_ROLE_WORDS = _lex(
    "governor deputy chair chairman chairwoman chairperson president member board committee"
    " director official policymaker justice judge commissioner minister secretary delegate"
    " negotiator representative senator councillor councilor trustee regent officer executive"
    " ceo chief head sits seat seats appointed voted vote votes voting dissent dissented dissenting"
)
_RULE_WORDS = _lex(
    "rule law statute mandate requires require required must threshold majority unanimous"
    " unanimity quorum constitution bylaw regulation treaty procedure supermajority plurality"
    " eligibility ratify ratification binding provision clause article amendment charter"
)
# The subset of rule language that names a *decision procedure*: a rule using one of these
# words is the tally/authority rule that the compiled decision-rule object stands for.
_RULE_PROCEDURE_WORDS = _lex(
    "majority unanimous unanimity quorum supermajority plurality threshold consensus consent"
    " veto ratify ratification two-thirds"
)
_EVENT_WORDS = _lex(
    "meeting decision election hearing vote summit deadline scheduled announce announcement"
    " release publish publication convene convenes session sitting conference ballot runoff"
    " referendum verdict ruling"
)
_DOCUMENT_WORDS = _lex(
    "statement minutes report communique communiqué filing letter memo memorandum publication"
    " bulletin notice dossier brief briefing transcript guidance projection forecast dot"
    " summary release document record"
)
_CHANNEL_WORDS = _lex(
    "website portal press wire feed broadcast newsletter dispatch bulletin channel platform"
    " account handle"
)
_RESOURCE_WORDS = _lex(
    "reserves budget capacity funding resource stockpile supply inventory treasury liquidity"
    " capital troops materiel allocation"
)
_VARIABLE_WORDS = _lex(
    "rate inflation unemployment index price gdp growth level percentage bps basis yield spread"
    " ratio margin deficit surplus balance count share poll approval probability"
)
_ACTION_WORDS = _lex(
    "voted decided cut hiked held raised lowered ruled signed announced approved rejected vetoed"
    " passed blocked issued withdrew resigned appointed nominated endorsed opposed supported abstained"
)


def _entity_kind(entity: str, propositions: str) -> CandidateKind | None:
    """Classify a named entity by its surface form + the language used about it."""

    words = entity.split()
    low = entity.lower()
    tokens = set(_WORD.findall(low))
    if tokens & _ORG_WORDS or low.split()[-1] in _ORG_SUFFIX:
        return CandidateKind.ORGANIZATION
    # An all-caps acronym referenced as a body (FOMC, ECB, SCOTUS, UN).
    if 2 <= len(entity) <= 8 and entity.isupper() and entity.isalpha():
        return CandidateKind.ORGANIZATION
    if tokens & _POPULATION_WORDS:
        return CandidateKind.POPULATION_GROUP
    # A multi-word proper name is person-like; or a single name spoken about with a
    # personal role verb.
    if len(words) >= 2 and entity[:1].isupper():
        return CandidateKind.PERSON
    return None


def _has_words(text: str, lexicon: frozenset[str]) -> bool:
    return bool(set(_WORD.findall(text.lower())) & lexicon)


# ---------------------------------------------------------------------------
# Deterministic candidate inventory
# ---------------------------------------------------------------------------


def build_candidate_inventory(
    view: EvidenceView,
    contract: ResolutionContract,
    *,
    signal_claim_ids: frozenset[str] = frozenset(),
) -> tuple[EvidenceCandidate, ...]:
    """Build the complete candidate inventory from the verified evidence store.

    Deterministic and stable: the same evidence always yields the same inventory.
    ``signal_claim_ids`` are claim ids the causal/uncertainty frame already keys on;
    a resource/variable/event backed by such a claim is treated as material because
    the model itself judged it outcome-relevant.
    """

    ctx = _MaterialityContext(
        body_norm=_norm(contract.decision_body),
        subject_norm=_norm(contract.subject_entity),
        signal_claim_ids=signal_claim_ids,
        required_fact_ids=frozenset(
            cid for f in contract.required_reality_facts for cid in f.evidence_claim_ids
        ),
        as_of=contract.as_of,
        horizon=contract.horizon,
        rule_kind=contract.decision_rule.kind,
    )
    return _inventory(view, ctx, contract)


def build_inventory_from(
    view: EvidenceView,
    *,
    decision_body: str = "",
    subject_entity: str = "",
    as_of: datetime,
    horizon: datetime,
    rule_kind: str = "majority",
    required_fact_ids: frozenset[str] = frozenset(),
    signal_claim_ids: frozenset[str] = frozenset(),
) -> tuple[EvidenceCandidate, ...]:
    """Build the inventory from loose parameters, before a full contract exists.

    Used by the live compiler to enumerate candidates *and hand them to the LLM* so it
    cannot silently forget a verified item during compilation.
    """

    ctx = _MaterialityContext(
        body_norm=_norm(decision_body),
        subject_norm=_norm(subject_entity),
        signal_claim_ids=signal_claim_ids,
        required_fact_ids=required_fact_ids,
        as_of=as_of,
        horizon=horizon,
        rule_kind=rule_kind,
    )
    return _inventory(view, ctx, None)


def _inventory(
    view: EvidenceView, ctx: _MaterialityContext, contract: ResolutionContract | None
) -> tuple[EvidenceCandidate, ...]:
    claims = view.available()
    direct = _entity_candidates(claims, ctx)
    direct += _claim_kind_candidates(claims, ctx)
    derived = _derived_candidates(direct, ctx)
    resolution = _resolution_requirements(contract) if contract is not None else []

    everything = direct + derived + resolution
    # Stable order: material first, then by kind then identity.
    everything.sort(key=lambda c: (not c.is_material, c.kind.value, c.canonical_identity))
    return tuple(everything)


def evidence_checklist(
    view: EvidenceView,
    *,
    decision_body: str = "",
    subject_entity: str = "",
    as_of: datetime,
    horizon: datetime,
    max_items: int = 80,
) -> str:
    """A compact, kind-grouped enumeration of the *material* evidence candidates, for
    injection into an LLM compile prompt so the model is handed an explicit inventory
    of what verified reality contains rather than being trusted to recall it."""

    candidates = build_inventory_from(
        view,
        decision_body=decision_body,
        subject_entity=subject_entity,
        as_of=as_of,
        horizon=horizon,
    )
    material = [c for c in candidates if c.is_material][:max_items]
    if not material:
        return "(no material candidates detected)"
    by_kind: dict[str, list[EvidenceCandidate]] = {}
    for c in material:
        by_kind.setdefault(c.kind.value, []).append(c)
    lines: list[str] = []
    for kind in sorted(by_kind):
        items = "; ".join(
            f"{c.canonical_identity} [{','.join(c.claim_ids[:4])}]" for c in by_kind[kind]
        )
        lines.append(f"- {kind}: {items}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _MaterialityContext:
    """Everything the deterministic materiality rules need, in one place."""

    body_norm: str
    subject_norm: str
    signal_claim_ids: frozenset[str]
    required_fact_ids: frozenset[str]
    as_of: datetime
    horizon: datetime
    rule_kind: str

    def signal_linked(self, claim_ids: tuple[str, ...]) -> bool:
        return bool(set(claim_ids) & self.signal_claim_ids)

    def required_linked(self, claim_ids: tuple[str, ...]) -> bool:
        return bool(set(claim_ids) & self.required_fact_ids)


@dataclass
class _EntityGroup:
    kind: CandidateKind
    identity: str
    claims: dict[str, EvidenceClaim] = field(default_factory=dict)
    lineage: set[str] = field(default_factory=set)
    role: bool = False


def _entity_candidates(
    claims: list[EvidenceClaim], ctx: _MaterialityContext
) -> list[EvidenceCandidate]:
    """People / organizations / population groups, one candidate per distinct entity.

    Claims about the same entity are *merged* into a single candidate (this is the
    canonical de-duplication the coverage rule requires — one entity, one world
    object), preserving every contributing claim id and lineage id.
    """

    grouped: dict[tuple[CandidateKind, str], _EntityGroup] = {}
    for c in claims:
        prop = c.proposition
        for ent in c.entities:
            ent = ent.strip()
            if not ent:
                continue
            kind = _entity_kind(ent, prop)
            if kind is None:
                continue
            norm_id = _norm(ent)
            group = grouped.setdefault((kind, norm_id), _EntityGroup(kind=kind, identity=ent))
            group.claims[c.id] = c
            group.lineage.add(c.lineage_event_id)
            # A person is a decision-maker (material) when spoken about with a role or
            # vote verb, or when they co-occur in a claim naming the decision body.
            if _has_words(prop, _ROLE_WORDS) or (ctx.body_norm and ctx.body_norm in _norm(prop)):
                group.role = True

    out: list[EvidenceCandidate] = []
    for (kind, norm_id), group in grouped.items():
        claim_ids = tuple(sorted(group.claims))
        conflicted = _conflicting(list(group.claims.values()))
        materiality = _entity_materiality(kind, group.role, norm_id, claim_ids, conflicted, ctx)
        out.append(
            EvidenceCandidate(
                candidate_id=content_id("cand", kind.value, norm_id),
                kind=kind,
                canonical_identity=group.identity,
                description=f"{kind.value}: {group.identity}",
                claim_ids=claim_ids,
                lineage_ids=tuple(sorted(group.lineage)),
                materiality=materiality,
            )
        )
    return out


def _entity_materiality(
    kind: CandidateKind,
    has_role: bool,
    norm_id: str,
    claim_ids: tuple[str, ...],
    conflicted: bool,
    ctx: _MaterialityContext,
) -> Materiality:
    if conflicted:
        return Materiality.CONFLICTED
    if kind is CandidateKind.PERSON:
        # A decision-maker is material; a person named only incidentally is not.
        return Materiality.MATERIAL if has_role else Materiality.IMMATERIAL
    if kind is CandidateKind.ORGANIZATION:
        # The decision body / subject organization is always material.
        if _overlaps(norm_id, ctx.body_norm) or _overlaps(norm_id, ctx.subject_norm):
            return Materiality.MATERIAL
        return Materiality.MATERIAL if ctx.signal_linked(claim_ids) else Materiality.IMMATERIAL
    if kind is CandidateKind.POPULATION_GROUP:
        # A population is material when it is the deciding/subject group or the frame
        # keys on it; a group mentioned in passing is not.
        if _overlaps(norm_id, ctx.body_norm) or _overlaps(norm_id, ctx.subject_norm):
            return Materiality.MATERIAL
        return Materiality.MATERIAL if ctx.signal_linked(claim_ids) else Materiality.IMMATERIAL
    return Materiality.UNKNOWN


def _overlaps(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a in b or b in a)


@dataclass
class _ClaimGroup:
    kind: CandidateKind
    base: Materiality
    rep: EvidenceClaim
    claims: dict[str, EvidenceClaim] = field(default_factory=dict)
    lineage: set[str] = field(default_factory=set)


_CLAIM_LEXICONS: tuple[tuple[CandidateKind, frozenset[str]], ...] = (
    (CandidateKind.RULE, _RULE_WORDS),
    (CandidateKind.SCHEDULED_EVENT, _EVENT_WORDS),
    (CandidateKind.DOCUMENT, _DOCUMENT_WORDS),
    (CandidateKind.CHANNEL, _CHANNEL_WORDS),
    (CandidateKind.RESOURCE, _RESOURCE_WORDS),
    (CandidateKind.VARIABLE, _VARIABLE_WORDS),
    (CandidateKind.PRIOR_ACTION, _ACTION_WORDS),
)


def _claim_kind_candidates(
    claims: list[EvidenceClaim], ctx: _MaterialityContext
) -> list[EvidenceCandidate]:
    """Rules, scheduled events, documents, channels, resources, variables and prior
    actions. Grouped per (kind, underlying event) so claims from one source about one
    kind become a single candidate."""

    # One candidate per (kind, claim): a single source can carry many distinct facts,
    # so grouping by the source's lineage would conflate them and lose per-claim
    # attributes (a scheduled date, a distinct rule). Entity-level merging — where
    # canonical de-duplication matters — happens in ``_entity_candidates``.
    grouped: dict[tuple[str, str], _ClaimGroup] = {}
    for c in claims:
        text = c.proposition + " " + c.normalized_value
        for kind, lex in _CLAIM_LEXICONS:
            if not _has_words(text, lex):
                continue
            group = grouped.setdefault(
                (kind.value, c.id), _ClaimGroup(kind=kind, base=Materiality.UNKNOWN, rep=c)
            )
            group.claims[c.id] = c
            group.lineage.add(c.lineage_event_id)

    out: list[EvidenceCandidate] = []
    for group in grouped.values():
        rep = group.rep
        claim_ids = tuple(sorted(group.claims))
        conflicted = _conflicting(list(group.claims.values()))
        materiality = _claim_materiality(group.kind, rep, claim_ids, conflicted, ctx)
        identity = _snippet(rep.proposition)
        out.append(
            EvidenceCandidate(
                candidate_id=content_id("cand", group.kind.value, rep.lineage_event_id, identity),
                kind=group.kind,
                canonical_identity=identity,
                description=rep.proposition,
                claim_ids=claim_ids,
                lineage_ids=tuple(sorted(group.lineage)),
                valid_time=(rep.valid_from.isoformat() if rep.valid_from else None),
                materiality=materiality,
            )
        )
    return out


def _claim_materiality(
    kind: CandidateKind,
    rep: EvidenceClaim,
    claim_ids: tuple[str, ...],
    conflicted: bool,
    ctx: _MaterialityContext,
) -> Materiality:
    """Deterministic materiality for claim-derived candidates.

    Conservative on purpose: an item is enforced-material only when the evidence ties
    it to the outcome, so incidental mentions are recorded exclusions rather than
    false refusals. Conflicts are always material — they must be resolved, not dropped.
    """

    if conflicted:
        return Materiality.CONFLICTED
    if ctx.signal_linked(claim_ids) or ctx.required_linked(claim_ids):
        # The frame or a required reality fact keys on this claim -> outcome-relevant.
        return Materiality.MATERIAL
    if kind is CandidateKind.RULE and _has_words(rep.proposition, _RULE_PROCEDURE_WORDS):
        # A rule that names a decision procedure (majority/unanimity/quorum/...) is a
        # binding rule of the decision and must be represented faithfully.
        return Materiality.MATERIAL
    if kind is CandidateKind.SCHEDULED_EVENT and _in_window(rep, ctx):
        # An event scheduled to occur inside the forecast window can change the outcome.
        return Materiality.MATERIAL
    return Materiality.IMMATERIAL


def _in_window(claim: EvidenceClaim, ctx: _MaterialityContext) -> bool:
    when = claim.valid_from
    return when is not None and ctx.as_of < when <= ctx.horizon


def _derived_candidates(
    direct: list[EvidenceCandidate], ctx: _MaterialityContext
) -> list[EvidenceCandidate]:
    """Structural candidates inferred from combinations of verified claims.

    A membership/authority relationship is not usually stated in one sentence; it is
    implied by a person and the decision body co-occurring in role language. Derived
    candidates are marked ``is_inference`` and cite the claims they were inferred from.
    """

    out: list[EvidenceCandidate] = []
    body_org = next(
        (
            c
            for c in direct
            if c.kind is CandidateKind.ORGANIZATION and c.materiality is Materiality.MATERIAL
        ),
        None,
    )
    if body_org is None:
        return out
    for person in direct:
        if person.kind is not CandidateKind.PERSON or not person.is_material:
            continue
        infer_from = tuple(sorted(set(person.claim_ids) | set(body_org.claim_ids)))
        out.append(
            EvidenceCandidate(
                candidate_id=content_id("cand", "membership", person.canonical_identity),
                kind=CandidateKind.MEMBERSHIP,
                canonical_identity=f"{person.canonical_identity} in {body_org.canonical_identity}",
                description=(
                    f"{person.canonical_identity} is a member of {body_org.canonical_identity}"
                ),
                claim_ids=person.claim_ids,
                lineage_ids=person.lineage_ids,
                relationships=(("member_of", body_org.canonical_identity),),
                materiality=Materiality.MATERIAL,
                is_inference=True,
                inferred_from=infer_from,
            )
        )
    return out


def _resolution_requirements(contract: ResolutionContract) -> list[EvidenceCandidate]:
    """The terminal predicate and every required reality fact are, by definition,
    materially relevant: the question cannot resolve without them."""

    out: list[EvidenceCandidate] = []
    spec = contract.terminal_predicate
    out.append(
        EvidenceCandidate(
            candidate_id=content_id("cand", "resolution", spec.mechanism, spec.yes_condition),
            kind=CandidateKind.RESOLUTION_REQUIREMENT,
            canonical_identity=f"{spec.mechanism}:{spec.yes_condition}:{spec.target_option}",
            description=(
                f"terminal resolves via {spec.mechanism} when {spec.yes_condition} "
                f"({spec.target_option or spec.target_action})"
            ),
            claim_ids=spec.evidence_claim_ids,
            lineage_ids=(),
            materiality=Materiality.MATERIAL,
            is_inference=True,
            inferred_from=spec.evidence_claim_ids,
        )
    )
    for fact in contract.required_reality_facts:
        out.append(
            EvidenceCandidate(
                candidate_id=content_id("cand", "requirement", fact.key),
                kind=CandidateKind.RESOLUTION_REQUIREMENT,
                canonical_identity=f"required:{fact.key}",
                description=fact.description,
                claim_ids=fact.evidence_claim_ids,
                lineage_ids=(),
                materiality=Materiality.MATERIAL,
                is_inference=True,
                inferred_from=fact.evidence_claim_ids,
            )
        )
    return out


def _conflicting(claims: list[EvidenceClaim]) -> bool:
    """True when this group carries a decisive contradiction: a recorded contradiction
    id inside the group, or two claims asserting different normalized values for what
    is plainly the same proposition."""

    ids = {c.id for c in claims}
    for c in claims:
        if set(c.contradiction_ids) & ids:
            return True
    by_prop: dict[str, set[str]] = {}
    for c in claims:
        by_prop.setdefault(_norm(c.proposition), set()).add(c.normalized_value.strip().lower())
    return any(len(values) > 1 for values in by_prop.values())


def _snippet(text: str, *, words: int = 8) -> str:
    parts = text.split()
    return " ".join(parts[:words])


# ---------------------------------------------------------------------------
# Deterministic coverage comparison + causal-use check
# ---------------------------------------------------------------------------


@dataclass
class _Assessment:
    dispositions: list[CandidateDisposition] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def assess_coverage(
    candidates: tuple[EvidenceCandidate, ...],
    spec: WorldSpecView,
) -> CompilationCoverageReport:
    """Compare the candidate inventory against the compiled world and assign one
    disposition to every candidate. A material candidate that is not represented, or
    represented but not causally wired, is recorded as missing."""

    a = _Assessment()
    claimed_objects: dict[str, str] = {}  # object_id -> first candidate_id that used it

    for cand in candidates:
        matched = _match_objects(cand, spec)
        wired = [o for o in matched if o.wired]

        if cand.materiality is Materiality.CONFLICTED:
            # Never silently pick a side: block until the conflict is resolved.
            disp = Disposition.UNCERTAIN if wired else Disposition.REQUIRED_BUT_UNRESOLVED
            a.dispositions.append(
                CandidateDisposition(
                    candidate_id=cand.candidate_id,
                    disposition=disp,
                    compiled_object_ids=tuple(o.object_id for o in matched),
                    reason="conflicting verified evidence; resolve before simulating",
                )
            )
            a.missing.append(_label(cand, "conflicting evidence unresolved"))
            continue

        if wired:
            primary = wired[0]
            merged = primary.object_id in claimed_objects
            claimed_objects.setdefault(primary.object_id, cand.candidate_id)
            a.dispositions.append(
                CandidateDisposition(
                    candidate_id=cand.candidate_id,
                    disposition=Disposition.MERGED if merged else Disposition.INCLUDED,
                    compiled_object_ids=tuple(o.object_id for o in wired),
                    reason=(
                        f"represented by {primary.name!r} via {primary.kind}"
                        if not merged
                        else f"same canonical object as an earlier candidate ({primary.name!r})"
                    ),
                    causal_uses=primary.uses,
                )
            )
            continue

        # Represented somewhere but not wired into the simulation == not covered.
        if matched and not wired:
            a.dispositions.append(
                CandidateDisposition(
                    candidate_id=cand.candidate_id,
                    disposition=(
                        Disposition.REQUIRED_BUT_UNRESOLVED
                        if cand.is_material
                        else Disposition.EXCLUDED_IRRELEVANT
                    ),
                    compiled_object_ids=tuple(o.object_id for o in matched),
                    reason="present in the world specification but disconnected from the simulation",
                )
            )
            if cand.is_material:
                a.missing.append(_label(cand, "represented but not causally wired"))
            continue

        # Not represented at all.
        if cand.is_material:
            a.dispositions.append(
                CandidateDisposition(
                    candidate_id=cand.candidate_id,
                    disposition=Disposition.REQUIRED_BUT_UNRESOLVED,
                    reason="materially relevant but absent from the compiled world",
                )
            )
            a.missing.append(_label(cand, "material but absent"))
        else:
            # An immaterial item may be excluded, but the exclusion is recorded with a
            # concrete, evidence-grounded reason — it never silently disappears.
            a.dispositions.append(
                CandidateDisposition(
                    candidate_id=cand.candidate_id,
                    disposition=Disposition.EXCLUDED_IRRELEVANT,
                    reason=_exclusion_reason(cand),
                )
            )

    return _report(candidates, a)


def _match_objects(cand: EvidenceCandidate, spec: WorldSpecView) -> list[WorldObject]:
    """Objects that represent this candidate, by claim-id overlap or identity match.

    For documents/channels, representation means *accessibility*: an actor must be
    able to perceive the underlying claims, so a document whose claims are reachable
    by some actor is considered wired even without a dedicated object."""

    cand_claims = set(cand.claim_ids)
    ident = _norm(cand.canonical_identity)
    desc = _norm(cand.description)
    matched: list[WorldObject] = []
    for obj in spec.objects:
        if cand_claims and cand_claims & set(obj.claim_ids):
            matched.append(obj)
            continue
        oname = _norm(obj.name)
        if oname and ident and (oname in ident or ident in oname):
            matched.append(obj)
            continue
        if _semantic_match(cand, desc, obj):
            matched.append(obj)

    is_doc = cand.kind in (CandidateKind.DOCUMENT, CandidateKind.CHANNEL)
    if is_doc and cand_claims and (cand_claims & spec.accessible_claim_ids):
        matched.append(
            WorldObject(
                object_id=f"access:{cand.candidate_id}",
                kind="information_access",
                name=cand.canonical_identity,
                claim_ids=cand.claim_ids,
                wired=True,
                uses=("actor_view",),
            )
        )
    return matched


def _semantic_match(cand: EvidenceCandidate, desc: str, obj: WorldObject) -> bool:
    """Match a candidate to a structural object by meaning when claim ids and names do
    not overlap — the compiled object may not have been attributed the exact claim id.

    A rule candidate corresponds to the compiled decision rule when both name the same
    decision procedure (majority/unanimity/quorum/...). A scheduled event that names
    the decision itself is carried by the terminal/institution objects.
    """

    desc_words = set(_WORD.findall(desc))
    if cand.kind is CandidateKind.RULE and obj.kind in ("rule", "terminal", "institution"):
        shared = desc_words & _RULE_PROCEDURE_WORDS & set(_WORD.findall(_norm(obj.name)))
        return bool(shared)
    if cand.kind is CandidateKind.SCHEDULED_EVENT and obj.kind in ("terminal", "institution"):
        # The decision/vote event of the deciding body is represented by the terminal.
        return bool(desc_words & {"decision", "vote", "meeting", "ruling", "verdict"})
    return False


def _label(cand: EvidenceCandidate, why: str) -> str:
    return f"[{cand.kind.value}] {cand.canonical_identity} — {why}"


def _exclusion_reason(cand: EvidenceCandidate) -> str:
    if cand.kind is CandidateKind.PERSON:
        return "named incidentally; no role, vote, or membership signal in the evidence"
    return f"no outcome-relevant signal for this {cand.kind.value} in the evidence"


def _report(
    candidates: tuple[EvidenceCandidate, ...], a: _Assessment
) -> CompilationCoverageReport:
    def count(d: Disposition) -> int:
        return sum(1 for x in a.dispositions if x.disposition is d)

    material = sum(1 for c in candidates if c.is_material)
    verdict = CoverageVerdict.COMPLETE if not a.missing else CoverageVerdict.INCOMPLETE
    return CompilationCoverageReport(
        total_candidates=len(candidates),
        material_candidates=material,
        included_candidates=count(Disposition.INCLUDED),
        excluded_candidates=count(Disposition.EXCLUDED_IRRELEVANT),
        merged_candidates=count(Disposition.MERGED),
        uncertain_candidates=count(Disposition.UNCERTAIN),
        unresolved_candidates=count(Disposition.REQUIRED_BUT_UNRESOLVED),
        missing_material_candidates=tuple(a.missing),
        coverage_verdict=verdict,
        candidates=candidates,
        dispositions=tuple(a.dispositions),
        notes=tuple(a.notes),
    )


def enforce_coverage(report: CompilationCoverageReport) -> None:
    """Block simulation when verified reality was lost during compilation."""

    if report.is_complete:
        return
    raise WorldIntegrityError(
        "verified evidence was lost during world compilation — simulation refused",
        details={
            "missing_material_candidates": list(report.missing_material_candidates),
            "material_candidates": report.material_candidates,
            "included": report.included_candidates,
            "unresolved": report.unresolved_candidates,
            "uncertain": report.uncertain_candidates,
        },
    )
