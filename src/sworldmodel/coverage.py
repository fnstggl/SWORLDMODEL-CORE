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
      -> the whole inventory handed to the compiler (evidence_checklist)
      -> LLM world compilation using the inventory (world_compiler / corpus)
      -> deterministic coverage comparison         (assess_coverage)
      -> targeted research / compilation repair     (api._compile_with_repair)
      -> world-integrity gate                       (enforce_coverage + verify_reality)
      -> simulation

The checklist is the *whole* inventory, not the material subset. Materiality grows
with inputs that are read off the compiled world (the identities it names, the claims
it keys on), which do not exist yet when the compiler is prompted; showing only what
is material without them would ask the compiler for less than the gate enforces and
make every difference an unsatisfiable repair round. See :func:`evidence_checklist`.

The comparison runs against the *exact compiled WorldSpec that will be simulated*
(see :func:`sworldmodel.world_compiler.world_spec_view`), so nothing can be verified
here and then quietly differ in the world the engine actually executes. Nothing in this
module assumes a committee, a vote, or any question family: a world element is a
person, org, rule, document, resource, event, channel, variable, or requirement.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .errors import WorldIntegrityError
from .evidence import ConflictScreen, ConflictScreening, EvidenceClaim, EvidenceView
from .ids import content_id
from .models import ResolutionContract

_WORD = re.compile(r"[a-z0-9]+")

# An independent second opinion on an exclusion: given a candidate the deterministic
# rules would drop, return True iff omitting it could plausibly change the outcome.
ExclusionReviewer = Callable[["EvidenceCandidate"], bool]


def _norm(text: str) -> str:
    """Case-folded word tokens, with accents folded onto their base letters.

    The token pattern is ASCII, so without the fold "Rodríguez" tokenized to "rodr guez"
    and never matched "Rodriguez" — one source spelling a name with its diacritic and
    another without was enough to make the same person two people, and then to refuse a
    roster for omitting one of them. Names are the primary key of this whole system; they
    have to survive a source that drops an accent.
    """

    folded = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(_WORD.findall(stripped))


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
    # The evidence speaks about this name in role or authority terms, rather than
    # merely mentioning it near the subject. Only this supports demanding a seat.
    role_attested: bool = False

    @property
    def is_material(self) -> bool:
        # CONFLICTED items are treated as material (they must be resolved, not dropped).
        return self.materiality in (Materiality.MATERIAL, Materiality.CONFLICTED)


@dataclass(frozen=True)
class WorldObject:
    """A thing that actually exists in the compiled world, with the claims it stands
    for and — critically — whether it is *wired* into the simulation and where."""

    object_id: str
    # As emitted by ``world_compiler.world_spec_view``: "actor", "rule", "signal",
    # "document", "resource", "channel", "scheduled_event", "terminal", "world_fact",
    # or a non-actor entity's own compiled kind.
    kind: str
    name: str
    claim_ids: tuple[str, ...] = ()
    wired: bool = False
    # Causal-use tags read off the compiled program: actor_view / action / authority /
    # reaction / branch / process / terminal / effect_target / resource.
    uses: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldSpecView:
    """A read-only description of the compiled world, sufficient to check coverage and
    causal use without importing the whole runtime."""

    objects: tuple[WorldObject, ...]
    accessible_claim_ids: frozenset[str] = frozenset()  # claims some actor can perceive
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
    # How far conflict screening got on the store this inventory came from. The default
    # is deliberately the *unknown* value, so a report that was never handed a screen can
    # never be read as one that was handed a clean one.
    conflict_screening: ConflictScreening = ConflictScreening.NOT_SCREENED
    conflict_screen_summary: str = ""

    @property
    def is_complete(self) -> bool:
        return self.coverage_verdict is CoverageVerdict.COMPLETE

    @property
    def conflicts_certified(self) -> bool:
        """Whether this report is entitled to say the evidence holds no unresolved conflict.

        A coverage report has always been read as a certificate over the evidence that
        entered the world. It is only that for a store whose candidate conflicts were
        enumerated *and* ruled on; over an unscreened store it certifies coverage and
        nothing about agreement, and saying so is the difference between a check that
        looked and a check that declined to.
        """

        return self.conflict_screening is ConflictScreening.FULLY_ADJUDICATED

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
            "conflict_screening": self.conflict_screening.value,
            "conflicts_certified": self.conflicts_certified,
            "conflict_screen_summary": self.conflict_screen_summary,
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
# Nouns that name an *instrument* — a thing drafted, signed, published or enacted. A
# name ending in one of these is a document, not somebody. This is a fact about English
# noun phrases, not about any subject area.
_INSTRUMENT_NOUNS = _lex(
    "agreement treaty accord protocol pact deal contract convention covenant memorandum"
    " communique communiqué declaration statement report minutes bill act ordinance"
    " decree resolution ruling judgment judgement opinion decision letter release note"
    " paper review summary transcript filing prospectus"
)
# Lowercase particles that belong inside a personal name.
_NAME_PARTICLES = _lex("van von de del della der den di da dos das du la le bin ibn al of and")
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


# Language that treats a name as an acting body rather than a place, product or month.
# Deliberately about *agency* — deciding, producing, announcing, meeting, reporting —
# because that is what makes something a causal element of somebody's world.
_BODY_CONTEXT = _ACTION_WORDS | _EVENT_WORDS | _ORG_WORDS | _RULE_PROCEDURE_WORDS


def _near(identity: str, text: str, lexicon: frozenset[str], *, before: bool = False) -> bool:
    """Whether a word from ``lexicon`` appears next to this name, inside one clause.

    A bag-of-words test over the whole passage would call any capitalized token in a
    sentence about a decision an organization — including the month the decision falls
    in and the country it happens in. Requiring the word to sit within a short span of
    the name is the difference between "Mercosur signed" and "signed in Brazil".

    ``before`` also accepts the word preceding the name, which is where English puts a
    title: "Governor Andrew Bailey" attests a role exactly as "Andrew Bailey, Governor
    of the Bank of England" does.
    """

    name = rf"(?<![a-z0-9]){re.escape(identity.lower())}(?![a-z0-9])"
    words = rf"\b(?:{'|'.join(sorted(lexicon))})\b"
    low = text.lower()
    if re.search(rf"{name}[^.;]{{0,48}}?{words}", low):
        return True
    return before and re.search(rf"{words}[^.;]{{0,48}}?{name}", low) is not None


def _acts_in(identity: str, propositions: str) -> bool:
    """Whether the evidence shows this name *doing* something, close to the name itself."""

    return _near(identity, propositions, _BODY_CONTEXT)


def _entity_kind(entity: str, propositions: str) -> CandidateKind | None:
    """Classify a named entity by its surface form + the language used about it."""

    # "Andrew Bailey, Governor of the Bank of England" is a person with their office
    # appended. Classifying the whole string reads the office and calls the person an
    # organization, so the identity is taken from before the appositive — but only when
    # what precedes the comma is already a full name. "Smith, John" is one name written
    # backwards, and truncating it to "Smith" loses the person.
    head = entity.split(",")[0].strip()
    identity = head if len(head.split()) > 1 else entity
    words = identity.split()
    low = identity.lower()
    tokens = set(_WORD.findall(low))
    if tokens & _ORG_WORDS or (low.split() and low.split()[-1] in _ORG_SUFFIX):
        return CandidateKind.ORGANIZATION
    # An acronym referenced as a body (FOMC, ECB, OPEC+, S&P). Punctuation is stripped
    # before the shape test, because an organization does not stop being one for having
    # a "+" in its name — and OPEC+ was the subject of an entire acceptance question
    # that this inventory could not see. A single letter stays excluded: that is an
    # initial, not an organization.
    squashed = "".join(ch for ch in identity if ch.isalnum())
    if len(squashed) > 1 and squashed.isupper() and squashed.isalpha():
        return CandidateKind.ORGANIZATION
    if tokens & _POPULATION_WORDS:
        return CandidateKind.POPULATION_GROUP
    # A capitalized single token — Tesla, Mercosur, Banxico — is a real name that the
    # shape alone cannot tell from a month, a place or a product. The evidence can: when
    # the claims about it speak of it acting, deciding, producing or announcing, it is a
    # body. It is classified as an ORGANIZATION and never as a person, which is what
    # makes this safe — organizations are not counted as participants, so recognizing a
    # mononym can never manufacture a seat the reality gate then demands be filled.
    if (
        len(words) == 1
        and identity[:1].isupper()
        and not _is_calendar_shaped(identity)
        and _acts_in(identity, propositions)
    ):
        return CandidateKind.ORGANIZATION
    # An instrument: a thing that is signed, published or enacted. Names ending in one
    # of these are documents whatever else they look like, and reading them as people is
    # how a live EU–Mercosur run came to demand a seat for the "Mercosur Agreement" and
    # for the "Signed Trade Agreement" the world had already compiled as a document.
    if low.split() and low.split()[-1] in _INSTRUMENT_NOUNS:
        return CandidateKind.DOCUMENT
    # A capitalized name of more than one token is person-like — but *every* token has
    # to be capitalized. "EU member states" is a collective written the way collectives
    # are written, and calling it a person made the reality gate demand a seat for it
    # beside the European Union and the European Council, which is where its member
    # states already were. Lowercase particles are part of how names are written and do
    # not break the rule.
    if len(words) > 1 and all(w[:1].isupper() or w.lower() in _NAME_PARTICLES for w in words):
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
    focal_identities: tuple[str, ...] = (),
) -> tuple[EvidenceCandidate, ...]:
    """Build the complete candidate inventory from the verified evidence store.

    Deterministic and stable: the same evidence always yields the same inventory.
    ``signal_claim_ids`` are claim ids the compiled world already keys on (a field an
    uncertainty branches over, an action's supporting evidence); an item backed by such
    a claim is material because the compiled world itself judged it outcome-relevant.
    ``focal_identities`` are additional entity names the compiled world names — passing
    them lets an organization or population that the world treats as an actor count as
    material without any hardcoded notion of a "decision body".
    """

    ctx = _MaterialityContext(
        focal_norms=_focal_norms((contract.subject_entity, *focal_identities)),
        signal_claim_ids=signal_claim_ids,
        required_fact_ids=frozenset(
            cid for f in contract.required_reality_facts for cid in f.evidence_claim_ids
        ),
        as_of=contract.as_of,
        horizon=contract.horizon,
    )
    return _inventory(view, ctx, contract)


def build_inventory_from(
    view: EvidenceView,
    *,
    subject_entity: str = "",
    focal_identities: tuple[str, ...] = (),
    as_of: datetime,
    horizon: datetime,
    required_fact_ids: frozenset[str] = frozenset(),
    signal_claim_ids: frozenset[str] = frozenset(),
) -> tuple[EvidenceCandidate, ...]:
    """Build the inventory from loose parameters, before a full contract exists.

    Used by the live compiler to enumerate candidates *and hand them to the LLM* so it
    cannot silently forget a verified item during compilation.
    """

    ctx = _MaterialityContext(
        focal_norms=_focal_norms((subject_entity, *focal_identities)),
        signal_claim_ids=signal_claim_ids,
        required_fact_ids=required_fact_ids,
        as_of=as_of,
        horizon=horizon,
    )
    return _inventory(view, ctx, None)


def _focal_norms(identities: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(n for n in (_norm(i) for i in identities) if n))


def _inventory(
    view: EvidenceView, ctx: _MaterialityContext, contract: ResolutionContract | None
) -> tuple[EvidenceCandidate, ...]:
    claims = view.available()
    direct = _entity_candidates(claims, ctx)
    direct += _claim_kind_candidates(claims, ctx)
    derived = _derived_candidates(direct, ctx)
    resolution = _resolution_requirements(contract) if contract is not None else []

    everything = direct + derived + resolution
    # Stable order: material first, then directly-observed candidates before derived
    # inferences (so a person takes the INCLUDED slot and its inferred membership
    # MERGES into it, not the reverse), then by kind and identity.
    everything.sort(
        key=lambda c: (not c.is_material, c.is_inference, c.kind.value, c.canonical_identity)
    )
    return tuple(everything)


def evidence_checklist(
    view: EvidenceView,
    *,
    subject_entity: str = "",
    focal_identities: tuple[str, ...] = (),
    as_of: datetime,
    horizon: datetime,
    signal_claim_ids: frozenset[str] = frozenset(),
    required_fact_ids: frozenset[str] = frozenset(),
) -> str:
    """The kind-grouped enumeration of evidence candidates handed to the LLM compiler.

    It lists the **whole** inventory, not the subset that is material under the inputs
    available before compilation. That is not a stylistic choice. Materiality grows
    monotonically with the inputs: a candidate can only move from immaterial to
    material as focal identities and signal claim ids arrive, and both of those are
    read off the compiled world, so they exist at gate time and not here. Filtering
    here by a materiality computed without them would hand the compiler a strictly
    smaller list than the gate later enforces — the compiler would be judged on items
    it was never asked for, and every such item would drive a repair round that no
    amount of further research can satisfy.

    Listing everything makes the checklist independent of those inputs, so whatever the
    gate demands was on the list. ``signal_claim_ids`` / ``required_fact_ids`` /
    ``focal_identities`` are still accepted so a caller that already knows them gets
    the same materiality labels the gate will compute.
    """

    candidates = build_inventory_from(
        view,
        subject_entity=subject_entity,
        focal_identities=focal_identities,
        as_of=as_of,
        horizon=horizon,
        required_fact_ids=required_fact_ids,
        signal_claim_ids=signal_claim_ids,
    )
    if not candidates:
        return "(the verified evidence implies no world elements)"
    by_kind: dict[str, list[EvidenceCandidate]] = {}
    for c in candidates:
        by_kind.setdefault(c.kind.value, []).append(c)
    lines: list[str] = [
        "Every item below was found in the verified evidence and is checked against the "
        "world you compile. Anything you leave out must be genuinely incidental to the "
        "outcome.",
    ]
    for kind in sorted(by_kind):
        for c in by_kind[kind]:
            cites = ",".join(c.claim_ids) or "no direct citation"
            lines.append(f"- {kind} | {c.canonical_identity} [{cites}]")
    return "\n".join(lines)


@dataclass(frozen=True)
class _MaterialityContext:
    """Everything the deterministic materiality rules need, in one place.

    ``focal_norms`` are the normalized identities the question actually turns on — the
    subject entity plus any entity the compiled world already names. They are the
    universal replacement for a hardcoded "decision body": an organization or
    population that overlaps a focal identity is material, whatever kind of process
    the question involves.
    """

    focal_norms: tuple[str, ...]
    signal_claim_ids: frozenset[str]
    required_fact_ids: frozenset[str]
    as_of: datetime
    horizon: datetime

    def signal_linked(self, claim_ids: tuple[str, ...]) -> bool:
        return bool(set(claim_ids) & self.signal_claim_ids)

    def required_linked(self, claim_ids: tuple[str, ...]) -> bool:
        return bool(set(claim_ids) & self.required_fact_ids)

    def mentions_focal(self, text: str) -> bool:
        norm = _norm(text)
        return any(f and f in norm for f in self.focal_norms)

    def overlaps_focal(self, norm_id: str) -> bool:
        return any(_overlaps(norm_id, f) for f in self.focal_norms)


@dataclass
class _EntityGroup:
    kind: CandidateKind
    identity: str
    claims: dict[str, EvidenceClaim] = field(default_factory=dict)
    lineage: set[str] = field(default_factory=set)
    role: bool = False
    role_attested: bool = False


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
            # A date is never a world entity. The guard existed only where the roster
            # was checked, so the checklist handed to the compiler could still open with
            # "person | Q3 2026" — asking it to model a quarter as somebody.
            if _is_calendar_shaped(ent):
                continue
            norm_id = _norm(ent)
            group = grouped.setdefault((kind, norm_id), _EntityGroup(kind=kind, identity=ent))
            group.claims[c.id] = c
            group.lineage.add(c.lineage_event_id)
            # A person is a decision-maker (material) when spoken about with a role or
            # vote verb, or when they co-occur in a claim naming the decision body.
            if _has_words(prop, _ROLE_WORDS):
                group.role = True
                # Attested in role or authority terms — a much stronger signal than
                # merely appearing beside the subject, and the only one strong enough
                # to demand a seat at the table. Which is why the role word has to sit
                # next to the name: read across a whole claim, "EU member states must
                # ratify what the board agreed" attests a role for every capitalized
                # thing in it.
                group.role_attested = group.role_attested or _near(
                    ent, prop, _ROLE_WORDS, before=True
                )
            elif ctx.mentions_focal(prop):
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
                role_attested=group.role_attested,
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
        if ctx.overlaps_focal(norm_id):
            return Materiality.MATERIAL
        return Materiality.MATERIAL if ctx.signal_linked(claim_ids) else Materiality.IMMATERIAL
    if kind is CandidateKind.POPULATION_GROUP:
        # A population is material when it is the deciding/subject group or the frame
        # keys on it; a group mentioned in passing is not.
        if ctx.overlaps_focal(norm_id):
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
        # The whole proposition is the identity. Truncating it to a fixed number of
        # words shortened what the compiler was shown and changed which compiled
        # objects the identity could match, on nothing but the size of the cutoff.
        identity = " ".join(rep.proposition.split())
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
    """The declarative terminal and every required reality fact are, by definition,
    materially relevant: the question cannot resolve without them.

    The terminal is a compiled :class:`~sworldmodel.worldspec.TerminalExpression`, so it
    is described by its own plain-language condition — there is no mechanism or family
    to name here."""

    out: list[EvidenceCandidate] = []
    terminal = contract.terminal
    identity = terminal.description or contract.target_outcome or "terminal condition"
    out.append(
        EvidenceCandidate(
            candidate_id=content_id("cand", "resolution", identity),
            kind=CandidateKind.RESOLUTION_REQUIREMENT,
            canonical_identity=f"terminal:{identity}",
            description=f"the question resolves YES when: {identity}",
            claim_ids=(),
            lineage_ids=(),
            materiality=Materiality.MATERIAL,
            is_inference=True,
            inferred_from=(),
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


# ---------------------------------------------------------------------------
# Participants the evidence itself names
# ---------------------------------------------------------------------------


def evidence_named_participants(
    view: EvidenceView,
    contract: ResolutionContract,
    *,
    focal_identities: tuple[str, ...] = (),
    signal_claim_ids: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """The distinct persons the verified claims themselves name as decision-relevant.

    This is the evidence-side anchor for the participant count. A person counts when
    the evidence speaks about them in role or authority terms, or when a claim naming
    them also names a focal identity — the same materiality rule the coverage gate
    applies, so the two cannot disagree.

    Organizations and population groups are deliberately not counted. The evidence does
    not say whether such a body is one participant or a container for many, and turning
    it into a number would invent the very fact the caller is trying to verify.

    An empty result means the evidence does not establish a participant set at all. It
    is not a count of zero, and a caller must not report a check it could not run.
    """

    inventory = build_candidate_inventory(
        view,
        contract,
        signal_claim_ids=signal_claim_ids,
        focal_identities=focal_identities,
    )
    return tuple(
        sorted(
            {
                c.canonical_identity
                for c in inventory
                if c.kind is CandidateKind.PERSON
                and c.is_material
                and _speaks_of_a_role(c)
                and not _is_calendar_shaped(c.canonical_identity)
            }
        )
    )


@dataclass(frozen=True)
class ParticipantSupport:
    """The evidence that stands behind one participant the compiled world names.

    ``claim_ids`` are the available claims that are actually *about* this participant —
    it is one of their declared entities, or its name occurs in the claim's own text.
    Claim ids the compiler merely *attached* to an actor do not appear here unless the
    claims themselves mention it, which is the whole point: a live Bank of England world
    gave the Monetary Policy Committee six claim ids, and not one of the thirteen claims
    in that store mentions the Monetary Policy Committee anywhere.
    """

    identity: str
    claim_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    disposition_axes: tuple[str, ...] = ()
    missing_disposition_axes: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        """Whether any verified claim speaks about this participant at all.

        False means the world contains a participant the evidence never mentions. That
        is a fabricated participant, whatever claim ids were stapled to it.
        """

        return bool(self.claim_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "supported": self.is_supported,
            "claim_ids": list(self.claim_ids),
            "lineage_ids": list(self.lineage_ids),
            "disposition_axes": list(self.disposition_axes),
            "missing_disposition_axes": list(self.missing_disposition_axes),
        }


def participant_evidence_support(
    view: EvidenceView, identities: tuple[str, ...]
) -> tuple[ParticipantSupport, ...]:
    """Trace every compiled participant back to the evidence that speaks about it.

    This is a *fact*, deliberately not a gate: it reports, per named participant, which
    available claims mention it, and what retrieval established about its disposition.
    The compiler reads it to know which of its participants it has nothing to go on for;
    a reviewer reads it to see an actor with an empty list and know the world invented
    somebody. Whether an unsupported participant is fatal is a decision for whoever owns
    the refusal, not for the function that establishes the fact.
    """

    out: list[ParticipantSupport] = []
    for identity in identities:
        claims = view.about(identity)
        disposition = view.disposition(identity)
        out.append(
            ParticipantSupport(
                identity=identity,
                claim_ids=tuple(sorted(c.id for c in claims)),
                lineage_ids=tuple(sorted({c.lineage_event_id for c in claims})),
                disposition_axes=tuple(a.axis.value for a in disposition.axes),
                missing_disposition_axes=tuple(a.value for a in disposition.missing_axes),
            )
        )
    return tuple(out)


def unsupported_participants(view: EvidenceView, identities: tuple[str, ...]) -> tuple[str, ...]:
    """Those named participants that no available claim speaks about.

    An empty result is meaningful here — ``identities`` is what was checked, so "nothing
    unsupported" is a statement about a list that was actually examined. Pass no
    identities and you get no answer, not a pass.
    """

    return tuple(
        s.identity for s in participant_evidence_support(view, identities) if not s.is_supported
    )


def _speaks_of_a_role(candidate: EvidenceCandidate) -> bool:
    """Whether the evidence describes this name in role or authority terms.

    Materiality is a low bar on purpose — a name mentioned alongside the subject is
    worth showing the compiler. Demanding a *seat* for that name is a far stronger
    claim, and mere co-occurrence does not support it: a claim about a production
    decision that also mentions a country and a quarter does not make either one a
    decision-maker.
    """

    return candidate.role_attested


# Calendar vocabulary. A capitalized two-token name can be a date ("July 2026", "Q1
# 2026"), and reading one as a person makes the reality gate demand a seat for a month.
# This is a property of how dates are written, not knowledge about any domain.
_MONTHS = frozenset(
    [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
)
_QUARTER = re.compile(r"^(q[1-4]|h[12]|fy)$", re.IGNORECASE)


def _is_calendar_shaped(identity: str) -> bool:
    tokens = identity.split()
    if not tokens:
        return True
    for token in tokens:
        bare = token.strip(",.").lower()
        if bare in _MONTHS or _QUARTER.match(bare) or bare.isdigit():
            return True
    return False


def participants_absent_from(
    identities: tuple[str, ...], roster_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Those ``identities`` that no name on ``roster_names`` refers to.

    Matching is the same normalized containment the coverage comparison uses, so an
    identity written "Governor Ada North" in one place and "Ada North" in the other is
    one person, not two.
    """

    roster = [n for n in (_norm(r) for r in roster_names) if n]
    return tuple(
        ident for ident in identities if not any(_overlaps(_norm(ident), name) for name in roster)
    )


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
    *,
    exclusion_reviewer: ExclusionReviewer | None = None,
    conflict_screen: ConflictScreen | None = None,
) -> CompilationCoverageReport:
    """Compare the candidate inventory against the compiled world and assign one
    disposition to every candidate. A material candidate that is not represented, or
    represented but not causally wired, is recorded as missing.

    ``exclusion_reviewer`` is an independent second opinion (a separate review pass
    from the compiler): given a candidate the deterministic rules would exclude as
    irrelevant, it answers whether omitting it could still plausibly change an actor's
    knowledge, authority, feasible actions, a constraint, a branch, timing, or the
    outcome. If it says yes, the exclusion is invalid — the candidate becomes
    UNCERTAIN and blocks, per the exclusion-challenge rule. Disagreement never
    silently resolves in favor of dropping the item.

    ``conflict_screen`` is the store's own record of which candidate conflicts were
    enumerated and ruled on. Supplied and inconclusive, the unexamined pairs are recorded
    as missing and the report is INCOMPLETE: a world compiled out of evidence whose
    disagreements nobody looked at has not been shown to be compiled out of anything
    coherent. Not supplied, the report says so in ``conflict_screening`` and in a note,
    and certifies coverage only — it never reports agreement it was not shown.
    """

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
            if cand.is_material:
                a.dispositions.append(
                    CandidateDisposition(
                        candidate_id=cand.candidate_id,
                        disposition=Disposition.REQUIRED_BUT_UNRESOLVED,
                        compiled_object_ids=tuple(o.object_id for o in matched),
                        reason="present in the world specification but disconnected from the simulation",
                    )
                )
                a.missing.append(_label(cand, "represented but not causally wired"))
            else:
                _record_exclusion(a, cand, exclusion_reviewer, objects=matched)
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
            _record_exclusion(a, cand, exclusion_reviewer)

    _record_conflict_screening(a, conflict_screen)
    return _report(candidates, a, conflict_screen)


def _record_conflict_screening(a: _Assessment, screen: ConflictScreen | None) -> None:
    """Say what is known about the evidence's own disagreements — including "nothing"."""

    if screen is None:
        a.notes.append(
            "conflict screening state was not supplied to this coverage assessment: this "
            "report certifies that compiled objects cover the evidence, and says nothing "
            "about whether that evidence agrees with itself"
        )
        return
    a.notes.append(f"conflict screening: {screen.summary()}")
    if screen.is_conclusive:
        return
    pairs = ", ".join(f"{c.a_id}<>{c.b_id}" for c in screen.unexamined[:12])
    a.missing.append(
        f"[evidence] {len(screen.unexamined)} candidate conflict(s) were never adjudicated "
        "— the evidence this world was compiled from has not been shown to agree with "
        f"itself ({pairs})"
    )


def _record_exclusion(
    a: _Assessment,
    cand: EvidenceCandidate,
    reviewer: ExclusionReviewer | None,
    *,
    objects: list[WorldObject] | None = None,
) -> None:
    """Record an EXCLUDED_IRRELEVANT disposition — unless an independent reviewer says
    the item could still matter, in which case the exclusion is invalid and it becomes
    UNCERTAIN and blocks (the exclusion challenge).

    Source provenance is not put to the reviewer at all. There is no compiled world in
    which "the document was published on 18 September 2025" could be represented, so a
    challenge to it could only ever be unsatisfiable — the compiler would be told to
    include something that is not world content, and no recompile could comply."""

    if reviewer is not None and not _is_source_provenance(cand) and reviewer(cand):
        a.dispositions.append(
            CandidateDisposition(
                candidate_id=cand.candidate_id,
                disposition=Disposition.UNCERTAIN,
                compiled_object_ids=tuple(o.object_id for o in (objects or [])),
                reason="exclusion challenged: an independent review found it could change the outcome",
                reviewer_stage="exclusion_challenge",
            )
        )
        a.missing.append(_label(cand, "exclusion rejected by independent review"))
        return
    a.dispositions.append(
        CandidateDisposition(
            candidate_id=cand.candidate_id,
            disposition=Disposition.EXCLUDED_IRRELEVANT,
            compiled_object_ids=tuple(o.object_id for o in (objects or [])),
            reason=(
                "present in the world specification but disconnected from the simulation"
                if objects
                else _exclusion_reason(cand)
            ),
        )
    )


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
    """Match a candidate to a structural object when claim ids and names do not overlap
    — the compiled object may not have been attributed the exact claim id.

    The only such match is a *two-sided* one: a rule candidate corresponds to a
    compiled term when the compiled term itself names the same decision procedure the
    evidence names (majority/unanimity/quorum/...). Both sides must use the word, so
    the compiled world has actually represented the procedure.

    There is deliberately no one-sided match. A scheduled event used to be counted as
    represented whenever its description merely contained a word like "decision" or
    "meeting", which matched it to the always-present terminal object: any such
    candidate was recorded INCLUDED while being wired into nothing, and the word list
    doing the deciding was a scenario assumption in code. A candidate now counts as
    represented only where it is actually wired — an entity, action, field, document,
    resource, process node, external process, or a terminal term that names it.
    """

    if cand.kind is CandidateKind.RULE and obj.kind in ("rule", "terminal"):
        desc_words = set(_WORD.findall(desc))
        shared = desc_words & _RULE_PROCEDURE_WORDS & set(_WORD.findall(_norm(obj.name)))
        return bool(shared)
    return False


def _label(cand: EvidenceCandidate, why: str) -> str:
    return f"[{cand.kind.value}] {cand.canonical_identity} — {why}"


def _exclusion_reason(cand: EvidenceCandidate) -> str:
    if _is_source_provenance(cand):
        return "provenance of a source, not a thing inside the world"
    if cand.kind is CandidateKind.PERSON:
        return "named incidentally; no role, vote, or membership signal in the evidence"
    return f"no outcome-relevant signal for this {cand.kind.value} in the evidence"


# A self-referential subject: the extractor talking about the page it was handed rather
# than about anything in the world. "The Bank of England published its minutes" names a
# real body and is world content; "the document was published" names nothing.
_SELF_REFERENCE = re.compile(
    r"\b(?:the|this)\s+(?:document|article|page|web\s?page|website|site|text|source|url)\b"
)
_PROVENANCE_PREDICATE = re.compile(
    r"\b(?:was|is|were|are|has\s+been)\s+(?:last\s+)?"
    r"(?:published|posted|updated|modified|dated|titled|entitled|written|authored|"
    r"bylined|retrieved|accessed|hosted|archived|captured)\b"
    r"|\b(?:appears?|appeared)\s+(?:on|at)\b"
    r"|\bcarries\s+a\s+byline\b"
    r"|\bhas\s+the\s+(?:url|title)\b"
)


def _is_source_provenance(cand: EvidenceCandidate) -> bool:
    """Whether this candidate describes a *source* rather than the world.

    When a page went online, who bylined it, what it is titled, where it lives: that is
    provenance. It is already recorded against every claim the page supports, and it is
    not a thing that exists inside the simulated world, so it can be neither material
    nor challenged into blocking a run. A live Bank of England run was refused because
    an independent reviewer challenged the exclusion of "The document was published on
    September 18, 2025" — there is no compiled world in which that could be represented.

    Both halves are required: a self-referential subject *and* a provenance predicate.
    A named body publishing a named document is an event in the world and is untouched.
    """

    text = " ".join(cand.description.split()).lower()
    # Drop the extractor's topic namespace ("context: ...") before matching.
    _, _, body = text.partition(": ")
    body = body or text
    return bool(_SELF_REFERENCE.search(body) and _PROVENANCE_PREDICATE.search(body))


def _report(
    candidates: tuple[EvidenceCandidate, ...],
    a: _Assessment,
    screen: ConflictScreen | None = None,
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
        conflict_screening=(
            screen.status if screen is not None else ConflictScreening.NOT_SCREENED
        ),
        conflict_screen_summary=(screen.summary() if screen is not None else ""),
    )


def enforce_coverage(report: CompilationCoverageReport) -> None:
    """Block simulation when verified reality was lost during compilation."""

    if report.is_complete:
        return
    raise WorldIntegrityError(
        "verified evidence was lost during world compilation — simulation refused",
        details={
            "failure": "coverage_incomplete",
            "missing_material_candidates": list(report.missing_material_candidates),
            "material_candidates": report.material_candidates,
            "included": report.included_candidates,
            "unresolved": report.unresolved_candidates,
            "uncertain": report.uncertain_candidates,
            "conflict_screening": report.conflict_screening.value,
            "conflicts_certified": report.conflicts_certified,
        },
    )
