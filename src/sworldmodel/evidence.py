"""The canonical evidence store, cutoff enforcement, and lineage de-duplication.

This is the first persistent structure built after the resolution contract. It is
never truncated to a fixed character budget: the *whole* structured store is kept,
and task-specific, token-limited *views* are derived from it on demand — each view
still renders full claims with their ids.

Cutoff enforcement is mechanical, not prompt-wording: downstream code only ever
receives an :class:`EvidenceView` bound to an ``as_of``. A claim whose
``available_at`` is after ``as_of`` is physically unreachable through that view,
even though it still lives in the store (for later, sealed, post-outcome analysis).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from .errors import CutoffViolationError, EvidenceError
from .models import AuthorityLevel, EpistemicType, SourceType

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _lex(words: str) -> frozenset[str]:
    return frozenset(words.split())


# Ordinary English function words, reporting verbs and modals. They are shared by any
# two texts on any subject, so their presence cannot show that two claims are about the
# same matter. Nothing here is about a subject area: this list must stay a fact about
# English, or a screen built on it becomes a scenario assumption in code.
_FUNCTION_WORDS = _lex(
    """a about according after already also amid an and are as at be been before being but by can
context could did do does during for from had has have having he her his how i in inference
into is it its just least less may might more most must my no nor not note of on only or our
over per previously remain remaining remains reported reporting reports said say saying says
shall she should so state stated statement states stating still such tells than that the
their them then these they this those to told under very was we were what when where which
while who whom whose why will with would yet you your"""
)


def _stem(word: str) -> str:
    """A crude, purely morphological suffix strip, so "cut" and "cuts" are one term.

    Deliberately not a lexicon: it applies English plural/participle endings and nothing
    else, so it carries no assumption about any subject area.
    """

    for suffix in ("ing", "ies"):
        if len(word) > 5 and word.endswith(suffix):
            return word[: -len(suffix)]
    for suffix in ("es", "ed"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _content_terms(text: str, *, drop: frozenset[str] = frozenset()) -> frozenset[str]:
    """The terms that say what a text is *about*: stemmed, minus function words.

    Numbers are kept whatever their length — "5 members" against "9 members" is the
    canonical disagreement, and a length filter would silently drop it.
    """

    out = {
        _stem(w)
        for w in _WORD.findall(text.lower())
        if w not in _FUNCTION_WORDS and (w.isdigit() or len(w) > 2)
    }
    return frozenset(out - drop)


def _claim_matter(text: str) -> str:
    """A claim's proposition with the extractor's namespace prefix removed.

    Extraction writes "context: ...", "statement: ...", "inference: ..." in front of
    many propositions. That prefix says which extraction slot the claim came out of, not
    what the claim is about, and treating it as the subject is what put "a cut is on the
    way" and "cuts are off the table" in different buckets so they were never compared.
    """

    head, sep, rest = text.partition(":")
    return rest.strip() if sep and rest.strip() and len(head.split()) <= 2 else text


@dataclass(frozen=True)
class EvidenceClaim:
    """A single, fully-attributed evidence claim.

    ``lineage_event_id`` ties claims that come from the *same underlying event*
    together. Multiple qualitative implications may be drawn from one meeting, but
    they share a lineage id and must never be counted as independent statistical
    cases.

    The provenance fields (``retrieved_url``, ``archived_at``, ``content_sha256``,
    ``extraction_prompt_sha256``) are what make a claim re-checkable: together with
    ``supporting_excerpt`` and ``retrieved_at`` they identify the exact document the
    claim came from, including which URL was actually requested when that differs from
    the publisher's URL. They default to empty for stores built from an authored corpus,
    where the corpus file is itself the record.
    """

    id: str
    proposition: str
    normalized_value: str
    entities: tuple[str, ...]
    valid_from: datetime | None
    valid_until: datetime | None
    published_at: datetime
    available_at: datetime
    source_id: str
    source_url: str
    source_title: str
    source_type: SourceType
    authority_level: AuthorityLevel
    supporting_excerpt: str
    lineage_event_id: str
    epistemic_type: EpistemicType
    confidence: float
    retrieved_at: datetime
    contradiction_ids: tuple[str, ...] = ()
    retrieved_url: str = ""
    archived_at: datetime | None = None
    content_sha256: str = ""
    extraction_prompt_sha256: str = ""

    def provenance(self) -> dict[str, object]:
        """Everything a reader needs to re-verify this claim against its document."""

        return {
            "claim_id": self.id,
            "proposition": self.proposition,
            "normalized_value": self.normalized_value,
            "source_url": self.source_url,
            "retrieved_url": self.retrieved_url or self.source_url,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "published_at": self.published_at.isoformat(),
            "available_at": self.available_at.isoformat(),
            "content_sha256": self.content_sha256,
            "extraction_prompt_sha256": self.extraction_prompt_sha256,
            "supporting_excerpt": self.supporting_excerpt,
            "source_type": self.source_type.value,
            "authority_level": int(self.authority_level),
            "lineage_event_id": self.lineage_event_id,
            "contradiction_ids": list(self.contradiction_ids),
        }

    def __post_init__(self) -> None:
        if self.available_at < self.published_at:
            # A fact cannot be *available* before it was published.
            raise EvidenceError(
                f"Claim {self.id}: available_at ({self.available_at}) precedes "
                f"published_at ({self.published_at})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise EvidenceError(f"Claim {self.id}: confidence must be in [0,1]")

    def is_available_at(self, as_of: datetime) -> bool:
        return self.available_at <= as_of

    def render(self) -> str:
        """Full, id-tagged rendering for a prompt view."""

        tags = f"[{self.id} · {self.epistemic_type.value} · {self.source_type.value}]"
        return f"{tags} {self.proposition} (value={self.normalized_value})"

    def tokens(self) -> list[str]:
        return _tokens(self.proposition + " " + " ".join(self.entities))


# ---------------------------------------------------------------------------
# Conflict screening: what was examined, and what was not
# ---------------------------------------------------------------------------


class ConflictVerdict(StrEnum):
    """What an adjudicator concluded about one candidate pair.

    ``UNADJUDICATED`` is the starting state and it is not a clean bill of health: it
    means the pair was found and nobody has ruled on it yet.
    """

    UNADJUDICATED = "unadjudicated"
    DECISIVE = "decisive"  # the world cannot have it both ways; blocks
    RECONCILED = "reconciled"  # both readings can be true at once


class ConflictScreening(StrEnum):
    """How far conflict screening got on a store. The whole point of this enum is that
    ``NOT_SCREENED`` and ``FULLY_ADJUDICATED`` can never render as the same empty list."""

    NOT_SCREENED = "not_screened"
    SCREENED_UNADJUDICATED = "screened_unadjudicated"
    PARTIALLY_ADJUDICATED = "partially_adjudicated"
    FULLY_ADJUDICATED = "fully_adjudicated"


@dataclass(frozen=True)
class ConflictCandidate:
    """Two claims about the same matter carrying different answers.

    A candidate is **not** an assertion that the claims contradict each other. It is a
    deterministic finding that they are comparable and that they differ, which is the
    precondition for an adjudicator to rule. Precision belongs to the adjudicator; the
    screen's job is recall, because a pair that is never enumerated is never examined
    and its absence is indistinguishable from agreement.
    """

    a_id: str
    b_id: str
    shared_subject: tuple[str, ...]  # the entities both claims are about
    shared_matter: tuple[str, ...]  # the content terms that make them comparable
    a_value: str
    b_value: str
    verdict: ConflictVerdict = ConflictVerdict.UNADJUDICATED
    adjudicator: str = ""
    reason: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return (self.a_id, self.b_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a_id,
            "b": self.b_id,
            "shared_subject": list(self.shared_subject),
            "shared_matter": list(self.shared_matter),
            "a_value": self.a_value,
            "b_value": self.b_value,
            "verdict": self.verdict.value,
            "adjudicator": self.adjudicator,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConflictScreen:
    """The record of one conflict screening pass: what was enumerated, and what a
    verdict was actually reached on.

    This exists because an empty conflict list has been read three separate ways in this
    repository — "nobody looked", "we looked and could not decide", and "we looked and
    it is clean" — and only the third licenses calling a store verified. Every consumer
    reads :attr:`status` rather than inferring from an empty list.
    """

    as_of: datetime | None
    screened_claim_ids: tuple[str, ...]
    candidates: tuple[ConflictCandidate, ...] = ()
    screener: str = "same_subject_different_answer"

    @property
    def unexamined(self) -> tuple[ConflictCandidate, ...]:
        """The candidate pairs nobody has ruled on. Non-empty means the screen is not a
        finding yet, whatever the recorded contradictions say."""

        return tuple(c for c in self.candidates if c.verdict is ConflictVerdict.UNADJUDICATED)

    @property
    def adjudicated(self) -> tuple[ConflictCandidate, ...]:
        return tuple(c for c in self.candidates if c.verdict is not ConflictVerdict.UNADJUDICATED)

    @property
    def decisive(self) -> tuple[ConflictCandidate, ...]:
        return tuple(c for c in self.candidates if c.verdict is ConflictVerdict.DECISIVE)

    @property
    def status(self) -> ConflictScreening:
        if not self.candidates:
            # Nothing to rule on: the screen ran over these claims and found no pair that
            # is even comparable-and-different. That IS a finding, and a conclusive one.
            return ConflictScreening.FULLY_ADJUDICATED
        if not self.adjudicated:
            return ConflictScreening.SCREENED_UNADJUDICATED
        if self.unexamined:
            return ConflictScreening.PARTIALLY_ADJUDICATED
        return ConflictScreening.FULLY_ADJUDICATED

    @property
    def is_conclusive(self) -> bool:
        """True only when every enumerated candidate carries a verdict.

        A caller may report "this store has no unresolved conflicts" if and only if this
        is True. Otherwise the honest statement names how many pairs went unexamined.
        """

        return self.status is ConflictScreening.FULLY_ADJUDICATED

    def summary(self) -> str:
        if not self.candidates:
            return (
                f"{len(self.screened_claim_ids)} claims screened; no pair was both "
                "comparable and in disagreement"
            )
        return (
            f"{len(self.screened_claim_ids)} claims screened; {len(self.candidates)} "
            f"candidate conflict(s), {len(self.adjudicated)} adjudicated, "
            f"{len(self.unexamined)} NOT examined, {len(self.decisive)} decisive"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "screener": self.screener,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "claims_screened": len(self.screened_claim_ids),
            "status": self.status.value,
            "conclusive": self.is_conclusive,
            "candidates": [c.as_dict() for c in self.candidates],
            "unexamined_pairs": [list(c.pair) for c in self.unexamined],
            "decisive_pairs": [list(c.pair) for c in self.decisive],
            "summary": self.summary(),
        }


def screen_claim_conflicts(claims: list[EvidenceClaim]) -> tuple[ConflictCandidate, ...]:
    """Enumerate pairs of claims that are about the same matter and give different answers.

    Two claims are a candidate when all three hold:

    * they name at least one entity in common — they have a shared subject;
    * their propositions share at least one content term beyond that subject's own name
      — they are about the same matter, not merely about the same person;
    * their normalized values are not the same answer.

    Deliberately deterministic and lexicon-free beyond ordinary English function words.
    It never decides that two claims contradict each other: "different answer to the same
    question" is what can be established without judgement, and judgement is the
    adjudicator's job. Deciding here that one value is merely more detail than another —
    "chair" against "former chair" looks exactly like "3.75%" against "3.75% after the
    vote" — would be that judgement smuggled into a screen, so it is not attempted.
    """

    out: list[ConflictCandidate] = []
    for a, b in itertools.combinations(sorted(claims, key=lambda c: c.id), 2):
        subject = {e.strip().lower() for e in a.entities} & {e.strip().lower() for e in b.entities}
        if not subject:
            continue
        if a.normalized_value.strip().lower() == b.normalized_value.strip().lower():
            continue
        subject_terms = frozenset(_stem(w) for name in subject for w in _WORD.findall(name.lower()))
        matter = _content_terms(_claim_matter(a.proposition), drop=subject_terms) & _content_terms(
            _claim_matter(b.proposition), drop=subject_terms
        )
        if not matter:
            continue
        if _content_terms(a.normalized_value, drop=subject_terms) == _content_terms(
            b.normalized_value, drop=subject_terms
        ):
            # The same answer written two ways is not a disagreement.
            continue
        out.append(
            ConflictCandidate(
                a_id=a.id,
                b_id=b.id,
                shared_subject=tuple(sorted(subject)),
                shared_matter=tuple(sorted(matter)),
                a_value=a.normalized_value,
                b_value=b.normalized_value,
            )
        )
    # Strongest shared matter first: that is the order an adjudicator with a bounded
    # budget should spend it in, so a truncated pass examines the most comparable pairs.
    out.sort(key=lambda c: (-len(c.shared_matter), c.a_id, c.b_id))
    return tuple(out)


@dataclass
class EvidenceStore:
    """The complete, mutable-at-build-time canonical store. After it is built it is
    only ever *read* (through views); the runtime never edits claims in place."""

    claims: dict[str, EvidenceClaim] = field(default_factory=dict)
    # ``None`` means conflict screening never ran on this store. It is not "clean".
    conflict_screen: ConflictScreen | None = None

    def add(self, claim: EvidenceClaim) -> None:
        if claim.id in self.claims:
            raise EvidenceError(f"Duplicate claim id {claim.id}")
        self.claims[claim.id] = claim

    def get(self, claim_id: str) -> EvidenceClaim:
        try:
            return self.claims[claim_id]
        except KeyError as exc:
            raise EvidenceError(f"Unknown claim id {claim_id}") from exc

    def all(self) -> list[EvidenceClaim]:
        return list(self.claims.values())

    def lineage_groups(self) -> dict[str, list[EvidenceClaim]]:
        """Group claims by their underlying event id."""

        groups: dict[str, list[EvidenceClaim]] = {}
        for claim in self.claims.values():
            groups.setdefault(claim.lineage_event_id, []).append(claim)
        return groups

    def independent_event_ids(self) -> set[str]:
        """The set of distinct underlying events. ``len()`` of this is the maximum
        number of *independent* observations available — not the claim count."""

        return {c.lineage_event_id for c in self.claims.values()}

    def contradictions(self) -> list[tuple[str, str]]:
        """Return decisive contradiction pairs (a_id, b_id) recorded on claims.

        This reads only what :func:`record_contradiction` (or a corpus) put there. A
        caller that reports "no conflicts" from an empty result is only entitled to do
        so if contradiction detection actually ran on the claims in this store.
        """

        pairs: set[tuple[str, str]] = set()
        for claim in self.claims.values():
            for other in claim.contradiction_ids:
                pairs.add(tuple(sorted((claim.id, other))))  # type: ignore[arg-type]
        return sorted(pairs)

    def record_contradiction(self, a_id: str, b_id: str) -> None:
        """Record a decisive contradiction between two stored claims, symmetrically."""

        if a_id == b_id:
            raise EvidenceError(f"Claim {a_id} cannot contradict itself")
        na, nb = mark_contradiction(self.get(a_id), self.get(b_id))
        self.claims[na.id] = na
        self.claims[nb.id] = nb

    # -- conflict screening ---------------------------------------------------

    def screen_conflicts(self, as_of: datetime | None = None) -> ConflictScreen:
        """Enumerate every candidate conflict in this store and remember that it ran.

        Bounded by ``as_of`` when given, so a post-cutoff claim cannot manufacture a
        conflict that blocks a pastcast. Re-screening after new claims arrive keeps the
        verdicts already reached, so a growing store never loses adjudications — and
        never inherits one for a pair that did not exist when the earlier pass ran.
        """

        claims = self.view(as_of).available() if as_of is not None else self.all()
        prior = {
            c.pair: c for c in (self.conflict_screen.candidates if self.conflict_screen else ())
        }
        candidates = tuple(
            replace(
                cand,
                verdict=prior[cand.pair].verdict,
                adjudicator=prior[cand.pair].adjudicator,
                reason=prior[cand.pair].reason,
            )
            if cand.pair in prior
            else cand
            for cand in screen_claim_conflicts(claims)
        )
        screen = ConflictScreen(
            as_of=as_of,
            screened_claim_ids=tuple(sorted(c.id for c in claims)),
            candidates=candidates,
        )
        self.conflict_screen = screen
        return screen

    def conflicts_examined(self) -> bool:
        """Whether anything ever looked for conflicts in this store at all."""

        return self.conflict_screen is not None

    def record_adjudication(
        self,
        a_id: str,
        b_id: str,
        *,
        decisive: bool,
        adjudicator: str,
        reason: str = "",
    ) -> None:
        """Record one adjudicator's verdict on a screened candidate pair.

        A decisive verdict is also written onto the claims themselves, so
        :meth:`contradictions` keeps returning exactly what it always returned. A
        non-decisive verdict changes no claim — but it does move the pair out of
        ``unexamined``, which is what turns an empty conflict list into a finding.
        """

        screen = self.conflict_screen
        if screen is None:
            raise EvidenceError(
                "cannot adjudicate a conflict before screening: call screen_conflicts() "
                "first, so the pair being ruled on is one this store actually enumerated"
            )
        pair = (a_id, b_id)
        if not any(c.pair == pair for c in screen.candidates):
            raise EvidenceError(f"pair {a_id}<>{b_id} is not a screened candidate of this store")
        verdict = ConflictVerdict.DECISIVE if decisive else ConflictVerdict.RECONCILED
        self.conflict_screen = replace(
            screen,
            candidates=tuple(
                replace(c, verdict=verdict, adjudicator=adjudicator, reason=reason)
                if c.pair == pair
                else c
                for c in screen.candidates
            ),
        )
        if decisive:
            self.record_contradiction(a_id, b_id)

    def conflict_certificate(self) -> dict[str, object]:
        """What this store is entitled to say about its own conflicts.

        Callers must read ``conclusive`` before reporting "no unresolved conflicts". An
        unscreened store answers ``screened: False`` — which is the whole difference
        between a check that declined to look and a check that looked and approved.
        """

        screen = self.conflict_screen
        if screen is None:
            return {
                "screened": False,
                "conclusive": False,
                "status": ConflictScreening.NOT_SCREENED.value,
                "recorded_contradictions": [list(p) for p in self.contradictions()],
                "summary": (
                    f"{len(self.claims)} claims were never screened for conflicts; this "
                    "store makes no claim about whether they disagree"
                ),
            }
        return {
            "screened": True,
            "conclusive": screen.is_conclusive,
            "status": screen.status.value,
            "recorded_contradictions": [list(p) for p in self.contradictions()],
            "summary": screen.summary(),
            "screen": screen.as_dict(),
        }

    def provenance_records(self) -> list[dict[str, object]]:
        """Per-claim provenance for the audit trail, ordered by claim id."""

        return [c.provenance() for c in sorted(self.claims.values(), key=lambda c: c.id)]

    def view(self, as_of: datetime) -> EvidenceView:
        """Return the cutoff-bounded, read-only view used by all downstream code."""

        return EvidenceView(store=self, as_of=as_of)


# ---------------------------------------------------------------------------
# The disposition axis: what a participant wants, has done, is bound by, and how it
# has reacted
# ---------------------------------------------------------------------------


class DispositionAxis(StrEnum):
    """The four things a simulated participant needs about itself in order to act like
    itself rather than like a generic occupant of its role.

    Retrieval that establishes only existence produces an agent that can be *named* and
    not *played*: it knows the nine OPEC+ members are in the room and nothing about what
    Saudi Arabia would push for or what Russia would resist. These axes are the evidence
    that difference is made of. They are roles a claim plays about a participant, not
    subject-area categories — the same four apply to a central banker, a trade
    negotiator, a producer state and a constructed population stratum.
    """

    WANTS = "wants"  # stated position, preference, goal or intent
    HAS_DONE = "has_done"  # what it actually did before, in this kind of situation
    CONSTRAINED_BY = "constrained_by"  # mandate, rule, limit or obligation binding it
    REACTS_TO = "reacts_to"  # how it responded when comparable moves were made

    @property
    def question(self) -> str:
        return _AXIS_QUESTIONS[self]


_AXIS_QUESTIONS: dict[DispositionAxis, str] = {
    DispositionAxis.WANTS: "what does this participant want, prefer or push for",
    DispositionAxis.HAS_DONE: "what has this participant previously done in this situation",
    DispositionAxis.CONSTRAINED_BY: "what mandate, rule or limit binds this participant",
    DispositionAxis.REACTS_TO: "how has this participant responded to comparable moves",
}

# Search phrasings per axis. These are what makes disposition something retrieval goes
# LOOKING for, rather than something it hopes falls out of a topic query.
_AXIS_QUERY_FORMS: dict[DispositionAxis, tuple[str, ...]] = {
    DispositionAxis.WANTS: ("{who} what it wants position", "{who} said wants preference goal"),
    DispositionAxis.HAS_DONE: (
        "{who} previous decision track record",
        "{who} last time voted acted",
    ),
    DispositionAxis.CONSTRAINED_BY: (
        "{who} mandate rules constraints limits",
        "{who} required by law obliged",
    ),
    DispositionAxis.REACTS_TO: ("{who} responded reaction to", "{who} how it reacted previously"),
}

# Per-axis vocabulary. Each list is ordinary English for a *mode of speaking about an
# agent* — intending, having acted, being bound, responding — and carries no subject
# matter. A word only counts when it sits beside the participant's own name (see
# :func:`_beside`), which is the difference between "Bailey signalled" and a sentence
# that happens to contain both "Bailey" and "signalled" about different things.
_WANT_WORDS = _lex(
    """advocate advocated advocates aim aimed aims ambition appropriate argue argued argues back
backed backs believe believed believes call called calls demand demanded demands described
describes expect expected expects favor favored favors favour favoured favours goal goals
hint hinted hints indicate indicated indicates insist insisted insists intend intended
intends objective objectives oppose opposed opposes plan planned plans prefer preferred
prefers prepared priorities priority push pushed pushes ready remarked resist resisted
resists said say says seek seeking seeks signal signaled signalled signals stated states
stating support supported supports target told urge urged urges view views want wanted wants
warned welcomed willing"""
)
# Past-tense and participle forms only. A bare stem like "cut", "hold" or "vote" is as
# often the noun for a future move as the verb for a past one, and reading "a rate cut
# is on the way" as a record of prior conduct is inventing a disposition — the exact
# failure this axis exists to avoid.
_PRIOR_ACTION_WORDS = _lex(
    """abstained agreed already announced appointed approved began blocked decided delayed earlier
endorsed held hiked imposed issued kept launched lifted lowered maintained nominated passed
postponed precedent previously published raised refused rejected released removed resigned
reversed ruled signed stopped vetoed voted withdrew"""
)
_CONSTRAINT_WORDS = _lex(
    """allowed authorised authority authorized binding bound cannot cap ceiling conditional
constrained constraint contingent deadline depends eligible entitled floor forbidden law
laws limit limited limits majority mandate mandated mandates must obligation obligations
obliged permitted prohibited quorum remit required requirement requires restricted rule
rules statute statutory subject threshold unable unanimity"""
)
_REACTION_WORDS = _lex(
    """adjust adjusted adjusts answered backlash change changed changes counter countered
countering followed following move moved moves prompted pushback react reacted reaction
reacts replied respond responded responds response retaliate retaliated retaliation revise
revised revises shift shifted shifts triggered"""
)

_AXIS_WORDS: dict[DispositionAxis, frozenset[str]] = {
    DispositionAxis.WANTS: _WANT_WORDS,
    DispositionAxis.HAS_DONE: _PRIOR_ACTION_WORDS,
    DispositionAxis.CONSTRAINED_BY: _CONSTRAINT_WORDS,
    DispositionAxis.REACTS_TO: _REACTION_WORDS,
}


def _name_forms(identity: str) -> tuple[str, ...]:
    """The forms a source may use for this participant: the full name and its own
    distinctive parts.

    Sources shorten names — "Andrew Bailey" is "Bailey" from the second paragraph on —
    and a proximity test keyed only on the full form misses everything after the first
    mention. Parts are only ever used *within a claim already established to be about
    this participant*, so a shared surname cannot pull in somebody else's record.
    """

    whole = identity.strip().lower()
    if not whole:
        return ()
    parts = [p for p in _WORD.findall(whole) if len(p) > 2 and p not in _FUNCTION_WORDS]
    return tuple(dict.fromkeys([whole, *parts]))


def disposition_queries(participant: str, *, forms_per_axis: int = 1) -> tuple[str, ...]:
    """The searches that would establish one participant's disposition.

    Retrieval has to *ask* what a participant wants, has done, is bound by and how it
    reacted. A topic query returns the participants; only these return the material that
    lets one of them be played rather than named — which is why they are issued rather
    than hoped for. ``forms_per_axis`` bounds the cost: one phrasing per axis is the
    floor at which every axis is still searched.
    """

    who = participant.strip()
    if not who:
        return ()
    return tuple(
        dict.fromkeys(
            form.format(who=who)
            for axis in DispositionAxis
            for form in _AXIS_QUERY_FORMS[axis][: max(1, forms_per_axis)]
        )
    )


def _beside(identity: str, text: str, lexicon: frozenset[str]) -> bool:
    """Whether a word from ``lexicon`` sits next to this participant's name, in one clause.

    A bag-of-words test over a whole passage attributes every verb in a paragraph to
    every name in it. Requiring the word within a short span of the name, either side, is
    what keeps "Bailey signalled readiness" apart from "the market signalled, and Bailey
    was in Basel".

    When no form of the name occurs in the text at all, the claim's own attribution is
    the only attribution available, and the lexicon is matched against the whole text.
    That is the honest fallback: this text was retrieved *because* the claim declares
    itself to be about this participant.
    """

    low = text.lower()
    words = rf"\b(?:{'|'.join(sorted(lexicon))})\b"
    forms = _name_forms(identity)
    present = [
        f for f in forms if re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", low) is not None
    ]
    if not present:
        return re.search(words, low) is not None
    for form in present:
        name = rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])"
        if re.search(rf"{name}[^.;]{{0,64}}?{words}", low) or re.search(
            rf"{words}[^.;]{{0,64}}?{name}", low
        ):
            return True
    return False


@dataclass(frozen=True)
class AxisEvidence:
    """Everything the evidence records about one participant along one axis.

    ``contested`` means the claims on this axis do not agree with each other. That is a
    finding to hand the compiler, not a problem to resolve by picking the newest or the
    most authoritative: a participant whose record disagrees about where it stands is
    exactly the participant a simulation exists to play forward.
    """

    axis: DispositionAxis
    claim_ids: tuple[str, ...]
    values: tuple[str, ...]
    renderings: tuple[str, ...]
    contested: bool = False
    conflicting_pairs: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "question": self.axis.question,
            "claim_ids": list(self.claim_ids),
            "values": list(self.values),
            "contested": self.contested,
            "conflicting_pairs": [list(p) for p in self.conflicting_pairs],
            "renderings": list(self.renderings),
        }


@dataclass(frozen=True)
class ParticipantDisposition:
    """What retrieval established about one participant's disposition, and what it did not.

    ``missing_axes`` is the load-bearing field. An axis with no supporting claim is
    reported missing and left empty — never filled from the role, the institution, or
    what a participant of this kind usually wants. An invented disposition is worse than
    a missing one: it is a fabricated value wearing a citation, which is this
    repository's entire defect history.
    """

    participant: str
    axes: tuple[AxisEvidence, ...] = ()
    missing_axes: tuple[DispositionAxis, ...] = ()
    searched_axes: tuple[DispositionAxis, ...] = ()

    @property
    def is_grounded(self) -> bool:
        return bool(self.axes)

    @property
    def supporting_claim_ids(self) -> tuple[str, ...]:
        return tuple(sorted({cid for a in self.axes for cid in a.claim_ids}))

    @property
    def contested_axes(self) -> tuple[DispositionAxis, ...]:
        return tuple(a.axis for a in self.axes if a.contested)

    @property
    def conflicting_claim_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted({p for a in self.axes for p in a.conflicting_pairs}))

    def axis(self, axis: DispositionAxis) -> AxisEvidence | None:
        return next((a for a in self.axes if a.axis is axis), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "participant": self.participant,
            "grounded": self.is_grounded,
            "axes": [a.as_dict() for a in self.axes],
            "missing_axes": [a.value for a in self.missing_axes],
            "searched_axes": [a.value for a in self.searched_axes],
            "contested_axes": [a.value for a in self.contested_axes],
            "supporting_claim_ids": list(self.supporting_claim_ids),
        }


@dataclass(frozen=True)
class EvidenceView:
    """A read-only projection of the store bounded by ``as_of``.

    Post-cutoff claims are simply not returned by any accessor here. Attempting to
    fetch one by id raises :class:`CutoffViolationError` so a leak fails loudly
    instead of silently poisoning a pastcast.
    """

    store: EvidenceStore
    as_of: datetime

    def available(self) -> list[EvidenceClaim]:
        return [c for c in self.store.all() if c.is_available_at(self.as_of)]

    def get(self, claim_id: str) -> EvidenceClaim:
        claim = self.store.get(claim_id)
        if not claim.is_available_at(self.as_of):
            raise CutoffViolationError(
                f"Claim {claim_id} (available_at={claim.available_at.isoformat()}) is "
                f"after the cutoff {self.as_of.isoformat()} and cannot be used"
            )
        return claim

    def by_entity(self, entity: str) -> list[EvidenceClaim]:
        e = entity.lower()
        return [c for c in self.available() if any(e == x.lower() for x in c.entities)]

    def observations(self) -> list[EvidenceClaim]:
        return [c for c in self.available() if c.epistemic_type is EpistemicType.OBSERVATION]

    def independent_event_ids(self) -> set[str]:
        return {c.lineage_event_id for c in self.available()}

    def relevant(self, query: str, *, limit: int) -> list[EvidenceClaim]:
        """Rank available claims by lexical relevance to ``query``, then by authority.

        This is an *operational* retrieval heuristic; it encodes no social outcome. The
        two criteria are applied in order rather than combined by a weight, so no
        invented exchange rate decides how much authority is worth how much overlap. It
        never deletes anything: it selects the top ``limit`` claims and renders them
        fully, ids intact.
        """

        q = set(_tokens(query))
        ordered = sorted(
            self.available(),
            key=lambda c: (-len(q & set(c.tokens())), -int(c.authority_level), c.id),
        )
        return ordered[:limit]

    def render_view(self, query: str, *, limit: int) -> tuple[str, tuple[str, ...]]:
        """Render a token-limited, id-preserving evidence view for a prompt.

        Returns ``(text, claim_ids)``. The canonical store is untouched.
        """

        claims = self.relevant(query, limit=limit)
        text = "\n".join(c.render() for c in claims)
        return text, tuple(c.id for c in claims)

    # -- the disposition axis -------------------------------------------------

    def about(self, participant: str) -> list[EvidenceClaim]:
        """Available claims that are actually *about* this participant.

        Either it is one of the claim's declared entities, or its name occurs as a whole
        phrase in the claim's own text. Substring matching is not used: it is how "Ada"
        came to match inside "Adamant" and how an actor acquired somebody else's record.
        """

        key = participant.strip().lower()
        if not key:
            return []
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])")
        out: list[EvidenceClaim] = []
        for claim in self.available():
            if (
                any(key == e.strip().lower() for e in claim.entities)
                or pattern.search(claim.proposition.lower())
                or pattern.search(claim.supporting_excerpt.lower())
            ):
                out.append(claim)
        return out

    def disposition(self, participant: str) -> ParticipantDisposition:
        """What the evidence establishes about how this participant would act.

        Every axis is searched. An axis with supporting claims is returned with them and
        with their claim ids; an axis with none is returned in ``missing_axes`` and left
        empty. Nothing is inferred across axes and nothing is filled in from the
        participant's role — an actor's disposition is either cited or absent.

        Where the claims on one axis disagree with each other, the axis is marked
        ``contested`` and carries the conflicting pairs. This is the same deterministic
        screen the store uses for conflicts, so a disagreement that is material enough to
        contest a disposition is the same disagreement the conflict certificate reports.
        """

        claims = self.about(participant)
        axes: list[AxisEvidence] = []
        missing: list[DispositionAxis] = []
        for axis in DispositionAxis:
            words = _AXIS_WORDS[axis]
            hits = [
                c
                for c in claims
                if _beside(participant, f"{c.proposition} {c.supporting_excerpt}", words)
            ]
            if not hits:
                missing.append(axis)
                continue
            conflicts = screen_claim_conflicts(hits)
            axes.append(
                AxisEvidence(
                    axis=axis,
                    claim_ids=tuple(sorted(c.id for c in hits)),
                    values=tuple(dict.fromkeys(c.normalized_value for c in hits)),
                    renderings=tuple(c.render() for c in sorted(hits, key=lambda c: c.id)),
                    contested=bool(conflicts),
                    # Screen order: strongest shared matter first, so a reader with
                    # limited attention sees the sharpest disagreement first.
                    conflicting_pairs=tuple(c.pair for c in conflicts),
                )
            )
        return ParticipantDisposition(
            participant=participant,
            axes=tuple(axes),
            missing_axes=tuple(missing),
            searched_axes=tuple(DispositionAxis),
        )

    def disposition_queries(self, participant: str) -> tuple[str, ...]:
        """The searches that would establish this participant's disposition."""

        return disposition_queries(participant, forms_per_axis=2)


def check_lineage_independence(
    claims: list[EvidenceClaim],
) -> tuple[int, int]:
    """Return ``(claim_count, independent_event_count)``.

    A caller that wants to treat these claims as independent samples must use the
    *event* count, never the claim count.
    """

    return len(claims), len({c.lineage_event_id for c in claims})


def mark_contradiction(a: EvidenceClaim, b: EvidenceClaim) -> tuple[EvidenceClaim, EvidenceClaim]:
    """Return copies of ``a`` and ``b`` with each other recorded as a contradiction."""

    return (
        replace(a, contradiction_ids=tuple(sorted({*a.contradiction_ids, b.id}))),
        replace(b, contradiction_ids=tuple(sorted({*b.contradiction_ids, a.id}))),
    )
