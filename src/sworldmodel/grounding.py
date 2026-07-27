"""Actor grounding: model the specific real person, not a generic role template.

An audit of a completed live run found that named actors were only weakly grounded.
Every actor's compiled ``current_inclination`` had been flattened to the board's common
position, overwriting each member's *verified* previous vote; every actor carried the
same frame-level reaction rules; and the previous observed action never appeared in the
prompt at all. The five prompts were ~80% identical — differing essentially in name,
role, authority, and retrieved memories.

This module fixes that generally. It builds a typed :class:`ActorGroundingProfile` per
actor in which every element carries an explicit epistemic mark:

    VERIFIED_OBSERVATION  — directly supported by evidence claims that survived the
                            store's citation check
    SUPPORTED_INFERENCE   — reasoned from surviving cited evidence, labeled as inference
    UNSUPPORTED           — no citation survived; kept for the audit trail, never
                            rendered into an actor's prompt

There is no mark for "not found". What was looked for and not found is recorded in
``missing_information`` and rendered to the actor as such — it is not an element of
the profile, because there is nothing to be an element of.

Three rules are structural, not stylistic:

* A mark may never outrun its citations. An item marked VERIFIED_OBSERVATION or
  SUPPORTED_INFERENCE without a surviving claim id cannot be constructed at all
  (:meth:`GroundedItem.__post_init__` raises), because the compiler upstream deletes
  claim ids that are not in the evidence store — so an actor memory whose citations
  were stripped would otherwise reach the live actor stamped as an observed fact.
* A verified historical action may never be overwritten by an inferred current
  position. ``previous_observed_actions`` and ``current_evidence_grounded_inclination``
  are separate fields with separate provenance.
* Similarity between actors is allowed — actors genuinely share positions — but it must
  come from evidence, never from a generic template. An actor that carries no cited,
  self-attributed record of its own is a defect the gate refuses, naming the actor.

Nothing here is scenario-specific: the same structure grounds a committee member, a
head of state, an organization's representative, or a constructed population stratum.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .epistemics import (
    EpistemicClass,
    GroundingAssessment,
    GroundingLevel,
    is_first_person,
    private_state_class,
)
from .errors import CutoffViolationError, EvidenceError, WorldIntegrityError
from .evidence import DispositionAxis, EvidenceView, ParticipantDisposition

_WORD = re.compile(r"[a-z0-9]+")


class ClaimAttestation(StrEnum):
    """Whether this actor's cited claims actually mention this actor.

    An actor's ``claim_ids`` are *assigned* by the compiler. Nothing about that assignment
    establishes that the claims say anything about the actor, and a live Bank of England
    world put the Monetary Policy Committee into the simulation at grounding level
    OFFICIAL_ROLE carrying six claim ids, none of which mentions a committee anywhere.

    The tri-state is the point. ``NOT_CHECKED`` is the default and must never read as a
    pass: it says nobody compared the claims against the name. Only :func:`attest_profiles`
    can move a profile off it, and only ``UNATTESTED`` withdraws the office.
    """

    NOT_CHECKED = "not_checked"
    ATTESTED = "attested"
    UNATTESTED = "unattested"


class Provenance(StrEnum):
    """How a grounding element is known. An inference is never presented as a fact,
    and an uncited item is never presented as either."""

    VERIFIED_OBSERVATION = "verified_observation"
    SUPPORTED_INFERENCE = "supported_inference"
    UNSUPPORTED = "unsupported"


# The marks that assert evidential support. Both require at least one surviving claim id.
_SUPPORTED = frozenset({Provenance.VERIFIED_OBSERVATION, Provenance.SUPPORTED_INFERENCE})


@dataclass(frozen=True)
class GroundedItem:
    """One grounding element with its epistemic status and evidence lineage."""

    content: str
    provenance: Provenance
    claim_ids: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()
    valid_time: str | None = None

    def __post_init__(self) -> None:
        # The mark exists to track the citations that actually survived. A supported
        # mark with no claim id would be rendered to a live actor as established fact
        # with nothing behind it, which is the worst failure this system has.
        if self.provenance in _SUPPORTED and not self.claim_ids:
            raise WorldIntegrityError(
                f"a {self.provenance.value} element must cite at least one evidence claim",
                details={"content": self.content},
            )

    @property
    def is_verified(self) -> bool:
        return self.provenance is Provenance.VERIFIED_OBSERVATION

    @property
    def is_supported(self) -> bool:
        """True when evidence actually stands behind this item. Only supported items
        are rendered into an actor's prompt."""

        return self.provenance in _SUPPORTED

    def render(self) -> str:
        """Render with the epistemic mark and its citations visible, so a model reading
        the prompt can never mistake an inference for an observed fact."""

        cites = f" [{','.join(self.claim_ids)}]" if self.claim_ids else ""
        when = f" ({self.valid_time})" if self.valid_time else ""
        return f"{self.content}{when} — {self.provenance.value.upper()}{cites}"


def observation(
    content: str, claim_ids: tuple[str, ...], *, valid_time: str | None = None
) -> GroundedItem:
    """Something the evidence records about this actor.

    VERIFIED_OBSERVATION when at least one citation survived the evidence store's
    check; UNSUPPORTED when none did. The mark is decided by the citations, never by
    the caller's confidence.
    """

    mark = Provenance.VERIFIED_OBSERVATION if claim_ids else Provenance.UNSUPPORTED
    return GroundedItem(content, mark, claim_ids, valid_time=valid_time)


def inference(content: str, claim_ids: tuple[str, ...]) -> GroundedItem:
    """Something reasoned from evidence. SUPPORTED_INFERENCE when a citation survived,
    UNSUPPORTED when none did — an inference with nothing behind it is not support."""

    mark = Provenance.SUPPORTED_INFERENCE if claim_ids else Provenance.UNSUPPORTED
    return GroundedItem(content, mark, claim_ids)


@dataclass(frozen=True)
class ActorGroundingProfile:
    """Everything known about one specific actor, with provenance for every element.

    ``is_constructed_representative`` marks a deliberately synthetic agent (a population
    stratum) — such an actor is permitted to be generic, carries a population weight,
    and may never impersonate a named real person.

    ``unsupported_records`` holds elements whose citations did not survive. They are
    kept so the trace can show what was dropped and why; they are never rendered.
    """

    actor_id: str
    canonical_identity: str
    role: str
    authority: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    valid_time: str | None = None

    previous_observed_actions: tuple[GroundedItem, ...] = ()
    direct_statements: tuple[GroundedItem, ...] = ()
    stated_preferences: tuple[GroundedItem, ...] = ()
    inferred_preferences: tuple[GroundedItem, ...] = ()
    goals: tuple[GroundedItem, ...] = ()
    constraints: tuple[GroundedItem, ...] = ()
    commitments: tuple[GroundedItem, ...] = ()
    relationships: tuple[GroundedItem, ...] = ()
    information_access: tuple[GroundedItem, ...] = ()
    relevant_documents: tuple[GroundedItem, ...] = ()

    current_evidence_grounded_inclination: GroundedItem | None = None
    conditional_reaction_model: tuple[GroundedItem, ...] = ()
    # Where the retrieved record does not agree about this actor. Never resolved here:
    # an actor the sources disagree about is exactly the actor a simulation exists to
    # play forward, and silently picking the newest or the most authoritative reading
    # would decide the question before anybody acts.
    contested_dispositions: tuple[GroundedItem, ...] = ()

    claim_ids: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    unsupported_records: tuple[GroundedItem, ...] = ()

    claim_attestation: ClaimAttestation = ClaimAttestation.NOT_CHECKED
    name_attested_claim_ids: tuple[str, ...] = ()

    is_constructed_representative: bool = False
    population_weight: float | None = None

    def with_attestation(
        self, attestation: ClaimAttestation, claim_ids: tuple[str, ...]
    ) -> ActorGroundingProfile:
        """Return this profile carrying the result of a claim-attestation pass."""

        return replace(
            self, claim_attestation=attestation, name_attested_claim_ids=tuple(sorted(claim_ids))
        )

    # ---- derived views -----------------------------------------------------

    @property
    def own_record_groups(self) -> tuple[tuple[GroundedItem, ...], ...]:
        """The groups that hold this actor's own record of what it did, said, holds, or
        is bound by — as opposed to context, inferences about it, or documents."""

        return (
            self.previous_observed_actions,
            self.direct_statements,
            self.stated_preferences,
            self.commitments,
        )

    @property
    def previous_observed_action(self) -> GroundedItem | None:
        """The leading entry of this actor's own verified history, if it has one.

        Entries are in construction order (an explicitly compiled previous action
        first, then seeded memories in compiled order); nothing here claims to know
        which is most recent. Never an inference.
        """

        for item in self.previous_observed_actions:
            if item.is_verified:
                return item
        return None

    def identity_keys(self) -> frozenset[str]:
        """The name forms by which this actor may be referred to in prose.

        A single character is a letter, not a name: it carries no discriminating signal
        and would fire on the article "a" or an initial inside any record, so it is not
        used to decide who a record is about.
        """

        keys = {self.actor_id.lower(), self.canonical_identity.lower()}
        keys.update(a.lower() for a in self.aliases)
        return frozenset(k for k in keys if len(k) > 1)

    def all_items(self) -> list[GroundedItem]:
        groups = (
            *self.own_record_groups,
            self.inferred_preferences,
            self.goals,
            self.constraints,
            self.relationships,
            self.information_access,
            self.relevant_documents,
            self.conditional_reaction_model,
        )
        items = [i for g in groups for i in g]
        if self.current_evidence_grounded_inclination is not None:
            items.append(self.current_evidence_grounded_inclination)
        return items

    @property
    def evidence_claim_ids(self) -> tuple[str, ...]:
        ids = set(self.claim_ids)
        for item in self.all_items():
            ids.update(item.claim_ids)
        return tuple(sorted(ids))

    @property
    def own_cited_records(self) -> tuple[GroundedItem, ...]:
        """This actor's own record: verified (hence cited) *and* self-attributed —
        written in the actor's own voice, or explicitly naming the actor.

        A cited fact about the world is not grounding for a person; a first-person
        record with no surviving citation is not evidence. Only items that are both
        count.
        """

        keys = self.identity_keys()
        return tuple(
            item
            for group in self.own_record_groups
            for item in group
            if item.is_verified and _self_attributed(item.content, keys)
        )

    @property
    def has_own_cited_record(self) -> bool:
        """True when the actor carries at least one cited, self-attributed record of a
        prior action, statement, stated position or commitment of its own.

        This is now a *promoter* to :attr:`GroundingLevel.DIRECT_RECORD`, not a
        condition of existing. See :meth:`grounding_assessment`.
        """

        return bool(self.own_cited_records)

    def grounding_assessment(self) -> GroundingAssessment:
        """Where this actor sits in the evidence hierarchy, and what that licenses.

        The level is decided by which *citations survived*, never by the compiler's
        confidence. Each rung is checked in order and the strongest reached wins, so an
        actor with a first-person record is grounded at level 1 even though its office
        would independently ground it at level 2.

        The crucial property is what happens at levels 2-6: the actor is admitted, and
        its disposition is downgraded to INFERRED or HYPOTHETICAL rather than the actor
        being deleted. A trade commissioner whose office and authority are a matter of
        public record does not stop existing because no retrieved source quotes them in
        the first person — their private position is simply not a fact, and the
        simulation is what resolves it.
        """

        keys = self.identity_keys()

        def cited(items: tuple[GroundedItem, ...]) -> tuple[GroundedItem, ...]:
            return tuple(i for i in items if i.is_supported)

        own = self.own_cited_records
        first_person = tuple(i for i in own if is_first_person(i.content))
        if first_person:
            return self._assess(
                GroundingLevel.DIRECT_RECORD,
                "carries a cited record in this actor's own voice",
                first_person,
            )

        # An office is a structural fact. It is verifiable, it is what puts the actor
        # inside the causal boundary, and it is the level at which most real
        # decision-makers are knowable before the fact.
        #
        # "Verified office" means the *evidence* records the office, so the claims cited
        # for it have to mention the actor. Without that test, a role string, one
        # authority and any claim ids at all admitted an actor nothing in the evidence
        # refers to — the exact hole the comment at ROLE_LEVEL_BEHAVIOR below describes,
        # one rung higher and therefore reached first. A live Bank of England world put
        # the Monetary Policy Committee here on six claims that never mention it.
        #
        # UNATTESTED withdraws the office; NOT_CHECKED does not grant one it verified —
        # it leaves the level as it was and says so in the reason, because a check that
        # never ran must not read as a check that passed.
        if (
            self.role.strip()
            and self.authority
            and self.claim_ids
            and self.claim_attestation is not ClaimAttestation.UNATTESTED
        ):
            checked = self.claim_attestation is ClaimAttestation.ATTESTED
            office_claims = self.name_attested_claim_ids if checked else self.claim_ids
            return self._assess(
                GroundingLevel.OFFICIAL_ROLE,
                f"holds the verified office {self.role!r} with cited authority "
                f"({', '.join(self.authority)})"
                + (
                    ""
                    if checked
                    else " — NOT CHECKED: no pass has confirmed that these claims mention "
                    "this actor"
                ),
                (),
                extra_claims=office_claims,
            )

        if own:
            return self._assess(
                GroundingLevel.DOCUMENTED_PRIOR_BEHAVIOR,
                "carries cited records of conduct attributed to this actor by name",
                own,
            )

        policy = cited(self.constraints) + cited(self.relevant_documents)
        if policy:
            return self._assess(
                GroundingLevel.INSTITUTIONAL_POLICY,
                "grounded in cited institutional policy or governing documents",
                policy,
            )

        named = tuple(i for i in self.all_items() if i.is_supported and _any_name(i.content, keys))
        if named:
            return self._assess(
                GroundingLevel.CONTEMPORANEOUS_REPORTING,
                "named by cited contemporaneous sources",
                named,
            )

        # Role level still requires the *entity* to be cited. Without that, one
        # irrelevant claim id attached to an invented person's memory seed, plus any
        # non-empty role string, was enough to admit an actor nothing in the evidence
        # refers to.
        role_level = tuple(i for i in self.all_items() if i.is_supported)
        if (
            role_level
            and self.role.strip()
            and self.claim_ids
            and self.claim_attestation is not ClaimAttestation.UNATTESTED
        ):
            return self._assess(
                GroundingLevel.ROLE_LEVEL_BEHAVIOR,
                f"grounded only at the level of the role {self.role!r}, not the individual",
                role_level,
            )

        return GroundingAssessment(
            actor_id=self.actor_id,
            level=GroundingLevel.NONE,
            reason=(
                "no surviving citation attaches this actor to the world: neither a "
                "record of its own, nor a cited office and authority, nor any cited "
                "source that names it"
            )
            + (
                " — every claim assigned to this actor was checked and none of them "
                "mentions it, so it is a participant the evidence does not contain"
                if self.claim_attestation is ClaimAttestation.UNATTESTED
                else ""
            ),
            disposition_class=EpistemicClass.UNSUPPORTED,
            missing=self.missing_information,
        )

    def _assess(
        self,
        level: GroundingLevel,
        reason: str,
        items: tuple[GroundedItem, ...],
        *,
        extra_claims: tuple[str, ...] = (),
    ) -> GroundingAssessment:
        claims = {c for i in items for c in i.claim_ids} | set(extra_claims)
        return GroundingAssessment(
            actor_id=self.actor_id,
            level=level,
            reason=reason,
            supporting_claim_ids=tuple(sorted(claims)),
            disposition_class=private_state_class(level),
            missing=self.missing_information,
        )

    def render_grounding(self) -> str:
        """The ACTOR-SPECIFIC GROUNDING block placed in this actor's prompt.

        Only items with surviving citations are rendered, each carrying its own mark
        and claim ids. An element whose citations did not survive is never shown here
        — the actor is told how many were withheld, not what they said.
        """

        assessment = self.grounding_assessment()
        lines: list[str] = [f"You are {self.canonical_identity}."]
        if self.aliases:
            lines.append(f"Also referred to as: {', '.join(self.aliases)}")
        lines.append(f"Role: {self.role}")
        lines.append(f"Authority: {', '.join(self.authority) or '(none recorded)'}")
        lines.append(
            f"HOW YOU ARE GROUNDED: {assessment.level.label} "
            f"(hierarchy level {int(assessment.level)}). Your own current position is "
            f"{assessment.disposition_class.value.upper()} at this level."
        )
        if assessment.disposition_class is not EpistemicClass.VERIFIED:
            lines.append(
                "No source records what you privately intend here. Reason from your "
                "office, obligations and record — and do not pretend to a position you "
                "have not taken."
            )
        if self.is_constructed_representative:
            lines.append(
                "NOTE: you are a CONSTRUCTED REPRESENTATIVE agent standing for a "
                f"population segment (weight {self.population_weight}); you are not a "
                "specific named individual."
            )

        def block(title: str, items: tuple[GroundedItem, ...]) -> None:
            shown = [i for i in items if i.is_supported]
            if shown:
                lines.append(f"\n{title}:")
                lines.extend(f"  - {i.render()}" for i in shown)

        block(
            "YOUR OWN RECORD, FROM CITED EVIDENCE (what you previously did, said, or held)",
            self.previous_observed_actions,
        )
        block("YOUR OWN PUBLIC STATEMENTS", self.direct_statements)
        block("YOUR STATED PREFERENCES", self.stated_preferences)
        block("YOUR COMMITMENTS", self.commitments)
        block("YOUR GOALS", self.goals)
        block("YOUR CONSTRAINTS", self.constraints)
        block("YOUR RELATIONSHIPS", self.relationships)
        block("YOUR INFORMATION ACCESS", self.information_access)
        block("PREFERENCES INFERRED ABOUT YOU (not observed)", self.inferred_preferences)
        incl = self.current_evidence_grounded_inclination
        if incl is not None and incl.is_supported:
            lines.append(
                "\nYOUR CURRENT INCLINATION (an inference about now — it does NOT replace "
                "your own record above):"
            )
            lines.append(f"  - {incl.render()}")
        block("WHAT WOULD CHANGE YOUR POSITION", self.conditional_reaction_model)
        contested = [i for i in self.contested_dispositions if i.is_supported]
        if contested:
            lines.append(
                "\nWHERE THE RECORD DISAGREES ABOUT YOU (both readings are cited; neither "
                "has been settled, and it is not settled by you assuming one):"
            )
            lines.extend(f"  - {i.render()}" for i in contested)
        if self.missing_information:
            lines.append("\nNOT KNOWN ABOUT YOU (do not invent these):")
            lines.extend(f"  - {m}" for m in self.missing_information)
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        def ser(items: tuple[GroundedItem, ...]) -> list[dict[str, object]]:
            return [
                {
                    "content": i.content,
                    "provenance": i.provenance.value,
                    "claim_ids": list(i.claim_ids),
                    "valid_time": i.valid_time,
                }
                for i in items
            ]

        prev = self.previous_observed_action
        incl = self.current_evidence_grounded_inclination
        return {
            "grounding": self.grounding_assessment().as_dict(),
            "actor_id": self.actor_id,
            "canonical_identity": self.canonical_identity,
            "aliases": list(self.aliases),
            "role": self.role,
            "authority": list(self.authority),
            "valid_time": self.valid_time,
            "previous_observed_actions": ser(self.previous_observed_actions),
            "previous_observed_action": prev.content if prev else None,
            "direct_statements": ser(self.direct_statements),
            "stated_preferences": ser(self.stated_preferences),
            "inferred_preferences": ser(self.inferred_preferences),
            "goals": ser(self.goals),
            "constraints": ser(self.constraints),
            "commitments": ser(self.commitments),
            "relationships": ser(self.relationships),
            "information_access": ser(self.information_access),
            "relevant_documents": ser(self.relevant_documents),
            "current_evidence_grounded_inclination": (
                {"content": incl.content, "provenance": incl.provenance.value} if incl else None
            ),
            "conditional_reaction_model": ser(self.conditional_reaction_model),
            "contested_dispositions": ser(self.contested_dispositions),
            "claim_attestation": self.claim_attestation.value,
            "name_attested_claim_ids": list(self.name_attested_claim_ids),
            "own_cited_records": ser(self.own_cited_records),
            "unsupported_records": ser(self.unsupported_records),
            "claim_ids": list(self.evidence_claim_ids),
            "lineage_ids": list(self.lineage_ids),
            "missing_information": list(self.missing_information),
            "is_constructed_representative": self.is_constructed_representative,
            "population_weight": self.population_weight,
        }


def _self_attributed(content: str, keys: frozenset[str]) -> bool:
    """True when ``content`` is this actor's own record: written in the first person,
    or explicitly naming the actor."""

    if is_first_person(content):
        return True
    low = content.lower()
    return any(_names(k, low) for k in keys)


def _any_name(content: str, keys: frozenset[str]) -> bool:
    low = content.lower()
    return any(_names(k, low) for k in keys)


def _names(key: str, text: str) -> bool:
    """True when ``key`` occurs in ``text`` as a whole word/phrase (not a substring of
    a longer word), so an id like "ada" cannot match inside "adamant"."""

    return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text) is not None


# ---------------------------------------------------------------------------
# Deterministic actor-grounding coverage gate
# ---------------------------------------------------------------------------


# The only disposition an actor-grounding pass assigns: this claim reached this
# actor's profile. Deciding that a claim is irrelevant, a duplicate, or unresolved is
# the coverage gate's job (see :class:`sworldmodel.coverage.Disposition`); mirroring
# that vocabulary here declared five outcomes this module never produces.
_INCLUDED_IN_ACTOR_PROFILE = "included_in_actor_profile"


@dataclass(frozen=True)
class ActorGroundingReport:
    profiles: tuple[ActorGroundingProfile, ...]
    ungrounded_actors: tuple[str, ...] = ()
    misattributed: tuple[str, ...] = ()
    missing_previous_actions: tuple[str, ...] = ()
    dropped_uncited: tuple[str, ...] = ()
    dispositions: tuple[tuple[str, str, str], ...] = ()  # (claim_id, actor_id, disposition)
    assessments: tuple[GroundingAssessment, ...] = ()
    # Actors whose assigned claims were checked and found not to mention them. Reported
    # separately from ``ungrounded_actors`` because it is a different finding: not
    # "weakly evidenced" but "the evidence does not contain this participant".
    unattested_actors: tuple[str, ...] = ()
    unchecked_actors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not (self.ungrounded_actors or self.misattributed or self.missing_previous_actions)

    @property
    def level_histogram(self) -> tuple[tuple[str, int], ...]:
        counts: dict[str, int] = {}
        for a in self.assessments:
            counts[a.level.name.lower()] = counts.get(a.level.name.lower(), 0) + 1
        return tuple(sorted(counts.items()))

    def as_dict(self) -> dict[str, object]:
        return {
            "complete": self.is_complete,
            "ungrounded_actors": list(self.ungrounded_actors),
            "misattributed": list(self.misattributed),
            "missing_previous_actions": list(self.missing_previous_actions),
            "dropped_uncited": list(self.dropped_uncited),
            "unattested_actors": list(self.unattested_actors),
            "actors_whose_claims_were_never_checked": list(self.unchecked_actors),
            "profiles": [p.as_dict() for p in self.profiles],
            "grounding_assessments": [a.as_dict() for a in self.assessments],
            "grounding_levels": dict(self.level_histogram),
            "dispositions": [
                {"claim_id": c, "actor_id": a, "disposition": d} for c, a, d in self.dispositions
            ],
            "notes": list(self.notes),
        }


def assess_actor_grounding(
    profiles: tuple[ActorGroundingProfile, ...],
    *,
    require_previous_action: bool = False,
) -> ActorGroundingReport:
    """Check that each actor is grounded as the specific entity it claims to be.

    An actor passes when the surviving citations place it somewhere on the evidence
    hierarchy — a record of its own, a verified office and authority, documented prior
    conduct, institutional policy, contemporaneous reporting that names it, or evidence
    at the level of its role. What fails is an actor with *no* surviving citation of any
    kind: a name with nothing behind it.

    What the level changes is not whether the actor exists but what may be said about
    it. Below a direct record, its disposition is marked INFERRED or HYPOTHETICAL and
    the simulation resolves it, rather than the compiler asserting it.

    No actor may carry another actor's personal record; that check is unchanged, and it
    is the one that actually catches fabrication.
    """

    ungrounded: list[str] = []
    misattributed: list[str] = []
    missing_prev: list[str] = []
    dropped: list[str] = []
    dispositions: list[tuple[str, str, str]] = []
    assessments: list[GroundingAssessment] = []

    # Cross-assignment check. Sharing a *source* is legitimate — one set of minutes or
    # one poll can back every actor — so claim-id overlap is not misattribution. The
    # real defect is an actor's personal record whose CONTENT is about someone else
    # (a profile carrying "<another actor> stated: ...").
    keys_by_actor = {p.actor_id: p.identity_keys() for p in profiles}
    for p in profiles:
        mine = keys_by_actor[p.actor_id]
        foreign = {k for other, keys in keys_by_actor.items() if other != p.actor_id for k in keys}
        for item in list(p.previous_observed_actions) + list(p.direct_statements):
            low = item.content.lower()
            if any(_names(m, low) for m in mine):
                continue  # the record names this actor: correctly attributed
            hit = next((k for k in sorted(foreign) if _names(k, low)), None)
            if hit:
                misattributed.append(
                    f"{p.actor_id} carries a personal record naming {hit!r}: {item.content!r}"
                )

    unattested: list[str] = []
    unchecked: list[str] = []
    for p in profiles:
        if p.claim_attestation is ClaimAttestation.UNATTESTED:
            unattested.append(p.actor_id)
        elif p.claim_attestation is ClaimAttestation.NOT_CHECKED:
            unchecked.append(p.actor_id)
        for item in p.unsupported_records:
            dropped.append(f"{p.actor_id}: {item.content!r} had no surviving evidence citation")
        if p.is_constructed_representative:
            # A deliberately synthetic stratum is allowed to be generic, but it must be
            # labeled and carry a population weight.
            if p.population_weight is None:
                ungrounded.append(f"{p.actor_id}: constructed representative without a weight")
            continue
        assessment = p.grounding_assessment()
        assessments.append(assessment)
        if not assessment.admissible:
            ungrounded.append(
                f"{p.actor_id} ({p.canonical_identity}): {assessment.reason} — this "
                "actor has no referent in the evidence and may not enter the world"
            )
        if require_previous_action and p.previous_observed_action is None:
            missing_prev.append(f"{p.actor_id} ({p.canonical_identity}): no previous action found")
        for cid in p.evidence_claim_ids:
            dispositions.append((cid, p.actor_id, _INCLUDED_IN_ACTOR_PROFILE))

    notes: list[str] = []
    if unchecked:
        notes.append(
            "claim attestation NOT CHECKED for "
            + ", ".join(sorted(unchecked))
            + ": nothing has compared the claims assigned to these actors against their "
            "names, so their citations are assignments, not attestations"
        )
    if unattested:
        notes.append(
            "claims checked and found not to mention: "
            + ", ".join(sorted(unattested))
            + " — these are participants the evidence does not contain"
        )
    return ActorGroundingReport(
        profiles=profiles,
        ungrounded_actors=tuple(ungrounded),
        misattributed=tuple(sorted(set(misattributed))),
        missing_previous_actions=tuple(missing_prev),
        dropped_uncited=tuple(dropped),
        dispositions=tuple(dispositions),
        assessments=tuple(assessments),
        unattested_actors=tuple(sorted(unattested)),
        unchecked_actors=tuple(sorted(unchecked)),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Attestation: do this actor's claims mention this actor?
# ---------------------------------------------------------------------------


def attest_profiles(
    profiles: tuple[ActorGroundingProfile, ...], view: EvidenceView
) -> tuple[ActorGroundingProfile, ...]:
    """Check each actor's assigned claims against the actor's own name.

    A claim attests an actor when the actor is one of its declared entities, or when the
    actor's full name or one of its declared aliases occurs in the claim's proposition or
    supporting excerpt. Nothing else counts: the compiler assigning a claim id to an actor
    is the compiler's opinion, and it is precisely the opinion this check exists to test.

    **Whole names only, never parts.** A source that writes "Bailey" after introducing
    "Andrew Bailey" is matched through ``aliases``, which is what that field is for — so
    a compiler that knows an actor is referred to by a short form must record the short
    form. Matching on parts instead would defeat the check outright: an invented "Terminal
    operations manager" shares the token "terminal" with any claim about a ferry terminal,
    and would attest itself out of a claim that is not about a person at all.

    Returns the profiles with :attr:`ActorGroundingProfile.claim_attestation` set. It
    refuses nothing itself — the assessment and the report say what was found, and
    :func:`enforce_actor_grounding` is where an unattested actor stops being admissible.
    """

    out: list[ActorGroundingProfile] = []
    for profile in profiles:
        if profile.is_constructed_representative:
            # A stratum stands for a population, not for a name in a source; requiring a
            # claim to mention it by name would refuse every legitimate one.
            out.append(profile)
            continue
        attesting = tuple(
            cid for cid in profile.evidence_claim_ids if _claim_names(view, cid, profile)
        )
        out.append(
            profile.with_attestation(
                ClaimAttestation.ATTESTED if attesting else ClaimAttestation.UNATTESTED,
                attesting,
            )
        )
    return tuple(out)


def _claim_names(view: EvidenceView, claim_id: str, profile: ActorGroundingProfile) -> bool:
    """Whether the stored claim ``claim_id`` actually speaks about this actor."""

    try:
        claim = view.get(claim_id)
    except (EvidenceError, CutoffViolationError):
        # A claim id that is not in the store, or not available at the cutoff, attests
        # nothing. It is not evidence that the actor is absent either — the other ids
        # decide that — so this is a "no" for this id and nothing more.
        return False
    keys = profile.identity_keys()
    if any(e.strip().lower() in keys for e in claim.entities):
        return True
    text = f"{claim.proposition} {claim.supporting_excerpt}".lower()
    return any(_names(k, text) for k in keys)


# ---------------------------------------------------------------------------
# Disposition: what this actor wants, has done, is bound by, and how it reacted
# ---------------------------------------------------------------------------


# Each retrieval axis lands in the profile field that already existed for it:
# WANTS -> stated_preferences, HAS_DONE -> previous_observed_actions,
# CONSTRAINED_BY -> constraints, REACTS_TO -> conditional_reaction_model. Those fields
# were never filled from evidence — a compiled actor carried its claim ids and not one
# word of what they said, so every actor's prompt described a role rather than a person.


def ground_disposition(
    profile: ActorGroundingProfile, disposition: ParticipantDisposition
) -> ActorGroundingProfile:
    """Fold retrieved disposition evidence into an actor's profile, with its marks.

    Every item lands as a VERIFIED_OBSERVATION carrying the claim ids it came from —
    these are quotations of the record, not readings of it — except the contested block,
    where the item is the *fact that the record disagrees*, which is itself observed.

    An axis retrieval did not establish is added to ``missing_information`` naming the
    axis, and no field is filled for it. That is the rule the whole module turns on: an
    invented disposition is a fabricated value wearing a citation, and a missing one is
    merely a thing the actor must reason about without.
    """

    by_axis: dict[DispositionAxis, tuple[GroundedItem, ...]] = {}
    contested: list[GroundedItem] = []
    for evidence in disposition.axes:
        by_axis[evidence.axis] = tuple(
            observation(text, (claim_id,))
            for claim_id, text in zip(evidence.claim_ids, evidence.renderings, strict=True)
        )
        for a_id, b_id in evidence.conflicting_pairs:
            contested.append(
                observation(
                    f"the record disagrees about {evidence.axis.question}: "
                    f"{a_id} and {b_id} give different answers",
                    (a_id, b_id),
                )
            )

    def added(axis: DispositionAxis) -> tuple[GroundedItem, ...]:
        return by_axis.get(axis, ())

    missing = list(profile.missing_information)
    for axis in disposition.missing_axes:
        missing.append(
            f"{axis.value.replace('_', ' ')}: no retrieved source establishes "
            f"{axis.question} — this is not established and must not be assumed"
        )
    return replace(
        profile,
        stated_preferences=profile.stated_preferences + added(DispositionAxis.WANTS),
        previous_observed_actions=(
            profile.previous_observed_actions + added(DispositionAxis.HAS_DONE)
        ),
        constraints=profile.constraints + added(DispositionAxis.CONSTRAINED_BY),
        conditional_reaction_model=(
            profile.conditional_reaction_model + added(DispositionAxis.REACTS_TO)
        ),
        contested_dispositions=profile.contested_dispositions + tuple(contested),
        missing_information=tuple(dict.fromkeys(missing)),
    )


def enforce_actor_grounding(report: ActorGroundingReport) -> None:
    """Refuse to simulate a world whose actors are not grounded in cited evidence."""

    if report.is_complete:
        return
    raise WorldIntegrityError(
        "actors are not grounded in cited evidence — simulation refused",
        details={
            # Grounding failed for this compilation. Compiling again from the same
            # evidence is legitimate; inventing support for an actor is not.
            "failure": "actors_ungrounded",
            "recompilable": True,
            "ungrounded_actors": list(report.ungrounded_actors),
            "misattributed_evidence": list(report.misattributed),
            "missing_previous_actions": list(report.missing_previous_actions),
            "dropped_uncited_records": list(report.dropped_uncited),
        },
    )


# ---------------------------------------------------------------------------
# Building profiles from compiled member specs
# ---------------------------------------------------------------------------


def profile_from_member(
    *,
    actor_id: str,
    name: str,
    role: str,
    authority: tuple[str, ...],
    previous_action: str | None,
    previous_action_claim_ids: tuple[str, ...] = (),
    memory_seeds: tuple[tuple[str, tuple[str, ...]], ...] = (),
    inclination: str | None = None,
    inclination_claim_ids: tuple[str, ...] = (),
    reaction_rules: tuple[tuple[str, str], ...] = (),
    reaction_rule_claim_ids: tuple[str, ...] = (),
    valid_time: str | None = None,
) -> ActorGroundingProfile:
    """Build a grounded profile from an actor's compiled evidence.

    Every element's mark is decided by the citations that survived: an item with a
    surviving claim id becomes a VERIFIED_OBSERVATION (or, for the inclination and the
    reaction rules, a SUPPORTED_INFERENCE); an item with none becomes UNSUPPORTED, is
    held in ``unsupported_records`` for the trace, and never reaches the actor's
    prompt. The verified record is kept in its own field so the inferred current
    inclination can never overwrite it.

    Seeded memories are not sorted by their wording. Guessing from words like "voted"
    which memory is an "action" and which a "statement" is a scenario assumption, and
    it would put the gate at the mercy of vocabulary; a seed is simply this actor's own
    record, and whether it is self-attributed is checked directly.
    """

    records: list[GroundedItem] = []
    dropped: list[GroundedItem] = []

    def keep(item: GroundedItem, into: list[GroundedItem]) -> None:
        (into if item.is_supported else dropped).append(item)

    if previous_action:
        keep(
            observation(
                f"took the action: {previous_action}",
                previous_action_claim_ids,
                valid_time=valid_time,
            ),
            records,
        )
    for content, cids in memory_seeds:
        text = content.strip()
        if text:
            keep(observation(text, cids), records)

    inclination_item: GroundedItem | None = None
    if inclination:
        item = inference(f"currently expected to favor: {inclination}", inclination_claim_ids)
        if item.is_supported:
            inclination_item = item
        else:
            dropped.append(item)

    reactions: list[GroundedItem] = []
    for signal, option in reaction_rules:
        keep(
            inference(
                f"if {signal} crosses its threshold, would move toward {option}",
                reaction_rule_claim_ids,
            ),
            reactions,
        )

    profile = ActorGroundingProfile(
        actor_id=actor_id,
        canonical_identity=name,
        role=role,
        authority=authority,
        valid_time=valid_time,
        previous_observed_actions=tuple(records),
        current_evidence_grounded_inclination=inclination_item,
        conditional_reaction_model=tuple(reactions),
        unsupported_records=tuple(dropped),
    )

    missing: list[str] = []
    if not profile.has_own_cited_record:
        missing.append(
            "your own words: no retrieved source records a statement or action of yours "
            "on this matter, so your position here is not established"
        )
    if dropped:
        missing.append(
            f"{len(dropped)} compiled element(s) had no surviving evidence citation and "
            "were withheld from this briefing"
        )
    return replace(profile, missing_information=tuple(missing))


# ---------------------------------------------------------------------------
# Citation support: does the cited record support what it is attached to?
#
# :func:`attest_profiles` above settled the actor half of this question — a claim
# grounds an actor only when it actually names that actor, never merely by being in the
# store. Everything below is the same move applied to the other things a plan cites:
# the VALUE of an uncertainty alternative, and the prose claims (a zero-actor
# justification, a single-multiplier exemption) a plan asks to be believed on evidence.
#
# The defect being closed is one predicate, repeated: a check that reads the PRESENCE
# of a citation as proof of SUPPORT. `semantic_plan.cited()` returned True for any id in
# the store, so `c-f1` — "The Kestrel Bay ferry terminal recorded 320000 crossings" —
# grounded an aquifer world's 0.9 runoff fraction, and with it D3's straddling gate,
# D4's exemption, D5's zero-actor justification and FD-11's filler gate at once.
#
# Two rules shape what follows, both learned the hard way in this repo:
#
# * **A gate that refuses correct worlds is worse than the hole it closes.** So support
#   is decided by the most generous reading of the record that still means something: a
#   number is supported by any stated value or any stated range that contains it (read
#   through percentages, because "a 3% rise" legitimately supports a 1.03 multiplier),
#   and a categorical value is supported by any significant word it shares with the
#   record. What is refused is a citation with *nothing* to do with the value it is
#   attached to.
# * **A check that could not run must never read as a check that passed.** Support is
#   tri-state. :attr:`Support.UNDECIDABLE` is returned whenever the claim texts were not
#   supplied or the value is of a kind no textual test can read, and callers are expected
#   to fall back to the weaker existence check *and say so* — never to treat it as a pass.
# ---------------------------------------------------------------------------


class Support(StrEnum):
    """Whether a citation supports the thing it is attached to.

    The tri-state is load-bearing. ``UNDECIDABLE`` is not a soft ``SUPPORTED``: it says
    the comparison never happened, either because no claim text was available or because
    the value is of a kind this module cannot read. A caller that maps it to "fine"
    reintroduces exactly the defect this module exists to close, one layer up.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNDECIDABLE = "undecidable"


@dataclass(frozen=True)
class CitedRecord:
    """One stored claim, reduced to the text a support test reads.

    Built from an :class:`~sworldmodel.evidence.EvidenceClaim`, from a plain mapping, or
    from anything carrying the same attribute names — the support tests run in the
    validator, which is handed whatever the caller has, and a claim it cannot read is
    reported as unreadable rather than assumed good.
    """

    claim_id: str
    proposition: str = ""
    normalized_value: str = ""
    supporting_excerpt: str = ""
    entities: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return f"{self.proposition} {self.normalized_value} {self.supporting_excerpt}"


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


def cited_record(claim_id: str, claim: Any) -> CitedRecord:
    """Read one stored claim into the shape the support tests use.

    Accepts the evidence store's own claim objects (``proposition`` /
    ``normalized_value`` / ``supporting_excerpt`` / ``entities``) and the plain
    dictionaries the compilers and tests carry, where the normalized value is written
    ``value``. Fields that are absent are empty, never invented.
    """

    def field(*names: str) -> Any:
        for name in names:
            if isinstance(claim, Mapping):
                if name in claim:
                    return claim[name]
            elif hasattr(claim, name):
                return getattr(claim, name)
        return None

    raw_entities = field("entities") or ()
    entities = (
        tuple(str(e) for e in raw_entities) if isinstance(raw_entities, (list, tuple)) else ()
    )
    return CitedRecord(
        claim_id=claim_id,
        proposition=_as_str(field("proposition")),
        normalized_value=_as_str(field("normalized_value", "value")),
        supporting_excerpt=_as_str(field("supporting_excerpt", "excerpt")),
        entities=entities,
    )


# A number as a record writes one: optional sign, digits with optional thousands
# separators, optional decimals. Guarded on both sides so an identifier like "c-a3" or a
# version "v2.1.4" is not read as a quantity the record states.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])")

# Connectors that make two adjacent numbers the ends of one stated range. "and" is only
# a range connector after the word "between", because "240 acre-feet and 800 acre-feet"
# states two quantities and no interval between them.
_RANGE_CONNECTORS = frozenset({"to", "-", "–", "—", "through", "..", "..."})
_PERCENT = re.compile(r"\s*(?:%|percent|per cent|percentage)", re.IGNORECASE)


def _is_percent_at(text: str, position: int) -> bool:
    return _PERCENT.match(text, position) is not None


def _readings(number: float, *, percent: bool) -> tuple[float, ...]:
    """Every quantity one written number can legitimately stand for.

    A bare number stands for itself. A percentage additionally stands for its fraction
    and for the multipliers either side of it, because a record that says "3%" is the
    normal way a real source supports a 1.03 or 0.97 factor — refusing that would refuse
    correct worlds for a formatting difference, which is the failure mode this module is
    under standing instruction to avoid.
    """

    if not percent:
        return (number,)
    return (number, number / 100.0, 1.0 + number / 100.0, 1.0 - number / 100.0)


def _interval_readings(
    low: float, high: float, *, percent: bool
) -> tuple[tuple[float, float], ...]:
    if not percent:
        return ((low, high),)
    return (
        (low, high),
        (low / 100.0, high / 100.0),
        (1.0 + low / 100.0, 1.0 + high / 100.0),
        (1.0 - high / 100.0, 1.0 - low / 100.0),
    )


def stated_quantities(text: str) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    """The values and the ranges a record states, with percentages read both ways.

    Returns ``(points, intervals)``. Nothing here interprets units: a support test asks
    whether the record contains this quantity at all, not whether it is denominated the
    same way, because the plan's own dimension gates already judge units and duplicating
    that here would refuse a correct world over a rendering.
    """

    points: list[float] = []
    intervals: list[tuple[float, float]] = []
    matches = list(_NUMBER.finditer(text))
    values: list[tuple[float, bool]] = []
    for match in matches:
        try:
            number = float(match.group().replace(",", ""))
        except ValueError:  # pragma: no cover - the pattern only matches numerals
            continue
        percent = _is_percent_at(text, match.end())
        values.append((number, percent))
        points.extend(_readings(number, percent=percent))
    for index, (left, right) in enumerate(zip(matches, matches[1:], strict=False)):
        connector = text[left.end() : right.start()].strip().lower()
        joined = connector in _RANGE_CONNECTORS or (
            connector == "and"
            and text[max(0, left.start() - 12) : left.start()].lower().find("between") >= 0
        )
        if not joined:
            continue
        low_value, low_percent = values[index]
        high_value, high_percent = values[index + 1]
        if low_value > high_value:
            low_value, high_value = high_value, low_value
        intervals.extend(
            _interval_readings(low_value, high_value, percent=low_percent or high_percent)
        )
    return tuple(points), tuple(intervals)


def _quantity_stated(text: str, value: float) -> bool:
    points, intervals = stated_quantities(text)
    if any(math.isclose(value, p, rel_tol=1e-6, abs_tol=1e-12) for p in points):
        return True
    scale = max(abs(value), 1.0)
    slack = 1e-9 * scale
    return any(low - slack <= value <= high + slack for low, high in intervals)


def _significant_words(text: str) -> frozenset[str]:
    """The words in a phrase long enough to identify what it is about.

    Deliberately crude, and deliberately the same crudeness the world review already
    uses for the settled-record subject check: short words carry no subject, so they
    cannot decide that a record is about a value.
    """

    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return frozenset(w for w in cleaned.split() if len(w) >= 4)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ClaimSupport:
    """The evidence store, as the gates that read a citation as support see it.

    ``records`` is ``None`` when the caller had only claim *ids* — the store's texts
    were never handed over. In that state every support question answers
    :attr:`Support.UNDECIDABLE`, which is why :attr:`can_read_claims` exists: a gate must
    be able to say "this was not checked" rather than quietly returning to the existence
    check that was the defect.
    """

    known: frozenset[str] | None = None
    records: Mapping[str, CitedRecord] | None = None

    @classmethod
    def from_claims(
        cls, claims: Mapping[str, Any] | None, *, known: frozenset[str] | None = None
    ) -> ClaimSupport:
        """Build from ``{claim_id: claim}`` — store objects or plain mappings alike."""

        if claims is None:
            return cls(known=known, records=None)
        records = {cid: cited_record(cid, claim) for cid, claim in claims.items()}
        return cls(known=frozenset(records) if known is None else known, records=records)

    @classmethod
    def from_view(cls, view: Any) -> ClaimSupport:
        """Build from an :class:`~sworldmodel.evidence.EvidenceView`, at its own cutoff.

        Only claims the view itself admits are read, so a post-cutoff claim can never
        support anything — the same rule :func:`_claim_names` follows for actors.
        """

        claims = {c.id: c for c in view.available()}
        return cls.from_claims(claims)

    @property
    def can_read_claims(self) -> bool:
        return self.records is not None

    def exists(self, ids: Iterable[str]) -> bool:
        """The old predicate, kept under its real name: these ids are in the store.

        Existence is a precondition for support and is never support by itself. It stays
        because a citation to a claim that is not there grounds nothing at all, which is
        a different (and worse) finding than a claim that is there and says something
        else.
        """

        wanted = tuple(ids)
        if not wanted:
            return False
        return True if self.known is None else all(i in self.known for i in wanted)

    def _readable(self, ids: Iterable[str]) -> list[CitedRecord]:
        if self.records is None:
            return []
        return [self.records[i] for i in ids if i in self.records]

    def supports_quantity(self, ids: Iterable[str], value: float) -> Support:
        """Does any cited record state this quantity, or a range that contains it?

        This is the whole of D3's own correction boundary made mechanical — "anchor each
        in cited evidence (a published range, a recorded distribution, a stated
        forecast)" — and it is what separates legitimate indirect support from arbitrary
        citation. A claim recording a range 0.6-0.9 supports a 0.9 alternative although
        it never mentions that alternative; a claim recording 320000 ferry crossings
        supports neither 0.9 nor 0.6, whatever else it is in the store for.
        """

        wanted = tuple(ids)
        if not wanted or not self.can_read_claims:
            return Support.UNDECIDABLE
        readable = self._readable(wanted)
        if not readable:
            return Support.UNDECIDABLE
        if any(_quantity_stated(r.text, value) for r in readable):
            return Support.SUPPORTED
        return Support.UNSUPPORTED

    def supports_value(self, ids: Iterable[str], value: Any) -> Support:
        """Support for an alternative's declared value, numeric or categorical.

        A categorical value is judged far more loosely than a number — one significant
        word shared with the record is enough — because a plan legitimately paraphrases
        what a source says, and there is no way to tell a paraphrase from an invention
        without reading for meaning. What that loose test still refuses is the shape FD-11
        was: a filler alternative ("some other regime", "other") hung on a claim that
        shares not one word with it.
        """

        number = _numeric(value)
        if number is not None:
            return self.supports_quantity(ids, number)
        text = _as_str(value).strip()
        wanted = tuple(ids)
        if not text or not wanted or not self.can_read_claims:
            return Support.UNDECIDABLE
        readable = self._readable(wanted)
        if not readable:
            return Support.UNDECIDABLE
        words = _significant_words(text)
        if not words:
            return Support.UNDECIDABLE
        for record in readable:
            if words & _significant_words(record.text):
                return Support.SUPPORTED
        return Support.UNSUPPORTED

    def names_any_of(self, ids: Iterable[str], names: Iterable[str]) -> Support:
        """Is any cited record about something this world has actually considered?

        The test is :func:`attest_profiles`' test, moved from an actor to a world: a
        record speaks about a name when it declares it among its entities, or writes it
        out as a whole word or phrase. Whole names only, never parts — an invented
        "Terminal operations manager" shares the token "terminal" with any claim about a
        ferry terminal, and part-matching would let it attest itself.

        Used for the prose claims a plan asks to be believed on evidence, where there is
        no value to check. It is the widest possible reading of "relevant": every entity
        the world includes, every candidate it deliberately excluded, and its subject all
        count, so the only thing it refuses is a citation with no connection to the world
        at all.
        """

        wanted = tuple(ids)
        keys = frozenset(n.strip().lower() for n in names if n and len(n.strip()) > 1)
        if not wanted or not keys or not self.can_read_claims:
            return Support.UNDECIDABLE
        readable = self._readable(wanted)
        if not readable:
            return Support.UNDECIDABLE
        for record in readable:
            if any(e.strip().lower() in keys for e in record.entities):
                return Support.SUPPORTED
            low = record.text.lower()
            if any(_names(k, low) for k in keys):
                return Support.SUPPORTED
        return Support.UNSUPPORTED
