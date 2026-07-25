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

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from .epistemics import (
    EpistemicClass,
    GroundingAssessment,
    GroundingLevel,
    is_first_person,
    private_state_class,
)
from .errors import WorldIntegrityError

_WORD = re.compile(r"[a-z0-9]+")


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

    claim_ids: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    unsupported_records: tuple[GroundedItem, ...] = ()

    is_constructed_representative: bool = False
    population_weight: float | None = None

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
        if self.role.strip() and self.authority and self.claim_ids:
            return self._assess(
                GroundingLevel.OFFICIAL_ROLE,
                f"holds the verified office {self.role!r} with cited authority "
                f"({', '.join(self.authority)})",
                (),
                extra_claims=self.claim_ids,
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

        named = tuple(
            i for i in self.all_items() if i.is_supported and _any_name(i.content, keys)
        )
        if named:
            return self._assess(
                GroundingLevel.CONTEMPORANEOUS_REPORTING,
                "named by cited contemporaneous sources",
                named,
            )

        role_level = tuple(i for i in self.all_items() if i.is_supported)
        if role_level and self.role.strip():
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

    for p in profiles:
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

    return ActorGroundingReport(
        profiles=profiles,
        ungrounded_actors=tuple(ungrounded),
        misattributed=tuple(sorted(set(misattributed))),
        missing_previous_actions=tuple(missing_prev),
        dropped_uncited=tuple(dropped),
        dispositions=tuple(dispositions),
        assessments=tuple(assessments),
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
