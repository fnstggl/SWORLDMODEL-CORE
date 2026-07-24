"""Actor grounding: model the specific real person, not a generic role template.

An audit of a completed live run found that named actors were only weakly grounded.
Every actor's compiled ``current_inclination`` had been flattened to the board's common
position, overwriting each member's *verified* previous vote; every actor carried the
same frame-level reaction rules; and the previous observed action never appeared in the
prompt at all. The five prompts were ~80% identical — differing essentially in name,
role, authority, and retrieved memories.

This module fixes that generally. It builds a typed :class:`ActorGroundingProfile` per
actor in which every element carries an explicit epistemic mark:

    VERIFIED_OBSERVATION  — directly supported by cited evidence
    SUPPORTED_INFERENCE   — reasoned from cited evidence, labeled as inference
    SIMULATED_HYPOTHESIS  — an explicitly represented unknown explored in simulation
    UNKNOWN               — not found; recorded as missing, never invented

Two rules are structural, not stylistic:

* A verified historical action may never be overwritten by an inferred current
  position. ``previous_observed_action`` and ``current_evidence_grounded_inclination``
  are separate fields with separate provenance.
* Similarity between actors is allowed — actors genuinely share positions — but it must
  come from evidence, never from a generic template. An actor known to exist whose
  grounding is only a name plus a generic role is a defect the gate refuses.

Nothing here is scenario-specific: the same structure grounds a committee member, a
head of state, an organization's representative, or a constructed population stratum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from .errors import WorldIntegrityError


class Provenance(StrEnum):
    """How a grounding element is known. An inference is never presented as a fact."""

    VERIFIED_OBSERVATION = "verified_observation"
    SUPPORTED_INFERENCE = "supported_inference"
    SIMULATED_HYPOTHESIS = "simulated_hypothesis"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GroundedItem:
    """One grounding element with its epistemic status and evidence lineage."""

    content: str
    provenance: Provenance
    claim_ids: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()
    valid_time: str | None = None

    @property
    def is_verified(self) -> bool:
        return self.provenance is Provenance.VERIFIED_OBSERVATION

    def render(self) -> str:
        """Render with the epistemic mark visible, so a model reading the prompt can
        never mistake an inference for an observed fact."""

        cites = f" [{','.join(self.claim_ids)}]" if self.claim_ids else ""
        when = f" ({self.valid_time})" if self.valid_time else ""
        return f"{self.content}{when} — {self.provenance.value.upper()}{cites}"


def verified(content: str, claim_ids: tuple[str, ...] = (), **kw: object) -> GroundedItem:
    return GroundedItem(content, Provenance.VERIFIED_OBSERVATION, claim_ids, **kw)  # type: ignore[arg-type]


def inferred(content: str, claim_ids: tuple[str, ...] = (), **kw: object) -> GroundedItem:
    return GroundedItem(content, Provenance.SUPPORTED_INFERENCE, claim_ids, **kw)  # type: ignore[arg-type]


def unknown(content: str) -> GroundedItem:
    return GroundedItem(content, Provenance.UNKNOWN, ())


@dataclass(frozen=True)
class ActorGroundingProfile:
    """Everything known about one specific actor, with provenance for every element.

    ``is_constructed_representative`` marks a deliberately synthetic agent (a population
    stratum) — such an actor is permitted to be generic, carries a population weight,
    and may never impersonate a named real person.
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
    unresolved_private_hypotheses: tuple[GroundedItem, ...] = ()

    claim_ids: tuple[str, ...] = ()
    lineage_ids: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()

    is_constructed_representative: bool = False
    population_weight: float | None = None

    # ---- derived views -----------------------------------------------------

    @property
    def previous_observed_action(self) -> GroundedItem | None:
        """The most recent verified action, if any. Never an inference."""

        for item in self.previous_observed_actions:
            if item.is_verified:
                return item
        return self.previous_observed_actions[0] if self.previous_observed_actions else None

    def all_items(self) -> list[GroundedItem]:
        groups = (
            self.previous_observed_actions,
            self.direct_statements,
            self.stated_preferences,
            self.inferred_preferences,
            self.goals,
            self.constraints,
            self.commitments,
            self.relationships,
            self.information_access,
            self.relevant_documents,
            self.conditional_reaction_model,
            self.unresolved_private_hypotheses,
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
    def is_only_name_and_role(self) -> bool:
        """True when nothing actor-specific beyond identity/role/authority is known —
        i.e. the actor is effectively a generic role template."""

        substantive = (
            self.previous_observed_actions
            + self.direct_statements
            + self.stated_preferences
            + self.commitments
            + self.goals
        )
        return not any(i.provenance is not Provenance.UNKNOWN for i in substantive)

    def render_grounding(self) -> str:
        """The ACTOR-SPECIFIC GROUNDING block placed in this actor's prompt."""

        lines: list[str] = [f"You are {self.canonical_identity}."]
        if self.aliases:
            lines.append(f"Also referred to as: {', '.join(self.aliases)}")
        lines.append(f"Role: {self.role}")
        lines.append(f"Authority: {', '.join(self.authority) or '(none recorded)'}")
        if self.is_constructed_representative:
            lines.append(
                "NOTE: you are a CONSTRUCTED REPRESENTATIVE agent standing for a "
                f"population segment (weight {self.population_weight}); you are not a "
                "specific named individual."
            )

        def block(title: str, items: tuple[GroundedItem, ...]) -> None:
            if items:
                lines.append(f"\n{title}:")
                lines.extend(f"  - {i.render()}" for i in items)

        block("YOUR OWN PREVIOUS ACTIONS (historical record)", self.previous_observed_actions)
        block("YOUR OWN PUBLIC STATEMENTS", self.direct_statements)
        block("YOUR STATED PREFERENCES", self.stated_preferences)
        block("YOUR COMMITMENTS", self.commitments)
        block("YOUR GOALS", self.goals)
        block("YOUR CONSTRAINTS", self.constraints)
        block("YOUR RELATIONSHIPS", self.relationships)
        block("YOUR INFORMATION ACCESS", self.information_access)
        block("PREFERENCES INFERRED ABOUT YOU (not observed)", self.inferred_preferences)
        if self.current_evidence_grounded_inclination is not None:
            lines.append(
                "\nYOUR CURRENT INCLINATION (an inference about now — it does NOT replace "
                "your previous actions above):"
            )
            lines.append(f"  - {self.current_evidence_grounded_inclination.render()}")
        block("WHAT WOULD CHANGE YOUR POSITION", self.conditional_reaction_model)
        block("OPEN QUESTIONS ABOUT YOUR OWN VIEW", self.unresolved_private_hypotheses)
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
            "unresolved_private_hypotheses": ser(self.unresolved_private_hypotheses),
            "claim_ids": list(self.evidence_claim_ids),
            "lineage_ids": list(self.lineage_ids),
            "missing_information": list(self.missing_information),
            "is_constructed_representative": self.is_constructed_representative,
            "population_weight": self.population_weight,
        }


# ---------------------------------------------------------------------------
# Deterministic actor-grounding coverage gate
# ---------------------------------------------------------------------------


class ActorClaimDisposition(StrEnum):
    INCLUDED_IN_ACTOR_PROFILE = "included_in_actor_profile"
    INCLUDED_AS_SHARED_CONTEXT = "included_as_shared_context"
    EXCLUDED_IRRELEVANT = "excluded_irrelevant"
    MERGED_DUPLICATE = "merged_duplicate"
    UNCERTAIN = "uncertain"
    REQUIRED_BUT_UNRESOLVED = "required_but_unresolved"


@dataclass(frozen=True)
class ActorGroundingReport:
    profiles: tuple[ActorGroundingProfile, ...]
    generic_actors: tuple[str, ...] = ()
    misattributed: tuple[str, ...] = ()
    missing_previous_actions: tuple[str, ...] = ()
    dispositions: tuple[tuple[str, str, str], ...] = ()  # (claim_id, actor_id, disposition)
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not (self.generic_actors or self.misattributed or self.missing_previous_actions)

    def as_dict(self) -> dict[str, object]:
        return {
            "complete": self.is_complete,
            "generic_actors": list(self.generic_actors),
            "misattributed": list(self.misattributed),
            "missing_previous_actions": list(self.missing_previous_actions),
            "profiles": [p.as_dict() for p in self.profiles],
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
    """Check that each named actor is grounded as a specific person rather than a
    generic role template, and that no actor carries another actor's history."""

    generic: list[str] = []
    misattributed: list[str] = []
    missing_prev: list[str] = []
    dispositions: list[tuple[str, str, str]] = []

    # Cross-assignment check. Sharing a *source* is legitimate — one set of minutes or
    # one poll can back every actor — so claim-id overlap is not misattribution. The
    # real defect is an actor's personal record whose CONTENT is about someone else
    # (a profile carrying "<another actor> stated: ...").
    others: dict[str, set[str]] = {}
    for p in profiles:
        keys = {p.actor_id.lower(), p.canonical_identity.lower()}
        keys.update(a.lower() for a in p.aliases)
        # Very short handles ("a", "b") are matched as whole words only; they carry no
        # discriminating signal inside prose and would otherwise fire on any substring.
        others[p.actor_id] = {k for k in keys if len(k) >= 3}
    for p in profiles:
        mine = others[p.actor_id]
        foreign = {k for other, keys in others.items() if other != p.actor_id for k in keys}
        for item in list(p.previous_observed_actions) + list(p.direct_statements):
            low = item.content.lower()
            if any(_names(m, low) for m in mine):
                continue  # the record names this actor: correctly attributed
            hit = next((k for k in sorted(foreign) if _names(k, low)), None)
            if hit:
                misattributed.append(
                    f"{p.actor_id} carries a personal record naming {hit!r}: {item.content[:80]!r}"
                )

    for p in profiles:
        if p.is_constructed_representative:
            # A deliberately synthetic stratum is allowed to be generic, but it must be
            # labeled and carry a population weight.
            if p.population_weight is None:
                generic.append(f"{p.actor_id}: constructed representative without a weight")
            continue
        if p.is_only_name_and_role:
            generic.append(
                f"{p.actor_id} ({p.canonical_identity}): only a name and a generic role — "
                "no verified action, statement, preference, or commitment"
            )
        if require_previous_action and p.previous_observed_action is None:
            missing_prev.append(f"{p.actor_id} ({p.canonical_identity}): no previous action found")
        for cid in p.evidence_claim_ids:
            dispositions.append(
                (cid, p.actor_id, ActorClaimDisposition.INCLUDED_IN_ACTOR_PROFILE.value)
            )

    return ActorGroundingReport(
        profiles=profiles,
        generic_actors=tuple(generic),
        misattributed=tuple(sorted(set(misattributed))),
        missing_previous_actions=tuple(missing_prev),
        dispositions=tuple(dispositions),
    )


def _names(key: str, text: str) -> bool:
    """True when ``key`` occurs in ``text`` as a whole word/phrase (not a substring of
    a longer word), so an id like "ada" cannot match inside "adamant"."""

    return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text) is not None


def enforce_actor_grounding(report: ActorGroundingReport) -> None:
    """Refuse to simulate a world whose named actors are generic role templates."""

    if report.is_complete:
        return
    raise WorldIntegrityError(
        "actors are not grounded as specific people — simulation refused",
        details={
            "generic_actors": list(report.generic_actors),
            "misattributed_evidence": list(report.misattributed),
            "missing_previous_actions": list(report.missing_previous_actions),
        },
    )


# ---------------------------------------------------------------------------
# Building profiles from compiled member specs
# ---------------------------------------------------------------------------

_ACTION_HINTS = ("voted", "vote", "supported", "opposed", "dissent", "preferred", "backed")
_STATEMENT_HINTS = ("said", "stated", "told", "wrote", "speech", "interview", "remarked")


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
    valid_time: str | None = None,
) -> ActorGroundingProfile:
    """Build a grounded profile from an actor's compiled evidence.

    The verified previous action is preserved as its own field; the inferred current
    inclination is recorded separately and explicitly marked as an inference. Seeded
    memories are sorted into previous actions vs statements by their own wording so an
    actor's personal history reaches the prompt as history, not as a generic note.
    """

    prev: list[GroundedItem] = []
    statements: list[GroundedItem] = []
    other: list[GroundedItem] = []
    if previous_action:
        prev.append(
            GroundedItem(
                content=f"took the action: {previous_action}",
                provenance=Provenance.VERIFIED_OBSERVATION,
                claim_ids=previous_action_claim_ids,
                valid_time=valid_time,
            )
        )
    for content, cids in memory_seeds:
        text = content.strip()
        if not text:
            continue
        low = text.lower()
        item = GroundedItem(
            content=text, provenance=Provenance.VERIFIED_OBSERVATION, claim_ids=cids
        )
        if any(h in low for h in _ACTION_HINTS):
            prev.append(item)
        elif any(h in low for h in _STATEMENT_HINTS):
            statements.append(item)
        else:
            other.append(item)

    inclination_item = None
    if inclination:
        inclination_item = GroundedItem(
            content=f"currently expected to favor: {inclination}",
            provenance=Provenance.SUPPORTED_INFERENCE,
            claim_ids=inclination_claim_ids,
        )
    reactions = tuple(
        GroundedItem(
            content=f"if {signal} crosses its threshold, would move toward {option}",
            provenance=Provenance.SUPPORTED_INFERENCE,
        )
        for signal, option in reaction_rules
    )
    missing: list[str] = []
    if not prev:
        missing.append("no previous outcome-relevant action found in evidence")
    if not statements:
        missing.append("no direct public statement found in evidence")

    return ActorGroundingProfile(
        actor_id=actor_id,
        canonical_identity=name,
        role=role,
        authority=authority,
        valid_time=valid_time,
        previous_observed_actions=tuple(prev),
        direct_statements=tuple(statements),
        stated_preferences=tuple(other),
        current_evidence_grounded_inclination=inclination_item,
        conditional_reaction_model=reactions,
        missing_information=tuple(missing),
    )


def with_simulated_hypothesis(
    profile: ActorGroundingProfile, hypothesis: str
) -> ActorGroundingProfile:
    """Attach an explicitly-simulated private-state hypothesis (never a claimed fact)."""

    item = GroundedItem(hypothesis, Provenance.SIMULATED_HYPOTHESIS, ())
    return replace(
        profile, unresolved_private_hypotheses=(*profile.unresolved_private_hypotheses, item)
    )
