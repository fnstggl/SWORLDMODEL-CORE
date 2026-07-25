"""The four epistemic classes, and the evidence hierarchy that grounds an actor.

A social world model cannot directly verify a private belief, an intention, an
interpretation, a future action, an organizational response or a population's behavior.
Requiring it to try produces a system that refuses every real question. Allowing it to
skip the distinction produces a system that presents a guess as a fact. The way out is
not a threshold but a *taxonomy*, carried end to end:

    VERIFIED      a directly supported real-world fact
    INFERRED      a defensible conclusion from one or more verified facts
    HYPOTHETICAL  a plausible unresolved alternative, represented as uncertainty
    UNSUPPORTED   no defensible evidentiary or causal basis — it may not enter the world

The classes are not confidence bands. They are different *kinds* of statement, and the
rules that apply to each differ:

* VERIFIED may be shown to an actor as established fact.
* INFERRED may be shown to an actor, always marked as reasoning rather than fact, and
  always carrying the evidence it was reasoned from.
* HYPOTHETICAL may never be shown as either. It belongs to the branch structure: it is
  the thing the simulation exists to resolve.
* UNSUPPORTED never enters the world at all.

Structural facts — who exists, what office they hold, what authority that office
carries, what the decision rule is, when the deadline falls — must be VERIFIED, because
getting them wrong means simulating a different world. Private state must not be, because
demanding it means simulating no world.

# The grounding hierarchy

An actor does not need a quotation to exist. It needs to be *the real occupant of a real
role inside the causal boundary*. The hierarchy below ranks how strongly the evidence
attaches a specific actor to the world, from a first-person record down to
institution-level behavior. Levels 1-6 all admit an actor; they differ in how much of
that actor's disposition is VERIFIED rather than INFERRED. Level 7 is not evidence at
all — it is the labeled-uncertainty fallback, and an actor with nothing above it is a
name with no referent.

This is the correction of a real defect. The previous gate required every actor to carry
a cited, first-person record of its own. Faced with a trade negotiation or a quarterly
production figure — where role and authority are a matter of public record but no
first-person quotation exists in the retrieved sources — a compiler that obeyed the rule
had exactly one safe move: emit no actors. The runtime then refused the world for having
nobody in it. The requirement did not prevent invention; it prevented representation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum

__all__ = [
    "EpistemicClass",
    "GroundingLevel",
    "GroundingAssessment",
    "ADMISSIBLE_LEVELS",
    "is_first_person",
    "level_admits_actor",
    "private_state_class",
]


class EpistemicClass(StrEnum):
    """What kind of statement this is. See the module docstring for the rules."""

    VERIFIED = "verified"
    INFERRED = "inferred"
    HYPOTHETICAL = "hypothetical"
    UNSUPPORTED = "unsupported"

    @property
    def may_reach_an_actor(self) -> bool:
        """VERIFIED and INFERRED may be rendered into a prompt (marked as what they
        are). HYPOTHETICAL belongs to the branch structure, not to anybody's knowledge;
        UNSUPPORTED belongs nowhere."""

        return self in (EpistemicClass.VERIFIED, EpistemicClass.INFERRED)

    @property
    def may_be_stated_as_fact(self) -> bool:
        return self is EpistemicClass.VERIFIED


class GroundingLevel(IntEnum):
    """How strongly the evidence attaches an actor to the world. Lower is stronger.

    The ordering is the product requirement's hierarchy, unchanged. It is an ordering of
    *evidence kinds*, not of certainty: level 2 (an official role) is often far more
    reliable than level 5 (a newspaper paraphrase), but level 2 says less about what the
    actor privately wants, which is why the levels also decide how much of the actor's
    disposition may be stated rather than inferred.
    """

    DIRECT_RECORD = 1  # the actor's own action or first-person statement
    OFFICIAL_ROLE = 2  # verified office, membership and authority
    DOCUMENTED_PRIOR_BEHAVIOR = 3  # recorded past conduct attributed to this actor
    INSTITUTIONAL_POLICY = 4  # official organizational or institutional policy
    CONTEMPORANEOUS_REPORTING = 5  # dated reporting that names this actor
    ROLE_LEVEL_BEHAVIOR = 6  # behavior evidenced at the role or institution level
    LABELED_UNCERTAIN = 7  # nothing but explicitly labeled alternatives
    NONE = 8  # no evidence of any kind

    @property
    def label(self) -> str:
        return _LEVEL_LABELS[self]


_LEVEL_LABELS: dict[GroundingLevel, str] = {
    GroundingLevel.DIRECT_RECORD: "own action or first-person statement",
    GroundingLevel.OFFICIAL_ROLE: "verified office, membership and authority",
    GroundingLevel.DOCUMENTED_PRIOR_BEHAVIOR: "documented prior behavior",
    GroundingLevel.INSTITUTIONAL_POLICY: "official institutional policy",
    GroundingLevel.CONTEMPORANEOUS_REPORTING: "contemporaneous reporting naming this actor",
    GroundingLevel.ROLE_LEVEL_BEHAVIOR: "role-level or institution-level behavioral evidence",
    GroundingLevel.LABELED_UNCERTAIN: "labeled uncertain alternatives only",
    GroundingLevel.NONE: "no evidence",
}


# Levels that admit an actor into the simulation. Everything at or above
# LABELED_UNCERTAIN is a placeholder for evidence rather than evidence, and an actor
# resting on it alone would be invented.
ADMISSIBLE_LEVELS = frozenset(
    {
        GroundingLevel.DIRECT_RECORD,
        GroundingLevel.OFFICIAL_ROLE,
        GroundingLevel.DOCUMENTED_PRIOR_BEHAVIOR,
        GroundingLevel.INSTITUTIONAL_POLICY,
        GroundingLevel.CONTEMPORANEOUS_REPORTING,
        GroundingLevel.ROLE_LEVEL_BEHAVIOR,
    }
)


def level_admits_actor(level: GroundingLevel) -> bool:
    return level in ADMISSIBLE_LEVELS


def private_state_class(level: GroundingLevel) -> EpistemicClass:
    """What the actor's *disposition* may be called at this grounding level.

    Only a direct record lets the system say what this actor holds. A verified office
    supports a defensible inference about priorities and obligations, not a statement
    about preferences. Below that, the disposition is an open alternative and belongs to
    the branch structure rather than to the actor's briefing.
    """

    if level is GroundingLevel.DIRECT_RECORD:
        return EpistemicClass.VERIFIED
    if level in (
        GroundingLevel.OFFICIAL_ROLE,
        GroundingLevel.DOCUMENTED_PRIOR_BEHAVIOR,
        GroundingLevel.INSTITUTIONAL_POLICY,
        GroundingLevel.CONTEMPORANEOUS_REPORTING,
    ):
        return EpistemicClass.INFERRED
    if level is GroundingLevel.ROLE_LEVEL_BEHAVIOR:
        return EpistemicClass.HYPOTHETICAL
    return EpistemicClass.UNSUPPORTED


@dataclass(frozen=True)
class GroundingAssessment:
    """Why this actor is admissible, at what level, and what that implies.

    ``supporting_claim_ids`` are the citations that actually established the level — not
    every claim the actor touches. A reader of the trace can check the level by checking
    those claims.
    """

    actor_id: str
    level: GroundingLevel
    reason: str
    supporting_claim_ids: tuple[str, ...] = ()
    disposition_class: EpistemicClass = EpistemicClass.UNSUPPORTED
    missing: tuple[str, ...] = ()

    @property
    def admissible(self) -> bool:
        return level_admits_actor(self.level)

    def as_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "grounding_level": int(self.level),
            "grounding_level_name": self.level.name.lower(),
            "grounding_basis": self.level.label,
            "reason": self.reason,
            "supporting_claim_ids": list(self.supporting_claim_ids),
            "disposition_epistemic_class": self.disposition_class.value,
            "admissible": self.admissible,
            "missing": list(self.missing),
        }


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")

# Grammatical first person: a closed class of English pronouns. This is a property of
# *voice*, not a domain vocabulary, so it reads the same for a central banker, a trade
# negotiator or a constructed population stratum.
_FIRST_PERSON = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)


def is_first_person(content: str) -> bool:
    """True when this record is written in the actor's own voice.

    First person is now a *promoter* — it raises grounding to level 1 and lets the
    actor's disposition be stated as verified — rather than a requirement for existing.
    """

    return bool(set(_WORD.findall(content.lower())) & _FIRST_PERSON)
