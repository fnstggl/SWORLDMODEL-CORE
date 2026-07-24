"""Typed integrity failures and universal repair routing.

The repair loop used to understand exactly one failure — a coverage-gate omission —
because it keyed on the ``missing_material_candidates`` detail. Every other refusal
(no actors, participant mismatch, ungrounded actor, unverified required fact)
propagated immediately, so live runs failed on problems that targeted research could
have fixed.

This module classifies a :class:`WorldIntegrityError` into a typed
:class:`IntegrityFailure` carrying what is missing and what would resolve it, so the
repair loop can route each failure to the right response instead of one generic retry.

The gates are NOT weakened. A failure is repairable only when more evidence could
honestly satisfy it. Structurally false worlds — a surplus roster, a duplicated
participant, a decisive contradiction — are deliberately non-repairable and refuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from .errors import WorldIntegrityError


class FailureType(StrEnum):
    NO_CAUSALLY_RELEVANT_ACTORS = "no_causally_relevant_actors"
    ACTOR_GROUNDING_INCOMPLETE = "actor_grounding_incomplete"
    ACTOR_IDENTITY_UNVERIFIED = "actor_identity_unverified"
    PARTICIPANT_COUNT_MISMATCH = "participant_count_mismatch"
    PARTICIPANT_SURPLUS = "participant_surplus"
    DUPLICATED_PARTICIPANT = "duplicated_participant"
    MISSING_MATERIAL_EVIDENCE = "missing_material_evidence"
    MISSING_REQUIRED_REALITY_FACT = "missing_required_reality_fact"
    CONTRADICTORY_LOAD_BEARING_EVIDENCE = "contradictory_load_bearing_evidence"
    WORLD_COVERAGE_FAILURE = "world_coverage_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntegrityFailure:
    """A structured description of why a world was refused, and what would fix it."""

    failure_type: FailureType
    message: str
    retryable: bool = False
    affected_actor_ids: tuple[str, ...] = ()
    missing_candidates: tuple[str, ...] = ()
    expected_value: int | None = None
    represented_value: int | None = None
    conflicting_claim_ids: tuple[str, ...] = ()
    targeted_research_needs: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "failure_type": self.failure_type.value,
            "message": self.message,
            "retryable": self.retryable,
            "affected_actor_ids": list(self.affected_actor_ids),
            "missing_candidates": list(self.missing_candidates),
            "expected_value": self.expected_value,
            "represented_value": self.represented_value,
            "conflicting_claim_ids": list(self.conflicting_claim_ids),
            "targeted_research_needs": list(self.targeted_research_needs),
        }


def _strs(value: object) -> tuple[str, ...]:
    return tuple(str(x) for x in value) if isinstance(value, list | tuple) else ()


def _int(value: object) -> int | None:
    return value if isinstance(value, int) else None


_NAME = re.compile(r"^(?:\[[^\]]+\]\s*)?([^—:(]+)")


def _research_need(label: str) -> str:
    """The searchable identity inside a gate label like '[person] Ada Lin — absent'."""

    m = _NAME.match(label.strip())
    return (m.group(1) if m else label).strip()


def classify_failure(exc: WorldIntegrityError) -> IntegrityFailure:
    """Turn a gate refusal into a typed, routable failure."""

    details = exc.details or {}
    msg = str(exc).splitlines()[0]
    low = msg.lower()

    missing_material = _strs(details.get("missing_material_candidates"))
    if missing_material:
        return IntegrityFailure(
            failure_type=FailureType.MISSING_MATERIAL_EVIDENCE,
            message=msg,
            retryable=True,
            missing_candidates=missing_material,
            targeted_research_needs=tuple(_research_need(m) for m in missing_material),
        )

    # Actor grounding: a named actor exists but carries no verified personal record.
    generic = _strs(details.get("generic_actors"))
    missing_prev = _strs(details.get("missing_previous_actions"))
    if generic or missing_prev:
        needs = tuple(_research_need(x) for x in (*generic, *missing_prev))
        return IntegrityFailure(
            failure_type=FailureType.ACTOR_GROUNDING_INCOMPLETE,
            message=msg,
            retryable=True,
            affected_actor_ids=needs,
            targeted_research_needs=needs,
        )
    # One actor carrying another's record is a compilation defect, not an evidence gap.
    if _strs(details.get("misattributed_evidence")) or _strs(details.get("misattributed")):
        return IntegrityFailure(
            failure_type=FailureType.ACTOR_IDENTITY_UNVERIFIED, message=msg, retryable=False
        )

    if "no actors" in low or "no participants" in low:
        return IntegrityFailure(
            failure_type=FailureType.NO_CAUSALLY_RELEVANT_ACTORS,
            message=msg,
            retryable=True,
            targeted_research_needs=("who or what actually decides this outcome",),
        )

    if "duplicated" in low:
        return IntegrityFailure(
            failure_type=FailureType.DUPLICATED_PARTICIPANT, message=msg, retryable=False
        )

    expected, represented = (
        _int(details.get("expected participants")),
        _int(details.get("represented participants")),
    )
    if expected is None:
        expected = _int(details.get("expected voting seats"))
    if represented is None:
        represented = _int(details.get("verified and represented seats"))
    if expected is not None and represented is not None and expected != represented:
        # A SHORTFALL is an evidence gap: research can find the missing participants.
        # A SURPLUS is a structurally false world and must never be "repaired" away.
        shortfall = represented < expected
        return IntegrityFailure(
            failure_type=(
                FailureType.PARTICIPANT_COUNT_MISMATCH
                if shortfall
                else FailureType.PARTICIPANT_SURPLUS
            ),
            message=msg,
            retryable=shortfall,
            expected_value=expected,
            represented_value=represented,
            targeted_research_needs=(
                (f"the complete list of all {expected} participants and their identities",)
                if shortfall
                else ()
            ),
        )

    missing_facts = _strs(details.get("missing"))
    if missing_facts:
        return IntegrityFailure(
            failure_type=FailureType.MISSING_REQUIRED_REALITY_FACT,
            message=msg,
            retryable=True,
            missing_candidates=missing_facts,
            targeted_research_needs=tuple(_research_need(m) for m in missing_facts),
        )

    if "contradiction" in low or "conflict" in low:
        return IntegrityFailure(
            failure_type=FailureType.CONTRADICTORY_LOAD_BEARING_EVIDENCE,
            message=msg,
            retryable=False,
        )

    return IntegrityFailure(failure_type=FailureType.UNKNOWN, message=msg, retryable=False)
