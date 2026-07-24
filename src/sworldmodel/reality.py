"""Reality-integrity: refuse to simulate a structurally false world.

The manifest owns the determination of whether the world is faithful enough to
simulate. The simulator does not begin behavioral rollout until load-bearing reality
facts are verified. A nine-seat board can never become five modeled units; a
threshold can never be rescaled to fit a smaller roster; a missing voter can never be
dropped. Any such mismatch stops the run with :class:`WorldIntegrityError`.
"""

from __future__ import annotations

from .actors import ActorState
from .errors import EvidenceError, WorldIntegrityError
from .evidence import EvidenceView
from .models import (
    DecisionRule,
    InstitutionSpec,
    IntegrityVerdict,
    RealityManifest,
    ResolutionContract,
)


def expected_majority_threshold(total_seats: int) -> int:
    """Strict majority: floor(n/2) + 1."""

    return total_seats // 2 + 1


def _threshold_consistent(rule: DecisionRule) -> bool:
    if rule.kind == "majority":
        return rule.threshold == expected_majority_threshold(rule.total_seats)
    if rule.kind == "unanimous":
        return rule.threshold == rule.total_seats
    if rule.kind == "supermajority":
        # A supermajority must exceed a strict majority and not exceed all seats.
        return expected_majority_threshold(rule.total_seats) < rule.threshold <= rule.total_seats
    if rule.kind == "plurality":
        return 1 <= rule.threshold <= rule.total_seats
    return False


def verify_reality(
    contract: ResolutionContract,
    evidence: EvidenceView,
    actors: dict[str, ActorState],
    institution: InstitutionSpec | None,
) -> RealityManifest:
    """Build the manifest and RAISE if the world is not faithful.

    Returns a VERIFIED manifest only when every check passes.
    """

    rule = contract.decision_rule
    voting = [a for a in actors.values() if a.definition.is_voting_seat]
    represented = len(voting)
    expected = contract.expected_voting_seats

    details_base: dict[str, object] = {
        "expected voting seats": expected,
        "verified and represented seats": represented,
    }

    # 1. Seat count must match the verified roster exactly.
    if expected is not None and represented != expected:
        missing = expected - represented
        raise WorldIntegrityError(
            "roster does not match verified reality — simulation refused",
            details={**details_base, "missing seats": missing},
        )

    # 2. The decision rule's total_seats must match the roster.
    if rule.total_seats != represented:
        raise WorldIntegrityError(
            "decision rule total_seats disagrees with represented roster — refused",
            details={"rule.total_seats": rule.total_seats, "represented seats": represented},
        )

    # 3. The threshold must be consistent with the verified rule (no rescaling).
    if not _threshold_consistent(rule):
        raise WorldIntegrityError(
            "threshold is inconsistent with the verified decision rule — refused",
            details={
                "rule.kind": rule.kind,
                "rule.threshold": rule.threshold,
                "rule.total_seats": rule.total_seats,
                "expected_majority": expected_majority_threshold(rule.total_seats),
            },
        )

    # 4. Seat vote powers must match the institution's verified powers.
    if institution is not None:
        declared = dict(institution.seat_vote_powers)
        for a in voting:
            expected_power = declared.get(a.actor_id)
            if expected_power is not None and expected_power != a.definition.vote_power:
                raise WorldIntegrityError(
                    "seat vote power disagrees with verified institution — refused",
                    details={
                        "actor": a.actor_id,
                        "represented power": a.definition.vote_power,
                        "verified power": expected_power,
                    },
                )

    # 5. No duplicated seat (same underlying person occupying two seats).
    names = [a.definition.name.strip().lower() for a in voting]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise WorldIntegrityError(
            "duplicated voting seat detected — refused",
            details={"duplicated": dupes},
        )

    # 6. Every required reality fact must be satisfied by available evidence.
    available_ids = {c.id for c in evidence.available()}
    missing_facts: list[str] = []
    for fact in contract.required_reality_facts:
        if not fact.evidence_claim_ids:
            missing_facts.append(f"{fact.key}: no evidence attached")
            continue
        if not all(cid in available_ids for cid in fact.evidence_claim_ids):
            missing_facts.append(f"{fact.key}: cited evidence not available by cutoff")

    # 7. Decisive evidence contradictions block rollout.
    conflicts = [f"{a} <> {b}" for a, b in evidence.store.contradictions()]

    verdict = (
        IntegrityVerdict.VERIFIED
        if not missing_facts and not conflicts
        else IntegrityVerdict.REFUSED
    )

    coverage = _coverage(contract, available_ids)
    manifest = RealityManifest(
        verified_entities=tuple(sorted(a.definition.name for a in actors.values())),
        verified_roles=tuple((a.actor_id, a.definition.role) for a in actors.values()),
        verified_institutions=(institution.name,) if institution else (),
        verified_memberships=(
            ((institution.institution_id, institution.member_actor_ids),) if institution else ()
        ),
        verified_authorities=tuple((a.actor_id, a.definition.authority) for a in actors.values()),
        verified_rules=(f"{rule.kind}:{rule.threshold}/{rule.total_seats}",),
        verified_previous_actions=tuple(
            sorted(
                f"{a.actor_id}:{a.definition.conditional_behavior.current_inclination}"
                for a in voting
            )
        ),
        unresolved_conflicts=tuple(conflicts),
        missing_required_facts=tuple(missing_facts),
        evidence_coverage=coverage,
        integrity_verdict=verdict,
        expected_voting_seats=expected,
        represented_voting_seats=represented,
    )

    if verdict is IntegrityVerdict.REFUSED:
        if conflicts:
            raise EvidenceError(
                "decisive evidence contradictions block rollout: " + "; ".join(conflicts)
            )
        raise WorldIntegrityError(
            "required reality facts are unverified — simulation refused",
            details={"missing": manifest.missing_required_facts},
        )

    return manifest


def _coverage(contract: ResolutionContract, available_ids: set[str]) -> float:
    facts = contract.required_reality_facts
    if not facts:
        return 1.0
    satisfied = sum(
        1
        for f in facts
        if f.evidence_claim_ids and all(cid in available_ids for cid in f.evidence_claim_ids)
    )
    return satisfied / len(facts)
