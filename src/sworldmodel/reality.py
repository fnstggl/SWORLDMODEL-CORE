"""Reality-integrity: refuse to simulate a structurally false world.

The manifest owns the determination of whether the world is faithful enough to
simulate. Behavioral rollout does not begin until load-bearing reality facts are
verified. The checks are *generic*: they apply to any compiled world (a decision
body, a negotiation table, a set of population strata, a set of organizations).

The central invariant is preserved without any committee assumption: if the compiled
world claims a specific number of decision-relevant participants, that many must be
verified from evidence — a nine-participant body can never become five modeled units.
Any such mismatch stops the run with :class:`WorldIntegrityError`.
"""

from __future__ import annotations

from .actors import ActorState
from .errors import EvidenceError, WorldIntegrityError
from .evidence import EvidenceView
from .models import IntegrityVerdict, RealityManifest, ResolutionContract


def verify_reality(
    contract: ResolutionContract,
    evidence: EvidenceView,
    actors: dict[str, ActorState],
) -> RealityManifest:
    """Build the manifest and RAISE if the world is not faithful. Returns a VERIFIED
    manifest only when every check passes."""

    represented = len(actors)
    expected = contract.expected_participants
    details_base: dict[str, object] = {
        "expected participants": expected,
        "verified and represented participants": represented,
    }

    # 0. There must be at least one verified actor to simulate.
    if not actors:
        raise WorldIntegrityError(
            "no actors were verified from evidence — simulation refused", details=details_base
        )

    # 1. If the world claims a specific participant count, it must match the roster
    #    exactly (no silent compression of a larger body into fewer modeled units).
    if expected is not None and represented != expected:
        shortfall = expected - represented
        details: dict[str, object] = {**details_base, "difference": shortfall}
        if shortfall > 0:
            # A shortfall is repairable by research: name it in the same label shape the
            # coverage gate uses, so the existing targeted-research repair loop can go
            # looking for the participants the compiled world is missing. This never
            # invents a participant — if research cannot find them, the gate refuses again.
            named = ", ".join(sorted(a.entity.name for a in actors.values()))
            details["missing_material_candidates"] = [
                f"[person] the remaining {shortfall} of {expected} participants in "
                f"{contract.subject_entity or 'the deciding body'} — "
                f"only {represented} verified so far ({named})"
            ]
        raise WorldIntegrityError(
            "participant roster does not match verified reality — simulation refused",
            details=details,
        )

    # 2. No duplicated participant (same underlying person occupying two slots).
    names = [a.entity.name.strip().lower() for a in actors.values()]
    dupes = sorted({n for n in names if n and names.count(n) > 1})
    if dupes:
        raise WorldIntegrityError(
            "duplicated participant detected — refused", details={"duplicated": dupes}
        )

    # 3. Every required reality fact must be satisfied by available evidence.
    available_ids = {c.id for c in evidence.available()}
    missing_facts: list[str] = []
    for fact in contract.required_reality_facts:
        if not fact.evidence_claim_ids:
            missing_facts.append(f"{fact.key}: no evidence attached")
            continue
        if not all(cid in available_ids for cid in fact.evidence_claim_ids):
            missing_facts.append(f"{fact.key}: cited evidence not available by cutoff")

    # 4. Decisive evidence contradictions block rollout.
    conflicts = [f"{a} <> {b}" for a, b in evidence.store.contradictions()]

    verdict = (
        IntegrityVerdict.VERIFIED
        if not missing_facts and not conflicts
        else IntegrityVerdict.REFUSED
    )
    coverage = _coverage(contract, available_ids)
    manifest = RealityManifest(
        verified_entities=tuple(sorted(a.entity.name for a in actors.values())),
        verified_roles=tuple((a.actor_id, a.role) for a in actors.values()),
        verified_authorities=tuple((a.actor_id, a.authority) for a in actors.values()),
        verified_rules=(contract.target_outcome,) if contract.target_outcome else (),
        verified_previous_actions=(),
        unresolved_conflicts=tuple(conflicts),
        missing_required_facts=tuple(missing_facts),
        evidence_coverage=coverage,
        integrity_verdict=verdict,
        expected_participants=expected,
        represented_participants=represented,
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
