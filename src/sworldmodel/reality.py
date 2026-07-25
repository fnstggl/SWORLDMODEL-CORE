"""Reality-integrity: refuse to simulate a structurally false world.

The manifest owns the determination of whether the world is faithful enough to
simulate. Behavioral rollout does not begin until load-bearing reality facts are
verified. The checks are *generic*: they apply to any compiled world (a decision
body, a negotiation table, a set of population strata, a set of organizations).

The central invariant is preserved without any committee assumption: every participant
the *verified evidence* names must appear in the compiled roster — a nine-participant
body can never become five modeled units. The anchor is the evidence, not the
compiler's own declaration: comparing a compiled roster against a count the same model
response supplied only ever detects that model disagreeing with itself, which is not a
reality check. The compiler's declared count is still compared against its own roster,
but it is reported as what it is — an internal-consistency check.

When the evidence names no decision-relevant person, the participant check cannot run.
The manifest then reports ``expected_participants`` as ``None`` and records why, rather
than reporting a check that did not happen.
"""

from __future__ import annotations

import re

from .actors import ActorState
from .coverage import evidence_named_participants, participants_absent_from
from .errors import EvidenceError, WorldIntegrityError
from .evidence import EvidenceView
from .models import IntegrityVerdict, RealityManifest, ResolutionContract
from .worldspec import WorldSpec


def verify_reality(
    contract: ResolutionContract,
    evidence: EvidenceView,
    actors: dict[str, ActorState],
    spec: WorldSpec,
) -> RealityManifest:
    """Build the manifest and RAISE if the world is not faithful. Returns a VERIFIED
    manifest only when every check that could be run passed, and records in
    ``notes`` every check that could not be run."""

    represented = len(actors)
    roster_names = tuple(a.entity.name for a in actors.values())
    notes: list[str] = []

    # 0. Something must exist that can produce the outcome.
    #
    #    Not necessarily a *person*. A quarterly delivery count is produced by
    #    production throughput, factory schedules, inventory and logistics; requiring a
    #    named individual there would mean inventing an executive who personally decides
    #    how many cars get built, which is a less faithful world than one with no
    #    individuals in it at all. What may never be missing is a causal pathway: some
    #    modeled thing whose operation produces the answer.
    if not actors and not spec.external_processes:
        raise WorldIntegrityError(
            "the compiled world has no causal producer — no actor whose decisions and "
            "no external or operational process whose behavior could produce this "
            "outcome, so there is nothing to simulate",
            details={
                "failure": "no_causal_producer",
                "verified and represented participants": 0,
                "external processes": 0,
                "recompilable": True,
            },
        )
    if not actors:
        notes.append(
            "no individual or organizational actor was compiled: this world resolves "
            f"through {len(spec.external_processes)} modeled process(es). That is "
            "correct only if the outcome is genuinely produced by throughput or "
            "administration rather than by anyone's decision"
        )

    # 1. Every participant the verified evidence names must be present in the compiled
    #    world. This is the only participant check anchored outside the compiler's own
    #    output.
    #
    #    Present in the world — not necessarily as an LLM actor. Whether a named person
    #    should be a deliberating actor, a non-deciding entity, or part of an
    #    organization acting as one unit is a *representation* question the evidence
    #    answers differently for a five-seat board and for a manufacturer's quarterly
    #    output. Forcing every named person into an actor slot is how a system ends up
    #    inventing an executive who personally decides a delivery count. What the
    #    evidence does establish, and what this gate enforces, is that a person it names
    #    in role terms cannot silently vanish from the world.
    entity_names = tuple(e.name for e in spec.entities)
    named = evidence_named_participants(evidence, contract, focal_identities=entity_names)
    expected: int | None
    if not named:
        expected = None
        notes.append(
            "participant count not established: the verified claims name no "
            "decision-relevant person, so the compiled roster was not checked against "
            "evidence and no participant count is reported"
        )
    else:
        expected = len(named)
        absent = participants_absent_from(named, entity_names)
        # A person can be present in a world through the body they act within. Asked
        # whether the EU and Mercosur will sign, a compiler that models the Commission,
        # the Council, the Parliament and the member states has represented the
        # Commission President — inside the institution she leads — and adding her as a
        # separate actor beside her own institution would double-count the same
        # authority. What this gate must not permit is a person the world has no place
        # for at all, so the association has to come from the evidence: some claim that
        # names the person must also name a compiled entity.
        if absent:
            covered, absent = _covered_by_an_organization(absent, evidence, entity_names, spec)
            if covered:
                notes.append(
                    "represented through their institution: "
                    + "; ".join(f"{p} (via {org})" for p, org in sorted(covered))
                )
        if absent:
            raise WorldIntegrityError(
                "the compiled world omits participants the verified evidence names — "
                "simulation refused",
                details={
                    "failure": "participants_omitted",
                    "participants named by evidence": list(named),
                    "compiled entities": list(entity_names),
                    "compiled actor roster": list(roster_names),
                    "absent from the world": list(absent),
                    "recompilable": True,
                },
            )
        present_not_acting = participants_absent_from(named, roster_names)
        if present_not_acting:
            scales = {e.name: e.representation_scale for e in spec.entities}
            notes.append(
                "represented but not deliberating: "
                + "; ".join(
                    f"{n} (as {scales.get(n) or 'entity'})" for n in sorted(present_not_acting)
                )
                + " — the evidence names them, and this world models their effect "
                "through the process rather than through their own decisions"
            )

    # 2. The compiled world's own declared participant count must match the roster it
    #    emitted. A model that says "nine seats" and then emits five has contradicted
    #    itself; this catches that, and nothing more — it is not evidence.
    declared = contract.expected_participants
    if declared is not None and represented != declared:
        # This is the compiler contradicting itself, not evidence contradicting the
        # compiler. It is a defect in one compilation, and the caller may recompile;
        # the detail says so, so a bounded retry can act on it.
        raise WorldIntegrityError(
            "participant roster does not match the count the compiled world declares "
            "for itself — simulation refused",
            details={
                "failure": "roster_count_contradiction",
                "declared by the compiled world": declared,
                "verified and represented participants": represented,
                "difference": (declared - represented),
                "recompilable": True,
            },
        )

    # 3. No duplicated participant (same underlying person occupying two slots).
    names = [n.strip().lower() for n in roster_names]
    dupes = sorted({n for n in names if n and names.count(n) > 1})
    if dupes:
        raise WorldIntegrityError(
            "duplicated participant detected — refused",
            details={"failure": "duplicate_participant", "duplicated": dupes},
        )

    # 4. Every required reality fact must be satisfied by available evidence.
    available_ids = {c.id for c in evidence.available()}
    missing_facts: list[str] = []
    for fact in contract.required_reality_facts:
        if not fact.evidence_claim_ids:
            missing_facts.append(f"{fact.key}: no evidence attached")
            continue
        if not all(cid in available_ids for cid in fact.evidence_claim_ids):
            missing_facts.append(f"{fact.key}: cited evidence not available by cutoff")

    # 5. Decisive evidence contradictions block rollout.
    conflicts = [f"{a} <> {b}" for a, b in evidence.store.contradictions()]

    verdict = (
        IntegrityVerdict.VERIFIED
        if not missing_facts and not conflicts
        else IntegrityVerdict.REFUSED
    )
    coverage, coverage_note = _coverage(contract, available_ids)
    if coverage_note:
        notes.append(coverage_note)
    manifest = RealityManifest(
        verified_entities=tuple(sorted(roster_names)),
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
        notes=tuple(notes),
    )

    if verdict is IntegrityVerdict.REFUSED:
        if conflicts:
            raise EvidenceError(
                "decisive evidence contradictions block rollout: " + "; ".join(conflicts)
            )
        raise WorldIntegrityError(
            "required reality facts are unverified — simulation refused",
            details={
                "failure": "required_facts_unverified",
                "recompilable": True,
                "missing": list(manifest.missing_required_facts),
            },
        )
    return manifest


def _covered_by_an_organization(
    absent: tuple[str, ...],
    evidence: EvidenceView,
    entity_names: tuple[str, ...],
    spec: WorldSpec,
) -> tuple[list[tuple[str, str]], tuple[str, ...]]:
    """Split the absent people into those an included body speaks for, and the rest.

    The link is evidential, not assumed: a claim must name both the person and a
    compiled entity. Co-occurrence in a verified claim is weak on its own, which is why
    it is only ever used to excuse a person from having their *own* slot in a world that
    already models their institution — never to add anyone, and never to establish a
    role.
    """

    claims = list(evidence.available())
    covered: list[tuple[str, str]] = []
    still_absent: list[str] = []
    bodies = {
        e.name.strip().lower()
        for e in spec.entities
        # Only a *body* can speak for a person. A document, an object or a channel that
        # happens to be mentioned in the same sentence cannot, and unanchored substring
        # matching let one do exactly that.
        if e.kind in ("organization", "person") or e.representation_scale in _BODY_SCALES
    }
    for person in absent:
        key = person.strip().lower()
        org = next(
            (
                name
                for claim in claims
                if _mentions(claim.proposition, key)
                or any(key == e.strip().lower() for e in claim.entities)
                for name in entity_names
                if name.strip().lower() != key
                and name.strip().lower() in bodies
                and _mentions(claim.proposition, name.strip().lower())
            ),
            None,
        )
        if org:
            covered.append((person, org))
        else:
            still_absent.append(person)
    return covered, tuple(still_absent)


# Representation scales that can act on a person's behalf.
_BODY_SCALES = frozenset({"organization", "subunit", "network", "population_stratum"})


def _mentions(text: str, name: str) -> bool:
    """Whole-name occurrence, so "Ada" does not match inside "Adams" and a two-word
    entity name is not found by half of it."""

    return re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text.lower()) is not None


def _coverage(contract: ResolutionContract, available_ids: set[str]) -> tuple[float, str]:
    """The fraction of required reality facts whose cited evidence is available.

    Returns ``(fraction, note)``. With no required facts there is nothing to cover, so
    the fraction is vacuous rather than a passing score, and the note says so.
    """

    facts = contract.required_reality_facts
    if not facts:
        return 1.0, (
            "evidence_coverage is vacuous: the compiled world declared no required "
            "reality facts, so no fact was checked for evidential support"
        )
    satisfied = sum(
        1
        for f in facts
        if f.evidence_claim_ids and all(cid in available_ids for cid in f.evidence_claim_ids)
    )
    return satisfied / len(facts), ""
