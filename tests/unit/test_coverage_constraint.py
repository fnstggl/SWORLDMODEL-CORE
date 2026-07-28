"""Coverage must certify representation, never a mention.

`_match_objects` identifies a candidate with a compiled object three ways: shared
claim ids, name containment, and the two-sided structural match in `_semantic_match`.
The name-containment branch was one-sided in the one case where it cannot mean
anything: a *claim-derived* candidate's `canonical_identity` is its whole proposition
(`_claim_kind_candidates`), so "the object's name occurs in the candidate's identity"
degenerates to "this claim mentions that object by name". Every claim retrieved for a
question mentions its subject, so the branch certified as REPRESENTED every claim that
named the subject entity, whether or not the world contained anything standing for it.

Measured on the recorded eight-party OPEC+ world (society bench arm
``treat_accounting_clean`` run 06, 8 entities / 8 parties holding acts / 7 send
effects): of 19 verified claims the world dropped 8 entirely — no compiled object
carried them and no actor could perceive them. The gate certified 6 of those 8 as
covered, because their propositions contain the string "OPEC" and the world has an
entity named "OPEC+". The remaining 2 fell through to the exclusion challenge, one was
challenged and the world was refused. Which absent claim gets challenged was decided by
a substring test on the claim's prose:

    "context: OPEC+ has begun gradually restoring reduced production levels, with
     full resumption of voluntary cuts scheduled for the coming months."
        -> MERGED, "represented by 'OPEC+' via actor"    (certified covered)

    "The plan includes reviving the remaining one-third of a 1.65 million bpd supply
     cutback (roughly 550,000 bpd) in three monthly stages."
        -> EXCLUDED -> challenged -> UNCERTAIN            (refused the world)

Both describe the same staged unwinding of the same cuts. The world represents
neither. `_semantic_match` already refuses exactly this one-sided match and says why in
its own docstring ("A scheduled event used to be counted as represented whenever its
description merely contained a word like 'decision' ... any such candidate was recorded
INCLUDED while being wired into nothing"). These tests extend that rule to the branch
it never reached.

The direction of the fix is TIGHTENING: it converts false passes into exclusions, so a
world that drops verified evidence is refused more often, not less.
"""

from __future__ import annotations

from sworldmodel.coverage import (
    CandidateKind,
    Disposition,
    EvidenceCandidate,
    Materiality,
    WorldObject,
    WorldSpecView,
    assess_coverage,
)

# --------------------------------------------------------------------------- #
# Fixtures modelled on the recorded run: one wired actor named "OPEC+", and the
# two claim-derived candidates above, neither of which the world represents.
# --------------------------------------------------------------------------- #

SUBJECT_ACTOR = WorldObject(
    object_id="opec",
    kind="actor",
    name="OPEC+",
    claim_ids=("c-in-world",),
    wired=True,
    uses=("actor_view", "action"),
)


def _claim_candidate(cid: str, proposition: str, claim: str) -> EvidenceCandidate:
    """A claim-derived candidate, built the way `_claim_kind_candidates` builds one:
    the whole proposition is the canonical identity."""

    return EvidenceCandidate(
        candidate_id=cid,
        kind=CandidateKind.RESOURCE,
        canonical_identity=proposition,
        description=proposition,
        claim_ids=(claim,),
        lineage_ids=("ev",),
        materiality=Materiality.IMMATERIAL,
    )


NAMES_THE_SUBJECT = _claim_candidate(
    "c_names_subject",
    "context: OPEC+ has begun gradually restoring reduced production levels, with full "
    "resumption of voluntary cuts scheduled for the coming months.",
    "c-restoring",
)
DOES_NOT_NAME_THE_SUBJECT = _claim_candidate(
    "c_silent",
    "The plan includes reviving the remaining one-third of a 1.65 million bpd supply "
    "cutback (roughly 550,000 bpd) in three monthly stages.",
    "c-revival",
)
WORLD = WorldSpecView(objects=(SUBJECT_ACTOR,), accessible_claim_ids=frozenset({"c-in-world"}))


# --------------------------------------------------------------------------- #
# 1. The measured pair: two claims equally absent must be disposed of equally.
# --------------------------------------------------------------------------- #


def test_two_equally_absent_claims_get_the_same_disposition() -> None:
    """The refusal that killed the eight-party world. Neither claim is carried by any
    compiled object and neither is perceivable by any actor, so nothing distinguishes
    them except whether the proposition happens to contain the subject's name."""

    report = assess_coverage((NAMES_THE_SUBJECT, DOES_NOT_NAME_THE_SUBJECT), WORLD)

    names = report.disposition_for("c_names_subject")
    silent = report.disposition_for("c_silent")
    assert names is not None and silent is not None
    assert names.disposition is silent.disposition, (
        "one claim was certified as covered and the other refused the world, and the "
        "only difference between them is a substring of their prose"
    )
    assert names.disposition is Disposition.EXCLUDED_IRRELEVANT


def test_a_mention_is_not_a_representation() -> None:
    """Naming the subject inside a claim does not make the compiled subject stand for
    that claim. The audit record must not say it does."""

    report = assess_coverage((NAMES_THE_SUBJECT,), WORLD)
    disposition = report.disposition_for("c_names_subject")
    assert disposition is not None
    assert disposition.disposition is not Disposition.INCLUDED
    assert disposition.disposition is not Disposition.MERGED
    assert "opec" not in [o.lower() for o in disposition.compiled_object_ids]
    assert disposition.causal_uses == ()


def test_a_mentioned_material_claim_blocks_instead_of_passing() -> None:
    """Tightening, not loosening: the same claim, when the deterministic rules call it
    material, must now block rather than be waved through by the mention."""

    material = EvidenceCandidate(
        candidate_id="c_material",
        kind=CandidateKind.RULE,
        canonical_identity=(
            "rule: OPEC+ approved a new mechanism to reassess maximum sustainable "
            "production capacities."
        ),
        description="rule: OPEC+ approved a new mechanism to reassess capacities.",
        claim_ids=("c-mechanism",),
        lineage_ids=("ev",),
        materiality=Materiality.MATERIAL,
    )
    report = assess_coverage((material,), WORLD)
    disposition = report.disposition_for("c_material")
    assert disposition is not None
    assert disposition.disposition is Disposition.REQUIRED_BUT_UNRESOLVED
    assert not report.is_complete


def test_the_mention_no_longer_hides_an_absent_claim_from_the_reviewer() -> None:
    """The exclusion challenge is the gate's last line on an item nobody represented.
    A mention used to route the item past it entirely — so which absent claim was ever
    reviewed depended on its wording."""

    challenged: list[str] = []

    def review(candidate: EvidenceCandidate) -> bool:
        challenged.append(candidate.candidate_id)
        return False

    assess_coverage(
        (NAMES_THE_SUBJECT, DOES_NOT_NAME_THE_SUBJECT), WORLD, exclusion_reviewer=review
    )
    assert challenged == ["c_names_subject", "c_silent"]


# --------------------------------------------------------------------------- #
# 2. What must NOT change: real name matching, and the claim-id path.
# --------------------------------------------------------------------------- #


def test_a_name_variant_still_matches_the_entity_it_names() -> None:
    """An entity candidate's identity IS a name, so containment between it and an
    object's name is evidence of identity — the case the branch exists for."""

    saudi = EvidenceCandidate(
        candidate_id="c_saudi",
        kind=CandidateKind.PERSON,
        canonical_identity="Saudi Arabia",
        description="person: Saudi Arabia",
        claim_ids=("c-saudi",),
        lineage_ids=("ev",),
        materiality=Materiality.MATERIAL,
    )
    world = WorldSpecView(
        objects=(
            WorldObject(
                object_id="saudi_arabia",
                kind="actor",
                name="Kingdom of Saudi Arabia",
                wired=True,
                uses=("actor_view", "action"),
            ),
        )
    )
    report = assess_coverage((saudi,), world)
    disposition = report.disposition_for("c_saudi")
    assert disposition is not None
    assert disposition.disposition is Disposition.INCLUDED
    assert report.is_complete


def test_a_claim_carried_by_a_wired_object_is_still_covered() -> None:
    """The legitimate path for a claim-derived candidate is shared claim ids: the
    compiler put the claim into a world fact, a field, or an actor's memory, and the
    view reports it. Nothing here narrows that."""

    covered = _claim_candidate(
        "c_covered",
        "The seven OPEC+ countries are scheduled to meet next on August 2, 2026.",
        "c-meeting",
    )
    world = WorldSpecView(
        objects=(
            SUBJECT_ACTOR,
            WorldObject(
                object_id="fact_0",
                kind="world_fact",
                name="the seven countries meet on 2 August 2026",
                claim_ids=("c-meeting",),
                wired=True,
                uses=("actor_view",),
            ),
        ),
        accessible_claim_ids=frozenset({"c-in-world", "c-meeting"}),
    )
    report = assess_coverage((covered,), world)
    disposition = report.disposition_for("c_covered")
    assert disposition is not None
    assert disposition.disposition is Disposition.INCLUDED
    assert disposition.causal_uses == ("actor_view",)
    assert report.is_complete


def test_an_object_that_names_the_whole_proposition_still_represents_it() -> None:
    """The other direction of containment is untouched: an object whose own name
    carries the candidate's entire identity has named the thing, not mentioned it."""

    short = _claim_candidate("c_short", "the strait reopens", "c-strait")
    world = WorldSpecView(
        objects=(
            WorldObject(
                object_id="uncertainty:strait_open",
                kind="scheduled_event",
                name="whether the strait reopens before the meeting",
                wired=True,
                uses=("reaction", "branch"),
            ),
        )
    )
    report = assess_coverage((short,), world)
    disposition = report.disposition_for("c_short")
    assert disposition is not None
    assert disposition.disposition is Disposition.INCLUDED


def test_conflicted_claims_still_block_regardless_of_wording() -> None:
    """A conflicted candidate never depended on the name branch and must not start to."""

    conflicted = EvidenceCandidate(
        candidate_id="c_conflict",
        kind=CandidateKind.RESOURCE,
        canonical_identity="OPEC+ output rises by 550,000 bpd and also does not rise.",
        description="conflicting verified evidence",
        claim_ids=("c-a", "c-b"),
        lineage_ids=("ev",),
        materiality=Materiality.CONFLICTED,
    )
    report = assess_coverage((conflicted,), WORLD)
    disposition = report.disposition_for("c_conflict")
    assert disposition is not None
    assert disposition.disposition is Disposition.REQUIRED_BUT_UNRESOLVED
    assert not report.is_complete
