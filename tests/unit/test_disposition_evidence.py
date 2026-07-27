"""Retrieval must establish *disposition*, and must never certify what it never checked.

Three defects are pinned here, all of them measured on the live
``artifacts/ab/individual_semantic`` store — thirteen real, genuinely contradictory
claims about the Governor of the Bank of England:

1. **A store whose conflicts were never examined certified itself conflict-free.**
   ``EvidenceStore.contradictions()`` returned ``[]`` and every downstream reader took
   that as "checked, and clean". The live detector never paired the contradictory
   claims at all, because it keyed candidates on the *exact* entity tuple plus the
   extractor's namespace prefix, so "a cut is on the way" and "cuts are off the table"
   sat in different buckets and were never compared.
2. **A compiled participant could be admitted on claims that never mention it.** The
   Monetary Policy Committee entered that world at grounding level OFFICIAL_ROLE
   carrying six claim ids, none of which names it anywhere.
3. **Retrieval had no axis for disposition.** Twelve claims record what Bailey wants,
   has done and is constrained by; the compiled actor carried the twelve ids and not
   one word of their content.

Every test here fails against the pre-change tree.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import pytest

from sworldmodel.coverage import participant_evidence_support, unsupported_participants
from sworldmodel.epistemics import GroundingLevel
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.evidence import (
    ConflictScreening,
    ConflictVerdict,
    DispositionAxis,
    EvidenceClaim,
    EvidenceStore,
    EvidenceView,
)
from sworldmodel.grounding import (
    ActorGroundingProfile,
    ClaimAttestation,
    GroundedItem,
    Provenance,
    assess_actor_grounding,
    attest_profiles,
    ground_disposition,
)
from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType

T = datetime.fromisoformat

# The live fixture: the run the adversary measured.
_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "artifacts"
    / "ab"
    / "individual_semantic"
    / "evidence_store.json"
)

# The claims the adversary named, by id.
CUT_ON_THE_WAY = "c-8125ca75ac21"  # "a UK interest rate cut is 'on the way'"
CUTS_OFF_THE_TABLE = "c-c370b40ec24d"  # "previously expected rate cuts are off the table"
IMPLIES_INCREASES = "c-caac4f8dc302"  # inference: scenario B "leading to possible rate increases"

AS_OF = T("2026-07-26T00:00:00+00:00")


def _live_store() -> EvidenceStore:
    """The real 13-claim store, loaded exactly as it was written by the run."""

    store = EvidenceStore()
    for raw in json.loads(_FIXTURE.read_text()):
        store.add(
            EvidenceClaim(
                id=raw["id"],
                proposition=raw["proposition"],
                normalized_value=raw["normalized_value"],
                entities=tuple(raw["entities"]),
                valid_from=None,
                valid_until=None,
                published_at=T(raw["published_at"]),
                available_at=T(raw["available_at"]),
                source_id=raw["source_id"],
                source_url=raw.get("source_url", ""),
                source_title=raw.get("source_title", ""),
                source_type=SourceType(raw["source_type"]),
                authority_level=AuthorityLevel(raw["authority_level"]),
                supporting_excerpt=raw.get("supporting_excerpt", ""),
                lineage_event_id=raw["lineage_event_id"],
                epistemic_type=EpistemicType(raw["epistemic_type"]),
                confidence=float(raw.get("confidence", 0.5)),
                retrieved_at=T(raw["published_at"]),
            )
        )
    return store


def _claim(
    cid: str,
    proposition: str,
    value: str,
    entities: tuple[str, ...],
    *,
    epistemic: EpistemicType = EpistemicType.OBSERVATION,
    excerpt: str = "",
) -> EvidenceClaim:
    pub = T("2026-01-01T00:00:00+00:00")
    return EvidenceClaim(
        id=cid,
        proposition=proposition,
        normalized_value=value,
        entities=entities,
        valid_from=None,
        valid_until=None,
        published_at=pub,
        available_at=pub,
        source_id="s",
        source_url="",
        source_title="",
        source_type=SourceType.CONTEMPORANEOUS_REPORTING,
        authority_level=AuthorityLevel.MEDIUM,
        supporting_excerpt=excerpt or proposition,
        lineage_event_id=f"ev_{cid}",
        epistemic_type=epistemic,
        confidence=0.5,
        retrieved_at=pub,
    )


# ---------------------------------------------------------------------------
# 1. An empty conflict list is not a finding until something looked
# ---------------------------------------------------------------------------


def test_an_unscreened_store_reports_that_it_was_never_screened() -> None:
    """ "No conflicts recorded" and "nobody looked" must not read identically.

    This is the whole defect: ``contradictions()`` returns ``[]`` for a store nothing
    examined and for a store that was examined and found clean, and every gate
    downstream trusted the second reading.
    """

    store = _live_store()
    assert store.contradictions() == []  # unchanged: nothing was ever recorded
    assert store.conflict_screen is None, "a store nobody screened must not carry a screen"
    cert = store.conflict_certificate()
    assert cert["screened"] is False
    assert cert["conclusive"] is False
    assert cert["status"] == ConflictScreening.NOT_SCREENED.value
    assert not store.conflicts_examined()


def test_screening_the_live_store_pairs_the_claims_that_actually_disagree() -> None:
    """The specific conflicts the adversary named must be found, by claim id.

    Before the change these three claims were never compared with one another: the
    candidate key was the extractor's namespace prefix ("", "statement", "inference")
    plus the *exact* entity tuple, and all three differ on both.
    """

    store = _live_store()
    screen = store.screen_conflicts(AS_OF)

    pairs = {(c.a_id, c.b_id) for c in screen.candidates}
    assert (CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE) in pairs, (
        "'a cut is on the way' and 'cuts are off the table' are the same matter with "
        "opposite answers and must be put to an adjudicator"
    )
    # The inference implying increases contradicts both of the statements above.
    assert {IMPLIES_INCREASES, CUT_ON_THE_WAY} in [{a, b} for a, b in pairs]
    assert {IMPLIES_INCREASES, CUTS_OFF_THE_TABLE} in [{a, b} for a, b in pairs]

    headline = next(
        c for c in screen.candidates if (c.a_id, c.b_id) == (CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE)
    )
    assert "cut" in headline.shared_matter and "rate" in headline.shared_matter
    assert headline.verdict is ConflictVerdict.UNADJUDICATED

    # Screening alone is not a verdict. Nothing has adjudicated anything yet.
    assert screen.status is ConflictScreening.SCREENED_UNADJUDICATED
    assert not screen.is_conclusive
    assert store.conflicts_examined()


def test_a_screen_only_pairs_claims_about_the_same_matter() -> None:
    """A shared name is not a shared subject: two unrelated facts about one person are
    not a candidate conflict, or the screen is noise and the real conflict is lost."""

    store = EvidenceStore()
    store.add(_claim("k1", "Ada Lovelace chairs the board", "chair", ("Ada Lovelace",)))
    store.add(_claim("k2", "Ada Lovelace lives in Kent", "kent", ("Ada Lovelace",)))
    store.add(_claim("k3", "Ada Lovelace left the board", "former chair", ("Ada Lovelace",)))
    screen = store.screen_conflicts(AS_OF)
    pairs = {frozenset((c.a_id, c.b_id)) for c in screen.candidates}
    assert frozenset(("k1", "k3")) in pairs, "same matter, different answer"
    assert frozenset(("k1", "k2")) not in pairs, "different matter is not a disagreement"


def test_the_same_answer_written_two_ways_is_not_a_disagreement() -> None:
    """The screen decides "same matter, different answer" and nothing more. Whether one
    value is merely more detailed than another is the adjudicator's judgement, and
    encoding it here would be that judgement smuggled into a deterministic screen — the
    same lexical shape distinguishes "chair"/"former chair", which IS a disagreement."""

    store = EvidenceStore()
    store.add(_claim("d1", "The board cut the reference rate", "rate cut", ("Board",)))
    store.add(_claim("d2", "The board cut the reference rate", "cut the rate", ("Board",)))
    assert store.screen_conflicts(AS_OF).candidates == ()


def test_adjudicating_every_candidate_is_what_makes_an_empty_result_a_finding() -> None:
    store = _live_store()
    screen = store.screen_conflicts(AS_OF)
    for cand in screen.candidates:
        store.record_adjudication(
            cand.a_id, cand.b_id, decisive=False, adjudicator="test", reason="stance, not fact"
        )
    final = store.conflict_screen
    assert final is not None
    assert final.status is ConflictScreening.FULLY_ADJUDICATED
    assert final.is_conclusive
    assert store.contradictions() == []  # still empty — but now it *means* something
    assert store.conflict_certificate()["conclusive"] is True


def test_a_partly_adjudicated_screen_names_the_pairs_nobody_examined() -> None:
    """Budget exhaustion must be visible as unexamined pairs, never as a clean store."""

    store = _live_store()
    screen = store.screen_conflicts(AS_OF)
    first = screen.candidates[0]
    store.record_adjudication(first.a_id, first.b_id, decisive=False, adjudicator="test")
    partial = store.conflict_screen
    assert partial is not None
    assert partial.status is ConflictScreening.PARTIALLY_ADJUDICATED
    assert not partial.is_conclusive
    assert len(partial.unexamined) == len(screen.candidates) - 1
    assert (first.a_id, first.b_id) not in {(c.a_id, c.b_id) for c in partial.unexamined}


def test_a_decisive_adjudication_is_recorded_on_the_claims_themselves() -> None:
    store = _live_store()
    store.screen_conflicts(AS_OF)
    store.record_adjudication(
        CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE, decisive=True, adjudicator="test", reason="both ways"
    )
    assert store.contradictions() == [tuple(sorted((CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE)))]
    assert CUTS_OFF_THE_TABLE in store.get(CUT_ON_THE_WAY).contradiction_ids


def test_screening_is_bounded_by_the_cutoff_view() -> None:
    """A post-cutoff claim may not manufacture a conflict that blocks a pastcast."""

    store = EvidenceStore()
    store.add(_claim("early", "Bank Rate was cut at the June meeting", "cut", ("Bank Rate",)))
    late = _claim("late", "Bank Rate was held at the June meeting", "held", ("Bank Rate",))
    store.add(
        EvidenceClaim(
            **{
                **{f.name: getattr(late, f.name) for f in late.__dataclass_fields__.values()},
                "published_at": T("2027-01-01T00:00:00+00:00"),
                "available_at": T("2027-01-01T00:00:00+00:00"),
            }
        )
    )
    assert store.screen_conflicts(AS_OF).candidates == ()
    assert store.screen_conflicts(T("2027-06-01T00:00:00+00:00")).candidates


# ---------------------------------------------------------------------------
# 2. A compiled participant must be traceable to evidence about it
# ---------------------------------------------------------------------------


def test_the_monetary_policy_committee_is_reported_as_having_no_evidence() -> None:
    """The compiled world named it, held an affordance for it, and gave it six claim
    ids. Not one of the thirteen claims mentions it."""

    view = _live_store().view(AS_OF)
    support = {
        s.identity: s
        for s in participant_evidence_support(view, ("Andrew Bailey", "Monetary Policy Committee"))
    }

    assert support["Andrew Bailey"].is_supported
    assert support["Andrew Bailey"].claim_ids

    mpc = support["Monetary Policy Committee"]
    assert not mpc.is_supported
    assert mpc.claim_ids == ()
    assert unsupported_participants(view, ("Andrew Bailey", "Monetary Policy Committee")) == (
        "Monetary Policy Committee",
    )


def _office_profile(actor_id: str, name: str, claim_ids: tuple[str, ...]) -> ActorGroundingProfile:
    return ActorGroundingProfile(
        actor_id=actor_id,
        canonical_identity=name,
        role="Monetary policy decision-making body",
        authority=("mpc_hold_meeting_and_decide",),
        claim_ids=claim_ids,
    )


def test_an_unattested_office_may_not_be_grounded_as_a_verified_office() -> None:
    """Grounding level OFFICIAL_ROLE means "the evidence records this office". An actor
    whose cited claims never name it has not got one."""

    profile = _office_profile("mpc", "Monetary Policy Committee", ("c-1",))
    assert profile.grounding_assessment().level is GroundingLevel.OFFICIAL_ROLE

    unattested = profile.with_attestation(ClaimAttestation.UNATTESTED, ())
    assessment = unattested.grounding_assessment()
    assert assessment.level is GroundingLevel.NONE
    assert not assessment.admissible
    report = assess_actor_grounding((unattested,))
    assert not report.is_complete
    assert any("Monetary Policy Committee" in u for u in report.ungrounded_actors)


def test_an_unrun_attestation_never_reads_as_a_passed_one() -> None:
    """The check that declines to look must not be mistaken for the check that looked."""

    profile = _office_profile("mpc", "Monetary Policy Committee", ("c-1",))
    assert profile.claim_attestation is ClaimAttestation.NOT_CHECKED
    report = assess_actor_grounding((profile,))
    assert report.unattested_actors == ()
    assert any("not checked" in note.lower() for note in report.notes), (
        "a report over profiles nobody attested must say so"
    )


def test_attesting_against_the_live_store_separates_bailey_from_the_committee() -> None:
    view = _live_store().view(AS_OF)
    bailey = ActorGroundingProfile(
        actor_id="andrew_bailey",
        canonical_identity="Andrew Bailey",
        role="Governor of the Bank of England",
        authority=("signal_support_for_further_cut",),
        claim_ids=(CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE),
    )
    mpc = _office_profile(
        "monetary_policy_committee",
        "Monetary Policy Committee",
        ("c-0ffeac961b66", "c-395c610621b4", "c-9452f1917289"),
    )
    attested = {p.actor_id: p for p in attest_profiles((bailey, mpc), view)}

    assert attested["andrew_bailey"].claim_attestation is ClaimAttestation.ATTESTED
    assert attested["andrew_bailey"].grounding_assessment().level is GroundingLevel.OFFICIAL_ROLE

    assert attested["monetary_policy_committee"].claim_attestation is ClaimAttestation.UNATTESTED
    assert attested["monetary_policy_committee"].name_attested_claim_ids == ()
    report = assess_actor_grounding(tuple(attested.values()))
    assert report.unattested_actors == ("monetary_policy_committee",)
    assert not report.is_complete


# ---------------------------------------------------------------------------
# 3. A disposition axis in retrieval
# ---------------------------------------------------------------------------


def test_retrieval_has_an_axis_for_what_a_participant_wants_and_has_done() -> None:
    view = _live_store().view(AS_OF)
    disp = view.disposition("Andrew Bailey")

    axes = {a.axis for a in disp.axes}
    assert DispositionAxis.WANTS in axes, "twelve claims record what Bailey is pushing for"
    assert disp.supporting_claim_ids, "an axis with no citation is not an axis"
    for axis in disp.axes:
        assert axis.claim_ids, "every axis entry must cite the claims it came from"
        assert set(axis.claim_ids) <= {c.id for c in view.available()}


def test_a_participant_with_no_evidence_has_every_axis_missing_and_none_invented() -> None:
    """An invented disposition is worse than a missing one."""

    view = _live_store().view(AS_OF)
    disp = view.disposition("Monetary Policy Committee")
    assert disp.axes == ()
    assert set(disp.missing_axes) == set(DispositionAxis)
    assert disp.supporting_claim_ids == ()
    assert not disp.is_grounded


def test_a_contested_disposition_is_reported_contested_not_silently_resolved() -> None:
    """The record says Bailey is for a cut, against a cut, and leaning toward increases.
    Retrieval must hand the compiler all three, marked as disagreeing."""

    view = _live_store().view(AS_OF)
    disp = view.disposition("Andrew Bailey")
    assert disp.contested_axes, "the record plainly disagrees about this actor"
    contested_ids = {cid for axis in disp.axes if axis.contested for cid in axis.claim_ids}
    assert {CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE} <= contested_ids
    conflicts = {frozenset(p) for p in disp.conflicting_claim_pairs}
    assert frozenset((CUT_ON_THE_WAY, CUTS_OFF_THE_TABLE)) in conflicts


def test_disposition_queries_search_every_axis_by_name() -> None:
    """Retrieval must go looking for disposition, not hope it falls out of a topic query."""

    queries = EvidenceView(store=EvidenceStore(), as_of=AS_OF).disposition_queries("Saudi Arabia")
    assert len(queries) >= len(DispositionAxis)
    assert all("Saudi Arabia" in q for q in queries)
    joined = " ".join(queries).lower()
    for probe in ("want", "previous", "constrain", "respond"):
        assert probe in joined


def test_grounding_a_disposition_reaches_the_actor_prompt_with_its_marks() -> None:
    view = _live_store().view(AS_OF)
    profile = ActorGroundingProfile(
        actor_id="andrew_bailey",
        canonical_identity="Andrew Bailey",
        role="Governor of the Bank of England",
        authority=("signal_support_for_further_cut",),
        claim_ids=(CUT_ON_THE_WAY,),
    )
    grounded = ground_disposition(profile, view.disposition("Andrew Bailey"))

    assert grounded.stated_preferences or grounded.previous_observed_actions
    assert grounded.contested_dispositions, "a contested record must reach the actor"
    for item in grounded.contested_dispositions:
        assert item.claim_ids
        assert item.provenance is not Provenance.UNSUPPORTED
    rendered = grounded.render_grounding()
    assert "DISAGREE" in rendered.upper()
    assert CUT_ON_THE_WAY in rendered and CUTS_OFF_THE_TABLE in rendered


def test_a_disposition_item_can_never_outrun_its_citations() -> None:
    with pytest.raises(WorldIntegrityError):
        GroundedItem("wants a cut", Provenance.VERIFIED_OBSERVATION, ())


def test_grounding_a_missing_axis_records_it_as_not_known() -> None:
    view = _live_store().view(AS_OF)
    profile = _office_profile("mpc", "Monetary Policy Committee", ())
    grounded = ground_disposition(profile, view.disposition("Monetary Policy Committee"))
    assert grounded.stated_preferences == ()
    assert grounded.previous_observed_actions == ()
    assert any(
        "not established" in m or "no retrieved source" in m for m in grounded.missing_information
    )
    assert all(
        axis.value.replace("_", " ") in " ".join(grounded.missing_information)
        for axis in DispositionAxis
    )
