"""The integrity gates that stand between verified evidence and a live actor.

Each test here corresponds to a way the system was found to be lying: an uncited
memory reaching an actor stamped VERIFIED_OBSERVATION, a checklist that asked the
compiler for less than the gate enforced, a candidate counted as represented because
its wording contained a domain word, and a participant count that only ever compared
the compiler against itself.
"""

from __future__ import annotations

import copy
from datetime import datetime

import pytest

from _fakes import build_bundle
from _worlds import scheduled_multiparty_world
from sworldmodel.actors import ActorState
from sworldmodel.coverage import (
    CandidateKind,
    Disposition,
    EvidenceCandidate,
    Materiality,
    WorldObject,
    WorldSpecView,
    assess_coverage,
    build_candidate_inventory,
    evidence_checklist,
)
from sworldmodel.epistemics import EpistemicClass, GroundingLevel
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.evidence import EvidenceClaim, EvidenceStore, EvidenceView
from sworldmodel.grounding import (
    GroundedItem,
    Provenance,
    assess_actor_grounding,
    enforce_actor_grounding,
    profile_from_member,
)
from sworldmodel.models import AuthorityLevel, EpistemicType, ResolutionContract, SourceType
from sworldmodel.reality import verify_reality
from sworldmodel.world_compiler import build_base_world
from sworldmodel.worldspec import (
    ActorSpec,
    EntitySpec,
    Expr,
    ProcessGraph,
    TerminalExpression,
    WorldSpec,
)

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")
PUBLISHED = datetime.fromisoformat("2026-04-01T00:00:00+00:00")

BOARD = "the Governing Board"


def _claim(cid: str, proposition: str, entities: tuple[str, ...]) -> EvidenceClaim:
    return EvidenceClaim(
        id=cid,
        proposition=proposition,
        normalized_value="recorded",
        entities=entities,
        valid_from=PUBLISHED,
        valid_until=None,
        published_at=PUBLISHED,
        available_at=PUBLISHED,
        source_id="src",
        source_url="https://example.test/doc",
        source_title="fixture source",
        source_type=SourceType.OFFICIAL_INSTITUTIONAL,
        authority_level=AuthorityLevel.AUTHORITATIVE,
        supporting_excerpt=proposition,
        lineage_event_id=f"ev_{cid}",
        epistemic_type=EpistemicType.OBSERVATION,
        confidence=0.95,
        retrieved_at=PUBLISHED,
    )


def _view(*claims: EvidenceClaim) -> EvidenceView:
    store = EvidenceStore()
    for claim in claims:
        store.add(claim)
    return store.view(AS_OF)


def _contract(**kw: object) -> ResolutionContract:
    return ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=str(kw.get("subject_entity", "the measure")),
        resolution_units="the outcome",
        terminal=TerminalExpression(Expr("const", (True,))),
        target_outcome="the measure carries",
    )


def _actor(entity_id: str, name: str) -> ActorState:
    entity = EntitySpec(
        entity_id, name, "person", is_actor=True, role="member", authority=("decide",)
    )
    return ActorState.from_spec(entity, ActorSpec(entity_id), default_time=AS_OF)


def _spec(actors: dict[str, ActorState]) -> WorldSpec:
    """The minimum WorldSpec the reality gate reads: which entities the world contains
    and whether anything non-agent runs in it."""

    return WorldSpec(
        title="fixture",
        entities=tuple(a.entity for a in actors.values()),
        actors=tuple(ActorSpec(aid) for aid in actors),
        fields=(),
        resources=(),
        channels=(),
        documents=(),
        actions=(),
        process=ProcessGraph(()),
        terminal=TerminalExpression(Expr("const", (True,))),
    )


def _people_view() -> EvidenceView:
    return _view(
        _claim("k_ada", f"Ada North is a voting member of {BOARD}.", ("Ada North",)),
        _claim("k_ben", f"Ben East is a voting member of {BOARD}.", ("Ben East",)),
        _claim("k_cara", f"Cara West is a voting member of {BOARD}.", ("Cara West",)),
    )


# --------------------------------------------------------------------------- #
# 1. Nothing uncited is ever rendered to an actor as an established fact.
# --------------------------------------------------------------------------- #


def test_uncited_memory_seed_is_never_rendered_as_a_verified_observation() -> None:
    """The compiler deletes claim ids that are not in the evidence store. A seed left
    with none is not an observation, must not reach the prompt, and must not satisfy
    the gate."""

    stripped = profile_from_member(
        actor_id="ada",
        name="Ada North",
        role="member",
        authority=("decide",),
        previous_action=None,
        memory_seeds=(("I voted to hold at the last session.", ()),),
    )
    rendered = stripped.render_grounding()
    assert "I voted to hold" not in rendered
    assert "VERIFIED_OBSERVATION" not in rendered
    assert [i.provenance for i in stripped.unsupported_records] == [Provenance.UNSUPPORTED]
    assert not stripped.has_own_cited_record

    report = assess_actor_grounding((stripped,))
    assert not report.is_complete
    with pytest.raises(WorldIntegrityError) as exc:
        enforce_actor_grounding(report)
    assert "ada" in str(exc.value)  # the refusal names the actor

    # The same seed, with its citation intact, is rendered and satisfies the gate.
    cited = profile_from_member(
        actor_id="ada",
        name="Ada North",
        role="member",
        authority=("decide",),
        previous_action=None,
        memory_seeds=(("I voted to hold at the last session.", ("k_ada",)),),
    )
    assert "I voted to hold at the last session. — VERIFIED_OBSERVATION [k_ada]" in (
        cited.render_grounding()
    )
    assert assess_actor_grounding((cited,)).is_complete


def test_a_supported_mark_cannot_be_constructed_without_a_citation() -> None:
    """The invariant is structural, not a convention a future caller can forget."""

    for mark in (Provenance.VERIFIED_OBSERVATION, Provenance.SUPPORTED_INFERENCE):
        with pytest.raises(WorldIntegrityError):
            GroundedItem("she favours a hold", mark, ())


def test_a_cited_record_that_is_not_the_actors_own_does_not_ground_it_as_its_own() -> None:
    """Citations alone are not a personal record: a fact about the world, attached to a
    person, is still not that person's own statement.

    What follows from that is a *level*, not a deletion. The actor is admitted — a real
    member of a real board does not stop existing because no source quotes them — but
    the evidence reaches only their institution, so their own position is HYPOTHETICAL
    and the simulation is what resolves it. Under the previous rule this actor was
    removed from the world, which is how live runs ended up with nobody in them.
    """

    ambient = profile_from_member(
        actor_id="ada",
        name="Ada North",
        role="member",
        authority=("decide",),
        previous_action=None,
        memory_seeds=((f"{BOARD} met on Tuesday.", ("k_ada",)),),
    )
    assert not ambient.has_own_cited_record
    assessment = ambient.grounding_assessment()
    assert assessment.level is GroundingLevel.ROLE_LEVEL_BEHAVIOR
    assert assessment.disposition_class is EpistemicClass.HYPOTHETICAL
    assert assessment.admissible
    assert assess_actor_grounding((ambient,)).is_complete
    # And the actor is told, in its own prompt, that its position is not established.
    assert "role-level or institution-level" in ambient.render_grounding()


def test_a_verified_office_is_enough_to_model_a_real_decision_maker() -> None:
    """The correction this system needed most.

    A commissioner, governor or minister whose office and authority are a matter of
    public record is a real causal producer. No retrieved source quotes their private
    preference — that is exactly the thing the simulation exists to resolve — and
    demanding one before they may exist is what left EU–Mercosur and Tesla with zero
    compiled participants.
    """

    from dataclasses import replace as _replace

    officeholder = _replace(
        profile_from_member(
            actor_id="sef",
            name="the Trade Commissioner",
            role="Commissioner for Trade",
            authority=("sign_on_behalf_of_the_union",),
            previous_action=None,
        ),
        claim_ids=("k_office",),  # the entity's own citation, attesting the office
    )
    assert not officeholder.has_own_cited_record
    assessment = officeholder.grounding_assessment()
    assert assessment.level is GroundingLevel.OFFICIAL_ROLE
    assert assessment.disposition_class is EpistemicClass.INFERRED
    assert assessment.supporting_claim_ids == ("k_office",)
    assert assess_actor_grounding((officeholder,)).is_complete


def test_a_name_with_no_citation_of_any_kind_is_still_refused() -> None:
    """The hierarchy admits weak grounding, not absent grounding. An actor no surviving
    claim attaches to the world in any way is invented, and stays refused."""

    invented = profile_from_member(
        actor_id="ghost",
        name="A. Person",
        role="",
        authority=(),
        previous_action=None,
        memory_seeds=(("I intend to vote against.", ()),),  # citation did not survive
    )
    assessment = invented.grounding_assessment()
    assert assessment.level is GroundingLevel.NONE
    assert not assessment.admissible
    report = assess_actor_grounding((invented,))
    assert not report.is_complete
    with pytest.raises(WorldIntegrityError) as exc:
        enforce_actor_grounding(report)
    assert "ghost" in str(exc.value)


# --------------------------------------------------------------------------- #
# 2. The compiler is asked for exactly what it will be judged on.
# --------------------------------------------------------------------------- #


def test_checklist_lists_every_candidate_the_gate_can_later_demand() -> None:
    """The checklist is built before the world exists, so it cannot know the focal
    identities or signal claims the gate reads off the compiled world. Any candidate
    those inputs can promote to material must therefore already be on the list."""

    view = _view(_claim("k_org", f"{BOARD} is the body that decides the measure.", (BOARD,)))
    contract = _contract()

    without_world = build_candidate_inventory(view, contract)
    org = next(c for c in without_world if c.kind is CandidateKind.ORGANIZATION)
    assert not org.is_material  # nothing yet says this organization matters...

    with_world = build_candidate_inventory(view, contract, focal_identities=(BOARD,))
    promoted = next(c for c in with_world if c.kind is CandidateKind.ORGANIZATION)
    assert promoted.is_material  # ...but the compiled world makes it material at the gate

    checklist = evidence_checklist(view, as_of=AS_OF, horizon=HORIZON)
    assert BOARD in checklist, "the gate can demand it, so the compiler must be shown it"


# --------------------------------------------------------------------------- #
# 3. Representation means wiring, never vocabulary.
# --------------------------------------------------------------------------- #


def test_an_event_is_not_represented_merely_by_saying_decision() -> None:
    """The terminal object is always present in every compiled world. Matching an
    event to it because the event's wording contains "decision" recorded coverage for
    something the world never modelled."""

    event = EvidenceCandidate(
        candidate_id="c_event",
        kind=CandidateKind.SCHEDULED_EVENT,
        canonical_identity="An extraordinary decision meeting is convened on 1 June.",
        description="An extraordinary decision meeting is convened on 1 June.",
        claim_ids=("k_event",),
        lineage_ids=("ev_event",),
        materiality=Materiality.MATERIAL,
    )
    terminal_only = WorldSpecView(
        objects=(
            WorldObject(
                object_id="terminal",
                kind="terminal",
                name="YES when the measure carries",
                wired=True,
                uses=("terminal",),
            ),
        )
    )
    report = assess_coverage((event,), terminal_only)
    disposition = report.disposition_for("c_event")
    assert disposition is not None
    assert disposition.disposition is Disposition.REQUIRED_BUT_UNRESOLVED
    assert not report.is_complete

    # Wired into an actual process node, the same candidate is covered.
    with_node = WorldSpecView(
        objects=(
            WorldObject(
                object_id="node:june_session",
                kind="scheduled_event",
                name="june_session the extraordinary session",
                claim_ids=("k_event",),
                wired=True,
                uses=("process",),
            ),
        )
    )
    assert assess_coverage((event,), with_node).is_complete


# --------------------------------------------------------------------------- #
# 4. The participant count is anchored in evidence, or it is not reported.
# --------------------------------------------------------------------------- #


def test_roster_is_checked_against_the_participants_the_evidence_names() -> None:
    actors = {"ada": _actor("ada", "Ada North"), "ben": _actor("ben", "Ben East")}
    with pytest.raises(WorldIntegrityError) as exc:
        verify_reality(_contract(), _people_view(), actors, _spec(actors))
    assert "Cara West" in str(exc.value)

    actors["cara"] = _actor("cara", "Cara West")
    manifest = verify_reality(_contract(), _people_view(), actors, _spec(actors))
    assert manifest.expected_participants == 3
    assert manifest.represented_participants == 3


def test_an_unestablished_participant_count_is_reported_as_unestablished() -> None:
    """Evidence that names no decision-relevant person cannot establish a count. The
    manifest must say so rather than report a check that never ran."""

    view = _view(_claim("k_ctx", "conditions were unchanged over the quarter.", ()))
    actors = {"ada": _actor("ada", "Ada North")}
    manifest = verify_reality(_contract(), view, actors, _spec(actors))
    assert manifest.expected_participants is None
    assert any("not established" in n for n in manifest.notes)


# --------------------------------------------------------------------------- #
# 5. An actor with no entity is refused, never materialized.
# --------------------------------------------------------------------------- #


def test_an_actor_without_a_matching_entity_is_refused() -> None:
    """Inventing the entity would put a person in the simulation that verified reality
    never described, and the coverage gate — which enumerates entities — would never
    see them."""

    data = copy.deepcopy(scheduled_multiparty_world())
    dropped = data["world_spec"]["entities"].pop()["entity_id"]
    bundle = build_bundle(data)
    contract = _contract()

    with pytest.raises(WorldIntegrityError) as exc:
        build_base_world(bundle.spec, contract, bundle.evidence_store.view(AS_OF), ())
    assert dropped in str(exc.value)


# --------------------------------------------------------------------------- #
# 6. Verification is loose about notation and strict about support.
# --------------------------------------------------------------------------- #


def test_a_date_written_in_prose_supports_a_claim_normalized_to_iso() -> None:
    """The exact claim a live EU–Mercosur run discarded, which was its whole store.

    A Council page reading "Brussels, 17 January 2026 — ... signed ..." was refused
    because the ISO form 2026-01-17 decomposes into 2026, 1 and 17, and the phantom "1"
    from the month can never appear in prose that writes "January". The claim was
    rejected for its own formatting, the store was left empty, the compiler was handed
    nothing, and the run reported that no decision-maker could be found.
    """

    from sworldmodel.source_extract import verify_claim

    document = (
        "Brussels, 17 January 2026 - The European Union and Mercosur signed the "
        "Partnership Agreement at a ceremony in Brazil."
    )
    assert (
        verify_claim(
            proposition="The European Union and Mercosur signed the trade agreement",
            normalized_value="2026-01-17",
            entities=("European Union", "Mercosur"),
            excerpt="The European Union and Mercosur signed the Partnership Agreement",
            document=document,
        )
        == ""
    )


def test_verification_still_refuses_an_unsupported_value_and_a_wrong_date() -> None:
    """Loosening notation may not loosen support. Both of these must still fail — the
    second is the hole that opens if `normalized_value` stops being checked at all,
    since the compiler's evidence listing shows exactly that value."""

    from sworldmodel.source_extract import verify_claim

    wrong_date = verify_claim(
        proposition="the agreement was signed",
        normalized_value="2026-02-17",
        entities=("agreement",),
        excerpt="the agreement was signed on 17 January 2026",
        document="the agreement was signed on 17 January 2026",
    )
    assert "date" in wrong_date

    smuggled = verify_claim(
        proposition="the rate was held",
        normalized_value="8.50",
        entities=("Committee",),
        excerpt="the Committee held the rate unchanged",
        document="the Committee held the rate unchanged",
    )
    assert "value" in smuggled
