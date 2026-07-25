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
        expected_participants=kw.get("expected_participants"),  # type: ignore[arg-type]
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

    from dataclasses import replace as _replace

    ambient = _replace(
        profile_from_member(
            actor_id="ada",
            name="Ada North",
            role="member",
            authority=("decide",),
            previous_action=None,
            memory_seeds=((f"{BOARD} met on Tuesday.", ("k_ada",)),),
        ),
        # What the compiler attaches to a real entity: the claims that establish it.
        claim_ids=("k_ada",),
    )
    assert not ambient.has_own_cited_record
    assessment = ambient.grounding_assessment()
    # A cited office with cited authority is level 2, and that is the right answer: what
    # is *not* established is the actor's own position, which becomes an inference.
    assert assessment.level is GroundingLevel.OFFICIAL_ROLE
    assert assessment.disposition_class is EpistemicClass.INFERRED
    assert assessment.admissible
    assert assess_actor_grounding((ambient,)).is_complete
    assert "verified office" in ambient.render_grounding()

    # The same person with no authority recorded falls to role level, and their position
    # becomes an open alternative rather than an inference.
    role_only = _replace(ambient, authority=())
    weaker = role_only.grounding_assessment()
    assert weaker.level is GroundingLevel.ROLE_LEVEL_BEHAVIOR
    assert weaker.disposition_class is EpistemicClass.HYPOTHETICAL
    assert weaker.admissible
    assert "role-level or institution-level" in role_only.render_grounding()


def test_role_level_grounding_still_needs_the_entity_itself_to_be_cited() -> None:
    """The floor under the weakest admissible level.

    Without this, one irrelevant claim id hung on an invented person's memory seed —
    plus any non-empty role string — was enough to admit an actor that nothing in the
    evidence refers to. The entity must carry a surviving citation of its own.
    """

    uncited_entity = profile_from_member(
        actor_id="ghost",
        name="A. Person",
        role="member",  # a role string proves nothing on its own
        authority=(),
        previous_action=None,
        memory_seeds=(("The meeting took place.", ("k_ada",)),),  # cited, but not about them
    )
    assessment = uncited_entity.grounding_assessment()
    assert assessment.level is GroundingLevel.NONE
    assert not assessment.admissible


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


def test_source_provenance_is_excluded_and_never_put_to_the_reviewer() -> None:
    """A live Bank of England run was refused because an independent reviewer challenged
    the exclusion of "The document was published on September 18, 2025". Provenance is
    recorded against every claim the page supports and exists nowhere in the world, so
    the challenge could only ever be unsatisfiable — no recompile can include it."""

    def challenge_everything(_: EvidenceCandidate) -> bool:
        return True

    provenance = EvidenceCandidate(
        candidate_id="c_prov",
        kind=CandidateKind.DOCUMENT,
        canonical_identity="context: The document was published on September 18, 2025.",
        description="context: The document was published on September 18, 2025.",
        claim_ids=("k_prov",),
        lineage_ids=("ev_prov",),
        materiality=Materiality.IMMATERIAL,
    )
    empty = WorldSpecView(objects=())
    report = assess_coverage((provenance,), empty, exclusion_reviewer=challenge_everything)
    disposition = report.disposition_for("c_prov")
    assert disposition is not None
    assert disposition.disposition is Disposition.EXCLUDED_IRRELEVANT
    assert "provenance" in disposition.reason
    assert report.is_complete

    # A named body publishing a named document is an event in the world, not provenance,
    # and the challenge still blocks.
    world_event = EvidenceCandidate(
        candidate_id="c_event",
        kind=CandidateKind.SCHEDULED_EVENT,
        canonical_identity="date: The Bank of England published its minutes on 18 September 2025.",
        description="date: The Bank of England published its minutes on 18 September 2025.",
        claim_ids=("k_event",),
        lineage_ids=("ev_event",),
        materiality=Materiality.IMMATERIAL,
    )
    blocked = assess_coverage((world_event,), empty, exclusion_reviewer=challenge_everything)
    challenged = blocked.disposition_for("c_event")
    assert challenged is not None
    assert challenged.disposition is Disposition.UNCERTAIN
    assert not blocked.is_complete


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


def test_a_declared_roster_the_world_never_populated_is_refused() -> None:
    """The motivating failure: the compiler says nine seats and puts five people in the
    world. Counting what the world contains is what catches it."""

    actors = {"ada": _actor("ada", "Ada North")}
    with pytest.raises(WorldIntegrityError) as exc:
        verify_reality(_contract(expected_participants=9), _view(), actors, _spec(actors))
    assert exc.value.details["failure"] == "declared_participants_not_represented"
    assert exc.value.details["represented in the world"] == 1
    assert exc.value.details["recompilable"] is True


def test_participants_carried_by_a_process_do_not_contradict_the_declared_count() -> None:
    """A live OPEC+ run declared eight producer countries, modelled the group as one
    deliberating unit and the eight beside it as entities the process carries, and was
    refused for it — by the gate that sits directly below the one which had just
    accepted exactly that representation. Representation scale is not a contradiction,
    and representing more of the world than the count names is not one either.
    """

    deciding = _actor("opec_plus", "OPEC+")
    carried = tuple(
        EntitySpec(f"e{i}", name, "organization", is_actor=False)
        for i, name in enumerate(("Saudi Arabia", "Russia", "Iraq", "UAE"))
    )
    spec = WorldSpec(
        title="fixture",
        entities=(deciding.entity, *carried),
        actors=(ActorSpec("opec_plus"),),
        fields=(),
        resources=(),
        channels=(),
        documents=(),
        actions=(),
        process=ProcessGraph(()),
        terminal=TerminalExpression(Expr("const", (True,))),
    )
    manifest = verify_reality(
        _contract(expected_participants=4), _view(), {"opec_plus": deciding}, spec
    )
    assert manifest.represented_participants == 1


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


def test_a_lone_expression_argument_is_read_as_a_one_argument_list() -> None:
    """`args` is a list by schema, and a model will still write the single argument
    bare: {"op": "const", "args": false}. Iterating that raised TypeError from inside
    the parser and killed a live Bank of England run before it could write any
    diagnosis at all. Reading a lone argument as a one-argument list changes no meaning.
    """

    from sworldmodel.worldspec import parse_expr

    assert parse_expr({"op": "const", "args": False}) == Expr("const", (False,))
    assert parse_expr({"op": "field", "args": "rate"}) == Expr("field", ("rate",))
    assert parse_expr({"op": "const", "args": None}) == Expr("const", ())
    # The ordinary shape is unchanged.
    nested = parse_expr({"op": "equals", "args": [{"op": "field", "args": ["x"]}, 3]})
    assert nested.op == "equals" and nested.args[1] == 3


def test_an_unparsable_compilation_is_a_refusal_the_repair_loop_can_act_on() -> None:
    """A shape the parser cannot read is a defect in one compilation, not a fact about
    the world. It must arrive as a recompilable refusal carrying its parser error —
    never as a raw TypeError from inside a parser, which leaves no diagnosis behind."""

    from sworldmodel.errors import WorldIntegrityError
    from sworldmodel.repair import plan_repair

    exc = WorldIntegrityError(
        "the compiled world could not be parsed",
        details={
            "failure": "malformed_compilation",
            "recompilable": True,
            "parser_error": "TypeError: 'bool' object is not iterable",
        },
    )
    plan = plan_repair(exc, "will x happen?")
    assert plan is not None
    assert not plan.needs_research  # nothing is missing from the evidence
    assert "could not be parsed" in plan.instruction
    assert "args" in plan.instruction  # it names the shape to fix


def test_a_quantity_must_come_from_the_quoted_sentence_or_an_adjacent_line() -> None:
    """The model picks both the claim and the quote, having read the whole page, so any
    reach measured in characters lets it quote one sentence and assert a number from
    another. A 150-character window still accepted a staff headcount four sentences on.

    The boundary that works is linguistic: the sentence that was quoted, plus the lines
    around it — which is what a table is, a heading on one line and its value beneath.
    """

    from sworldmodel.source_extract import verify_claim

    borrowed = verify_claim(
        proposition="The Committee voted 4750 to maintain Bank Rate",
        normalized_value="4750",
        entities=("Committee",),
        excerpt="The Committee voted to maintain Bank Rate",
        document=(
            "The Committee voted to maintain Bank Rate. Further discussion followed. "
            "More talk. Then more. The Bank employs 4750 staff across its sites."
        ),
    )
    assert "value" in borrowed, "a number from an unrelated sentence was accepted"

    # A table heading and the value on the next line is the case the region exists for.
    assert (
        verify_claim(
            proposition="Tesla delivered 389407 vehicles in Q2 2026",
            normalized_value="389407",
            entities=("Tesla",),
            excerpt="Q2 2026 total vehicle deliveries",
            document=(
                "Quarterly deliveries\nTesla Q2 2026 total vehicle deliveries\n"
                "389,407 units delivered"
            ),
        )
        == ""
    )


def test_a_decisive_contradiction_is_repairable_before_it_is_fatal() -> None:
    """A live OPEC+ run was refused because one source reported quotas held for 2026 and
    another said cuts would be unwound in 2026 — a report of a decision and a projection
    about what comes next, which are not even inconsistent.

    Two properties follow. A disagreement about what *will* happen is the uncertainty
    the simulation exists to resolve, so it is not decisive at all. And a genuine
    contradiction about a matter of fact should first send the researcher after a source
    that settles it: only one that survives that ends the run.
    """

    from sworldmodel.repair import plan_repair

    exc = WorldIntegrityError(
        "decisive evidence contradictions block rollout: a <> b",
        details={
            "failure": "decisive_evidence_contradiction",
            "recompilable": True,
            "contradictions": ["the board has five members <> the board has nine members"],
            "claim pairs": ["a <> b"],
        },
    )
    plan = plan_repair(exc, "will the board decide x?", subject_entity="the board")
    assert plan is not None
    assert plan.needs_research, "a contradiction should send us looking for what settles it"
    assert any("official" in q for q in plan.queries)
    # And the instruction offers the honest alternative rather than forcing a choice.
    assert "uncertainty" in plan.instruction
    assert "without a source" in plan.instruction


def test_every_gate_failure_code_has_a_repair_plan() -> None:
    """A failure code with no plan is a refusal the loop cannot act on.

    `terminal_reads_no_world_state` shipped without one and surfaced in a live run as
    "no repair plan for this failure" — a gate I had added, refusing correctly, and then
    dead-ending because nothing knew what to do about it. The set is small enough to
    check exhaustively, so it is.
    """

    from sworldmodel.repair import _PLANS

    emitted = {
        "no_causal_producer",
        "actors_ungrounded",
        "participants_omitted",
        "declared_participants_not_represented",
        "duplicate_participant",
        "required_facts_unverified",
        "decisive_evidence_contradiction",
        "orphan_actors",
        "nothing_can_act",
        "nothing_scheduled",
        "terminal_has_no_producer",
        "terminal_reads_no_world_state",
        "actors_cannot_reach_terminal",
        "environment_presets_terminal",
        "unknown_expression_operator",
        "malformed_compilation",
        "coverage_incomplete",
    }
    # `duplicate_participant` is deliberately unplanned: the same person in two seats is
    # not something more evidence or a rewritten instruction can fix.
    unplanned = emitted - set(_PLANS)
    assert unplanned == {"duplicate_participant"}, unplanned


def test_a_committee_still_needs_every_member_the_evidence_names() -> None:
    """The relaxation below must not touch collective decisions.

    Several actors can perform the terminal-producing action here, so this is not one
    person's own act and every named member is still required.
    """

    actors = {"ada": _actor("ada", "Ada North"), "ben": _actor("ben", "Ben East")}
    with pytest.raises(WorldIntegrityError) as exc:
        verify_reality(_contract(), _people_view(), actors, _spec(actors))
    assert "Cara West" in str(exc.value)


def test_one_persons_own_act_does_not_require_the_people_around_them() -> None:
    """Asked whether one named person will do something, the people around them are
    context, not co-producers.

    A live Bank of England run compiled Andrew Bailey — whose own statement is the
    entire outcome — and was refused for omitting two other committee members the
    evidence names in role terms. The compiler then correctly resisted adding them
    through three repair rounds, because they do not produce Bailey's statement. The
    demand belongs to worlds whose outcome is produced collectively.
    """

    from sworldmodel.worldspec import ActionDefinition, Effect

    entity = EntitySpec(
        "ada", "Ada North", "person", is_actor=True, role="member", authority=("speak",)
    )
    actors = {"ada": ActorState.from_spec(entity, ActorSpec("ada"), default_time=AS_OF)}
    spec = WorldSpec(
        title="one person's own act",
        entities=(entity,),
        actors=(ActorSpec("ada"),),
        fields=(),
        resources=(),
        channels=(),
        documents=(),
        # Only Ada can perform the action that writes the terminal term.
        actions=(
            ActionDefinition(
                action_id="speak",
                meaning="say it publicly",
                eligible_actors=("ada",),
                effects=(Effect("set_field", (("field", "said"), ("value", True))),),
            ),
        ),
        process=ProcessGraph(()),
        terminal=TerminalExpression(
            Expr("equals", (Expr("field", ("said",)), True)), Expr("const", (False,))
        ),
    )
    contract = _contract(subject_entity="Ada North")

    # Ben and Cara are named in role terms by the evidence and are not in the world.
    manifest = verify_reality(contract, _people_view(), actors, spec)
    assert manifest.integrity_verdict.value == "verified"
    # Nothing is silent: the manifest records who was named and why they were omitted.
    assert any("named but not modelled" in n for n in manifest.notes)
    assert any("Ben East" in n and "Cara West" in n for n in manifest.notes)
