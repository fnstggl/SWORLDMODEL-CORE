"""Actor invariants: persistent memory, local views, intentions not consequences."""

from __future__ import annotations

import pytest

from _helpers import base_corpus, compile_dict
from sworldmodel.actors import ActorRuntime
from sworldmodel.errors import IntentValidationError
from sworldmodel.gateway import DeterministicGateway, GatewayRequest, GatewayResponse
from sworldmodel.intents import Environment
from sworldmodel.models import EventKind, IntentKind, Visibility


def _compiled():
    return compile_dict(base_corpus())


def _open_decision(world, chair: str, option: str):
    env = Environment()
    ev1 = env.environment_event(
        world,
        kind=EventKind.PROPOSAL_INTRODUCED,
        actor_id=chair,
        payload={"proposal_id": "p1", "option": option, "text": "proposal", "revision_of": None},
        time=world.time,
    )
    world = world.apply([ev1]).with_stage("positions")
    ev2 = env.environment_event(world, kind=EventKind.DECISION_OPENED, payload={}, time=world.time)
    world = world.apply([ev2])
    return world, env


def _vote(world, actor_id: str) -> str:
    rt = ActorRuntime(DeterministicGateway())
    intent, _, _, _ = rt.step(world.actors[actor_id], world.view_for(actor_id), seed=0)
    return str(intent.payload_dict.get("option", ""))


def test_actors_have_non_empty_persistent_memory() -> None:
    compiled = _compiled()
    for actor in compiled.base_world.actors.values():
        assert len(actor.memory) >= 1


def test_actor_cannot_perceive_an_undelivered_message() -> None:
    compiled = _compiled()
    world = compiled.base_world
    env = Environment()
    # A raw MESSAGE_SENT (not a MESSAGE_DELIVERED) is never surfaced.
    sent = env.environment_event(
        world,
        kind="message_sent",
        actor_id="b",
        payload={"text": "psst"},
        time=world.time,
        visibility=Visibility.PRIVATE,
        audience=("a",),
    )
    # A private delivered message to C is not visible to A.
    to_c = env.environment_event(
        world,
        kind=EventKind.MESSAGE_DELIVERED,
        actor_id="b",
        payload={"text": "for carol"},
        time=world.time,
        visibility=Visibility.PRIVATE,
        audience=("c",),
    )
    world = world.apply([sent, to_c])
    view = world.view_for("a")
    kinds = {o.obs_id for o in view.observations}
    assert sent.event_id not in kinds
    assert to_c.event_id not in kinds


def test_actor_cannot_perceive_another_actors_private_state_or_vote() -> None:
    compiled = _compiled()
    world, _ = _open_decision(compiled.base_world, "a", "hold")
    env = Environment()
    vote = env.environment_event(
        world,
        kind=EventKind.VOTE_CAST,
        actor_id="c",
        payload={"option": "cut"},
        time=world.time,
        visibility=Visibility.PRIVATE,
        audience=(),
    )
    world = world.apply([vote])
    view = world.view_for("a")
    assert all(o.obs_id != vote.event_id for o in view.observations)
    assert view.public_votes == ()  # ballots are secret during the meeting


def test_changing_a_proposal_changes_a_vote_without_changing_personality() -> None:
    compiled = _compiled()
    w_hold, _ = _open_decision(compiled.base_world, "a", "hold")
    w_cut, _ = _open_decision(compiled.base_world, "a", "cut")
    # Same actor 'a' (same definition/personality), different focal proposal.
    assert _vote(w_hold, "a") != _vote(w_cut, "a")


def test_changing_a_memory_changes_a_later_decision() -> None:
    compiled = _compiled()
    world, _ = _open_decision(compiled.base_world, "a", "cut")
    baseline = _vote(world, "a")  # accepts the focal cut proposal

    # Give 'a' a durable commitment-to-hold memory; the vote should change.
    a2 = world.actors["a"].clone()
    a2.memory.add_memory(
        "I publicly committed to hold at the last meeting.",
        kind="episodic",
        importance=0.9,
        created=world.time,
        tags=("commitment:hold",),
    )
    world2 = world.with_actor(a2)
    with_commitment = _vote(world2, "a")
    assert baseline != with_commitment


def test_waiting_creates_a_pending_need_then_a_vote_when_info_arrives() -> None:
    compiled = _compiled()
    rt = ActorRuntime(DeterministicGateway())
    # initial stage: no proposal -> the actor waits with a pending need
    world = compiled.base_world
    intent, waited_actor, _, _ = rt.step(world.actors["a"], world.view_for("a"), seed=0)
    assert intent.kind == IntentKind.WAIT
    assert waited_actor.pending_questions  # a pending information need was recorded

    # later: proposal introduced + decision opened -> the same actor now votes
    world = world.with_actor(waited_actor)
    world, _ = _open_decision(world, "a", "hold")
    intent2, _, _, _ = rt.step(world.actors["a"], world.view_for("a"), seed=0)
    assert intent2.kind == IntentKind.CAST_VOTE


def test_canonical_kind_normalizes_live_model_synonyms() -> None:
    from sworldmodel.actors import _canonical_kind

    # Canonical values pass through untouched.
    assert _canonical_kind("cast_vote") == IntentKind.CAST_VOTE
    assert _canonical_kind("make_statement") == IntentKind.MAKE_STATEMENT
    # Common natural shorthands a live LLM emits are canonicalized (label only).
    assert _canonical_kind("vote") == IntentKind.CAST_VOTE
    assert _canonical_kind("Vote") == IntentKind.CAST_VOTE
    assert _canonical_kind("cast a vote") == IntentKind.CAST_VOTE
    assert _canonical_kind("statement") == IntentKind.MAKE_STATEMENT
    assert _canonical_kind("hold") == IntentKind.WAIT
    assert _canonical_kind("abstain") == IntentKind.WAIT
    # A genuinely unknown label is left unchanged, so _to_intent still rejects it.
    assert _canonical_kind("coalition_formed") == "coalition_formed"


def test_closest_option_maps_loose_votes_without_crashing() -> None:
    from sworldmodel.actors import _closest_option

    opts = ("unanimous_hold", "non_unanimous_hold", "cut")
    # Substring containment -> the tightest containing option.
    assert _closest_option("hold", opts) == "unanimous_hold"
    assert _closest_option("cut", ("hold", "cut", "hike")) == "cut"
    # Token overlap when there is no containment.
    assert _closest_option("rate hike", ("hold", "cut", "hike")) == "hike"
    # No match at all -> None (the caller then rejects the vote rather than guessing).
    assert _closest_option("teleport", ("hold", "cut")) is None
    assert _closest_option("hold", ()) is None


def test_live_synonym_vote_becomes_a_cast_vote_intent() -> None:
    class SynonymGateway(DeterministicGateway):
        def _generate(self, request: GatewayRequest) -> GatewayResponse:
            if request.task_kind == "actor_decision":
                stage = request.context.get("stage")
                if stage == "decision":
                    return GatewayResponse(
                        task_kind=request.task_kind,
                        data={"kind": "vote", "vote_option": "hold", "rationale": "steady"},
                        raw_text="{}",
                        model=self.model_id,
                        params={},
                        seed=request.seed,
                        prompt_hash="x",
                        tokens_in=1,
                        tokens_out=1,
                    )
            return super()._generate(request)

    compiled = _compiled()
    world = compiled.base_world
    world, _ = _open_decision(world, "a", "hold")
    world = world.with_stage("decision")
    rt = ActorRuntime(SynonymGateway())
    intent, _, _, _ = rt.step(world.actors["a"], world.view_for("a"), seed=0)
    assert intent.kind == IntentKind.CAST_VOTE
    assert intent.payload_dict.get("option") == "hold"


def test_actor_output_cannot_mark_another_actor_persuaded() -> None:
    class ConsequenceGateway(DeterministicGateway):
        def _generate(self, request: GatewayRequest) -> GatewayResponse:
            if request.task_kind == "actor_decision":
                return GatewayResponse(
                    task_kind=request.task_kind,
                    data={"kind": "coalition_formed", "rationale": "we agreed"},
                    raw_text="{}",
                    model=self.model_id,
                    params={},
                    seed=request.seed,
                    prompt_hash="x",
                    tokens_in=1,
                    tokens_out=1,
                )
            return super()._generate(request)

    compiled = _compiled()
    world, _ = _open_decision(compiled.base_world, "a", "hold")
    rt = ActorRuntime(ConsequenceGateway())
    with pytest.raises(IntentValidationError):
        rt.step(world.actors["a"], world.view_for("a"), seed=0)


def test_actor_intent_cannot_directly_resolve_the_terminal() -> None:
    compiled = _compiled()
    world, env = _open_decision(compiled.base_world, "a", "hold")
    rt = ActorRuntime(DeterministicGateway())
    intent, new_actor, _, _ = rt.step(world.actors["a"], world.view_for("a"), seed=0)
    world = world.with_actor(new_actor)
    events = env.execute(env.validate(intent, world), world)
    world = world.apply(events)
    # Casting a vote does NOT set the terminal — only deterministic tally can.
    assert world.terminal_state is None
