"""Actor invariants: persistent memory, actor-local views, intentions not consequences."""

from __future__ import annotations

from dataclasses import replace

from _helpers import base_corpus, compile_dict
from sworldmodel.actors import ActorRuntime
from sworldmodel.effects import EffectExecutor
from sworldmodel.executor import ActionExecutor
from sworldmodel.gateway import DeterministicGateway
from sworldmodel.models import Visibility
from sworldmodel.worldspec import ActionChoice


def _exec() -> ActionExecutor:
    return ActionExecutor(DeterministicGateway(), EffectExecutor())


def test_actors_have_non_empty_persistent_memory() -> None:
    for actor in compile_dict(base_corpus()).base_world.actors.values():
        assert len(actor.memory) >= 1


def test_actor_cannot_perceive_a_private_event_meant_for_another() -> None:
    world = compile_dict(base_corpus()).base_world
    eff = EffectExecutor()
    ev = eff.raw_event(
        world,
        kind="deliver_information",
        actor_id="b",
        payload={"text": "for carol only"},
        visibility=Visibility.PRIVATE,
        audience=("c",),
    )
    world = world.apply([ev])
    assert all(o.obs_id != ev.event_id for o in world.view_for("a").observations)
    assert any(o.obs_id == ev.event_id for o in world.view_for("c").observations)


def test_actor_emits_intention_environment_produces_the_consequence() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("decide")
    actor = world.actors["a"]
    choice = ActionChoice(
        mode="compiled_action", action_id="record_position", params=(("position", "hold"),)
    )
    outcome = _exec().execute(actor, choice, world, compiled.spec, seed=0)
    assert outcome.status == "executed"
    world2 = world.apply(outcome.events)
    # The consequence is a world record; the actor's turn never sets the terminal.
    assert world2.get_records("votes")
    assert world2.terminal_state is None


def test_unauthorized_compiled_action_is_rejected() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("decide")
    stripped = world.actors["a"]
    stripped = replace(stripped, entity=replace(stripped.entity, authority=()))
    choice = ActionChoice(
        mode="compiled_action", action_id="record_position", params=(("position", "hold"),)
    )
    outcome = _exec().execute(stripped, choice, world, compiled.spec, seed=0)
    assert outcome.status == "rejected"
    assert "authority" in outcome.reason
    # No decisive record was written.
    assert not world.apply(outcome.events).get_records("votes")


def test_observed_field_change_flips_the_reacted_action_not_the_identity() -> None:
    compiled = compile_dict(base_corpus())
    spec = compiled.spec
    action = spec.action("record_position")
    assert action is not None
    ex = _exec()
    rt = ActorRuntime(DeterministicGateway())

    def choose(shock: float) -> str:
        world = compiled.base_world.with_stage("decide")
        ev = EffectExecutor().raw_event(
            world,
            kind="release_data",
            actor_id=None,
            payload={"fields": {"shock": shock}},
            visibility=Visibility.PUBLIC,
        )
        world = world.apply([ev])
        view = replace(
            world.view_for("c"), feasible_actions=(ex.action_card(action),), allow_novel=False
        )
        choice, _, _, _ = rt.step(world.actors["c"], view, seed=0)
        return dict(choice.params).get("position", "")

    # Same actor 'c' (same identity/memory), different observed shock -> different action.
    assert choose(0.0) == "hold"
    assert choose(0.6) == "cut"
