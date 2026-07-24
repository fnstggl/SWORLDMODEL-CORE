"""Interaction invariants: authority, stage/timing, preconditions, targets, resources."""

from __future__ import annotations

from dataclasses import replace

from _helpers import base_corpus, compile_dict
from _worlds import negotiation_world
from sworldmodel.effects import EffectExecutor
from sworldmodel.executor import ActionExecutor
from sworldmodel.gateway import DeterministicGateway
from sworldmodel.worldspec import ActionChoice, ActionDefinition, Effect, parse_expr


def _exec() -> ActionExecutor:
    return ActionExecutor(DeterministicGateway(), EffectExecutor())


def test_action_is_gated_by_its_stage() -> None:
    compiled = compile_dict(base_corpus())
    action = compiled.spec.action("record_position")
    world_brief = compiled.base_world.with_stage("brief")
    world_decide = compiled.base_world.with_stage("decide")
    ex = _exec()
    assert not ex.is_feasible(world_brief, world_brief.actors["a"], action)
    assert ex.is_feasible(world_decide, world_decide.actors["a"], action)


def test_precondition_gates_the_action() -> None:
    compiled = compile_dict(negotiation_world())
    sign = compiled.spec.action("sign_deal")
    world = compiled.base_world.with_stage("bargain")  # gap starts at 20 -> not signable
    ex = _exec()
    assert not ex.is_feasible(world, world.actors["buyer"], sign)
    closed = replace(world, fields=(("gap", -2.0), ("hostility", 0.0)))
    assert ex.is_feasible(closed, closed.actors["buyer"], sign)


def test_resource_cost_gates_feasibility_and_execution() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("act")
    costly = ActionDefinition(
        action_id="spend",
        meaning="spend a token",
        eligible_actors=("*",),
        required_authority=("decide",),
        stages=("act",),
        resource_costs=(("token", 1.0),),
        effects=(Effect(op="append_record", params=(("collection", "log"), ("value", "spent"))),),
    )
    spec = replace(compiled.spec, actions=compiled.spec.actions + (costly,))
    ex = _exec()
    # No tokens -> infeasible, and execution is rejected.
    assert not ex.is_feasible(world, world.actors["a"], costly)
    outcome = ex.execute(
        world.actors["a"], ActionChoice(mode="compiled_action", action_id="spend"), world, spec, 0
    )
    assert outcome.status == "rejected"
    # With a token -> feasible and consumed.
    funded = replace(world, resources=(("token@a", 1.0),))
    assert ex.is_feasible(funded, funded.actors["a"], costly)
    done = ex.execute(
        funded.actors["a"], ActionChoice(mode="compiled_action", action_id="spend"), funded, spec, 0
    )
    assert done.status == "executed"
    assert funded.apply(done.events).get_resource("token", "a") == 0.0


def test_target_validation_rejects_an_invalid_target() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("act")
    messaging = ActionDefinition(
        action_id="ping",
        meaning="ping a colleague",
        eligible_actors=("*",),
        required_authority=("decide",),
        stages=("act",),
        valid_targets=("b", "c"),
        effects=(Effect(op="deliver_information", params=(("text", "ping"), ("to", ["$target"]))),),
    )
    spec = replace(compiled.spec, actions=compiled.spec.actions + (messaging,))
    ex = _exec()
    bad = ex.execute(
        world.actors["a"],
        ActionChoice(mode="compiled_action", action_id="ping", target="zzz"),
        world,
        spec,
        0,
    )
    assert bad.status == "rejected" and "target" in bad.reason
    good = ex.execute(
        world.actors["a"],
        ActionChoice(mode="compiled_action", action_id="ping", target="b"),
        world,
        spec,
        0,
    )
    assert good.status == "executed"


def test_out_of_stage_action_is_rejected_on_execution() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("brief")  # decision stage is 'decide'
    outcome = _exec().execute(
        world.actors["a"],
        ActionChoice(
            mode="compiled_action", action_id="record_position", params=(("position", "hold"),)
        ),
        world,
        compiled.spec,
        0,
    )
    assert outcome.status == "rejected"


def test_precondition_expr_parses_from_json() -> None:
    expr = parse_expr({"op": "less_or_equal", "args": [{"field": "gap"}, 0]})
    assert expr.op == "less_or_equal"
