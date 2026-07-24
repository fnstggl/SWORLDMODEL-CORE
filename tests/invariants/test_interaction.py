"""Interaction invariants: real statements, delivered and reacted to."""

from __future__ import annotations

from _helpers import base_corpus, compile_dict, run_dict
from sworldmodel.actors import ActorRuntime
from sworldmodel.gateway import DeterministicGateway
from sworldmodel.intents import Environment
from sworldmodel.models import EventKind, Visibility


def _bare_decision(world):
    """Open the decision with NO proposal on the table, isolating statement effects."""

    env = Environment()
    ev = env.environment_event(world, kind=EventKind.DECISION_OPENED, payload={}, time=world.time)
    return world.apply([ev]), env


def _deliver_statement(world, env, speaker: str, info_signals: dict[str, float]):
    ev = env.environment_event(
        world,
        kind=EventKind.STATEMENT_MADE,
        actor_id=speaker,
        payload={
            "statement_text": "Colleagues, the incoming data shows a decisive shock.",
            "favored_option": "cut",
            "info_signals": info_signals,
        },
        time=world.time,
        visibility=Visibility.PUBLIC,
    )
    return world.apply([ev]), ev


def _vote(world, actor_id: str):
    rt = ActorRuntime(DeterministicGateway())
    intent, _, _, _ = rt.step(world.actors[actor_id], world.view_for(actor_id), seed=0)
    return intent


def test_substantive_statement_is_delivered_with_full_content() -> None:
    _, ctx, _ = run_dict(base_corpus())
    statements = [e for e in ctx.run_result.event_ledger if e.kind == EventKind.STATEMENT_MADE]
    assert statements, "the protocol produced no statements"
    assert all(e.payload_dict.get("statement_text") for e in statements)


def test_recipient_can_react_to_a_statement_and_removing_it_changes_the_vote() -> None:
    compiled = compile_dict(base_corpus())

    # With a statement carrying a decisive shock, the recipient reacts (moves to cut).
    world, env = _bare_decision(compiled.base_world)
    world_with, stmt = _deliver_statement(world, env, "b", {"shock": 0.6})
    intent_with = _vote(world_with, "a")

    # Without the statement, the same actor keeps its hold position.
    world_without, _ = _bare_decision(compiled.base_world)
    intent_without = _vote(world_without, "a")

    assert intent_with.payload_dict.get("option") != intent_without.payload_dict.get("option")
    # The reacting actor actually referenced the delivered statement as an observation.
    assert stmt.event_id in intent_with.referenced_observation_ids


def test_deliberation_is_not_scalar_averaging() -> None:
    result, _, _ = run_dict(base_corpus())
    options = set(result.contract.outcome_space)
    for b in result.branch_outcomes:
        for _, option in b.votes:
            # Every vote is a discrete option, never a float position on a scale.
            assert option in options
