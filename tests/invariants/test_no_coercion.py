"""The environment decides what is possible. The actor decides what it wants.

Neither may do the other's job. These tests exist because the failure they guard
against is invisible in a finished run: a trace can show an actor "making a statement"
when what it actually did was try to vote early, and nothing in the output says so.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from _fakes import ProgrammableGateway, act, build_bundle, propose, wait_decision
from _worlds import scheduled_multiparty_world, single_response_world
from sworldmodel.engine import run
from sworldmodel.errors import GatewayError
from sworldmodel.models import ResolutionContract
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def _compile(data: dict, gateway: ProgrammableGateway):
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="test question",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=4,
    )


def _gateway(decision) -> ProgrammableGateway:
    return ProgrammableGateway(
        {
            "actor_decision": decision,
            "reflect": {"beliefs_update": [], "new_memories": []},
            "interpret_novel": lambda ctx: {
                "representable": True,
                "reason": "authored mapping",
                "required_authority": list(
                    (ctx.get("parameters") or {}).get("required_authority", []) or []
                ),
                "effects": (ctx.get("parameters") or {}).get("effects", []),
            },
        }
    )


# ---------------------------------------------------------------------------
# A premature act is refused. It does not become a different act.
# ---------------------------------------------------------------------------


def test_a_premature_action_is_rejected_and_nothing_is_recorded() -> None:
    """The canonical failure: acting before the process opens that act.

    The action must be refused, the world must record no position, and — critically —
    no *other* action may appear in its place.
    """

    data = scheduled_multiparty_world()
    attempted = {"count": 0}

    def decide(ctx: dict) -> dict:
        if ctx["stage"] == "preparation":
            attempted["count"] += 1
            return act("record_position", {"position": "hold"})  # not open yet
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    assert attempted["count"] > 0
    rejected = [d for d in result.actor_decisions if d.validation_status == "rejected"]
    assert rejected, "the premature action was not rejected"
    assert "not available in stage" in rejected[0].validation_reason

    # Nothing was recorded, and no substitute action was invented.
    assert not world.get_records("positions")
    for d in result.actor_decisions:
        if d.stage == "preparation":
            assert d.intent["action_id"] in ("record_position", "")
            assert d.intent["mode"] in ("compiled_action", "wait")


def test_the_rejection_reason_comes_back_to_the_actor_who_was_refused() -> None:
    """A refusal is information. The actor is told, and may choose something else."""

    data = scheduled_multiparty_world()
    seen_rejections: list[str] = []

    def decide(ctx: dict) -> dict:
        for obs in ctx.get("observations", []):
            if obs["kind"] == "action_rejected":
                seen_rejections.append(obs["summary"])
                return act("circulate_note", {"text": "I will raise it at the session instead"})
        if ctx["stage"] == "preparation":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    assert seen_rejections, "the actor was never told its attempt was refused"
    # The replacement action is the actor's own second decision, recorded separately.
    notes = [d for d in result.actor_decisions if d.intent["action_id"] == "circulate_note"]
    assert notes


def test_a_missing_parameter_is_refused_rather_than_chosen_for_the_actor() -> None:
    """The runtime must never pick an actor's option.

    An action whose parameter decides the substance of the act — which way you go, how
    much you offer, what you answer — is meaningless without it. Filling it in with
    "the first declared choice" would be the simulator casting the actor's decision.
    """

    data = scheduled_multiparty_world()

    def decide(ctx: dict) -> dict:
        if ctx["stage"] == "session":
            return act("record_position", {})  # no position stated
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    rejected = [d for d in result.actor_decisions if d.validation_status == "rejected"]
    assert rejected
    assert "missing required parameter" in rejected[0].validation_reason
    assert not world.get_records("positions"), "a position was recorded that nobody stated"


def test_an_unusable_provider_response_fails_rather_than_becoming_a_wait() -> None:
    """A response the runtime cannot read is a provider failure, not a decision to
    wait. Turning it into a wait would silently insert behavior nobody chose."""

    data = single_response_world()
    gw = ProgrammableGateway(
        {"actor_decision": {"reasoning": "no action_mode at all"}, "reflect": {}}
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    assert result.branch_outcomes
    for b in result.branch_outcomes:
        assert not b.resolved
        assert "provider_failure" in (b.unresolved_reason or "")


def test_an_unnamed_compiled_action_is_not_guessed() -> None:
    data = single_response_world()
    gw = ProgrammableGateway(
        {
            "actor_decision": {
                "plan_disposition": "continue",
                "action_mode": "compiled_action",
                "compiled_action_id": "",
            },
            "reflect": {},
        }
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    assert all(not b.resolved for b in result.branch_outcomes)
    assert "without naming one" in (result.branch_outcomes[0].unresolved_reason or "")


def test_an_unknown_plan_disposition_is_a_failure_not_a_default() -> None:
    from sworldmodel.actors import ActorRuntime

    runtime = ActorRuntime(ProgrammableGateway())
    with pytest.raises(GatewayError):
        runtime._disposition({"plan_disposition": "vibes"})


# ---------------------------------------------------------------------------
# Novel actions: authorized ones run, unauthorized ones do not, and neither is
# quietly mapped onto something that already exists.
# ---------------------------------------------------------------------------


def test_an_authorized_novel_action_executes() -> None:
    data = scheduled_multiparty_world()

    def decide(ctx: dict) -> dict:
        if ctx["stage"] == "preparation" and ctx["actor_id"] == "member_0":
            return propose(
                "request that the session be moved a week later",
                [
                    {
                        "op": "create_event",
                        "event_type": "reschedule_requested",
                        "text": "member_0 asks to move the session",
                    }
                ],
            )
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    assert any(
        e.payload_dict.get("event_type") == "reschedule_requested" for e in world.event_history
    )


def test_an_unauthorized_novel_action_is_rejected() -> None:
    data = scheduled_multiparty_world()

    def decide(ctx: dict) -> dict:
        if ctx["stage"] == "preparation" and ctx["actor_id"] == "member_0":
            return propose(
                "unilaterally set the outcome",
                [{"op": "set_field", "field": "external_signal", "value": 99}],
                **{},
            ) | {
                "novel_action": {
                    "description": "unilaterally set the outcome",
                    "target": "",
                    "parameters": {
                        "effects": [{"op": "set_field", "field": "external_signal", "value": 99}],
                        "required_authority": ["set_outcome_unilaterally"],
                    },
                    "intended_effect": "decide alone",
                }
            }
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    rejected = [
        d
        for d in result.actor_decisions
        if d.validation_status == "rejected" and d.intent["mode"] == "novel_action"
    ]
    assert rejected
    assert "lacks authority" in rejected[0].validation_reason
    assert world.get_field("external_signal") != 99


def test_an_unrepresentable_novel_action_is_refused_not_approximated() -> None:
    data = scheduled_multiparty_world()
    gw = ProgrammableGateway(
        {
            "actor_decision": lambda ctx: (
                propose("do something the world cannot express", [])
                if ctx["stage"] == "preparation"
                else wait_decision()
            ),
            "reflect": {"beliefs_update": [], "new_memories": []},
            "interpret_novel": {
                "representable": False,
                "reason": "no safe universal representation",
            },
        }
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    rejected = [d for d in result.actor_decisions if d.validation_status == "rejected"]
    assert rejected
    assert "unrepresentable" in rejected[0].validation_reason
    # And nothing else happened in its place.
    assert all(d.intent["mode"] != "compiled_action" for d in rejected)


# ---------------------------------------------------------------------------
# Acting is not succeeding
# ---------------------------------------------------------------------------


def test_an_action_whose_completion_condition_fails_does_not_take_effect() -> None:
    """Started, then overtaken by events, then failed — visibly."""

    data = single_response_world()
    data["world_spec"]["fields"].append(
        {"field_id": "channel_open", "value_type": "bool", "initial": True}
    )
    data["world_spec"]["actions"][0]["duration_seconds"] = 86400 * 3
    data["world_spec"]["actions"][0]["completion_conditions"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["channel_open"]}, True],
    }
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "channel_closes",
            "description": "the channel closes while the reply is being written",
            "occurrences": [
                {
                    "at": "2026-05-21T00:00:00+00:00",
                    "description": "channel closed",
                    "effects": [{"op": "set_field", "field": "channel_open", "value": False}],
                }
            ],
        }
    ]

    def decide(ctx: dict) -> dict:
        if ctx["current_action"] is None and not ctx.get("observations"):
            return act("send_reply", {"answer": "yes"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    assert not world.get_records("replies"), "an action succeeded in a world that had moved on"
    assert any(e.kind == "action_failed" for e in world.event_history)
    assert world.get_field("reply_sent") is not True
