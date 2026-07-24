"""Acceptance tests for the rebuilt+universal merge.

These lock the properties that only hold once the live-research/evidence-integrity line
and the universal-simulator line are actually fused: one production path, the coverage
gate assessing the exact WorldSpec that gets simulated, event-driven actor invocation,
persistent actor state, and rejection (never coercion) of invalid intentions.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import pytest

from _helpers import base_corpus, compile_dict
from _worlds import _corpus, _person, run_corpus
from sworldmodel.effects import EffectExecutor
from sworldmodel.executor import ActionExecutor
from sworldmodel.gateway import DeterministicGateway
from sworldmodel.worldspec import ActionChoice

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "sworldmodel"
SUPERSEDED = (
    "compiler.py",
    "universal_compiler.py",
    "runtime.py",
    "mechanisms.py",
    "protocols.py",
    "intents.py",
)


# --- 15/16. one production path; the old committee runtime is unreachable ----


def test_superseded_modules_are_deleted() -> None:
    for name in SUPERSEDED:
        assert not (SRC / name).exists(), f"superseded module {name} still present"


def test_no_superseded_module_is_reachable_from_forecast() -> None:
    import sworldmodel.api  # noqa: F401  (loads the whole production path)

    run_corpus(base_corpus())
    loaded = {m.split(".")[-1] for m in sys.modules if m.startswith("sworldmodel.")}
    assert not loaded & {n[:-3] for n in SUPERSEDED}


def test_exactly_one_compiler_one_engine_one_evaluator() -> None:
    from sworldmodel import api
    from sworldmodel.engine import evaluate_terminal, run
    from sworldmodel.world_compiler import compile_world

    assert api.compile_world is compile_world  # one production compiler
    assert api.run is run  # one production engine
    assert evaluate_terminal.__module__ == "sworldmodel.engine"  # one terminal evaluator


# --- 3/4. verified research feeds, and coverage assesses, the exact WorldSpec -


def test_coverage_gate_assesses_the_exact_worldspec_that_is_simulated() -> None:
    result, ctx = run_corpus(base_corpus())
    spec = ctx.compiled.spec
    report = ctx.compiled.coverage_report
    assert report.is_complete

    # The very spec the gate passed is the one the engine executed.
    assert ctx.run_result.final_worlds
    from sworldmodel.world_compiler import world_spec_view

    view, _ = world_spec_view(
        spec, ctx.compiled.base_world, ctx.bundle.uncertainties, ctx.bundle.world_facts
    )
    object_ids = {o.object_id for o in view.objects}
    for action in spec.actions:
        assert f"action:{action.action_id}" in object_ids
    for node in spec.process.nodes:
        assert f"node:{node.node_id}" in object_ids
    assert "terminal" in object_ids
    assert result.probability_source == "weighted_simulated_trajectories"


def test_evidence_claims_reach_the_compiled_world() -> None:
    _, ctx = run_corpus(base_corpus())
    cited = {cid for e in ctx.compiled.spec.entities for cid in e.evidence_claim_ids}
    available = {c.id for c in ctx.bundle.evidence_store.view(ctx.as_of).available()}
    assert cited and cited <= available  # the world cites only verified, in-cutoff claims


# --- 10. actors are invoked from events, not on a fixed schedule -------------


def _two_actor_world(*, eligible_only_a: bool, ping: bool = False) -> dict[str, Any]:
    """A world with no fields and no uncertainty, so nothing is broadcast at branch
    start: an actor is invoked only if the compiled world actually gives it a trigger."""

    effects: list[dict[str, Any]] = [
        # Private: acting leaves no trace another actor could perceive, so the only way
        # 'b' can be invoked is if the world genuinely reaches it (the ``ping`` case).
        {
            "op": "append_record",
            "collection": "log",
            "key": "$actor",
            "value": "acted",
            "visibility": "private",
        }
    ]
    if ping:
        # The action also reaches the *other* actor, which is a genuine trigger for them.
        effects.append(
            {"op": "deliver_information", "text": "your turn", "to": ["b"], "visibility": "private"}
        )
    spec = {
        "title": "trigger world",
        "subject_entity": "the log",
        "entities": [_person("a", ["act"]), _person("b", ["act"])],
        "actors": [
            {"entity_id": "a", "policy": {"default_action_id": "do_it"}},
            {"entity_id": "b", "policy": {"default_action_id": "do_it"}},
        ],
        "fields": [],
        "actions": [
            {
                "action_id": "do_it",
                "meaning": "act",
                "eligible_actors": ["a"] if eligible_only_a else ["*"],
                "required_authority": ["act"],
                "stages": ["act"],
                "effects": effects,
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "act",
                    "stage": "act",
                    "participants": ["a", "b"],
                    "action_ids": ["do_it"],
                }
            ]
        },
        "terminal": {
            "yes_when": {"op": "greater_or_equal", "args": [{"op": "count", "args": ["log"]}, 1]},
            "description": "YES iff anything was logged",
        },
    }
    return _corpus(spec, actor_ids=["a", "b"], expected_participants=2, target="anything logged")


def test_actor_with_no_trigger_is_never_invoked() -> None:
    # 'b' is a declared participant but is eligible for no action and receives nothing.
    _, ctx = run_corpus(_two_actor_world(eligible_only_a=True))
    invoked = {d.actor_id for d in ctx.run_result.actor_decisions}
    assert "a" in invoked
    assert "b" not in invoked, "an actor nothing affects must not be invoked"


def test_every_invocation_carries_a_concrete_trigger() -> None:
    _, ctx = run_corpus(base_corpus())
    assert ctx.run_result.actor_decisions
    for d in ctx.run_result.actor_decisions:
        assert d.trigger, f"{d.actor_id} was invoked with no recorded trigger"
        assert any(
            t in d.trigger
            for t in ("information", "opportunity", "pending_need", "response_to_completed_action")
        )


def test_a_completed_action_can_reinvoke_another_actor() -> None:
    # 'a' acts; its action delivers information to 'b', which is a trigger for 'b'.
    _, ctx = run_corpus(_two_actor_world(eligible_only_a=True, ping=True))
    by_actor: dict[str, list[str]] = {}
    for d in ctx.run_result.actor_decisions:
        by_actor.setdefault(d.actor_id, []).append(d.trigger)
    assert "b" in by_actor, "the delivered information should have invoked b"
    assert any("information" in t or "response" in t for t in by_actor["b"])


# --- 11. persistent actor state ---------------------------------------------


def test_actors_keep_persistent_memory_and_pending_needs() -> None:
    compiled = compile_dict(base_corpus())
    for actor in compiled.base_world.actors.values():
        assert len(actor.memory) >= 1  # seeded from evidence, never empty
        assert actor.active_plan
        assert isinstance(actor.pending_questions, tuple)  # the field persists on the actor

    _, ctx = run_corpus(base_corpus())
    world = next(iter(ctx.run_result.final_worlds.values()))
    for actor in world.actors.values():
        # memory grew as the actor perceived the world, and observations are remembered
        assert len(actor.memory) >= 1
        assert actor.last_observed_event_ids or not world.event_history


# --- 12. invalid intentions are rejected, never semantically coerced ---------


def test_out_of_stage_action_is_rejected_not_rewritten() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("brief")  # the action belongs to stage 'decide'
    executor = ActionExecutor(DeterministicGateway(), EffectExecutor())
    outcome = executor.execute(
        world.actors["a"],
        ActionChoice(
            mode="compiled_action", action_id="record_position", params=(("position", "hold"),)
        ),
        world,
        compiled.spec,
        0,
    )
    assert outcome.status == "rejected"
    after = world.apply(outcome.events)
    assert not after.get_records("votes")  # nothing recorded
    # and nothing else was substituted in its place
    assert all(e.kind == "action_rejected" for e in outcome.events)


def test_out_of_choice_parameter_is_rejected() -> None:
    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("decide")
    executor = ActionExecutor(DeterministicGateway(), EffectExecutor())
    outcome = executor.execute(
        world.actors["a"],
        ActionChoice(
            mode="compiled_action", action_id="record_position", params=(("position", "banana"),)
        ),
        world,
        compiled.spec,
        0,
    )
    assert outcome.status == "rejected"
    assert "banana" in outcome.reason
    assert not world.apply(outcome.events).get_records("votes")


# --- 1/2. question-only entry, real provider for production ------------------


def test_live_config_requires_a_real_provider_and_live_research() -> None:
    from sworldmodel import DeterministicGateway as DG
    from sworldmodel import ForecastConfig
    from sworldmodel.deepseek_gateway import DeepSeekGateway
    from sworldmodel.http import FakeTransport
    from sworldmodel.live_research import LiveResearchBackend

    real = DeepSeekGateway(FakeTransport(), api_key="k")
    live_backend = LiveResearchBackend(real, FakeTransport())
    assert ForecastConfig(gateway=real, research_backend=live_backend).is_live
    # A deterministic (test-only) gateway can never satisfy the live gate.
    assert not ForecastConfig(gateway=DG(), research_backend=live_backend).is_live


def test_forecast_signature_is_question_cutoff_horizon_only() -> None:
    import inspect

    from sworldmodel import forecast

    params = list(inspect.signature(forecast).parameters)
    assert params == ["question", "as_of", "horizon", "config"]
    # No corpus argument anywhere on the public entry point.
    assert not any("corpus" in p for p in params)


@pytest.mark.parametrize("name", ["compile_world", "run"])
def test_api_module_exposes_one_of_each_stage(name: str) -> None:
    from sworldmodel import api

    assert hasattr(api, name)
