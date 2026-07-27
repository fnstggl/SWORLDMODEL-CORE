"""A novel action states an intention. It does not get to be the world.

The decision prompt promises every actor: *"A novel action does NOT automatically
happen: the world decides. You state an intention only; you may never assert a
consequence."* The runtime did not hold that line, and the hole was exact.

``_blocked_by_world_authority`` looked for *compiled actions* producing the same effect
and, finding none, wrote ``continue``. So whenever a terminal term's only producer was a
process node — a scheduled release, a quarterly run, the weather — the gatekeeper list
came back empty, the check returned "not blocked", and the only thing left between an
invented action and the answer was the interpreter model's self-declared
``required_authority``, which can be ``[]``. An actor could invent an action that sets
the outcome, and the sole objection was a model declaring that its own invention needed
no standing.

Two defects, and the second is the general one:

1. **An effect only the environment produces is not an effect any actor has standing to
   perform.** Rain is not an action. A scheduled data release is not an action. A
   quarterly production run is not an action. Process-node effects and external-process
   occurrences are therefore producers that confer no standing at all.
2. **An empty producer set read as "nobody objects" when it means "nobody has
   standing."** That is the fourth appearance this week of a check whose *found nothing*
   is indistinguishable from *did not look* — FD-34 (a provider outage laundering
   mechanical findings), FD-42 (blank records evaluating to an answer), FD-45 (a crashed
   check reading as an approved one). The three answers are now named and distinct:
   ``STANDING_ACTOR``, ``STANDING_ENVIRONMENT``, ``STANDING_UNPRODUCED``.

The opposite failure is guarded just as hard. The novel route is this system's only
escape from pre-enumeration, so a gate that refuses correct worlds costs real agency: an
actor writing a field a compiled action already lets it write must still be permitted,
and inventing something genuinely new — a term nothing in the world produces and the
terminal does not read — must still be permitted.

These tests drive the real ``resolve_novel`` with the real ``EffectExecutor``: nothing
below stubs out the thing under test.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from _fakes import ProgrammableGateway
from sworldmodel.actors import ActorState
from sworldmodel.effects import EffectExecutor
from sworldmodel.evidence import EvidenceStore
from sworldmodel.models import ResolutionContract
from sworldmodel.novel import (
    STANDING_ACTOR,
    STANDING_ENVIRONMENT,
    STANDING_UNPRODUCED,
    resolve_novel,
    terminal_terms,
    world_standing,
)
from sworldmodel.world import WorldState
from sworldmodel.world_compiler import build_base_world
from sworldmodel.worldspec import (
    ActionChoice,
    Effect,
    WorldSpec,
    effect_terms,
    parse_effect,
    parse_world_spec,
)

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


# ---------------------------------------------------------------------------
# A world whose terminal term is written only by a process node.
# ---------------------------------------------------------------------------


def _world_dict() -> dict[str, Any]:
    """A refinery quarter. The operator can publish and can report; the *run itself* is
    the only thing that meets the target, and the terminal reads exactly that."""

    return {
        "title": "a production quarter",
        "subject_entity": "the refinery",
        "resolution_units": "target met",
        "entities": [
            {
                "entity_id": "operator",
                "name": "The Operator",
                "kind": "organization",
                "is_actor": True,
                "role": "operator",
                "authority": ["publish_statement", "report_output"],
            },
            {
                "entity_id": "observer",
                "name": "A Trade Observer",
                "kind": "organization",
                "is_actor": True,
                "role": "observer",
                "authority": [],
            },
        ],
        "actors": [
            {"entity_id": "operator", "reasoning": "runs the plant"},
            {"entity_id": "observer", "reasoning": "watches, and has no standing"},
        ],
        "fields": [
            {"field_id": "production_target_met", "value_type": "bool", "initial": False},
            {"field_id": "reported_output", "value_type": "number", "initial": 0},
            {"field_id": "published_index", "value_type": "number", "initial": 0},
        ],
        "actions": [
            {
                "action_id": "publish_statement",
                "meaning": "say something publicly",
                "eligible_actors": ["role:operator"],
                "required_authority": ["publish_statement"],
                "effects": [{"op": "create_event", "event_type": "statement_published"}],
            },
            {
                "action_id": "report_output",
                "meaning": "report the quarter's output",
                "eligible_actors": ["role:operator"],
                "required_authority": ["report_output"],
                "effects": [{"op": "set_field", "field": "reported_output", "value": 0}],
            },
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "quarterly_run",
                    "description": "the plant runs the quarter and meets its target or does not",
                    "stage": "quarter",
                    "at": "2026-06-10T00:00:00+00:00",
                    "participants": ["*"],
                    "effects": [
                        {"op": "set_field", "field": "production_target_met", "value": True}
                    ],
                }
            ]
        },
        "external_processes": [
            {
                "process_id": "index_publication",
                "description": "the scheduled index release",
                "occurrences": [
                    {
                        "at": "2026-06-01T00:00:00+00:00",
                        "description": "index published",
                        "effects": [{"op": "release_data", "fields": {"published_index": 7.5}}],
                    }
                ],
            }
        ],
        "terminal": {
            "yes_when": {
                "op": "equals",
                "args": [{"op": "field", "args": ["production_target_met"]}, True],
            }
        },
    }


def _spec(mutate: Any = None) -> WorldSpec:
    data = _world_dict()
    if mutate is not None:
        mutate(data)
    return parse_world_spec(data)


def _world(spec: WorldSpec) -> WorldState:
    contract = ResolutionContract(
        question="does the quarter meet its target?",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=spec.subject_entity,
        resolution_units=spec.resolution_units,
        terminal=spec.terminal,
    )
    return build_base_world(spec, contract, EvidenceStore().view(AS_OF), ())


def _actor(world: WorldState, actor_id: str = "operator") -> ActorState:
    return world.actors[actor_id]


def _gateway(effects: list[dict[str, Any]], declared: list[str] | None = None):
    """An interpreter that maps the intention onto exactly these effects, and declares
    exactly this authority. ``declared=[]`` is the interesting case: the model saying its
    own invention needs no standing at all."""

    return ProgrammableGateway(
        {
            "interpret_novel": {
                "representable": True,
                "reason": "authored mapping",
                "required_authority": list(declared or []),
                "effects": effects,
            }
        }
    )


def _propose(
    effects: list[dict[str, Any]],
    *,
    actor_id: str = "operator",
    declared: list[str] | None = None,
    mutate: Any = None,
):
    spec = _spec(mutate)
    world = _world(spec)
    choice = ActionChoice(
        mode="novel_action",
        novel_description="a thing the compiler did not anticipate",
        novel_intended_effect="the described consequence",
    )
    resolution, _produced = resolve_novel(
        actor=_actor(world, actor_id),
        choice=choice,
        world=world,
        spec=spec,
        gateway=_gateway(effects, declared),
        executor=EffectExecutor(),
        seed=0,
    )
    return resolution


# ---------------------------------------------------------------------------
# 1. The hole itself: a process-produced terminal term, written by an invented act.
# ---------------------------------------------------------------------------


def test_a_novel_action_cannot_write_a_terminal_term_only_a_process_produces() -> None:
    """The reproducer. Before the fix this was PERMITTED and the branch resolved YES.

    Nothing but ``quarterly_run`` sets ``production_target_met``, and the terminal reads
    it. The operator invents "declare the target met" and the interpreter — the same
    model that proposed it — declares that this needs no authority whatsoever.
    """

    resolution = _propose(
        [{"op": "set_field", "field": "production_target_met", "value": True}],
        declared=[],
    )

    assert not resolution.executed, "a novel action wrote the answer"
    assert "only the environment produces" in resolution.reason
    # The refusal names *what* and *who*, so the actor is told something it can act on.
    assert "production_target_met" in resolution.reason
    assert "process_node:quarterly_run" in resolution.reason

    (standing,) = resolution.standing
    assert standing.term == "field:production_target_met"
    assert standing.standing == STANDING_ENVIRONMENT
    assert standing.actor_producers == ()
    assert standing.environment_producers == ("process_node:quarterly_run",)
    assert standing.decides_terminal


def test_a_scheduled_release_is_not_an_action_either() -> None:
    """The external-process half. ``release_data`` writes world fields on the calendar;
    an actor asserting the release has happened is asserting a consequence."""

    resolution = _propose(
        [{"op": "set_field", "field": "published_index", "value": 99}],
        declared=[],
    )

    assert not resolution.executed
    assert "only the environment produces" in resolution.reason
    assert "external_process:index_publication#0" in resolution.reason
    assert resolution.standing[0].standing == STANDING_ENVIRONMENT


def test_the_refusal_survives_renaming_the_operation() -> None:
    """``adjust_field`` and ``set_field`` reach the same field.

    Standing is a property of the state written, not of the verb used to write it, so
    swapping the op must not open a second door to the same term.
    """

    resolution = _propose(
        [{"op": "adjust_field", "field": "production_target_met", "delta": 1}],
        declared=[],
    )
    assert not resolution.executed
    assert "only the environment produces" in resolution.reason


def test_the_refusal_survives_writing_the_field_as_a_data_release() -> None:
    """``release_data`` writes every key of its ``fields`` map straight into world state.

    It is the one op that writes fields without naming one in ``field``, so a check that
    only knows ``set_field``/``adjust_field`` has a door standing open beside the one it
    is guarding — and this is the shape the environment itself uses.
    """

    resolution = _propose(
        [{"op": "release_data", "fields": {"production_target_met": True}}],
        declared=[],
    )
    assert not resolution.executed
    assert "only the environment produces" in resolution.reason
    assert resolution.standing[0].term == "field:production_target_met"


def test_the_refusal_survives_spelling_the_parameter_the_other_way() -> None:
    """A model writes ``field_id`` where the schema says ``field`` — the exact synonym
    the compiled path already normalizes (a live Bank of England run emitted it).

    The gate and the executor must read one object: if the check normalizes and the
    executor does not, the check guards an effect nobody runs; if the executor
    normalizes and the check does not, the check is blind to the effect that runs.
    """

    resolution = _propose(
        [{"op": "set_field", "field_id": "production_target_met", "value": True}],
        declared=[],
    )
    assert not resolution.executed
    assert "only the environment produces" in resolution.reason
    assert resolution.standing[0].term == "field:production_target_met"


def test_a_descriptive_stage_parameter_is_not_read_as_writing_the_stage() -> None:
    """No universal op moves the stage — the engine does, when a process node fires. An
    effect that merely says which stage it belongs to must not be refused as though it
    had tried to move the world's, which is a false positive on ordinary metadata."""

    resolution = _propose([{"op": "create_event", "event_type": "a_new_thing", "stage": "quarter"}])
    assert resolution.executed, resolution.reason
    assert "stage:" not in {s.term for s in resolution.standing}


def test_one_permitted_effect_does_not_carry_a_refused_one_in_with_it() -> None:
    """A proposal is refused whole. Bundling the gated write with an innocuous one is
    the oldest way past a check that stops at the first thing it likes."""

    resolution = _propose(
        [
            {"op": "create_event", "event_type": "a_brand_new_kind_of_happening"},
            {"op": "set_field", "field": "production_target_met", "value": True},
        ],
        declared=[],
    )
    assert not resolution.executed
    assert "only the environment produces" in resolution.reason
    assert {s.standing for s in resolution.standing} == {
        STANDING_UNPRODUCED,
        STANDING_ENVIRONMENT,
    }


# ---------------------------------------------------------------------------
# 2. The legitimate case survives. A gate that refuses correct worlds is worse than
#    the hole it closes, and this is the system's only escape from pre-enumeration.
# ---------------------------------------------------------------------------


def test_a_novel_action_writing_what_a_compiled_action_writes_is_permitted() -> None:
    """The operator already has ``report_output``, which writes ``reported_output``.

    Reaching the same field by an invented route is not an escalation — it is the same
    standing, exercised differently, and refusing it would make the novel path useless.
    """

    resolution = _propose([{"op": "set_field", "field": "reported_output", "value": 412}])

    assert resolution.executed, resolution.reason
    (standing,) = resolution.standing
    assert standing.standing == STANDING_ACTOR
    assert standing.satisfied_by == "report_output"
    # The permitted resolution records the standing it acted on, not merely that it ran.
    assert resolution.required_authority == ("report_output",)


def test_a_novel_action_reaching_new_ground_is_permitted() -> None:
    """Nothing in this world creates a ``reschedule_requested`` event and the terminal
    does not read one. That is exactly what the novel route exists for."""

    resolution = _propose(
        [{"op": "create_event", "event_type": "reschedule_requested", "text": "move the run"}]
    )

    assert resolution.executed, resolution.reason
    (standing,) = resolution.standing
    assert standing.standing == STANDING_UNPRODUCED
    assert not standing.decides_terminal


def test_communicating_is_still_permitted() -> None:
    """``deliver_information`` writes no world state the terminal can read. Refusing it
    would silence the actors in the name of authority."""

    resolution = _propose(
        [{"op": "deliver_information", "to": ["observer"], "text": "the run is behind"}]
    )
    assert resolution.executed, resolution.reason
    assert resolution.standing[0].term == "op:deliver_information"


def test_an_unproduced_term_that_decides_the_answer_is_still_refused() -> None:
    """The completion of the principle. When nothing at all produces the term the
    terminal reads, "invent an action" and "write the answer" are the same move — so the
    permissive branch stops exactly at the outcome and nowhere short of it."""

    def no_producer(data: dict[str, Any]) -> None:
        data["process"]["nodes"][0]["effects"] = []

    resolution = _propose(
        [{"op": "set_field", "field": "production_target_met", "value": True}],
        declared=[],
        mutate=no_producer,
    )
    assert not resolution.executed
    assert "nothing in this compiled world produces" in resolution.reason
    assert resolution.standing[0].standing == STANDING_UNPRODUCED


# ---------------------------------------------------------------------------
# 3. A self-declared `required_authority` may tighten. It may never unlock.
# ---------------------------------------------------------------------------


def test_a_declared_empty_authority_does_not_unlock_a_gated_effect() -> None:
    """The observer holds nothing and is eligible for no compiled action. It proposes
    the effect ``report_output`` gates, and the interpreter declares ``[]``."""

    resolution = _propose(
        [{"op": "set_field", "field": "reported_output", "value": 412}],
        actor_id="observer",
        declared=[],
    )

    assert not resolution.executed
    assert "requires standing" in resolution.reason
    (standing,) = resolution.standing
    assert standing.standing == STANDING_ACTOR
    assert standing.satisfied_by == ""
    assert standing.actor_producers == ("report_output",)
    # The requirement on the record is the WORLD's, not the model's empty claim.
    assert resolution.required_authority == ("report_output",)


def test_a_declared_empty_authority_does_not_unlock_an_environment_effect() -> None:
    resolution = _propose(
        [{"op": "set_field", "field": "production_target_met", "value": True}],
        actor_id="observer",
        declared=[],
    )
    assert not resolution.executed
    assert "only the environment produces" in resolution.reason


@pytest.mark.parametrize("declared", [[], ["something_irrelevant"]])
def test_no_declaration_can_talk_the_world_out_of_its_verdict(declared: list[str]) -> None:
    """Whatever the interpreter says — nothing, or a token beside the point — the
    process-produced term stays shut."""

    resolution = _propose(
        [{"op": "set_field", "field": "production_target_met", "value": True}],
        declared=declared,
    )
    assert not resolution.executed


def test_a_declaration_can_only_add_a_requirement() -> None:
    """The operator may legitimately write ``reported_output``; the interpreter says the
    act also needs a token the operator does not hold. The stricter answer wins."""

    permitted = _propose([{"op": "set_field", "field": "reported_output", "value": 412}])
    assert permitted.executed

    tightened = _propose(
        [{"op": "set_field", "field": "reported_output", "value": 412}],
        declared=["sign_the_annual_return"],
    )
    assert not tightened.executed
    assert "lacks authority" in tightened.reason
    assert "sign_the_annual_return" in tightened.reason
    # The declared token joins the world's own requirement rather than replacing it.
    assert tightened.required_authority == ("report_output", "sign_the_annual_return")


# ---------------------------------------------------------------------------
# 4. "Found nothing" and "did not look" are now different answers.
# ---------------------------------------------------------------------------


def test_every_effect_names_at_least_one_term() -> None:
    """An empty reach would be a silence meaning two things at once: an effect that
    writes nothing nameable, and an effect this vocabulary failed to classify."""

    from sworldmodel.effects import UNIVERSAL_OPS

    for op in sorted(UNIVERSAL_OPS):
        assert effect_terms(Effect(op=op)), f"{op} named nothing at all"
        assert effect_terms(parse_effect({"op": op, "field": "x", "collection": "x"}))


def test_every_proposed_term_is_classified_into_exactly_one_named_bucket() -> None:
    """The structural claim: ``world_standing`` returns a verdict per term, every
    verdict is one of the three named answers, and none of them is silence."""

    spec = _spec()
    actor = _actor(_world(spec))
    effects = (
        Effect(op="set_field", params=(("field", "production_target_met"), ("value", True))),
        Effect(op="set_field", params=(("field", "reported_output"), ("value", 1))),
        Effect(op="create_event", params=(("event_type", "something_new"),)),
    )
    standings = world_standing(spec, effects, actor)

    assert [s.term for s in standings] == [
        "field:production_target_met",
        "field:reported_output",
        "event:something_new",
    ]
    assert [s.standing for s in standings] == [
        STANDING_ENVIRONMENT,
        STANDING_ACTOR,
        STANDING_UNPRODUCED,
    ]
    assert all(s.standing for s in standings)


def test_an_effect_the_world_gates_is_not_confused_with_one_it_has_never_heard_of() -> None:
    """The exact distinction the old check could not make. Both came back as an empty
    gatekeeper list; both were permitted. They are now different verdicts with different
    outcomes."""

    spec = _spec()
    actor = _actor(_world(spec), "observer")

    (environment,) = world_standing(
        spec,
        (Effect(op="set_field", params=(("field", "production_target_met"), ("value", True))),),
        actor,
    )
    (unheard_of,) = world_standing(
        spec, (Effect(op="create_event", params=(("event_type", "a_new_thing"),)),), actor
    )

    assert environment.standing != unheard_of.standing
    assert environment.refusal() and not unheard_of.refusal()


def test_the_gate_fires_inside_a_real_run() -> None:
    """The same refusal, reached the way the runtime reaches it.

    ``scheduled_multiparty_world``'s ``external_signal`` is written by a scheduled
    ``release_data`` and by nothing else — the paradigm case the audit names, "a
    scheduled data release is not an action". A member invents "publish the measurement
    myself", the interpreter declares it needs no authority, and the world says no.
    """

    from _fakes import act, build_bundle, propose, wait_decision
    from _worlds import scheduled_multiparty_world
    from sworldmodel.engine import run
    from sworldmodel.world_compiler import compile_world

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "preparation" and ctx["actor_id"] == "member_0":
            return propose(
                "publish the measurement myself",
                [{"op": "set_field", "field": "external_signal", "value": 99}],
            )
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gateway = ProgrammableGateway(
        {
            "actor_decision": decide,
            "reflect": {"beliefs_update": [], "new_memories": []},
            # The interpreter maps it faithfully and declares no authority at all.
            "interpret_novel": lambda ctx: {
                "representable": True,
                "reason": "a field write",
                "required_authority": [],
                "effects": [{"op": "set_field", "field": "external_signal", "value": 99}],
            },
        }
    )

    bundle = build_bundle(scheduled_multiparty_world())
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
    compiled = compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=4,
    )
    result = run(compiled, gateway, seed=0)

    refused = [
        d
        for d in result.actor_decisions
        if d.validation_status == "rejected" and d.intent["mode"] == "novel_action"
    ]
    assert refused, "the invented release was permitted inside a real run"
    assert "only the environment produces" in refused[0].validation_reason
    for world in result.final_worlds.values():
        assert world.get_field("external_signal") != 99


def test_the_terminal_terms_include_both_legs() -> None:
    """``unresolved_when`` decides the answer exactly as much as ``yes_when``: writing it
    turns a determined branch into an undetermined one."""

    def add_unresolved(data: dict[str, Any]) -> None:
        data["terminal"]["unresolved_when"] = {
            "op": "greater_than",
            "args": [{"op": "count", "args": ["formal_challenges"]}, 0],
        }

    spec = _spec(add_unresolved)
    assert terminal_terms(spec) == frozenset(
        {"field:production_target_met", "collection:formal_challenges"}
    )
