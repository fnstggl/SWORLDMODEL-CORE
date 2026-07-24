"""The event-driven scheduler: actor calls are caused, never scheduled.

These are the load-bearing tests for the runtime's central claim — that how often an
actor is invoked is an *output* of what happened to it. Each one would pass trivially
under a fixed "one turn per actor per stage" loop only if that loop were doing the
right thing by accident, and most of them cannot pass under one at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import scheduled_multiparty_world, single_response_world
from sworldmodel.engine import (
    KIND_DECISION,
    WAKE_DEADLINE,
    WAKE_NEED_FAILED,
    WAKE_NEED_MET,
    WAKE_OPPORTUNITY,
    WAKE_RULE,
    RunBudget,
    run,
)
from sworldmodel.models import ResolutionContract
from sworldmodel.schedule import Schedule, make_entry
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def _compile(data: dict, gateway: ProgrammableGateway, *, max_branches: int = 4):
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="test question",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        required_reality_facts=bundle.required_reality_facts,
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
        max_branches=max_branches,
    )


def _gateway(decision) -> ProgrammableGateway:
    return ProgrammableGateway(
        {"actor_decision": decision, "reflect": {"beliefs_update": [], "new_memories": []}}
    )


# ---------------------------------------------------------------------------
# 1. No constant limits an actor to a fixed number of calls
# ---------------------------------------------------------------------------


def test_no_fixed_actor_turn_structure_in_the_runtime() -> None:
    """The compiled world has no notion of rounds and the engine has no turn loop."""

    from sworldmodel import engine, worldspec

    assert not hasattr(worldspec.ProcessNode, "rounds")
    for path in (engine.__file__, worldspec.__file__):
        text = Path(path).read_text()
        assert "for _ in range(" not in text, f"{path} contains a fixed repetition loop"
    # A process node describes a moment and who may act, not how many times.
    node = worldspec.ProcessNode(node_id="n")
    assert not any("round" in f for f in node.__dataclass_fields__)


def test_actor_invocation_counts_are_not_uniform() -> None:
    """One actor gets extra information, so it is invoked more often than the others.

    Under a per-stage turn schedule every actor would be called the same number of
    times regardless of what reached them.
    """

    data = scheduled_multiparty_world()

    def decide(ctx: dict) -> dict:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        if ctx["actor_id"] == "member_0" and ctx["stage"] == "preparation":
            return act("circulate_note", {"text": "a note for the others"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    counts = next(iter(result.diagnostics.values())).actor_call_counts
    assert len(set(counts.values())) > 1, f"all actors invoked equally often: {counts}"


def test_two_branches_invoke_the_same_actor_different_numbers_of_times() -> None:
    """Different worlds, different events, different amounts of attention."""

    data = scheduled_multiparty_world()
    data["uncertainties"] = [
        {
            "variable": "external_signal",
            "why_unknown": "the measurement is not yet published",
            "reversal_capable": True,
            "release_at": "2026-06-09T12:00:00+00:00",
            "outcomes": [
                {
                    "value": "high",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 9.0]],
                },
                {
                    "value": "low",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 1.0]],
                },
            ],
        }
    ]

    def decide(ctx: dict) -> dict:
        signal = float(ctx.get("observed_fields", {}).get("external_signal", 0) or 0)
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold" if signal < 5 else "change"})
        # Only a high reading is worth writing to the others about.
        if signal > 5 and ctx["actor_id"] == "member_0":
            return act("circulate_note", {"text": "the reading is high"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    per_branch = {b: dict(d.actor_call_counts) for b, d in result.diagnostics.items()}
    assert len(per_branch) == 2
    totals = [sum(v.values()) for v in per_branch.values()]
    assert totals[0] != totals[1], f"branches invoked actors identically: {per_branch}"


# ---------------------------------------------------------------------------
# 2. Visibility, delivery, noticing and relevance are four different things
# ---------------------------------------------------------------------------


def test_visible_but_irrelevant_event_does_not_force_a_decision_call() -> None:
    """An actor notices an ambient public event and carries on.

    The world publishes something every actor can see, and no wake rule covers it. It
    must enter memory without costing a model call.
    """

    data = scheduled_multiparty_world()
    data["world_spec"]["wake_rules"] = []  # nothing in this world says the signal matters
    data["world_spec"]["process"]["nodes"] = [
        n for n in data["world_spec"]["process"]["nodes"] if n["node_id"] != "preparation"
    ]
    data["world_spec"]["process"]["nodes"][0]["at"] = "2026-06-25T09:00:00+00:00"

    calls: list[str] = []

    def decide(ctx: dict) -> dict:
        calls.append(f"{ctx['actor_id']}@{ctx['branch_time']}")
        return act("record_position", {"position": "hold"})

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    # The external release happens on 2026-06-09, well before the session on the 25th.
    early = [c for c in calls if "-06-09" in c]
    assert not early, f"an irrelevant public release triggered decisions: {early}"
    # But it was still perceived: it is in the ledger and was delivered.
    world = next(iter(result.final_worlds.values()))
    assert any(e.kind == "release_data" for e in world.event_history)
    assert any(d.noticed for d in world.deliveries)


def test_a_compiled_wake_rule_does_trigger_a_decision() -> None:
    """The same event, with a compiled rule saying it matters, does wake people."""

    data = scheduled_multiparty_world()  # keeps the signal_matters wake rule
    data["world_spec"]["process"]["nodes"] = [
        n for n in data["world_spec"]["process"]["nodes"] if n["node_id"] != "preparation"
    ]

    woken: list[str] = []

    def decide(ctx: dict) -> dict:
        woken.append(str(ctx["why_you_are_deciding_now"]["trigger"]))
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)
    assert WAKE_RULE in woken


def test_an_event_invisible_to_an_actor_never_reaches_it() -> None:
    """Private information is delivered only to its audience — not to everyone else."""

    data = scheduled_multiparty_world()
    data["world_spec"]["actions"][1]["visibility"] = "private"
    data["world_spec"]["actions"][1]["effects"] = [
        {
            "op": "deliver_information",
            "to": ["member_1"],
            "text": "$param.text",
            "visibility": "private",
        }
    ]

    def decide(ctx: dict) -> dict:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        if ctx["actor_id"] == "member_0":
            return act("circulate_note", {"text": "for member_1 only"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    note = next(e for e in world.event_history if e.kind == "deliver_information")
    recipients = {d.actor_id for d in world.deliveries if d.event_id == note.event_id}
    assert recipients == {"member_1"}, recipients


# ---------------------------------------------------------------------------
# 3. Waiting is a real state with real future consequences
# ---------------------------------------------------------------------------


def test_an_unmet_information_need_wakes_the_actor_at_its_deadline() -> None:
    """Waiting for something that never comes is itself an event."""

    data = single_response_world()
    seen: list[str] = []

    def decide(ctx: dict) -> dict:
        trigger = str(ctx["why_you_are_deciding_now"]["trigger"])
        seen.append(trigger)
        if trigger == WAKE_NEED_FAILED:
            return act("send_reply", {"answer": "no"})
        return {
            **wait_decision("waiting for confirmation before answering"),
            "information_needs": [
                {
                    "question": "has budget been confirmed?",
                    "asked_of": "",
                    "deadline": "2026-06-02T09:00:00+00:00",
                }
            ],
        }

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    assert WAKE_NEED_FAILED in seen, seen
    world = next(iter(result.final_worlds.values()))
    assert world.get_records("replies"), "the actor never got to act on the failed wait"


def test_a_deadline_wakes_the_participants() -> None:
    data = single_response_world(reply_deadline="2026-06-10T00:00:00+00:00")
    seen: list[str] = []

    def decide(ctx: dict) -> dict:
        seen.append(str(ctx["why_you_are_deciding_now"]["trigger"]))
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)
    assert WAKE_DEADLINE in seen, seen


def test_a_self_set_revisit_time_creates_a_real_future_wake_up() -> None:
    data = single_response_world()
    times: list[str] = []

    def decide(ctx: dict) -> dict:
        times.append(str(ctx["branch_time"]))
        if len(times) == 1:
            return {
                **wait_decision("I will look at this again on the 5th"),
                "revisit_when": [
                    {"description": "revisit the request", "at": "2026-06-05T08:00:00+00:00"}
                ],
            }
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)
    assert any("2026-06-05" in t for t in times), times


def test_an_answer_arriving_wakes_the_actor_that_asked() -> None:
    data = scheduled_multiparty_world()
    triggers: list[str] = []

    def decide(ctx: dict) -> dict:
        trigger = str(ctx["why_you_are_deciding_now"]["trigger"])
        triggers.append(trigger)
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        if ctx["actor_id"] == "member_1":
            return act("circulate_note", {"text": "here is the answer you asked for"})
        if ctx["actor_id"] == "member_0":
            return {
                **wait_decision("waiting on member_1"),
                "information_needs": [{"question": "what is the figure?", "asked_of": "member_1"}],
            }
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)
    assert WAKE_NEED_MET in triggers, triggers


def test_three_messages_at_three_times_produce_three_invocations() -> None:
    """Materially different things arriving at different real times each get a reaction.

    A fixed schedule gives one turn regardless. Here the count is three because three
    things happened.
    """

    data = single_response_world()
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "incoming",
            "description": "three separate messages arrive on three days",
            "occurrences": [
                {
                    "at": f"2026-05-2{day}T09:00:00+00:00",
                    "description": f"message {day}",
                    "effects": [
                        {
                            "op": "deliver_information",
                            "to": ["recipient"],
                            "text": f"message {day}: a materially different request",
                        }
                    ],
                }
                for day in (2, 4, 6)
            ],
        }
    ]
    data["world_spec"]["process"]["nodes"] = []  # nothing but the messages

    times: list[str] = []

    def decide(ctx: dict) -> dict:
        times.append(str(ctx["branch_time"])[:10])
        return wait_decision("noted")

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)

    assert sorted(set(times)) == ["2026-05-22", "2026-05-24", "2026-05-26"], times


def test_another_participant_speaking_is_a_reason_but_a_data_release_is_not() -> None:
    """A person saying something reaches you differently from the world changing."""

    data = scheduled_multiparty_world()
    data["world_spec"]["wake_rules"] = []  # no compiled rule covers either event
    reasons: list[str] = []

    def decide(ctx: dict) -> dict:
        reasons.append(str(ctx["why_you_are_deciding_now"]["trigger"]))
        if ctx["stage"] == "preparation" and ctx["actor_id"] == "member_0":
            return act("circulate_note", {"text": "my reading of the situation"})
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)

    assert "communication_from_another_actor" in reasons, reasons
    # The external measurement is public and no rule covers it: nobody was woken for it.
    assert all(r != "compiled_wake_rule" for r in reasons)


# ---------------------------------------------------------------------------
# 4. Plans persist; only material events interrupt them
# ---------------------------------------------------------------------------


def test_an_actor_busy_with_an_action_is_not_interrupted_by_a_mere_opportunity() -> None:
    """A long action holds the actor; an opportunity node does not break into it."""

    data = single_response_world()
    data["world_spec"]["actions"][0]["duration_seconds"] = 86400 * 5
    data["world_spec"]["process"]["nodes"].append(
        {
            "node_id": "nudge",
            "stage": "open",
            "at": "2026-05-21T10:00:00+00:00",
            "description": "another opportunity",
            "participants": ["recipient"],
        }
    )
    contexts: list[dict] = []

    def decide(ctx: dict) -> dict:
        contexts.append(ctx)
        current = ctx["current_action"]
        if current is not None and current["status"] == "in_progress":
            raise AssertionError("actor was interrupted mid-action by a mere opportunity")
        if current is None:
            return act("send_reply", {"answer": "yes"})
        return wait_decision("already replied")

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    # The nudge node fires on 05-21, while the reply is still in flight (05-20 -> 05-25).
    # It must not have produced an invocation.
    assert not [c for c in contexts if "2026-05-21" in c["branch_time"]]
    world = next(iter(result.final_worlds.values()))
    assert world.get_records("replies")


def test_plan_identity_survives_a_revision() -> None:
    data = single_response_world()
    plans: list[dict | None] = []

    def decide(ctx: dict) -> dict:
        plans.append(ctx.get("active_plan"))
        if len(plans) == 1:
            return {
                "plan_disposition": "replace",
                "plan_update": {
                    "goal": "answer once the figures are checked",
                    "basis": "the request names a deadline",
                    "steps": [{"description": "check figures", "at": "2026-06-02T09:00:00+00:00"}],
                },
                "action_mode": "wait",
                "reasoning": "not yet",
            }
        if len(plans) == 2:
            return {
                "plan_disposition": "revise",
                "plan_update": {"goal": "answer today", "steps": []},
                "action_mode": "wait",
                "reasoning": "narrowing the plan",
            }
        return act("send_reply", {"answer": "yes"})

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)

    named = [p for p in plans if p]
    assert len(named) >= 2
    # A revision keeps the plan's identity so the trace shows one plan changing.
    assert named[-1]["plan_id"] == named[0]["plan_id"]
    assert named[-1]["revision_count"] >= 1


def test_a_planned_step_with_a_time_schedules_a_real_future_opportunity() -> None:
    data = single_response_world()
    times: list[str] = []

    def decide(ctx: dict) -> dict:
        times.append(str(ctx["branch_time"]))
        if len(times) == 1:
            return {
                "plan_disposition": "replace",
                "plan_update": {
                    "goal": "reply after the review",
                    "basis": "the request names a deadline I must meet",
                    "steps": [
                        {
                            "description": "send the reply",
                            "intended_action_id": "send_reply",
                            "at": "2026-06-11T09:00:00+00:00",
                        }
                    ],
                },
                "action_mode": "wait",
                "reasoning": "review first",
            }
        return act("send_reply", {"answer": "yes"})

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    run(compiled, gw, seed=0)
    assert any("2026-06-11" in t for t in times), times


def test_a_scheduled_effect_actually_schedules() -> None:
    """An effect stamped in the future must not happen now, and must not drag the clock.

    Before, a future-stamped effect was applied on the spot and pulled the branch clock
    along with it, so everything genuinely due in between was skipped — which is the
    exact opposite of what scheduling means.
    """

    data = single_response_world()
    data["world_spec"]["process"]["nodes"][0]["effects"] = [
        {
            "op": "deliver_information",
            "to": ["recipient"],
            "text": "the request has arrived",
        },
        {
            "op": "schedule_event",
            "at": "2026-06-15T09:00:00+00:00",
            "event_type": "follow_up",
            "text": "a follow-up, three weeks later",
        },
    ]
    times: list[str] = []

    def decide(ctx: dict) -> dict:
        times.append(str(ctx["branch_time"]))
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))

    follow = [e for e in world.event_history if e.payload_dict.get("event_type") == "follow_up"]
    assert follow, "the scheduled event never fired"
    # It fired at its own time, not at the moment it was scheduled.
    assert follow[0].time.date().isoformat() == "2026-06-15"
    # And the clock did not leap there: the actor was seen on 05-20 first.
    assert any("2026-05-20" in t for t in times), times


# ---------------------------------------------------------------------------
# 5. Real time, real ordering, real limits
# ---------------------------------------------------------------------------


def test_actor_invocations_carry_real_branch_datetimes() -> None:
    data = scheduled_multiparty_world()
    gw = _gateway(
        lambda ctx: (
            act("record_position", {"position": "hold"})
            if ctx["stage"] == "session"
            else wait_decision()
        )
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    for d in result.actor_decisions:
        t = datetime.fromisoformat(d.branch_time)
        assert AS_OF <= t <= HORIZON
    session = [d for d in result.actor_decisions if d.stage == "session"]
    assert session and all("2026-06-25" in d.branch_time for d in session)


def test_same_timestamp_entries_pop_as_one_simultaneous_batch() -> None:
    t = datetime.fromisoformat("2026-06-01T00:00:00+00:00")
    s = Schedule()
    s = s.push(
        make_entry(at=t, kind=KIND_DECISION, actor_id="b", payload={"wake_reason": "x"}),
        make_entry(at=t, kind=KIND_DECISION, actor_id="a", payload={"wake_reason": "x"}),
        make_entry(at=t + timedelta(hours=1), kind=KIND_DECISION, actor_id="c"),
    )
    s2, batch = s.pop_batch(horizon=t + timedelta(days=1))
    assert {e.actor_id for e in batch} == {"a", "b"}
    assert len(s2) == 1


def test_schedule_order_does_not_depend_on_insertion_order() -> None:
    t = datetime.fromisoformat("2026-06-01T00:00:00+00:00")
    entries = [
        make_entry(at=t + timedelta(hours=h), kind=KIND_DECISION, actor_id=a)
        for h, a in [(2, "c"), (0, "a"), (1, "b")]
    ]
    a = Schedule().push(*entries)
    b = Schedule().push(*reversed(entries))
    assert [e.entry_id for e in a.entries] == [e.entry_id for e in b.entries]


def test_entries_beyond_the_horizon_are_reported_not_executed() -> None:
    data = single_response_world()
    data["world_spec"]["process"]["nodes"].append(
        {
            "node_id": "after_the_end",
            "stage": "later",
            "at": "2026-08-01T09:00:00+00:00",
            "description": "a step that falls outside the question's window",
            "participants": ["recipient"],
        }
    )
    gw = _gateway(lambda ctx: wait_decision())
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    diag = next(iter(result.diagnostics.values()))
    assert diag.pending_beyond_horizon
    assert all("2026-08-01" in e["at"] for e in diag.pending_beyond_horizon)


def test_a_budget_stop_leaves_the_branch_honest_not_finished() -> None:
    """Hitting a runtime limit must never manufacture an outcome."""

    data = scheduled_multiparty_world()
    gw = _gateway(lambda ctx: wait_decision("thinking"))
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_actor_calls=2))

    diag = next(iter(result.diagnostics.values()))
    assert "budget exhausted" in diag.stop_reason
    outcome = result.branch_outcomes[0]
    assert not outcome.resolved and outcome.outcome is None


def test_a_cut_short_trajectory_cannot_report_a_resolved_outcome() -> None:
    """Stopping early must not look like watching the process finish.

    The compiled terminal here has `unresolved_when: false`, so a naive evaluation of a
    truncated world would resolve it to NO. It must not.
    """

    data = single_response_world()
    data["world_spec"]["terminal"]["unresolved_when"] = {"op": "const", "args": [False]}
    # A long tail of scheduled work the branch will not get to.
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "drumbeat",
            "description": "many scheduled events before the horizon",
            "occurrences": [
                {
                    "at": f"2026-06-{day:02d}T09:00:00+00:00",
                    "description": f"tick {day}",
                    "effects": [
                        {"op": "deliver_information", "to": ["recipient"], "text": f"tick {day}"}
                    ],
                }
                for day in range(1, 21)
            ],
        }
    ]
    gw = _gateway(lambda ctx: wait_decision("thinking"))
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    diag = next(iter(result.diagnostics.values()))
    assert diag.unfired_in_horizon > 0
    outcome = result.branch_outcomes[0]
    assert not outcome.resolved
    assert "cut short" in (outcome.unresolved_reason or "")


def test_opportunity_is_the_wake_reason_for_a_process_node() -> None:
    data = single_response_world()
    gw = _gateway(lambda ctx: wait_decision())
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    assert any(WAKE_OPPORTUNITY in d.wake_reason for d in result.actor_decisions)


def test_every_invocation_records_the_full_causal_context() -> None:
    """Requirement: each actor call carries its trigger, time, what it noticed, what it
    retrieved, its plan before and after, its intention, and the world's verdict."""

    data = scheduled_multiparty_world()
    gw = _gateway(
        lambda ctx: (
            act("record_position", {"position": "hold"})
            if ctx["stage"] == "session"
            else wait_decision()
        )
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    assert result.actor_decisions
    for d in result.actor_decisions:
        assert d.wake_reason and d.wake_detail
        assert d.branch_time
        assert isinstance(d.delivered_observation_ids, list)
        assert isinstance(d.noticed_observation_ids, list)
        assert isinstance(d.retrieved_memory_ids, list)
        assert d.plan_disposition
        assert d.state_before is not None and d.state_after is not None
        assert d.intent["mode"]
        assert d.validation_status and d.validation_reason
        assert d.world_version_at_decision >= 0


def test_removing_the_event_removes_the_wake_up_it_caused() -> None:
    """Delete the cause, lose the invocation. This is what makes the trajectory real."""

    def run_with(nodes_filter) -> int:
        data = single_response_world()
        data["world_spec"]["process"]["nodes"] = nodes_filter(
            data["world_spec"]["process"]["nodes"]
        )
        gw = _gateway(lambda ctx: act("send_reply", {"answer": "yes"}))
        compiled = _compile(data, gw)
        result = run(compiled, gw, seed=0)
        return len(result.actor_decisions)

    with_event = run_with(lambda ns: ns)
    without_event = run_with(lambda ns: [])
    assert with_event > 0
    assert without_event == 0, "actors were invoked with nothing to invoke them"


@pytest.mark.parametrize("world", [scheduled_multiparty_world, single_response_world])
def test_the_same_runtime_executes_structurally_different_worlds(world) -> None:
    """One engine, two worlds with different actors, actions, graphs and terminals."""

    data = world()

    def decide(ctx: dict) -> dict:
        for card in ctx.get("feasible_actions", []):
            params = {}
            for p in card.get("parameters", []):
                if p.get("choices"):
                    params[p["name"]] = p["choices"][0]
                elif p.get("required"):
                    params[p["name"]] = "text"
            return act(card["action_id"], params)
        return wait_decision()

    gw = _gateway(decide)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    assert result.branch_outcomes
