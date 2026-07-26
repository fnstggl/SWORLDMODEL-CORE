"""Actor-lifecycle vertical slices (Phase 2: ACT-5, ACT-6, ACT-7).

Production-path harnesses: real compiler, real engine, real executor, real actor
runtime — the only stand-in is the scripted provider (``ProgrammableGateway``), so
every assertion below is about what the runtime actually did with the scripted
answers, never about a stand-in reasoning plausibly on the runtime's behalf.

* ACT-5 — one complete information → delivery → notice → interpretation → action
  lifecycle: an actor receives information mid-run, notices it, and acts because
  of it, with the causal chain visible in the records.
* ACT-6 — a multi-actor communication chain send → deliver → notice → interpret →
  respond → consequence, with all four separations (sending ≠ delivery ≠ notice ≠
  agreement) recorded as distinct transitions at distinct times.
* ACT-7 — a rejected intention followed by the actor reconsidering at its next
  wake: the refusal is on the record, and the subsequent intent is a different
  decision, not a rewrite of the first.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import AS_OF as AS_OF_S
from _worlds import HORIZON as HORIZON_S
from _worlds import single_response_world
from sworldmodel.compiled import CompiledWorld
from sworldmodel.engine import (
    WAKE_DIRECTED,
    WAKE_OWN_ACTION,
    WAKE_RULE,
    RunResult,
    run,
)
from sworldmodel.models import Event, ResolutionContract, Visibility
from sworldmodel.temporal_report import compute_temporal_report
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat(AS_OF_S)
HORIZON = datetime.fromisoformat(HORIZON_S)


def _compile(data: dict[str, Any], gateway: ProgrammableGateway) -> CompiledWorld:
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
        max_branches=4,
    )


def _gateway(decision: Any) -> ProgrammableGateway:
    return ProgrammableGateway(
        {"actor_decision": decision, "reflect": {"beliefs_update": [], "new_memories": []}}
    )


def _only_world(result: RunResult) -> Any:
    return next(iter(result.final_worlds.values()))


def _events(result: RunResult, kind: str) -> list[Event]:
    return [e for e in result.event_ledger if e.kind == kind]


# ---------------------------------------------------------------------------
# ACT-5: information → delivery → notice → interpretation → action
# ---------------------------------------------------------------------------


def test_act5_information_delivery_notice_interpretation_action_lifecycle() -> None:
    """An actor receives information mid-run, notices it, and acts because of it.

    Every transition of the lifecycle is a separate recorded step, and the causal
    chain — information event id → delivery record → notice → the decision the
    notice triggered → the action it produced → the world consequence — is
    asserted link by link on the run's own records.
    """

    data = single_response_world()
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "budget_confirmation",
            "description": "the awaited confirmation arrives mid-run",
            "occurrences": [
                {
                    "at": "2026-06-01T09:00:00+00:00",
                    "description": "budget confirmed",
                    "effects": [
                        {
                            "op": "deliver_information",
                            "to": ["recipient"],
                            "text": "budget confirmed: proceed with the reply",
                        }
                    ],
                }
            ],
        }
    ]

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        confirming = [
            o["obs_id"] for o in ctx["observations"] if "budget confirmed" in o["summary"]
        ]
        if confirming:
            return act(
                "send_reply",
                {"answer": "yes"},
                reasoning="the budget is confirmed, so I reply yes",
                referenced_observation_ids=confirming,
            )
        return wait_decision("waiting for budget confirmation")

    gw = _gateway(decide)
    result = run(_compile(data, gw), gw, seed=0)
    world = _only_world(result)

    # The INFORMATION: a real mid-run event, produced by the environment at its own
    # scheduled time — not present at seeding.
    info = next(
        e
        for e in _events(result, "deliver_information")
        if "budget confirmed" in str(e.payload_dict.get("text", ""))
    )
    assert info.actor_id is None
    assert info.time == datetime.fromisoformat("2026-06-01T09:00:00+00:00")

    # The DELIVERY and the NOTICE: two separately recorded transitions for this
    # actor and this event.
    delivery = next(
        d for d in world.deliveries if d.event_id == info.event_id and d.actor_id == "recipient"
    )
    assert delivery.available_at == info.time
    assert delivery.noticed_at is not None
    assert delivery.noticed_at >= delivery.available_at

    # Before the information arrived the actor decided too — and decided to wait.
    before = [
        d for d in result.actor_decisions if datetime.fromisoformat(d.branch_time) < info.time
    ]
    assert before, "the actor was never invoked before the information arrived"
    assert all(d.validation_status == "wait" for d in before)

    # The INTERPRETATION → ACTION: the wake the notice triggered names the event as
    # its cause, shows it among what was noticed, and produced the action.
    acted = next(d for d in result.actor_decisions if d.intent.get("action_id") == "send_reply")
    assert WAKE_DIRECTED in acted.wake_reason
    assert info.event_id in acted.trigger_event_ids
    assert info.event_id in acted.noticed_observation_ids
    assert any(
        "budget confirmed" in str(o.get("summary", ""))
        for o in acted.decision_context.get("observations", [])
    )
    assert acted.intent["params"] == {"answer": "yes"}
    assert "budget is confirmed" in str(acted.intent["rationale"])
    assert acted.validation_status == "started"
    assert datetime.fromisoformat(acted.branch_time) == delivery.noticed_at

    # The CONSEQUENCE: the action started, took its real duration, completed, and
    # only then did the world change — after the information, because of it.
    started = next(e for e in _events(result, "action_started") if e.event_id in acted.event_ids)
    assert str(started.payload_dict.get("action_id")) == "send_reply"
    assert started.time == info.time  # the decision followed the notice, that instant
    (completed,) = _events(result, "action_completed")
    # 1800s of real duration: acting is not succeeding, and succeeding took time.
    assert completed.time == datetime.fromisoformat("2026-06-01T09:30:00+00:00")
    (reply,) = world.get_records("replies")
    assert reply["value"] == "yes"
    assert reply["time"] > info.time

    (outcome,) = result.branch_outcomes
    assert outcome.resolved and outcome.outcome == "YES"


# ---------------------------------------------------------------------------
# ACT-6: send → deliver → notice → interpret → respond → consequence
# ---------------------------------------------------------------------------


def _claim(cid: str, proposition: str) -> dict[str, Any]:
    return {"id": cid, "proposition": proposition, "supporting_excerpt": proposition}


def two_actor_exchange_world() -> dict[str, Any]:
    """Two people who can only reach each other by sending things that take real
    time to write, real time to arrive, and real time to be read."""

    def entity(eid: str, name: str, role: str) -> dict[str, Any]:
        return {
            "entity_id": eid,
            "name": name,
            "kind": "person",
            "is_actor": True,
            "role": role,
            "authority": ["correspond"],
            "representation_scale": "individual",
            "evidence_claim_ids": [f"c_{eid}"],
        }

    return {
        "reality": {
            "as_of": AS_OF_S,
            "horizon": HORIZON_S,
            "subject_entity": "the exchange",
            "resolution_units": "a recorded answer",
            "target_outcome": "the reviewer answers yes",
            "expected_participants": 2,
        },
        "claims": [
            _claim("c_proposer", "the proposer's role and mandate are documented"),
            _claim("c_reviewer", "the reviewer's role and mandate are documented"),
            _claim("c_exchange", "the exchange opens on 2026-05-20"),
        ],
        "world_spec": {
            "title": "a two-person exchange",
            "subject_entity": "the exchange",
            "resolution_units": "a recorded answer",
            "entities": [
                entity("proposer", "The Proposer", "proposer"),
                entity("reviewer", "The Reviewer", "reviewer"),
            ],
            "actors": [
                {
                    "entity_id": "proposer",
                    "reasoning": "opens the exchange and waits for the answer",
                    "memory_seeds": [
                        {
                            "content": "I must send the request out this week.",
                            "kind": "episodic",
                            "importance": 0.8,
                            "evidence_claim_ids": ["c_proposer"],
                        }
                    ],
                },
                {
                    "entity_id": "reviewer",
                    "reasoning": "answers requests that actually reach them",
                    "memory_seeds": [
                        {
                            "content": "I answer correspondence once I have read it.",
                            "kind": "episodic",
                            "importance": 0.8,
                            "evidence_claim_ids": ["c_reviewer"],
                        }
                    ],
                },
            ],
            "fields": [],
            "actions": [
                {
                    "action_id": "send_request",
                    "meaning": "write and send the request",
                    "eligible_actors": ["proposer"],
                    "required_authority": ["correspond"],
                    "parameters": [{"name": "text", "type": "string", "required": True}],
                    "duration_seconds": 3600,
                    "delivery_delay_seconds": 7200,
                    "notice_delay_seconds": 1800,
                    "effects": [
                        {"op": "deliver_information", "to": ["reviewer"], "text": "$param.text"}
                    ],
                    "evidence_claim_ids": ["c_exchange"],
                },
                {
                    "action_id": "send_answer",
                    "meaning": "consider the request and send the answer",
                    "eligible_actors": ["reviewer"],
                    "required_authority": ["correspond"],
                    "parameters": [
                        {
                            "name": "answer",
                            "type": "option",
                            "required": True,
                            "choices": ["yes", "no"],
                        }
                    ],
                    "duration_seconds": 1800,
                    "delivery_delay_seconds": 600,
                    "notice_delay_seconds": 300,
                    "effects": [
                        {
                            "op": "deliver_information",
                            "to": ["proposer"],
                            "text": "my answer: $param.answer",
                        },
                        {
                            "op": "append_record",
                            "collection": "answers",
                            "key": "$actor",
                            "value": "$param.answer",
                        },
                    ],
                    "evidence_claim_ids": ["c_exchange"],
                },
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "opening",
                        "stage": "correspondence",
                        "at": "2026-05-20T09:00:00+00:00",
                        "description": "the proposer opens the exchange",
                        "participants": ["proposer"],
                        "action_ids": ["send_request"],
                    }
                ]
            },
            "wake_rules": [],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [
                        {
                            "op": "count",
                            "args": [
                                "answers",
                                {
                                    "op": "equals",
                                    "args": [{"op": "item", "args": ["value"]}, "yes"],
                                },
                            ],
                        },
                        1,
                    ],
                },
                "unresolved_when": {"op": "const", "args": [False]},
                "description": "YES when the reviewer's recorded answer is yes",
            },
        },
    }


def test_act6_two_actor_chain_keeps_all_four_separations() -> None:
    """send → deliver → notice → interpret → respond → consequence, across two
    actors, with sending ≠ delivery ≠ notice ≠ agreement each its own recorded
    transition at its own time."""

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["actor_id"] == "proposer":
            if ctx["current_action"] is None and not ctx["observations"]:
                return act(
                    "send_request",
                    {"text": "please confirm the figure by Friday"},
                    reasoning="opening the exchange",
                )
            return wait_decision("the request is out; waiting")
        if any("please confirm" in o["summary"] for o in ctx["observations"]):
            return act(
                "send_answer", {"answer": "yes"}, reasoning="the request reached me; I answer"
            )
        return wait_decision("nothing has reached me")

    data = two_actor_exchange_world()
    gw = _gateway(decide)
    result = run(_compile(data, gw), gw, seed=0)
    world = _only_world(result)

    # 1. SENDING happened when the writing finished — not when the actor decided.
    request = next(
        e
        for e in _events(result, "deliver_information")
        if "please confirm" in str(e.payload_dict.get("text", ""))
    )
    assert request.actor_id == "proposer"
    sent_at = request.time
    assert sent_at == datetime.fromisoformat("2026-05-20T10:00:00+00:00")
    decided = next(d for d in result.actor_decisions if d.intent.get("action_id") == "send_request")
    assert datetime.fromisoformat(decided.branch_time) < sent_at  # deciding ≠ sending

    # 2. DELIVERY is later than sending (the channel takes real time).
    to_reviewer = next(
        d for d in world.deliveries if d.event_id == request.event_id and d.actor_id == "reviewer"
    )
    assert to_reviewer.available_at == datetime.fromisoformat("2026-05-20T12:00:00+00:00")
    assert to_reviewer.available_at > sent_at

    # 3. NOTICE is later than delivery (reading is not receiving).
    assert to_reviewer.noticed_at == datetime.fromisoformat("2026-05-20T12:30:00+00:00")
    assert to_reviewer.noticed_at > to_reviewer.available_at

    # 4. INTERPRETATION → RESPONSE: the reviewer's wake names the request as its
    # cause; its response is its own validated decision, not an echo.
    response = next(d for d in result.actor_decisions if d.intent.get("action_id") == "send_answer")
    assert response.actor_id == "reviewer"
    assert WAKE_DIRECTED in response.wake_reason
    assert request.event_id in response.trigger_event_ids
    assert request.event_id in response.noticed_observation_ids
    assert datetime.fromisoformat(response.branch_time) == to_reviewer.noticed_at
    assert response.validation_status == "started"

    # NOTICE ≠ AGREEMENT: at the moment of noticing, nothing in the world had been
    # agreed — the recorded answer exists only after the reviewer's own action
    # completed, strictly later.
    (answer,) = world.get_records("answers")
    assert answer["by"] == "reviewer" and answer["value"] == "yes"
    answered_at = answer["time"]
    assert answered_at == datetime.fromisoformat("2026-05-20T13:00:00+00:00")
    assert answered_at > to_reviewer.noticed_at

    # 5. CONSEQUENCE: the answer travelled back and the proposer noticed it — again
    # through delivery and notice, not instantly.
    reply_ev = next(
        e
        for e in _events(result, "deliver_information")
        if str(e.payload_dict.get("text", "")).startswith("my answer:")
    )
    back = next(
        d for d in world.deliveries if d.event_id == reply_ev.event_id and d.actor_id == "proposer"
    )
    assert back.available_at == datetime.fromisoformat("2026-05-20T13:10:00+00:00")
    assert back.noticed_at == datetime.fromisoformat("2026-05-20T13:15:00+00:00")

    # The whole chain is strictly ordered: no two lifecycle stages collapsed.
    chain = [
        sent_at,
        to_reviewer.available_at,
        to_reviewer.noticed_at,
        answered_at,
        back.available_at,
        back.noticed_at,
    ]
    assert chain == sorted(chain) and len(set(chain)) == len(chain)

    (outcome,) = result.branch_outcomes
    assert outcome.resolved and outcome.outcome == "YES"


# ---------------------------------------------------------------------------
# ACT-7: a rejected intention, then a genuine reconsideration
# ---------------------------------------------------------------------------


def test_act7_rejected_action_is_recorded_and_reconsidered_at_the_next_wake() -> None:
    """The environment refuses an infeasible intention with the exact reason; the
    actor is woken *for the refusal* and its next intent is a different decision."""

    data = single_response_world()
    attempts: list[str] = []

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        trigger = str(ctx["why_you_are_deciding_now"]["trigger"])
        if WAKE_OWN_ACTION in trigger:
            # The refusal reached me; I choose something the world allows.
            return act("send_reply", {"answer": "no"}, reasoning="scaling back after the refusal")
        if not attempts:
            attempts.append("first")
            return act(
                "send_reply", {"answer": "maybe"}, reasoning="hedging outside the permitted answers"
            )
        return wait_decision("nothing further")

    gw = _gateway(decide)
    result = run(_compile(data, gw), gw, seed=0)
    world = _only_world(result)

    # The REJECTION record: the environment refused, said exactly why, and told the
    # actor — privately, as the world's verdict on its attempt.
    rejected = next(d for d in result.actor_decisions if d.validation_status == "rejected")
    assert rejected.intent["params"] == {"answer": "maybe"}
    assert "not one of" in rejected.validation_reason
    (rejection_ev,) = _events(result, "action_rejected")
    assert rejection_ev.visibility is Visibility.PRIVATE
    assert rejection_ev.audience == ("recipient",)
    assert "not one of" in str(rejection_ev.payload_dict.get("reason", ""))
    # A refusal changes nothing in the world: no reply existed when it was issued.
    assert all(r["time"] > rejection_ev.time for r in world.get_records("replies"))

    # The RECONSIDERATION: the very next wake of this actor is caused by its own
    # refused attempt, and produces a *changed* intent that the world accepts.
    after = [
        d
        for d in result.actor_decisions
        if d.actor_id == "recipient"
        and datetime.fromisoformat(d.branch_time) >= datetime.fromisoformat(rejected.branch_time)
        and d is not rejected
    ]
    assert after, "the actor was never woken again after the refusal"
    reconsidered = after[0]
    assert WAKE_OWN_ACTION in reconsidered.wake_reason
    assert "refused" in reconsidered.wake_detail
    assert reconsidered.intent["action_id"] == "send_reply"
    assert reconsidered.intent["params"] == {"answer": "no"}
    assert reconsidered.intent["params"] != rejected.intent["params"]
    assert reconsidered.validation_status == "started"

    # The reconsidered action really happened: started, completed, recorded.
    assert len(_events(result, "action_started")) == 1
    assert len(_events(result, "action_completed")) == 1
    (reply,) = world.get_records("replies")
    assert reply["value"] == "no"
    (outcome,) = result.branch_outcomes
    assert outcome.resolved and outcome.outcome == "NO"


# ---------------------------------------------------------------------------
# Staged-release harness: two actors, real message traffic, and two uncertainties
# whose evidence dates them nineteen days apart (ACT-5/6, TMP-4, COM-1)
# ---------------------------------------------------------------------------

EARLY_RELEASE = "2026-06-01T09:00:00+00:00"
LATE_RELEASE = "2026-06-20T09:00:00+00:00"


def staged_release_exchange_world() -> dict[str, Any]:
    """A world the live geopolitical run could not be: two people who reach each other
    only by sending things, and TWO uncertain published figures with DIFFERENT release
    dates.

    The live Phase-2 run had one actor, no message traffic at all and a single
    uncertainty, so it could not tell a fixed release calendar from a merged one, nor a
    working communication counter from a broken one — every branch looked the same
    whichever way the runtime behaved. Here the analyst can only learn the early figure
    when it is published, can only tell the decider by sending a note that takes real
    time to arrive and be read, and the decider's own figure is not published until
    nineteen days later. A runtime that merges the two releases, or that publishes an
    undated value at a neighbour's date, produces a visibly different world.
    """

    def entity(eid: str, name: str, role: str, authority: list[str]) -> dict[str, Any]:
        return {
            "entity_id": eid,
            "name": name,
            "kind": "person",
            "is_actor": True,
            "role": role,
            "authority": authority,
            "representation_scale": "individual",
            "evidence_claim_ids": [f"c_{eid}"],
        }

    return {
        "reality": {
            "as_of": AS_OF_S,
            "horizon": HORIZON_S,
            "subject_entity": "the determination",
            "resolution_units": "a recorded determination",
            "target_outcome": "the decider records go",
            "expected_participants": 2,
        },
        "claims": [
            _claim("c_analyst", "the analyst's role and mandate are documented"),
            _claim("c_decider", "the decider's role and mandate are documented"),
            _claim("c_early", f"the early figure is published on {EARLY_RELEASE}"),
            _claim("c_late", f"the confirming figure is published on {LATE_RELEASE}"),
        ],
        "world_spec": {
            "title": "a staged two-figure determination",
            "subject_entity": "the determination",
            "resolution_units": "a recorded determination",
            "entities": [
                entity("analyst", "The Analyst", "analyst", ["report"]),
                entity("decider", "The Decider", "decider", ["determine"]),
            ],
            "actors": [
                {
                    "entity_id": "analyst",
                    "reasoning": "reports a published figure to the decider once it exists",
                    "memory_seeds": [
                        {
                            "content": "I pass published figures on as soon as they exist.",
                            "kind": "episodic",
                            "importance": 0.8,
                            "evidence_claim_ids": ["c_analyst"],
                        }
                    ],
                },
                {
                    "entity_id": "decider",
                    "reasoning": "determines once the confirming figure is out",
                    "memory_seeds": [
                        {
                            "content": "I do not determine before the confirming figure exists.",
                            "kind": "episodic",
                            "importance": 0.8,
                            "evidence_claim_ids": ["c_decider"],
                        }
                    ],
                },
            ],
            "fields": [],
            "actions": [
                {
                    "action_id": "report_reading",
                    "meaning": "send the decider what the early figure turned out to be",
                    "eligible_actors": ["analyst"],
                    "required_authority": ["report"],
                    "parameters": [{"name": "text", "type": "string", "required": True}],
                    "duration_seconds": 1800,
                    "delivery_delay_seconds": 3600,
                    "notice_delay_seconds": 1800,
                    "effects": [
                        {"op": "deliver_information", "to": ["decider"], "text": "$param.text"}
                    ],
                    "evidence_claim_ids": ["c_early"],
                },
                {
                    "action_id": "record_determination",
                    "meaning": "record the determination",
                    "eligible_actors": ["decider"],
                    "required_authority": ["determine"],
                    "parameters": [
                        {
                            "name": "call",
                            "type": "option",
                            "required": True,
                            "choices": ["go", "hold"],
                        }
                    ],
                    "duration_seconds": 900,
                    "effects": [
                        {
                            "op": "append_record",
                            "collection": "determinations",
                            "key": "$actor",
                            "value": "$param.call",
                        }
                    ],
                    "evidence_claim_ids": ["c_late"],
                },
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "determination",
                        "stage": "determination",
                        "at": "2026-06-21T09:00:00+00:00",
                        "description": "the decider makes the call",
                        "participants": ["decider"],
                        "action_ids": ["record_determination"],
                    }
                ]
            },
            "wake_rules": [
                {
                    "rule_id": "early_figure_reaches_the_analyst",
                    "wakes": ["analyst"],
                    "reason": "the early figure has been published",
                    "on_field_change": "early_signal",
                },
                {
                    "rule_id": "late_figure_reaches_the_decider",
                    "wakes": ["decider"],
                    "reason": "the confirming figure has been published",
                    "on_field_change": "late_signal",
                },
            ],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [
                        {
                            "op": "count",
                            "args": [
                                "determinations",
                                {"op": "equals", "args": [{"op": "item", "args": ["value"]}, "go"]},
                            ],
                        },
                        1,
                    ],
                },
                "unresolved_when": {
                    "op": "less_than",
                    "args": [{"op": "count", "args": ["determinations"]}, 1],
                },
                "description": "YES when the decider records go",
            },
        },
        "uncertainties": [
            {
                "variable": "early_signal",
                "why_unknown": "the early figure is published after the cutoff",
                "reversal_capable": True,
                "release_at": EARLY_RELEASE,
                "evidence_claim_ids": ["c_early"],
                "outcomes": [
                    {
                        "value": "high",
                        "weight": 0.5,
                        "provenance": "symmetric_ignorance_assumption",
                        "field_effects": [["early_signal", 9.0]],
                    },
                    {
                        "value": "low",
                        "weight": 0.5,
                        "provenance": "symmetric_ignorance_assumption",
                        "field_effects": [["early_signal", 1.0]],
                    },
                ],
            },
            {
                "variable": "late_signal",
                "why_unknown": "the confirming figure is published nineteen days later",
                "reversal_capable": True,
                "release_at": LATE_RELEASE,
                "evidence_claim_ids": ["c_late"],
                "outcomes": [
                    {
                        "value": "high",
                        "weight": 0.5,
                        "provenance": "symmetric_ignorance_assumption",
                        "field_effects": [["late_signal", 9.0]],
                    },
                    {
                        "value": "low",
                        "weight": 0.5,
                        "provenance": "symmetric_ignorance_assumption",
                        "field_effects": [["late_signal", 1.0]],
                    },
                ],
            },
        ],
    }


def staged_release_decisions(ctx: dict[str, Any]) -> dict[str, Any]:
    """The scripted script for :func:`staged_release_exchange_world`.

    Each actor answers only from what it can actually see: the analyst reports the
    early figure once it is in its own observed fields, and the decider calls it only
    once the confirming figure is in its. Nothing here inspects the branch id — an
    actor that can see the future would betray itself by acting early.
    """

    fields = ctx.get("observed_fields") or {}
    trigger = str(ctx["why_you_are_deciding_now"]["trigger"])
    if ctx["actor_id"] == "analyst":
        # Reports at the wake the publication itself caused — nothing test-side keeps
        # track of whether it has already reported, because branches run concurrently
        # and shared state would let one branch answer for another.
        if WAKE_RULE in trigger and "early_signal" in fields:
            return act(
                "report_reading",
                {"text": f"the early figure is {fields['early_signal']}"},
                reasoning="the early figure is out; it must be passed on",
            )
        return wait_decision("nothing to report")
    if ctx["stage"] == "determination" and "late_signal" in fields:
        call = "go" if float(fields["late_signal"]) > 5 else "hold"
        return act("record_determination", {"call": call}, reasoning="both figures are in")
    return wait_decision("not both figures are in")


def test_staged_release_harness_runs_the_whole_production_path() -> None:
    """The harness the live Phase-2 run could not be: two actors, real message
    traffic, two uncertainties dated nineteen days apart — one real compilation, one
    real run, all four branches.

    It is here because the live geopolitical run could not discriminate a fix from
    the bug it was meant to prove: one actor, zero messages and a single uncertainty
    produce the same artifacts whether releases are merged or separate and whether the
    message counters work or not. This world's branches differ in what each actor
    could see, when, and what it sent as a result.
    """

    gw = _gateway(staged_release_decisions)
    result = run(_compile(staged_release_exchange_world(), gw), gw, seed=0)
    early = datetime.fromisoformat(EARLY_RELEASE)
    late = datetime.fromisoformat(LATE_RELEASE)

    assert set(result.final_worlds) == {
        "sc_early_signal:high_late_signal:high",
        "sc_early_signal:high_late_signal:low",
        "sc_early_signal:low_late_signal:high",
        "sc_early_signal:low_late_signal:low",
    }

    for branch_id, world in result.final_worlds.items():
        # TMP-4: each figure became public at its OWN date, carrying only itself.
        hyps = [
            (e.time, dict(e.payload_dict.get("fields") or {}))
            for e in world.event_history
            if e.kind == "release_data"
        ]
        assert [t for t, _ in hyps] == [early, late], f"{branch_id}: {hyps}"
        assert set(hyps[0][1]) == {"early_signal"} and set(hyps[1][1]) == {"late_signal"}

        # ACT-5/ACT-6/COM-1: the analyst learned the early figure, sent it, and the
        # decider received it — send, delivery and notice at three different times.
        (note,) = [e for e in world.event_history if e.kind == "deliver_information"]
        assert note.actor_id == "analyst"
        assert str(hyps[0][1]["early_signal"]) in str(note.payload_dict.get("text", ""))
        (delivery,) = [d for d in world.deliveries if d.event_id == note.event_id]
        assert delivery.actor_id == "decider"
        assert note.time < delivery.available_at < delivery.noticed_at

        # The determination used the LATE figure, so it could not have been made
        # before that figure existed.
        (determination,) = world.get_records("determinations")
        assert determination["time"] > late
        want = "go" if hyps[1][1]["late_signal"] > 5 else "hold"
        assert determination["value"] == want, f"{branch_id}: decided on the wrong figure"

    # No actor saw the later figure before it was published — checked on what each
    # decision was actually shown.
    for d in result.actor_decisions:
        if datetime.fromisoformat(d.branch_time) < late:
            assert "late_signal" not in d.decision_context.get("observed_fields", {}), (
                f"{d.branch_id}/{d.actor_id}@{d.branch_time} read an unpublished figure"
            )

    # The branches genuinely disagree — the run is not four copies of one trajectory.
    outcomes = {b.branch_id: b.outcome for b in result.branch_outcomes}
    assert set(outcomes.values()) == {"YES", "NO"}

    report = compute_temporal_report(result)
    for branch in report["branches"].values():
        # Two hypothesis releases per branch, and NOT a single process update: this
        # world's processes move nothing on their own.
        assert branch["scenario_releases"] == 2
        assert branch["process_updates"] == 0
        # One message, one recipient, delivered and read: sent and delivered agree
        # because they are now the same unit.
        assert branch["messages"] == {
            "send_events": 1,
            "sent": 1,
            "delivered": 1,
            "undelivered": 0,
            "noticed": 1,
            "missed": 0,
        }
        # Real delays were compiled, so nothing was noticed the instant it was sent.
        assert branch["same_timestamp_communications"] == 0


def test_a_deadline_does_not_re_ask_a_participant_who_already_acted() -> None:
    """FD-18: a node queues both an opportunity and a deadline wake per participant.
    An actor that took the opportunity has done the thing the deadline exists to
    catch; waking it again about the same node, the same day, with nothing changed in
    between is the duplicate the live run's audit flagged. An actor that DECLINED the
    opportunity still gets its deadline — a closing window is real news to someone who
    has not acted."""

    data = single_response_world(reply_deadline="2026-05-20T18:00:00+00:00")

    def act_at_once(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["current_action"] is None and ctx["last_decision_time"] is None:
            return act("send_reply", {"answer": "yes"})
        return wait_decision("already replied")

    gw = _gateway(act_at_once)
    acted_run = run(_compile(data, gw), gw, seed=0)
    wakes = [(d.branch_time, d.wake_reason) for d in acted_run.actor_decisions]
    assert any(d.validation_status == "started" for d in acted_run.actor_decisions), (
        "probe shape lost: the actor never took the opportunity"
    )
    assert not any("deadline_reached" in reason for _, reason in wakes), (
        f"the actor was re-asked at the deadline after already acting: {wakes}"
    )

    # The same world, same deadline — but the actor waits. The deadline still reaches
    # it, and the report attributes the repeat to the calendar, not to a phantom
    # arrival of information.
    gw2 = _gateway(lambda ctx: wait_decision("not yet"))
    waited_run = run(_compile(data, gw2), gw2, seed=0)
    waited = [(d.branch_time, d.wake_reason) for d in waited_run.actor_decisions]
    assert any("deadline_reached" in reason for _, reason in waited), (
        f"a participant that had NOT acted was never woken at the deadline: {waited}"
    )
    flagged = compute_temporal_report(waited_run)["branches"]["baseline"]["wake_ups"]
    assert flagged["repeated_without_new_information_by_cause"]["information_driven"] == 0
    assert flagged["repeated_without_new_information_by_cause"]["calendar_driven"] >= 1


def test_the_act8_headline_separates_calendar_repeats_from_phantom_arrivals() -> None:
    """FD-18(2): ACT-8's headline number splits by wake cause.

    An information-driven repeat is a defect every time — the world said something
    reached this actor and the actor's own noticed set is empty of it. A calendar-driven
    repeat is a deadline or an opportunity arriving with nothing new, which is ordinary
    life. Reported under one number, a live run's "8 of 12 wakes repeated without new
    information" was about half false alarms, and a number that is half false alarms is
    a number readers stop reading.
    """

    data = two_actor_exchange_world()
    # A calendar wake for the reviewer long after everything has been said and read.
    data["world_spec"]["process"]["nodes"].append(
        {
            "node_id": "reviewer_diary",
            "stage": "correspondence",
            "at": "2026-06-15T09:00:00+00:00",
            "description": "the reviewer looks at their calendar",
            "participants": ["reviewer"],
            "action_ids": ["send_answer"],
        }
    )

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["actor_id"] == "proposer":
            if ctx["current_action"] is None and not ctx["observations"]:
                return act("send_request", {"text": "please confirm the figure by Friday"})
            return wait_decision("waiting")
        if any("please confirm" in o["summary"] for o in ctx["observations"]):
            return act("send_answer", {"answer": "yes"})
        return wait_decision("nothing has reached me")

    gw = _gateway(decide)
    result = run(_compile(data, gw), gw, seed=0)
    wake_ups = compute_temporal_report(result)["branches"]["baseline"]["wake_ups"]

    diary = [
        f
        for f in wake_ups["flagged_repeats"]
        if f["actor_id"] == "reviewer" and f["branch_time"].startswith("2026-06-15")
    ]
    assert diary, "probe shape lost: the late diary wake was not a flagged repeat"
    assert diary[0]["cause_class"] == "calendar_driven"
    by_cause = wake_ups["repeated_without_new_information_by_cause"]
    assert sum(by_cause.values()) == wake_ups["repeated_without_new_information"], (
        f"the split does not account for every flagged repeat: {by_cause}"
    )
    assert by_cause["calendar_driven"] >= 1
    assert by_cause["unclassified"] == 0, (
        f"a wake cause the split does not know about: "
        f"{[f['wake_reason'] for f in wake_ups['flagged_repeats']]}"
    )
