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
    RunResult,
    run,
)
from sworldmodel.models import Event, ResolutionContract, Visibility
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
