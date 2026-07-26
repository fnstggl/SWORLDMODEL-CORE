"""The §9 temporal report (TMP-2, FD-13) and honest release timing (TMP-4, FD-7).

Everything here runs the REAL production path — real compiler, real engine, real
trace writer — with the scripted provider as the only stand-in. The report is
computed from the run's own record, so each test builds a run whose §9 counters
are known in advance and asserts them exactly.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import AS_OF as AS_OF_S
from _worlds import HORIZON as HORIZON_S
from _worlds import scheduled_multiparty_world, single_response_world
from sworldmodel.compiled import CompiledWorld
from sworldmodel.engine import WAKE_OWN_ACTION, RunResult, run
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.ids import canonical_json
from sworldmodel.models import ResolutionContract
from sworldmodel.outcomes import aggregate
from sworldmodel.rundir import PIPELINE_ARTIFACTS, prepare_run_dir
from sworldmodel.schedule import (
    KIND_SCENARIO_RELEASE,
    ORIGIN_EXTERNAL,
    Schedule,
    make_entry,
)
from sworldmodel.temporal_report import (
    TEMPORAL_REPORT_FILENAME,
    compute_temporal_report,
    write_temporal_report,
)
from sworldmodel.tracing import TraceContext
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat(AS_OF_S)
HORIZON = datetime.fromisoformat(HORIZON_S)


def _compile_pair(
    data: dict[str, Any], gateway: ProgrammableGateway
) -> tuple[ResolutionContract, CompiledWorld]:
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
    return contract, compiled


def _gateway(decision: Any) -> ProgrammableGateway:
    return ProgrammableGateway(
        {"actor_decision": decision, "reflect": {"beliefs_update": [], "new_memories": []}}
    )


# ---------------------------------------------------------------------------
# TMP-2: every §9 counter, computed from a run whose shape is known exactly
# ---------------------------------------------------------------------------


def test_every_section9_counter_on_a_known_run() -> None:
    """One branch, two wakes, one message, one action, one long silent jump.

    The run: the request lands on 05-20 10:00 (message sent, delivered and noticed
    at that instant); the actor waits and sets itself a revisit for 06-05 08:00
    (which coincides with the node deadline, so the two causes merge into one
    wake); at that wake — which carries nothing materially new — it acts; the
    action takes 1800s; the terminal is recorded at the horizon.
    """

    data = single_response_world(reply_deadline="2026-06-05T08:00:00+00:00")
    calls: list[str] = []

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        calls.append(str(ctx["branch_time"]))
        if len(calls) == 1:
            return {
                **wait_decision("I will come back to this"),
                "revisit_when": [
                    {"description": "look at the request again", "at": "2026-06-05T08:00:00+00:00"}
                ],
            }
        return act("send_reply", {"answer": "yes"})

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    report = compute_temporal_report(result)

    assert set(report["branches"]) == {"baseline"}
    b = report["branches"]["baseline"]

    # Distinct timestamps: 05-20 10:00 (request + wait), 06-05 08:00 (wake, start),
    # 06-05 08:30 (effects land), horizon (terminal recorded).
    assert b["distinct_timestamps"] == 4
    assert b["first_timestamp"] == "2026-05-20T10:00:00+00:00"
    assert b["last_timestamp"] == "2026-06-25T23:59:59+00:00"

    # The largest silent jump is the tail: 06-05 08:30 -> horizon.
    assert b["largest_jump"] == {
        "seconds": 1783799.0,
        "from": "2026-06-05T08:30:00+00:00",
        "to": "2026-06-25T23:59:59+00:00",
    }

    # The 1800s action is not a zero-duration action; the request, delivered with
    # no channel delay, IS a same-instant communication — counted, not hidden.
    assert b["zero_duration_actions"] == 0
    assert b["same_timestamp_communications"] == 1

    # Wake novelty (the ACT-8 measurement): the first wake carried the request —
    # materially new; the revisit/deadline wake carried nothing new and is flagged.
    assert b["wake_ups"]["total"] == 2
    assert b["wake_ups"]["with_new_information"] == 1
    assert b["wake_ups"]["without_new_information"] == 1
    assert b["wake_ups"]["repeated_without_new_information"] == 1
    (flag,) = b["wake_ups"]["flagged_repeats"]
    assert flag["actor_id"] == "recipient"
    assert flag["branch_time"] == "2026-06-05T08:00:00+00:00"
    # ...and the repeat is attributed to the cause that produced it: the actor's own
    # revisit condition and the node deadline are both moments arriving, not the world
    # claiming something reached it.
    assert flag["cause_class"] == "calendar_driven"
    assert b["wake_ups"]["repeated_without_new_information_by_cause"] == {
        "calendar_driven": 1,
        "information_driven": 0,
        "unclassified": 0,
    }

    # The novelty trail makes both verdicts auditable, wake by wake.
    trail = b["wake_ups"]["trail"]
    assert [t["materially_new_information"] for t in trail] == [True, False]
    assert [t["cause_class"] for t in trail] == ["information_driven", "calendar_driven"]

    # Every action accounted for; every message accounted for, in ONE unit. The single
    # request was addressed to one recipient, so one send event is one (message,
    # recipient) pair, which is what delivered/noticed/missed have always counted. The
    # terminal's own result_recorded line is not a message and does not appear here.
    assert b["actions"] == {"started": 1, "completed": 1, "failed": 0, "rejected": 0}
    assert b["messages"] == {
        "send_events": 1,
        "sent": 1,
        "delivered": 1,
        "undelivered": 0,
        "noticed": 1,
        "missed": 0,
    }

    # The request's arrival is the one non-actor process update; this branch has no
    # uncertainty, so nothing of its own was released; the terminal was checked once
    # before anything ran and recorded once at the end.
    assert b["process_updates"] == 1
    assert b["scenario_releases"] == 0
    assert b["terminal_checks"] == {"pre_simulation": 1, "final_recorded": 1}

    # Run totals are the branch totals for a one-branch run.
    r = report["run"]
    assert r["branch_count"] == 1
    assert r["distinct_timestamps"] == 4
    assert r["largest_jump_seconds"] == 1783799.0
    assert r["wake_ups"]["repeated_without_new_information"] == 1
    assert r["wake_ups"]["repeated_without_new_information_by_cause"] == {
        "calendar_driven": 1,
        "information_driven": 0,
        "unclassified": 0,
    }
    assert r["actions"] == b["actions"]
    assert r["messages"] == b["messages"]
    # The content key ACT-8 rests on ships its own limits with every report.
    assert report["known_limitations"]["wake_novelty_content_key"]


def test_zero_duration_actions_are_counted() -> None:
    data = single_response_world()
    data["world_spec"]["actions"][0]["duration_seconds"] = 0

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["current_action"] is None:
            return act("send_reply", {"answer": "yes"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    report = compute_temporal_report(run(compiled, gw, seed=0))
    b = report["branches"]["baseline"]
    assert b["zero_duration_actions"] == 1
    assert b["actions"]["started"] == 1 and b["actions"]["completed"] == 1


def test_rejected_and_failed_actions_are_counted_separately() -> None:
    """A refusal is not a failure and neither is a completion; the report keeps the
    action lifecycle's four ends apart."""

    # Rejected: an intention outside the permitted parameter choices.
    data = single_response_world()
    first: list[str] = []

    def reject_then_wait(ctx: dict[str, Any]) -> dict[str, Any]:
        if not first:
            first.append("x")
            return act("send_reply", {"answer": "maybe"})
        return wait_decision()

    gw = _gateway(reject_then_wait)
    _, compiled = _compile_pair(data, gw)
    b = compute_temporal_report(run(compiled, gw, seed=0))["branches"]["baseline"]
    assert b["actions"] == {"started": 0, "completed": 0, "failed": 0, "rejected": 1}

    # Failed: started honestly, overtaken by the world, failed at completion time.
    data2 = single_response_world()
    data2["world_spec"]["fields"].append(
        {"field_id": "channel_open", "value_type": "bool", "initial": True}
    )
    data2["world_spec"]["actions"][0]["duration_seconds"] = 86400 * 3
    data2["world_spec"]["actions"][0]["completion_conditions"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["channel_open"]}, True],
    }
    data2["world_spec"]["external_processes"] = [
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

    def act_once(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["current_action"] is None and not ctx.get("observations"):
            return act("send_reply", {"answer": "yes"})
        return wait_decision()

    gw2 = _gateway(act_once)
    _, compiled2 = _compile_pair(data2, gw2)
    b2 = compute_temporal_report(run(compiled2, gw2, seed=0))["branches"]["baseline"]
    assert b2["actions"]["started"] == 1
    assert b2["actions"]["failed"] == 1
    assert b2["actions"]["completed"] == 0


def test_the_report_is_computed_per_branch_and_totalled() -> None:
    """Two branches with different trajectories produce different counters, and the
    run block is their sum."""

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

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        signal = float(ctx.get("observed_fields", {}).get("external_signal", 0) or 0)
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold" if signal < 5 else "change"})
        if signal > 5 and ctx["actor_id"] == "member_0":
            return act("circulate_note", {"text": "the reading is high"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    report = compute_temporal_report(run(compiled, gw, seed=0))

    assert set(report["branches"]) == {"sc_external_signal:high", "sc_external_signal:low"}
    high = report["branches"]["sc_external_signal:high"]
    low = report["branches"]["sc_external_signal:low"]
    # Only the high branch had a note circulated: one send event, reaching the four
    # other members — four (message, recipient) pairs, the same unit `delivered`
    # counts. The note's accompanying note_circulated record is not a message.
    assert low["messages"] == {
        "send_events": 0,
        "sent": 0,
        "delivered": 0,
        "undelivered": 0,
        "noticed": 0,
        "missed": 0,
    }
    assert high["messages"]["send_events"] == 1
    assert high["messages"]["sent"] == 4
    assert high["messages"]["delivered"] == high["messages"]["sent"]
    assert high["wake_ups"]["total"] > low["wake_ups"]["total"]
    # Each branch released its own hypothesis once, at the release date — counted as
    # branch construction, never as a process update. The one process update per
    # branch is the compiled external occurrence.
    for branch in (high, low):
        assert branch["scenario_releases"] == 1
        assert branch["process_updates"] == 1
    run_block = report["run"]
    assert run_block["branch_count"] == 2
    assert run_block["messages"]["sent"] == high["messages"]["sent"] + low["messages"]["sent"]
    assert run_block["scenario_releases"] == 2
    assert run_block["process_updates"] == 2
    assert run_block["wake_ups"]["total"] == high["wake_ups"]["total"] + low["wake_ups"]["total"]


# ---------------------------------------------------------------------------
# ACT-8 wake novelty: noticed-based, content-keyed, never self-contaminated
# ---------------------------------------------------------------------------


def test_a_wake_between_delivery_and_notice_is_not_credited_with_unread_mail() -> None:
    """Adversary probe shape: the reviewer glances at its calendar at 12:10 — after
    the request was DELIVERED (12:00) but before it is NOTICED (12:30). The diary
    wake must not be credited with the unread message, and the 12:30 wake at which
    the actor actually reads it must be the one credited — never flagged as a
    repeat."""

    import test_actor_lifecycles as act_lc

    data = act_lc.two_actor_exchange_world()
    data["world_spec"]["process"]["nodes"].append(
        {
            "node_id": "reviewer_diary",
            "stage": "correspondence",
            "at": "2026-05-20T12:10:00+00:00",
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
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    b = compute_temporal_report(result)["branches"]["baseline"]

    by_wake = {(t["actor_id"], t["branch_time"]): t for t in b["wake_ups"]["trail"]}
    diary = by_wake[("reviewer", "2026-05-20T12:10:00+00:00")]
    reading = by_wake[("reviewer", "2026-05-20T12:30:00+00:00")]
    assert diary["materially_new_information"] is False, (
        "a wake was credited with information the actor had not noticed"
    )
    assert reading["materially_new_information"] is True, (
        "the wake that actually read the message was not credited"
    )
    assert not any(
        f["actor_id"] == "reviewer" and f["branch_time"] == "2026-05-20T12:30:00+00:00"
        for f in b["wake_ups"]["flagged_repeats"]
    ), "the wake that truly brought new information was flagged as a repeat"


def test_byte_identical_redelivered_content_is_not_materially_new() -> None:
    """Adversary probe shape: the same reminder sent twice under fresh event ids.
    Novelty is keyed on content, so the wake noticing the second, identical
    reminder is a repeat without materially new information — an id-keyed rule
    would credit it and a nagging cascade would never be flagged."""

    data = single_response_world()
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "nagger",
            "description": "the same reminder is sent twice",
            "occurrences": [
                {
                    "at": f"2026-06-0{day}T09:00:00+00:00",
                    "description": "reminder",
                    "effects": [
                        {
                            "op": "deliver_information",
                            "to": ["recipient"],
                            "text": "REMINDER: please reply",
                        }
                    ],
                }
                for day in (1, 3)
            ],
        }
    ]

    gw = _gateway(lambda ctx: wait_decision("never acting"))
    _, compiled = _compile_pair(data, gw)
    b = compute_temporal_report(run(compiled, gw, seed=0))["branches"]["baseline"]

    trail = {t["branch_time"]: t for t in b["wake_ups"]["trail"]}
    first = trail["2026-06-01T09:00:00+00:00"]
    second = trail["2026-06-03T09:00:00+00:00"]
    assert first["materially_new_information"] is True
    assert second["materially_new_information"] is False, (
        "byte-identical re-delivered content was counted as materially new"
    )
    assert any(
        f["branch_time"] == "2026-06-03T09:00:00+00:00" for f in b["wake_ups"]["flagged_repeats"]
    )


def test_the_reconsideration_after_a_rejection_counts_as_materially_new() -> None:
    """The refusal the world hands back IS new information: the ACT-7
    reconsideration wake notices the rejection and must never be flagged as a
    repeat (a live OPEC+ run flagged exactly this wake). And the record of the
    decision that CAUSED the rejection must not list the rejection among what had
    been delivered to it — the snapshot is taken before the decision's own
    consequences apply."""

    data = single_response_world()
    attempts: list[str] = []

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if WAKE_OWN_ACTION in str(ctx["why_you_are_deciding_now"]["trigger"]):
            return act("send_reply", {"answer": "no"})
        if not attempts:
            attempts.append("x")
            return act("send_reply", {"answer": "maybe"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)

    rejected = next(d for d in result.actor_decisions if d.validation_status == "rejected")
    (rejection_ev,) = [e for e in result.event_ledger if e.kind == "action_rejected"]
    assert rejection_ev.event_id not in rejected.delivered_observation_ids, (
        "the decision's record credits it with its own consequence"
    )

    b = compute_temporal_report(result)["branches"]["baseline"]
    recon = next(t for t in b["wake_ups"]["trail"] if WAKE_OWN_ACTION in t["wake_reason"])
    assert recon["materially_new_information"] is True, (
        "the reconsideration wake — which noticed the world's refusal — was not "
        "credited with new information"
    )
    assert rejection_ev.event_id in recon["noticed_event_ids"]
    assert not any(WAKE_OWN_ACTION in f["wake_reason"] for f in b["wake_ups"]["flagged_repeats"]), (
        "the required ACT-7 reconsideration was flagged as an inert repeat"
    )


def test_the_terminal_result_line_is_never_counted_as_a_message() -> None:
    """M1(a): the block counts the ENGINE's communications, so a run in which nobody
    said anything reports zero messages sent.

    ``_finalize`` writes one ``create_event`` ``result_recorded`` line per branch as
    the terminal's own record. It is addressed to nobody and is never propagated, so
    it produces no delivery — but while the report counted ``create_event`` as a
    communication it was counted as a message *sent*, inflating the counter by exactly
    the branch count on every run. A live four-branch run reported 4 messages sent
    having sent none, which is the number a reader would use to judge COM-1.
    """

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

    # Nobody circulates anything: positions are recorded, which is a record, not a
    # message. The only create_event in the run is the terminal's own line.
    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    report = compute_temporal_report(result)

    terminal_lines = [
        e
        for e in result.event_ledger
        if e.kind == "create_event"
        and str(e.payload_dict.get("event_type", "")) == "result_recorded"
    ]
    assert len(terminal_lines) == report["run"]["branch_count"] == 2, (
        "probe shape lost: the run must emit one terminal line per branch"
    )
    assert report["run"]["terminal_checks"]["final_recorded"] == len(terminal_lines)
    assert report["run"]["messages"] == {
        "send_events": 0,
        "sent": 0,
        "delivered": 0,
        "undelivered": 0,
        "noticed": 0,
        "missed": 0,
    }, "the terminal's own result line was counted as a message somebody sent"


def test_a_missed_commitment_is_counted_as_a_missed_message() -> None:
    """M1(b): ``update_commitment`` is a communication to the engine — it wakes a
    recipient and it generates delivery records — so it must be one to the report.

    Dropping it made a genuinely missed commitment vanish: here the reviewer commits
    to something, the commitment reaches the proposer, and the proposer never reads it
    before the horizon. That is exactly what ``missed`` exists to show, and while the
    block counted a different kind set it showed zero.
    """

    import test_actor_lifecycles as act_lc

    data = act_lc.two_actor_exchange_world()
    answer = data["world_spec"]["actions"][1]
    assert answer["action_id"] == "send_answer"
    # The reviewer's answer is an undertaking recorded, not a note sent: the only
    # message it produces is the commitment itself, so `missed` isolates it.
    answer["effects"] = [
        {
            "op": "update_commitment",
            "to": ["proposer"],
            "text": "I will stand behind the figure I answered",
            "tag": "answer",
        },
        {
            "op": "append_record",
            "collection": "answers",
            "key": "$actor",
            "value": "$param.answer",
        },
    ]
    # ...and it is never read: the notice falls beyond the question's horizon.
    answer["notice_delay_seconds"] = 86400 * 60

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["actor_id"] == "proposer":
            if ctx["current_action"] is None and not ctx["observations"]:
                return act("send_request", {"text": "please confirm the figure by Friday"})
            return wait_decision("waiting")
        if any("please confirm" in o["summary"] for o in ctx["observations"]):
            return act("send_answer", {"answer": "yes"})
        return wait_decision("nothing has reached me")

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    world = result.final_worlds["baseline"]

    (commitment,) = [e for e in result.event_ledger if e.kind == "update_commitment"]
    deliveries = [d for d in world.deliveries if d.event_id == commitment.event_id]
    assert deliveries, "probe shape lost: the commitment reached nobody"
    assert all(d.noticed_at is None for d in deliveries), (
        "probe shape lost: the commitment was read after all"
    )

    m = compute_temporal_report(result)["branches"]["baseline"]["messages"]
    assert m["missed"] == len(deliveries) == 1, (
        f"a delivered-but-never-read commitment is absent from `missed`: {m}"
    )
    assert m["send_events"] == 2, f"the commitment is not counted as a message at all: {m}"


def test_sent_and_delivered_are_counted_in_the_same_unit() -> None:
    """M1(c): every counter in the block is per (message, recipient).

    ``sent`` used to count events while delivered/noticed/missed counted delivery
    records, so a world in which one note reaches four people reported more delivered
    than sent — "16 of 5 messages delivered" on the adversary's probe. The arithmetic
    now closes: sent = delivered + undelivered, delivered = noticed + missed, and the
    message count itself is kept under its own name.
    """

    data = scheduled_multiparty_world()

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        if ctx["actor_id"] in ("member_0", "member_1"):
            return act("circulate_note", {"text": f"a note from {ctx['actor_id']}"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    m = compute_temporal_report(run(compiled, gw, seed=0))["branches"]["baseline"]["messages"]

    assert m["send_events"] >= 2, f"probe shape lost: too few messages sent: {m}"
    assert m["delivered"] > m["send_events"], (
        "probe shape lost: this world must reach more recipients than it sends "
        f"messages, or it cannot detect a unit mismatch: {m}"
    )
    assert m["sent"] == m["delivered"] + m["undelivered"], f"sent is not per-recipient: {m}"
    assert m["delivered"] == m["noticed"] + m["missed"], f"deliveries do not close: {m}"


# ---------------------------------------------------------------------------
# TMP-4 / FD-7: a dated future release fires at its release time, not at t0
# ---------------------------------------------------------------------------


def test_a_future_release_is_not_applied_at_seed_time() -> None:
    """The BoE defect class: `release_at` in the future must mean the branch state
    honestly lacks the value until the release fires — the world before the release
    shows the pre-release level, the clock does not leap to the release date, and
    the hypothesis still lands (after any compiled placeholder occurrence)."""

    data = scheduled_multiparty_world()
    data["uncertainties"] = [
        {
            "variable": "external_signal",
            "why_unknown": "the measurement is published after the cutoff",
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

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    release_time = datetime.fromisoformat("2026-06-09T12:00:00+00:00")

    for branch_id, world in result.final_worlds.items():
        value = 9.0 if branch_id.endswith("high") else 1.0
        # The branch's hypothesis release exists, and it fired AT its release time.
        hyp = [
            e
            for e in world.event_history
            if e.kind == "release_data"
            and dict(e.payload_dict.get("fields") or {}).get("external_signal") == value
        ]
        assert hyp, f"{branch_id}: the branch hypothesis never entered the world"
        assert all(e.time == release_time for e in hyp)
        # The value the branch is hypothesizing about was NOT in world state before
        # the release: every earlier event ledger state knew only the initial 3.5.
        pre_release_releases = [
            e for e in world.event_history if e.kind == "release_data" and e.time < release_time
        ]
        assert not pre_release_releases, f"{branch_id}: a release fired before its date"
        # And the hypothesis survived the compiled baseline occurrence at the same
        # instant: the branch condition, not the placeholder, is the world's value.
        assert world.get_field("external_signal") == value

    # The seed did not drag the clock to the release date: the preparation node
    # fired at its compiled time, and the actors deciding there saw the honest
    # pre-release level.
    prep = [d for d in result.actor_decisions if d.stage == "preparation"]
    assert prep
    first = min(datetime.fromisoformat(d.branch_time) for d in prep)
    assert first == datetime.fromisoformat("2026-06-01T09:00:00+00:00")
    at_prep_time = [d for d in prep if d.branch_time == "2026-06-01T09:00:00+00:00"]
    assert at_prep_time
    for d in at_prep_time:
        observed = d.decision_context.get("observed_fields", {})
        assert observed.get("external_signal") == 3.5, (
            f"{d.branch_id}/{d.actor_id} saw the future: {observed}"
        )


def _two_dated_uncertainties(early: str | None, late: str) -> list[dict[str, Any]]:
    """Two uncertainties over two different fields, released at two different moments
    (``early=None`` means the evidence never established a timing for the first)."""

    def variable(name: str, release_at: str | None, high: float, low: float) -> dict[str, Any]:
        return {
            "variable": name,
            "why_unknown": f"{name} is published after the cutoff",
            "reversal_capable": True,
            "release_at": release_at,
            "outcomes": [
                {
                    "value": "high",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [[name, high]],
                },
                {
                    "value": "low",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [[name, low]],
                },
            ],
        }

    return [variable("early_signal", early, 9.0, 1.0), variable("late_signal", late, 8.0, 2.0)]


def test_each_uncertainty_becomes_public_at_its_own_release_date() -> None:
    """TMP-4/FD-7 (probe_release_merge): a branch is not one announcement.

    Two uncertainties dated nineteen days apart. Collapsing a branch's releases to one
    moment — historically ``release_at=min(releases)``, one merged
    ``KIND_SCENARIO_RELEASE`` per branch — published the later value on the earlier
    date, giving every actor nineteen simulated days in which to decide on information
    that did not exist yet. Each release must fire at its own moment, carrying only
    the fields that release reveals.
    """

    early = datetime.fromisoformat("2026-06-01T09:00:00+00:00")
    late = datetime.fromisoformat("2026-06-20T09:00:00+00:00")
    data = scheduled_multiparty_world()
    data["uncertainties"] = _two_dated_uncertainties(early.isoformat(), late.isoformat())

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    assert len(result.final_worlds) == 4, "the two uncertainties did not cross into 4 branches"

    for branch_id, world in result.final_worlds.items():
        want_early = 9.0 if "early_signal:high" in branch_id else 1.0
        want_late = 8.0 if "late_signal:high" in branch_id else 2.0
        hypotheses = [
            (e.time, dict(e.payload_dict.get("fields") or {}))
            for e in world.event_history
            if e.kind == "release_data" and "branch_conditions" in e.payload_dict
        ]
        assert [t for t, _ in hypotheses] == [early, late], (
            f"{branch_id}: the branch's releases did not fire at their own dates: "
            f"{[t.isoformat() for t, _ in hypotheses]}"
        )
        # Each release carries ONLY what it reveals — the later field is absent from
        # the earlier announcement, which is what makes the run honest rather than
        # merely differently timed.
        assert hypotheses[0][1] == {"early_signal": want_early}
        assert hypotheses[1][1] == {"late_signal": want_late}
        assert world.get_field("early_signal") == want_early
        assert world.get_field("late_signal") == want_late

    # Nobody could read the later figure before it existed — checked on what the
    # actors were actually shown, not on world state.
    for d in result.actor_decisions:
        if datetime.fromisoformat(d.branch_time) < late:
            observed = d.decision_context.get("observed_fields", {})
            assert "late_signal" not in observed, (
                f"{d.branch_id}/{d.actor_id}@{d.branch_time} saw a figure published "
                f"on {late.isoformat()}: {observed}"
            )

    # And the §9 report attributes them to branch construction, not to the world's
    # processes: two releases per branch, neither counted as a process update.
    report = compute_temporal_report(result)
    for branch in report["branches"].values():
        assert branch["scenario_releases"] == 2
    assert report["run"]["scenario_releases"] == 8


def test_an_undated_uncertainty_is_never_released_on_its_neighbours_date() -> None:
    """TMP-4/FD-7, the other half: a value whose timing the evidence never established
    is a standing condition of the branch from t0.

    Merging the branch's releases gave such a value the *other* variable's date — the
    exact "dropped at an invented midpoint" the runtime's own docstring forbids. It is
    known from the start, and the dated one still waits for its date.
    """

    late = datetime.fromisoformat("2026-06-20T09:00:00+00:00")
    data = scheduled_multiparty_world()
    data["uncertainties"] = _two_dated_uncertainties(None, late.isoformat())

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)

    for branch_id, world in result.final_worlds.items():
        want_early = 9.0 if "early_signal:high" in branch_id else 1.0
        hypotheses = [
            (e.time, dict(e.payload_dict.get("fields") or {}))
            for e in world.event_history
            if e.kind == "release_data" and "branch_conditions" in e.payload_dict
        ]
        assert len(hypotheses) == 2, f"{branch_id}: expected a standing seed and one release"
        assert hypotheses[0] == (AS_OF, {"early_signal": want_early}), (
            f"{branch_id}: the undated value did not stand from the cutoff: {hypotheses[0]}"
        )
        assert hypotheses[1][0] == late
        assert set(hypotheses[1][1]) == {"late_signal"}

    # The undated value is known at the first decision; the dated one is not.
    first = min(datetime.fromisoformat(d.branch_time) for d in result.actor_decisions)
    for d in result.actor_decisions:
        observed = d.decision_context.get("observed_fields", {})
        if datetime.fromisoformat(d.branch_time) == first:
            assert "early_signal" in observed, (
                f"a standing branch condition was withheld from the actors: {observed}"
            )
        if datetime.fromisoformat(d.branch_time) < late:
            assert "late_signal" not in observed


def test_two_scenario_releases_at_one_instant_are_refused_not_hash_ordered() -> None:
    """Residual guard: the ordering class puts the scenario release last within its
    instant, but it does not order two scenario releases against EACH OTHER — they
    share (at, class, microstep, kind) and fall through to entry-id hash order. The
    branch builder groups a scenario's releases by moment so this cannot arise; the
    schedule refuses it outright rather than leaving the invariant one refactor from
    silently returning."""

    at = datetime.fromisoformat("2026-06-09T12:00:00+00:00")
    first = make_entry(
        at=at,
        kind=KIND_SCENARIO_RELEASE,
        payload={"fields": {"a": 1.0}},
        origin=ORIGIN_EXTERNAL,
        origin_detail="scenario_release:sc_a",
    )
    second = make_entry(
        at=at,
        kind=KIND_SCENARIO_RELEASE,
        payload={"fields": {"b": 2.0}},
        origin=ORIGIN_EXTERNAL,
        origin_detail="scenario_release:sc_b",
    )
    # The same entry pushed twice is a duplicate and is simply ignored.
    assert len(Schedule().push(first).push(first)) == 1
    with pytest.raises(WorldIntegrityError) as exc:
        Schedule().push(first).push(second)
    assert "same instant" in str(exc.value)
    # ...and in one push, too — the check does not depend on the order of arrival.
    with pytest.raises(WorldIntegrityError):
        Schedule().push(first, second)


def test_a_same_instant_deferred_placeholder_never_overwrites_the_hypothesis() -> None:
    """Adversary regression (probe_tie_flip): the compiled world announces at 06-01
    that the value WILL be released at 06-09 — an ``at``-stamped deferred
    ``release_data`` placeholder that lands at the very instant of the branch's own
    scenario release. This exact shape (process id and hypothesis value found by
    the probe's search) used to be decided by schedule-entry-id hash order, and the
    placeholder overwrote the branch's defining condition (final field 3.5 instead
    of the hypothesized 2.0). The scenario release now carries a dedicated ordering
    class and fires strictly last at its instant, for every shape."""

    data = scheduled_multiparty_world()
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "signal_release",
            "description": "announcement now, release later",
            "occurrences": [
                {
                    "at": "2026-06-01T08:00:00+00:00",
                    "description": "the release is scheduled",
                    "effects": [
                        {
                            "op": "release_data",
                            "fields": {"external_signal": 3.5},
                            "at": "2026-06-09T12:00:00+00:00",
                        }
                    ],
                }
            ],
            "evidence_claim_ids": ["c_session"],
        }
    ]
    data["uncertainties"] = [
        {
            "variable": "external_signal",
            "why_unknown": "published after the cutoff",
            "reversal_capable": True,
            "release_at": "2026-06-09T12:00:00+00:00",
            "outcomes": [
                {
                    "value": "low",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 2.0]],
                },
                {
                    "value": "other",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 99.0]],
                },
            ],
        }
    ]

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["stage"] == "session":
            return act("record_position", {"position": "hold"})
        return wait_decision()

    gw = _gateway(decide)
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    release_time = datetime.fromisoformat("2026-06-09T12:00:00+00:00")

    for branch_id, world in result.final_worlds.items():
        want = 2.0 if branch_id.endswith("low") else 99.0
        releases = [e for e in world.event_history if e.kind == "release_data"]
        values = [dict(e.payload_dict.get("fields") or {}).get("external_signal") for e in releases]
        assert all(e.time == release_time for e in releases)
        # The placeholder DID fire — it is real compiled world, not suppressed —
        # and the branch's hypothesis fired strictly after it.
        assert 3.5 in values, f"{branch_id}: the compiled placeholder never fired"
        assert values[-1] == want, (
            f"{branch_id}: the placeholder overwrote the branch's defining condition "
            f"(releases in apply order: {values})"
        )
        assert world.get_field("external_signal") == want


# ---------------------------------------------------------------------------
# Emission: TraceContext.write persists the report; prepare_run_dir clears it
# ---------------------------------------------------------------------------


def _write_real_trace(tmp_path: Path) -> tuple[RunResult, Path]:
    data = single_response_world()
    gw = _gateway(
        lambda ctx: (
            act("send_reply", {"answer": "yes"})
            if ctx["last_decision_time"] is None
            else wait_decision("already replied")
        )
    )
    bundle = build_bundle(data)
    contract, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    forecast = aggregate(
        result.branch_outcomes,
        truncated_mass=result.truncated_mass,
        truncated_reason=result.truncated_reason,
        contract=contract,
        manifest=compiled.manifest,
        trajectory_summaries=result.trajectory_summaries,
        trace_location=str(tmp_path),
        model_call_count=0,
        token_usage=0,
        limitations=(),
    )
    ctx = TraceContext(
        contract=contract,
        evidence_store=bundle.evidence_store,
        as_of=AS_OF,
        bundle=bundle,
        compiled=compiled,
        run_result=result,
        forecast=forecast,
        model_id="programmable",
    )
    ctx.write(tmp_path)
    return result, tmp_path / TEMPORAL_REPORT_FILENAME


def test_trace_write_emits_the_temporal_report(tmp_path: Path) -> None:
    result, path = _write_real_trace(tmp_path)
    assert path.exists(), "TraceContext.write did not emit temporal_report.json"
    on_disk = json.loads(path.read_text())
    assert on_disk == json.loads(canonical_json(compute_temporal_report(result)))
    # The report reflects this real run, not a stub: the reply action is in it.
    assert on_disk["run"]["actions"]["started"] == 1
    assert on_disk["run"]["terminal_checks"]["final_recorded"] == 1


def test_temporal_report_is_registered_and_cleared_as_a_pipeline_artifact(
    tmp_path: Path,
) -> None:
    assert TEMPORAL_REPORT_FILENAME in PIPELINE_ARTIFACTS
    _, path = _write_real_trace(tmp_path)
    assert path.exists()
    prepare_run_dir(
        tmp_path, question="another question", as_of=AS_OF, horizon=HORIZON, mode="semantic"
    )
    assert not path.exists(), "a stale temporal report survived prepare_run_dir"


def test_write_temporal_report_writes_canonical_json(tmp_path: Path) -> None:
    data = single_response_world()
    gw = _gateway(lambda ctx: wait_decision())
    _, compiled = _compile_pair(data, gw)
    result = run(compiled, gw, seed=0)
    write_temporal_report(tmp_path, result)
    text = (tmp_path / TEMPORAL_REPORT_FILENAME).read_text()
    assert text == canonical_json(compute_temporal_report(result)) + "\n"
