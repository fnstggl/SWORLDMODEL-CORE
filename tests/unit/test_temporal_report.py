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

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import AS_OF as AS_OF_S
from _worlds import HORIZON as HORIZON_S
from _worlds import scheduled_multiparty_world, single_response_world
from sworldmodel.compiled import CompiledWorld
from sworldmodel.engine import RunResult, run
from sworldmodel.ids import canonical_json
from sworldmodel.models import ResolutionContract
from sworldmodel.outcomes import aggregate
from sworldmodel.rundir import PIPELINE_ARTIFACTS, prepare_run_dir
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

    # Every action accounted for; every message accounted for.
    assert b["actions"] == {"started": 1, "completed": 1, "failed": 0, "rejected": 0}
    assert b["messages"] == {"sent": 1, "delivered": 1, "noticed": 1, "missed": 0}

    # The request's arrival is the one non-actor process update; the terminal was
    # checked once before anything ran and recorded once at the end.
    assert b["process_updates"] == 1
    assert b["terminal_checks"] == {"pre_simulation": 1, "final_recorded": 1}

    # Run totals are the branch totals for a one-branch run.
    r = report["run"]
    assert r["branch_count"] == 1
    assert r["distinct_timestamps"] == 4
    assert r["largest_jump_seconds"] == 1783799.0
    assert r["wake_ups"]["repeated_without_new_information"] == 1
    assert r["actions"] == b["actions"]
    assert r["messages"] == b["messages"]


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
    # Only the high branch had a note circulated: one more message, more wakes.
    assert high["messages"]["sent"] == low["messages"]["sent"] + 1
    assert high["wake_ups"]["total"] > low["wake_ups"]["total"]
    run_block = report["run"]
    assert run_block["branch_count"] == 2
    assert run_block["messages"]["sent"] == high["messages"]["sent"] + low["messages"]["sent"]
    assert run_block["wake_ups"]["total"] == high["wake_ups"]["total"] + low["wake_ups"]["total"]


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
