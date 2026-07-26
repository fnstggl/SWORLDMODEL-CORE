"""The observability set (OBS-1..4, OBS-7, OBS-8) and the D7 shared replay core.

Everything here is mechanical: hand-built RunResults and bundles, no network, no LLM.
The properties under test are the load-bearing ones:

* initial state + ordered diffs reconstruct every branch's final fields EXACTLY,
  with zero model calls (OBS-7);
* the production trace writer persists the complete observability set at
  trace-write time, and ``prepare_run_dir`` clears all of it (OBS-1..4, OBS-8);
* a delivered-but-unnoticed communication is recorded honestly (noticed=false), and
  a communication no decision ever consumed is ``noticed="unknown"`` — never guessed;
* the replay core's counterfactual answer equals the forensic script's answer on the
  same fixture — one replay implementation, two consumers (D7).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sworldmodel import replaycore
from sworldmodel.compiled import CompiledWorld
from sworldmodel.coverage import CompilationCoverageReport, CoverageVerdict
from sworldmodel.engine import ActorDecisionRecord, BranchDiagnostics, RunResult
from sworldmodel.evidence import EvidenceStore, EvidenceView
from sworldmodel.grounding import ActorGroundingReport
from sworldmodel.models import (
    BranchOutcome,
    BranchWeight,
    Event,
    ForecastResult,
    ForecastStatus,
    IntegrityVerdict,
    RealityManifest,
    ResolutionContract,
    Visibility,
    WeightProvenance,
    make_payload,
)
from sworldmodel.research import ResearchBundle
from sworldmodel.rundir import PIPELINE_ARTIFACTS, prepare_run_dir
from sworldmodel.tracing import TraceContext
from sworldmodel.uncertainty import Scenario, ScenarioSet
from sworldmodel.world import WorldState
from sworldmodel.worldspec import (
    Expr,
    FieldSpec,
    ProcessGraph,
    TerminalExpression,
    WorldSpec,
)

AS_OF = datetime(2026, 5, 1, tzinfo=UTC)
LATER = datetime(2026, 5, 10, tzinfo=UTC)
HORIZON = datetime(2026, 6, 1, tzinfo=UTC)

BRANCH = "primary/sc_cond:x"

OBSERVABILITY_ARTIFACTS = (
    "branch_initial_state.json",
    "state_diffs.jsonl",
    "communications.jsonl",
    "process_transitions.jsonl",
    "run_dossier.html",
)


# ---------------------------------------------------------------------------
# Builders — the smallest real objects the trace writer reads
# ---------------------------------------------------------------------------


def _terminal() -> TerminalExpression:
    return TerminalExpression(
        yes_when=Expr("equals", (Expr("field", ("approved",)), True)),
        description="approved is true",
    )


def _contract() -> ResolutionContract:
    return ResolutionContract(
        question="Will the proposal be approved?",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity="proposal",
        resolution_units="binary",
        terminal=_terminal(),
    )


def _spec() -> WorldSpec:
    return WorldSpec(
        title="approval world",
        entities=(),
        actors=(),
        fields=(
            FieldSpec(field_id="approved", value_type="bool", initial=None),
            FieldSpec(field_id="cond", value_type="string", initial=None),
        ),
        resources=(),
        channels=(),
        documents=(),
        actions=(),
        process=ProcessGraph(),
        terminal=_terminal(),
    )


def _event(
    kind: str,
    payload: dict[str, Any],
    *,
    event_id: str,
    actor_id: str | None = None,
    at: datetime = AS_OF,
    visibility: Visibility = Visibility.PUBLIC,
) -> Event:
    return Event(
        event_id=event_id,
        branch_id=BRANCH,
        time=at,
        kind=kind,
        actor_id=actor_id,
        target_ids=(),
        payload=make_payload(payload),
        visibility=visibility,
    )


def _ledger() -> list[Event]:
    return [
        # The branch's scenario seed: its hypothesis about the uncertain value.
        _event(
            "release_data",
            {
                "fields": {"cond": "x"},
                "branch_conditions": {"cond": "x"},
                "epistemic_type": "hypothesis",
            },
            event_id="ev-seed",
        ),
        # Delivered to bob, never noticed by any recorded decision.
        _event(
            "deliver_information",
            {"text": "the briefing"},
            event_id="ev-deliver",
            visibility=Visibility.PRIVATE,
        ),
        # A public announcement no decision ever consumed.
        _event(
            "create_event",
            {"event_type": "press_release"},
            event_id="ev-orphan",
            at=LATER,
        ),
        # The actor's decisive act.
        _event(
            "set_field",
            {"field": "approved", "value": True},
            event_id="ev-approve",
            actor_id="alice",
            at=LATER,
        ),
    ]


def _decision(
    actor_id: str,
    *,
    delivered: list[str],
    noticed: list[str],
) -> ActorDecisionRecord:
    return ActorDecisionRecord(
        branch_id=BRANCH,
        actor_id=actor_id,
        branch_time=LATER.isoformat(),
        stage="initial",
        wake_reason="information_delivered",
        wake_detail="",
        trigger_event_ids=[],
        delivered_observation_ids=delivered,
        noticed_observation_ids=noticed,
        retrieved_memory_ids=[],
        plan_before=None,
        plan_after=None,
        plan_disposition="continue",
        state_before={},
        state_after={},
        decision_context={},
        intent={"mode": "wait", "action_id": ""},
        validation_status="executed",
        validation_reason="",
        event_ids=[],
        world_version_at_decision=0,
        prompt_hash="",
        model="",
        tokens_out=0,
    )


def _run_result() -> RunResult:
    branch = BranchOutcome(
        branch_id=BRANCH,
        parent_lineage=("root",),
        weight=1.0,
        resolved=True,
        outcome="YES",
        unresolved_reason=None,
        truncated=False,
        key_conditions=(("cond", "x"),),
        records=(("field:approved", "True"), ("field:cond", "x")),
        event_count=4,
    )
    world = WorldState(
        branch_id=BRANCH,
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.SYMMETRIC_IGNORANCE, "test"),
        time=AS_OF,
        contract=_contract(),
        evidence=EvidenceView(store=EvidenceStore(), as_of=AS_OF),
    )
    return RunResult(
        branch_outcomes=(branch,),
        trajectory_summaries=(),
        event_ledger=_ledger(),
        actor_decisions=[
            # bob had ev-deliver delivered and did NOT notice it,
            _decision("bob", delivered=["ev-deliver"], noticed=[]),
            # alice woke without consuming either communication.
            _decision("alice", delivered=[], noticed=[]),
        ],
        final_worlds={BRANCH: world},
        truncated_mass=0.0,
        truncated_reason="",
        diagnostics={BRANCH: BranchDiagnostics()},
    )


def _executed_compilation() -> dict[str, Any]:
    return {
        "world_spec": {
            "fields": [
                {"field_id": "approved", "value_type": "bool", "initial": None},
                {"field_id": "cond", "value_type": "string", "initial": None},
            ],
            "terminal": {
                "yes_when": {"op": "equals", "args": [{"op": "field", "args": ["approved"]}, True]},
                "unresolved_when": False,
                "description": "approved is true",
            },
        }
    }


def _bundle() -> ResearchBundle:
    return ResearchBundle(
        evidence_store=EvidenceStore(),
        spec=_spec(),
        uncertainties=(),
        world_facts=(),
        required_reality_facts=(),
        subject_entity="proposal",
        resolution_units="binary",
        target_outcome="approved",
        expected_participants=None,
        authoritative_sources=(),
        horizon=HORIZON,
        research_plan=("terminal", "<- evidence"),
        as_of=AS_OF,
        executed_compilation=_executed_compilation(),
    )


def _compiled() -> CompiledWorld:
    contract = _contract()
    base = WorldState(
        branch_id="root",
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.SYMMETRIC_IGNORANCE, "test"),
        time=AS_OF,
        contract=contract,
        evidence=EvidenceView(store=EvidenceStore(), as_of=AS_OF),
    )
    return CompiledWorld(
        base_world=base,
        spec=_spec(),
        scenario_set=ScenarioSet(
            scenarios=(
                Scenario(
                    scenario_id=BRANCH,
                    weight=1.0,
                    provenance=WeightProvenance.SYMMETRIC_IGNORANCE,
                    provenance_detail="test",
                    field_levels=(("cond", "x"),),
                    conditions=(("cond", "x"),),
                ),
            ),
            truncated_mass=0.0,
            truncated_reason="",
        ),
        manifest=RealityManifest(
            verified_entities=(),
            verified_roles=(),
            verified_authorities=(),
            verified_rules=(),
            verified_previous_actions=(),
            unresolved_conflicts=(),
            missing_required_facts=(),
            evidence_coverage=1.0,
            integrity_verdict=IntegrityVerdict.VERIFIED,
        ),
        coverage_report=CompilationCoverageReport(
            total_candidates=0,
            material_candidates=0,
            included_candidates=0,
            excluded_candidates=0,
            merged_candidates=0,
            uncertain_candidates=0,
            unresolved_candidates=0,
            missing_material_candidates=(),
            coverage_verdict=CoverageVerdict.COMPLETE,
        ),
        actor_grounding=ActorGroundingReport(profiles=()),
    )


def _forecast(run: RunResult) -> ForecastResult:
    return ForecastResult(
        question="Will the proposal be approved?",
        contract=_contract(),
        integrity_manifest=_compiled().manifest,
        status=ForecastStatus.RESOLVED,
        simulation_probability=1.0,
        lower_bound=1.0,
        upper_bound=1.0,
        resolved_mass=1.0,
        unresolved_mass=0.0,
        resolved_yes_mass=1.0,
        resolved_no_mass=0.0,
        trajectory_summaries=(),
        branch_outcomes=run.branch_outcomes,
        trace_location="(test)",
        limitations=(),
        model_call_count=0,
        token_usage=0,
    )


def _context(run: RunResult) -> TraceContext:
    return TraceContext(
        contract=_contract(),
        evidence_store=EvidenceStore(),
        as_of=AS_OF,
        bundle=_bundle(),
        compiled=_compiled(),
        run_result=run,
        forecast=_forecast(run),
        model_id="test-model",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# OBS-7: initial state + ordered diffs reconstruct the world exactly
# ---------------------------------------------------------------------------


def test_initial_state_plus_ordered_diffs_reconstructs_final_fields_exactly() -> None:
    """The replay identity, on a ledger exercising every field-writing op."""

    events: list[dict[str, Any]] = [
        {
            "event_id": "e1",
            "branch_id": BRANCH,
            "time": "2026-05-01T00:00:00+00:00",
            "kind": "release_data",
            "actor_id": None,
            "payload": {"fields": {"score": 10, "cond": "x"}, "branch_conditions": {"cond": "x"}},
        },
        {
            "event_id": "e2",
            "branch_id": BRANCH,
            "time": "2026-05-02T00:00:00+00:00",
            "kind": "set_field",
            "actor_id": "alice",
            "payload": {"field": "flag", "value": True},
        },
        {
            "event_id": "e3",
            "branch_id": BRANCH,
            "time": "2026-05-03T00:00:00+00:00",
            "kind": "adjust_field",
            "actor_id": None,
            "payload": {"field": "score", "delta": 5},
        },
        {
            "event_id": "e4",
            "branch_id": BRANCH,
            "time": "2026-05-04T00:00:00+00:00",
            "kind": "create_event",
            "actor_id": None,
            "payload": {"event_type": "announcement"},
        },
    ]
    initials = {"score": None, "flag": None}
    initial = replaycore.branch_initial_state(initials, events)
    assert initial == {"score": 10, "flag": None, "cond": "x"}

    diffs = replaycore.derive_state_diffs(
        events,
        initial_by_branch={replaycore.branch_key(BRANCH): initial},
        terminal_fields=("flag",),
    )
    # The seed release repeats the initial state, so it produces no diff; the
    # count-only event changes no field, so it produces none either.
    assert [d["event_id"] for d in diffs] == ["e2", "e3"]
    assert diffs[0]["state_before"] == initial
    assert diffs[0]["triggered_by"] == "alice"
    assert diffs[0]["terminal_read_fields_affected"] == ["flag"]
    assert diffs[1]["diff"] == {"score": {"from": 10, "to": 15.0}}
    assert diffs[1]["triggered_by"] == "process:adjust_field"
    assert diffs[1]["terminal_read_fields_affected"] == []

    # The identity itself: initial + ordered diffs == full replay final state.
    state = dict(initial)
    for d in diffs:
        for name, change in d["diff"].items():
            state[name] = change["to"]
    replayed, counts = replaycore.replay_fields(events)
    assert state == {**initial, **replayed}
    assert state == {"score": 15.0, "flag": True, "cond": "x"}
    assert counts == {"announcement": 1}


def test_replay_accepts_live_event_objects_and_artifact_dicts_identically() -> None:
    """The same replay runs at trace-write time (dataclasses) and from disk (dicts)."""

    live = _ledger()
    as_dicts = [
        {
            "event_id": e.event_id,
            "branch_id": e.branch_id,
            "time": e.time.isoformat(),
            "kind": e.kind,
            "actor_id": e.actor_id,
            "payload": e.payload_dict,
            "visibility": e.visibility.value,
            "evidence_claim_ids": list(e.evidence_claim_ids),
        }
        for e in live
    ]
    assert replaycore.replay_fields(live) == replaycore.replay_fields(as_dicts)
    assert replaycore.fields_written_by_events(live) == replaycore.fields_written_by_events(
        as_dicts
    )
    # Namespaced and bare branch ids join to the same key (FD-16).
    assert replaycore.branch_key("primary/sc_cond:x") == replaycore.branch_key("sc_cond:x")


def test_a_field_never_written_reads_none_and_is_never_invented() -> None:
    fields, counts = replaycore.replay_fields([])
    assert fields == {}
    world = replaycore.ReplayWorld(fields, counts)
    assert world.get_field("anything") is None
    assert world.get_records("anything") == []


def test_terminal_arithmetic_over_an_undetermined_value_replays_as_unresolved() -> None:
    """The engine reports arithmetic over a never-determined value as an honest
    unresolved outcome; the replay must say exactly the same — not NO, not a crash."""

    terminal = {
        "yes_when": {
            "op": "greater_than",
            "args": [
                {
                    "op": "multiply",
                    "args": [
                        {"op": "field", "args": ["known"]},
                        {"op": "field", "args": ["never_set"]},
                    ],
                },
                10,
            ],
        }
    }
    answer = replaycore.counterfactual_outcome(
        [],
        lambda e: True,
        initial={"known": 4.0},
        terminal=terminal,
        rendered={},
    )
    assert answer == "UNRESOLVED"


# ---------------------------------------------------------------------------
# OBS-1..4, OBS-8: the trace writer persists the set; prepare_run_dir clears it
# ---------------------------------------------------------------------------


def test_trace_write_persists_the_complete_observability_set(tmp_path: Path) -> None:
    run = _run_result()
    ctx = _context(run)
    ctx.write(tmp_path)

    for name in OBSERVABILITY_ARTIFACTS:
        assert (tmp_path / name).exists(), f"{name} not written by TraceContext.write"
        assert name in PIPELINE_ARTIFACTS, f"{name} written but never cleared"

    # OBS-1: complete per-branch initial state — initials plus scenario conditions;
    # the never-initialized, never-conditioned field is present and honestly null.
    initial_states = json.loads((tmp_path / "branch_initial_state.json").read_text())
    assert set(initial_states) == {BRANCH}
    entry = initial_states[BRANCH]
    assert entry["conditions"] == {"cond": "x"}
    assert entry["fields"] == {"approved": None, "cond": "x"}

    # OBS-2/OBS-7: state diffs chain from that initial state and reconstruct the
    # branch's final fields exactly; terminal relevance is derived from the spec.
    diffs = _read_jsonl(tmp_path / "state_diffs.jsonl")
    assert [d["event_id"] for d in diffs] == ["ev-approve"]
    assert diffs[0]["state_before"] == entry["fields"]
    assert diffs[0]["terminal_read_fields_affected"] == ["approved"]
    state = dict(entry["fields"])
    for d in diffs:
        for name, change in d["diff"].items():
            state[name] = change["to"]
    replayed, _ = replaycore.replay_fields(run.event_ledger)
    assert state == {**entry["fields"], **replayed} == {"approved": True, "cond": "x"}

    # OBS-4: non-actor transitions with inputs/outputs/evidence.
    procs = _read_jsonl(tmp_path / "process_transitions.jsonl")
    assert [p["event_id"] for p in procs] == ["ev-seed", "ev-deliver", "ev-orphan"]
    assert procs[0]["inputs"] == {"cond": "x"}

    # OBS-8: the dossier is self-contained, rendered from the same reconstruction.
    dossier = (tmp_path / "run_dossier.html").read_text()
    assert "Run dossier" in dossier
    assert "RECONSTRUCTED" in dossier
    assert "Will the proposal be approved?" in dossier
    assert "prefers-color-scheme:dark" in dossier


def test_prepare_run_dir_clears_the_observability_set(tmp_path: Path) -> None:
    ctx = _context(_run_result())
    ctx.write(tmp_path)
    for name in OBSERVABILITY_ARTIFACTS:
        assert (tmp_path / name).exists()

    prepare_run_dir(
        tmp_path,
        question="a different question",
        as_of=AS_OF,
        horizon=HORIZON,
        mode="semantic",
    )
    for name in OBSERVABILITY_ARTIFACTS:
        assert not (tmp_path / name).exists(), f"stale {name} survived prepare_run_dir"


# ---------------------------------------------------------------------------
# OBS-3: the delivery→notice join is honest
# ---------------------------------------------------------------------------


def test_delivered_but_unnoticed_and_unconsumed_communications_are_honest(
    tmp_path: Path,
) -> None:
    ctx = _context(_run_result())
    ctx.write(tmp_path)
    comms = {c["event_id"]: c for c in _read_jsonl(tmp_path / "communications.jsonl")}
    assert set(comms) == {"ev-deliver", "ev-orphan"}

    # Delivered to bob, consumed by a decision, never noticed: false — not guessed.
    delivered = comms["ev-deliver"]
    assert delivered["noticed"] is False
    assert delivered["deliveries"] == [{"actor_id": "bob", "noticed": False, "noticed_at": None}]
    assert delivered["content"] == "the briefing"

    # No recorded decision ever consumed the announcement: unknown — never guessed.
    orphan = comms["ev-orphan"]
    assert orphan["noticed"] == "unknown"
    assert orphan["deliveries"] == []


def test_a_noticed_communication_records_who_and_when() -> None:
    events = _ledger()
    decisions = [_decision("bob", delivered=["ev-deliver"], noticed=["ev-deliver"])]
    comms = {c["event_id"]: c for c in replaycore.extract_communications(events, decisions)}
    assert comms["ev-deliver"]["noticed"] is True
    assert comms["ev-deliver"]["deliveries"] == [
        {"actor_id": "bob", "noticed": True, "noticed_at": LATER.isoformat()}
    ]


# ---------------------------------------------------------------------------
# D7: one replay implementation — the forensic script and the core agree
# ---------------------------------------------------------------------------


def test_replaycore_counterfactual_matches_the_forensic_script(tmp_path: Path) -> None:
    """The forensic script's all-actors-removed answer IS the replay core's answer,
    on the same fixture — produced by the production trace writer itself."""

    ctx = _context(_run_result())
    ctx.write(tmp_path)

    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import forensics

    result = forensics.reconstruct(tmp_path, "fixture")
    assert result["verdict"] == "RECONSTRUCTED"
    (branch,) = result["branches"]
    assert branch["recomputed_outcome"] == "YES"

    # The same counterfactual, straight through the core, on the persisted ledger.
    events = _read_jsonl(tmp_path / "event_ledger.jsonl")
    world = json.loads((tmp_path / "compiled_world.json").read_text())
    initial = replaycore.branch_initial_state(
        replaycore.initial_fields_from_world(world) or {}, events
    )
    core_answer = replaycore.counterfactual_outcome(
        events,
        lambda e: not replaycore.event_actor_id(e),
        initial=initial,
        terminal=replaycore.terminal_ast_from_world(world),
        rendered={},
    )
    assert core_answer == branch["counterfactuals"]["all_actor_output_removed"]
    # And the answer is the honest one: with the actor's act deleted, the terminal
    # reads the never-written field as None and resolves NO — not YES, not a guess.
    assert core_answer == "NO"
