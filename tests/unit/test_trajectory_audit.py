"""The post-simulation trajectory auditor, exercised on hand-built runs.

Every check here is mechanical: no gateway, no network, no model. The runs are the
smallest RunResults that exhibit each defect, built directly from the engine's and the
models' own dataclasses so the auditor is tested against the real shapes it will read.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sworldmodel.compiled import CompiledWorld
from sworldmodel.coverage import CompilationCoverageReport, CoverageVerdict
from sworldmodel.engine import ActorDecisionRecord, RunResult
from sworldmodel.evidence import EvidenceStore, EvidenceView
from sworldmodel.grounding import ActorGroundingReport
from sworldmodel.models import (
    BranchOutcome,
    BranchWeight,
    Event,
    IntegrityVerdict,
    RealityManifest,
    ResolutionContract,
    Visibility,
    WeightProvenance,
    make_payload,
)
from sworldmodel.trajectory_audit import (
    TrajectoryAudit,
    audit_trajectory,
    mechanical_trajectory_checks,
)
from sworldmodel.uncertainty import Scenario, ScenarioSet
from sworldmodel.world import WorldState
from sworldmodel.world_review import AuditFinding, _from_findings, _parse_findings
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


# ---------------------------------------------------------------------------
# Builders — the smallest real objects the auditor reads
# ---------------------------------------------------------------------------


def _terminal() -> TerminalExpression:
    return TerminalExpression(
        yes_when=Expr("eq", (Expr("field", ("decision",)), "approved")),
        description="decision is approved",
    )


def _spec(field_cites: tuple[str, ...] = ()) -> WorldSpec:
    return WorldSpec(
        title="t",
        entities=(),
        actors=(),
        fields=(
            FieldSpec(
                field_id="decision",
                value_type="string",
                initial="approved" if field_cites else None,
                evidence_claim_ids=field_cites,
            ),
        ),
        resources=(),
        channels=(),
        documents=(),
        actions=(),
        process=ProcessGraph(),
        terminal=_terminal(),
    )


def _world(branch_id: str, events: tuple[Event, ...] = ()) -> WorldState:
    contract = ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity="s",
        resolution_units="binary",
        terminal=_terminal(),
    )
    return WorldState(
        branch_id=branch_id,
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.SYMMETRIC_IGNORANCE, "test"),
        time=AS_OF,
        contract=contract,
        evidence=EvidenceView(store=EvidenceStore(), as_of=AS_OF),
        event_history=events,
    )


def _scenario(
    sid: str,
    conditions: tuple[tuple[str, str], ...] = (),
    provenance: WeightProvenance = WeightProvenance.SYMMETRIC_IGNORANCE,
) -> Scenario:
    return Scenario(
        scenario_id=sid,
        weight=0.5,
        provenance=provenance,
        provenance_detail="test",
        field_levels=(),
        conditions=conditions,
    )


def _compiled(spec: WorldSpec, scenarios: tuple[Scenario, ...]) -> CompiledWorld:
    return CompiledWorld(
        base_world=_world("root"),
        spec=spec,
        scenario_set=ScenarioSet(scenarios=scenarios, truncated_mass=0.0, truncated_reason=""),
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


def _branch(
    bid: str,
    outcome: str | None,
    conditions: tuple[tuple[str, str], ...] = (),
) -> BranchOutcome:
    return BranchOutcome(
        branch_id=bid,
        parent_lineage=("root",),
        weight=0.5,
        resolved=outcome is not None,
        outcome=outcome,
        unresolved_reason=None if outcome else "the process never reached its end",
        truncated=False,
        key_conditions=conditions,
    )


def _event(
    branch_id: str,
    kind: str,
    payload: dict[str, Any],
    *,
    event_id: str,
    at: datetime = AS_OF,
    actor_id: str | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        branch_id=branch_id,
        time=at,
        kind=kind,
        actor_id=actor_id,
        target_ids=(),
        payload=make_payload(payload),
        visibility=Visibility.PUBLIC,
    )


def _decision(
    branch_id: str = "b1",
    actor_id: str = "a1",
    intent: dict[str, object] | None = None,
    prompt: str = "",
) -> ActorDecisionRecord:
    return ActorDecisionRecord(
        branch_id=branch_id,
        actor_id=actor_id,
        branch_time=AS_OF.isoformat(),
        stage="initial",
        wake_reason="process_opportunity",
        wake_detail="",
        trigger_event_ids=[],
        delivered_observation_ids=[],
        noticed_observation_ids=[],
        retrieved_memory_ids=[],
        plan_before=None,
        plan_after=None,
        plan_disposition="continue",
        state_before={},
        state_after={},
        decision_context={"rendered_prompt": prompt} if prompt else {},
        intent=intent or {"mode": "wait", "action_id": ""},
        validation_status="executed",
        validation_reason="",
        event_ids=[],
        world_version_at_decision=0,
        prompt_hash="",
        model="",
        tokens_out=0,
    )


def _run(
    branches: tuple[BranchOutcome, ...],
    decisions: tuple[ActorDecisionRecord, ...] = (),
    ledger: tuple[Event, ...] = (),
    final_worlds: dict[str, WorldState] | None = None,
) -> RunResult:
    return RunResult(
        branch_outcomes=branches,
        trajectory_summaries=(),
        event_ledger=list(ledger),
        actor_decisions=list(decisions),
        final_worlds=final_worlds or {},
        truncated_mass=0.0,
        truncated_reason="",
    )


def _by_key(audit_or_findings: TrajectoryAudit | tuple[AuditFinding, ...]) -> dict[str, list[str]]:
    findings = (
        audit_or_findings.findings
        if isinstance(audit_or_findings, TrajectoryAudit)
        else audit_or_findings
    )
    out: dict[str, list[str]] = {}
    for f in findings:
        out.setdefault(f.key, []).append(f.severity)
    return out


def _produced_world(bid: str) -> WorldState:
    """A world whose terminal term was actually written, at two distinct times."""

    return _world(
        bid,
        events=(
            _event(
                bid,
                "set_field",
                {"field": "decision", "value": "approved"},
                event_id=f"{bid}-e1",
            ),
            _event(
                bid,
                "create_event",
                {"event_type": "result_recorded", "text": "done"},
                event_id=f"{bid}-e2",
                at=LATER,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Mechanical checks
# ---------------------------------------------------------------------------


def test_repeated_equivalent_calls_are_flagged_high() -> None:
    intent: dict[str, object] = {"mode": "compiled_action", "action_id": "vote", "params": {}}
    run = _run(
        branches=(_branch("b1", "NO"),),
        decisions=(
            _decision(intent=dict(intent)),
            _decision(intent=dict(intent)),
            _decision(actor_id="a2", intent={"mode": "wait", "action_id": ""}),
        ),
    )
    compiled = _compiled(_spec(), (_scenario("b1"),))
    severities = _by_key(mechanical_trajectory_checks(compiled, run))
    assert severities["repeated_equivalent_calls"] == ["HIGH"]


def test_distinct_intents_are_not_flagged() -> None:
    run = _run(
        branches=(_branch("b1", "NO"),),
        decisions=(
            _decision(intent={"mode": "compiled_action", "action_id": "vote"}),
            _decision(intent={"mode": "wait", "action_id": ""}),
        ),
    )
    compiled = _compiled(_spec(), (_scenario("b1"),))
    severities = _by_key(mechanical_trajectory_checks(compiled, run))
    assert severities["repeated_equivalent_calls"] == ["PASS"]


def test_a_yes_with_unproduced_lineage_is_critical_and_the_run_invalid() -> None:
    # The branch resolves YES, but its final world contains no event writing the
    # terminal term and the spec cites no evidence establishing it: nothing produced it.
    run = _run(
        branches=(_branch("b1", "YES"),),
        final_worlds={"b1": _world("b1")},
    )
    compiled = _compiled(_spec(), (_scenario("b1"),))

    severities = _by_key(mechanical_trajectory_checks(compiled, run))
    assert severities["unproduced_yes"] == ["CRITICAL"]

    audit = audit_trajectory(compiled, run, None, question="q")
    assert audit.classification == "invalid"
    assert audit.error == ""
    assert "unproduced_yes" in _by_key(audit)


def test_a_resolved_run_with_zero_actor_calls_is_operational() -> None:
    run = _run(
        branches=(_branch("b1", "YES"),),
        ledger=tuple(_produced_world("b1").event_history),
        final_worlds={"b1": _produced_world("b1")},
    )
    compiled = _compiled(_spec(), (_scenario("b1"),))
    audit = audit_trajectory(compiled, run, None, question="q")
    assert audit.classification == "operational_process_simulation"
    # The produced YES is clean and time advanced, so nothing mechanical blocks.
    assert not [f for f in audit.findings if f.severity in ("CRITICAL", "HIGH")]


def test_a_run_where_nothing_resolved_is_unresolved() -> None:
    run = _run(branches=(_branch("b1", None), _branch("b2", None)))
    compiled = _compiled(_spec(), (_scenario("b1"), _scenario("b2")))
    audit = audit_trajectory(compiled, run, None, question="q")
    assert audit.classification == "unresolved"


def test_a_resolved_run_with_actor_calls_is_genuine() -> None:
    run = _run(
        branches=(_branch("b1", "YES"),),
        decisions=(_decision(intent={"mode": "compiled_action", "action_id": "vote"}),),
        ledger=tuple(_produced_world("b1").event_history),
        final_worlds={"b1": _produced_world("b1")},
    )
    compiled = _compiled(_spec(), (_scenario("b1"),))
    audit = audit_trajectory(compiled, run, None, question="q")
    assert audit.classification == "genuine_actor_simulation"
    d = audit.as_dict()
    assert d["classification"] == "genuine_actor_simulation"
    assert {f["key"] for f in d["findings"]} >= {"unproduced_yes", "time_advanced"}


def test_an_evidence_established_terminal_is_a_factual_resolution() -> None:
    # The spec cites evidence for the terminal term's initial value and nothing at
    # runtime wrote it: the record answered the question before the window opened.
    run = _run(
        branches=(_branch("b1", "YES"),),
        final_worlds={"b1": _world("b1")},
    )
    compiled = _compiled(_spec(field_cites=("c1",)), (_scenario("b1"),))
    audit = audit_trajectory(compiled, run, None, question="q")
    assert audit.classification == "factual_resolution"


def test_outcomes_that_mirror_symmetric_ignorance_conditions_are_flagged() -> None:
    branches = (
        _branch("b_up", "YES", conditions=(("market", "up"),)),
        _branch("b_down", "NO", conditions=(("market", "down"),)),
    )
    run = _run(
        branches=branches,
        ledger=tuple(_produced_world("b_up").event_history),
        final_worlds={"b_up": _produced_world("b_up"), "b_down": _produced_world("b_down")},
    )
    compiled = _compiled(
        _spec(),
        (
            _scenario("b_up", conditions=(("market", "up"),)),
            _scenario("b_down", conditions=(("market", "down"),)),
        ),
    )
    findings = mechanical_trajectory_checks(compiled, run)
    flagged = [f for f in findings if f.key == "result_equals_initialization"]
    assert [f.severity for f in flagged] == ["HIGH"]
    assert flagged[0].finding == "the forecast repeats its initialization"

    # The same outcomes over empirically grounded weights are not the prior read back.
    grounded = _compiled(
        _spec(),
        (
            _scenario(
                "b_up",
                conditions=(("market", "up"),),
                provenance=WeightProvenance.MARKET_SURVEY,
            ),
            _scenario(
                "b_down",
                conditions=(("market", "down"),),
                provenance=WeightProvenance.MARKET_SURVEY,
            ),
        ),
    )
    severities = _by_key(mechanical_trajectory_checks(grounded, run))
    assert severities["result_equals_initialization"] == ["PASS"]


def test_branch_condition_named_in_a_prompt_before_release_is_flagged_medium() -> None:
    leaky = _decision(prompt="You are in branch job_market_slowing; decide accordingly.")
    run = _run(branches=(_branch("b1", "NO"),), decisions=(leaky,))
    compiled = _compiled(_spec(), (_scenario("b1", conditions=(("job_market", "slowing"),)),))
    severities = _by_key(mechanical_trajectory_checks(compiled, run))
    assert severities["branch_label_leakage"] == ["MEDIUM"]

    # After a release_data event delivered the value, the same prompt is legitimate.
    released = _run(
        branches=(_branch("b1", "NO"),),
        decisions=(leaky,),
        ledger=(
            _event(
                "b1",
                "release_data",
                {"fields": {"job_market": "slowing"}},
                event_id="b1-r1",
                at=AS_OF.replace(day=1, hour=0),
            ),
        ),
    )
    severities = _by_key(mechanical_trajectory_checks(compiled, released))
    assert severities["branch_label_leakage"] == ["PASS"]


def test_a_resolved_branch_frozen_at_one_instant_is_flagged() -> None:
    frozen = (
        _event("b1", "set_field", {"field": "decision", "value": "rejected"}, event_id="e1"),
        _event("b1", "create_event", {"event_type": "x"}, event_id="e2"),
    )
    run = _run(branches=(_branch("b1", "NO"),), ledger=frozen)
    compiled = _compiled(_spec(), (_scenario("b1"),))
    severities = _by_key(mechanical_trajectory_checks(compiled, run))
    assert severities["time_advanced"] == ["MEDIUM"]


def test_the_audit_never_raises() -> None:
    class Broken:
        @property
        def scenario_set(self) -> object:
            raise RuntimeError("malformed")

    audit = audit_trajectory(
        Broken(),  # type: ignore[arg-type]
        _run(branches=()),
        None,
        question="q",
    )
    assert "could not run" in audit.error


# ---------------------------------------------------------------------------
# world_review: parsing model output into findings (pure, no gateway)
# ---------------------------------------------------------------------------


def test_parse_findings_normalizes_and_enforces_the_evidence_rule() -> None:
    findings = _parse_findings(
        {
            "findings": [
                {
                    "key": "terminal_preresolved",
                    "severity": "critical",
                    "finding": "the terminal is already true at the start",
                    "evidence_basis": "field decision starts at the YES value",
                },
                {
                    # A blocking severity with no stated evidence basis must demote to LOW.
                    "key": "numbers_have_evidence",
                    "severity": "HIGH",
                    "finding": "a capacity looks invented",
                    "evidence_basis": "",
                },
                {
                    # Unknown keys are dropped, not invented into the questionnaire.
                    "key": "not_a_question",
                    "severity": "HIGH",
                    "finding": "x",
                    "evidence_basis": "y",
                },
                {
                    "key": "decorative_actors",
                    "severity": "PASS",
                    "finding": "every actor can move the outcome",
                    "evidence_basis": "each actor's actions write terminal terms",
                },
            ]
        }
    )
    by_key = {f.key: f for f in findings}
    assert set(by_key) == {"terminal_preresolved", "numbers_have_evidence", "decorative_actors"}
    assert by_key["terminal_preresolved"].severity == "CRITICAL"
    assert by_key["numbers_have_evidence"].severity == "LOW"
    assert by_key["decorative_actors"].severity == "PASS"


def test_findings_assemble_into_a_backward_compatible_review() -> None:
    findings = (
        AuditFinding("terminal_preresolved", "CRITICAL", "already true", "initial value"),
        AuditFinding("process_is_represented", "HIGH", "process absent", "no producer node"),
        AuditFinding("decorative_actors", "MEDIUM", "one actor is scenery", "no effect path"),
        AuditFinding("representation_scale_is_right", "PASS", "scale fits", "roster matches"),
    )
    review = _from_findings(findings)
    assert review.should_repair
    assert review.failed_blocking == ("terminal_preresolved", "process_is_represented")
    assert ("representation_scale_is_right", True, "scale fits") in review.answers
    assert ("terminal_preresolved", False, "already true") in review.answers
    assert review.concerns == ("decorative_actors: one actor is scenery",)

    d = review.as_dict()
    assert {f["key"]: f["severity"] for f in d["findings"]}["process_is_represented"] == "HIGH"
    assert d["blocking_failures"] == ["terminal_preresolved", "process_is_represented"]
    assert {a["question"]: a["ok"] for a in d["answers"]}["representation_scale_is_right"] is True

    instruction = review.repair_instruction()
    assert "terminal_preresolved [CRITICAL]" in instruction
    assert "evidence basis: no producer node" in instruction
    assert "decorative_actors" not in instruction  # MEDIUM does not block


def test_parse_findings_tolerates_garbage_shapes() -> None:
    assert _parse_findings(None) == ()
    assert _parse_findings({"findings": "not a list"}) == ()
    assert _parse_findings({"findings": [42, {"severity": "HIGH"}]}) == ()
    # An unknown severity is visible but never blocking.
    (f,) = _parse_findings(
        {
            "findings": [
                {
                    "key": "branch_weights_arbitrary",
                    "severity": "SEVERE",
                    "finding": "x",
                    "evidence_basis": "y",
                }
            ]
        }
    )
    assert f.severity == "MEDIUM"


def test_a_factual_resolution_skips_the_realism_review_it_cannot_fail() -> None:
    """A geopolitical run resolved YES from the cited record and the model layer then
    filed five CRITICALs against it — "no actor calls exist", "time advanced in a
    single jump" — attacking the absence of a trajectory the run is not supposed to
    have. When the deterministic classifier says factual_resolution, the realism
    questions are not applicable and no model call is spent on them."""

    run = _run(branches=(_branch("b1", "YES"),), final_worlds={"b1": _world("b1")})
    compiled = _compiled(_spec(field_cites=("c1",)), (_scenario("b1"),))

    class _Exploding:
        def generate(self, request):  # pragma: no cover - must never be reached
            raise AssertionError("the realism review must not run for a factual resolution")

    audit = audit_trajectory(compiled, run, _Exploding(), question="q")
    assert audit.classification == "factual_resolution"
    assert not audit.error
    by_key = {f.key: f for f in audit.findings}
    assert by_key["factual_resolution_basis"].severity == "PASS"
    assert "actor_calls_causally_motivated" not in by_key


def test_a_report_line_survives_an_intentless_wake() -> None:
    """A no-feasible-action wake records intent={}, and the report renderer's hard
    c['mode'] crashed the whole trace write of a completed direct-mode run 559s in —
    an exit-4 diagnosis beside a finished forecast. The line must render for every
    record shape the engine writes."""

    from dataclasses import replace as dc_replace

    from sworldmodel.tracing import _invocation_line

    empty = dc_replace(_decision(), intent={})
    line = _invocation_line(empty)
    assert "intent none" in line
    assert "executed" in line

    normal = _decision(intent={"mode": "act", "action_id": "sign_it"})
    assert "intent act sign_it" in _invocation_line(normal)


def test_a_conjunction_of_conditions_is_caught_like_a_single_separating_variable() -> None:
    """A Bank of England run resolved YES exactly on (job_market=slowing AND
    inflation=other). The check only tested one variable at a time, so it reported PASS
    with the words 'not a pure function of their symmetric-ignorance conditions' — while
    the outcomes were precisely that function, of both variables jointly. An independent
    forensic audit found the stated finding broader than the code that produced it."""

    from sworldmodel.trajectory_audit import mechanical_trajectory_checks

    def cell(job: str, infl: str, outcome: str) -> object:
        return _branch(
            f"sc_job:{job}_infl:{infl}",
            outcome,
            conditions=(("job_market", job), ("inflation", infl)),
        )

    run = _run(
        branches=(
            cell("not slowing", "other", "NO"),
            cell("not slowing", "elevated", "NO"),
            cell("slowing", "other", "YES"),
            cell("slowing", "elevated", "NO"),
        ),
        final_worlds={},
    )
    compiled = _compiled(
        _spec(),
        tuple(
            _scenario(f"sc_job:{j}_infl:{i}")
            for j in ("not slowing", "slowing")
            for i in ("other", "elevated")
        ),
    )
    by_key = {f.key: f for f in mechanical_trajectory_checks(compiled, run)}
    finding = by_key["result_equals_initialization"]
    assert finding.severity == "MEDIUM", "a joint-function outcome must not report PASS"
    assert "condition tuple" in finding.finding


def test_the_merged_ledger_is_namespaced_like_every_other_collection() -> None:
    """_merge prefixed branch ids on outcomes, worlds, diagnostics and decisions but
    not on the event ledger. The mechanical checks key ledger data by branch_id and
    look it up by the outcome's prefixed id, so every lookup missed and two checks
    returned their vacuous PASS on every run — publishing "every resolved branch's
    events span more than one timestamp" for a branch whose ledger held one event, and
    "not a pure function of their symmetric-ignorance conditions" for outcomes that
    were exactly that function. An independent forensic audit found both PASSes false."""

    from sworldmodel.api import _merge

    res = _run(
        branches=(_branch("b1", "YES"),),
        ledger=(_event("b1", "create_event", {}, event_id="e1"),),
        decisions=(_decision(branch_id="b1"),),
        final_worlds={"b1": _world("b1")},
    )
    merged = _merge([(1.0, "primary", True, res)])

    outcome_ids = {b.branch_id for b in merged.branch_outcomes}
    ledger_ids = {e.branch_id for e in merged.event_ledger}
    assert outcome_ids == {"primary/b1"}
    assert ledger_ids == outcome_ids, (
        "ledger branch ids must join with outcome branch ids, or every mechanical "
        "check that reads the ledger per branch silently passes"
    )
    assert set(merged.final_worlds) == outcome_ids
    assert {d.branch_id for d in merged.actor_decisions} == outcome_ids


def test_time_advanced_actually_fires_on_a_single_instant_branch() -> None:
    """The check that proves the defect above is closed: a resolved branch whose events
    all share one timestamp must be flagged, not passed."""

    from sworldmodel.trajectory_audit import mechanical_trajectory_checks

    run = _run(
        branches=(_branch("primary/b1", "YES"),),
        ledger=(
            _event("primary/b1", "create_event", {}, event_id="e1", at=AS_OF),
            _event("primary/b1", "create_event", {}, event_id="e2", at=AS_OF),
        ),
        final_worlds={"primary/b1": _world("primary/b1")},
    )
    by_key = {f.key: f for f in mechanical_trajectory_checks(_compiled(_spec(), ()), run)}
    assert by_key["time_advanced"].severity != "PASS", (
        "a branch frozen at one instant must not report PASS"
    )
