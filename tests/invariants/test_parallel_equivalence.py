"""Serial-versus-parallel equivalence: concurrency is a wall-clock decision only.

Branches, structures and the review/assessment pair run concurrently in production.
Nothing about a trajectory may depend on which other work was running at the time:
these tests execute the same worlds serially and in parallel and require the results
to be identical — probability, branch table, event ledger, actor decisions and the
call log's deterministic serialization. The gateway's identical-request memo carries
the same burden in the other direction: two byte-identical requests racing from
parallel branches must produce ONE provider call whose answer both reuse, or a run's
answers would depend on thread timing.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from _fakes import FixtureResearchBackend, ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import scheduled_multiparty_world
from sworldmodel.config import ForecastConfig
from sworldmodel.engine import run
from sworldmodel.gateway import GatewayRequest, GatewayResponse, ModelGateway
from sworldmodel.ids import canonical_json, prompt_hash
from sworldmodel.models import ResolutionContract
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def _split_world() -> dict:
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
    return data


def _signal_sensitive(ctx: dict) -> dict:
    if ctx["stage"] != "session":
        return wait_decision()
    signal = float(ctx.get("observed_fields", {}).get("external_signal", 0) or 0)
    return act("record_position", {"position": "hold" if signal < 5 else "change"})


def _gateway() -> ProgrammableGateway:
    return ProgrammableGateway(
        {
            "actor_decision": _signal_sensitive,
            "reflect": {"beliefs_update": [], "new_memories": []},
            "world_review": {"findings": []},
            "trajectory_audit": {"findings": []},
            "assess_structure": {
                "is_material": True,
                "reason": "an informal path could also decide it",
                "primary_weight": 0.8,
                "alternatives": [
                    {
                        "structure_id": "informal_path",
                        "what_differs": "an informal path binds instead",
                        "rationale": "the record leaves it open",
                        "weight": 0.2,
                        "provenance": "symmetric_ignorance_assumption",
                        "could_reverse_outcome": True,
                    }
                ],
            },
        }
    )


def _compiled(gw: ProgrammableGateway):
    bundle = build_bundle(_split_world())
    contract = ResolutionContract(
        question="q",
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
        gateway=gw,
        seed=0,
        max_branches=8,
    )


def _run_digest(result: Any) -> str:
    """A canonical serialization of everything a replay reads from a run."""

    return canonical_json(
        {
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "weight": b.weight,
                    "resolved": b.resolved,
                    "outcome": b.outcome,
                    "unresolved_reason": b.unresolved_reason,
                    "conditions": dict(b.key_conditions),
                    "records": dict(b.records),
                }
                for b in result.branch_outcomes
            ],
            "ledger": [
                {"branch": e.branch_id, "kind": e.kind, "time": e.time.isoformat()}
                for e in result.event_ledger
            ],
            "decisions": [
                {
                    "branch": d.branch_id,
                    "actor": d.actor_id,
                    "time": d.branch_time,
                    "intent": d.intent,
                    "status": d.validation_status,
                }
                for d in result.actor_decisions
            ],
            "truncated": result.truncated_mass,
        }
    )


def test_branch_parallelism_is_invisible_in_the_result() -> None:
    gw_serial = _gateway()
    serial = run(_compiled(gw_serial), gw_serial, seed=0, max_concurrent_branches=1)
    gw_parallel = _gateway()
    parallel = run(_compiled(gw_parallel), gw_parallel, seed=0, max_concurrent_branches=4)
    assert _run_digest(serial) == _run_digest(parallel)
    # The same provider work was done, call for call.
    assert gw_serial.call_count == gw_parallel.call_count


def _forecast_payload(result: Any) -> str:
    return canonical_json(
        {
            "probability": result.simulation_probability,
            "source": result.probability_source,
            "yes": result.resolved_yes_mass,
            "no": result.resolved_no_mass,
            "unresolved": result.unresolved_mass,
            "lower": result.lower_bound,
            "upper": result.upper_bound,
            "branches": [
                {"id": b.branch_id, "w": b.weight, "resolved": b.resolved, "outcome": b.outcome}
                for b in result.branch_outcomes
            ],
        }
    )


def test_the_full_pipeline_is_serial_parallel_equivalent() -> None:
    """run_forecast with concurrency 1/1 versus 4/2 — same probability, same branch
    table, same unresolved mass (the unrepresentable alternative refuses identically
    on both paths), and the same deterministic call-log serialization."""

    from sworldmodel.api import run_forecast
    from sworldmodel.tracing import TraceContext

    def _run(branches: int, structures: int) -> tuple[Any, Any, ProgrammableGateway]:
        gw = _gateway()
        config = ForecastConfig(
            gateway=gw,
            research_backend=FixtureResearchBackend(build_bundle(_split_world())),
            max_concurrent_branches=branches,
            max_concurrent_structures=structures,
        )
        result, ctx = run_forecast("q", AS_OF, HORIZON, config)
        return result, ctx, gw

    serial_result, serial_ctx, serial_gw = _run(1, 1)
    parallel_result, parallel_ctx, parallel_gw = _run(4, 2)

    assert _forecast_payload(serial_result) == _forecast_payload(parallel_result)
    assert _run_digest(serial_ctx.run_result) == _run_digest(parallel_ctx.run_result)
    # The deterministic call-log artifact is identical however the threads ran.
    serial_ctx._calls_override = serial_gw.calls
    parallel_ctx._calls_override = parallel_gw.calls
    assert TraceContext.llm_call_lines(serial_ctx) == TraceContext.llm_call_lines(parallel_ctx)


# ---------------------------------------------------------------------------
# The identical-request memo, and its in-flight deduplication
# ---------------------------------------------------------------------------


class _CountingMemoGateway(ModelGateway):
    """A gateway with memoization ON whose provider counts and delays its calls."""

    memoize_identical_requests = True

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.provider_calls = 0
        self.delay = delay

    @property
    def model_id(self) -> str:
        return "counting"

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        with self._lock:
            self.provider_calls += 1
        if self.delay:
            time.sleep(self.delay)
        return GatewayResponse(
            task_kind=request.task_kind,
            data={"echo": request.prompt},
            raw_text=request.prompt,
            model="counting",
            params={},
            seed=request.seed,
            prompt_hash=prompt_hash(request.prompt),
            tokens_in=1,
            tokens_out=1,
        )


def _req(prompt: str) -> GatewayRequest:
    return GatewayRequest(task_kind="actor_decision", prompt=prompt, context={}, seed=7)


def test_a_byte_identical_request_is_answered_once_and_reused() -> None:
    gw = _CountingMemoGateway()
    first = gw.generate(_req("same view"))
    second = gw.generate(_req("same view"))
    third = gw.generate(_req("a different view"))
    assert gw.provider_calls == 2
    assert gw.call_count == 2, "a memo reuse is not a provider call"
    assert gw.memo_hits == 1
    assert second.data == first.data
    assert second.params.get("memo_hit") is True
    assert third.params.get("memo_hit") is None
    # The reuse is on the record: three entries in the call log, one marked.
    assert len(gw.calls) == 3
    # A reuse hands back its own copy: mutating it cannot rewrite the recorded answer.
    second.data["echo"] = "tampered"
    assert gw.generate(_req("same view")).data == {"echo": "same view"}


def test_racing_identical_requests_collapse_to_one_provider_call() -> None:
    """The serial-parallel guarantee at the request level: N identical requests in
    flight at once produce exactly one provider call, and every caller gets that
    call's answer — never a second sample that thread timing decides."""

    gw = _CountingMemoGateway(delay=0.05)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: gw.generate(_req("racing view")), range(8)))
    assert gw.provider_calls == 1
    assert gw.memo_hits == 7
    assert {canonical_json(r.data) for r in responses} == {canonical_json({"echo": "racing view"})}


def test_a_failed_inflight_request_does_not_wedge_the_waiters() -> None:
    from sworldmodel.errors import GatewayError

    class _FailsOnce(_CountingMemoGateway):
        def _generate(self, request: GatewayRequest) -> GatewayResponse:
            with self._lock:
                first = self.provider_calls == 0
            if first:
                with self._lock:
                    self.provider_calls += 1
                raise GatewayError("scripted first failure")
            return super()._generate(request)

    gw = _FailsOnce()
    try:
        gw.generate(_req("view"))
        raise AssertionError("the first call must fail")
    except GatewayError:
        pass
    # The failure released the in-flight slot: the next identical request performs
    # the call itself instead of waiting forever on the failed one.
    assert gw.generate(_req("view")).data == {"echo": "view"}
