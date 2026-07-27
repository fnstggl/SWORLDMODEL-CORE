"""Two properties, from one live 698-second frozen-store run that refused.

That run's own record made two claims about itself that were not true, and both are
the same defect: **a thing that did not happen was written down as a thing that
happened and came back empty.**

1. Its diagnosis named ``archive_coverage_failure`` — *"19 claim(s) stored, none
   admissible at the cutoff … the compiler saw an empty record"*. All nineteen were
   admissible; the frozen backend simply never wrote ``admissible_claim_count``, and
   ``int(trace.get(...) or 0)`` turned a counter nobody wrote into a measured zero.
   The same record also named ``discovery_failure`` — *"no candidate URL was
   discovered at all"* — about a backend that issues no queries by construction. This
   is FD-34, FD-45 and FD-42's shape (a check that did not run reading as a check that
   ran and approved) arriving in the root-cause classifier.

2. The repair record itself has the same failure mode. The repair loop DID consume
   that run's ``actors_ungrounded`` refusal — five attempts, seventeen provider calls —
   and that property is pinned below because it is what "refused" versus "repaired
   into a world" rests on. But three paths reached a refusal while leaving
   ``repair_attempts: []``, which reads exactly like a loop that looked and had
   nothing to do: an attempt that RAISED (recorded only after it returns), a replan
   that declined without saying so, and — the reachability defect proper — a bundle
   assembled outside the frozen backend's guard, so an assembly refusal carried no
   research, was filed as a research failure, and never reached
   ``_replan_initial_compile`` at all.

Everything here runs the real ``run_forecast`` over the real ``FrozenResearchBackend``
on a scripted gateway. No network, no provider.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import sworldmodel.api as api
from _fakes import FixtureResearchBackend, ProgrammableGateway, build_bundle, wait_decision
from _worlds import AS_OF, HORIZON, scheduled_multiparty_world
from sworldmodel.config import ForecastConfig
from sworldmodel.diagnosis import ForecastRefused, RunDiagnosis
from sworldmodel.errors import GatewayError, WorldIntegrityError
from sworldmodel.repair import RepairLog

# ``scripts/frozen_forecast.py`` holds the backend under test; it is a script, not a
# package, so it is reached by path rather than by import path.
_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

AS_OF_DT = datetime.fromisoformat(AS_OF)
HORIZON_DT = datetime.fromisoformat(HORIZON)
QUESTION = "will both members record hold?"

# The trace a frozen-store run writes. Three keys, no research pass of any kind — no
# `admissible_claim_count`, no `attempted_urls`, no `queries`. Verbatim in shape from
# the 698s run's research_trace.json.
FROZEN_TRACE: dict[str, Any] = {
    "frozen_store": True,
    "claim_count": 19,
    "semantic_compilation": {"plan": {}},
}


# --------------------------------------------------------------------------- #
# Scripting
# --------------------------------------------------------------------------- #


def _ungrounded_world() -> dict[str, Any]:
    """The v5 OPEC+ shape: a compiled coalition that is a constructed representative
    carrying no population weight. ``compile_world`` refuses it ``actors_ungrounded``
    with ``recompilable: True`` — a compile-stage refusal that advertises itself as
    repairable, which is the exact case in question."""

    world = copy.deepcopy(scheduled_multiparty_world(members=2, threshold=2))
    for entity in world["world_spec"]["entities"]:
        if entity["entity_id"] == "member_1":
            entity["representation_scale"] = "population_stratum"
            entity["kind"] = "population_group"
    return world


def _gateway() -> ProgrammableGateway:
    gw = ProgrammableGateway(
        {
            "actor_decision": lambda ctx: wait_decision("waiting"),
            "reflect": {"beliefs_update": [], "new_memories": []},
            "world_review": {"findings": []},
        }
    )
    gw.is_live = True  # type: ignore[attr-defined]
    return gw


def _frozen_run(
    monkeypatch: pytest.MonkeyPatch,
    compile_script: Any,
    tmp_path: Path,
) -> tuple[BaseException | None, int]:
    """Run the pipeline exactly as ``scripts/frozen_forecast.py`` does — its real
    backend, the real ``run_forecast`` — with the compiler scripted."""

    import frozen_forecast

    seed = build_bundle(_ungrounded_world())
    ref: dict[str, Any] = {}
    config = ForecastConfig(
        gateway=_gateway(),
        research_backend=frozen_forecast.FrozenResearchBackend(seed.evidence_store, ref),
        compiler_mode="direct",
        max_branches=2,
        trace_dir=tmp_path,
    )
    ref["config"] = config

    calls = {"n": 0}

    def scripted(cfg: Any, question: str, as_of: Any, horizon: Any, view: Any, **kw: Any) -> Any:
        calls["n"] += 1
        return compile_script(calls["n"])

    monkeypatch.setattr(api, "compile_for_mode", scripted)
    monkeypatch.setattr(frozen_forecast, "compile_for_mode", scripted)
    try:
        api.run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, config)
    except BaseException as exc:  # noqa: BLE001 — the refusal is what is under test
        return exc, calls["n"]
    return None, calls["n"]


def _attempts(exc: BaseException | None) -> list[dict[str, object]]:
    log = getattr(exc, "repair_log", None)
    return list(getattr(log, "attempts", []) or [])


# --------------------------------------------------------------------------- #
# 1. The repair path IS reachable for a compile-stage refusal, and says it was.
# --------------------------------------------------------------------------- #


def test_a_recompilable_gate_refusal_is_consumed_by_the_repair_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The standing property, not a regression: this holds before and after.

    It is pinned because everything downstream rests on it. If a compile-stage refusal
    carrying ``recompilable: True`` could not reach repair, then "refused" and
    "repaired into a world" were never different outcomes, and every conclusion drawn
    from a refusal — including "the planner cannot produce this world" — would be
    confounded by a repair that never ran. The live 698s run recorded five attempts on
    exactly this failure; here the same failure records its attempts on a frozen store
    that cannot grow, and the refusal that survives them is the real one.
    """

    exc, compiles = _frozen_run(monkeypatch, lambda n: copy.deepcopy(_ungrounded_world()), tmp_path)

    assert isinstance(exc, ForecastRefused)
    assert exc.stage == "compilation"
    details = getattr(exc.__cause__, "details", {}) or {}
    assert details["failure"] == "actors_ungrounded"
    assert details["recompilable"] is True
    attempts = _attempts(exc)
    assert attempts, "a repairable compile-stage refusal reached a refusal with no attempt on it"
    assert compiles > 1, "the recompile the plan asked for never happened"
    assert [a["failure"] for a in attempts] == ["actors_ungrounded"] * len(attempts)
    assert all(a["plan"] for a in attempts), "an attempt was recorded with no plan behind it"
    assert "repair exhausted" in str(attempts[-1]["outcome"])


def test_a_repair_attempt_that_raises_is_still_on_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The round was recorded only after ``_repair_once`` RETURNED, so an attempt that
    raised — a final refusal re-raised by ``_recompile``, a stop, or any shape outside
    its catch list — left the run reporting ``repair_attempts: []``: a repair that was
    planned, started and paid for, on a record saying it was never tried."""

    def script(n: int) -> Any:
        if n == 1:
            return copy.deepcopy(_ungrounded_world())
        raise AttributeError("'NoneType' object has no attribute 'view'")

    exc, compiles = _frozen_run(monkeypatch, script, tmp_path)

    assert compiles == 2, "the repair recompile did not run, so this proves nothing"
    attempts = _attempts(exc)
    assert len(attempts) == 1, "the attempt that ended the run is missing from its own record"
    assert attempts[0]["failure"] == "actors_ungrounded"
    assert attempts[0]["plan"], "the plan that was acted on is not on the record"
    outcome = str(attempts[0]["outcome"])
    assert "AttributeError" in outcome, outcome
    # And the failure still propagates: recording it must not swallow it.
    assert isinstance(exc, ForecastRefused)


def test_a_bundle_assembly_refusal_reaches_the_compile_stage_replan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``assemble_bundle`` refuses in its own right — a compilation that declares no
    horizon, a world_spec the parser cannot read. It sat OUTSIDE the frozen backend's
    guard, so those refusals reached ``run_forecast`` carrying no research at all:
    ``_checkpoint_partial`` returned False, the run was filed as a *research* failure,
    the whole evidence record went unwritten, and ``_replan_initial_compile`` — which
    exists precisely to give a recompilable compile-stage refusal its registered
    repair — was never reached."""

    def script(n: int) -> Any:
        world = copy.deepcopy(_ungrounded_world())
        world["reality"].pop("horizon", None)
        return world

    exc, _compiles = _frozen_run(monkeypatch, script, tmp_path)

    assert isinstance(exc, ForecastRefused)
    assert exc.stage == "compilation", (
        "a run that compiled a world and could not assemble it did not fail at research"
    )
    assert (tmp_path / "evidence_store.json").exists(), "the evidence record was thrown away"
    assert (tmp_path / "research_trace.json").exists()
    # The replan was reached and said what it did about it, rather than being skipped.
    assert _attempts(exc), "the refusal reached a diagnosis with an empty repair record"


def test_the_replan_records_every_reason_it_declines() -> None:
    """``_replan_initial_compile`` used to return None silently on three paths, so a
    refusal it never looked at produced the same ``repair_attempts: []`` as one it
    looked at and had nothing to do about. That is this file's other defect in the
    repair record itself, and a reader must be able to tell the two apart."""

    bundle = build_bundle(_ungrounded_world())

    def _recompilable() -> WorldIntegrityError:
        exc = WorldIntegrityError(
            "the semantic plan is invalid after validator rounds",
            details={"failure": "semantic_plan_invalid", "recompilable": True},
        )
        exc.partial_evidence_store = bundle.evidence_store  # type: ignore[attr-defined]
        exc.partial_live_trace = {}  # type: ignore[attr-defined]
        return exc

    dead = ForecastConfig(
        gateway=ProgrammableGateway({}),  # is_live is False
        research_backend=FixtureResearchBackend(bundle),
    )

    log = RepairLog()
    assert (
        api._replan_initial_compile("q?", AS_OF_DT, HORIZON_DT, dead, _recompilable(), log) is None
    )
    assert len(log.attempts) == 1
    assert "not live" in str(log.attempts[0]["outcome"])
    assert log.attempts[0]["failure"] == "semantic_plan_invalid"

    final = WorldIntegrityError(
        "the independent reality review abstained",
        details={"failure": "semantic_review_abstained", "recompilable": False},
    )
    final.partial_evidence_store = bundle.evidence_store  # type: ignore[attr-defined]
    log = RepairLog()
    assert api._replan_initial_compile("q?", AS_OF_DT, HORIZON_DT, dead, final, log) is None
    assert len(log.attempts) == 1
    assert "final" in str(log.attempts[0]["outcome"])

    log = RepairLog()
    unnamed = GatewayError("the provider returned nothing usable")
    assert api._replan_initial_compile("q?", AS_OF_DT, HORIZON_DT, dead, unnamed, log) is None
    assert len(log.attempts) == 1
    assert log.attempts[0]["failure"] == "GatewayError"


# --------------------------------------------------------------------------- #
# 2. A run states only what it observed, and says what it could not.
# --------------------------------------------------------------------------- #


def _frozen_diagnosis(gate: str = "actors_ungrounded") -> RunDiagnosis:
    bundle = dc_replace(
        build_bundle(scheduled_multiparty_world(members=2, threshold=2)),
        live_trace=dict(FROZEN_TRACE),
    )
    return RunDiagnosis(
        question="Will the seven participating OPEC+ countries agree a further increase?",
        as_of=AS_OF_DT,
        horizon=HORIZON_DT,
        bundle=bundle,
        failure=WorldIntegrityError(
            "actors are not grounded in cited evidence — simulation refused",
            details={"failure": gate, "recompilable": True},
        ),
        failure_stage="compilation",
        compiler_mode="semantic",
    )


def test_a_counter_nobody_wrote_is_not_a_measured_zero() -> None:
    """``int(trace.get("admissible_claim_count") or 0)``: a backend that never counted
    and a backend that counted nothing produced the same number, and the classifier
    read the second meaning off the first."""

    d = _frozen_diagnosis()
    assert d.extraction()["claims_stored"] > 0
    assert d.extraction()["claims_admissible_at_cutoff"] is None, (
        "an unwritten counter must not be readable as a measurement of zero"
    )

    # A backend that DID count keeps reporting its count, zero included.
    measured = dc_replace(d.bundle, live_trace={**FROZEN_TRACE, "admissible_claim_count": 0})  # type: ignore[arg-type]
    d.bundle = measured
    assert d.extraction()["claims_admissible_at_cutoff"] == 0
    d.bundle = dc_replace(measured, live_trace={**FROZEN_TRACE, "admissible_claim_count": 19})
    assert d.extraction()["claims_admissible_at_cutoff"] == 19


def test_the_archive_cause_needs_a_counter_that_was_actually_written() -> None:
    """The live sentence this removes: *"19 claim(s) stored, none admissible at the
    cutoff … so the compiler saw an empty record"*, about a store whose nineteen claims
    were every one of them admissible."""

    d = _frozen_diagnosis()
    assert "archive_coverage_failure" not in [c["cause"] for c in d.root_cause()]

    # The defect this cause was added for is untouched: a run that measured an empty
    # admissible view still names it, and still names it first.
    d.bundle = dc_replace(d.bundle, live_trace={**FROZEN_TRACE, "admissible_claim_count": 0})  # type: ignore[arg-type]
    causes = [c["cause"] for c in d.root_cause()]
    assert causes[0] == "archive_coverage_failure"


def test_a_discovery_pass_that_never_ran_is_not_a_discovery_failure() -> None:
    """*"no candidate URL was discovered at all"*, about a backend that issues no
    queries by construction. The old guard asked whether SOME trace existed; a frozen
    trace holds three keys, so it passed."""

    d = _frozen_diagnosis()
    assert "discovery_failure" not in [c["cause"] for c in d.root_cause()]

    # A real pass writes `attempted_urls` and `queries` unconditionally, so an empty
    # list is a search that found nothing — and that is still a discovery failure.
    d.bundle = dc_replace(  # type: ignore[arg-type]
        d.bundle, live_trace={**FROZEN_TRACE, "attempted_urls": [], "queries": []}
    )
    causes = [c["cause"] for c in d.root_cause()]
    assert "discovery_failure" in causes


def test_a_diagnosis_that_could_not_observe_something_says_so() -> None:
    """Declining to infer is only half of it. A reader left with no line at all cannot
    tell "the run checked this and it was fine" from "the run never looked", which is
    the same confusion one level up."""

    d = _frozen_diagnosis()
    said = {u["measurement"]: u for u in d.unobserved()}
    assert set(said) == {"claims_admissible_at_cutoff", "urls_considered"}
    assert said["claims_admissible_at_cutoff"]["cause_neither_inferred_nor_ruled_out"] == (
        "archive_coverage_failure"
    )
    assert said["urls_considered"]["cause_neither_inferred_nor_ruled_out"] == "discovery_failure"
    assert d.as_dict()["unobserved"] == d.unobserved()

    # The run still names the mechanism it DID observe.
    assert [c["cause"] for c in d.root_cause()] == ["actor_not_attested_by_evidence"]

    # A run whose research actually ran claims nothing unobserved.
    d.bundle = dc_replace(  # type: ignore[arg-type]
        d.bundle,
        live_trace={
            **FROZEN_TRACE,
            "admissible_claim_count": 19,
            "attempted_urls": ["https://x.test/a"],
            "queries": [{"query": "q", "channel": "general"}],
        },
    )
    assert d.unobserved() == []


# --------------------------------------------------------------------------- #
# 3. No root cause asserts more than the run observed.
# --------------------------------------------------------------------------- #


def test_no_root_cause_is_a_verdict_on_whether_a_check_should_have_fired() -> None:
    """``over_strict_grounding_gate`` said the grounding gate had been wrong. Nothing
    in a run measures that, and in the run that provoked this the gate reported *"every
    claim assigned to this actor was checked and none of them mentions it"* — a gate
    behaving exactly as designed, filed under a name saying it had misbehaved. The same
    shape sat on the verifier twice, and a fourth was in the vocabulary unemitted."""

    from sworldmodel.diagnosis import ROOT_CAUSES

    verdicts = [
        c
        for c in ROOT_CAUSES
        if "too_strict" in c or "too_weak" in c or "over_strict" in c or "under_strict" in c
    ]
    assert verdicts == [], (
        "a root cause may state what the run observed, never whether a component "
        f"should have behaved as it did: {verdicts}"
    )


def test_the_ungrounded_actor_cause_quotes_the_gate_rather_than_judging_it() -> None:
    """The cause is now the gate's finding restated, carrying its own words — so the
    reader decides whether the gate was too strict, with the reason in front of them."""

    reason = (
        "seven_opecplus_countries (Seven_OPECPlus_Countries): no surviving citation "
        "attaches this actor to the world — every claim assigned to this actor was "
        "checked and none of them mentions it"
    )
    d = _frozen_diagnosis()
    d.failure = WorldIntegrityError(
        "actors are not grounded in cited evidence — simulation refused",
        details={
            "failure": "actors_ungrounded",
            "recompilable": True,
            "ungrounded_actors": [reason],
        },
    )
    cause = next(c for c in d.root_cause() if c["cause"] == "actor_not_attested_by_evidence")
    assert reason in cause["why"], "the gate's own reason must travel with the cause"
    assert "1 compiled actor(s)" in cause["why"]
    assert "strict" not in cause["why"], "the cause must not judge the gate for the reader"


def test_a_contradiction_between_verified_claims_is_not_a_verdict_on_the_verifier() -> None:
    """``claim_verification_too_weak`` asserted the verifier had let something through.
    What was observed is that two claims which BOTH passed disagree — and this
    repository's own repair module argues the commonest case is not a verification
    defect at all, but the uncertainty the simulation exists to resolve."""

    d = _frozen_diagnosis()
    d.failure = WorldIntegrityError(
        "decisive evidence contradictions block rollout",
        details={
            "failure": "decisive_evidence_contradiction",
            "recompilable": True,
            "contradictions": ["the board has five members <> the board has nine members"],
        },
    )
    causes = {c["cause"]: c["why"] for c in d.root_cause()}
    assert "verified_claims_contradict_each_other" in causes
    assert "five members" in causes["verified_claims_contradict_each_other"], (
        "the contradiction itself must be quoted, so a reader can judge whether it is one"
    )


def test_a_wake_up_loop_is_named_only_when_one_was_measured() -> None:
    """*The event loop did not settle* was asserted for EVERY simulation-stage failure,
    off nothing but the stage. W3 counts re-decisions per branch; the cause is claimed
    only on a positive count, and a simulation that failed some other way says so."""

    class _Ran(RunDiagnosis):
        convergence: dict[str, int] = {"repeat_decisions": 0}

        def runtime(self) -> dict[str, Any]:
            return {"ran": True, "actor_invocations": 12, "convergence": self.convergence}

    d = _Ran(
        question="q",
        as_of=AS_OF_DT,
        horizon=HORIZON_DT,
        failure=RuntimeError("something in the runtime"),
        failure_stage="simulation",
    )
    assert [c["cause"] for c in d.root_cause()] == ["unclassified"], (
        "a simulation failure with no measured repeat must not be called a wake-up loop"
    )

    d.convergence = {"repeat_decisions": 7}
    cause = next(c for c in d.root_cause() if c["cause"] == "repeated_wake_up_loop")
    assert "7 actor call(s)" in cause["why"] and "12 invocation(s)" in cause["why"]
