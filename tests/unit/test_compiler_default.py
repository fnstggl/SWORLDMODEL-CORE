"""The semantic compiler is the default and canonical path, everywhere.

The five rules this file pins:

1. invoking the CLI or API without a compiler argument selects semantic;
2. direct runs only when explicitly requested;
3. a semantic refusal remains a semantic refusal and never triggers direct;
4. resume rejects artifacts created under a different compiler mode;
5. the mode is recorded on every compilation and every diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

import pytest

from _fakes import FakeTransport, FixtureResearchBackend, ProgrammableGateway, build_bundle
from _worlds import single_response_world
from sworldmodel.cli import build_parser
from sworldmodel.config import ForecastConfig
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.gateway import ModelGateway
from sworldmodel.live_research import LiveResearchBackend

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def test_the_cli_selects_semantic_when_no_compiler_is_named() -> None:
    args = build_parser().parse_args(
        [
            "forecast",
            "--question",
            "q?",
            "--as-of",
            AS_OF.isoformat(),
            "--horizon",
            HORIZON.isoformat(),
        ]
    )
    assert args.compiler == "semantic"


def test_direct_runs_only_when_explicitly_requested() -> None:
    args = build_parser().parse_args(
        [
            "forecast",
            "--question",
            "q?",
            "--as-of",
            AS_OF.isoformat(),
            "--horizon",
            HORIZON.isoformat(),
            "--compiler",
            "direct",
        ]
    )
    assert args.compiler == "direct"

    gw = ProgrammableGateway({})
    config = ForecastConfig(
        gateway=gw,
        research_backend=FixtureResearchBackend(build_bundle(single_response_world())),
        compiler_mode="direct",
    )
    import contextlib

    from sworldmodel.api import compile_for_mode

    # Direct mode consults ONLY the direct compiler; whether the empty scripted
    # response survives its parsing is not the point — the calls are.
    with contextlib.suppress(Exception):
        compile_for_mode(
            config,
            "q?",
            AS_OF,
            HORIZON,
            build_bundle(single_response_world()).evidence_store.view(AS_OF),
        )
    kinds = {r.task_kind for r in gw.seen}
    assert "compile_world_spec" in kinds
    assert not kinds & {"semantic_plan", "semantic_plan_delta", "semantic_review"}


def test_config_and_backend_default_to_semantic() -> None:
    gw = ProgrammableGateway({})
    config = ForecastConfig(
        gateway=gw, research_backend=FixtureResearchBackend(build_bundle(single_response_world()))
    )
    assert config.compiler_mode == "semantic"
    backend = LiveResearchBackend(gw, FakeTransport())
    assert backend.compiler_mode == "semantic"


def test_a_config_that_never_states_a_mode_compiles_semantically() -> None:
    """compile_for_mode's own default — the protocol getattr — is semantic too, so a
    minimal config shim without the attribute cannot silently become direct."""

    @dataclass(frozen=True)
    class BareConfig:
        gateway: ModelGateway

    gw = ProgrammableGateway({})  # nothing scripted: the semantic plan will refuse
    with pytest.raises(WorldIntegrityError) as exc:
        from sworldmodel.api import compile_for_mode

        compile_for_mode(
            BareConfig(gateway=gw),
            "q?",
            AS_OF,
            HORIZON,
            build_bundle(single_response_world()).evidence_store.view(AS_OF),
        )
    assert exc.value.details["failure"] == "semantic_plan_invalid"
    kinds = {r.task_kind for r in gw.seen}
    assert "semantic_plan" in kinds
    assert "compile_world_spec" not in kinds


def test_a_semantic_refusal_never_falls_back_to_direct() -> None:
    """Rule 3. The scripted plan is unreadable in every round; the refusal must
    propagate as the semantic pipeline's own, with zero direct-compiler calls."""

    from sworldmodel.api import compile_for_mode

    gw = ProgrammableGateway({"compile_world_spec": {"world_spec": {}}})  # bait: never taken
    config = ForecastConfig(
        gateway=gw,
        research_backend=FixtureResearchBackend(build_bundle(single_response_world())),
        compiler_mode="semantic",
    )
    with pytest.raises(WorldIntegrityError) as exc:
        compile_for_mode(
            config,
            "q?",
            AS_OF,
            HORIZON,
            build_bundle(single_response_world()).evidence_store.view(AS_OF),
        )
    assert exc.value.details["failure"] == "semantic_plan_invalid"
    assert not [r for r in gw.seen if r.task_kind == "compile_world_spec"]


def test_an_empty_admissible_view_refuses_before_any_model_call() -> None:
    """A store with nothing admissible at the cutoff cannot ground any world, so no
    plan it produced could cite anything and every repair round would re-derive the
    same impossibility. The Banxico pastcast measured it: 939 seconds — the entire
    compile-and-repair budget — spent discovering that nothing can cite nothing, then
    refusing anyway. The refusal must arrive immediately, name the archive gap rather
    than the compiler, and cost zero model calls; and it must be final, not
    recompilable."""

    from sworldmodel.api import compile_for_mode
    from sworldmodel.evidence import EvidenceStore

    empty = EvidenceStore().view(AS_OF)
    for mode in ("semantic", "direct"):
        gw = ProgrammableGateway({})
        config = ForecastConfig(
            gateway=gw,
            research_backend=FixtureResearchBackend(build_bundle(single_response_world())),
            compiler_mode=mode,
        )
        with pytest.raises(WorldIntegrityError) as exc:
            compile_for_mode(config, "q?", AS_OF, HORIZON, empty)
        assert exc.value.details["failure"] == "no_admissible_evidence"
        assert exc.value.details["recompilable"] is False
        assert gw.call_count == 0, f"{mode}: an ungroundable world must cost no calls"


def test_the_empty_view_refusal_names_the_record_not_the_compiler() -> None:
    from sworldmodel.diagnosis import RunDiagnosis
    from sworldmodel.errors import WorldIntegrityError as WIE

    diagnosis = RunDiagnosis(
        question="q?",
        as_of=AS_OF,
        horizon=HORIZON,
        failure=WIE("no evidence", details={"failure": "no_admissible_evidence"}),
        failure_stage="compilation",
    )
    causes = [c["cause"] for c in diagnosis.root_cause()]
    assert "archive_coverage_failure" in causes
    assert "compiler_omission" not in causes


def test_resume_rejects_artifacts_from_the_other_compiler_mode() -> None:
    """Rule 4. A bundle whose research trace was recorded under direct mode cannot be
    extended by a semantic run (or vice versa) — the mismatch refuses with its own
    failure code instead of splicing modes silently."""

    gw = ProgrammableGateway({})
    backend = LiveResearchBackend(gw, FakeTransport())
    prior = replace(
        build_bundle(single_response_world()),
        live_trace={"compiler_mode": "direct", "queries": []},
    )
    with pytest.raises(WorldIntegrityError) as exc:
        backend.augment_targeted("q?", AS_OF, HORIZON, ["find the roster"], prior)
    assert exc.value.details["failure"] == "compiler_mode_mismatch"
    assert exc.value.details["recompilable"] is False
    assert exc.value.details["recorded_mode"] == "direct"
    assert exc.value.details["run_mode"] == "semantic"

    # Same mode resumes fine as far as the mode guard is concerned (it then proceeds
    # into retrieval, which the empty transport ends quietly).
    same = replace(
        build_bundle(single_response_world()),
        live_trace={"compiler_mode": "semantic", "queries": []},
    )
    assert backend.augment_targeted("q?", AS_OF, HORIZON, ["find the roster"], same) is None


def test_a_prior_plan_from_another_mode_is_never_consumed() -> None:
    from sworldmodel.api import _latest_semantic_plan

    trace = {
        "compiler_mode": "direct",
        "semantic_compilation": {"plan": {"resolution": {}}},
    }
    assert _latest_semantic_plan(trace) is None
    trace_semantic = {
        "compiler_mode": "semantic",
        "semantic_compilation": {"plan": {"resolution": {}}},
    }
    assert _latest_semantic_plan(trace_semantic) == {"resolution": {}}


def test_the_diagnosis_records_semantic_by_default() -> None:
    from sworldmodel.diagnosis import RunDiagnosis

    d = RunDiagnosis(question="q?", as_of=AS_OF, horizon=HORIZON)
    assert d.compiler_mode == "semantic"
    assert d.as_dict()["compiler_mode"] == "semantic"
