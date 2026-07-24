"""Full question-only pipeline through the live modules — mocked HTTP + fake LLM."""

from __future__ import annotations

from datetime import datetime

from _live_helpers import AS_OF, HORIZON, FakeLLM, widget_transport
from sworldmodel import ForecastConfig, run_forecast
from sworldmodel.deepseek_gateway import DeepSeekGateway
from sworldmodel.http import FakeTransport
from sworldmodel.live_research import LiveResearchBackend, ResearchBudget

NOW = datetime.fromisoformat("2027-03-01T00:00:00+00:00")
QUESTION = "Will the Widget Standards Board adopt the standard at its next meeting?"


def _run():
    llm = FakeLLM()
    transport = widget_transport()
    backend = LiveResearchBackend(llm, transport, budget=ResearchBudget(max_rounds=1), now=NOW)
    config = ForecastConfig(gateway=llm, research_backend=backend, seed=0, max_branches=4)
    result, ctx = run_forecast(QUESTION, AS_OF, HORIZON, config)
    return result, ctx, config


def test_question_only_pipeline_produces_a_trajectory_forecast() -> None:
    result, _, _ = _run()
    assert result.probability_source == "weighted_simulated_trajectories"
    assert result.integrity_manifest.is_verified
    assert result.integrity_manifest.represented_voting_seats == 3
    for b in result.branch_outcomes:
        if b.resolved:
            assert len(b.votes) == 3  # every seat casts a final vote


def test_pipeline_separates_llm_stages() -> None:
    _, _, config = _run()
    stages = config.gateway.stage_call_counts()
    for stage in (
        "research_plan",
        "extract_claims",
        "compile_reality",
        "compile_uncertainty",
        "compile_world",
        "actor_decision",
    ):
        assert stages.get(stage, 0) > 0, f"stage {stage} never called"


def test_config_is_live_requires_a_live_gateway_and_backend() -> None:
    # The fake LLM is a test double (is_live=False) -> the config is NOT live.
    llm = FakeLLM()
    backend = LiveResearchBackend(llm, widget_transport(), now=NOW)
    assert not ForecastConfig(gateway=llm, research_backend=backend).is_live

    # A real DeepSeek gateway + live backend IS live (no network call made here).
    real = DeepSeekGateway(FakeTransport(), api_key="k")
    live_backend = LiveResearchBackend(real, FakeTransport())
    assert ForecastConfig(gateway=real, research_backend=live_backend).is_live


def test_production_config_rejects_deterministic_gateway() -> None:
    from sworldmodel import DeterministicGateway

    llm = FakeLLM()
    backend = LiveResearchBackend(llm, widget_transport(), now=NOW)
    # A deterministic gateway can never satisfy the live gate.
    assert not getattr(DeterministicGateway(), "is_live", False)
    assert not ForecastConfig(gateway=DeterministicGateway(), research_backend=backend).is_live
