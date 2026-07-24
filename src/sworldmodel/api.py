"""The single public forecasting entry point.

    forecast(question, as_of, horizon, config) -> ForecastResult

The full causal route is readable and direct:

    research  -> evidence + compiled WorldSpec (actors, actions, process, terminal)
              -> contract (immutable question definition, locks the declarative terminal)
              -> compile  (verified base world + reality-integrity gate + uncertainty branches)
              -> event runtime (one universal engine executes the compiled process graph)
              -> terminal evaluator (deterministic declarative predicate over world state)
              -> trajectory aggregation -> report

There is exactly one normal path, and it does not branch on the kind of question. No
phase adapters, no profiles, no mechanism families, no fallbacks.
"""

from __future__ import annotations

from datetime import datetime

from .config import ForecastConfig
from .engine import run
from .models import ForecastResult, ResolutionContract
from .outcomes import aggregate
from .research import ResearchBundle
from .tracing import TraceContext
from .world_compiler import compile_world


def _build_contract(
    question: str, as_of: datetime, horizon: datetime, bundle: ResearchBundle
) -> ResolutionContract:
    return ResolutionContract(
        question=question,
        as_of=as_of,
        horizon=horizon,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome or bundle.spec.terminal.description,
        authoritative_resolution_sources=bundle.authoritative_sources,
        required_reality_facts=bundle.required_reality_facts,
        expected_participants=bundle.expected_participants,
    )


def _limitations(config: ForecastConfig) -> tuple[str, ...]:
    return (
        f"actor behavior produced by gateway {config.gateway.model_id!r}; offline runs use a "
        "deterministic calibrated-behavior reasoner, not a frontier LLM.",
        "branch weights on uncertain future data are epistemic (symmetric-ignorance / "
        "explicit-model); the reported bounds expose that sensitivity.",
    )


def run_forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> tuple[ForecastResult, TraceContext]:
    """Run the full pipeline and return the result plus a trace context for writing."""

    bundle = config.research_backend.research(question, as_of, horizon)
    evidence_view = bundle.evidence_store.view(as_of)
    contract = _build_contract(question, as_of, horizon, bundle)

    compiled = compile_world(
        contract,
        evidence_view,
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        seed=config.seed,
        max_branches=config.max_branches,
        compile_responses=bundle.compile_responses,
    )
    run_result = run(compiled, config.gateway, seed=config.seed)

    diagnostics: tuple[tuple[str, str], ...] = ()
    if config.include_reference_class_diagnostic and bundle.reference_class:
        diagnostics = tuple(sorted(bundle.reference_class.items()))

    trace_location = str(config.trace_dir) if config.trace_dir else "(not written)"
    result = aggregate(
        run_result.branch_outcomes,
        truncated_mass=run_result.truncated_mass,
        truncated_reason=run_result.truncated_reason,
        contract=contract,
        manifest=compiled.manifest,
        trajectory_summaries=run_result.trajectory_summaries,
        trace_location=trace_location,
        model_call_count=config.gateway.call_count,
        token_usage=config.gateway.total_tokens,
        limitations=_limitations(config),
        diagnostics=diagnostics,
    )
    ctx = TraceContext(
        contract=contract,
        evidence_store=bundle.evidence_store,
        as_of=as_of,
        bundle=bundle,
        compiled=compiled,
        run_result=run_result,
        forecast=result,
        model_id=config.gateway.model_id,
    )
    return result, ctx


def forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> ForecastResult:
    result, ctx = run_forecast(question, as_of, horizon, config)
    if config.trace_dir is not None:
        ctx.write(config.trace_dir, gateway_calls=config.gateway.calls)
    return result
