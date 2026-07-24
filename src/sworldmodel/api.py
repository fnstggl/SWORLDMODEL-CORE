"""The single public forecasting entry point.

    forecast(question, as_of, horizon, config) -> ForecastResult

The full causal route is readable and direct:

    research -> evidence -> contract -> integrity -> compile
             -> initialize worlds -> event runtime -> terminal evaluator
             -> trajectory aggregation -> report

There is exactly one normal path. No phase adapters, no profiles, no fallbacks.
"""

from __future__ import annotations

from datetime import datetime

from .compiler import CompiledWorld, compile_world
from .config import ForecastConfig
from .errors import WorldIntegrityError
from .models import ForecastResult, ResolutionContract
from .outcomes import aggregate
from .research import ResearchBundle
from .runtime import RunResult, run
from .tracing import TraceContext


def _build_contract(
    question: str, as_of: datetime, horizon: datetime, bundle: ResearchBundle
) -> ResolutionContract:
    return ResolutionContract(
        question=question,
        as_of=as_of,
        horizon=horizon,
        outcome_space=bundle.frame.options,
        target_outcome=f"{bundle.terminal_spec.yes_condition}:{bundle.target_option}",
        subject_entity=bundle.subject_entity,
        decision_body=bundle.decision_body,
        resolution_units=bundle.resolution_units,
        terminal_predicate=bundle.terminal_spec,
        decision_rule=bundle.decision_rule,
        authoritative_resolution_sources=bundle.authoritative_sources,
        required_reality_facts=bundle.required_reality_facts,
        expected_voting_seats=bundle.expected_voting_seats,
    )


def _compile_with_repair(
    question: str,
    as_of: datetime,
    horizon: datetime,
    bundle: ResearchBundle,
    config: ForecastConfig,
    *,
    max_attempts: int = 3,
) -> tuple[ResearchBundle, CompiledWorld]:
    """Compile the world, and if the coverage gate finds a materially relevant
    candidate missing, run targeted follow-up research and recompile before giving up.

    The gate itself is non-negotiable: if the missing candidates still cannot be
    represented after the backend has exhausted its follow-up research, the final
    :class:`WorldIntegrityError` propagates and simulation is refused. A backend that
    cannot augment (offline corpus/mock) simply blocks on the first failure.
    """

    backend = config.research_backend
    for attempt in range(max_attempts):
        evidence_view = bundle.evidence_store.view(as_of)
        contract = _build_contract(question, as_of, horizon, bundle)
        try:
            compiled = compile_world(
                contract,
                evidence_view,
                bundle,
                config.gateway,
                seed=config.seed,
                max_branches=config.max_branches,
            )
            return bundle, compiled
        except WorldIntegrityError as exc:
            missing = exc.details.get("missing_material_candidates")
            augment = getattr(backend, "augment_for_coverage", None)
            if not isinstance(missing, list) or augment is None or attempt == max_attempts - 1:
                raise
            augmented = augment(question, as_of, horizon, [str(m) for m in missing], bundle)
            if augmented is None:
                raise
            bundle = augmented
    raise AssertionError("unreachable")  # pragma: no cover


def _limitations(config: ForecastConfig, run_result: RunResult) -> tuple[str, ...]:
    lims = [
        f"actor behavior produced by gateway {config.gateway.model_id!r}; offline runs use a "
        "deterministic calibrated-behavior reasoner, not a frontier LLM.",
        "branch weights on uncertain future data are epistemic (symmetric-ignorance / "
        "explicit-model); the reported bounds expose that sensitivity.",
    ]
    return tuple(lims)


def run_forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> tuple[ForecastResult, TraceContext]:
    """Run the full pipeline and return the result plus a trace context for writing."""

    bundle = config.research_backend.research(question, as_of, horizon)
    bundle, compiled = _compile_with_repair(question, as_of, horizon, bundle, config)
    contract = _build_contract(question, as_of, horizon, bundle)
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
        limitations=_limitations(config, run_result),
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
