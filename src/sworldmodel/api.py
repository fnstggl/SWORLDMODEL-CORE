"""The single public forecasting entry point.

    forecast(question, as_of, horizon, config) -> ForecastResult

The full causal route is readable and direct, and there is exactly one of it:

    question
      -> live research            (cited evidence store built from the question alone)
      -> verified evidence        (fetch + verify + lineage + cutoff)
      -> candidate inventory      (what verified reality contains)
      -> LLM-compiled WorldSpec   (entities, actions, process graph, terminal)
      -> reality + coverage gates (assessed against the exact WorldSpec to be simulated)
      -> targeted research repair (when coverage finds a material item missing)
      -> persistent possible worlds
      -> event-driven persistent actors
      -> validated intentions
      -> universal world effects
      -> declarative terminal evaluation
      -> weighted trajectory aggregation

No phase adapters, no profiles, no mechanism families, no fallbacks. The route does
not branch on the kind of question.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from .compiled import CompiledWorld
from .config import ForecastConfig
from .engine import RunResult, run
from .errors import WorldIntegrityError
from .models import ForecastResult, ResolutionContract
from .outcomes import aggregate
from .repair import classify_failure
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


def _compile_with_repair(
    question: str,
    as_of: datetime,
    horizon: datetime,
    bundle: ResearchBundle,
    config: ForecastConfig,
    *,
    max_attempts: int = 3,
) -> tuple[ResearchBundle, CompiledWorld]:
    """Compile the world; when a gate refuses for a reason more evidence could honestly
    fix, run targeted follow-up research and recompile before giving up.

    Every refusal is classified into a typed :class:`IntegrityFailure` and routed by
    kind — a coverage omission, a world with no actors, a participant shortfall, an
    ungrounded actor, or an unverified required fact each produce their own targeted
    research need. The gates are never weakened: a structurally false world (surplus
    roster, duplicated participant, decisive contradiction) is classified as
    non-repairable and refuses immediately. If targeted research still cannot satisfy
    the gate, the final :class:`WorldIntegrityError` propagates and simulation is
    refused. A backend that cannot augment simply blocks on the first failure.
    """

    backend = config.research_backend
    repairs: list[dict[str, object]] = []
    for attempt in range(max_attempts):
        evidence_view = bundle.evidence_store.view(as_of)
        contract = _build_contract(question, as_of, horizon, bundle)
        try:
            compiled = compile_world(
                contract,
                evidence_view,
                bundle.spec,
                bundle.uncertainties,
                bundle.world_facts,
                gateway=config.gateway,
                seed=config.seed,
                max_branches=config.max_branches,
                compile_responses=bundle.compile_responses,
            )
            return replace(bundle, repair_log=tuple(repairs)), compiled
        except WorldIntegrityError as exc:
            failure = classify_failure(exc)
            augment = getattr(backend, "augment_for_coverage", None)
            repairs.append({**failure.as_dict(), "attempt": attempt})
            if not failure.retryable or augment is None or attempt == max_attempts - 1:
                raise
            needs = list(failure.targeted_research_needs) or list(failure.missing_candidates)
            if not needs:
                raise
            augmented = augment(question, as_of, horizon, needs, bundle)
            if augmented is None:
                raise
            bundle = augmented
    raise AssertionError("unreachable")  # pragma: no cover


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
    bundle, compiled = _compile_with_repair(question, as_of, horizon, bundle, config)
    contract = _build_contract(question, as_of, horizon, bundle)

    run_result: RunResult = run(compiled, config.gateway, seed=config.seed)

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
