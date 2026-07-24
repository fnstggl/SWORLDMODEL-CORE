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

from datetime import datetime

from .compiled import CompiledWorld
from .config import ForecastConfig
from .engine import RunResult, run
from .errors import WorldIntegrityError
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
                bundle.spec,
                bundle.uncertainties,
                bundle.world_facts,
                gateway=config.gateway,
                seed=config.seed,
                max_branches=config.max_branches,
                compile_responses=bundle.compile_responses,
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
    """State honestly what this particular run's number does and does not rest on."""

    out = [
        f"actor behavior was produced by {config.gateway.model_id!r}; every actor decision in "
        "the trace is a real provider call, and deleting those calls deletes the forecast.",
        "branch weights on uncertain future values are epistemic (symmetric-ignorance or "
        "explicit-model); the reported unconditional bounds expose that sensitivity.",
    ]
    stops = {
        d.stop_reason
        for d in run_result.diagnostics.values()
        if d.stop_reason and d.stop_reason != "schedule exhausted"
    }
    for stop in sorted(stops):
        out.append(f"at least one branch ended early: {stop}")
    beyond = sum(len(d.pending_beyond_horizon) for d in run_result.diagnostics.values())
    if beyond:
        out.append(
            f"{beyond} scheduled world events fall after the horizon and were never executed; "
            "the question's window closed before that part of the process."
        )
    return tuple(out)


def run_forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> tuple[ForecastResult, TraceContext]:
    """Run the full pipeline and return the result plus a trace context for writing."""

    bundle = config.research_backend.research(question, as_of, horizon)
    bundle, compiled = _compile_with_repair(question, as_of, horizon, bundle, config)
    contract = _build_contract(question, as_of, horizon, bundle)

    run_result: RunResult = run(compiled, config.gateway, seed=config.seed, budget=config.budget)

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
        diagnostics=(),
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
