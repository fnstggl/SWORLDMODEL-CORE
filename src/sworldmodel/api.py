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
from typing import Any

from .compiled import CompiledWorld
from .config import ForecastConfig
from .engine import RunResult, run
from .errors import GatewayError, WorldIntegrityError
from .models import ForecastResult, ResolutionContract
from .outcomes import aggregate
from .research import ResearchBundle, assemble_bundle
from .structures import (
    StructuralAlternative,
    StructuralAssessment,
    alternative_compile_instruction,
    assess_structure,
)
from .tracing import TraceContext
from .world_compiler import compile_world, compile_world_spec_live, render_evidence


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


def _structural_limitations(assessment: StructuralAssessment) -> tuple[str, ...]:
    if not assessment.is_material:
        return (f"causal structure treated as determined: {assessment.reason}",)
    out = [
        f"the causal structure itself is uncertain ({assessment.reason}); "
        f"{len(assessment.alternatives) + 1} structures were simulated and their masses combined."
    ]
    for alt in assessment.alternatives:
        out.append(
            f"alternative structure {alt.structure_id!r} (weight {alt.weight:.2f}, "
            f"{alt.provenance.value}): {alt.what_differs}"
        )
    return tuple(out)


def _compile_alternative(
    question: str,
    as_of: datetime,
    horizon: datetime,
    bundle: ResearchBundle,
    config: ForecastConfig,
    alt: StructuralAlternative,
) -> tuple[ResearchBundle, CompiledWorld]:
    """Compile one alternative causal structure from the *same* verified evidence.

    No new research happens: the same evidence store is reused, so the two structures
    are genuinely two readings of one body of facts rather than two different worlds
    built from two different sets of facts.
    """

    data, _ = compile_world_spec_live(
        config.gateway,
        question,
        as_of,
        horizon,
        bundle.evidence_store.view(as_of),
        extra_instruction=alternative_compile_instruction(alt),
        structure_id=alt.structure_id,
    )
    alt_bundle = assemble_bundle(bundle.evidence_store, data)
    contract = _build_contract(question, as_of, horizon, alt_bundle)
    compiled = compile_world(
        contract,
        alt_bundle.evidence_store.view(as_of),
        alt_bundle.spec,
        alt_bundle.uncertainties,
        alt_bundle.world_facts,
        gateway=config.gateway,
        seed=config.seed,
        max_branches=config.max_branches,
    )
    return alt_bundle, compiled


def _merge(results: list[tuple[float, str, RunResult]]) -> RunResult:
    """Combine per-structure runs into one trajectory set.

    Every branch weight is scaled by the weight of the structure it happened in, and
    branch ids are namespaced by structure, so the branch table the report prints still
    reconstructs the probability by hand.
    """

    outcomes: list[Any] = []
    summaries: list[Any] = []
    ledger: list[Any] = []
    decisions: list[Any] = []
    worlds: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    truncated = 0.0
    reasons: list[str] = []

    for weight, structure_id, res in results:
        prefix = f"{structure_id}/"
        for b in res.branch_outcomes:
            outcomes.append(replace(b, branch_id=prefix + b.branch_id, weight=b.weight * weight))
        for s in res.trajectory_summaries:
            summaries.append(replace(s, branch_id=prefix + s.branch_id, weight=s.weight * weight))
        ledger.extend(res.event_ledger)
        for d in res.actor_decisions:
            d.branch_id = prefix + d.branch_id
            decisions.append(d)
        worlds.update({prefix + k: v for k, v in res.final_worlds.items()})
        diagnostics.update({prefix + k: v for k, v in res.diagnostics.items()})
        truncated += res.truncated_mass * weight
        if res.truncated_reason:
            reasons.append(f"{structure_id}: {res.truncated_reason}")

    return RunResult(
        branch_outcomes=tuple(outcomes),
        trajectory_summaries=tuple(summaries),
        event_ledger=ledger,
        actor_decisions=decisions,
        final_worlds=worlds,
        truncated_mass=truncated,
        truncated_reason="; ".join(reasons),
        diagnostics=diagnostics,
    )


def run_forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> tuple[ForecastResult, TraceContext]:
    """Run the full pipeline and return the result plus a trace context for writing."""

    bundle = config.research_backend.research(question, as_of, horizon)
    bundle, compiled = _compile_with_repair(question, as_of, horizon, bundle, config)
    contract = _build_contract(question, as_of, horizon, bundle)

    # Is this even the right world? Ordinary uncertainty asks what a value turns out to
    # be; this asks whether the causal structure we compiled is the one that decides the
    # question. When the evidence leaves that open, each structure is simulated.
    assessment, structure_response = assess_structure(
        contract,
        bundle.evidence_store.view(as_of),
        compiled.spec,
        gateway=config.gateway,
        seed=config.seed,
        max_alternatives=config.max_structures - 1,
        evidence_render=render_evidence(bundle.evidence_store.view(as_of)),
    )

    runs: list[tuple[float, str, RunResult]] = [
        (
            assessment.primary_weight,
            compiled.spec.structure_id,
            run(compiled, config.gateway, seed=config.seed, budget=config.budget),
        )
    ]
    unrepresentable: list[tuple[StructuralAlternative, str]] = []
    for alt in assessment.alternatives:
        try:
            _, alt_compiled = _compile_alternative(question, as_of, horizon, bundle, config, alt)
        except (WorldIntegrityError, GatewayError, ValueError, KeyError) as exc:
            # A possibility we could not faithfully represent is not a possibility we
            # get to ignore. Its mass stays unresolved and widens the bounds.
            unrepresentable.append((alt, f"{type(exc).__name__}: {exc}"))
            continue
        runs.append(
            (
                alt.weight,
                alt.structure_id,
                run(alt_compiled, config.gateway, seed=config.seed, budget=config.budget),
            )
        )

    run_result = _merge(runs)
    unrepresentable_mass = sum(a.weight for a, _ in unrepresentable)

    trace_location = str(config.trace_dir) if config.trace_dir else "(not written)"
    result = aggregate(
        run_result.branch_outcomes,
        truncated_mass=run_result.truncated_mass + unrepresentable_mass,
        truncated_reason="; ".join(
            [run_result.truncated_reason]
            + [
                f"structure {a.structure_id!r} could not be represented: {why}"
                for a, why in unrepresentable
            ]
        ).strip("; "),
        contract=contract,
        manifest=compiled.manifest,
        trajectory_summaries=run_result.trajectory_summaries,
        trace_location=trace_location,
        model_call_count=config.gateway.call_count,
        token_usage=config.gateway.total_tokens,
        limitations=_limitations(config, run_result) + _structural_limitations(assessment),
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
        structure_assessment=assessment,
        structure_response=structure_response,
    )
    return result, ctx


def forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> ForecastResult:
    result, ctx = run_forecast(question, as_of, horizon, config)
    if config.trace_dir is not None:
        ctx.write(config.trace_dir, gateway_calls=config.gateway.calls)
    return result
