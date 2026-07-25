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
from .diagnosis import ForecastRefused
from .engine import RunResult, run
from .errors import GatewayError, SWorldModelError, WorldIntegrityError
from .models import ForecastResult, ResolutionContract
from .outcomes import aggregate
from .repair import RepairLog, RepairPlan, plan_repair
from .research import ResearchBundle, assemble_bundle
from .structures import (
    StructuralAlternative,
    StructuralAssessment,
    alternative_compile_instruction,
    assess_structure,
)
from .tracing import TraceContext
from .world_compiler import compile_world, compile_world_spec_live, render_evidence
from .world_review import review_world


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


# A ceiling on repair attempts, not a policy. Repair stops when it stops making
# progress; this only bounds a pathological alternation between two failures that each
# "fix" the other. It is deliberately far above the number of rounds real repair takes.
_REPAIR_CEILING = 12


def _compile_with_repair(
    question: str,
    as_of: datetime,
    horizon: datetime,
    bundle: ResearchBundle,
    config: ForecastConfig,
    *,
    log: RepairLog | None = None,
) -> tuple[ResearchBundle, CompiledWorld]:
    """Compile the world; when a gate refuses, repair the exact element it named.

    Each refusal carries a machine-readable ``failure`` code. :func:`plan_repair` turns
    that code into targeted research (a missing office-holder sends the researcher after
    rosters; a missing mechanism sends it after procedural rules) and a specific compiler
    instruction. Where the failure is the compiler contradicting itself or dropping
    something already in the evidence store, no research is warranted and only the
    instruction changes.

    The loop continues while repair is *achieving something*: while each attempt either
    changes the diagnosed failure or adds new claims to the evidence store. When an
    attempt does neither, there is no further defensible source or representation path,
    and the refusal is real rather than an artifact of the attempt budget. Then it
    propagates: the gates themselves are never negotiable.
    """

    log = log if log is not None else RepairLog()
    seen_failures: set[str] = set()

    for _ in range(_REPAIR_CEILING):
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
            failure = str(exc.details.get("failure") or "unclassified")
            before = len(bundle.evidence_store.claims)
            plan = plan_repair(exc, question, subject_entity=bundle.subject_entity)
            if plan is None:
                log.record(
                    None,
                    failure=failure,
                    message=str(exc),
                    claims_before=before,
                    claims_after=before,
                    outcome="no repair plan for this failure",
                )
                raise

            repaired = _repair_once(question, as_of, horizon, bundle, config, plan)
            after = len(repaired.evidence_store.claims) if repaired else before
            if repaired is None:
                log.record(
                    plan,
                    failure=failure,
                    message=str(exc),
                    claims_before=before,
                    claims_after=before,
                    outcome="repair could not be attempted (no live gateway or backend)",
                )
                raise

            # Progress means new evidence, or a failure we have not diagnosed before.
            # Repeating a diagnosis with nothing new to read is the definition of a
            # reroll, and it is where an honest run stops.
            new_evidence = after > before
            new_diagnosis = failure not in seen_failures
            seen_failures.add(failure)
            log.record(
                plan,
                failure=failure,
                message=str(exc),
                claims_before=before,
                claims_after=after,
                outcome=(
                    "retrying"
                    if (new_evidence or new_diagnosis)
                    else "no new evidence and no new diagnosis — repair exhausted"
                ),
            )
            if not (new_evidence or new_diagnosis):
                raise
            bundle = repaired
    # Reachable: twelve alternating diagnoses, each new the first time it appears. It
    # must arrive as a refusal like any other — an AssertionError is not a
    # SWorldModelError, so it would escape the ForecastRefused wrapper and leave the run
    # with a traceback and no diagnosis, which is the failure mode this whole run exists
    # to remove.
    raise WorldIntegrityError(
        f"repair did not converge after {_REPAIR_CEILING} attempts — each attempt "
        "changed the diagnosis without ever producing a compilable world",
        details={
            "failure": "repair_did_not_converge",
            "recompilable": False,
            "diagnoses seen": sorted(seen_failures),
            "attempts": _REPAIR_CEILING,
        },
    )


def _repair_once(
    question: str,
    as_of: datetime,
    horizon: datetime,
    bundle: ResearchBundle,
    config: ForecastConfig,
    plan: RepairPlan,
) -> ResearchBundle | None:
    """Carry out one repair: targeted research if the plan calls for it, then recompile
    with the plan's specific instruction.

    Research extends the existing evidence store — every previously verified claim keeps
    its id and lineage — so a follow-up search can only ever add to what is known.
    """

    if plan.needs_research:
        augment = getattr(config.research_backend, "augment_targeted", None)
        if augment is not None:
            extended = augment(question, as_of, horizon, list(plan.queries), bundle)
            if extended is not None:
                bundle = extended
    return _recompile(question, as_of, horizon, bundle, config, plan.instruction)


def _recompile(
    question: str,
    as_of: datetime,
    horizon: datetime,
    bundle: ResearchBundle,
    config: ForecastConfig,
    reason: str,
) -> ResearchBundle | None:
    """Compile the world again, with an instruction naming exactly what to fix.

    The evidence store is whatever the repair left it as — unchanged when the failure
    was the compiler's, extended when targeted research found more. Either way this is a
    fresh reading of a known body of facts, never a search for facts that fit a
    conclusion.
    """

    if not getattr(config.gateway, "is_live", False):
        return None
    try:
        data, _ = compile_world_spec_live(
            config.gateway,
            question,
            as_of,
            horizon,
            bundle.evidence_store.view(as_of),
            extra_instruction=(
                "A previous compilation of this question was rejected. Fix exactly this "
                "and change nothing else about how you read the evidence:\n"
                f"{reason}\n"
                "Do not invent support for anything."
            ),
            structure_id="primary",
        )
    except (GatewayError, WorldIntegrityError, ValueError, KeyError):
        return None
    return assemble_bundle(bundle.evidence_store, data)


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

    log = RepairLog()
    try:
        bundle = config.research_backend.research(question, as_of, horizon)
    except SWorldModelError as exc:
        raise ForecastRefused(exc, stage="research", repair_log=log) from exc
    except (TypeError, ValueError, KeyError) as exc:
        # A parser or provider shape nobody anticipated. It is still a run that stopped,
        # and it still owes a diagnosis rather than a traceback.
        raise ForecastRefused(exc, stage="research", repair_log=log) from exc
    try:
        bundle, compiled = _compile_with_repair(question, as_of, horizon, bundle, config, log=log)
    except SWorldModelError as exc:
        # Everything the run learned before it stopped travels with the refusal, so the
        # caller can write a diagnosis. A refusal that leaves only a traceback is how
        # four of five acceptance questions became undiagnosable.
        raise ForecastRefused(exc, stage="compilation", bundle=bundle, repair_log=log) from exc
    contract = _build_contract(question, as_of, horizon, bundle)

    # Before the rollout budget: is this obviously not the right world? The gates are
    # mechanical and have already passed it; this catches what they cannot check —
    # a resolution condition that answers a nearby question, a detail nobody sourced, a
    # date that was plausible rather than published. One call, and a clear "no" goes to
    # repair rather than into several minutes of simulating the wrong thing.
    review = review_world(
        compiled,
        bundle.evidence_store.view(as_of),
        config.gateway,
        question=question,
        evidence_render=render_evidence(bundle.evidence_store.view(as_of)),
    )
    if review.should_repair:
        repaired = _recompile(question, as_of, horizon, bundle, config, review.repair_instruction())
        if repaired is not None:
            try:
                bundle, compiled = _compile_with_repair(
                    question, as_of, horizon, repaired, config, log=log
                )
                contract = _build_contract(question, as_of, horizon, bundle)
            except SWorldModelError:
                # The review is advisory. A recompilation that the mechanical gates then
                # refuse is worse than the world we already had, which they passed.
                pass

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
    unrepresentable_mass = sum(a.weight for a, _ in unrepresentable) + assessment.undescribed_mass

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
    ctx.repair_log = log
    ctx.world_review = review
    return result, ctx


def forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> ForecastResult:
    result, ctx = run_forecast(question, as_of, horizon, config)
    if config.trace_dir is not None:
        ctx.write(config.trace_dir, gateway_calls=config.gateway.calls)
    return result
