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

import time
from dataclasses import replace
from datetime import datetime
from typing import Any

from .compiled import CompiledWorld
from .config import ForecastConfig
from .diagnosis import ForecastRefused
from .engine import RunResult, run
from .errors import GatewayError, RunInterrupted, SWorldModelError, WorldIntegrityError
from .ids import canonical_json
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
from .trajectory_audit import audit_trajectory
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
    attempted: list[ResearchBundle] | None = None,
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
    seen_signatures: set[str] = set()
    deadline = time.monotonic() + max(0.0, config.max_compile_seconds)

    for _ in range(_REPAIR_CEILING):
        # Every world this loop actually tried, in order, so a refusal can report the
        # one that was refused rather than the one the caller handed in.
        if attempted is not None:
            attempted.append(bundle)
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
                    outcome=(
                        "the repaired compilation could not be produced or read — no "
                        "live gateway, a provider error, or a world the parser could "
                        "not make sense of"
                    ),
                )
                raise

            # Progress means new evidence, or a failure we have not diagnosed before,
            # or the same kind of failure about something different. Repeating a
            # diagnosis with nothing new to read is the definition of a reroll, and it
            # is where an honest run stops — but a live Bank of England run was stopped
            # after two rounds for "no new diagnosis" while the compiler was in fact
            # changing the world each time: the orphan term moved from
            # bailey_public_stance to bailey_vote. Same code, different world, and the
            # second attempt was never made. The ceiling above is what bounds a compiler
            # that cycles forever.
            signature = f"{failure}|{_failure_signature(exc)}"
            new_evidence = after > before
            new_diagnosis = failure not in seen_failures or signature not in seen_signatures
            seen_failures.add(failure)
            seen_signatures.add(signature)
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
            if time.monotonic() > deadline:
                # Out of time, not out of ideas. The refusal that propagates is the last
                # gate's own, so the record says what the world was still missing rather
                # than only that a clock ran out — and the run ends with a diagnosis
                # instead of being killed from outside with nothing written.
                log.record(
                    plan,
                    failure=failure,
                    message=str(exc),
                    claims_before=before,
                    claims_after=after,
                    outcome=(
                        f"repair budget of {config.max_compile_seconds:.0f}s exhausted; "
                        "the last diagnosis stands"
                    ),
                )
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


def _failure_signature(exc: WorldIntegrityError) -> str:
    """What this refusal is *about*, so two refusals with the same code can be told apart.

    Only the details that name world elements — the orphan terms, the absent
    participants, the missing candidates — not counts or free text, which move for
    reasons that are not a different world.
    """

    parts: list[str] = []
    for key in sorted(exc.details):
        value = exc.details[key]
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            parts.append(f"{key}={sorted(value)}")
    return ";".join(parts)


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
        # Carry the research record forward. A compiler-only repair does no new
        # research, so `assemble_bundle` has no trace to build — and without this the
        # record of every query, source and rejection made before the repair was dropped
        # on the floor. The run that first completed reported "0 queries, 0 sources
        # fetched, 0 claims" in its own diagnosis while its audit showed 38 extractions
        # and 399 HTTP requests.
        #
        # Parsing is inside the try for a reason. A live OPEC+ run died on
        # `float(None)` in the resource parser *here*, during a repair recompile, where
        # nothing was catching it — past every gate that would have turned it into a
        # diagnosis, out through run_forecast, leaving a traceback and no artifacts at
        # all. A repair that cannot be read is a repair that did not happen.
        return replace(assemble_bundle(bundle.evidence_store, data), live_trace=bundle.live_trace)
    except (GatewayError, WorldIntegrityError, ValueError, KeyError, TypeError, IndexError):
        return None


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


def _checkpoint_research(config: ForecastConfig, bundle: ResearchBundle) -> None:
    """Persist what research produced, the moment it produces it.

    Research is the expensive stage — minutes of network, dozens of model calls — and
    for a long time its record only reached disk if everything after it also succeeded.
    A live EU-Mercosur run spent twenty minutes gathering evidence and then died in a
    truncated HTTP read, leaving an empty trace directory: there was nothing to diagnose
    because nothing had been written. Checkpointing here means a later stage can fail,
    crash or be killed and the research still survives.

    Best effort by construction: a checkpoint that raised would itself become a way to
    lose a run, which is the opposite of the point.
    """

    out = config.trace_dir
    if out is None:
        return
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "research_trace.json").write_text(canonical_json(bundle.live_trace or {}) + "\n")
        (out / "evidence_store.json").write_text(
            canonical_json(
                [
                    {
                        "id": c.id,
                        "proposition": c.proposition,
                        "normalized_value": c.normalized_value,
                        "entities": list(c.entities),
                        "epistemic_type": c.epistemic_type.value,
                        "source_url": c.source_url,
                        "supporting_excerpt": c.supporting_excerpt,
                        "available_at": c.available_at.isoformat(),
                    }
                    for c in bundle.evidence_store.all()
                ]
            )
            + "\n"
        )
    except OSError:
        pass


def run_forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> tuple[ForecastResult, TraceContext]:
    """Run the full pipeline and return the result plus a trace context for writing."""

    log = RepairLog()
    try:
        bundle = config.research_backend.research(question, as_of, horizon)
    except RunInterrupted:
        # A stop is not a refusal. Reporting it as one puts a root cause on the run's
        # record — "no candidate URL was discovered at all" — that describes how far it
        # had got, not why it ended.
        raise
    except SWorldModelError as exc:
        raise ForecastRefused(exc, stage="research", repair_log=log) from exc
    except (TypeError, ValueError, KeyError) as exc:
        # A parser or provider shape nobody anticipated. It is still a run that stopped,
        # and it still owes a diagnosis rather than a traceback.
        raise ForecastRefused(exc, stage="research", repair_log=log) from exc
    _checkpoint_research(config, bundle)
    attempted: list[ResearchBundle] = []
    try:
        bundle, compiled = _compile_with_repair(
            question, as_of, horizon, bundle, config, log=log, attempted=attempted
        )
    except RunInterrupted:
        raise
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
        # Not a gate: a shape nobody anticipated, from a parser or a provider payload.
        # It is still a run that stopped, and it still owes a diagnosis rather than a
        # traceback — a live OPEC+ run died on `float(None)` inside the resource parser
        # and wrote no artifacts at all, which is precisely the failure mode this whole
        # run exists to remove.
        raise ForecastRefused(
            exc,
            stage="compilation",
            bundle=attempted[-1] if attempted else bundle,
            repair_log=log,
        ) from exc
    except SWorldModelError as exc:
        # Everything the run learned before it stopped travels with the refusal, so the
        # caller can write a diagnosis. A refusal that leaves only a traceback is how
        # four of five acceptance questions became undiagnosable.
        #
        # The bundle that travels is the *last one tried*, not the first. Repair rebinds
        # its own local, so reporting the caller's variable described the world the run
        # started from: a Banxico refusal reading "no actions and no external processes"
        # was filed beside a compiled world with three actions in it, which is a
        # diagnosis of a world nobody refused.
        raise ForecastRefused(
            exc, stage="compilation", bundle=attempted[-1] if attempted else bundle, repair_log=log
        ) from exc
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
        outcome = "recompile produced nothing; the reviewed world was simulated"
        repaired = _recompile(question, as_of, horizon, bundle, config, review.repair_instruction())
        if repaired is not None:
            try:
                bundle, compiled = _compile_with_repair(
                    question, as_of, horizon, repaired, config, log=log
                )
                contract = _build_contract(question, as_of, horizon, bundle)
                outcome = "recompiled; the world simulated is not the world reviewed here"
            except SWorldModelError as exc:
                # The review is advisory. A recompilation that the mechanical gates then
                # refuse is worse than the world we already had, which they passed.
                outcome = (
                    f"recompile refused by the gates ({exc}); the reviewed world was simulated"
                )
        review = replace(review, disposition=outcome)

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
    # The second adversary: attack the trajectory the way the reality auditor attacked
    # the world. Mechanical checks first (repeated calls, an unproduced YES, a forecast
    # that merely repeats its initialization), then one model pass over the per-branch
    # digest. Advisory and never raising — it classifies what happened, it does not
    # decide whether the run was allowed.
    ctx.trajectory_audit = audit_trajectory(compiled, run_result, config.gateway, question=question)
    return result, ctx


def forecast(
    question: str, as_of: datetime, horizon: datetime, config: ForecastConfig
) -> ForecastResult:
    result, ctx = run_forecast(question, as_of, horizon, config)
    if config.trace_dir is not None:
        ctx.write(config.trace_dir, gateway_calls=config.gateway.calls)
    return result
