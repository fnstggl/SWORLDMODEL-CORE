"""Terminal aggregation — the forecast is the weighted frequency of YES trajectories.

The reported forecast comes only from simulated trajectory outcomes. No historical
prior, generic outcome prior, numerical consensus model, or combiner may override
the completed trajectories. When mass is genuinely unresolved, we report bounds
rather than inventing a point estimate; if nothing is validly resolved we return no
fabricated probability.

Forecast integrity: a forecast must never silently repeat an arbitrary initialization.
Alongside the point estimate, :func:`aggregate` reports the same aggregation taken over
each branch's *pre-simulation* terminal answer, the shift the simulation produced,
whether every branch weight is grounded in an identified distribution, and bounds that
survive any redistribution of the ungrounded weights. The live failure this guards: a
0.5/0.5 symmetric-ignorance split whose two branches resolved YES and NO, reported as
0.5000 — the prior repeated, with the simulation as decoration.

Three gates run here, all universal — no question, family or domain appears in this
module:

* **D1/FI-1 — the validity triple.** ``trace_reproducible``,
  ``causal_simulation_valid`` and ``point_estimate_calibrated`` are three separate
  fields with three separate bases. Each defaults to "not assessed", never to true.
* **D2/FI-2 — headline suppression.** A point estimate is withheld when ungrounded
  branch weights disagree, when it would condition on a minority of branch mass, or
  when ungrounded numeric alternatives straddle the terminal threshold. What is
  published instead is: no point estimate, the honest bounds, and the reason. The
  scenario average survives in diagnostics and is never the answer. FD-17 is why the
  rule is a disjunction: a live run published 1.0000 from ONE YES branch with three
  unresolved siblings at equal ungrounded weights, where the disagreement test cannot
  fire because the others were unresolved rather than opposed.
* **D6/FI-3 — the responsibility gate.** When a
  :class:`~sworldmodel.models.ResponsibilityReport` is supplied and its classification
  is not one that may publish, the forecast carries no answer at all.
"""

from __future__ import annotations

from .errors import MassConservationError
from .models import (
    PROBABILITY_SOURCE,
    PROBABILITY_SOURCE_ESTABLISHED,
    PROBABILITY_SOURCE_NOT_PUBLISHABLE,
    PROBABILITY_SOURCE_SUPPRESSED,
    PROBABILITY_SOURCE_UNGROUNDED_WEIGHTS,
    SUPPRESSED_RESOLVED_MASS_IS_A_MINORITY,
    SUPPRESSED_RESPONSIBILITY_GATE,
    SUPPRESSED_THRESHOLD_STRADDLING,
    SUPPRESSED_UNGROUNDED_WEIGHTS_DISAGREE,
    SUPPRESSED_WEIGHTS_UNGROUNDED_FOR_A_CAUSED_RESULT,
    BranchOutcome,
    ForecastIntegrity,
    ForecastResult,
    ForecastStatus,
    ForecastValidity,
    RealityManifest,
    ResolutionContract,
    ResponsibilityReport,
    TrajectorySummary,
    ValidityState,
)
from .responsibility import threshold_straddling_variables

# One sentence per suppression reason, for the artifact and the console. Universal.
_SUPPRESSION_TEXT: dict[str, str] = {
    SUPPRESSED_UNGROUNDED_WEIGHTS_DISAGREE: (
        "branches whose weights are not grounded in any identified distribution "
        "disagree about the answer, so the number is the arbitrary split between them"
    ),
    SUPPRESSED_RESOLVED_MASS_IS_A_MINORITY: (
        "the point estimate would condition on a minority of the branch mass; most of "
        "the world's possibilities never reached an answer"
    ),
    SUPPRESSED_THRESHOLD_STRADDLING: (
        "ungrounded numeric alternatives sit on opposite sides of the terminal "
        "threshold, so the answer is the choice of those numbers"
    ),
    SUPPRESSED_RESPONSIBILITY_GATE: (
        "the responsibility classification forbids publishing an answer for this run"
    ),
    SUPPRESSED_WEIGHTS_UNGROUNDED_FOR_A_CAUSED_RESULT: (
        "the trajectory produced the answer, but the branch weights are not grounded, "
        "so its magnitude is not a calibrated probability"
    ),
}


def aggregate(
    branch_outcomes: tuple[BranchOutcome, ...],
    *,
    truncated_mass: float,
    truncated_reason: str,
    contract: ResolutionContract,
    manifest: RealityManifest,
    trajectory_summaries: tuple[TrajectorySummary, ...],
    trace_location: str,
    model_call_count: int,
    token_usage: int,
    limitations: tuple[str, ...],
    diagnostics: tuple[tuple[str, str], ...] = (),
    world_review_blocking: tuple[str, ...] | None = None,
    responsibility: ResponsibilityReport | None = None,
) -> ForecastResult:
    """Aggregate the branch table into the published result.

    ``world_review_blocking`` is the D1 seam for ``causal_simulation_valid`` and is
    three-valued on purpose: ``None`` (the default) means no world review was assessed
    for the world that was actually simulated and the leg reads "not assessed"; ``()``
    means a review ran and no blocking finding survived; a non-empty tuple names the
    blocking findings that did survive. It never defaults to valid.

    ``responsibility`` is the D6 publication gate, computed in-run by
    :func:`sworldmodel.responsibility.classify_responsibility`. When it is omitted no
    gate is applied — the caller has simply not run it — and that fact is reported
    rather than assumed away.
    """

    yes = sum(b.weight for b in branch_outcomes if b.resolved and b.outcome == "YES")
    no = sum(b.weight for b in branch_outcomes if b.resolved and b.outcome == "NO")
    unresolved = sum(b.weight for b in branch_outcomes if not b.resolved)
    resolved = yes + no
    total_initial = resolved + unresolved + truncated_mass

    if abs(total_initial - 1.0) > 1e-6:
        raise MassConservationError(
            f"branch mass not conserved: yes={yes} no={no} unresolved={unresolved} "
            f"truncated={truncated_mass} total={total_initial}"
        )

    # The scenario average, conditional on *resolved* mass — the fraction of valid
    # terminal worlds that ended YES. No prior fills the unresolved share. Under D2 this
    # figure is a diagnostic; whether any of it is published is decided below.
    scenario_average = (yes / resolved) if resolved > 1e-12 else None
    unresolved_total = unresolved + truncated_mass
    resolved_mass_share = resolved / total_initial if total_initial > 1e-12 else 0.0

    # Bounds. The existing semantics — unresolved mass could fall either way — are kept
    # and extended: mass whose *weight* is ungrounded cannot pin a bound either, because
    # the point weight of such a branch is an enumeration artifact that could be pushed
    # toward or away from YES without contradicting any evidence. Exact redistribution
    # would need the sibling structure of each uncertainty enumeration, which the branch
    # table does not carry, so the sound conservative form is used: only grounded
    # resolved mass narrows the interval (lower = grounded YES mass, upper = 1 minus
    # grounded NO mass). When every weight is grounded this reduces exactly to the
    # previous bounds: yes/total and (yes + unresolved)/total.
    grounded_yes = sum(
        b.weight for b in branch_outcomes if b.resolved and b.outcome == "YES" and b.weight_grounded
    )
    grounded_no = sum(
        b.weight for b in branch_outcomes if b.resolved and b.outcome == "NO" and b.weight_grounded
    )
    lower = grounded_yes / total_initial
    upper = (total_initial - grounded_no) / total_initial

    if resolved <= 1e-12:
        status = ForecastStatus.UNRESOLVED
    elif unresolved_total <= 1e-9:
        status = ForecastStatus.RESOLVED
    else:
        status = ForecastStatus.PARTIALLY_RESOLVED

    # -- forecast integrity ---------------------------------------------------
    # The same aggregation, taken over what the terminal already said when each branch
    # world was initialized (conditions applied, nothing simulated). If the number
    # after simulation is the number before it, the trajectories added nothing.
    pre_yes = sum(b.weight for b in branch_outcomes if b.pre_resolved and b.pre_outcome == "YES")
    pre_no = sum(b.weight for b in branch_outcomes if b.pre_resolved and b.pre_outcome == "NO")
    pre_resolved_mass = pre_yes + pre_no
    pre_unresolved_mass = (
        sum(b.weight for b in branch_outcomes if not b.pre_resolved) + truncated_mass
    )
    p_before = (pre_yes / pre_resolved_mass) if pre_resolved_mass > 1e-12 else None
    shift = (
        (scenario_average - p_before)
        if scenario_average is not None and p_before is not None
        else None
    )

    weights_grounded_all = all(b.weight_grounded for b in branch_outcomes)
    ungrounded_variables = tuple(
        sorted(
            {
                name
                for b in branch_outcomes
                if not b.weight_grounded
                for name, _value in b.key_conditions
            }
        )
    )
    # The point estimate depends materially on arbitrary weights exactly when branches
    # carrying ungrounded weights disagree about the answer: shifting mass between them
    # then moves the number, and nothing grounds where that mass sits.
    ungrounded_answers = {
        b.outcome for b in branch_outcomes if b.resolved and not b.weight_grounded
    }
    answer_depends_on_arbitrary_weights = len(ungrounded_answers) > 1
    # A run in which every resolved branch already carried its final answer at t0 did
    # not produce that answer by simulating: the cited record did. Labeling it
    # "weighted_simulated_trajectories" claims a provenance the trace cannot support,
    # which is exactly how a live OPEC+ run published 1.00 with zero scheduling
    # batches and zero actor invocations under a trajectory label.
    # Restricted to YES on purpose, matching the compile gate's own framing: a terminal
    # already SATISFIED at t0 was carried in by the record, while a terminal that says
    # NO at t0 and still says NO is the ordinary open world in which the actors simply
    # never produced the outcome — a real simulated result, not a citation.
    resolved_branches = [b for b in branch_outcomes if b.resolved]
    established_before_simulation = bool(resolved_branches) and all(
        b.pre_resolved and b.pre_outcome == "YES" and b.outcome == "YES" for b in resolved_branches
    )

    # -- D3/FI-5, the aggregation-time backstop -------------------------------
    # The semantic validator refuses a threshold-straddling plan statically, at the
    # earliest recompilable stage. This is the late net: the same defect read off the
    # branch table the run actually produced, so a world the static gate never saw
    # cannot publish a number that is the choice of two invented values.
    straddling = threshold_straddling_variables(branch_outcomes)

    # -- D2/FI-2, headline suppression ----------------------------------------
    reasons: list[str] = []
    if answer_depends_on_arbitrary_weights:
        reasons.append(SUPPRESSED_UNGROUNDED_WEIGHTS_DISAGREE)
    # FD-17: a MINORITY of branch mass resolving is enough on its own. The live run this
    # comes from published 1.0000 from one YES branch at weight 0.25 with three
    # unresolved siblings — the disagreement test above cannot fire, because the other
    # three were unresolved rather than opposed. Exactly half is not a minority.
    if scenario_average is not None and resolved_mass_share < 0.5 - 1e-9:
        reasons.append(SUPPRESSED_RESOLVED_MASS_IS_A_MINORITY)
    if straddling:
        reasons.append(SUPPRESSED_THRESHOLD_STRADDLING)

    # -- D6/FI-3, the responsibility publication gate --------------------------
    answer_withheld = responsibility is not None and not responsibility.may_publish_answer
    if answer_withheld:
        reasons.append(SUPPRESSED_RESPONSIBILITY_GATE)
    elif responsibility is not None and not responsibility.point_estimate_permitted:
        reasons.append(SUPPRESSED_WEIGHTS_UNGROUNDED_FOR_A_CAUSED_RESULT)

    point_estimate_suppressed = scenario_average is not None and bool(reasons)
    published_probability = (
        None if (answer_withheld or point_estimate_suppressed) else scenario_average
    )
    suppression_reason = _suppression_sentence(reasons, responsibility)
    point_estimate_is_calibrated = published_probability is not None

    if answer_withheld:
        probability_source = PROBABILITY_SOURCE_NOT_PUBLISHABLE
    elif answer_depends_on_arbitrary_weights:
        # The historical label for this shape, kept because it says exactly what the
        # figure in diagnostics is: a scenario enumeration over ungrounded weights.
        probability_source = PROBABILITY_SOURCE_UNGROUNDED_WEIGHTS
    elif point_estimate_suppressed:
        probability_source = PROBABILITY_SOURCE_SUPPRESSED
    elif established_before_simulation:
        probability_source = PROBABILITY_SOURCE_ESTABLISHED
    else:
        probability_source = PROBABILITY_SOURCE

    integrity = ForecastIntegrity(
        probability_before_simulation=p_before,
        probability_after_simulation=scenario_average,
        simulation_shift=shift,
        pre_resolved_mass=pre_resolved_mass,
        pre_unresolved_mass=pre_unresolved_mass,
        weights_grounded_all=weights_grounded_all,
        ungrounded_variables=ungrounded_variables,
        point_estimate_is_calibrated=point_estimate_is_calibrated,
        counterfactual_note=_counterfactual_note(
            branch_outcomes, p_before, scenario_average, shift
        ),
        point_estimate_suppressed=point_estimate_suppressed,
        suppression_reasons=tuple(reasons),
        resolved_mass_share=resolved_mass_share,
        threshold_straddling_variables=straddling,
    )

    validity = _validity(
        published_probability=published_probability,
        suppression_reason=suppression_reason,
        world_review_blocking=world_review_blocking,
        responsibility=responsibility,
        branch_count=len(branch_outcomes),
    )

    return ForecastResult(
        question=contract.question,
        contract=contract,
        integrity_manifest=manifest,
        status=status,
        simulation_probability=published_probability,
        lower_bound=lower,
        upper_bound=upper,
        resolved_mass=resolved,
        unresolved_mass=unresolved_total,
        resolved_yes_mass=yes,
        resolved_no_mass=no,
        trajectory_summaries=trajectory_summaries,
        branch_outcomes=branch_outcomes,
        trace_location=trace_location,
        limitations=limitations
        + (
            (f"truncated mass {truncated_mass:.4f}: {truncated_reason}",)
            if truncated_mass > 0
            else ()
        )
        + ((suppression_reason,) if suppression_reason else ()),
        model_call_count=model_call_count,
        token_usage=token_usage,
        diagnostics=diagnostics,
        probability_source=probability_source,
        integrity=integrity,
        scenario_average=scenario_average,
        point_estimate_suppressed=point_estimate_suppressed,
        point_estimate_suppression_reason=suppression_reason,
        answer_withheld=answer_withheld,
        validity=validity,
        responsibility=responsibility,
    )


def _suppression_sentence(reasons: list[str], responsibility: ResponsibilityReport | None) -> str:
    """The published explanation: what is unavailable, and exactly why."""

    if not reasons:
        return ""
    head = (
        "No answer is published for this run"
        if responsibility is not None and not responsibility.may_publish_answer
        else "Point estimate unavailable"
    )
    why = "; ".join(_SUPPRESSION_TEXT.get(r, r) for r in reasons)
    tail = ""
    if responsibility is not None and not responsibility.may_publish_answer:
        tail = (
            f" Responsibility classification {responsibility.classification}: "
            f"{responsibility.reason}."
        )
    return (
        f"{head} — {why}. The honest result is the reported bounds; the scenario "
        f"average is retained in diagnostics only and is not the answer.{tail}"
    )


def _validity(
    *,
    published_probability: float | None,
    suppression_reason: str,
    world_review_blocking: tuple[str, ...] | None,
    responsibility: ResponsibilityReport | None,
    branch_count: int,
) -> ForecastValidity:
    """D1/FI-1: three questions, three answers, three bases. Never collapsed.

    ``trace_reproducible`` is deliberately the weakest of the three and says so in its
    own basis string. A forensic audit returned RECONSTRUCTED on three live runs and
    two of them were arithmetic over equal, ungrounded weights: reproducibility is a
    statement about arithmetic, and it must never be readable as trustworthiness.
    """

    if responsibility is not None:
        trace_state = (
            ValidityState.VALID if responsibility.trace_reproducible else ValidityState.INVALID
        )
        trace_basis = responsibility.trace_reproducible_basis
    else:
        trace_state = ValidityState.NOT_ASSESSED
        trace_basis = (
            f"the {branch_count} branch outcome(s) aggregate arithmetically and branch mass "
            "is conserved, but no ledger replay was run, so nothing confirms the recorded "
            "trajectory reproduces the published outcomes"
        )

    if world_review_blocking is None:
        causal_state = ValidityState.NOT_ASSESSED
        causal_basis = (
            "no world review was reported for the world that was actually simulated; "
            "this is the absence of an assessment, not a passing one"
        )
    elif world_review_blocking:
        causal_state = ValidityState.INVALID
        causal_basis = (
            "blocking world-review findings survive into the simulated world: "
            + ", ".join(sorted(world_review_blocking))
        )
    else:
        causal_state = ValidityState.VALID
        causal_basis = (
            "a world review ran against the world that was simulated and no blocking "
            "finding survived it"
        )
    if responsibility is not None and not responsibility.trace_reproducible:
        # A record that does not reconstruct cannot evidence a valid causal simulation,
        # whatever the pre-rollout review concluded about the world.
        causal_state = ValidityState.INVALID
        causal_basis = (
            "the recorded trajectory does not reproduce the published outcomes, so the "
            f"simulation record cannot evidence a valid causal world: "
            f"{responsibility.trace_reproducible_basis}"
        )

    if published_probability is not None:
        point_state = ValidityState.VALID
        point_basis = (
            "a point estimate is published: every branch weight bearing on it is grounded "
            "in an identified distribution, the resolved mass is not a minority, and no "
            "ungrounded numeric alternative straddles the terminal threshold"
        )
    else:
        point_state = ValidityState.INVALID
        point_basis = suppression_reason or (
            "no branch reached a terminal answer, so there is no point estimate to calibrate"
        )
    return ForecastValidity(
        trace_reproducible=trace_state,
        causal_simulation_valid=causal_state,
        point_estimate_calibrated=point_state,
        trace_reproducible_basis=trace_basis,
        causal_simulation_valid_basis=causal_basis,
        point_estimate_calibrated_basis=point_basis,
    )


def _counterfactual_note(
    branch_outcomes: tuple[BranchOutcome, ...],
    p_before: float | None,
    p_after: float | None,
    shift: float | None,
) -> str:
    """One sentence: what deleting every actor/process output would leave behind."""

    changed = [
        b for b in branch_outcomes if (b.pre_resolved, b.pre_outcome) != (b.resolved, b.outcome)
    ]
    if not changed:
        return (
            "Deleting every actor and process output would leave the answer unchanged: "
            "each branch already carried its final outcome when its world was initialized."
        )
    # event_count > 0 is how the branch table shows that something actually ran; the
    # engine counts each branch's applied ledger there, so an all-zero table means no
    # simulated event ever executed and the "simulation changed nothing" claim is not
    # made about a simulation that never happened.
    events_ran = any(b.event_count > 0 for b in branch_outcomes)
    if shift is not None and abs(shift) <= 1e-12 and events_ran:
        return (
            f"Actor and process events ran, yet the aggregate probability sits exactly at "
            f"its pre-simulation value ({_fmt(p_after)}): deleting their outputs would alter "
            f"{len(changed)} branch outcome(s) without moving the number."
        )
    return (
        f"Deleting actor and process outputs would change {len(changed)} of "
        f"{len(branch_outcomes)} branch outcomes, leaving a pre-simulation probability of "
        f"{_fmt(p_before)} instead of {_fmt(p_after)}."
    )


def _fmt(p: float | None) -> str:
    return "undetermined" if p is None else f"{p:.4f}"
