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
"""

from __future__ import annotations

from .errors import MassConservationError
from .models import (
    PROBABILITY_SOURCE,
    PROBABILITY_SOURCE_ESTABLISHED,
    PROBABILITY_SOURCE_UNGROUNDED_WEIGHTS,
    BranchOutcome,
    ForecastIntegrity,
    ForecastResult,
    ForecastStatus,
    RealityManifest,
    ResolutionContract,
    TrajectorySummary,
)


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
) -> ForecastResult:
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

    # Simulation probability is conditional on *resolved* mass — the fraction of
    # valid terminal worlds that ended YES. No prior fills the unresolved share.
    sim_prob = (yes / resolved) if resolved > 1e-12 else None
    unresolved_total = unresolved + truncated_mass

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
    shift = (sim_prob - p_before) if sim_prob is not None and p_before is not None else None

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
    if established_before_simulation:
        probability_source = PROBABILITY_SOURCE_ESTABLISHED
    elif answer_depends_on_arbitrary_weights:
        probability_source = PROBABILITY_SOURCE_UNGROUNDED_WEIGHTS
    else:
        probability_source = PROBABILITY_SOURCE
    point_estimate_is_calibrated = sim_prob is not None and not answer_depends_on_arbitrary_weights

    integrity = ForecastIntegrity(
        probability_before_simulation=p_before,
        probability_after_simulation=sim_prob,
        simulation_shift=shift,
        pre_resolved_mass=pre_resolved_mass,
        pre_unresolved_mass=pre_unresolved_mass,
        weights_grounded_all=weights_grounded_all,
        ungrounded_variables=ungrounded_variables,
        point_estimate_is_calibrated=point_estimate_is_calibrated,
        counterfactual_note=_counterfactual_note(branch_outcomes, p_before, sim_prob, shift),
    )

    return ForecastResult(
        question=contract.question,
        contract=contract,
        integrity_manifest=manifest,
        status=status,
        simulation_probability=sim_prob,
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
        ),
        model_call_count=model_call_count,
        token_usage=token_usage,
        diagnostics=diagnostics,
        probability_source=probability_source,
        integrity=integrity,
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
