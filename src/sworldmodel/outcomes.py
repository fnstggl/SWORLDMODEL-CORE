"""Terminal aggregation — the forecast is the weighted frequency of YES trajectories.

The reported forecast comes only from simulated trajectory outcomes. No historical
prior, generic outcome prior, numerical consensus model, or combiner may override
the completed trajectories. When mass is genuinely unresolved, we report bounds
rather than inventing a point estimate; if nothing is validly resolved we return no
fabricated probability.
"""

from __future__ import annotations

from .errors import MassConservationError
from .models import (
    BranchOutcome,
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
    lower = yes / total_initial
    upper = (yes + unresolved_total) / total_initial

    if resolved <= 1e-12:
        status = ForecastStatus.UNRESOLVED
    elif unresolved_total <= 1e-9:
        status = ForecastStatus.RESOLVED
    else:
        status = ForecastStatus.PARTIALLY_RESOLVED

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
    )
