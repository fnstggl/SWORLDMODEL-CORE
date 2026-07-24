"""Genuine uncertainty and branching.

Branches represent genuine unknown realities (an uncertain future data release, an
uncertain private reaction), not decorative personality variants. We build a small
joint scenario set — not the Cartesian product of every generated uncertainty — cap
it, and disclose any truncated mass rather than silently renormalizing it away. Every
branch weight carries its provenance.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .errors import MassConservationError, WorldIntegrityError
from .models import UncertaintyOutcome, UncertaintySpec, WeightProvenance

# Ordering from weakest (most epistemically humble) to strongest identification.
# When outcomes with different provenance combine, the branch takes the weakest.
_PROVENANCE_STRENGTH = {
    WeightProvenance.SENSITIVITY_ONLY: 0,
    WeightProvenance.SYMMETRIC_IGNORANCE: 1,
    WeightProvenance.EXPLICIT_MODEL: 2,
    WeightProvenance.CALIBRATED_BEHAVIOR: 3,
    WeightProvenance.MARKET_SURVEY: 4,
    WeightProvenance.DIRECT_EMPIRICAL: 5,
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    weight: float
    provenance: WeightProvenance
    provenance_detail: str
    field_levels: tuple[tuple[str, Any], ...]  # world-field name -> level under this branch
    conditions: tuple[tuple[str, str], ...]  # (variable, outcome_value)
    # When this branch's uncertain value actually becomes public, if the evidence says.
    # ``None`` means it is a standing condition of the branch rather than a dated
    # release: the runtime must not invent a date for it.
    release_at: datetime | None = None


@dataclass(frozen=True)
class ScenarioSet:
    scenarios: tuple[Scenario, ...]
    truncated_mass: float
    truncated_reason: str


def _weakest(provs: list[WeightProvenance]) -> WeightProvenance:
    return min(provs, key=lambda p: _PROVENANCE_STRENGTH[p])


def _refuse_unmodeled_dependence(specs: tuple[UncertaintySpec, ...]) -> None:
    """Refuse to cross two uncertainties the compiler itself said are dependent."""

    present = {s.variable for s in specs}
    for spec in specs:
        clashing = sorted(present & set(spec.depends_on))
        if clashing:
            raise WorldIntegrityError(
                f"uncertainty {spec.variable!r} declares dependence on {clashing}, which are "
                "also modeled as separate uncertainties. Crossing dependent unknowns as if "
                "independent invents a joint distribution. Express them as one uncertainty "
                "whose outcomes are joint states.",
                details={"variable": spec.variable, "depends_on": clashing},
            )


def enumerate_scenarios(
    uncertainties: tuple[UncertaintySpec, ...],
    baseline_fields: dict[str, Any],
    *,
    max_branches: int = 24,
) -> ScenarioSet:
    """Build joint scenarios from the world's uncertainty declarations.

    If there is no declared uncertainty, a single baseline scenario (weight 1.0)
    represents the world as verified.

    Otherwise, variables are combined **only where independence is defensible**. Any
    variable that declares a dependence on another present variable is refused: two
    dependent unknowns multiplied as if independent produce confident joint
    probabilities that nothing supports, and the fix is for the compiler to express
    them as one uncertainty with joint outcomes, not for this function to guess a
    correlation. Independent variables are crossed, ranked, capped at ``max_branches``,
    and any dropped mass is disclosed rather than renormalized away.
    """

    specs = uncertainties
    _refuse_unmodeled_dependence(specs)
    if not specs:
        baseline = Scenario(
            scenario_id="baseline",
            weight=1.0,
            provenance=WeightProvenance.DIRECT_EMPIRICAL,
            provenance_detail="no declared future uncertainty; world taken as verified",
            # Nothing is released into a baseline branch: the verified initial field
            # values are already in the base world. Re-delivering them as an event
            # would make the world's own starting state look like news.
            field_levels=(),
            conditions=(),
        )
        return ScenarioSet(scenarios=(baseline,), truncated_mass=0.0, truncated_reason="")

    # Normalize each variable's outcome weights to sum to 1 (per-variable conservation).
    per_var: list[list[UncertaintyOutcome]] = []
    var_names: list[str] = []
    for spec in specs:
        total = sum(o.weight.value for o in spec.outcomes)
        if total <= 0:
            raise MassConservationError(
                f"uncertainty {spec.variable!r} has non-positive total weight"
            )
        per_var.append(list(spec.outcomes))
        var_names.append(spec.variable)

    raw: list[Scenario] = []
    for combo in itertools.product(*per_var):
        weight = 1.0
        provs: list[WeightProvenance] = []
        details: list[str] = []
        # Only the genuinely uncertain values travel with the branch. The rest of the
        # world is already verified and needs no announcement.
        levels: dict[str, Any] = {}
        conditions: list[tuple[str, str]] = []
        for name, outcome, spec in zip(var_names, combo, specs, strict=True):
            var_total = sum(o.weight.value for o in spec.outcomes)
            weight *= outcome.weight.value / var_total
            provs.append(outcome.weight.provenance)
            details.append(f"{name}={outcome.value}({outcome.weight.provenance.value})")
            for fld, lvl in outcome.field_effects:
                levels[fld] = lvl
            conditions.append((name, outcome.value))
        sid = "sc_" + "_".join(f"{n}:{v}" for n, v in conditions)
        # A release date belongs to the uncertainty that has one. Borrowing the
        # earliest across all of them would give a branch a date for a value whose
        # timing the evidence never established.
        releases = [
            spec.release_at
            for spec, outcome in zip(specs, combo, strict=True)
            if spec.release_at is not None and outcome.field_effects
        ]
        raw.append(
            Scenario(
                scenario_id=sid,
                weight=weight,
                provenance=_weakest(provs),
                provenance_detail="; ".join(details),
                field_levels=tuple(sorted(levels.items())),
                conditions=tuple(conditions),
                release_at=min(releases) if releases else None,
            )
        )

    raw.sort(key=lambda s: (-s.weight, s.scenario_id))
    kept = raw[:max_branches]
    dropped = raw[max_branches:]
    truncated_mass = sum(s.weight for s in dropped)
    reason = (
        f"dropped {len(dropped)} lowest-weight scenarios beyond cap {max_branches}"
        if dropped
        else ""
    )

    total_kept = sum(s.weight for s in kept)
    if abs(total_kept + truncated_mass - 1.0) > 1e-9:
        raise MassConservationError(
            f"scenario mass not conserved: kept {total_kept} + truncated {truncated_mass}"
        )
    return ScenarioSet(
        scenarios=tuple(kept), truncated_mass=truncated_mass, truncated_reason=reason
    )
