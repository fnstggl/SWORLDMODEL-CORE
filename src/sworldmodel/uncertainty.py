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

from .errors import MassConservationError
from .models import ScenarioFrame, UncertaintyOutcome, WeightProvenance

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
    signal_levels: tuple[tuple[str, float], ...]
    conditions: tuple[tuple[str, str], ...]  # (signal, outcome_value)


@dataclass(frozen=True)
class ScenarioSet:
    scenarios: tuple[Scenario, ...]
    truncated_mass: float
    truncated_reason: str


def _weakest(provs: list[WeightProvenance]) -> WeightProvenance:
    return min(provs, key=lambda p: _PROVENANCE_STRENGTH[p])


def enumerate_scenarios(frame: ScenarioFrame, *, max_branches: int = 24) -> ScenarioSet:
    """Build joint scenarios from the frame's uncertainty declarations.

    If there is no declared uncertainty, a single baseline scenario (weight 1.0)
    represents the world as verified. Otherwise we take the product across uncertain
    signals (each is a distinct external event, so independence is defensible), rank
    by weight, cap at ``max_branches``, and disclose any dropped mass.
    """

    specs = frame.uncertainty
    if not specs:
        baseline = Scenario(
            scenario_id="baseline",
            weight=1.0,
            provenance=WeightProvenance.DIRECT_EMPIRICAL,
            provenance_detail="no declared future uncertainty; world taken as verified",
            signal_levels=tuple(sorted((s.name, s.baseline) for s in frame.signals)),
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
                f"uncertainty {spec.signal!r} has non-positive total weight"
            )
        per_var.append(list(spec.outcomes))
        var_names.append(spec.signal)

    baseline_levels = {s.name: s.baseline for s in frame.signals}
    raw: list[Scenario] = []
    for combo in itertools.product(*per_var):
        weight = 1.0
        provs: list[WeightProvenance] = []
        details: list[str] = []
        levels = dict(baseline_levels)
        conditions: list[tuple[str, str]] = []
        for name, outcome, spec in zip(var_names, combo, specs, strict=True):
            var_total = sum(o.weight.value for o in spec.outcomes)
            weight *= outcome.weight.value / var_total
            provs.append(outcome.weight.provenance)
            details.append(f"{name}={outcome.value}({outcome.weight.provenance.value})")
            for sig, lvl in outcome.signal_effects:
                levels[sig] = lvl
            conditions.append((name, outcome.value))
        sid = "sc_" + "_".join(f"{n}:{v}" for n, v in conditions)
        raw.append(
            Scenario(
                scenario_id=sid,
                weight=weight,
                provenance=_weakest(provs),
                provenance_detail="; ".join(details),
                signal_levels=tuple(sorted(levels.items())),
                conditions=tuple(conditions),
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
