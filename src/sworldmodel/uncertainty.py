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
class ScenarioRelease:
    """One moment at which *part* of a branch's hypothesis becomes public.

    A branch that crosses several uncertainties is not one announcement. Each
    uncertainty's value becomes public when *that* uncertainty's evidence says it does,
    and the fields it carries are only the fields that release actually reveals.

    ``at is None`` means the evidence never established a timing: the value is a
    standing condition of the branch from the start of the window, never a dated
    release at an invented moment.
    """

    at: datetime | None
    field_levels: tuple[tuple[str, Any], ...]  # world-field name -> level revealed here
    conditions: tuple[tuple[str, str], ...]  # (variable, outcome_value) established here


@dataclass(frozen=True)
class Scenario:
    """One joint hypothesis about the uncertain future, with its release calendar.

    There is deliberately **no** single ``release_at`` on a branch. Collapsing a
    branch's releases to one moment (the earliest, historically) published every other
    uncertain value at that borrowed date: two variables dated nineteen days apart both
    became public on the earlier date, and a variable whose timing the evidence never
    established was dropped at whatever date its neighbour happened to have. Timing
    belongs to the uncertainty that has one, so it lives per release in ``releases``.
    """

    scenario_id: str
    weight: float
    provenance: WeightProvenance
    provenance_detail: str
    field_levels: tuple[tuple[str, Any], ...]  # world-field name -> level under this branch
    conditions: tuple[tuple[str, str], ...]  # (variable, outcome_value)
    # Every moment at which some part of this branch's hypothesis becomes public,
    # standing conditions first and then in chronological order.
    releases: tuple[ScenarioRelease, ...] = ()

    def dated_releases(self, as_of: datetime) -> tuple[ScenarioRelease, ...]:
        """The releases that must still *fire* inside the run, at their own times.

        A release whose moment lies at or before the cutoff is already public: the
        branch is born knowing it. Only what is genuinely ahead of the cutoff is
        scheduled, so the branch state honestly lacks it until then (TMP-4/FD-7).
        """

        return tuple(
            r for r in self.releases if r.field_levels and r.at is not None and r.at > as_of
        )

    def standing_levels(self, as_of: datetime) -> tuple[tuple[str, Any], ...]:
        """The field levels this branch is born knowing: the ones whose timing the
        evidence never established, plus the ones already public at the cutoff."""

        levels: dict[str, Any] = {}
        for rel in self.releases:
            if rel.at is None or rel.at <= as_of:
                levels.update(dict(rel.field_levels))
        return tuple(sorted(levels.items()))


@dataclass(frozen=True)
class ScenarioSet:
    scenarios: tuple[Scenario, ...]
    truncated_mass: float
    truncated_reason: str


# Provenances that assert no identified distribution: a uniform split adopted for want
# of information ("symmetric_ignorance_assumption" in live artifacts), or a branch kept
# only to expose sensitivity. A weight from either is an enumeration artifact — it says
# how many branches were written, not how likely any of them is.
UNGROUNDED_PROVENANCES = frozenset(
    {WeightProvenance.SYMMETRIC_IGNORANCE, WeightProvenance.SENSITIVITY_ONLY}
)


def weights_grounded(scenario: Scenario) -> bool:
    """Whether this branch's weight is anchored in an identified distribution.

    ``Scenario.provenance`` is already the *weakest* provenance across the variables the
    branch combines (see :func:`_weakest`), so a single symmetric-ignorance component is
    enough to make the whole branch weight ungrounded — multiplying an arbitrary factor
    into an empirical one yields an arbitrary product.
    """

    return scenario.provenance not in UNGROUNDED_PROVENANCES


def _weakest(provs: list[WeightProvenance]) -> WeightProvenance:
    return min(provs, key=lambda p: _PROVENANCE_STRENGTH[p])


def _release_order(at: datetime | None) -> tuple[int, str]:
    """Standing conditions (no established timing) first, then chronological. Compared
    on the ISO string so a naive and an aware stamp never raise mid-sort."""

    return (0, "") if at is None else (1, at.isoformat())


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
        conditions: list[tuple[str, str]] = []
        # Only the genuinely uncertain values travel with the branch, and each travels
        # with ITS OWN moment. A release date belongs to the uncertainty that has one:
        # borrowing the earliest across all of them gave a branch a date for a value
        # whose timing the evidence never established, and published every other value
        # on that borrowed date too — nineteen simulated days in which actors decided
        # on information that did not exist yet (FD-7/TMP-4). Grouping by moment, never
        # merging across moments, is the whole fix.
        by_moment: dict[datetime | None, dict[str, Any]] = {}
        conds_by_moment: dict[datetime | None, list[tuple[str, str]]] = {}
        for name, outcome, spec in zip(var_names, combo, specs, strict=True):
            var_total = sum(o.weight.value for o in spec.outcomes)
            weight *= outcome.weight.value / var_total
            provs.append(outcome.weight.provenance)
            details.append(f"{name}={outcome.value}({outcome.weight.provenance.value})")
            conditions.append((name, outcome.value))
            if not outcome.field_effects:
                # An outcome that changes no world field announces nothing; it is a
                # condition of the branch, not a release.
                continue
            slot = by_moment.setdefault(spec.release_at, {})
            for fld, lvl in outcome.field_effects:
                slot[fld] = lvl
            conds_by_moment.setdefault(spec.release_at, []).append((name, outcome.value))
        sid = "sc_" + "_".join(f"{n}:{v}" for n, v in conditions)
        releases = tuple(
            ScenarioRelease(
                at=moment,
                field_levels=tuple(sorted(by_moment[moment].items())),
                conditions=tuple(sorted(conds_by_moment[moment])),
            )
            for moment in sorted(by_moment, key=_release_order)
        )
        # The branch's end-state levels, folded in the order the world will learn them,
        # so two uncertainties writing one field agree with what the trajectory does.
        levels: dict[str, Any] = {}
        for rel in releases:
            levels.update(dict(rel.field_levels))
        raw.append(
            Scenario(
                scenario_id=sid,
                weight=weight,
                provenance=_weakest(provs),
                provenance_detail="; ".join(details),
                field_levels=tuple(sorted(levels.items())),
                conditions=tuple(conditions),
                releases=releases,
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
