"""Forecast integrity: a forecast must never silently repeat an arbitrary initialization.

The live failure these tests pin down: a Bank of England run compiled one uncertainty
with outcomes True 0.5 / False 0.5, both provenance "symmetric_ignorance_assumption".
Branch True resolved YES, branch False resolved NO, and the final probability was
0.5000 — exactly the prior. The number carried no information the branch weights did
not already contain, and 0.5/0.5 was never grounded in any evidence. `aggregate` must
say so: in the probability source, in the bounds, and in the integrity record.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from sworldmodel.models import (
    BranchOutcome,
    IntegrityVerdict,
    RealityManifest,
    ResolutionContract,
)
from sworldmodel.outcomes import aggregate
from sworldmodel.worldspec import Expr, TerminalExpression

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")

_CONTRACT = ResolutionContract(
    question="q",
    as_of=AS_OF,
    horizon=HORIZON,
    subject_entity="subject",
    resolution_units="bool",
    terminal=TerminalExpression(yes_when=Expr("equals", (Expr("field", ("x",)), True))),
)

_MANIFEST = RealityManifest(
    verified_entities=(),
    verified_roles=(),
    verified_authorities=(),
    verified_rules=(),
    verified_previous_actions=(),
    unresolved_conflicts=(),
    missing_required_facts=(),
    evidence_coverage=1.0,
    integrity_verdict=IntegrityVerdict.VERIFIED,
)


def _branch(
    branch_id: str,
    weight: float,
    outcome: str | None,
    *,
    pre_outcome: str | None,
    weight_grounded: bool,
    resolved: bool | None = None,
    pre_resolved: bool | None = None,
    variable: str = "job_market_slowing",
    value: str = "",
    event_count: int = 3,
) -> BranchOutcome:
    return BranchOutcome(
        branch_id=branch_id,
        parent_lineage=("root",),
        weight=weight,
        resolved=outcome is not None if resolved is None else resolved,
        outcome=outcome,
        unresolved_reason=None if outcome is not None else "did not resolve",
        truncated=False,
        key_conditions=((variable, value or branch_id),),
        event_count=event_count,
        pre_outcome=pre_outcome,
        pre_resolved=pre_outcome is not None if pre_resolved is None else pre_resolved,
        weight_grounded=weight_grounded,
    )


def _aggregate(branches: tuple[BranchOutcome, ...], truncated_mass: float = 0.0):
    return aggregate(
        branches,
        truncated_mass=truncated_mass,
        truncated_reason="cap" if truncated_mass else "",
        contract=_CONTRACT,
        manifest=_MANIFEST,
        trajectory_summaries=(),
        trace_location="(test)",
        model_call_count=0,
        token_usage=0,
        limitations=(),
    )


# ---------------------------------------------------------------------------


def test_ungrounded_symmetric_split_is_flagged_not_presented_as_a_finding() -> None:
    """The Bank of England shape: 0.5/0.5 symmetric ignorance, branches YES and NO."""

    forecast = _aggregate(
        (
            _branch("sc_true", 0.5, "YES", pre_outcome="NO", weight_grounded=False, value="true"),
            _branch("sc_false", 0.5, "NO", pre_outcome="NO", weight_grounded=False, value="false"),
        )
    )
    # The point estimate is still exactly the weighted YES trajectories ...
    assert forecast.simulation_probability == pytest.approx(0.5)
    # ... but before simulation both branches said NO, so the simulation *did* move
    # the answer — the branch weights, not the trajectories, put it back at the prior.
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.probability_before_simulation == pytest.approx(0.0)
    assert integrity.probability_after_simulation == pytest.approx(0.5)
    assert integrity.simulation_shift == pytest.approx(0.5)
    # The weights are arbitrary and the branches disagree: the source says so, the
    # bounds refuse to be narrowed by arbitrary mass, and the point is uncalibrated.
    assert forecast.probability_source == "scenario_enumeration_ungrounded_weights"
    assert forecast.lower_bound == pytest.approx(0.0)
    assert forecast.upper_bound == pytest.approx(1.0)
    assert not integrity.point_estimate_is_calibrated
    assert not integrity.weights_grounded_all
    assert integrity.ungrounded_variables == ("job_market_slowing",)
    # The record is serializable for the trace, with every promised key.
    d = integrity.as_dict()
    assert set(d) == {
        "probability_before_simulation",
        "probability_after_simulation",
        "simulation_shift",
        "pre_resolved_mass",
        "pre_unresolved_mass",
        "weights_grounded_all",
        "ungrounded_variables",
        "point_estimate_is_calibrated",
        "counterfactual_note",
    }
    assert d["ungrounded_variables"] == ["job_market_slowing"]


def test_the_same_split_with_grounded_weights_is_calibrated() -> None:
    forecast = _aggregate(
        (
            _branch("sc_true", 0.5, "YES", pre_outcome="NO", weight_grounded=True, value="true"),
            _branch("sc_false", 0.5, "NO", pre_outcome="NO", weight_grounded=True, value="false"),
        )
    )
    assert forecast.simulation_probability == pytest.approx(0.5)
    # Source and bounds keep the existing behavior exactly.
    assert forecast.probability_source == "weighted_simulated_trajectories"
    assert forecast.lower_bound == pytest.approx(0.5)
    assert forecast.upper_bound == pytest.approx(0.5)
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.point_estimate_is_calibrated
    assert integrity.weights_grounded_all
    assert integrity.ungrounded_variables == ()


def test_agreeing_ungrounded_branches_do_not_flag_the_source() -> None:
    """When every ungrounded branch says the same thing, the arbitrary weights cannot
    move the number, so the point estimate does not depend on them."""

    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome="YES", weight_grounded=False, value="a"),
            _branch("b", 0.5, "YES", pre_outcome="YES", weight_grounded=False, value="b"),
        )
    )
    assert forecast.simulation_probability == pytest.approx(1.0)
    assert forecast.probability_source == "weighted_simulated_trajectories"
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.point_estimate_is_calibrated
    assert not integrity.weights_grounded_all  # still disclosed
    # The bounds still refuse the ungrounded mass: nothing grounded pins them.
    assert forecast.lower_bound == pytest.approx(0.0)
    assert forecast.upper_bound == pytest.approx(1.0)


def test_branches_the_simulation_never_changed_are_named_in_the_counterfactual() -> None:
    forecast = _aggregate(
        (
            _branch("a", 0.6, "YES", pre_outcome="YES", weight_grounded=True, value="a"),
            _branch("b", 0.4, "NO", pre_outcome="NO", weight_grounded=True, value="b"),
        )
    )
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.simulation_shift == pytest.approx(0.0)
    # Every branch already carried its final answer at initialization: deleting the
    # actor/process outputs would leave the forecast unchanged, and the note says so.
    assert "unchanged" in integrity.counterfactual_note
    assert integrity.probability_before_simulation == pytest.approx(
        integrity.probability_after_simulation
    )


def test_a_changed_forecast_reports_what_deletion_would_leave() -> None:
    forecast = _aggregate(
        (
            _branch("a", 0.6, "YES", pre_outcome="NO", weight_grounded=True, value="a"),
            _branch("b", 0.4, "NO", pre_outcome="NO", weight_grounded=True, value="b"),
        )
    )
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.simulation_shift == pytest.approx(0.6)
    assert "unchanged" not in integrity.counterfactual_note
    assert "1 of 2" in integrity.counterfactual_note


def test_unresolved_branches_stay_unresolved_before_and_after() -> None:
    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome="YES", weight_grounded=True, value="a"),
            _branch("b", 0.5, None, pre_outcome=None, weight_grounded=True, value="b"),
        )
    )
    # After: the unresolved branch is unresolved mass, not a filled-in outcome.
    assert forecast.unresolved_mass == pytest.approx(0.5)
    assert forecast.simulation_probability == pytest.approx(1.0)
    integrity = forecast.integrity
    assert integrity is not None
    # Before: the same branch contributes to pre-unresolved mass, and the
    # pre-simulation probability conditions on what did pre-resolve.
    assert integrity.pre_unresolved_mass == pytest.approx(0.5)
    assert integrity.pre_resolved_mass == pytest.approx(0.5)
    assert integrity.probability_before_simulation == pytest.approx(1.0)


def test_truncated_mass_counts_as_unresolved_in_both_aggregations() -> None:
    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome="YES", weight_grounded=True, value="a"),
            _branch("b", 0.25, "NO", pre_outcome="NO", weight_grounded=True, value="b"),
        ),
        truncated_mass=0.25,
    )
    assert forecast.unresolved_mass == pytest.approx(0.25)
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.pre_unresolved_mass == pytest.approx(0.25)
    assert integrity.probability_before_simulation == pytest.approx(0.5 / 0.75)


def test_nothing_pre_resolved_means_no_pre_probability_not_a_zero() -> None:
    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome=None, weight_grounded=True, value="a"),
            _branch("b", 0.5, "NO", pre_outcome=None, weight_grounded=True, value="b"),
        )
    )
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.probability_before_simulation is None
    assert integrity.simulation_shift is None
    assert integrity.pre_unresolved_mass == pytest.approx(1.0)
    # The simulation genuinely produced the answer here; that is the healthy case.
    assert integrity.point_estimate_is_calibrated


def test_old_style_branch_construction_still_works_with_safe_defaults() -> None:
    """Existing hand-built records (replays, older tests) omit the new fields."""

    legacy = BranchOutcome(
        branch_id="legacy",
        parent_lineage=("root",),
        weight=1.0,
        resolved=True,
        outcome="YES",
        unresolved_reason=None,
        truncated=False,
        key_conditions=(("v", "x"),),
    )
    assert legacy.pre_outcome is None
    assert legacy.pre_resolved is False
    assert legacy.weight_grounded is True
    forecast = _aggregate((legacy,))
    assert forecast.simulation_probability == pytest.approx(1.0)
    assert forecast.probability_source == "weighted_simulated_trajectories"


def test_scenario_groundedness_helper_reads_the_weakest_provenance() -> None:
    from sworldmodel.models import WeightProvenance
    from sworldmodel.uncertainty import Scenario, weights_grounded

    def scenario(prov: WeightProvenance) -> Scenario:
        return Scenario(
            scenario_id="s",
            weight=0.5,
            provenance=prov,
            provenance_detail="test",
            field_levels=(),
            conditions=(("v", "x"),),
        )

    assert not weights_grounded(scenario(WeightProvenance.SYMMETRIC_IGNORANCE))
    assert not weights_grounded(scenario(WeightProvenance.SENSITIVITY_ONLY))
    assert weights_grounded(scenario(WeightProvenance.DIRECT_EMPIRICAL))
    assert weights_grounded(scenario(WeightProvenance.MARKET_SURVEY))
    assert weights_grounded(scenario(WeightProvenance.CALIBRATED_BEHAVIOR))
    assert weights_grounded(scenario(WeightProvenance.EXPLICIT_MODEL))
