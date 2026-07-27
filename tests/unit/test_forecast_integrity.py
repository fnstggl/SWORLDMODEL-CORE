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
    PROBABILITY_SOURCE,
    RECONSTRUCTED_MEANS,
    REQUIRED_RESPONSIBILITY_TESTS,
    BranchOutcome,
    IntegrityVerdict,
    RealityManifest,
    ResolutionContract,
    ResponsibilityReport,
    ValidityState,
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


def _aggregate(
    branches: tuple[BranchOutcome, ...],
    truncated_mass: float = 0.0,
    **gates: object,
):
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
        **gates,  # type: ignore[arg-type]
    )


def _report(
    *,
    classification: str = "ACTOR_CAUSED",
    may_publish: bool | None = None,
    point_estimate_permitted: bool | None = None,
    weights_grounded: bool = True,
    trace_reproducible: bool = True,
    reason: str = "the actors produced the answer",
) -> ResponsibilityReport:
    """A responsibility report as :mod:`sworldmodel.responsibility` would emit it.

    The gate itself is exercised end-to-end in ``test_responsibility.py``; here it is a
    fixed input, so these tests pin what the *aggregation* does with each verdict.
    """

    publishes = (
        classification
        in ("ACTOR_CAUSED", "PROCESS_CAUSED", "ACTOR_AND_PROCESS_CAUSED", "FACTUALLY_RESOLVED")
        if may_publish is None
        else may_publish
    )
    return ResponsibilityReport(
        classification=classification,
        may_publish_answer=publishes,
        point_estimate_permitted=(
            publishes and weights_grounded
            if point_estimate_permitted is None
            else point_estimate_permitted
        ),
        reason=reason,
        weights_grounded=weights_grounded,
        trace_reproducible=trace_reproducible,
        trace_reproducible_basis=(
            "every branch outcome reproduces exactly by replaying the recorded ledger "
            "and re-evaluating the terminal, with no model call. This confirms the "
            "arithmetic, nothing more."
            if trace_reproducible
            else "the ledger replay disagrees with the published outcome: b1 does not reproduce"
        ),
        tests_completed=REQUIRED_RESPONSIBILITY_TESTS,
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
    # D2/FI-2: no point estimate is published at all. The scenario average is still
    # exactly the weighted YES trajectories, and it lives in diagnostics — never as the
    # answer, because a label beside a number does not stop the number being read.
    assert forecast.simulation_probability is None
    assert forecast.point_estimate_suppressed
    assert forecast.scenario_average == pytest.approx(0.5)
    assert "Point estimate unavailable" in forecast.point_estimate_suppression_reason
    # ... and before simulation both branches said NO, so the simulation *did* move
    # the scenario average — the branch weights, not the trajectories, put it back at
    # the prior.
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
        "point_estimate_suppressed",
        "suppression_reasons",
        "resolved_mass_share",
        "threshold_straddling_variables",
        "scenario_average_is_diagnostic_only",
    }
    assert d["ungrounded_variables"] == ["job_market_slowing"]
    assert d["suppression_reasons"] == ["ungrounded_branch_weights_disagree"]
    # The scenario average is in the record, and the record says what it is.
    assert d["probability_after_simulation"] == pytest.approx(0.5)
    assert d["scenario_average_is_diagnostic_only"] is True


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

    # The branches resolve YES through the simulation (pre_outcome=None): a branch
    # already YES at t0 is a record-established answer, which is a different claim with
    # its own source label and its own test below.
    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome=None, weight_grounded=False, value="a"),
            _branch("b", 0.5, "YES", pre_outcome=None, weight_grounded=False, value="b"),
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


def test_an_answer_the_record_already_carried_is_not_labeled_a_trajectory() -> None:
    """A live OPEC+ run published 1.00 under probability_source
    'weighted_simulated_trajectories' with zero scheduling batches, zero actor
    invocations, and a terminal already true at t0. The label claimed a provenance the
    trace could not support. When every resolved branch already carried its final
    answer before the first event fired, the source must say so."""

    from sworldmodel.models import PROBABILITY_SOURCE_ESTABLISHED

    settled = _aggregate(
        (_branch("b1", 1.0, "YES", pre_outcome="YES", weight_grounded=True, event_count=0),)
    )
    assert settled.probability_source == PROBABILITY_SOURCE_ESTABLISHED
    assert settled.simulation_probability == 1.0

    # Produced by the run rather than carried into it: the trajectory label stands.
    produced = _aggregate((_branch("b1", 1.0, "YES", pre_outcome=None, weight_grounded=True),))
    assert produced.probability_source == PROBABILITY_SOURCE

    # A settled branch beside one the simulation actually moved is NOT "established".
    mixed = _aggregate(
        (
            _branch("b1", 0.5, "YES", pre_outcome="YES", weight_grounded=True),
            _branch("b2", 0.5, "NO", pre_outcome=None, weight_grounded=True),
        )
    )
    assert mixed.probability_source != PROBABILITY_SOURCE_ESTABLISHED


# ===========================================================================
# FI-1 (D1) — the validity triple, three separate fields, honest defaults
# ===========================================================================


def test_the_validity_triple_is_three_separate_fields_not_one_verdict() -> None:
    forecast = _aggregate(
        (_branch("a", 1.0, "YES", pre_outcome=None, weight_grounded=True),),
        world_review_blocking=(),
    )
    validity = forecast.validity
    assert validity is not None
    assert validity.trace_reproducible is ValidityState.NOT_ASSESSED
    assert validity.causal_simulation_valid is ValidityState.VALID
    assert validity.point_estimate_calibrated is ValidityState.VALID
    d = validity.as_dict()
    assert set(d) == {
        "trace_reproducible",
        "trace_reproducible_basis",
        "causal_simulation_valid",
        "causal_simulation_valid_basis",
        "point_estimate_calibrated",
        "point_estimate_calibrated_basis",
        "all_three_valid",
        "reconstructed_means",
    }
    # Each leg carries its own basis: no reader can quote one as the others.
    assert d["trace_reproducible_basis"]
    assert d["causal_simulation_valid_basis"]
    assert d["point_estimate_calibrated_basis"]


def test_reconstructed_never_reads_as_trustworthy() -> None:
    """The forensic audit returned RECONSTRUCTED on all three live runs, and two of the
    three were arithmetic over equal ungrounded weights. Reproducibility is a statement
    about arithmetic and must say so in its own words."""

    report = _report(trace_reproducible=True, classification="ACTOR_CAUSED")
    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome=None, weight_grounded=False, value="a"),
            _branch("b", 0.5, "NO", pre_outcome=None, weight_grounded=False, value="b"),
        ),
        responsibility=report,
    )
    validity = forecast.validity
    assert validity is not None
    # The trace reproduces exactly ...
    assert validity.trace_reproducible is ValidityState.VALID
    # ... and the forecast is still not trustworthy: no point estimate survives.
    assert validity.point_estimate_calibrated is ValidityState.INVALID
    assert not validity.all_three_valid
    assert forecast.simulation_probability is None
    assert "confirms the arithmetic, nothing more" in validity.trace_reproducible_basis
    assert RECONSTRUCTED_MEANS in validity.as_dict()["reconstructed_means"]
    assert "not a claim" in RECONSTRUCTED_MEANS


def test_causal_validity_defaults_to_not_assessed_never_to_true() -> None:
    """The seam with the world review is three-valued. An absent review is an absent
    assessment; only an actual review with nothing blocking surviving is 'valid'."""

    branches = (_branch("a", 1.0, "YES", pre_outcome=None, weight_grounded=True),)

    absent = _aggregate(branches)
    assert absent.validity is not None
    assert absent.validity.causal_simulation_valid is ValidityState.NOT_ASSESSED
    assert "absence of an assessment" in absent.validity.causal_simulation_valid_basis

    clean = _aggregate(branches, world_review_blocking=())
    assert clean.validity is not None
    assert clean.validity.causal_simulation_valid is ValidityState.VALID

    blocked = _aggregate(
        branches, world_review_blocking=("terminal_preresolved", "decorative_actors")
    )
    assert blocked.validity is not None
    assert blocked.validity.causal_simulation_valid is ValidityState.INVALID
    assert "decorative_actors" in blocked.validity.causal_simulation_valid_basis
    # A world the review condemned does not stop the arithmetic reproducing, and the
    # two facts stay separate — that is the whole point of the triple.
    assert blocked.validity.point_estimate_calibrated is ValidityState.VALID


def test_a_trace_that_does_not_reproduce_invalidates_the_causal_leg_too() -> None:
    report = _report(trace_reproducible=False, classification="INVALID", may_publish=False)
    forecast = _aggregate(
        (_branch("a", 1.0, "YES", pre_outcome=None, weight_grounded=True),),
        world_review_blocking=(),
        responsibility=report,
    )
    validity = forecast.validity
    assert validity is not None
    assert validity.trace_reproducible is ValidityState.INVALID
    # Even though the world review passed, a record that does not reconstruct cannot
    # evidence a valid causal simulation.
    assert validity.causal_simulation_valid is ValidityState.INVALID
    assert "does not reproduce" in validity.causal_simulation_valid_basis


# ===========================================================================
# FI-2 (D2, FD-2, FD-17) — headline suppression
# ===========================================================================


def test_fd17_one_yes_branch_and_three_unresolved_publishes_no_point_estimate() -> None:
    """The live artifact artifacts/phase2/geopolitical2/forecast.json, exactly.

    Four branches at 0.25 with symmetric-ignorance weights; one resolved YES and three
    never resolved. The run published `simulation_probability: 1.0` under
    `weighted_simulated_trajectories` with `point_estimate_is_calibrated: true` and
    0.75 unresolved. FI-2's disagreement test cannot fire — the other three are
    unresolved rather than opposed — so the minority-mass rule is what catches it.
    """

    forecast = _aggregate(
        (
            _branch("sc_ff", 0.25, None, pre_outcome=None, weight_grounded=False, value="ff"),
            _branch("sc_ft", 0.25, None, pre_outcome=None, weight_grounded=False, value="ft"),
            _branch("sc_tf", 0.25, None, pre_outcome=None, weight_grounded=False, value="tf"),
            _branch("sc_tt", 0.25, "YES", pre_outcome=None, weight_grounded=False, value="tt"),
        )
    )
    # The published headline is empty. It was 1.0000.
    assert forecast.simulation_probability is None
    assert forecast.point_estimate_suppressed
    # The stale calibration flag comes out false. It was true.
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.point_estimate_is_calibrated is False
    assert integrity.suppression_reasons == ("resolved_mass_is_a_minority_of_branch_mass",)
    assert integrity.resolved_mass_share == pytest.approx(0.25)
    # The scenario average survives, labelled, in diagnostics only.
    assert forecast.scenario_average == pytest.approx(1.0)
    assert integrity.probability_after_simulation == pytest.approx(1.0)
    # What is published instead: no point estimate, honest bounds, and the reason.
    assert forecast.lower_bound == pytest.approx(0.0)
    assert forecast.upper_bound == pytest.approx(1.0)
    assert "minority of the branch mass" in forecast.point_estimate_suppression_reason
    assert forecast.probability_source == "point_estimate_withheld_scenario_bounds_only"
    # The reason travels with the published limitations, not only in a side record.
    assert any("Point estimate unavailable" in lim for lim in forecast.limitations)


def test_the_minority_rule_is_a_minority_not_a_majority_requirement() -> None:
    """Exactly half the mass resolving is not a minority; the estimate stands."""

    half = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome=None, weight_grounded=True, value="a"),
            _branch("b", 0.5, None, pre_outcome=None, weight_grounded=True, value="b"),
        )
    )
    assert half.simulation_probability == pytest.approx(1.0)
    assert not half.point_estimate_suppressed

    just_under = _aggregate(
        (
            _branch("a", 0.49, "YES", pre_outcome=None, weight_grounded=True, value="a"),
            _branch("b", 0.51, None, pre_outcome=None, weight_grounded=True, value="b"),
        )
    )
    assert just_under.simulation_probability is None
    assert just_under.point_estimate_suppressed


def test_the_minority_rule_fires_on_truncated_mass_too() -> None:
    """Mass dropped by the branch cap is mass that never reached an answer."""

    forecast = _aggregate(
        (_branch("a", 0.4, "YES", pre_outcome=None, weight_grounded=True, value="a"),),
        truncated_mass=0.6,
    )
    assert forecast.simulation_probability is None
    assert forecast.integrity is not None
    assert "resolved_mass_is_a_minority_of_branch_mass" in forecast.integrity.suppression_reasons


def test_fd2_ungrounded_disagreement_suppresses_whatever_the_resolved_share() -> None:
    """The Bank of England / Tesla shape: all mass resolved, and still no point estimate."""

    forecast = _aggregate(
        (
            _branch("a", 0.25, "YES", pre_outcome=None, weight_grounded=False, value="a"),
            _branch("b", 0.25, "NO", pre_outcome=None, weight_grounded=False, value="b"),
            _branch("c", 0.25, "NO", pre_outcome=None, weight_grounded=False, value="c"),
            _branch("d", 0.25, "NO", pre_outcome=None, weight_grounded=False, value="d"),
        )
    )
    assert forecast.resolved_mass == pytest.approx(1.0)
    assert forecast.simulation_probability is None
    assert forecast.scenario_average == pytest.approx(0.25)
    assert forecast.integrity is not None
    assert forecast.integrity.suppression_reasons == ("ungrounded_branch_weights_disagree",)
    assert "not grounded in any identified distribution" in (
        forecast.point_estimate_suppression_reason
    )


def test_grounded_weights_that_disagree_still_publish_a_point_estimate() -> None:
    """Suppression is about ungrounded weights, not about disagreement as such."""

    forecast = _aggregate(
        (
            _branch("a", 0.7, "YES", pre_outcome=None, weight_grounded=True, value="a"),
            _branch("b", 0.3, "NO", pre_outcome=None, weight_grounded=True, value="b"),
        )
    )
    assert forecast.simulation_probability == pytest.approx(0.7)
    assert not forecast.point_estimate_suppressed
    assert forecast.integrity is not None
    assert forecast.integrity.point_estimate_is_calibrated


# ===========================================================================
# FI-5 (D3) — the aggregation-time threshold-straddling backstop
# ===========================================================================


def test_fi5_ungrounded_numeric_alternatives_straddling_the_threshold_suppress() -> None:
    """The Tesla shape at aggregation time: factors 0.8 and 1.05 either side of 0.8331.

    The semantic validator refuses this statically. This is the late net, for a world
    the static gate never saw.
    """

    forecast = _aggregate(
        (
            _branch(
                "lo",
                0.5,
                "NO",
                pre_outcome=None,
                weight_grounded=False,
                variable="seasonal_factor",
                value="0.8",
            ),
            _branch(
                "hi",
                0.5,
                "YES",
                pre_outcome=None,
                weight_grounded=False,
                variable="seasonal_factor",
                value="1.05",
            ),
        )
    )
    assert forecast.simulation_probability is None
    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.threshold_straddling_variables == ("seasonal_factor",)
    assert "threshold_straddling_ungrounded_scenarios" in integrity.suppression_reasons
    assert "opposite sides of the terminal threshold" in (
        forecast.point_estimate_suppression_reason
    )


def test_fi5_does_not_fire_on_grounded_numeric_alternatives() -> None:
    """A cited distribution over numeric draws is a distribution, not an invention."""

    forecast = _aggregate(
        (
            _branch(
                "lo",
                0.5,
                "NO",
                pre_outcome=None,
                weight_grounded=True,
                variable="seasonal_factor",
                value="0.8",
            ),
            _branch(
                "hi",
                0.5,
                "YES",
                pre_outcome=None,
                weight_grounded=True,
                variable="seasonal_factor",
                value="1.05",
            ),
        )
    )
    assert forecast.integrity is not None
    assert forecast.integrity.threshold_straddling_variables == ()
    assert forecast.simulation_probability == pytest.approx(0.5)


# ===========================================================================
# FI-3 / FI-4 (D6, FD-4) — the responsibility publication gate
# ===========================================================================


def test_fd4_a_branch_weights_dominated_run_publishes_no_answer_at_all() -> None:
    """The defect: the classification lived only in the offline forensic script, so a
    BRANCH_WEIGHTS_DOMINATED run published 0.25 normally. It may now publish nothing."""

    report = _report(
        classification="BRANCH_WEIGHTS_DOMINATED",
        may_publish=False,
        point_estimate_permitted=False,
        reason="the probability is the weight of the winning cells",
    )
    forecast = _aggregate(
        (
            _branch("a", 0.25, "YES", pre_outcome=None, weight_grounded=False, value="a"),
            _branch("b", 0.25, "NO", pre_outcome=None, weight_grounded=False, value="b"),
            _branch("c", 0.25, "NO", pre_outcome=None, weight_grounded=False, value="c"),
            _branch("d", 0.25, "NO", pre_outcome=None, weight_grounded=False, value="d"),
        ),
        responsibility=report,
    )
    assert forecast.answer_withheld
    assert forecast.simulation_probability is None
    assert forecast.probability_source == "no_answer_published_responsibility_gate"
    assert "No answer is published for this run" in forecast.point_estimate_suppression_reason
    assert "BRANCH_WEIGHTS_DOMINATED" in forecast.point_estimate_suppression_reason
    # The report travels with the forecast so the artifact carries the whole gate.
    assert forecast.responsibility is report
    assert forecast.integrity is not None
    assert "responsibility_classification_forbids_publication" in (
        forecast.integrity.suppression_reasons
    )
    # The scenario average is still recoverable for diagnosis, and is not the answer.
    assert forecast.scenario_average == pytest.approx(0.25)


@pytest.mark.parametrize(
    "classification",
    ["INITIAL_ASSUMPTIONS_DOMINATED", "BRANCH_WEIGHTS_DOMINATED", "UNRESOLVED", "INVALID"],
)
def test_only_the_four_causing_classifications_may_publish(classification: str) -> None:
    blocked = _aggregate(
        (_branch("a", 1.0, "YES", pre_outcome=None, weight_grounded=True),),
        responsibility=_report(classification=classification, may_publish=False),
    )
    assert blocked.answer_withheld
    assert blocked.simulation_probability is None


@pytest.mark.parametrize(
    "classification",
    ["ACTOR_CAUSED", "PROCESS_CAUSED", "ACTOR_AND_PROCESS_CAUSED", "FACTUALLY_RESOLVED"],
)
def test_the_four_causing_classifications_publish_normally(classification: str) -> None:
    ok = _aggregate(
        (_branch("a", 1.0, "YES", pre_outcome=None, weight_grounded=True),),
        responsibility=_report(classification=classification),
    )
    assert not ok.answer_withheld
    assert ok.simulation_probability == pytest.approx(1.0)


def test_a_caused_result_without_grounded_weights_still_loses_its_point_estimate() -> None:
    """D6, second sentence: actor- and process-caused results still need grounded
    weights before the magnitude is a calibrated probability."""

    report = _report(
        classification="ACTOR_CAUSED",
        may_publish=True,
        point_estimate_permitted=False,
        weights_grounded=False,
    )
    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome=None, weight_grounded=False, value="a"),
            _branch("b", 0.5, "YES", pre_outcome=None, weight_grounded=False, value="b"),
        ),
        responsibility=report,
    )
    # The answer is not withheld — the actors really did produce it ...
    assert not forecast.answer_withheld
    # ... but there is no calibrated number.
    assert forecast.simulation_probability is None
    assert forecast.point_estimate_suppressed
    assert forecast.integrity is not None
    assert "caused_result_without_grounded_branch_weights" in (
        forecast.integrity.suppression_reasons
    )


def test_omitting_the_gate_applies_no_gate_and_says_so() -> None:
    """A caller that has not run the gate gets today's behaviour and an honest
    'not assessed', never a silent pass."""

    forecast = _aggregate((_branch("a", 1.0, "YES", pre_outcome=None, weight_grounded=True),))
    assert forecast.responsibility is None
    assert not forecast.answer_withheld
    assert forecast.simulation_probability == pytest.approx(1.0)
    assert forecast.validity is not None
    assert forecast.validity.trace_reproducible is ValidityState.NOT_ASSESSED
    assert "no ledger replay was run" in forecast.validity.trace_reproducible_basis


# ===========================================================================
# FI-6 (§15) — weight rules
# ===========================================================================


def test_fi6_every_grounded_provenance_names_an_evidence_class() -> None:
    from sworldmodel.models import WeightProvenance
    from sworldmodel.uncertainty import UNGROUNDED_PROVENANCES, WEIGHT_EVIDENCE_CLASSES

    grounded = set(WeightProvenance) - UNGROUNDED_PROVENANCES
    assert set(WEIGHT_EVIDENCE_CLASSES) == grounded
    text = " ".join(WEIGHT_EVIDENCE_CLASSES.values())
    for phrase in (
        "empirical frequency",
        "base rate",
        "current-state evidence",
        "survey or market evidence",
        "comparable cases",
        "explicit model estimate",
        "visible support and stated uncertainty",
    ):
        assert phrase in text


def test_fi6_equal_weights_are_never_automatically_probabilities() -> None:
    """A uniform split with no constraining evidence is symmetric ignorance whatever
    label it wears, and the demotion reaches the branch record."""

    from sworldmodel.models import (
        BranchWeight,
        UncertaintyOutcome,
        UncertaintySpec,
        WeightProvenance,
    )
    from sworldmodel.uncertainty import (
        effective_provenance,
        enumerate_scenarios,
        weight_grounding_defects,
        weights_grounded,
    )

    def outcome(value: str, weight: float) -> UncertaintyOutcome:
        return UncertaintyOutcome(
            value=value,
            weight=BranchWeight(weight, WeightProvenance.DIRECT_EMPIRICAL, "claimed empirical"),
            field_effects=(("s", value),),
        )

    uniform = UncertaintySpec("v", "unknown", True, (outcome("a", 0.5), outcome("b", 0.5)))
    assert effective_provenance(uniform, uniform.outcomes[0]) is (
        WeightProvenance.SYMMETRIC_IGNORANCE
    )
    defects = weight_grounding_defects(uniform)
    assert len(defects) == 2
    assert "equal weights are never automatically probabilities" in defects[0]
    assert all(not weights_grounded(s) for s in enumerate_scenarios((uniform,), {}).scenarios)

    # Cite the distribution the split came from and the claim stands.
    cited = UncertaintySpec(
        "v",
        "unknown",
        True,
        (outcome("a", 0.5), outcome("b", 0.5)),
        constraining_evidence_ids=("c-1e7c9d751ad5",),
    )
    assert effective_provenance(cited, cited.outcomes[0]) is WeightProvenance.DIRECT_EMPIRICAL
    assert weight_grounding_defects(cited) == ()
    assert all(weights_grounded(s) for s in enumerate_scenarios((cited,), {}).scenarios)

    # An asymmetric split is itself a claim someone made, and is not demoted.
    asymmetric = UncertaintySpec("v", "unknown", True, (outcome("a", 0.7), outcome("b", 0.3)))
    assert effective_provenance(asymmetric, asymmetric.outcomes[0]) is (
        WeightProvenance.DIRECT_EMPIRICAL
    )


def test_fi6_an_explicit_model_estimate_needs_visible_support_and_uncertainty() -> None:
    from sworldmodel.models import (
        BranchWeight,
        UncertaintyOutcome,
        UncertaintySpec,
        WeightProvenance,
    )
    from sworldmodel.uncertainty import effective_provenance, weight_grounding_defects

    def outcome(value: str, weight: float, detail: str) -> UncertaintyOutcome:
        return UncertaintyOutcome(
            value=value,
            weight=BranchWeight(weight, WeightProvenance.EXPLICIT_MODEL, detail),
            field_effects=(("s", value),),
        )

    bare = UncertaintySpec("v", "unknown", True, (outcome("a", 0.7, ""), outcome("b", 0.3, "")))
    assert effective_provenance(bare, bare.outcomes[0]) is WeightProvenance.SYMMETRIC_IGNORANCE
    assert "is not a model estimate" in weight_grounding_defects(bare)[0]

    supported = UncertaintySpec(
        "v",
        "unknown",
        True,
        (outcome("a", 0.7, "logit model, +/- 0.08"), outcome("b", 0.3, "logit model, +/- 0.08")),
        constraining_evidence_ids=("c-82fa536178c2",),
    )
    assert effective_provenance(supported, supported.outcomes[0]) is (
        WeightProvenance.EXPLICIT_MODEL
    )
    assert weight_grounding_defects(supported) == ()


def test_fi6_ungrounded_alternatives_stay_available_for_scenario_analysis() -> None:
    """§15 keeps them. What it forbids is their average masquerading as calibrated."""

    forecast = _aggregate(
        (
            _branch("a", 0.5, "YES", pre_outcome=None, weight_grounded=False, value="a"),
            _branch("b", 0.5, "NO", pre_outcome=None, weight_grounded=False, value="b"),
        )
    )
    # Every scenario is still reported, with its weight and its answer.
    assert len(forecast.branch_outcomes) == 2
    assert {b.outcome for b in forecast.branch_outcomes} == {"YES", "NO"}
    # The average exists, in diagnostics, and is not the answer.
    assert forecast.scenario_average == pytest.approx(0.5)
    assert forecast.simulation_probability is None
    assert forecast.validity is not None
    assert forecast.validity.point_estimate_calibrated is ValidityState.INVALID
