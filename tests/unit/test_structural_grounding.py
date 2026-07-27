"""H-1: an unevidenced structural split must reach the forecast-integrity machinery.

The structure weights come from the assess_structure model call — a guess about WHICH
WORLD WE ARE IN. ``_merge`` used to multiply them into branch mass while leaving every
branch's ``weight_grounded`` True, so ``weights_grounded_all`` stayed True,
``point_estimate_is_calibrated`` stayed True, and the bounds collapsed onto the point:
the whole integrity apparatus bypassed one level up.
"""

from __future__ import annotations

from datetime import datetime

from _fakes import ProgrammableGateway
from sworldmodel.api import _merge, _structure_weight_grounded
from sworldmodel.engine import RunResult
from sworldmodel.evidence import EvidenceStore
from sworldmodel.models import (
    BranchOutcome,
    IntegrityVerdict,
    RealityManifest,
    ResolutionContract,
    WeightProvenance,
)
from sworldmodel.outcomes import aggregate
from sworldmodel.structures import assess_structure
from sworldmodel.worldspec import Expr, ProcessGraph, TerminalExpression, WorldSpec

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

_SPEC = WorldSpec(
    title="t",
    entities=(),
    actors=(),
    fields=(),
    resources=(),
    channels=(),
    documents=(),
    actions=(),
    process=ProcessGraph(),
    terminal=_CONTRACT.terminal,
)


def _assessment(alternative_provenance: str | None):
    """Run the real assess_structure call with a scripted model reply. ``None`` leaves
    the provenance label off entirely, which must fall back to symmetric ignorance."""

    alt: dict[str, object] = {
        "structure_id": "informal_path",
        "what_differs": "the binding decision travels an informal path, not this one",
        "rationale": "the evidence does not settle which path binds",
        "weight": 0.4,
        "could_reverse_outcome": True,
    }
    if alternative_provenance is not None:
        alt["provenance"] = alternative_provenance
    gw = ProgrammableGateway(
        {
            "assess_structure": {
                "is_material": True,
                "reason": "two structures fit the evidence",
                "primary_weight": 0.6,
                "alternatives": [alt],
            }
        }
    )
    assessment, _ = assess_structure(
        _CONTRACT,
        EvidenceStore().view(AS_OF),
        _SPEC,
        gateway=gw,
        seed=0,
    )
    return assessment


def _one_branch_run(outcome: str) -> RunResult:
    """A run whose single branch resolved with a fully grounded scenario weight —
    exactly the input that used to let the structural guess hide."""

    branch = BranchOutcome(
        branch_id="baseline",
        parent_lineage=("root",),
        weight=1.0,
        resolved=True,
        outcome=outcome,
        unresolved_reason=None,
        truncated=False,
        key_conditions=(),
        records=(),
        event_count=3,
        pre_outcome=None,
        pre_resolved=False,
        weight_grounded=True,
    )
    return RunResult(
        branch_outcomes=(branch,),
        trajectory_summaries=(),
        event_ledger=[],
        actor_decisions=[],
        final_worlds={},
        truncated_mass=0.0,
        truncated_reason="",
    )


def _aggregate_merged(assessment) -> object:
    runs = [
        (
            assessment.primary_weight,
            "primary",
            _structure_weight_grounded(assessment, "primary"),
            _one_branch_run("YES"),
        ),
        (
            assessment.alternatives[0].weight,
            "informal_path",
            _structure_weight_grounded(assessment, "informal_path"),
            _one_branch_run("NO"),
        ),
    ]
    merged = _merge(runs)
    return aggregate(
        merged.branch_outcomes,
        truncated_mass=0.0,
        truncated_reason="",
        contract=_CONTRACT,
        manifest=_MANIFEST,
        trajectory_summaries=merged.trajectory_summaries,
        trace_location="",
        model_call_count=2,
        token_usage=10,
        limitations=(),
    )


def test_symmetric_ignorance_structure_weights_reach_the_integrity_record() -> None:
    assessment = _assessment("symmetric_ignorance_assumption")
    result = _aggregate_merged(assessment)

    assert result.integrity.weights_grounded_all is False
    assert result.integrity.point_estimate_is_calibrated is False
    # The structure choice surfaces beside every other ungrounded variable.
    assert "causal_structure" in result.integrity.ungrounded_variables
    # D2/FI-2: the two structures disagree and neither weight is grounded, so no point
    # estimate is published at all. The scenario average survives as a diagnostic, and
    # the bounds — which are the honest answer here — are wider than it at both ends.
    assert result.simulation_probability is None
    assert result.point_estimate_suppressed is True
    p = result.scenario_average
    assert p is not None
    assert result.lower_bound < p < result.upper_bound


def test_unlabeled_structure_provenance_falls_back_to_ungrounded() -> None:
    """structures.py maps an unrecognized/absent provenance label to symmetric
    ignorance — the least defensible kind — and the merge must treat it as such."""

    assessment = _assessment(None)
    assert assessment.alternatives[0].provenance is WeightProvenance.SYMMETRIC_IGNORANCE
    result = _aggregate_merged(assessment)
    assert result.integrity.weights_grounded_all is False
    assert result.integrity.point_estimate_is_calibrated is False


def test_primary_weight_is_ungrounded_when_its_complement_is_a_guess() -> None:
    """The primary's weight is 1 minus the alternatives': if any alternative's weight
    is a symmetric-ignorance guess, the primary's complement is a guess too."""

    assessment = _assessment("symmetric_ignorance_assumption")
    assert _structure_weight_grounded(assessment, "primary") is False
    assert _structure_weight_grounded(assessment, "informal_path") is False


def test_grounded_structural_split_keeps_the_point_calibrated() -> None:
    """A structural split anchored in an identified distribution is not a guess: the
    branches stay grounded and the point estimate keeps its calibration claim."""

    assessment = _assessment("market_or_survey_distribution")
    assert _structure_weight_grounded(assessment, "primary") is True
    assert _structure_weight_grounded(assessment, "informal_path") is True
    result = _aggregate_merged(assessment)
    assert result.integrity.weights_grounded_all is True
    assert result.integrity.point_estimate_is_calibrated is True
    assert result.lower_bound == result.simulation_probability == result.upper_bound


def test_immaterial_assessment_changes_nothing() -> None:
    """When the structure is determined, the single run merges exactly as before."""

    gw = ProgrammableGateway(
        {"assess_structure": {"is_material": False, "reason": "the record settles it"}}
    )
    assessment, _ = assess_structure(
        _CONTRACT, EvidenceStore().view(AS_OF), _SPEC, gateway=gw, seed=0
    )
    assert _structure_weight_grounded(assessment, "primary") is True
    merged = _merge([(assessment.primary_weight, "primary", True, _one_branch_run("YES"))])
    (branch,) = merged.branch_outcomes
    assert branch.weight == 1.0
    assert branch.weight_grounded is True
    assert branch.key_conditions == ()
