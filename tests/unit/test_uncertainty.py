from __future__ import annotations

from sworldmodel.models import (
    BranchWeight,
    ReactionRule,
    ScenarioFrame,
    SignalDef,
    UncertaintyOutcome,
    UncertaintySpec,
    WeightProvenance,
)
from sworldmodel.uncertainty import enumerate_scenarios


def _outcome(value: str, w: float, level: float) -> UncertaintyOutcome:
    return UncertaintyOutcome(
        value=value,
        weight=BranchWeight(w, WeightProvenance.EXPLICIT_MODEL, "test"),
        signal_effects=(("s", level),),
    )


def _frame(specs: tuple[UncertaintySpec, ...]) -> ScenarioFrame:
    return ScenarioFrame(
        options=("a", "b"),
        signals=(SignalDef("s", 0.0, "signal"),),
        reaction_rules=(ReactionRule("s", "above", 0.5, "b", "r"),),
        guidance_option="a",
        guidance_text="",
        acceptance_tolerance=0.5,
        uncertainty=specs,
    )


def test_no_uncertainty_gives_single_baseline() -> None:
    ss = enumerate_scenarios(_frame(()))
    assert len(ss.scenarios) == 1
    assert ss.scenarios[0].weight == 1.0
    assert ss.truncated_mass == 0.0


def test_product_weights_and_mass_conserved() -> None:
    specs = (
        UncertaintySpec("s", "unknown", True, (_outcome("lo", 0.6, 0.0), _outcome("hi", 0.4, 0.6))),
        UncertaintySpec("t", "unknown", True, (_outcome("lo", 0.5, 0.0), _outcome("hi", 0.5, 0.6))),
    )
    ss = enumerate_scenarios(_frame(specs))
    assert len(ss.scenarios) == 4
    assert abs(sum(s.weight for s in ss.scenarios) + ss.truncated_mass - 1.0) < 1e-9


def test_cap_discloses_truncated_mass() -> None:
    outcomes = tuple(_outcome(f"o{i}", 1.0, float(i)) for i in range(6))
    spec = UncertaintySpec("s", "unknown", True, outcomes)
    ss = enumerate_scenarios(_frame((spec,)), max_branches=3)
    assert len(ss.scenarios) == 3
    assert ss.truncated_mass > 0.0
    assert "dropped" in ss.truncated_reason
    assert abs(sum(s.weight for s in ss.scenarios) + ss.truncated_mass - 1.0) < 1e-9


def test_provenance_recorded_on_every_scenario() -> None:
    specs = (
        UncertaintySpec("s", "unknown", True, (_outcome("lo", 0.7, 0.0), _outcome("hi", 0.3, 0.6))),
    )
    ss = enumerate_scenarios(_frame(specs))
    assert all(isinstance(s.provenance, WeightProvenance) for s in ss.scenarios)
