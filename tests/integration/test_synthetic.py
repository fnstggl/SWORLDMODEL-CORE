"""Generalization proofs: distinct scenarios through one runtime."""

from __future__ import annotations

from _helpers import run_dict, synthetic_corpus


def test_seven_member_council_generalizes() -> None:
    result, _, _ = run_dict(synthetic_corpus("seven_member_council"))
    assert result.integrity_manifest.represented_voting_seats == 7
    assert result.status.value == "resolved"
    # quiet branch -> unanimous approve (YES); mobilized branch -> not unanimous
    outcomes = {b.branch_id: b.outcome for b in result.branch_outcomes}
    assert any(o == "YES" for o in outcomes.values())
    assert any(o == "NO" for o in outcomes.values())


def test_data_shock_changes_actor_plans() -> None:
    result, _, _ = run_dict(synthetic_corpus("data_shock_board"))
    # The crystallized-shock branch flips everyone off the maintain position.
    shock = next(b for b in result.branch_outcomes if "crystallizes" in b.branch_id)
    contained = next(b for b in result.branch_outcomes if "contained" in b.branch_id)
    assert set(dict(shock.votes).values()) == {"change"}  # data shock changed the plan
    assert set(dict(contained.votes).values()) == {"maintain"}
    assert shock.outcome == "NO" and contained.outcome == "YES"
