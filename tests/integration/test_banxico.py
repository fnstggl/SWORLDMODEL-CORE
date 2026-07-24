"""Banxico vertical slice — the expected roster/facts live HERE, not in core.

The committee-ness of Banxico is entirely compiled DATA (a WorldSpec). The core that
executes it is domain-free; the same runtime resolves the negotiation and population
worlds. This fixture only asserts that the compiled world is faithful and that the
forecast comes solely from simulated trajectories.
"""

from __future__ import annotations

from _helpers import banxico_corpus, run_dict

EXPECTED_ROSTER = {
    "Victoria Rodriguez Ceja",
    "Jonathan Heath",
    "Galia Borja Gomez",
    "Omar Mejia Castelazo",
    "Jose Gabriel Cuadra Garcia",
}


def test_banxico_verifies_the_real_five_seat_roster() -> None:
    result, ctx = run_dict(banxico_corpus())
    assert result.integrity_manifest.is_verified
    assert result.integrity_manifest.expected_participants == 5
    assert result.integrity_manifest.represented_participants == 5
    names = {a.entity.name for a in ctx.compiled.base_world.actors.values()}
    assert names == EXPECTED_ROSTER


def test_banxico_records_five_votes_per_resolved_branch() -> None:
    result, ctx = run_dict(banxico_corpus())
    resolved = [b for b in result.branch_outcomes if b.resolved]
    assert resolved
    for b in resolved:
        world = ctx.run_result.final_worlds[b.branch_id]
        assert len(world.get_records("votes")) == 5


def test_banxico_forecast_comes_only_from_trajectories() -> None:
    result, ctx = run_dict(banxico_corpus())
    assert result.probability_source == "weighted_simulated_trajectories"
    assert result.simulation_probability is not None
    yes = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "YES")
    no = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "NO")
    assert abs(result.simulation_probability - yes / (yes + no)) < 1e-9
    # the in-line baseline resolves to a unanimous hold (YES)
    baseline = next(b for b in result.branch_outcomes if "in_line" in b.branch_id)
    assert baseline.outcome == "YES"
    world = ctx.run_result.final_worlds[baseline.branch_id]
    assert {r["value"] for r in world.get_records("votes")} == {"hold"}


def test_banxico_prior_votes_share_the_may7_lineage() -> None:
    from _helpers import banxico_corpus as _bc
    from sworldmodel import build_bundle_from_dict

    bundle = build_bundle_from_dict(_bc())
    lineage = bundle.evidence_store.lineage_groups()
    assert len(lineage["banxico_2026_05_07_decision"]) >= 6
