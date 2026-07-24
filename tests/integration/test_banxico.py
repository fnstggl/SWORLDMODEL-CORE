"""Banxico vertical slice — the expected roster/facts live HERE, not in core."""

from __future__ import annotations

from _helpers import banxico_corpus, bundle_of, run_dict

# The expected reality — asserted only in this evaluation fixture, never in core.
EXPECTED_ROSTER = {
    "Victoria Rodriguez Ceja",
    "Jonathan Heath",
    "Galia Borja Gomez",
    "Omar Mejia Castelazo",
    "Jose Gabriel Cuadra Garcia",
}
EXPECTED_CUT_COALITION = {
    "victoria_rodriguez_ceja",
    "omar_mejia_castelazo",
    "jose_gabriel_cuadra_garcia",
}
EXPECTED_HOLDERS = {"galia_borja_gomez", "jonathan_heath"}


def test_banxico_verifies_the_real_five_seat_roster() -> None:
    result, ctx, _ = run_dict(banxico_corpus())
    assert result.integrity_manifest.is_verified
    assert result.integrity_manifest.expected_voting_seats == 5
    assert result.integrity_manifest.represented_voting_seats == 5
    names = {a.definition.name for a in ctx.compiled.base_world.actors.values()}
    assert names == EXPECTED_ROSTER


def test_banxico_prior_votes_are_researched_not_assumed() -> None:
    bundle = bundle_of(banxico_corpus())
    cutters = {m.actor_id for m in bundle.members if m.prior_action == "cut"}
    holders = {m.actor_id for m in bundle.members if m.prior_action == "hold"}
    assert cutters == EXPECTED_CUT_COALITION
    assert holders == EXPECTED_HOLDERS
    # every prior vote cites shared-lineage evidence from the one May-7 event
    lineage = bundle.evidence_store.lineage_groups()
    assert len(lineage["banxico_2026_05_07_decision"]) >= 6


def test_banxico_produces_five_final_votes_per_resolved_branch() -> None:
    result, _, _ = run_dict(banxico_corpus())
    resolved = [b for b in result.branch_outcomes if b.resolved]
    assert resolved
    for b in resolved:
        assert len(b.votes) == 5


def test_banxico_forecast_comes_only_from_trajectories() -> None:
    result, _, _ = run_dict(banxico_corpus())
    assert result.probability_source == "weighted_simulated_trajectories"
    assert result.simulation_probability is not None
    yes = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "YES")
    no = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "NO")
    assert abs(result.simulation_probability - yes / (yes + no)) < 1e-9
    # a quiet baseline resolves to a unanimous hold (YES)
    baseline = next(b for b in result.branch_outcomes if b.branch_id.endswith("in_line"))
    assert baseline.outcome == "YES"
    assert set(dict(baseline.votes).values()) == {"hold"}
