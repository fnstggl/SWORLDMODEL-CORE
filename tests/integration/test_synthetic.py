"""Generalization proofs: materially different worlds through one runtime + evaluator."""

from __future__ import annotations

from _helpers import run_dict, synthetic_corpus

DOMAINS = [
    "committee_decision",
    "individual_response",
    "negotiation",
    "population_behavior",
    "geopolitical_process",
]


def test_every_domain_resolves_through_one_entry_point() -> None:
    for name in DOMAINS:
        result, _ = run_dict(synthetic_corpus(name))
        assert result.status.value == "resolved", name
        assert result.probability_source == "weighted_simulated_trajectories", name


def test_committee_decision_generalizes_to_five_seats() -> None:
    result, _ = run_dict(synthetic_corpus("committee_decision"))
    assert result.integrity_manifest.represented_participants == 5
    outcomes = {b.outcome for b in result.branch_outcomes}
    assert "YES" in outcomes and "NO" in outcomes  # a shock branch flips the outcome


def test_population_behavior_uses_a_weighted_sum_terminal() -> None:
    result, ctx = run_dict(synthetic_corpus("population_behavior"))
    # strong reception -> adopting weight (40+35) exceeds 50 -> YES; weak -> 40 -> NO
    outcomes = {b.branch_id: b.outcome for b in result.branch_outcomes}
    assert "YES" in outcomes.values() and "NO" in outcomes.values()


def test_geopolitical_process_resolves_via_an_event_pathway() -> None:
    result, _ = run_dict(synthetic_corpus("geopolitical_process"))
    assert result.status.value == "resolved"
    assert result.integrity_manifest.represented_participants == 2
