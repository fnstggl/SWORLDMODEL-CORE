"""Generality: distinct process types through ONE runtime and entry point."""

from __future__ import annotations

from _helpers import run_dict, synthetic_corpus
from sworldmodel.models import EventKind


def test_individual_response_uses_the_actor_action_terminal() -> None:
    result, ctx, _ = run_dict(synthetic_corpus("individual_response"))
    assert result.contract.terminal_predicate.mechanism == "actor_action"
    assert result.status.value == "resolved"
    yes = next(b for b in result.branch_outcomes if b.outcome == "YES")
    no = next(b for b in result.branch_outcomes if b.outcome == "NO")
    # The YES branch resolves because the focal actor actually took an action.
    yes_world = ctx.run_result.final_worlds[yes.branch_id]
    assert any(e.kind == EventKind.COMMITMENT_MADE for e in yes_world.event_history)
    no_world = ctx.run_result.final_worlds[no.branch_id]
    assert not any(e.kind == EventKind.COMMITMENT_MADE for e in no_world.event_history)


def test_population_strata_use_weighted_majority() -> None:
    result, _, _ = run_dict(synthetic_corpus("population_strata"))
    assert result.integrity_manifest.represented_voting_seats == 3
    quiet = next(b for b in result.branch_outcomes if "quiet" in b.branch_id)
    # weight-40 + weight-35 approve vs weight-25 reject -> approve carries the mass.
    assert quiet.outcome == "YES"
    assert dict(quiet.votes)["stratum_opposed"] == "reject"  # heterogeneity preserved


def test_four_process_types_share_one_entry_point() -> None:
    names = ["data_shock_board", "seven_member_council", "population_strata", "individual_response"]
    for name in names:
        result, _, _ = run_dict(synthetic_corpus(name))
        assert result.probability_source == "weighted_simulated_trajectories"
        assert result.status.value == "resolved"
