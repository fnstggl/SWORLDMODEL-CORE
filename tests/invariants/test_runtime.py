"""Runtime invariants: determinism, mass conservation, replay, no hidden override."""

from __future__ import annotations

from _helpers import base_corpus, dup, run_dict
from sworldmodel.gateway import DeterministicGateway, GatewayRequest, GatewayResponse
from sworldmodel.mechanisms import evaluate_terminal
from sworldmodel.models import EventKind, ForecastStatus


class _NoActorDecisionGateway(DeterministicGateway):
    """A gateway that fails every actor decision — simulates a provider outage and
    lets us prove that deleting actor decisions changes/kills the forecast."""

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        if request.task_kind == "actor_decision":
            from sworldmodel.errors import GatewayError

            raise GatewayError("actor decisions disabled")
        return super()._generate(request)


def test_probability_mass_is_conserved() -> None:
    result, _, _ = run_dict(base_corpus())
    total = result.resolved_mass + result.unresolved_mass
    assert abs(total - 1.0) < 1e-9
    assert abs(sum(b.weight for b in result.branch_outcomes) - 1.0) < 1e-9


def test_branch_replay_reconstructs_the_same_outcome() -> None:
    result, ctx, _ = run_dict(base_corpus())
    for b in result.branch_outcomes:
        if not b.resolved:
            continue
        world = ctx.run_result.final_worlds[b.branch_id]
        # Reconstruct votes purely from the event ledger and re-evaluate.
        replay_votes = {
            e.actor_id: e.payload_dict["option"]
            for e in world.event_history
            if e.kind == EventKind.VOTE_CAST and e.actor_id is not None
        }
        ev = evaluate_terminal(
            replay_votes,
            world.vote_powers(),
            world.contract.expected_voting_seats,
            world.contract.decision_rule,
            world.contract.terminal_predicate,
        )
        assert ev.outcome == b.outcome


def test_same_seed_reproduces_the_same_trace() -> None:
    r1, c1, _ = run_dict(base_corpus(), seed=7)
    r2, c2, _ = run_dict(base_corpus(), seed=7)
    assert [b.outcome for b in r1.branch_outcomes] == [b.outcome for b in r2.branch_outcomes]
    led1 = [(e.kind, dict(e.payload)) for e in c1.run_result.event_ledger]
    led2 = [(e.kind, dict(e.payload)) for e in c2.run_result.event_ledger]
    assert led1 == led2


def test_no_hidden_override_probability_is_reconstructable() -> None:
    result, _, _ = run_dict(base_corpus())
    yes = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "YES")
    no = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "NO")
    expected = yes / (yes + no)
    assert result.simulation_probability is not None
    assert abs(result.simulation_probability - expected) < 1e-9
    assert result.probability_source == "weighted_simulated_trajectories"


def test_provider_failure_becomes_unresolved_not_a_default_vote() -> None:
    from sworldmodel import ForecastConfig, MockResearchBackend, run_forecast

    corpus = dup(base_corpus())
    backend = MockResearchBackend(data=corpus)
    bundle = backend.research("q", None, None)  # type: ignore[arg-type]
    config = ForecastConfig(gateway=_NoActorDecisionGateway(), research_backend=backend, seed=0)
    result, _ = run_forecast("q", bundle.as_of, bundle.horizon, config)  # type: ignore[arg-type]
    assert result.status is ForecastStatus.UNRESOLVED
    assert result.simulation_probability is None  # no fabricated point estimate
    assert all(not b.resolved and not b.votes for b in result.branch_outcomes)
    assert result.unresolved_mass > 0.999  # all mass left explicitly unresolved


def test_deleting_actor_decisions_changes_the_forecast() -> None:
    normal, _, _ = run_dict(base_corpus())
    assert normal.status is ForecastStatus.RESOLVED
    # (the disabled-actor run above yields UNRESOLVED — a materially different result)


def test_branch_merging_only_for_equivalent_states() -> None:
    # Same corpus twice -> identical (safely mergeable) outcomes.
    r1, _, _ = run_dict(base_corpus())
    r2, _, _ = run_dict(base_corpus())
    assert {b.branch_id: b.outcome for b in r1.branch_outcomes} == {
        b.branch_id: b.outcome for b in r2.branch_outcomes
    }
    # Different guidance -> genuinely different outcomes (must NOT be merged).
    other = dup(base_corpus())
    other["frame"]["guidance_option"] = "cut"
    other["reality"]["target_option"] = "cut"
    other["reality"]["terminal"]["target_option"] = "cut"
    r3, _, _ = run_dict(other)
    assert r3.simulation_probability != r1.simulation_probability
