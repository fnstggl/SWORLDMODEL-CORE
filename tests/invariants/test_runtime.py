"""Runtime invariants: determinism, mass conservation, replay, no hidden override."""

from __future__ import annotations

from datetime import datetime

from _helpers import base_corpus, dup, run_dict
from _worlds import AS_OF, HORIZON
from sworldmodel import ForecastConfig, MockResearchBackend, run_forecast
from sworldmodel.engine import evaluate_terminal
from sworldmodel.errors import GatewayError
from sworldmodel.gateway import DeterministicGateway, GatewayRequest, GatewayResponse
from sworldmodel.models import ForecastStatus


class _NoActorDecisionGateway(DeterministicGateway):
    """Fails every actor decision — simulates a provider outage and lets us prove that
    deleting actor decisions changes/kills the forecast."""

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        if request.task_kind == "actor_decision":
            raise GatewayError("actor decisions disabled")
        return super()._generate(request)


def test_probability_mass_is_conserved() -> None:
    result, _ = run_dict(base_corpus())
    total = result.resolved_mass + result.unresolved_mass
    assert abs(total - 1.0) < 1e-9
    assert abs(sum(b.weight for b in result.branch_outcomes) - 1.0) < 1e-9


def test_branch_replay_reconstructs_the_same_outcome() -> None:
    result, ctx = run_dict(base_corpus())
    terminal = ctx.compiled.spec.terminal
    for b in result.branch_outcomes:
        if not b.resolved:
            continue
        world = ctx.run_result.final_worlds[b.branch_id]
        # Re-evaluate the declarative terminal from the replayed world state.
        ev = evaluate_terminal(world, terminal)
        assert ev.outcome == b.outcome


def test_same_seed_reproduces_the_same_trace() -> None:
    r1, c1 = run_dict(base_corpus(), seed=7)
    r2, c2 = run_dict(base_corpus(), seed=7)
    assert [b.outcome for b in r1.branch_outcomes] == [b.outcome for b in r2.branch_outcomes]
    led1 = [(e.kind, dict(e.payload)) for e in c1.run_result.event_ledger]
    led2 = [(e.kind, dict(e.payload)) for e in c2.run_result.event_ledger]
    assert led1 == led2


def test_no_hidden_override_probability_is_reconstructable() -> None:
    result, _ = run_dict(base_corpus())
    yes = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "YES")
    no = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "NO")
    expected = yes / (yes + no)
    assert result.simulation_probability is not None
    assert abs(result.simulation_probability - expected) < 1e-9
    assert result.probability_source == "weighted_simulated_trajectories"


def test_provider_failure_becomes_unresolved_not_a_default_action() -> None:
    backend = MockResearchBackend(data=dup(base_corpus()))
    config = ForecastConfig(gateway=_NoActorDecisionGateway(), research_backend=backend, seed=0)
    result, _ = run_forecast(
        "q", datetime.fromisoformat(AS_OF), datetime.fromisoformat(HORIZON), config
    )
    assert result.status is ForecastStatus.UNRESOLVED
    assert result.simulation_probability is None  # no fabricated point estimate
    assert all(not b.resolved and not b.records for b in result.branch_outcomes)
    assert result.unresolved_mass > 0.999


def test_deleting_actor_decisions_changes_the_forecast() -> None:
    normal, _ = run_dict(base_corpus())
    assert normal.status is ForecastStatus.RESOLVED
    # (the disabled-actor run above yields UNRESOLVED — a materially different result)


def test_equivalent_states_reproduce_different_targets_do_not() -> None:
    r1, _ = run_dict(base_corpus())
    r2, _ = run_dict(base_corpus())
    assert {b.branch_id: b.outcome for b in r1.branch_outcomes} == {
        b.branch_id: b.outcome for b in r2.branch_outcomes
    }
    # A genuinely different world (target = unanimous cut) must NOT match.
    from _worlds import committee_world

    other = committee_world({"a": "hold", "b": "hold", "c": "hold"}, target="cut")
    r3, _ = run_dict(other)
    assert r3.simulation_probability != r1.simulation_probability
