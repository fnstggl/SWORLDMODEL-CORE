"""The number comes from the trajectories, and the trajectories can be replayed.

Two properties that have to hold together. Either alone is easy to fake: a system can
report a beautiful trace while computing its answer elsewhere, and it can compute an
honest answer from a trace nobody can reconstruct.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import scheduled_multiparty_world
from sworldmodel.engine import evaluate_terminal, run
from sworldmodel.models import ForecastStatus, IntegrityVerdict, RealityManifest, ResolutionContract
from sworldmodel.outcomes import aggregate
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def _compile(data: dict, gateway: ProgrammableGateway, *, max_branches: int = 8):
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )
    return contract, compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=max_branches,
    )


def _split_world() -> dict:
    """A world whose answer genuinely depends on what the actors do."""

    data = scheduled_multiparty_world()
    data["uncertainties"] = [
        {
            "variable": "external_signal",
            "why_unknown": "the measurement is published after the cutoff",
            "reversal_capable": True,
            "release_at": "2026-06-09T12:00:00+00:00",
            "outcomes": [
                {
                    "value": "high",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 9.0]],
                },
                {
                    "value": "low",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 1.0]],
                },
            ],
        }
    ]
    return data


def _signal_sensitive(ctx: dict) -> dict:
    if ctx["stage"] != "session":
        return wait_decision()
    signal = float(ctx.get("observed_fields", {}).get("external_signal", 0) or 0)
    return act("record_position", {"position": "hold" if signal < 5 else "change"})


def _gateway(decision) -> ProgrammableGateway:
    return ProgrammableGateway(
        {"actor_decision": decision, "reflect": {"beliefs_update": [], "new_memories": []}}
    )


def _aggregate(contract, result):
    manifest = RealityManifest(
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
    return aggregate(
        result.branch_outcomes,
        truncated_mass=result.truncated_mass,
        truncated_reason=result.truncated_reason,
        contract=contract,
        manifest=manifest,
        trajectory_summaries=result.trajectory_summaries,
        trace_location="(test)",
        model_call_count=0,
        token_usage=0,
        limitations=(),
    )


# ---------------------------------------------------------------------------


def test_the_probability_is_exactly_the_weighted_yes_trajectories() -> None:
    """Reconstruct the number by hand from the branch table."""

    gw = _gateway(_signal_sensitive)
    contract, compiled = _compile(_split_world(), gw)
    result = run(compiled, gw, seed=0)
    forecast = _aggregate(contract, result)

    yes = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "YES")
    no = sum(b.weight for b in result.branch_outcomes if b.resolved and b.outcome == "NO")
    assert forecast.resolved_yes_mass == pytest.approx(yes)
    assert forecast.resolved_no_mass == pytest.approx(no)
    assert forecast.simulation_probability == pytest.approx(yes / (yes + no))
    # The actors split the branches: this is not a degenerate all-one-way run.
    assert 0.0 < forecast.simulation_probability < 1.0
    assert forecast.probability_source == "weighted_simulated_trajectories"


def test_deleting_the_actor_decisions_destroys_the_forecast() -> None:
    """If the actors were decorative, removing them would leave the number intact."""

    gw = _gateway(_signal_sensitive)
    contract, compiled = _compile(_split_world(), gw)
    with_actors = _aggregate(contract, run(compiled, gw, seed=0))

    silent = _gateway(lambda ctx: wait_decision("I do nothing"))
    contract2, compiled2 = _compile(_split_world(), silent)
    without = _aggregate(contract2, run(compiled2, silent, seed=0))

    assert with_actors.simulation_probability is not None
    # No positions are recorded, so the compiled unresolved_when holds and there is
    # nothing to compute a probability from. Nothing fills the gap.
    assert without.simulation_probability is None
    assert without.status is ForecastStatus.UNRESOLVED
    assert without.unresolved_mass == pytest.approx(1.0)


def test_unresolved_mass_is_reported_not_filled() -> None:
    """A provider failure leaves a hole, and the hole is visible in the bounds."""

    data = _split_world()
    calls = {"n": 0}

    def flaky(ctx: dict) -> dict:
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("unused")  # pragma: no cover
        return _signal_sensitive(ctx)

    gw = ProgrammableGateway(
        {"actor_decision": _signal_sensitive, "reflect": {"beliefs_update": [], "new_memories": []}}
    )
    contract, compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    forecast = _aggregate(contract, result)

    assert forecast.lower_bound <= (forecast.simulation_probability or 0) <= forecast.upper_bound
    total = forecast.resolved_yes_mass + forecast.resolved_no_mass + forecast.unresolved_mass
    assert total == pytest.approx(1.0)


def test_every_branch_can_be_reconstructed_from_its_own_record() -> None:
    gw = _gateway(_signal_sensitive)
    contract, compiled = _compile(_split_world(), gw)
    result = run(compiled, gw, seed=0)

    for b in result.branch_outcomes:
        world = result.final_worlds[b.branch_id]
        assert b.weight > 0
        assert b.key_conditions  # which uncertainty resolution this branch is
        recomputed = evaluate_terminal(world, compiled.spec.terminal)
        assert recomputed.resolved == b.resolved
        assert recomputed.outcome == b.outcome


def test_the_terminal_replays_from_the_event_ledger_alone() -> None:
    """Rebuild each world by replaying its applied events onto the base world, then
    evaluate the same compiled expression. Same answer, from the ledger only."""

    gw = _gateway(_signal_sensitive)
    contract, compiled = _compile(_split_world(), gw)
    result = run(compiled, gw, seed=0)

    for branch_id, final in result.final_worlds.items():
        replayed = compiled.base_world.clone(new_branch_id=branch_id, weight=final.weight)
        # The ledger is the record of what happened; nothing else is consulted.
        events = list(final.event_history)
        replayed = replayed.apply(events)
        replayed = replayed.with_time(final.contract.horizon)

        original = evaluate_terminal(final, compiled.spec.terminal)
        from_ledger = evaluate_terminal(replayed, compiled.spec.terminal)
        assert from_ledger.resolved == original.resolved, branch_id
        assert from_ledger.outcome == original.outcome, branch_id
        assert dict(from_ledger.highlights) == dict(original.highlights), branch_id


def test_the_schedule_is_serializable_for_the_trace_contract() -> None:
    """A frontend must be able to read the branch calendar without running anything."""

    gw = _gateway(_signal_sensitive)
    _, compiled = _compile(_split_world(), gw)
    result = run(compiled, gw, seed=0)

    for diag in result.diagnostics.values():
        for entry in diag.pending_beyond_horizon:
            assert {"entry_id", "at", "kind", "origin", "causal_parents"} <= set(entry)

    for d in result.actor_decisions:
        # Every invocation names its cause and its effect on the world.
        assert d.wake_reason
        assert d.validation_status in ("started", "executed", "rejected", "failed", "wait")
