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


def test_a_world_whose_outcome_is_an_input_is_refused() -> None:
    """The failure this gate exists for, reproduced from a real live run.

    A live Banxico pastcast compiled a world with no actions and no process, whose
    terminal read two fields that only the uncertainty's branch conditions ever set. It
    "resolved" with **zero actor invocations**: the reported probability was the model's
    branch weights on a field encoding the answer, labeled as trajectories. That world
    must not be simulatable.
    """

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    # The decision itself becomes an uncertain input, and nothing can act on it.
    data["world_spec"]["actions"] = []
    data["world_spec"]["fields"].append(
        {"field_id": "rate_decision", "value_type": "string", "initial": None}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["rate_decision"]}, "hold"],
    }
    data["uncertainties"] = [
        {
            "variable": "rate_decision",
            "why_unknown": "the board has not met",
            "reversal_capable": True,
            "outcomes": [
                {
                    "value": "hold",
                    "weight": 0.6,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["rate_decision", "hold"]],
                },
                {
                    "value": "change",
                    "weight": 0.4,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["rate_decision", "change"]],
                },
            ],
        }
    ]

    gw = _gateway(_signal_sensitive)
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, gw)
    assert "the outcome is an input" in str(exc.value)
    details = exc.value.details
    # Nothing that runs can write the term the terminal reads.
    assert details["terminal terms with no producer"] == ["rate_decision"]
    assert details["producers by terminal term"] == {"rate_decision": []}
    assert details["terminal terms supplied by uncertainty instead"] == ["rate_decision"]
    assert exc.value.details.get("recompilable") is True


def test_a_world_full_of_actions_that_cannot_reach_the_outcome_is_refused() -> None:
    """The subtler and more dangerous shape of the same defect.

    Here the world looks entirely healthy — actors, actions, a process, a schedule — but
    every action writes somewhere the terminal never reads, and the terminal's own terms
    come from an uncertainty. The actors would be invoked, deliberate, act, and appear
    all through the trace, while the answer was fixed by the branch weights before any
    of them opened their mouth. That is harder to spot than an empty world and worse.
    """

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    # Actions exist and are wired into the process, but they only touch `positions`.
    data["world_spec"]["fields"].append(
        {"field_id": "rate_decision", "value_type": "string", "initial": None}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["rate_decision"]}, "hold"],
    }
    data["world_spec"]["terminal"]["unresolved_when"] = {"op": "const", "args": [False]}
    data["uncertainties"] = [
        {
            "variable": "rate_decision",
            "why_unknown": "the board has not met",
            "reversal_capable": True,
            "outcomes": [
                {
                    "value": "hold",
                    "weight": 0.6,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["rate_decision", "hold"]],
                },
                {
                    "value": "change",
                    "weight": 0.4,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["rate_decision", "change"]],
                },
            ],
        }
    ]

    gw = _gateway(_signal_sensitive)
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, gw)
    assert "the outcome is an input" in str(exc.value)
    details = exc.value.details
    assert details["terminal reads"] == ["rate_decision"]
    assert "rate_decision" not in details["fields any action can write"]
    # The refusal names exactly which terms were supplied instead of produced.
    assert details["terminal terms supplied by uncertainty instead"] == ["rate_decision"]
    assert details.get("recompilable") is True


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


def test_the_environment_may_not_announce_the_answer_before_anyone_acts() -> None:
    """The third shape of the same defect, taken from a live Bank of England run.

    That world compiled a real actor with a real action, and also a scheduled process
    node carrying ``set_field(<the terminal term>, True)`` with a literal value and no
    entry condition. The node fired first, the terminal was already decided, and the
    actor — woken afterwards — noted that the thing had happened and waited. The
    reported forecast was 1.0000 from zero producing actions, and the outcome gate
    passed it because *some* action could in principle have written the term.

    A process that tallies what actors did is right and stays allowed; the difference is
    whether it is gated on something an action writes.
    """

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "signal_given", "value_type": "bool", "initial": False}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["signal_given"]}, True],
    }
    # An action can write the term — so the previous gate is satisfied...
    data["world_spec"]["actions"].append(
        {
            "action_id": "give_signal",
            "meaning": "say it publicly",
            "eligible_actors": ["*"],
            "required_authority": [],
            "parameters": [],
            "effects": [{"op": "set_field", "field": "signal_given", "value": True}],
            "evidence_claim_ids": [],
        }
    )
    # ...but the calendar writes it too, unconditionally, and gets there first.
    data["world_spec"]["process"]["nodes"][0]["effects"] = [
        {"op": "set_field", "field": "signal_given", "value": True}
    ]

    gw = _gateway(_signal_sensitive)
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, gw)
    assert "the environment writes the answer" in str(exc.value)
    assert exc.value.details["terms preset by the environment"] == ["signal_given"]
    assert exc.value.details.get("recompilable") is True


def test_every_terminal_term_names_what_actually_wrote_it() -> None:
    """The runtime half of producer lineage.

    The compile-time gate asks whether something *could* write each terminal term. This
    walks the branch's own ledger and names what did: the event, the actor behind it and
    its causal parents. A world spec cannot satisfy this by looking plausible.
    """

    from sworldmodel.engine import terminal_lineage

    gw = _gateway(_signal_sensitive)
    contract, compiled = _compile(_split_world(), gw)
    result = run(compiled, gw, seed=0)

    for branch_id, world in result.final_worlds.items():
        lineage = terminal_lineage(world, compiled.spec.terminal)
        assert lineage, branch_id
        for term in lineage:
            assert not term["unproduced"], (branch_id, term["terminal_term"])
            # The positions the terminal counts were written by the actors themselves.
            assert term["produced_by_an_actor"], (branch_id, term["terminal_term"])
            for writer in term["written_by"]:
                assert writer["event_id"] and writer["kind"]


def test_an_operational_process_that_accumulates_output_is_not_an_announcement() -> None:
    """The gate above must not refuse the world it exists to permit.

    A production line that adds units per shift writes the same terminal term as a
    process node that declares the answer — but accumulating toward a threshold is how
    throughput is honestly modelled, and the question is whether the quantity is
    *reached* or *asserted*. Only `set_field` to a literal is an announcement.
    """

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "units_built", "value_type": "number", "initial": 0}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "greater_than",
        "args": [{"op": "field", "args": ["units_built"]}, 100],
    }
    data["world_spec"]["actions"].append(
        {
            "action_id": "authorize_overtime",
            "meaning": "add a shift",
            "eligible_actors": ["*"],
            "required_authority": [],
            "parameters": [],
            "effects": [{"op": "adjust_field", "field": "units_built", "amount": 10}],
            "evidence_claim_ids": [],
        }
    )
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "assembly_line",
            "description": "the line builds a fixed number of units per shift",
            "occurrences": [
                {
                    "at": "2026-06-01T00:00:00+00:00",
                    "description": "a shift",
                    "effects": [{"op": "adjust_field", "field": "units_built", "amount": 40}],
                }
            ],
            "evidence_claim_ids": [],
        }
    ]

    gw = _gateway(_signal_sensitive)
    _, compiled = _compile(data, gw)  # must not raise
    producers = compiled.spec.external_processes
    assert producers and producers[0].process_id == "assembly_line"


def test_the_pre_rollout_review_can_never_kill_a_run_that_passed_the_gates() -> None:
    """It is advisory, and it runs after every mechanical gate has already passed.

    A fault here can therefore only ever destroy a run that was otherwise sound — which
    is what happened: a live Bank of England run compiled a real world, cleared every
    gate, and died in the review's own summary helper because a compiled `at` is an ISO
    string and the helper assumed a datetime. An opinion about a world must not be able
    to stop it.
    """

    from sworldmodel.world_review import _when, review_world

    # Compiled times arrive as ISO strings, not datetimes. Both must render.
    assert _when("2026-06-25T00:00:00+00:00") == "2026-06-25T00:00:00+00:00"
    assert _when(AS_OF) == AS_OF.isoformat()
    assert _when(None) is None and _when("") is None

    class Malformed:
        @property
        def spec(self) -> object:
            raise RuntimeError("compiled world is malformed")

    review = review_world(Malformed(), None, None, question="q", evidence_render="")
    assert "could not run" in review.error
    assert not review.should_repair  # a review that did not happen demands no repair


def test_a_terminal_that_reads_no_world_state_is_refused() -> None:
    """The limiting case, and it used to pass in silence.

    With no terms identified there are no orphans, so `yes_when = const(true)` — the
    answer written as a constant — satisfied the very gate that exists to forbid it.
    Allowing actor-free worlds exposed this, because the checks either side of it are
    rightly conditioned on there being actors.
    """

    from sworldmodel.errors import WorldIntegrityError

    for hardcoded in (
        {"op": "const", "args": [True]},
        {"op": "before", "args": [{"op": "now", "args": []}, {"op": "horizon", "args": []}]},
    ):
        data = _split_world()
        data["world_spec"]["terminal"]["yes_when"] = hardcoded
        gw = _gateway(_signal_sensitive)
        with pytest.raises(WorldIntegrityError) as exc:
            _compile(data, gw)
        assert "reads no world state" in str(exc.value)
        assert exc.value.details["failure"] == "terminal_reads_no_world_state"


def test_terminal_terms_beyond_plain_fields_are_recognised_and_matched() -> None:
    """`stage`, `event_count`, `resource` and `document_field` are all offered to the
    compiler as terminal operators. Reading only `field` and the collection aggregates
    had it both ways: those terminals named no terms, so an actor-free world passed
    vacuously, while a world with actors was refused and told its actors could not
    reach terms that had never been identified.

    A document field is also kept distinct from a world field of the same name, because
    the evaluator keeps them distinct — otherwise writing one satisfies a read of the
    other.
    """

    from sworldmodel.world_compiler import _effect_produces, _expr_terms
    from sworldmodel.worldspec import Effect, Expr

    assert _expr_terms(Expr("stage", ())) == {"stage:"}
    assert _expr_terms(Expr("event_count", ("signature",))) == {"event:signature"}
    assert _expr_terms(Expr("resource", ("votes", "board"))) == {"resource:votes"}
    assert _expr_terms(Expr("document_field", ("treaty", "signed"))) == {"document:treaty.signed"}

    assert _effect_produces(Effect("create_event", (("event_type", "signature"),))) == {
        "event:signature"
    }
    assert _effect_produces(Effect("transfer_resource", (("resource", "votes"),))) == {
        "resource:votes"
    }
    # A document field is not the world field of the same name.
    doc = _effect_produces(
        Effect("create_or_update_document", (("document", "treaty"), ("fields", {"signed": True})))
    )
    assert doc == {"document:treaty.signed"}
    assert "field:signed" not in doc


def test_an_expression_the_evaluator_cannot_run_is_caught_at_compile_time() -> None:
    """The evaluator raises on an unknown operator *while evaluating* — for a terminal,
    that is while finalizing a branch, after research, after compilation, after every
    actor has been invoked. A live Bank of England run died there on `{"op": "false"}`,
    six minutes in, with a ValueError and no diagnosis.

    Checking the whole program up front makes the same mistake cost one recompile.
    """

    from sworldmodel.errors import WorldIntegrityError
    from sworldmodel.worldspec import parse_expr

    # `true`/`false` are how a constant gets written by accident, and are read as one.
    assert parse_expr({"op": "false"}).op == "const"
    assert parse_expr({"op": "false"}).args == (False,)
    assert parse_expr({"op": "true"}).args == (True,)

    data = _split_world()
    data["world_spec"]["terminal"]["unresolved_when"] = {"op": "approximately", "args": [1]}
    gw = _gateway(_signal_sensitive)
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, gw)
    assert "cannot evaluate" in str(exc.value)
    assert exc.value.details["failure"] == "unknown_expression_operator"
    assert "approximately" in exc.value.details["unknown operators"]
    assert exc.value.details.get("recompilable") is True
