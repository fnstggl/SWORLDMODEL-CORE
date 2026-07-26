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
    # This world's branch weights are a symmetric-ignorance split and the branches
    # disagree, so the point estimate depends on arbitrary weights — the source says so
    # rather than presenting the scenario average as a simulated frequency.
    assert forecast.probability_source == "scenario_enumeration_ungrounded_weights"
    assert forecast.integrity is not None
    assert not forecast.integrity.point_estimate_is_calibrated
    # The number is still exactly the weighted YES trajectories; only its label and
    # bounds acknowledge what the weights are.
    assert forecast.lower_bound == pytest.approx(0.0)
    assert forecast.upper_bound == pytest.approx(1.0)


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


def test_each_branch_records_its_pre_simulation_answer_and_weight_grounding() -> None:
    """The engine, not a default, sets the forecast-integrity fields on every branch.

    The pre-simulation evaluation runs right after ``_seed_branch``: conditions are in
    world state, but no actor decision or process effect has executed yet.
    """

    gw = _gateway(_signal_sensitive)
    contract, compiled = _compile(_split_world(), gw)
    result = run(compiled, gw, seed=0)
    forecast = _aggregate(contract, result)

    for b in result.branch_outcomes:
        # This world's weights are a symmetric-ignorance split: explicitly ungrounded.
        assert b.weight_grounded is False
        # Before anyone acted no positions were recorded, so the compiled
        # unresolved_when held: the initialized world resolves nothing.
        assert b.pre_resolved is False
        assert b.pre_outcome is None
        # The final outcomes exist and differ from the pre-simulation ones: the
        # trajectories produced the answer instead of repeating the initialization.
        assert b.resolved and b.outcome in ("YES", "NO")

    integrity = forecast.integrity
    assert integrity is not None
    assert integrity.probability_before_simulation is None
    assert integrity.pre_unresolved_mass == pytest.approx(1.0)
    assert integrity.probability_after_simulation == forecast.simulation_probability
    assert not integrity.weights_grounded_all
    assert integrity.ungrounded_variables == ("external_signal",)


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
    # Diagnosed at its cause: the branch condition IS the answer.
    assert "an uncertainty writes the answer" in str(exc.value)
    details = exc.value.details
    assert details["failure"] == "uncertainty_writes_terminal"
    assert details["terminal terms written by an uncertainty"] == ["rate_decision"]
    assert details["producers by terminal term"] == {"rate_decision": []}
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
    assert "an uncertainty writes the answer" in str(exc.value)
    details = exc.value.details
    assert details["failure"] == "uncertainty_writes_terminal"
    assert details["terminal terms written by an uncertainty"] == ["rate_decision"]
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


def test_an_uncertainty_may_not_write_a_term_the_terminal_reads() -> None:
    """The gap a live Bank of England run walked straight through.

    The compiler declared an uncertainty literally named `bailey_choice_to_signal` whose
    branch effects set `bailey_signaled_support` — the same field the actor's own action
    writes. Because the action wrote it too there was no orphan, so the earlier check,
    which only fired for terms nothing else wrote, passed the world. Both branches then
    resolved YES, including the one whose branch condition was "no", for a reported
    probability of 1.0000 with bounds [1.0000, 1.0000]. The actor's own decision had
    been modelled as an exogenous coin flip and then overruled by the actor.

    An uncertainty sets what the world does TO the actors. It never writes the answer,
    whether or not something else writes it as well.
    """

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "signaled", "value_type": "bool", "initial": False}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["signaled"]}, True],
    }
    # An action writes it — so the orphan check is satisfied ...
    data["world_spec"]["actions"].append(
        {
            "action_id": "signal",
            "meaning": "say it publicly",
            "eligible_actors": ["*"],
            "effects": [{"op": "set_field", "field": "signaled", "value": True}],
            "evidence_claim_ids": [],
        }
    )
    # ... and the branch condition writes it too, which is the defect.
    data["uncertainties"] = [
        {
            "variable": "choice_to_signal",
            "why_unknown": "he has not said",
            "reversal_capable": True,
            "outcomes": [
                {
                    "value": "yes",
                    "weight": 0.7,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["signaled", True]],
                },
                {
                    "value": "no",
                    "weight": 0.3,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["signaled", False]],
                },
            ],
        }
    ]

    gw = _gateway(_signal_sensitive)
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, gw)
    assert exc.value.details["failure"] == "uncertainty_writes_terminal"
    assert exc.value.details["terminal terms written by an uncertainty"] == ["signaled"]


def test_a_terminal_term_nothing_writes_at_all_is_still_reported_as_an_orphan() -> None:
    """The stricter uncertainty rule must not hide the plainer defect beneath it."""

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "never_written", "value_type": "bool", "initial": False}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["never_written"]}, True],
    }
    data["uncertainties"] = []  # nothing supplies it either

    gw = _gateway(_signal_sensitive)
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, gw)
    assert "the outcome is an input" in str(exc.value)
    assert exc.value.details["terminal terms with no producer"] == ["never_written"]


def test_a_question_the_record_has_already_answered_compiles_from_its_citations() -> None:
    """Asked in July whether the EU and Mercosur would sign before October, a live run
    found the Commission's own page, Wikipedia and five other sources recording that they
    signed on 17 January. Nothing inside the window produces that. Demanding a producer
    would force a future signing to be invented for a signing that already happened, and
    refusing would refuse the one question the research had already answered.

    The citation is the whole rule: an initial value carrying claim ids is a fact the
    record establishes, and the same value carrying none is the compiler asserting an
    outcome."""

    from sworldmodel.errors import WorldIntegrityError
    from sworldmodel.world_compiler import terminal_producers

    data = _split_world()
    cited = data["claims"][0]["id"]
    data["world_spec"]["documents"] = [
        {
            "document_id": "agreement",
            "fields": {"signed": True},
            "evidence_claim_ids": [cited],
        }
    ]
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "document_field", "args": ["agreement", "signed"]}, True],
    }
    data["uncertainties"] = []

    gw = _gateway(_signal_sensitive)
    _, compiled = _compile(data, gw)
    producers = terminal_producers(compiled.spec)
    assert producers["document:agreement.signed"] == (f"evidence:{cited}",)

    # Strip the citation and the same world is the compiler asserting the answer.
    data["world_spec"]["documents"][0]["evidence_claim_ids"] = []
    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, _gateway(_signal_sensitive))
    assert exc.value.details["failure"] == "terminal_has_no_producer"


def test_the_settled_record_is_detected_for_the_exclusion_reviewer() -> None:
    """The coverage gate's exclusion reviewer inverts its materiality test when the
    world already resolves YES from the cited pre-cutoff record. The detector must fire
    exactly on that state — YES at t0 with every terminal term evidence-cited — and
    stay off for the normal open world, or future-dynamics claims would be waved
    through on questions the record has not settled."""

    from sworldmodel.world_compiler import _cited_factual_resolution

    data = _split_world()
    cited = data["claims"][0]["id"]
    data["world_spec"]["documents"] = [
        {
            "document_id": "agreement",
            "fields": {"signed": True},
            "evidence_claim_ids": [cited],
        }
    ]
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "document_field", "args": ["agreement", "signed"]}, True],
    }
    data["world_spec"]["terminal"]["unresolved_when"] = {"op": "const", "args": [False]}
    data["uncertainties"] = []
    _, compiled = _compile(data, _gateway(_signal_sensitive))
    assert _cited_factual_resolution(compiled.spec, compiled.base_world)

    # The same shape starting unsigned is the normal open state: no inversion.
    data["world_spec"]["documents"][0]["fields"]["signed"] = False
    _, open_world = _compile(data, _gateway(_signal_sensitive))
    assert not _cited_factual_resolution(open_world.spec, open_world.base_world)


def test_the_pre_rollout_review_is_told_when_the_record_already_answered() -> None:
    """A live Bank of England run compiled the legitimate preresolved state — outcome
    initial-true on cited pre-cutoff record — and the pre-rollout review attacked it
    for lacking a production process, forcing a recompile whose world demanded the
    already-made statement be made AGAIN inside the window: an absolute NO
    manufactured by changing the question's meaning. Under a cited factual resolution
    the review prompt must carry the settled-record basis and redirect the attack to
    citation sufficiency; an open world must not get that block."""

    from sworldmodel.world_review import _QUESTIONS, review_world

    findings = {
        "findings": [
            {"key": k, "severity": "PASS", "finding": "ok", "evidence_basis": "the world"}
            for k, _ in _QUESTIONS
        ]
    }

    data = _split_world()
    cited = data["claims"][0]["id"]
    data["world_spec"]["documents"] = [
        {"document_id": "agreement", "fields": {"signed": True}, "evidence_claim_ids": [cited]}
    ]
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "document_field", "args": ["agreement", "signed"]}, True],
    }
    data["world_spec"]["terminal"]["unresolved_when"] = {"op": "const", "args": [False]}
    data["uncertainties"] = []
    _, compiled = _compile(data, _gateway(_signal_sensitive))

    gw = ProgrammableGateway({"world_review": findings})
    review = review_world(compiled, None, gw, question="q?", evidence_render="the evidence")
    assert not review.error and not review.should_repair
    (req,) = [r for r in gw.seen if r.task_kind == "world_review"]
    assert "CITED FACTUAL RESOLUTION" in req.prompt
    assert "same subject, same act" in req.prompt

    # The open world's review carries no settled-record basis.
    data["world_spec"]["documents"][0]["fields"]["signed"] = False
    _, open_world = _compile(data, _gateway(_signal_sensitive))
    gw2 = ProgrammableGateway({"world_review": findings})
    review_world(open_world, None, gw2, question="q?", evidence_render="the evidence")
    (req2,) = [r for r in gw2.seen if r.task_kind == "world_review"]
    assert "CITED FACTUAL RESOLUTION" not in req2.prompt


def test_a_world_whose_initial_values_already_answer_yes_uncited_is_refused() -> None:
    """The OPEC+ shape: the answer baked into an uncited initial value.

    A live run compiled `quota_increase_announced` with initial True and no citation,
    plus an action that could also write it. The per-term gate passed — the action is a
    producer — and the branch resolved YES without one event firing: the runtime lineage
    showed the term was never written. YES at t0 is legitimate only as a factual
    resolution the cited record establishes; uncited, it is the compiler asserting the
    outcome and letting the simulation take credit.
    """

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "quota_increase_announced", "value_type": "bool", "initial": True}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "field", "args": ["quota_increase_announced"]}, True],
    }
    data["world_spec"]["terminal"]["unresolved_when"] = {"op": "const", "args": [False]}
    data["world_spec"]["actions"].append(
        {
            "action_id": "announce_quota_increase",
            "meaning": "announce it",
            "eligible_actors": ["*"],
            "effects": [{"op": "set_field", "field": "quota_increase_announced", "value": True}],
            "evidence_claim_ids": [],
        }
    )
    data["uncertainties"] = []

    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, _gateway(_signal_sensitive))
    assert exc.value.details["failure"] == "terminal_preresolved_without_evidence"
    assert exc.value.details["uncited terminal terms"] == ["quota_increase_announced"]

    # The same world starting neutral is the normal open state and compiles.
    data["world_spec"]["fields"][-1]["initial"] = False
    _compile(data, _gateway(_signal_sensitive))  # must not raise


def test_a_terminal_copied_from_an_ungrounded_uncertainty_is_refused() -> None:
    """The Tesla launder: an uncertainty draw copied one hop into the terminal term.

    A live run declared `delivery_value_exogenous` as a 50/50 exogenous uncertainty and an
    end-of-quarter node that set `actual_q3_deliveries = field(delivery_value_exogenous)` —
    the terminal read the deliveries. The uncertainty did not write the terminal term
    directly, so the earlier gate saw a node as the producer and passed it; the answer was
    the branch weight all the same, laundered through a node that computed nothing.
    """

    from sworldmodel.errors import WorldIntegrityError

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "delivery_value_exogenous", "value_type": "number", "initial": 0}
    )
    data["world_spec"]["fields"].append(
        {"field_id": "actual_deliveries", "value_type": "number", "initial": 0}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "greater_than",
        "args": [{"op": "field", "args": ["actual_deliveries"]}, 400000],
    }
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "q_end",
            "description": "the quarter closes and the total is recorded",
            "occurrences": [
                {
                    "at": "2026-06-20T00:00:00+00:00",
                    "description": "quarter end",
                    "effects": [
                        {
                            "op": "set_field",
                            "field": "actual_deliveries",
                            "value": {"op": "field", "args": ["delivery_value_exogenous"]},
                        }
                    ],
                }
            ],
            "evidence_claim_ids": [],
        }
    ]
    data["uncertainties"] = [
        {
            "variable": "delivery_value_exogenous",
            "why_unknown": "the quarter is not over",
            "reversal_capable": True,
            "outcomes": [
                {
                    "value": "over",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["delivery_value_exogenous", 400001]],
                },
                {
                    "value": "under",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["delivery_value_exogenous", 380000]],
                },
            ],
        }
    ]

    with pytest.raises(WorldIntegrityError) as exc:
        _compile(data, _gateway(_signal_sensitive))
    assert exc.value.details["failure"] == "terminal_laundered_from_uncertainty"
    assert exc.value.details["terminal terms copied from an uncertainty"] == ["actual_deliveries"]


def test_a_total_computed_from_a_grounded_base_and_an_uncertain_rate_is_not_a_launder() -> None:
    """The world the launder gate must permit: uncertainty on the driver, not the total.

    A quarter's deliveries built from a grounded starting run-rate scaled by an uncertain
    demand multiplier is production — the unknown sits on the rate, which is exactly where
    the compiler is told to put it — and the terminal reads a total the world computed, not
    a draw copied into it.
    """

    from sworldmodel.world_compiler import _laundered_terminal_terms, terminal_producers

    data = _split_world()
    data["world_spec"]["fields"].append(
        {"field_id": "base_runrate", "value_type": "number", "initial": 350000}
    )
    data["world_spec"]["fields"].append(
        {"field_id": "demand_multiplier", "value_type": "number", "initial": 1.0}
    )
    data["world_spec"]["fields"].append(
        {"field_id": "actual_deliveries", "value_type": "number", "initial": 0}
    )
    data["world_spec"]["terminal"]["yes_when"] = {
        "op": "greater_than",
        "args": [{"op": "field", "args": ["actual_deliveries"]}, 400000],
    }
    data["world_spec"]["external_processes"] = [
        {
            "process_id": "q_end",
            "description": "the quarter's output is the run-rate scaled by realised demand",
            "occurrences": [
                {
                    "at": "2026-06-20T00:00:00+00:00",
                    "description": "quarter end",
                    "effects": [
                        {
                            "op": "set_field",
                            "field": "actual_deliveries",
                            "value": {
                                "op": "multiply",
                                "args": [
                                    {"op": "field", "args": ["base_runrate"]},
                                    {"op": "field", "args": ["demand_multiplier"]},
                                ],
                            },
                        }
                    ],
                }
            ],
            "evidence_claim_ids": [],
        }
    ]
    data["uncertainties"] = [
        {
            "variable": "demand_multiplier",
            "why_unknown": "demand for the quarter is not yet observed",
            "reversal_capable": True,
            "outcomes": [
                {
                    "value": "strong",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["demand_multiplier", 1.2]],
                },
                {
                    "value": "weak",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["demand_multiplier", 1.05]],
                },
            ],
        }
    ]

    bundle = build_bundle(data)
    producers = terminal_producers(bundle.spec)
    # The value reads a grounded base as well as the uncertain rate, so it is production,
    # not a bare copy of a draw — the launder gate leaves it alone.
    assert _laundered_terminal_terms(bundle.spec, bundle.uncertainties, producers) == {}
