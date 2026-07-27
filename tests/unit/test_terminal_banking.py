"""W1/W2 — banking a monotone terminal, and truncation as an execution failure.

The defect these exist for (G1, ``artifacts/ab/individual_semantic``): the terminal was
``greater_than(event_count('bailey_signals_support_for_further_cut'), 0)``, the agent
chose the satisfying action on twenty separate invocations, the actor-call budget ran
out three scheduled events short of the horizon, and the branch reported
``resolved=False``. The run published **0.0** — the average of the two branches that
happened to finish, both of which said NO. The system watched the answer happen twenty
times and published its opposite.

Two things had to be true at once for that. The terminal was read ONCE, at the end, on
a trajectory that never got there; and a guard that is right in general — a branch cut
short did not watch its process finish — was applied to an expression that cannot come
undone. The fix is not "resolve earlier". It is monotonicity, computed from the shape
of the expression and never declared by whoever wrote the world, so the tests below
come in three parts: the structure is analysed correctly, a monotone terminal survives
truncation, and everything else keeps refusing exactly as it does today.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import single_response_world
from sworldmodel.engine import (
    EXECUTION_COMPLETE,
    EXECUTION_INCOMPLETE,
    UNRESOLVED_WORLD_UNDETERMINED,
    RunBudget,
    _bankable,
    run,
)
from sworldmodel.models import ResolutionContract
from sworldmodel.world_compiler import compile_world
from sworldmodel.worldspec import (
    MONOTONE_DOWN,
    MONOTONE_FIXED,
    MONOTONE_UNKNOWN,
    MONOTONE_UP,
    TerminalExpression,
    expression_monotonicity,
    monotone_sources,
    parse_expr,
)

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")

SIGNAL_EVENT = "signals_support_for_further_cut"


def _compile(data: dict[str, Any], gateway: ProgrammableGateway, *, max_branches: int = 4):
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="test question",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        required_reality_facts=bundle.required_reality_facts,
        expected_participants=bundle.expected_participants,
    )
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=max_branches,
    )


def _gateway(decision) -> ProgrammableGateway:
    return ProgrammableGateway(
        {"actor_decision": decision, "reflect": {"beliefs_update": [], "new_memories": []}}
    )


def _drumbeat() -> list[dict[str, Any]]:
    """A long tail of scheduled work inside the window that the branch will not reach.

    This is what makes a branch *cut short* rather than finished: entries still due
    before the horizon when we stopped.
    """

    return [
        {
            "process_id": "drumbeat",
            "description": "many scheduled events before the horizon",
            "occurrences": [
                {
                    "at": f"2026-06-{day:02d}T09:00:00+00:00",
                    "description": f"tick {day}",
                    "effects": [
                        {"op": "deliver_information", "to": ["recipient"], "text": f"tick {day}"}
                    ],
                }
                for day in range(1, 21)
            ],
        }
    ]


def _signalling_world(terminal: dict[str, Any]) -> dict[str, Any]:
    """G1's shape: one actor whose action emits a countable event, and a terminal read
    off the count of that event. The action is otherwise the fixture's own."""

    data = copy.deepcopy(single_response_world())
    action = data["world_spec"]["actions"][0]
    action["effects"].append(
        {
            "op": "create_event",
            "event_type": SIGNAL_EVENT,
            "text": "support signalled for a further cut",
        }
    )
    data["world_spec"]["terminal"] = terminal
    data["world_spec"]["external_processes"] = _drumbeat()
    return data


def _always_signal(ctx: dict[str, Any]) -> dict[str, Any]:
    return act("send_reply", {"answer": "yes"})


# ---------------------------------------------------------------------------
# 1. Monotonicity is read off the expression, in both directions
# ---------------------------------------------------------------------------


def test_an_event_count_above_zero_is_monotone_toward_yes() -> None:
    expr = parse_expr({"op": "greater_than", "args": [{"op": "event_count", "args": ["x"]}, 0]})
    assert expression_monotonicity(expr) == MONOTONE_UP
    assert monotone_sources(expr) == ("event:x",)


def test_an_event_count_below_a_bound_is_monotone_toward_no() -> None:
    """The direction test. ``event_count(X) < 3`` is TRUE early and becomes false as the
    world produces more of X — banking it as a YES would publish an answer the world was
    in the middle of contradicting."""

    expr = parse_expr({"op": "less_than", "args": [{"op": "event_count", "args": ["x"]}, 3]})
    assert expression_monotonicity(expr) == MONOTONE_DOWN


def test_negation_inverts_the_direction() -> None:
    exists = {"op": "exists", "args": ["c"]}
    assert expression_monotonicity(parse_expr(exists)) == MONOTONE_UP
    assert expression_monotonicity(parse_expr({"op": "not", "args": [exists]})) == MONOTONE_DOWN


def test_a_field_threshold_is_not_monotone_at_all() -> None:
    """The case the conservative guard exists for: more simulation genuinely could move
    the number, in either direction."""

    expr = parse_expr(
        {"op": "greater_than", "args": [{"op": "field", "args": ["deliveries"]}, 400000]}
    )
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN
    assert monotone_sources(expr) == ()


def test_a_record_local_where_predicate_stays_monotone() -> None:
    expr = parse_expr(
        {
            "op": "greater_or_equal",
            "args": [
                {
                    "op": "count",
                    "args": [
                        "positions",
                        {"op": "equals", "args": [{"op": "item", "args": ["value"]}, "hold"]},
                    ],
                },
                5,
            ],
        }
    )
    assert expression_monotonicity(expr) == MONOTONE_UP
    assert monotone_sources(expr) == ("collection:positions",)


def test_a_where_predicate_that_reads_a_mutable_field_is_not_monotone() -> None:
    """An append-only collection counted through a rewritable filter is not append-only:
    flipping the field disqualifies records that already matched."""

    expr = parse_expr(
        {
            "op": "greater_than",
            "args": [
                {
                    "op": "count",
                    "args": ["positions", {"op": "field", "args": ["window_open"]}],
                },
                0,
            ],
        }
    )
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN


def test_a_collection_named_by_a_field_is_not_monotone() -> None:
    """``count(field('which_list'))`` counts a DIFFERENT list once the field moves, so
    its growth is not the collection's growth."""

    expr = parse_expr(
        {
            "op": "greater_than",
            "args": [{"op": "count", "args": [{"op": "field", "args": ["which_list"]}]}, 0],
        }
    )
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN


def test_equality_against_a_count_is_not_monotone() -> None:
    """A count passing through 3 satisfies ``equals(count, 3)`` on the way and falsifies
    it immediately after."""

    expr = parse_expr({"op": "equals", "args": [{"op": "count", "args": ["c"]}, 3]})
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN


def test_a_sum_over_an_append_only_collection_is_not_monotone() -> None:
    """A negative summand makes an append-only total fall, and nothing structural can
    know the sign."""

    expr = parse_expr({"op": "greater_than", "args": [{"op": "sum", "args": ["c"]}, 0]})
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN


def test_mixing_directions_under_a_conjunction_destroys_monotonicity() -> None:
    expr = parse_expr(
        {
            "op": "and",
            "args": [
                {"op": "exists", "args": ["a"]},
                {"op": "not", "args": [{"op": "exists", "args": ["b"]}]},
            ],
        }
    )
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN


def test_two_growing_counts_compared_against_each_other_are_not_monotone() -> None:
    """``count(a) > count(b)`` is true whenever a is ahead, and b can catch up."""

    expr = parse_expr(
        {
            "op": "greater_than",
            "args": [{"op": "count", "args": ["a"]}, {"op": "count", "args": ["b"]}],
        }
    )
    assert expression_monotonicity(expr) == MONOTONE_UNKNOWN


def test_a_conjunction_of_growing_counts_is_monotone_and_names_both_sources() -> None:
    expr = parse_expr(
        {
            "op": "and",
            "args": [
                {"op": "exists", "args": ["a"]},
                {"op": "greater_than", "args": [{"op": "event_count", "args": ["e"]}, 0]},
            ],
        }
    )
    assert expression_monotonicity(expr) == MONOTONE_UP
    assert monotone_sources(expr) == ("collection:a", "event:e")


def test_a_rising_unresolved_when_blocks_banking_the_merger_approval() -> None:
    """The one hole the adversary found, by name: BOTH legs monotone, both UP.

    ``yes_when: event_count('board_approves_merger') > 0`` is monotone toward YES and
    every check on it passes. ``unresolved_when: count('formal_challenges') > 0`` is
    built from the same append-only ops and rises too::

        t0  nothing happened                        -> resolved NO
        t1  the board approves          <- BANK?    -> resolved YES
        t2  a shareholder files a challenge         -> UNRESOLVED

    Banking at t1 would override the compiled world's own unresolvedness condition and
    publish YES for a world the terminal says is not determinable. Nothing about the
    ``yes_when`` analysis can catch this, and ``prompts.py`` actively instructs the
    compiler to use ``unresolved_when`` for exactly this kind of dispute counter — so
    banking requires ``unresolved_when`` to be monotone the OTHER way: once the world
    is determinate, it stays determinate.
    """

    yes_when = parse_expr(
        {
            "op": "greater_than",
            "args": [{"op": "event_count", "args": ["board_approves_merger"]}, 0],
        }
    )
    rising = parse_expr(
        {"op": "greater_than", "args": [{"op": "count", "args": ["formal_challenges"]}, 0]}
    )
    # Both legs pass every test that looks only at yes_when.
    assert expression_monotonicity(yes_when) == MONOTONE_UP
    assert expression_monotonicity(rising) == MONOTONE_UP

    attack = TerminalExpression(yes_when=yes_when, unresolved_when=rising, description="merger")
    assert _bankable(attack) is None, "a terminal that can become undeterminable was bankable"

    # The control: the same yes_when under an unresolvedness condition that can only
    # close is bankable, so the refusal above is about direction and not about
    # refusing everything with an unresolved_when.
    closing = parse_expr(
        {"op": "less_than", "args": [{"op": "count", "args": ["formal_challenges"]}, 1]}
    )
    assert expression_monotonicity(closing) == MONOTONE_DOWN
    assert _bankable(TerminalExpression(yes_when=yes_when, unresolved_when=closing)) == (
        "event:board_approves_merger",
    )
    assert _bankable(TerminalExpression(yes_when=yes_when)) == ("event:board_approves_merger",)


def test_a_constant_is_fixed_and_has_nothing_to_point_at() -> None:
    """Monotone-because-constant is not bankable: there is no cause to name."""

    expr = parse_expr({"op": "greater_than", "args": [1, 0]})
    assert expression_monotonicity(expr) == MONOTONE_FIXED
    assert monotone_sources(expr) == ()


# ---------------------------------------------------------------------------
# 2. W1 — the G1 regression
# ---------------------------------------------------------------------------


def test_a_monotone_terminal_satisfied_early_survives_truncation() -> None:
    """G1. The agent produced the satisfying event; the run was cut short afterwards.

    Before W1 this branch reported ``resolved=False`` with "trajectory cut short", and a
    run whose other branches finished NO published the opposite of what it had watched
    happen. Truncation cannot un-happen an event that already happened.
    """

    data = _signalling_world(
        {
            "yes_when": {
                "op": "greater_than",
                "args": [{"op": "event_count", "args": [SIGNAL_EVENT]}, 0],
            },
            "unresolved_when": {"op": "const", "args": [False]},
            "description": f"YES when {SIGNAL_EVENT} has happened at least once",
        }
    )
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    diag = next(iter(result.diagnostics.values()))
    assert diag.unfired_in_horizon > 0, "the branch was not actually cut short"

    outcome = result.branch_outcomes[0]
    assert outcome.resolved and outcome.outcome == "YES", (
        f"a monotone terminal the world already satisfied was discarded: "
        f"{outcome.unresolved_reason}"
    )


def test_the_bank_records_when_the_answer_became_true_and_what_made_it_true() -> None:
    """A bank that cannot name its own cause is an assertion. The forensic replay and
    the responsibility gate both read this."""

    data = _signalling_world(
        {
            "yes_when": {
                "op": "greater_than",
                "args": [{"op": "event_count", "args": [SIGNAL_EVENT]}, 0],
            },
            "unresolved_when": {"op": "const", "args": [False]},
            "description": "YES when support has been signalled",
        }
    )
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    branch_id, diag = next(iter(result.diagnostics.items()))
    banked = diag.banked_terminal
    assert banked is not None, "nothing was banked for a satisfied monotone terminal"
    assert banked.outcome == "YES"
    assert banked.sources == (f"event:{SIGNAL_EVENT}",)
    assert banked.caused_by_event_ids and banked.cause_count == len(banked.caused_by_event_ids)
    assert not diag.bank_contradicted

    # The banked instant is when the WORLD reached it, strictly before we stopped.
    assert datetime.fromisoformat(banked.at) <= compiled.base_world.contract.horizon

    # Every named cause is a real event in this branch's ledger, and every one of them
    # is the event the terminal counts.
    ledger = {e.event_id: e for e in result.final_worlds[branch_id].event_history}
    for eid in banked.caused_by_event_ids:
        assert eid in ledger, f"banked cause {eid} is not in the branch ledger"
        assert ledger[eid].payload_dict.get("event_type") == SIGNAL_EVENT

    # And it reaches the trace, not only the diagnostics.
    recorded = [
        e
        for e in result.final_worlds[branch_id].event_history
        if e.payload_dict.get("event_type") == "result_recorded"
    ]
    assert recorded, "no result was recorded"
    banked_in_trace = dict(recorded[-1].payload_dict["data"])["banked_terminal"]
    assert banked_in_trace["at"] == banked.at
    assert banked_in_trace["caused_by_event_ids"] == list(banked.caused_by_event_ids)


# ---------------------------------------------------------------------------
# 3. The conservative behaviour that must survive, unchanged
# ---------------------------------------------------------------------------


def test_a_non_monotone_terminal_still_refuses_to_resolve_when_cut_short() -> None:
    """The guard's reasoning is right in general and must still be true OF THE CODE.

    ``field('reply_sent') == true`` is satisfied here at the moment we stop — and a
    field is rewritable, so an evaluation taken where we stopped is a claim about a
    process we did not watch. Nothing about W1 may soften this.
    """

    data = _signalling_world(
        {
            "yes_when": {"op": "equals", "args": [{"op": "field", "args": ["reply_sent"]}, True]},
            "unresolved_when": {"op": "const", "args": [False]},
            "description": "YES when the reply has been sent",
        }
    )
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    branch_id, diag = next(iter(result.diagnostics.items()))
    assert diag.unfired_in_horizon > 0
    # The field really was set — this is a refusal, not an accident of the world.
    assert result.final_worlds[branch_id].get_field("reply_sent") is True
    assert diag.banked_terminal is None

    outcome = result.branch_outcomes[0]
    assert not outcome.resolved and outcome.outcome is None
    assert "cut short" in (outcome.unresolved_reason or "")


def test_a_terminal_monotone_toward_no_is_never_banked_as_yes() -> None:
    """Direction, at the engine and not only in the analyser.

    ``event_count(X) < 100`` reads the same append-only term as G1's terminal and is
    satisfied at the moment we stop — but it is monotone the OTHER way: every further
    event moves it toward NO, and the branch stopped before the world was finished
    producing them. Banking on "the terminal is true and it mentions ``event_count``"
    would publish an answer the world may have been in the middle of contradicting,
    which is worse than the defect being fixed.

    A yes_when monotone toward NO is true from the start by construction, so the world
    carries an ``unresolved_when`` that opens only once a reply exists — otherwise the
    compiler correctly refuses a world whose initial state already answers YES.
    """

    data = _signalling_world(
        {
            "yes_when": {
                "op": "less_than",
                "args": [{"op": "event_count", "args": [SIGNAL_EVENT]}, 100],
            },
            "unresolved_when": {
                "op": "less_than",
                "args": [{"op": "count", "args": ["replies"]}, 1],
            },
            "description": f"YES while {SIGNAL_EVENT} has happened fewer than 100 times",
        }
    )
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    branch_id, diag = next(iter(result.diagnostics.items()))
    assert diag.unfired_in_horizon > 0

    # The terminal really is satisfied where we stopped: the world determined itself
    # (a reply exists) and the count is under the bound. This is a refusal to bank a
    # true-right-now terminal, not an accident of the world never getting there.
    world = result.final_worlds[branch_id]
    fired = len(world.get_events(SIGNAL_EVENT))
    assert 0 < fired < 100
    assert len(world.get_records("replies")) >= 1

    assert diag.banked_terminal is None, "a monotone-toward-NO terminal was banked"
    outcome = result.branch_outcomes[0]
    assert not outcome.resolved and outcome.outcome is None
    assert "cut short" in (outcome.unresolved_reason or "")


def test_the_merger_approval_is_not_banked_while_a_challenge_is_still_scheduled() -> None:
    """The adversary's case, driven through the engine.

    The board approves early and the terminal reads YES. A shareholder challenge is
    scheduled inside the window and is one of the entries truncation prevented from
    firing — so at the instant we stop, the world's own unresolvedness condition is
    false and a bank would look perfectly safe. It is not: the thing that would have
    made the answer undeterminable is sitting unfired on the calendar, which is
    precisely what "cut short" means. The refusal is structural, taken before the run
    starts, and does not depend on noticing the challenge.
    """

    approval = "board_approves_merger"
    data = copy.deepcopy(single_response_world())
    data["world_spec"]["actions"][0]["effects"].append(
        {"op": "create_event", "event_type": approval, "text": "the board approved"}
    )
    data["world_spec"]["terminal"] = {
        "yes_when": {"op": "greater_than", "args": [{"op": "event_count", "args": [approval]}, 0]},
        "unresolved_when": {
            "op": "greater_than",
            "args": [{"op": "count", "args": ["formal_challenges"]}, 0],
        },
        "description": "YES when the board approves and nobody has formally challenged",
    }
    data["world_spec"]["external_processes"] = _drumbeat() + [
        {
            "process_id": "challenges",
            "description": "a shareholder files a formal challenge, later in the window",
            "occurrences": [
                {
                    "at": "2026-06-10T09:00:00+00:00",
                    "description": "a formal challenge is filed",
                    "effects": [
                        {
                            "op": "append_record",
                            "collection": "formal_challenges",
                            "key": "shareholder",
                            "value": "filed",
                        }
                    ],
                }
            ],
        }
    ]
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    branch_id, diag = next(iter(result.diagnostics.items()))
    world = result.final_worlds[branch_id]
    assert len(world.get_events(approval)) > 0, "the board never approved"
    assert world.get_records("formal_challenges") == [], "the challenge was supposed to be unfired"
    assert diag.unfired_in_horizon > 0

    assert diag.banked_terminal is None, "banked over a terminal that can become undeterminable"
    outcome = result.branch_outcomes[0]
    assert not outcome.resolved and outcome.outcome is None
    assert "cut short" in (outcome.unresolved_reason or "")


def test_a_monotone_terminal_the_world_never_satisfied_is_not_banked() -> None:
    """Banking is not "resolve earlier". A terminal that never became true stays exactly
    as unresolved as it is today."""

    data = _signalling_world(
        {
            "yes_when": {
                "op": "greater_than",
                "args": [{"op": "event_count", "args": [SIGNAL_EVENT]}, 0],
            },
            "unresolved_when": {"op": "const", "args": [False]},
            "description": "YES when support has been signalled",
        }
    )
    gw = _gateway(lambda ctx: wait_decision("thinking"))
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    diag = next(iter(result.diagnostics.values()))
    assert diag.unfired_in_horizon > 0
    assert diag.banked_terminal is None
    outcome = result.branch_outcomes[0]
    assert not outcome.resolved
    assert "cut short" in (outcome.unresolved_reason or "")


# ---------------------------------------------------------------------------
# 4. W2 — truncation is an execution failure, not uncertainty
# ---------------------------------------------------------------------------


def test_a_budget_stop_is_classified_as_our_failure_not_the_world_s_uncertainty() -> None:
    """Budget exhaustion produced unresolved mass that flowed into bounds and
    suppression as though REALITY were uncertain. It is not: our simulator ran out of
    calls, and that is a reliability statement about us."""

    data = _signalling_world(
        {
            "yes_when": {"op": "equals", "args": [{"op": "field", "args": ["reply_sent"]}, True]},
            "unresolved_when": {"op": "const", "args": [False]},
            "description": "YES when the reply has been sent",
        }
    )
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    diag = next(iter(result.diagnostics.values()))
    assert diag.execution_status == EXECUTION_INCOMPLETE
    assert diag.execution_limit
    assert diag.unresolved_class == EXECUTION_INCOMPLETE
    assert diag.unresolved_class != UNRESOLVED_WORLD_UNDETERMINED

    # The run-level seam: this is the share of the world we did not manage to run.
    assert result.execution_incomplete_mass > 0.0
    assert result.execution_incomplete_branches
    assert all(reason for _, reason in result.execution_incomplete_branches)

    # And it reads as a statement about the run, not about reality.
    reason = result.branch_outcomes[0].unresolved_reason or ""
    assert "the simulator stopped before the world did" in reason


def test_a_branch_that_finished_carries_no_execution_incomplete_mass() -> None:
    """The other half: a run we DID complete must not be tarred with our own failure
    class, or the signal is worthless."""

    data = copy.deepcopy(single_response_world())
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)

    assert result.execution_incomplete_mass == 0.0
    assert result.execution_incomplete_branches == ()
    for diag in result.diagnostics.values():
        assert diag.execution_status == EXECUTION_COMPLETE
        assert diag.stop_reason == "schedule exhausted"
        assert diag.unfired_in_horizon == 0


def test_a_banked_branch_is_resolved_even_though_execution_was_incomplete() -> None:
    """The two facts are independent and must both be readable. We failed to run this
    branch to the end AND the answer is known, because what makes it known already
    happened."""

    data = _signalling_world(
        {
            "yes_when": {
                "op": "greater_than",
                "args": [{"op": "event_count", "args": [SIGNAL_EVENT]}, 0],
            },
            "unresolved_when": {"op": "const", "args": [False]},
            "description": "YES when support has been signalled",
        }
    )
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_batches=6))

    diag = next(iter(result.diagnostics.values()))
    assert diag.execution_status == EXECUTION_INCOMPLETE
    assert diag.unresolved_class == "", "a resolved branch has no unresolved mass to class"
    assert result.execution_incomplete_mass > 0.0
    assert result.branch_outcomes[0].resolved


# ---------------------------------------------------------------------------
# 5. The call budget scales with the world
# ---------------------------------------------------------------------------


def test_the_call_budget_scales_with_participants_and_horizon() -> None:
    """A nine-participant world over ten weeks got the same 80 calls as a
    two-participant world over two weeks, so the budget — not the world — decided who
    got to act."""

    small = RunBudget.for_world(participants=2, horizon_days=14)
    large = RunBudget.for_world(participants=9, horizon_days=70)
    assert large.max_actor_calls > small.max_actor_calls

    # More participants over the same window, and the same roster over a longer window,
    # each buy more room.
    assert (
        RunBudget.for_world(participants=9, horizon_days=70).max_actor_calls
        > RunBudget.for_world(participants=4, horizon_days=70).max_actor_calls
    )
    assert (
        RunBudget.for_world(participants=9, horizon_days=70).max_actor_calls
        > RunBudget.for_world(participants=9, horizon_days=14).max_actor_calls
    )


def test_no_world_is_given_less_than_the_old_flat_budget() -> None:
    """The old constant is the FLOOR. Scaling must never take room away from the small
    worlds that were running fine on it."""

    assert RunBudget.for_world(participants=1, horizon_days=1).max_actor_calls == 80
    assert RunBudget.for_world(participants=0, horizon_days=0).max_actor_calls == 80
    for participants in (1, 2, 5, 9, 40):
        for days in (1, 14, 70, 400):
            budget = RunBudget.for_world(participants=participants, horizon_days=days)
            assert budget.max_actor_calls >= 80
            assert budget.max_actor_calls <= 400
            # Raising the call ceiling alone would only move the stop to the next
            # budget and change nothing a reader can see.
            assert budget.max_events >= budget.max_actor_calls
            assert budget.max_batches >= budget.max_actor_calls


def test_an_explicitly_pinned_budget_is_honoured_exactly() -> None:
    data = copy.deepcopy(single_response_world())
    gw = _gateway(_always_signal)
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0, budget=RunBudget(max_actor_calls=1))

    for diag in result.diagnostics.values():
        assert sum(diag.actor_call_counts.values()) <= 1
        assert diag.execution_status == EXECUTION_INCOMPLETE
