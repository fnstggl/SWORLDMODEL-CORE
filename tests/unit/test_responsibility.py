"""The responsibility gate (D6 / FI-3 / FI-4): what produced the number, computed in-run.

Every test here is a deterministic replay of a recorded ledger — no simulation, no
model. The live failures these pin down:

* **FD-4.** The classification existed only in ``scripts/forensics.py``, so publication
  was ungated: a Bank of England run classified BRANCH_WEIGHTS_DOMINATED *after* it had
  already published 0.25 as a forecast. The gate now runs before the artifact is
  written, and four of its eight values forbid an answer.
* **The Tesla shape.** Two ungrounded numeric alternatives, 0.8 and 1.05, straddling a
  break-even of 0.8331 — the published 0.50 was the count of invented branches. The
  perturbation test moves the draw inside the range the run itself enumerated and
  reports that the terminal answer changes.
"""

from __future__ import annotations

from typing import Any

import pytest

from sworldmodel.models import (
    ACTOR_AND_PROCESS_CAUSED,
    ACTOR_CAUSED,
    BRANCH_WEIGHTS_DOMINATED,
    FACTUALLY_RESOLVED,
    INITIAL_ASSUMPTIONS_DOMINATED,
    PROCESS_CAUSED,
    PUBLISHING_CLASSIFICATIONS,
    REQUIRED_RESPONSIBILITY_TESTS,
    RESPONSIBILITY_CLASSIFICATIONS,
    RESPONSIBILITY_INVALID,
    RESPONSIBILITY_UNRESOLVED,
    BranchOutcome,
)
from sworldmodel.responsibility import (
    classify_responsibility,
    equal_weight_probability,
    numeric_condition_ranges,
    threshold_straddling_variables,
    weighted_probability,
)

# --------------------------------------------------------------------------- #
# Fixtures: an executable world and a recorded ledger, in artifact shape
# --------------------------------------------------------------------------- #


def _world(
    yes_when: dict[str, Any],
    fields: dict[str, Any],
    unresolved_when: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The persisted ``compiled_world.json`` shape the replay core reads."""

    return {
        "world_spec": {
            "fields": [{"field_id": k, "initial": v} for k, v in fields.items()],
            "terminal": {"yes_when": yes_when, "unresolved_when": unresolved_when},
        }
    }


def _unset(field: str) -> dict[str, Any]:
    """The engine's own shape for "nothing determined this": the field is still unset."""

    return {"op": "equals", "args": [{"op": "field", "args": [{"const": field}]}, {"const": None}]}


def _count_terminal(event_type: str) -> dict[str, Any]:
    """YES when at least one event of ``event_type`` was created."""

    return {
        "op": "greater_than",
        "args": [{"op": "event_count", "args": [{"const": event_type}]}, {"const": 0}],
    }


def _threshold_terminal(field: str, limit: float) -> dict[str, Any]:
    return {
        "op": "greater_than",
        "args": [{"op": "field", "args": [{"const": field}]}, {"const": limit}],
    }


def _event(
    branch: str,
    kind: str,
    payload: dict[str, Any],
    *,
    actor_id: str | None = None,
    seq: int = 0,
) -> dict[str, Any]:
    return {
        "event_id": f"{branch}-{kind}-{seq}",
        "branch_id": branch,
        "kind": kind,
        "actor_id": actor_id,
        "payload": payload,
        "time": "2026-06-01T00:00:00+00:00",
    }


def _branch(
    branch_id: str,
    weight: float,
    outcome: str | None,
    *,
    weight_grounded: bool,
    conditions: tuple[tuple[str, str], ...] = (),
    event_count: int = 2,
) -> BranchOutcome:
    return BranchOutcome(
        branch_id=branch_id,
        parent_lineage=("root",),
        weight=weight,
        resolved=outcome is not None,
        outcome=outcome,
        unresolved_reason=None if outcome is not None else "did not resolve",
        truncated=False,
        key_conditions=conditions,
        event_count=event_count,
        weight_grounded=weight_grounded,
    )


# --------------------------------------------------------------------------- #
# The vocabulary is exactly D6's, and only four of it may publish
# --------------------------------------------------------------------------- #


def test_the_vocabulary_is_exactly_the_eight_values_d6_names() -> None:
    assert RESPONSIBILITY_CLASSIFICATIONS == (
        "ACTOR_CAUSED",
        "PROCESS_CAUSED",
        "ACTOR_AND_PROCESS_CAUSED",
        "FACTUALLY_RESOLVED",
        "INITIAL_ASSUMPTIONS_DOMINATED",
        "BRANCH_WEIGHTS_DOMINATED",
        "UNRESOLVED",
        "INVALID",
    )
    assert {
        "ACTOR_CAUSED",
        "PROCESS_CAUSED",
        "ACTOR_AND_PROCESS_CAUSED",
        "FACTUALLY_RESOLVED",
    } == PUBLISHING_CLASSIFICATIONS
    # The other four exist precisely so a run can be refused an answer.
    assert set(RESPONSIBILITY_CLASSIFICATIONS) - PUBLISHING_CLASSIFICATIONS == {
        "INITIAL_ASSUMPTIONS_DOMINATED",
        "BRANCH_WEIGHTS_DOMINATED",
        "UNRESOLVED",
        "INVALID",
    }


# --------------------------------------------------------------------------- #
# FI-4: the mandatory test set
# --------------------------------------------------------------------------- #


def test_every_mandatory_counterfactual_runs_before_publication() -> None:
    """§14 names seven tests. A gate that ran six of them has not run."""

    world = _world(_count_terminal("signal"), {"stance": "unknown"})
    events = [
        _event("b1", "set_field", {"field": "stance", "value": "engaged"}, actor_id="a1"),
        _event("b1", "create_event", {"event_type": "signal"}, actor_id="a1", seq=1),
        _event("b1", "set_field", {"field": "backdrop", "value": 1}, seq=2),
    ]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[{"actor_id": "a1"}],
        world=world,
    )
    assert report.tests_completed == REQUIRED_RESPONSIBILITY_TESTS
    assert report.tests_missing == ()
    assert report.gate_complete
    cf = report.branch_counterfactuals[0]
    # Every mandatory deletion produced an answer, and each is recorded separately.
    assert cf.recomputed_outcome == "YES"
    assert cf.all_actor_output_removed == "NO"  # the actor made the only signal
    assert cf.all_process_output_removed == "YES"  # the process wrote nothing decisive
    assert cf.initialization_preserved == "NO"  # nothing left, nothing counted
    assert cf.terminal_relevant_actions_removed == "NO"
    assert dict(cf.per_actor_removed) == {"a1": "NO"}
    assert dict(cf.per_process_removed) == {"set_field": "YES"}
    # And the weight-sensitivity substitution ran.
    assert report.equal_weight_probability == pytest.approx(1.0)
    assert report.weight_sensitivity_span == pytest.approx(0.0)
    d = report.as_dict()
    assert d["tests_completed"] == list(REQUIRED_RESPONSIBILITY_TESTS)
    assert "no model call" in d["method"]


# --------------------------------------------------------------------------- #
# The eight classifications, each computed rather than asserted
# --------------------------------------------------------------------------- #


def test_actor_caused_when_deleting_actor_output_changes_the_answer() -> None:
    world = _world(_count_terminal("signal"), {})
    events = [_event("b1", "create_event", {"event_type": "signal"}, actor_id="bailey")]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[{"actor_id": "bailey"}],
        world=world,
    )
    assert report.classification == ACTOR_CAUSED
    assert report.may_publish_answer
    assert report.point_estimate_permitted  # grounded weights, so the number may stand
    assert "actors produced the answer" in report.reason


def test_process_caused_when_only_the_non_actor_process_moves_the_terminal() -> None:
    world = _world(_threshold_terminal("output", 100.0), {"output": 0})
    events = [_event("b1", "set_field", {"field": "output", "value": 150})]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[],
        world=world,
    )
    assert report.classification == PROCESS_CAUSED
    assert report.may_publish_answer


def test_actor_and_process_caused_when_both_deletions_move_an_outcome() -> None:
    world = _world(
        {
            "op": "and",
            "args": [
                _count_terminal("signal"),
                _threshold_terminal("output", 100.0),
            ],
        },
        {"output": 0},
    )
    events = [
        _event("b1", "create_event", {"event_type": "signal"}, actor_id="a1"),
        _event("b1", "set_field", {"field": "output", "value": 150}, seq=1),
    ]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[{"actor_id": "a1"}],
        world=world,
    )
    assert report.classification == ACTOR_AND_PROCESS_CAUSED
    assert report.may_publish_answer


def test_factually_resolved_when_the_terminal_is_yes_with_every_event_deleted() -> None:
    """The OPEC+ shape: the record answered the question before the window opened."""

    world = _world(_threshold_terminal("quota", 0.5), {"quota": 1})
    events = [_event("b1", "set_field", {"field": "unrelated", "value": 7})]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[],
        world=world,
    )
    assert report.classification == FACTUALLY_RESOLVED
    assert report.may_publish_answer
    # A factual resolution needs no grounded weights: no weight enters the answer.
    ungrounded = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=False),),
        events=events,
        actor_decisions=[],
        world=world,
    )
    assert ungrounded.classification == FACTUALLY_RESOLVED
    assert ungrounded.point_estimate_permitted


def test_initial_assumptions_dominated_when_no_deletion_changes_anything() -> None:
    """A NO that was NO at t0 and is still NO, with events that touched nothing."""

    world = _world(_count_terminal("signal"), {})
    events = [_event("b1", "set_field", {"field": "noise", "value": 1}, actor_id="a1")]
    report = classify_responsibility(
        (_branch("b1", 1.0, "NO", weight_grounded=True),),
        events=events,
        actor_decisions=[{"actor_id": "a1"}],
        world=world,
    )
    assert report.classification == INITIAL_ASSUMPTIONS_DOMINATED
    assert not report.may_publish_answer
    assert "initialization read back" in report.reason


def test_branch_weights_dominated_outranks_actor_caused_the_boe_shape() -> None:
    """The live Bank of England run. The actor genuinely produced the only YES — and
    the NUMBER is still one of four equally weighted symmetric-ignorance cells."""

    world = _world(_count_terminal("signal"), {})
    events = [
        _event("sc_slowing_other", "create_event", {"event_type": "signal"}, actor_id="bailey")
    ]
    branches = (
        _branch(
            "sc_slowing_other",
            0.25,
            "YES",
            weight_grounded=False,
            conditions=(("job_market", "slowing"), ("inflation", "other")),
        ),
        _branch(
            "sc_slowing_easing",
            0.25,
            "NO",
            weight_grounded=False,
            conditions=(("job_market", "slowing"), ("inflation", "easing")),
        ),
        _branch(
            "sc_stable_other",
            0.25,
            "NO",
            weight_grounded=False,
            conditions=(("job_market", "stable"), ("inflation", "other")),
        ),
        _branch(
            "sc_stable_easing",
            0.25,
            "NO",
            weight_grounded=False,
            conditions=(("job_market", "stable"), ("inflation", "easing")),
        ),
    )
    report = classify_responsibility(
        branches, events=events, actor_decisions=[{"actor_id": "bailey"}], world=world
    )
    # The actor deletion DOES change an outcome — that fact is recorded ...
    changed = [c for c in report.branch_counterfactuals if c.all_actor_output_removed != c.outcome]
    assert [c.branch_id for c in changed] == ["sc_slowing_other"]
    # ... and the classification is still BRANCH_WEIGHTS_DOMINATED, because the
    # magnitude is the weight of the winning cell and nothing grounds that weight.
    assert report.classification == BRANCH_WEIGHTS_DOMINATED
    assert not report.may_publish_answer
    assert not report.point_estimate_permitted
    assert report.weighted_probability == pytest.approx(0.25)
    assert report.equal_weight_probability == pytest.approx(0.25)


def test_unresolved_when_no_branch_reached_an_answer() -> None:
    world = _world(_threshold_terminal("x", 1.0), {"x": None}, unresolved_when=_unset("x"))
    report = classify_responsibility(
        (_branch("b1", 1.0, None, weight_grounded=True),),
        events=[],
        actor_decisions=[],
        world=world,
    )
    assert report.classification == RESPONSIBILITY_UNRESOLVED
    assert not report.may_publish_answer


def test_invalid_when_the_ledger_replay_disagrees_with_the_published_outcome() -> None:
    """A published YES the recorded ledger does not reproduce is not a forecast."""

    world = _world(_count_terminal("signal"), {})
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=[_event("b1", "set_field", {"field": "noise", "value": 1})],
        actor_decisions=[],
        world=world,
    )
    assert report.classification == RESPONSIBILITY_INVALID
    assert not report.may_publish_answer
    assert not report.trace_reproducible
    assert "ledger replay disagrees" in report.trace_reproducible_basis


# --------------------------------------------------------------------------- #
# The gate fails closed
# --------------------------------------------------------------------------- #


def test_a_gate_that_cannot_run_publishes_nothing() -> None:
    """Without an executable terminal there is no counterfactual to compute. "The
    check did not run" must not read as "the check passed"."""

    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=[],
        actor_decisions=[],
        world=None,
    )
    assert report.classification == RESPONSIBILITY_INVALID
    assert not report.may_publish_answer
    assert not report.trace_reproducible
    assert report.tests_missing == REQUIRED_RESPONSIBILITY_TESTS
    assert not report.gate_complete
    assert "no executable terminal" in report.reason


def test_the_gate_never_raises_and_refuses_when_it_breaks() -> None:
    class Exploding:
        @property
        def spec(self) -> Any:
            raise RuntimeError("boom")

    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=[],
        actor_decisions=[],
        world=Exploding(),
    )
    assert report.classification == RESPONSIBILITY_INVALID
    assert not report.may_publish_answer
    assert "boom" in report.error


# --------------------------------------------------------------------------- #
# FI-4: weight substitution and numeric perturbation
# --------------------------------------------------------------------------- #


def test_equal_weight_substitution_measures_what_the_weights_are_worth() -> None:
    branches = (
        _branch("a", 0.9, "YES", weight_grounded=True, conditions=(("v", "a"),)),
        _branch("b", 0.1, "NO", weight_grounded=True, conditions=(("v", "b"),)),
    )
    assert weighted_probability(branches) == pytest.approx(0.9)
    assert equal_weight_probability(branches) == pytest.approx(0.5)
    world = _world(_count_terminal("signal"), {})
    events = [
        _event("a", "create_event", {"event_type": "signal"}, actor_id="a1"),
        _event("b", "set_field", {"field": "noise", "value": 1}, actor_id="a1", seq=1),
    ]
    report = classify_responsibility(
        branches, events=events, actor_decisions=[{"actor_id": "a1"}], world=world
    )
    assert report.weight_sensitivity_span == pytest.approx(0.4)


def test_the_perturbation_band_comes_from_the_runs_own_enumerated_draws() -> None:
    branches = (
        _branch("lo", 0.5, "NO", weight_grounded=False, conditions=(("factor", "0.8"),)),
        _branch("hi", 0.5, "YES", weight_grounded=False, conditions=(("factor", "1.05"),)),
    )
    assert numeric_condition_ranges(branches) == {"factor": (0.8, 1.05)}
    # A grounded variable is not perturbed: its value is not an invented draw.
    grounded = (
        _branch("lo", 0.5, "NO", weight_grounded=True, conditions=(("factor", "0.8"),)),
        _branch("hi", 0.5, "YES", weight_grounded=True, conditions=(("factor", "1.05"),)),
    )
    assert numeric_condition_ranges(grounded) == {}
    # A single draw spans no range at all, so there is nothing plausible to move within.
    single = (_branch("lo", 1.0, "NO", weight_grounded=False, conditions=(("factor", "0.8"),)),)
    assert numeric_condition_ranges(single) == {}


def test_the_tesla_shape_is_caught_by_straddling_and_by_perturbation() -> None:
    """480,126 x 0.8 = 384,100.8 (NO) and x 1.05 = 504,132.3 (YES), break-even 0.8331.

    Two invented factors on opposite sides of the threshold: the answer is the choice
    of those two numbers and the fact that there are two of them.
    """

    world = _world(_threshold_terminal("q3_deliveries", 400_000.0), {"q3_deliveries": None})
    events = [
        _event("lo", "set_field", {"field": "q3_deliveries", "value": 384_100.8}),
        _event("hi", "set_field", {"field": "q3_deliveries", "value": 504_132.3}),
    ]
    branches = (
        _branch("lo", 0.5, "NO", weight_grounded=False, conditions=(("factor", "0.8"),)),
        _branch("hi", 0.5, "YES", weight_grounded=False, conditions=(("factor", "1.05"),)),
    )
    assert threshold_straddling_variables(branches) == ("factor",)

    report = classify_responsibility(branches, events=events, actor_decisions=[], world=world)
    assert report.threshold_straddling_variables == ("factor",)
    assert report.classification == BRANCH_WEIGHTS_DOMINATED
    assert not report.may_publish_answer
    # The perturbation moved the draw inside the band the run itself enumerated, and
    # the terminal answer changed — reported per branch, naming the field.
    assert report.numeric_perturbation_findings
    assert any(
        "q3_deliveries" in finding and "NO -> YES" in finding
        for finding in report.numeric_perturbation_findings
    )


def test_a_variable_whose_numeric_draws_agree_does_not_straddle() -> None:
    """Same numbers, same answer on both sides: nothing to attribute to the draw."""

    branches = (
        _branch("lo", 0.5, "YES", weight_grounded=False, conditions=(("factor", "0.8"),)),
        _branch("hi", 0.5, "YES", weight_grounded=False, conditions=(("factor", "1.05"),)),
    )
    assert threshold_straddling_variables(branches) == ()


def test_straddling_compares_only_branches_that_differ_in_that_one_variable() -> None:
    """Two variables moving at once is not evidence about either of them alone."""

    branches = (
        _branch(
            "a",
            0.25,
            "NO",
            weight_grounded=False,
            conditions=(("factor", "0.8"), ("regime", "tight")),
        ),
        _branch(
            "b",
            0.25,
            "YES",
            weight_grounded=False,
            conditions=(("factor", "1.05"), ("regime", "loose")),
        ),
    )
    # Nothing holds the rest of the tuple fixed, so no variable is charged with it.
    assert threshold_straddling_variables(branches) == ()

    with_control = branches + (
        _branch(
            "c",
            0.5,
            "YES",
            weight_grounded=False,
            conditions=(("factor", "1.05"), ("regime", "tight")),
        ),
    )
    # Now 'tight' is held fixed while factor moves 0.8 -> 1.05 and the answer flips.
    assert threshold_straddling_variables(with_control) == ("factor",)


def test_a_categorical_alternative_is_never_perturbed_as_a_number() -> None:
    """True/False and named states are not draws inside a plausible range."""

    branches = (
        _branch("t", 0.5, "YES", weight_grounded=False, conditions=(("open", "True"),)),
        _branch("f", 0.5, "NO", weight_grounded=False, conditions=(("open", "False"),)),
    )
    assert numeric_condition_ranges(branches) == {}
    assert threshold_straddling_variables(branches) == ()


# --------------------------------------------------------------------------- #
# The replay core's blind spot is named, never guessed around
# --------------------------------------------------------------------------- #


def test_a_cardinality_terminal_replays_exactly() -> None:
    """`count('positions') >= 2` reads how many records exist, which the ledger has."""

    world = _world(
        {
            "op": "greater_or_equal",
            "args": [{"op": "count", "args": [{"const": "positions"}]}, {"const": 2}],
        },
        {},
    )
    events = [
        _event("b1", "append_record", {"collection": "positions", "value": "hold"}, actor_id="m0"),
        _event(
            "b1",
            "append_record",
            {"collection": "positions", "value": "hold"},
            actor_id="m1",
            seq=1,
        ),
    ]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[{"actor_id": "m0"}, {"actor_id": "m1"}],
        world=world,
    )
    assert report.trace_reproducible
    assert report.classification == ACTOR_CAUSED
    assert report.branch_counterfactuals[0].all_actor_output_removed == "NO"


def test_a_content_predicated_terminal_replays_its_content_and_the_gate_runs() -> None:
    """FD-42, and this test is the history of a blind spot rather than a plain assertion.

    ``ReplayWorld`` used to reconstruct collection CARDINALITY and present each record as
    an empty placeholder, so `count('positions', equals(item('value'), 'hold'))` replayed
    over blanks and answered zero — confidently and wrongly. Every actor world in this
    repo uses that shape. The gate could not tell a genuine reproduction failure from an
    artifact of the replay, so it failed closed and refused to publish any of them.

    Failing closed was right while the blind spot existed, and it is wrong now that it
    does not: refusing a world the replay CAN reconstruct withholds an answer the system
    has honestly earned. The replay reconstructs record content, so the terminal replays
    to the live answer and the mandatory counterfactuals actually run.

    The old assertions are kept in shape below — same world, same events, same branch —
    so that if the reconstruction ever regresses to blanks, this test fails rather than a
    refusal quietly returning and reading like caution.
    """

    world = _world(
        {
            "op": "greater_or_equal",
            "args": [
                {
                    "op": "count",
                    "args": [
                        {"const": "positions"},
                        {
                            "op": "equals",
                            "args": [
                                {"op": "item", "args": [{"const": "value"}]},
                                {"const": "hold"},
                            ],
                        },
                    ],
                },
                {"const": 2},
            ],
        },
        {},
    )
    events = [
        _event("b1", "append_record", {"collection": "positions", "value": "hold"}, actor_id="m0"),
        _event(
            "b1",
            "append_record",
            {"collection": "positions", "value": "hold"},
            actor_id="m1",
            seq=1,
        ),
    ]
    report = classify_responsibility(
        (_branch("b1", 1.0, "YES", weight_grounded=True),),
        events=events,
        actor_decisions=[{"actor_id": "m0"}, {"actor_id": "m1"}],
        world=world,
    )
    # Two records both reading "hold" satisfy `>= 2`, so the live answer is YES. Over
    # blank placeholders the same terminal counted zero holds and answered NO.
    assert report.trace_reproducible, "the replay must reach the answer the run reached"
    assert report.classification == ACTOR_CAUSED
    assert not report.error
    # The gate now runs rather than refusing: every mandatory counterfactual completed.
    assert report.tests_missing == ()
    assert report.gate_complete
    assert report.may_publish_answer
    # Deleting both actors' output removes both records, so the count cannot be met.
    assert report.branch_counterfactuals[0].all_actor_output_removed == "NO"
