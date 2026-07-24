"""An undetermined terminal term leaves a branch unresolved — it never crashes the run
and never fabricates a NO.

Defect found by a post-merge live run:

    question : "Will the Federal Reserve announce a reduction in the federal funds
               target range at its September 2026 FOMC meeting?"
    expected : the compiled terminal compared an announcement date the branch never set;
               a comparison against a value the world never determined must leave that
               branch UNRESOLVED (the forecast then reports bounds), because neither YES
               nor NO is honestly supported
    actual   : ``ValueError: not a time value: None`` propagated out of the terminal
               evaluator and aborted the whole forecast
    cause    : every coercion in ``expressions`` is total except ``_time``, which raised
               a bare ValueError on a missing value
    fix      : ``_time`` raises the typed ``UndeterminedExpressionError``; the engine
               turns it into an unresolved branch and the executor treats an undetermined
               precondition as "action not currently feasible"
"""

from __future__ import annotations

from typing import Any

from _worlds import _corpus, _person, run_corpus
from sworldmodel.models import ForecastStatus


def _date_terminal_world(*, set_the_date: bool) -> dict[str, Any]:
    """A world whose terminal asks whether an announcement happened before the horizon.

    When ``set_the_date`` is false nothing ever writes ``announced_at``, so the terminal
    reads a value the world never determined."""

    effects: list[dict[str, Any]] = [
        {"op": "append_record", "collection": "log", "key": "$actor", "value": "acted"}
    ]
    if set_the_date:
        effects.append(
            {"op": "set_field", "field": "announced_at", "value": "2024-02-01T00:00:00+00:00"}
        )
    spec = {
        "title": "announcement world",
        "subject_entity": "the announcement",
        "entities": [_person("a", ["announce"])],
        "actors": [
            {
                "entity_id": "a",
                "reasoning": "a announced before",
                "memory_seeds": [
                    {
                        "content": "I, a, announced a decision in the previous cycle.",
                        "kind": "episodic",
                        "importance": 0.7,
                        "evidence_claim_ids": ["r_a"],
                    }
                ],
                "policy": {"default_action_id": "announce"},
            }
        ],
        "fields": [{"field_id": "announced_at", "value_type": "string"}],
        "actions": [
            {
                "action_id": "announce",
                "meaning": "make the announcement",
                "eligible_actors": ["a"],
                "required_authority": ["announce"],
                "stages": ["act"],
                "effects": effects,
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "act",
                    "stage": "act",
                    "participants": ["a"],
                    "action_ids": ["announce"],
                }
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "before",
                "args": [{"field": "announced_at"}, {"op": "horizon", "args": []}],
            },
            "description": "YES iff the announcement happened before the horizon",
        },
    }
    return _corpus(
        spec, actor_ids=["a"], expected_participants=1, target="announced before horizon"
    )


def test_undetermined_terminal_leaves_the_branch_unresolved() -> None:
    result, ctx = run_corpus(_date_terminal_world(set_the_date=False))
    # The run completes rather than raising, and reports no fabricated outcome.
    assert result.status is ForecastStatus.UNRESOLVED
    assert result.simulation_probability is None
    assert result.unresolved_mass > 0.999
    for b in result.branch_outcomes:
        assert not b.resolved and b.outcome is None
        assert "never determined" in (b.unresolved_reason or "")


def test_a_determined_date_still_resolves_normally() -> None:
    result, _ = run_corpus(_date_terminal_world(set_the_date=True))
    assert result.status is ForecastStatus.RESOLVED
    assert result.simulation_probability == 1.0


def test_undetermined_precondition_only_makes_the_action_infeasible() -> None:
    from _helpers import base_corpus, compile_dict
    from sworldmodel.effects import EffectExecutor
    from sworldmodel.executor import ActionExecutor
    from sworldmodel.gateway import DeterministicGateway
    from sworldmodel.worldspec import ActionDefinition, parse_expr

    compiled = compile_dict(base_corpus())
    world = compiled.base_world.with_stage("act")
    gated = ActionDefinition(
        action_id="needs_a_date",
        meaning="only possible once a date exists",
        eligible_actors=("*",),
        required_authority=("decide",),
        stages=("act",),
        preconditions=parse_expr(
            {"op": "before", "args": [{"field": "never_set"}, {"op": "horizon", "args": []}]}
        ),
    )
    executor = ActionExecutor(DeterministicGateway(), EffectExecutor())
    # Not feasible, and asking does not raise.
    assert not executor.is_feasible(world, world.actors["a"], gated)
