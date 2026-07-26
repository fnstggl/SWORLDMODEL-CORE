"""Runtime honesty: an unknown must never silently become a confident answer.

Covers audit findings:

* H-5 — an ``adjust_field`` whose delta expression is undetermined used to stay
  feasible (the ``None`` exemption in ``_unusable_quantity``), be coerced to ``+0.0``,
  and DETERMINE a previously-unset field — defeating any ``equals(field, None)``
  unresolved guard and producing a confident NO from an unknown.
* H-5 corollary — a direct-mode terminal over a field the world never initializes,
  with the default ``unresolved_when = const(False)``, resolved a confident NO because
  comparison coerces an absent quantity to zero. The compile gate now demands an
  is-unset guard for every such field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from _fakes import ProgrammableGateway, build_bundle, wait_decision
from sworldmodel.effects import EffectExecutor
from sworldmodel.engine import run
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.evidence import EvidenceStore, EvidenceView
from sworldmodel.expressions import evaluate
from sworldmodel.models import BranchWeight, ResolutionContract, WeightProvenance
from sworldmodel.world import WorldState
from sworldmodel.world_compiler import compile_world
from sworldmodel.worldspec import Effect, TerminalExpression, parse_expr

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def _terminal() -> TerminalExpression:
    return TerminalExpression(
        description="YES when total reaches 100",
        yes_when=parse_expr(
            {"op": "greater_or_equal", "args": [{"op": "field", "args": ["total"]}, 100]}
        ),
        unresolved_when=parse_expr(
            {"op": "equals", "args": [{"op": "field", "args": ["total"]}, None]}
        ),
    )


def _world() -> WorldState:
    contract = ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity="s",
        resolution_units="binary",
        terminal=_terminal(),
    )
    return WorldState(
        branch_id="b",
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.DIRECT_EMPIRICAL, "test"),
        time=AS_OF,
        contract=contract,
        evidence=EvidenceView(store=EvidenceStore(), as_of=AS_OF),
    )


# The delta is base * mult with mult never set anywhere: the expression is
# undetermined, and no value for the field exists in the world.
_UNDETERMINED_DELTA = {
    "op": "multiply",
    "args": [{"op": "field", "args": ["base"]}, {"op": "field", "args": ["mult"]}],
}


# ---------------------------------------------------------------------------
# H-5: an undetermined adjust_field delta refuses instead of determining a field
# ---------------------------------------------------------------------------


def test_undetermined_adjust_field_delta_makes_the_action_infeasible() -> None:
    world = _world()
    eff = Effect(op="adjust_field", params=(("field", "total"), ("delta", _UNDETERMINED_DELTA)))
    ok, reason = EffectExecutor().can_apply(world, (eff,), {"actor": "a"})
    assert not ok
    assert "could not determine" in reason


def test_undetermined_adjust_field_never_determines_the_field() -> None:
    """Even when the event is built and applied (the deferred-effect path re-resolves
    parameters at fire time, past the feasibility check), the field stays unset and the
    unresolved guard over it keeps holding."""

    world = _world()
    eff = Effect(op="adjust_field", params=(("field", "total"), ("delta", _UNDETERMINED_DELTA)))
    events, deferred = EffectExecutor().build_events(world, (eff,), {"actor": "a"})
    assert not deferred
    assert events[0].payload_dict["delta"] is None  # never coerced to 0.0
    after = world.apply(events)
    assert after.get_field("total") is None, "an undetermined delta determined the field"
    assert evaluate(world.contract.terminal.unresolved_when, after) is True


def test_undetermined_transfer_and_consume_amounts_are_refused_and_move_nothing() -> None:
    world = _world()
    for op, params in (
        (
            "transfer_resource",
            (("resource", "funds"), ("from", "a"), ("to", "b"), ("amount", _UNDETERMINED_DELTA)),
        ),
        (
            "consume_resource",
            (("resource", "funds"), ("holder", "a"), ("amount", _UNDETERMINED_DELTA)),
        ),
    ):
        eff = Effect(op=op, params=params)
        ok, reason = EffectExecutor().can_apply(world, (eff,), {"actor": "a"})
        assert not ok, f"{op} with an undetermined amount stayed feasible"
        assert "could not determine" in reason
        events, _ = EffectExecutor().build_events(world, (eff,), {"actor": "a"})
        after = world.apply(events)
        assert dict(after.resources) == dict(world.resources), (
            f"{op} with an undetermined amount moved resources"
        )


def test_missing_actor_parameter_quantity_is_refused_not_zeroed() -> None:
    """``$param.amount`` with no such parameter resolves to None — the same hole, one
    binding step earlier. The action must refuse, not transfer nothing as done."""

    world = _world()
    eff = Effect(
        op="adjust_field", params=(("field", "total"), ("delta", "$param.amount"))
    )
    ok, reason = EffectExecutor().can_apply(world, (eff,), {"actor": "a", "params": {}})
    assert not ok
    assert "could not determine" in reason


# ---------------------------------------------------------------------------
# H-5 corollary: a terminal over a never-initialized field must carry an
# is-unset guard, or the compile gate refuses
# ---------------------------------------------------------------------------


def _unset_field_world(unresolved_when: dict[str, Any]) -> dict[str, Any]:
    """A world whose terminal reads ``total``, a field with NO compiled initial value,
    produced only if the operator's action runs."""

    return {
        "reality": {
            "as_of": AS_OF.isoformat(),
            "horizon": HORIZON.isoformat(),
            "subject_entity": "the total",
            "resolution_units": "units",
            "target_outcome": "the total reaches 100",
            "expected_participants": 1,
        },
        "claims": [
            {
                "id": "c_actor",
                "proposition": "the operator and their mandate are documented",
                "value": True,
                "supporting_excerpt": "the operator and their mandate are documented",
            },
            {
                "id": "c_cycle",
                "proposition": "the production cycle is scheduled",
                "value": True,
                "supporting_excerpt": "the production cycle is scheduled",
            },
        ],
        "world_spec": {
            "title": "unset terminal field",
            "subject_entity": "the total",
            "resolution_units": "units",
            "entities": [
                {
                    "entity_id": "operator",
                    "name": "The Operator",
                    "kind": "person",
                    "is_actor": True,
                    "role": "operator",
                    "authority": ["produce"],
                    "representation_scale": "individual",
                    "evidence_claim_ids": ["c_actor"],
                }
            ],
            "actors": [
                {
                    "entity_id": "operator",
                    "reasoning": "produces when scheduled",
                    "memory_seeds": [
                        {
                            "content": "I run the cycle.",
                            "kind": "episodic",
                            "importance": 0.9,
                            "evidence_claim_ids": ["c_actor"],
                        }
                    ],
                }
            ],
            "fields": [{"field_id": "total", "value_type": "number"}],  # no initial
            "actions": [
                {
                    "action_id": "produce",
                    "meaning": "run a production step",
                    "eligible_actors": ["role:operator"],
                    "required_authority": ["produce"],
                    "parameters": [],
                    "visibility": "public",
                    "effects": [{"op": "adjust_field", "field": "total", "delta": 60}],
                    "evidence_claim_ids": ["c_cycle"],
                }
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "cycle",
                        "stage": "cycle",
                        "at": "2026-06-01T09:00:00+00:00",
                        "description": "the cycle",
                        "participants": ["operator"],
                        "action_ids": ["produce"],
                        "allow_novel": False,
                    }
                ]
            },
            "external_processes": [],
            "wake_rules": [],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [{"op": "field", "args": ["total"]}, 100],
                },
                "unresolved_when": unresolved_when,
                "description": "YES when total reaches 100",
            },
        },
    }


def _compile(data: dict[str, Any], gw: ProgrammableGateway) -> Any:
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="will the total reach 100?",
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
        gateway=gw,
        seed=0,
        max_branches=2,
    )


def _wait_gateway() -> ProgrammableGateway:
    return ProgrammableGateway(
        {
            "actor_decision": lambda ctx: wait_decision("observing"),
            "reflect": {"beliefs_update": [], "new_memories": []},
        }
    )


def test_unguarded_terminal_over_unset_field_is_refused_at_compile() -> None:
    """The direct compiler's default ``unresolved_when = const(False)`` plus total
    comparison (absent -> 0.0) used to resolve a confident NO over a quantity nobody
    produced. The gate refuses with a recompilable failure naming the fields."""

    data = _unset_field_world({"op": "const", "args": [False]})
    with pytest.raises(WorldIntegrityError) as exc_info:
        _compile(data, _wait_gateway())
    details = exc_info.value.details
    assert details["failure"] == "terminal_unset_fields_unguarded"
    assert details["recompilable"] is True
    assert details["unguarded fields"] == ["total"]


def test_guarded_terminal_over_unset_field_compiles_and_stays_unresolved() -> None:
    """The exact guard shape the semantic lowerer derives for UNKNOWN states —
    ``equals(field(x), None)`` — passes the gate, and a branch in which nothing ever
    produces the field reports unresolved rather than a manufactured NO."""

    data = _unset_field_world(
        {"op": "equals", "args": [{"op": "field", "args": ["total"]}, None]}
    )
    gw = _wait_gateway()
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    (branch,) = result.branch_outcomes
    assert not branch.resolved
    assert branch.outcome is None
