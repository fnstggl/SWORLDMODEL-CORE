"""Runtime honesty: an unknown must never silently become a confident answer.

Covers audit findings:

* H-5 — an ``adjust_field`` whose delta expression is undetermined used to stay
  feasible (the ``None`` exemption in ``_unusable_quantity``), be coerced to ``+0.0``,
  and DETERMINE a previously-unset field — defeating any ``equals(field, None)``
  unresolved guard and producing a confident NO from an unknown.
"""

from __future__ import annotations

from datetime import datetime

from sworldmodel.effects import EffectExecutor
from sworldmodel.evidence import EvidenceStore, EvidenceView
from sworldmodel.expressions import evaluate
from sworldmodel.models import BranchWeight, ResolutionContract, WeightProvenance
from sworldmodel.world import WorldState
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
