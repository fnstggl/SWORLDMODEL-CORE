"""The universal effect language: binding resolution, safe execution, resource checks."""

from __future__ import annotations

from dataclasses import replace

from _helpers import base_corpus, compile_dict
from sworldmodel.effects import UNIVERSAL_OPS, EffectExecutor
from sworldmodel.worldspec import Effect


def _world():
    return compile_dict(base_corpus()).base_world


def _binding(world):
    actor = world.actors["a"]
    return {"actor": "a", "self": actor.entity, "params": {"position": "hold"}, "target": "b"}


def test_binding_strings_are_resolved() -> None:
    world = _world()
    eff = Effect(
        op="append_record",
        params=(("collection", "votes"), ("key", "$actor"), ("value", "$param.position")),
    )
    ex = EffectExecutor()
    events = ex.build_events(world, (eff,), _binding(world))
    assert len(events) == 1
    p = events[0].payload_dict
    assert p["key"] == "a" and p["value"] == "hold"
    world2 = world.apply(events)
    recs = world2.get_records("votes")
    assert recs and recs[0]["value"] == "hold"


def test_resource_transfer_is_checked_and_applied() -> None:
    world = replace(_world(), resources=(("gold@a", 10.0),))
    ex = EffectExecutor()
    ok_transfer = Effect(
        op="transfer_resource",
        params=(("amount", 4.0), ("from", "a"), ("resource", "gold"), ("to", "b")),
    )
    ok, _ = ex.can_apply(world, (ok_transfer,), _binding(world))
    assert ok
    world2 = world.apply(ex.build_events(world, (ok_transfer,), _binding(world)))
    assert world2.get_resource("gold", "a") == 6.0
    assert world2.get_resource("gold", "b") == 4.0

    too_much = Effect(
        op="transfer_resource",
        params=(("amount", 99.0), ("from", "a"), ("resource", "gold"), ("to", "b")),
    )
    ok2, reason = ex.can_apply(world, (too_much,), _binding(world))
    assert not ok2 and "lacks" in reason


def test_consume_resource_reduces_holding() -> None:
    world = replace(_world(), resources=(("budget@a", 5.0),))
    ex = EffectExecutor()
    eff = Effect(
        op="consume_resource", params=(("amount", 2.0), ("holder", "a"), ("resource", "budget"))
    )
    world2 = world.apply(ex.build_events(world, (eff,), _binding(world)))
    assert world2.get_resource("budget", "a") == 3.0


def test_non_universal_op_is_refused() -> None:
    world = _world()
    ex = EffectExecutor()
    ok, reason = ex.can_apply(world, (Effect(op="rewrite_reality", params=()),), _binding(world))
    assert not ok and "universal" in reason


def test_effect_op_set_is_small_and_closed() -> None:
    # The execution language is a fixed, small set — not a growing action ontology.
    assert "cast_vote" not in UNIVERSAL_OPS
    assert "introduce_proposal" not in UNIVERSAL_OPS
    assert {
        "append_record",
        "set_field",
        "deliver_information",
        "transfer_resource",
    } <= UNIVERSAL_OPS
    assert len(UNIVERSAL_OPS) <= 14
