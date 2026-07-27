"""Stocks, flows and cadences — conservation and repetition, on invented worlds.

Two live failures are the reason this file exists, and neither is about its domain.

FD-24: a compiled world subtracted deliveries from an inventory it did not have, every
week for a quarter, and ended at minus forty-eight thousand units with a negative order
backlog. The runtime has always refused to overdraw a resource — ``transfer_resource``
and ``consume_resource`` are checked — but the semantic vocabulary had no way to say
"this quantity is a physical stock", so every quantity lowered to bare arithmetic on a
field and the conservation was unreachable. The fix is a word, not a validator: a state
declares its kind, and a stock is lowered onto the machinery that already conserves.

FD-25: the same world applied a WEEKLY rate at two dates in a ten-week quarter, so the
quarter's total was the rate times two and every branch resolved NO. The published
number was an artifact of how many dates were typed. The fix is again a word: a process
declares its recurrence and code enumerates the firings.

Every world below is invented — a grain elevator on a branch line, a ferry berth, a
reservoir — and none of them is any acceptance question. What is proven is shape:

* a stock cannot be driven below zero, through the real engine and the real executor;
* a declared capacity is a ceiling by the same conservation, not a second check;
* a flow applied over a declared recurrence fires the number of times the window holds —
  thirteen for a weekly rate across a quarter, not two;
* an under-enumerated cadence is refused, and the refusal names the shortfall;
* recurrence generation is deterministic and bounded;
* a level is exactly as unconstrained as it has always been;
* the standing claims a reviewer must audit reach the artifact a reviewer reads.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from _fakes import ProgrammableGateway, build_bundle
from sworldmodel.effects import EffectExecutor
from sworldmodel.engine import run
from sworldmodel.models import ResolutionContract
from sworldmodel.semantic_lowering import lower_plan
from sworldmodel.semantic_plan import (
    RECURRENCE_OCCURRENCE_CAP,
    parse_semantic_plan,
    validate_semantic_plan,
)
from sworldmodel.world_compiler import build_base_world, compile_world
from sworldmodel.worldspec import parse_world_spec

AS_OF = datetime.fromisoformat("2026-01-05T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-04-10T23:59:59+00:00")

# A quarter, to the day: thirteen weekly firings from the first to the last inclusive.
QUARTER_START = "2026-01-12T09:00:00+00:00"
QUARTER_END = "2026-04-06T09:00:00+00:00"
WEEKLY_FIRINGS = 13

CLAIMS: dict[str, dict[str, Any]] = {
    "c-e1": {
        "proposition": "The Halvard Reach grain elevator held 10000 tonnes of grain at the "
        "start of January",
        "value": "10000 tonnes",
        "entities": ["Halvard Reach Grain Elevator"],
    },
    "c-e2": {
        "proposition": "The Halvard Reach elevator's branch-line contract loads out 4000 "
        "tonnes a week",
        "value": "4000 tonnes per week",
        "entities": ["Halvard Reach Grain Elevator"],
    },
    "c-e3": {
        "proposition": "The Halvard Reach elevator has shipped nothing against this "
        "season's contract so far",
        "value": "0 tonnes",
        "entities": ["Halvard Reach Grain Elevator"],
    },
    "c-b1": {
        "proposition": "6000 vehicles are booked and waiting on the Kestrel Reach ferry "
        "berth's standing queue",
        "value": "6000 vehicles",
        "entities": ["Kestrel Reach Ferry Berth"],
    },
    "c-b2": {
        "proposition": "The Kestrel Reach berth's published timetable sails 400 vehicles a "
        "week",
        "value": "400 vehicles per week",
        "entities": ["Kestrel Reach Ferry Berth"],
    },
    "c-b3": {
        "proposition": "The Kestrel Reach berth has carried no vehicles against this "
        "quarter's tally",
        "value": "0 vehicles",
        "entities": ["Kestrel Reach Ferry Berth"],
    },
    "c-r1": {
        "proposition": "Sorrel Gap reservoir stored 9000 acre-feet at the start of January "
        "against a full pool of 12000",
        "value": "9000 of 12000 acre-feet",
        "entities": ["Sorrel Gap Reservoir"],
    },
    "c-r2": {
        "proposition": "Sorrel Gap's gauged winter inflow runs at 800 acre-feet a week",
        "value": "800 acre-feet per week",
        "entities": ["Sorrel Gap Reservoir"],
    },
}


def _validate(plan: dict[str, Any]) -> list[str]:
    return validate_semantic_plan(
        parse_semantic_plan(plan),
        as_of=AS_OF,
        horizon=HORIZON,
        known_claim_ids=frozenset(CLAIMS),
    )


def _defects(errors: list[str]) -> set[str]:
    return {e.split(":", 1)[0] for e in errors}


def _entity(name: str, role: str, claim: str) -> dict[str, Any]:
    return {
        "name": name,
        "structural_type": "institution",
        "role": role,
        "representation_scale": "organization",
        "decides": False,
        "authority": f"operates {role}",
        "why_material": f"its own operation produces every unit the terminal counts ({role})",
        "terminal_state_it_can_change": "its operating cycle moves the quantity the "
        "terminal reads",
        "information_received": "its own operating records",
        "if_removed": "nothing operates and no quantity moves at all",
        "evidence_claim_ids": [claim],
    }


def _zero_actor(claims: list[str], what: str) -> dict[str, Any]:
    return {
        "no_material_decision": f"the {what} runs to a contracted schedule the record "
        "states, and no person or body named in the record can add to it or withhold it "
        "inside the window",
        "process_sufficiency": f"the starting quantity and the gauged rate together "
        f"determine every unit the {what} moves",
        "evidence_claim_ids": claims,
    }


# ---------------------------------------------------------------------------
# Domain 1 — a grain elevator on a branch line: a stock that runs out
# ---------------------------------------------------------------------------


def elevator_plan(*, held: int = 10000, threshold: int = 20000) -> dict[str, Any]:
    """A contracted weekly loadout drawing on a finite pile of grain.

    Thirteen weekly firings are declared and thirteen are generated; only two of them
    can actually happen, because after 8000 tonnes have gone out there is not another
    4000 tonnes in the elevator. A world that models the pile as a plain number ships
    52000 tonnes out of a 10000-tonne elevator and answers YES.
    """

    return {
        "resolution": {
            "question": f"Will the Halvard Reach elevator ship more than {threshold} "
            "tonnes against this season's contract by April?",
            "yes_condition": f"Tonnage shipped this season exceeds {threshold}.",
            "subject_entity": "Halvard Reach Grain Elevator",
            "resolution_units": "tonnes shipped",
            "target_outcome": f"more than {threshold} tonnes shipped",
            "expected_participants": None,
            "evidence_claim_ids": ["c-e1"],
        },
        "entities": [
            _entity("Halvard Reach Grain Elevator", "the loading berth and the pile", "c-e1")
        ],
        "excluded_candidates": [],
        "zero_actor_justification": _zero_actor(["c-e1", "c-e2"], "loadout"),
        "states": [
            {
                "name": "grain in the elevator",
                "owner": "Halvard Reach Grain Elevator",
                "state_type": "quantity",
                "kind": "stock",
                "unit": "tonnes",
                "initial": held,
                "why_material": "every tonne shipped comes out of this pile",
                "evidence_claim_ids": ["c-e1"],
            },
            {
                "name": "weekly loadout rate",
                "owner": "world",
                "state_type": "quantity",
                "kind": "flow",
                "period": "P1W",
                "unit": "tonnes",
                "initial": 4000,
                "why_material": "the contracted rate the branch line takes away",
                "evidence_claim_ids": ["c-e2"],
            },
            {
                "name": "grain shipped this season",
                "owner": "world",
                "state_type": "quantity",
                "kind": "level",
                "unit": "tonnes",
                "initial": 0,
                "why_material": "the terminal reads it",
                "not_a_stock_because": "the season's tally records tonnage that has already "
                "left; nothing draws grain back out of it and it fills no container",
                "evidence_claim_ids": ["c-e3"],
            },
        ],
        "events": [],
        "affordances": [],
        "processes": [
            {
                "name": "weekly loadout",
                "meaning": "the branch line takes the contracted tonnage out of the pile",
                "kind": "operational",
                "inputs": ["grain in the elevator"],
                "recurrence": {
                    "period": "P1W",
                    "start": QUARTER_START,
                    "end": QUARTER_END,
                    "description": "a contracted weekly loadout",
                    "changes": [
                        {
                            "op": "decrease",
                            "target": "grain in the elevator",
                            "amount": {"kind": "state", "state": "weekly loadout rate"},
                        },
                        {
                            "op": "increase",
                            "target": "grain shipped this season",
                            "amount": {"kind": "state", "state": "weekly loadout rate"},
                        },
                    ],
                },
                "occurrences": [],
                "evidence_claim_ids": ["c-e2"],
            }
        ],
        "uncertainties": [],
        "terminal": {
            "form": "quantity_comparison",
            "state": "grain shipped this season",
            "comparison": "greater_than",
            "threshold": threshold,
        },
        "terminal_producer_note": "The season's tally starts at nothing and rises only as "
        "the weekly loadout takes grain out of the pile; when the pile is short the "
        "loadout does not happen, so the tally can never exceed what the elevator held.",
        "world_facts": [],
    }


# ---------------------------------------------------------------------------
# Domain 2 — a ferry berth: the cadence itself
# ---------------------------------------------------------------------------


def berth_plan(*, hand_written: list[str] | None = None, **recurrence: Any) -> dict[str, Any]:
    """A weekly sailing rate applied across a quarter.

    Exactly the shape that failed: a rate quoted per week, a window a quarter long, and
    a total that is the rate times however many dates the calendar contains. With
    ``hand_written`` the calendar is typed out by hand — two dates for a thirteen-week
    quarter — and with the default it is declared once as a cadence.
    """

    rec = {
        "period": "P1W",
        "start": QUARTER_START,
        "end": QUARTER_END,
        "description": "the timetabled weekly sailing",
        "changes": [
            {
                "op": "decrease",
                "target": "vehicles waiting at the berth",
                "amount": {"kind": "state", "state": "weekly sailing capacity"},
            },
            {
                "op": "increase",
                "target": "vehicles carried this quarter",
                "amount": {"kind": "state", "state": "weekly sailing capacity"},
            },
        ],
    }
    rec.update(recurrence)
    process: dict[str, Any] = {
        "name": "timetabled sailings",
        "meaning": "the berth sails its timetable and the queue goes down",
        "kind": "operational",
        "inputs": ["vehicles waiting at the berth"],
        "recurrence": rec,
        "occurrences": [],
        "evidence_claim_ids": ["c-b2"],
    }
    if hand_written is not None:
        process["recurrence"] = None
        process["occurrences"] = [
            {"description": "a sailing", "at": at, "changes": list(rec["changes"])}
            for at in hand_written
        ]
    return {
        "resolution": {
            "question": "Will the Kestrel Reach berth carry more than 4000 vehicles this "
            "quarter?",
            "yes_condition": "Vehicles carried this quarter exceed 4000.",
            "subject_entity": "Kestrel Reach Ferry Berth",
            "resolution_units": "vehicles carried",
            "target_outcome": "more than 4000 vehicles carried",
            "expected_participants": None,
            "evidence_claim_ids": ["c-b1"],
        },
        "entities": [_entity("Kestrel Reach Ferry Berth", "the berth and its queue", "c-b1")],
        "excluded_candidates": [],
        "zero_actor_justification": _zero_actor(["c-b1", "c-b2"], "timetable"),
        "states": [
            {
                "name": "vehicles waiting at the berth",
                "owner": "Kestrel Reach Ferry Berth",
                "state_type": "quantity",
                "kind": "stock",
                "unit": "vehicles",
                "initial": 6000,
                "why_material": "every vehicle carried comes off this queue",
                "evidence_claim_ids": ["c-b1"],
            },
            {
                "name": "weekly sailing capacity",
                "owner": "world",
                "state_type": "quantity",
                "kind": "flow",
                "period": "P1W",
                "unit": "vehicles",
                "initial": 400,
                "why_material": "the timetabled rate the berth can move each week",
                "evidence_claim_ids": ["c-b2"],
            },
            {
                "name": "vehicles carried this quarter",
                "owner": "world",
                "state_type": "quantity",
                "kind": "level",
                "unit": "vehicles",
                "initial": 0,
                "why_material": "the terminal reads it",
                "not_a_stock_because": "the quarter's tally counts crossings already made; "
                "nothing takes a crossing back out of it",
                "evidence_claim_ids": ["c-b3"],
            },
        ],
        "events": [],
        "affordances": [],
        "processes": [process],
        "uncertainties": [],
        "terminal": {
            "form": "quantity_comparison",
            "state": "vehicles carried this quarter",
            "comparison": "greater_than",
            "threshold": 4000,
        },
        "terminal_producer_note": "The tally rises only as timetabled sailings take "
        "vehicles off the queue, at the published weekly rate, across the whole quarter.",
        "world_facts": [],
    }


# ---------------------------------------------------------------------------
# Domain 3 — a reservoir: a ceiling, and a level that is free to go negative
# ---------------------------------------------------------------------------


def reservoir_plan() -> dict[str, Any]:
    """Winter inflow filling a pool that has a top.

    The capacity is not a second check bolted onto the stock; it is the counterpart
    holder of the same conserved quantity, so filling past the full pool is drawing on a
    holder that has run out. Alongside it, a plain level records the basin's net balance
    against its downstream commitments, which really can go below zero — and does.
    """

    return {
        "resolution": {
            "question": "Will Sorrel Gap reservoir hold more than 11000 acre-feet by April?",
            "yes_condition": "Storage exceeds 11000 acre-feet.",
            "subject_entity": "Sorrel Gap Reservoir",
            "resolution_units": "acre-feet stored",
            "target_outcome": "more than 11000 acre-feet stored",
            "expected_participants": None,
            "evidence_claim_ids": ["c-r1"],
        },
        "entities": [_entity("Sorrel Gap Reservoir", "the dam and its pool", "c-r1")],
        "excluded_candidates": [],
        "zero_actor_justification": _zero_actor(["c-r1", "c-r2"], "winter inflow"),
        "states": [
            {
                "name": "water stored behind the dam",
                "owner": "Sorrel Gap Reservoir",
                "state_type": "quantity",
                "kind": "stock",
                "capacity": 12000,
                "unit": "acre-feet",
                "initial": 9000,
                "why_material": "the terminal reads it, and the dam cannot hold more than "
                "its full pool",
                "evidence_claim_ids": ["c-r1"],
            },
            {
                "name": "weekly gauged inflow",
                "owner": "world",
                "state_type": "quantity",
                "kind": "flow",
                "period": "P1W",
                "unit": "acre-feet",
                "initial": 800,
                "why_material": "the gauged rate at which the basin delivers water",
                "evidence_claim_ids": ["c-r2"],
            },
            {
                "name": "basin net balance against downstream commitments",
                "owner": "world",
                "state_type": "quantity",
                "kind": "level",
                "unit": "acre-feet",
                "initial": 0,
                "why_material": "the district's running over- or under-delivery position",
                "not_a_stock_because": "a net position against commitments is a difference, "
                "not water held anywhere: being 400 acre-feet behind the schedule is a real "
                "and negative state of that account",
                "evidence_claim_ids": ["c-r2"],
            },
        ],
        "events": [],
        "affordances": [],
        "processes": [
            {
                "name": "winter inflow",
                "meaning": "the gauged winter inflow arrives week by week and the account "
                "against downstream commitments moves with it",
                "kind": "operational",
                "inputs": ["water stored behind the dam"],
                "recurrence": {
                    "period": "P1W",
                    "start": QUARTER_START,
                    "end": QUARTER_END,
                    "description": "a week of gauged inflow",
                    "changes": [
                        {
                            "op": "increase",
                            "target": "water stored behind the dam",
                            "amount": {"kind": "state", "state": "weekly gauged inflow"},
                        },
                        {
                            "op": "decrease",
                            "target": "basin net balance against downstream commitments",
                            "amount": {"kind": "literal", "value": 100},
                        },
                    ],
                },
                "occurrences": [],
                "evidence_claim_ids": ["c-r2"],
            }
        ],
        "uncertainties": [],
        "terminal": {
            "form": "quantity_comparison",
            "state": "water stored behind the dam",
            "comparison": "greater_than",
            "threshold": 11000,
        },
        "terminal_producer_note": "Storage starts at the cited 9000 acre-feet and rises "
        "only as gauged inflow arrives, and only as far as the full pool allows.",
        "world_facts": [],
    }


# ---------------------------------------------------------------------------
# Running the invented worlds through the real compiler and the real engine
# ---------------------------------------------------------------------------


def _cited_ids(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "evidence_claim_ids" and isinstance(value, list):
                out |= {str(v) for v in value}
            else:
                out |= _cited_ids(value)
    elif isinstance(node, list):
        for item in node:
            out |= _cited_ids(item)
    return out


def _bundle(plan: dict[str, Any]) -> Any:
    compilation, _ = lower_plan(parse_semantic_plan(plan))
    cited = sorted(cid for cid in CLAIMS if cid in _cited_ids(plan))
    return build_bundle(
        {
            "world_spec": compilation["world_spec"],
            "uncertainties": compilation["uncertainties"],
            "world_facts": compilation["world_facts"],
            "required_reality_facts": compilation["required_reality_facts"],
            "reality": {
                "subject_entity": compilation["subject_entity"],
                "resolution_units": compilation["resolution_units"],
                "target_outcome": compilation["target_outcome"],
                "expected_participants": compilation["expected_participants"],
                "as_of": AS_OF.isoformat(),
                "horizon": HORIZON.isoformat(),
            },
            "claims": [
                dict(CLAIMS[cid], id=cid, published_at="2026-01-02T00:00:00+00:00")
                for cid in cited
            ],
        }
    )


def _contract(plan: dict[str, Any], bundle: Any) -> ResolutionContract:
    return ResolutionContract(
        question=str(plan["resolution"]["question"]),
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )


def _run(plan: dict[str, Any]) -> Any:
    """Compile and run an invented world on the real engine — no actors, no model."""

    bundle = _bundle(plan)
    gw = ProgrammableGateway({"reflect": {"beliefs_update": [], "new_memories": []}})
    compiled = compile_world(
        _contract(plan, bundle),
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gw,
        seed=0,
        max_branches=4,
    )
    return run(compiled, gw, seed=0)


def _final(result: Any) -> Any:
    (world,) = list(result.final_worlds.values())
    return world


# ---------------------------------------------------------------------------
# FD-24 — a stock cannot be driven below zero
# ---------------------------------------------------------------------------


def test_a_stock_cannot_be_driven_negative_through_the_real_engine() -> None:
    """Thirteen weekly loadouts are scheduled; the elevator only has grain for two.

    The refusal is the runtime's own: the loadout moves grain with the conserved
    operations, and a firing that would take more than the holder has does not happen at
    all — never in part, and never recorded as done. The season's tally can therefore
    never exceed what the elevator held, which is the whole difference between a
    simulated operating system and arithmetic that happens to be signed.
    """

    result = _run(elevator_plan())
    world = _final(result)
    fields = world.fields_dict()
    resources = dict(world.resources)

    assert fields["grain_in_the_elevator"] == 2000.0
    assert fields["grain_shipped_this_season"] == 8000.0
    assert resources["grain_in_the_elevator@halvard_reach_grain_elevator"] == 2000.0

    # No reading of the stock, at any point in the run, is below zero — the field the
    # world reports and the resource the executor conserves move together or not at all.
    assert all(
        quantity >= 0.0 for key, quantity in resources.items() if key.startswith("grain_in_")
    )
    for event in world.event_history:
        if event.kind == "adjust_field" and event.payload_dict.get("field") == (
            "grain_in_the_elevator"
        ):
            assert event.payload_dict["delta"] == -4000.0
    shipped = [e for e in world.event_history if e.kind == "consume_resource"]
    assert len(shipped) == 2, "only the firings the pile could pay for happened"

    # And the conservation is what decides the answer: unconserved, thirteen firings of
    # 4000 tonnes would ship 52000 out of a 10000-tonne elevator and resolve YES.
    outcomes = {(o.resolved, o.outcome) for o in result.branch_outcomes}
    assert outcomes == {(True, "NO")}


def test_the_executor_itself_refuses_the_overdrawing_move() -> None:
    """The same refusal on the actor path, from the executor's own dry run.

    ``EffectExecutor.can_apply`` is what :class:`ActionExecutor` consults before an
    action starts and again before it completes. Handed the compiled effects of a
    loadout against an elevator that cannot pay for it, it refuses by name — no new
    check was added anywhere for this to be true.
    """

    plan = elevator_plan(held=1000)
    compilation, _ = lower_plan(parse_semantic_plan(plan))
    bundle = _bundle(plan)
    world = build_base_world(
        bundle.spec, _contract(plan, bundle), bundle.evidence_store.view(AS_OF), ()
    )
    spec = parse_world_spec(compilation["world_spec"])
    (process,) = spec.external_processes
    effects = process.occurrences[0].effects

    ok, why = EffectExecutor().can_apply(world, effects, {"actor": None, "self": None})
    assert ok is False
    assert "lacks" in why and "grain_in_the_elevator" in why

    # With grain in the pile the identical effects are feasible: the refusal is about the
    # balance, not about the shape of the move.
    plenty = _bundle(elevator_plan(held=99000))
    stocked = build_base_world(
        plenty.spec, _contract(plan, plenty), plenty.evidence_store.view(AS_OF), ()
    )
    assert EffectExecutor().can_apply(stocked, effects, {"actor": None, "self": None})[0] is True


def test_a_declared_capacity_is_a_ceiling_by_the_same_conservation() -> None:
    """A stock fills to its capacity and stops, because the room above it runs out.

    Nine thousand acre-feet in a twelve-thousand acre-foot pool leaves room for three
    weeks of eight-hundred-acre-foot inflow and not a fourth: the fourth transfer would
    draw 800 from a counterpart holder holding 600, and the executor's conservation is
    the same refusal in the other direction.
    """

    world = _final(_run(reservoir_plan()))
    fields = world.fields_dict()
    resources = dict(world.resources)

    assert fields["water_stored_behind_the_dam"] == 11400.0
    assert resources["water_stored_behind_the_dam@sorrel_gap_reservoir"] == 11400.0
    assert resources["water_stored_behind_the_dam@beyond_water_stored_behind_the_dam"] == 600.0
    total = sum(v for k, v in resources.items() if k.startswith("water_stored_"))
    assert total == 12000.0, "the stock and the room above it always sum to the capacity"


def test_a_level_keeps_todays_unconstrained_behavior() -> None:
    """A net position is a difference, and differences go negative.

    The reservoir's account against its downstream commitments is declared a level with
    its reason, and nothing in this change constrains it: it is written with the same
    bare ``adjust_field`` it has always used and ends the quarter below zero, in the same
    run in which the stock beside it refused to.
    """

    world = _final(_run(reservoir_plan()))
    balance = world.fields_dict()["basin_net_balance_against_downstream_commitments"]
    assert balance == -1300.0, "thirteen weekly firings of -100, unconstrained"

    compilation, _ = lower_plan(parse_semantic_plan(reservoir_plan()))
    (process,) = compilation["world_spec"]["external_processes"]
    ops = {e["op"] for e in process["occurrences"][0]["effects"]}
    assert ops == {"transfer_resource", "adjust_field"}
    level_writes = [
        e
        for e in process["occurrences"][0]["effects"]
        if e["op"] == "adjust_field"
        and e["field"] == "basin_net_balance_against_downstream_commitments"
    ]
    assert level_writes == [
        {
            "op": "adjust_field",
            "field": "basin_net_balance_against_downstream_commitments",
            "delta": -100,
        }
    ]


def test_a_decrease_on_an_unexplained_quantity_level_is_refused() -> None:
    """The FD-24 shape at the earliest stage: the plan that could not say so is refused.

    A quantity something draws down is either a stock — in which case the runtime
    conserves it — or a level whose plan says why it may go below zero. There is no
    third option, because the third option is what shipped a quarter of negative
    inventory.
    """

    plan = elevator_plan()
    stock = next(s for s in plan["states"] if s["name"] == "grain in the elevator")
    stock["kind"] = "level"
    stock.pop("capacity", None)
    errors = _validate(plan)
    assert "UNCONSERVED_PHYSICAL_STOCK" in _defects(errors), errors
    message = next(e for e in errors if e.startswith("UNCONSERVED_PHYSICAL_STOCK"))
    assert "grain in the elevator" in message
    assert "kind=stock" in message and "not_a_stock_because" in message

    # And the escape hatch is real, because some quantities genuinely do go negative.
    stock["not_a_stock_because"] = "the elevator's book position can be short against "
    "forward sales, which is a negative number the trade recognises"
    assert "UNCONSERVED_PHYSICAL_STOCK" not in _defects(_validate(plan))


def test_an_accumulating_terminal_quantity_has_to_say_what_kind_it_is() -> None:
    """The answer turns on it, so the plan says whether it is held or merely counted."""

    plan = berth_plan()
    tally = next(s for s in plan["states"] if s["name"] == "vehicles carried this quarter")
    tally.pop("not_a_stock_because")
    errors = _validate(plan)
    assert "ACCUMULATING_QUANTITY_UNDECLARED" in _defects(errors), errors
    assert "vehicles carried this quarter" in errors[0]


def test_a_stock_declaration_that_cannot_be_conserved_is_refused() -> None:
    """A conserved quantity needs a known starting amount, a ceiling to grow into, and
    moves rather than assignments — each refused by name with its own correction."""

    unknown = elevator_plan()
    next(s for s in unknown["states"] if s["name"] == "grain in the elevator")["initial"] = "UNKNOWN"
    assert "STOCK_DECLARATION_INCOMPLETE" in _defects(_validate(unknown))

    grows = elevator_plan()
    grows["processes"][0]["recurrence"]["changes"][0]["op"] = "increase"
    errors = _validate(grows)
    assert "STOCK_DECLARATION_INCOMPLETE" in _defects(errors), errors
    assert "capacity" in next(e for e in errors if e.startswith("STOCK_DECLARATION"))

    assigned = elevator_plan()
    assigned["processes"][0]["recurrence"]["changes"][0] = {
        "op": "set",
        "target": "grain in the elevator",
        "value": {"kind": "literal", "value": 0},
    }
    assert "UNCONSERVED_PHYSICAL_STOCK" in _defects(_validate(assigned))


# ---------------------------------------------------------------------------
# FD-25 — a rate is applied as often as its window holds
# ---------------------------------------------------------------------------


def test_a_weekly_flow_over_a_quarter_is_applied_thirteen_times() -> None:
    """The Tesla shape, in a berth: thirteen firings, not two.

    The planner declares "every P1W from the twelfth of January to the sixth of April"
    and code produces the calendar. The quarter's total is then the rate times the number
    of weeks the quarter actually holds — a fact about the window rather than about how
    many dates somebody was willing to type.
    """

    plan = berth_plan()
    assert _validate(plan) == []

    compilation, mapping = lower_plan(parse_semantic_plan(plan))
    (process,) = compilation["world_spec"]["external_processes"]
    assert len(process["occurrences"]) == WEEKLY_FIRINGS

    dates = [o["at"] for o in process["occurrences"]]
    assert dates[0] == QUARTER_START and dates[-1] == QUARTER_END
    gaps = {
        datetime.fromisoformat(b) - datetime.fromisoformat(a)
        for a, b in zip(dates, dates[1:], strict=False)
    }
    assert len(gaps) == 1 and gaps.pop().days == 7

    world = _final(_run(plan))
    fields = world.fields_dict()
    assert fields["vehicles_carried_this_quarter"] == 400.0 * WEEKLY_FIRINGS == 5200.0
    assert fields["vehicles_waiting_at_the_berth"] == 6000.0 - 5200.0

    # The cadence, and the calendar it produced, are both in the mapping a reader gets.
    (record,) = [r for r in mapping if r["namespace"] == "recurrence"]
    assert "every P1W" in record["semantic"]
    assert f"{WEEKLY_FIRINGS} enumerated" in record["lowering_rule"]


def test_an_under_enumerated_cadence_is_refused_and_names_the_shortfall() -> None:
    """Two dates for a thirteen-week quarter is the defect, stated as arithmetic.

    This is exactly the published failure: a weekly rate fired twice across a quarter, so
    the total was the rate times two and every branch resolved NO on an artifact of the
    schedule. The refusal does not ask the planner to count occurrences — it asks them to
    declare the cadence, and says how far short the calendar they wrote falls.
    """

    plan = berth_plan(hand_written=[QUARTER_START, QUARTER_END])
    errors = _validate(plan)
    assert "UNDER_ENUMERATED_CADENCE" in _defects(errors), errors
    message = next(e for e in errors if e.startswith("UNDER_ENUMERATED_CADENCE"))
    assert "weekly sailing capacity" in message and "P1W" in message
    assert f"holds {WEEKLY_FIRINGS} firings" in message
    assert "enumerates only 2" in message and "short by 11" in message
    assert "declare recurrence" in message

    # Enumerating the real calendar by hand is legal — the rule is about the cadence
    # being honoured, not about which of the two ways says so.
    every_week = [
        (datetime.fromisoformat(QUARTER_START).replace(day=12) if i == 0 else None)
        for i in range(1)
    ]
    del every_week
    start = datetime.fromisoformat(QUARTER_START)
    weekly = [(start.replace() + (i * (datetime.fromisoformat(QUARTER_END) - start) / 12)) for i in range(13)]
    assert _validate(berth_plan(hand_written=[w.isoformat() for w in weekly])) == []


def test_a_process_may_not_carry_a_cadence_and_a_calendar_at_once() -> None:
    """Two calendars is no calendar: the reader cannot tell which one runs."""

    plan = berth_plan()
    plan["processes"][0]["occurrences"] = [
        {"description": "a sailing", "at": QUARTER_START, "changes": []}
    ]
    errors = _validate(plan)
    assert "RECURRENCE_DECLARATION_INVALID" in _defects(errors), errors
    assert "hand-written occurrence" in next(
        e for e in errors if e.startswith("RECURRENCE_DECLARATION_INVALID")
    )


def test_recurrence_generation_is_deterministic_and_bounded() -> None:
    """The same declaration always produces the same calendar, and never an unbounded one."""

    first, _ = lower_plan(parse_semantic_plan(berth_plan()))
    second, _ = lower_plan(parse_semantic_plan(copy.deepcopy(berth_plan())))
    assert first == second

    dates = [o["at"] for o in first["world_spec"]["external_processes"][0]["occurrences"]]
    assert dates == [o["at"] for o in second["world_spec"]["external_processes"][0]["occurrences"]]
    assert len(dates) == len(set(dates)) == WEEKLY_FIRINGS

    # A cadence finer than the mechanism is a sampling rate, and it is refused rather
    # than truncated into a calendar that quietly under-runs its own window.
    fine = berth_plan(period="PT1H")
    errors = _validate(fine)
    assert "RECURRENCE_DECLARATION_INVALID" in _defects(errors), errors
    assert str(RECURRENCE_OCCURRENCE_CAP) in next(
        e for e in errors if e.startswith("RECURRENCE_DECLARATION_INVALID")
    )
    assert parse_semantic_plan(fine).processes[0].occurrences == ()

    # A period longer than its own window fires once and means nothing.
    lazy = berth_plan(period="P1Y")
    assert "RECURRENCE_DECLARATION_INVALID" in _defects(_validate(lazy))

    # A cadence with no changes is a date, not a mechanism.
    empty = berth_plan(changes=[])
    assert "RECURRENCE_DECLARATION_INVALID" in _defects(_validate(empty))


def test_a_rate_and_the_cadence_that_applies_it_must_agree() -> None:
    """A weekly rate fired daily would deliver seven weeks of work a week, and this
    compiler will not silently rescale a number the record quoted."""

    errors = _validate(berth_plan(period="P1D"))
    assert "RECURRENCE_DECLARATION_INVALID" in _defects(errors), errors
    assert "weekly sailing capacity" in next(
        e for e in errors if "different period" in e
    )


def test_a_flow_without_a_period_is_not_a_rate() -> None:
    plan = berth_plan()
    next(s for s in plan["states"] if s["name"] == "weekly sailing capacity")["period"] = ""
    errors = _validate(plan)
    assert "FLOW_PERIOD_UNDECLARED" in _defects(errors), errors


# ---------------------------------------------------------------------------
# FD-26 — the claims a reviewer must audit reach the artifact a reviewer reads
# ---------------------------------------------------------------------------


def test_the_zero_actor_justification_survives_lowering_into_the_compilation() -> None:
    """A world with no decider stands on a cited claim, and the claim has to be readable.

    A live actor-free run supplied a zero-actor justification with four cited claims and
    the executable a reviewer opened reported it as null. The claim is now lowered into
    the compilation itself — at the top level, where a reviewer reads, as well as inside
    the representation record — and survives assembly into the persisted world.
    """

    from sworldmodel.research import assemble_bundle

    plan = elevator_plan()
    compilation, _ = lower_plan(parse_semantic_plan(plan))

    claim = compilation["zero_actor_justification"]
    assert claim is not None, "the plan supplied it; the executable must carry it"
    assert claim["evidence_claim_ids"] == ["c-e1", "c-e2"]
    assert claim["no_material_decision"] and claim["process_sufficiency"]
    assert compilation["representation_record"]["zero_actor_justification"] == claim
    assert compilation["single_multiplier_exemption"] is None

    # And it is still there in the compilation the run actually persists.
    bundle = _bundle(plan)
    compilation["reality"] = {
        "as_of": AS_OF.isoformat(),
        "horizon": HORIZON.isoformat(),
        "subject_entity": compilation["subject_entity"],
        "resolution_units": compilation["resolution_units"],
        "target_outcome": compilation["target_outcome"],
    }
    executed = assemble_bundle(bundle.evidence_store, compilation).executed_compilation
    assert executed["zero_actor_justification"] == claim
    assert executed["representation_record"]["zero_actor_justification"] == claim
