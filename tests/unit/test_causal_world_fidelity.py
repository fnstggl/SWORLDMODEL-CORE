"""Causal-world fidelity (Phase 3), proven on domains this system has never seen.

Every world here is invented — an aquifer recharge district, a bay ferry terminal — and
none of them resembles an acceptance question, because the point of these gates is that
they are about *shape*, not subject. The failure they exist to end was a real one: a live
run reduced a production question to one cited quarter multiplied by one invented factor,
with no actor, no intermediate state and no time, and published the count of its own
branches as a probability. Nothing in the fix names that domain, that quantity or that
threshold — every number below is derived from the plan being judged, and the same rules
fire on a harbor, a canal or a basin.

What is proven here, one requirement at a time:

* CWF-4 / FD-3 — ungrounded alternatives either side of the threshold are refused, and
  the break-even is computed from the plan's own arithmetic rather than assumed.
* CWF-3 / FD-1 — a terminal quantity produced in one non-agent step is refused; the
  documented single-multiplier exemption is real, recorded, and cannot license invented
  numbers.
* CWF-5 / FD-9 — a world with no deciding entity needs a cited justification; a
  decorative actor is refused; a necessary actor cannot be dropped.
* CWF-1 — the representation-scale record is an artifact section of the compilation.
* FD-10 / FD-11 — an alternative carries meaning or it carries nothing.
* CWF-6 — the pre-rollout reviewer decides the one-step attacks for itself.
* CWF-8 — a genuine operational process runs, through the real engine, with and without
  actors, and ungrounded weights never become a calibrated point.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from sworldmodel.engine import run
from sworldmodel.models import ResolutionContract
from sworldmodel.semantic_lowering import lower_plan
from sworldmodel.semantic_plan import parse_semantic_plan, validate_semantic_plan
from sworldmodel.uncertainty import UNGROUNDED_PROVENANCES, weights_grounded
from sworldmodel.world_compiler import compile_world
from sworldmodel.world_review import mechanical_world_checks

AS_OF = datetime.fromisoformat("2026-01-10T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-03-01T23:59:59+00:00")

# The evidence both invented domains draw on. A plan is checked against exactly the
# claims it cites, so the two variants of a world can honestly differ in what the record
# contains — which is the whole of the zero-actor question.
CLAIMS: dict[str, dict[str, Any]] = {
    "c-a1": {
        "proposition": "The Cald Basin Recharge District has recorded 240 acre-feet of "
        "managed recharge so far this season",
        "value": "240 acre-feet",
        "entities": ["Cald Basin Recharge District"],
    },
    "c-a2": {
        "proposition": "The Cald Basin Recharge District's diversion canals can take 800 "
        "acre-feet during a full storm cycle",
        "value": "800 acre-feet per cycle",
        "entities": ["Cald Basin Recharge District"],
    },
    "c-a3": {
        "proposition": "Cald Basin winter runoff fractions in the recorded series range "
        "from 0.6 to 0.9",
        "value": "0.6 to 0.9",
        "entities": ["Cald Basin Recharge District"],
    },
    "c-a4": {
        "proposition": "The Cald Basin Recharge District water engineer may release a 200 "
        "acre-foot emergency storage allocation into the recharge basins",
        "value": "200 acre-feet, engineer's authority",
        "entities": ["Cald Basin Recharge District", "District water engineer"],
    },
    "c-f1": {
        "proposition": "The Kestrel Bay ferry terminal recorded 320000 crossings last season",
        "value": "320000 crossings",
        "entities": ["Kestrel Bay Ferry Terminal"],
    },
    "c-f2": {
        "proposition": "Kestrel Bay's published operating model scales last season's "
        "crossings by a seasonal factor between 1.3 and 1.4",
        "value": "1.3 to 1.4",
        "entities": ["Kestrel Bay Ferry Terminal"],
    },
    # An actor needs a claim that names it. c-f1 records the terminal's crossings and
    # names no person, so a world that puts a manager on the terminal quantity while
    # citing c-f1 is citing the thing the manager is supposed to be deciding about —
    # which is how a fabricated participant used to attest itself. This claim gives the
    # manager the same standing c-a4 gives the district's water engineer.
    "c-f3": {
        "proposition": "Kestrel Bay's terminal operations manager sets the published "
        "sailing schedule and may add or withdraw sailings within the season",
        "value": "schedule authority, terminal operations manager",
        "entities": ["Kestrel Bay Ferry Terminal", "Terminal operations manager"],
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
    """The named defects a validation produced, ignoring their prose."""

    return {e.split(":", 1)[0] for e in errors}


# ---------------------------------------------------------------------------
# Domain 1 — a managed aquifer recharge district (a real operating process)
# ---------------------------------------------------------------------------


def recharge_plan(*, with_actor: bool = False, cited_driver: bool = True) -> dict[str, Any]:
    """Storm runoff is diverted in January and infiltrates in February.

    A genuine operational world: a cited starting volume, an intermediate state the
    mechanism itself produces, two dated stages inside the window, and the uncertainty on
    the driver rather than on the total. With ``with_actor`` the district's engineer can
    release an emergency allocation — a decision that really moves the recorded volume —
    and the world is the same world otherwise.
    """

    driver_ids = ["c-a3"] if cited_driver else []
    plan: dict[str, Any] = {
        "resolution": {
            "question": "Will the Cald Basin district record more than 900 acre-feet of "
            "managed recharge before March 1?",
            "yes_condition": "Recorded managed recharge exceeds 900 acre-feet.",
            "subject_entity": "Cald Basin Recharge District",
            "resolution_units": "acre-feet of managed recharge",
            "target_outcome": "more than 900 acre-feet recorded",
            "expected_participants": 1 if with_actor else None,
            "evidence_claim_ids": ["c-a1"],
        },
        "entities": [
            {
                "name": "Cald Basin Recharge District",
                "structural_type": "institution",
                "role": "operates the diversion canals and the recharge basins",
                "representation_scale": "organization",
                "decides": False,
                "authority": "operates the canals and keeps the recharge log",
                "why_material": "its canals and basins produce every acre-foot recorded",
                "terminal_state_it_can_change": "raises recorded recharge volume through "
                "the diversion and infiltration cycles",
                "information_received": "storm gauge readings from the basin network",
                "if_removed": "nothing is diverted, nothing infiltrates and no volume is "
                "recorded at all",
                "evidence_claim_ids": ["c-a1"],
            }
        ],
        "excluded_candidates": [
            {
                "name": "downstream orchard cooperative",
                "why_immaterial": "it draws water only after recharge has been logged, so "
                "removing it cannot add to or subtract from the recorded volume",
                "evidence_claim_ids": ["c-a1"],
            }
        ],
        "states": [
            {
                "name": "recorded recharge volume",
                "owner": "world",
                "state_type": "quantity",
                "unit": "acre-feet",
                "initial": 240,
                "why_material": "the terminal reads it",
                "not_a_stock_because": "the recharge log is the season's cumulative record "
                "of water already infiltrated, not water held somewhere: nothing draws it "
                "back down and it fills no container",
                "evidence_claim_ids": ["c-a1"],
            },
            {
                "name": "diverted storm runoff",
                "owner": "world",
                "state_type": "quantity",
                "unit": "acre-feet",
                "initial": "UNKNOWN",
                "why_material": "the volume the January storm cycle puts into the basins",
                "evidence_claim_ids": [],
            },
            {
                "name": "winter runoff fraction",
                "owner": "world",
                "state_type": "quantity",
                "unit": "fraction",
                "initial": "UNKNOWN",
                "why_material": "how much of the canals' capacity the storms actually fill",
                "evidence_claim_ids": [],
            },
        ],
        "events": [],
        "affordances": [],
        "processes": [
            {
                "name": "basin diversion and infiltration",
                "meaning": "storm runoff is diverted into the basins and infiltrates",
                "kind": "operational",
                "inputs": ["winter runoff fraction"],
                "occurrences": [
                    {
                        "description": "the January storm cycle diverts what the canals take",
                        "at": "2026-01-25T06:00:00+00:00",
                        "changes": [
                            {
                                "op": "set",
                                "target": "diverted storm runoff",
                                "value": {
                                    "kind": "product",
                                    "parts": [
                                        {"kind": "literal", "value": 800},
                                        {"kind": "state", "state": "winter runoff fraction"},
                                    ],
                                },
                            }
                        ],
                    },
                    {
                        "description": "February infiltration writes the diverted volume "
                        "into the recharge log",
                        "at": "2026-02-20T06:00:00+00:00",
                        "changes": [
                            {
                                "op": "increase",
                                "target": "recorded recharge volume",
                                "amount": {"kind": "state", "state": "diverted storm runoff"},
                            }
                        ],
                    },
                ],
                "evidence_claim_ids": ["c-a2"],
            }
        ],
        "uncertainties": [
            {
                "name": "winter runoff",
                "what_unknown": "the fraction of canal capacity the winter storms fill",
                "why_unknown": "the storms have not happened yet",
                "affects_state": "winter runoff fraction",
                "release_at": None,
                "alternatives": [
                    {
                        "value": 0.9,
                        "weight": None,
                        "provenance": "symmetric_ignorance_assumption",
                        "grounding": "the wet end of the cited recorded series",
                        "meaning": "the storms fill the canals to the wet end of the record",
                        "why_unresolved": "the season has not run and nothing in the record "
                        "picks a point in the range",
                        "changes": ["process_state"],
                        "terminal_sensitivity": "decides_the_terminal",
                        "evidence_claim_ids": driver_ids,
                    },
                    {
                        "value": 0.6,
                        "weight": None,
                        "provenance": "symmetric_ignorance_assumption",
                        "grounding": "the dry end of the cited recorded series",
                        "meaning": "the storms fill the canals to the dry end of the record",
                        "why_unresolved": "the season has not run and nothing in the record "
                        "picks a point in the range",
                        "changes": ["process_state"],
                        "terminal_sensitivity": "decides_the_terminal",
                        "evidence_claim_ids": driver_ids,
                    },
                ],
            }
        ],
        "terminal": {
            "form": "quantity_comparison",
            "state": "recorded recharge volume",
            "comparison": "greater_than",
            "threshold": 900,
        },
        "terminal_producer_note": "The log starts at the cited 240 acre-feet, the January "
        "cycle produces the diverted volume, and February's infiltration adds it; the "
        "uncertainty drives the storm fraction, never the recorded total.",
        "world_facts": [],
    }
    if not with_actor:
        plan["zero_actor_justification"] = {
            "no_material_decision": "the diversion schedule follows the district's standing "
            "storm rule and the record names no person or body able to add or withhold "
            "recharge inside the window",
            "process_sufficiency": "the storm runoff fraction and the fixed canal capacity "
            "together determine every acre-foot the log records",
            "evidence_claim_ids": ["c-a1", "c-a2"],
        }
        return plan

    plan["entities"].append(
        {
            "name": "District water engineer",
            "structural_type": "person",
            "role": "decides whether the emergency storage allocation is released",
            "representation_scale": "individual",
            "decides": True,
            "authority": "may release the 200 acre-foot emergency storage allocation",
            "why_material": "the emergency release adds recorded acre-feet the storm cycle "
            "cannot produce on its own",
            "terminal_state_it_can_change": "increases recorded recharge volume by the "
            "released allocation",
            "information_received": "the storm gauge readings and basin levels presented at "
            "the district operations meeting",
            "if_removed": "the allocation is never released and a dry season can no longer "
            "reach the threshold",
            "evidence_claim_ids": ["c-a4"],
        }
    )
    plan["affordances"].extend(
        [
            {
                "name": "release emergency storage",
                "meaning": "release the emergency storage allocation into the basins",
                "actor": "District water engineer",
                "authority_required": "emergency storage authority",
                "visibility": "public",
                "changes": [
                    {
                        "op": "increase",
                        "target": "recorded recharge volume",
                        "amount": {"kind": "literal", "value": 200},
                    }
                ],
                "evidence_claim_ids": ["c-a4"],
            },
            {
                "name": "hold emergency storage",
                "meaning": "keep the emergency storage allocation in reserve and tell the "
                "district so",
                "actor": "District water engineer",
                "authority_required": "emergency storage authority",
                "visibility": "public",
                "changes": [
                    {
                        "op": "send",
                        "target": "allocation notice",
                        "recipients": ["Cald Basin Recharge District"],
                        "detail": "the emergency allocation stays in reserve",
                    }
                ],
                "evidence_claim_ids": ["c-a4"],
            },
        ]
    )
    plan["processes"].append(
        {
            "name": "district operations meeting",
            "meaning": "the dated meeting at which the allocation is decided",
            "kind": "actor_moment",
            "participants": ["District water engineer"],
            "allowed_affordances": ["release emergency storage", "hold emergency storage"],
            "at": "2026-02-05T09:00:00+00:00",
            "occurrences": [],
            "evidence_claim_ids": ["c-a4"],
        }
    )
    return plan


# ---------------------------------------------------------------------------
# Domain 2 — a ferry terminal's quarterly tally (a report wearing a world's clothes)
# ---------------------------------------------------------------------------


def ferry_plan(
    *,
    factors: tuple[float, float] = (1.1, 1.4),
    cited_factors: bool = False,
    exemption: bool = False,
    base: int = 320000,
    threshold: int = 400000,
) -> dict[str, Any]:
    """Last season's crossings times one factor, and that is the whole world.

    One non-agent change, no intermediate state, one moment: the quantity is announced
    rather than produced. Parameterised so the same shape can be shown with invented
    factors, with cited ones, with the documented exemption claimed, and at any scale.
    """

    factor_ids = ["c-f2"] if cited_factors else []
    plan: dict[str, Any] = {
        "resolution": {
            "question": f"Will the Kestrel Bay ferry terminal record more than {threshold} "
            "crossings this quarter?",
            "yes_condition": f"Recorded crossings exceed {threshold} at quarter end.",
            "subject_entity": "Kestrel Bay Ferry Terminal",
            "resolution_units": "recorded crossings",
            "target_outcome": f"more than {threshold} crossings recorded",
            "expected_participants": None,
            "evidence_claim_ids": ["c-f1"],
        },
        "entities": [
            {
                "name": "Kestrel Bay Ferry Terminal",
                "structural_type": "institution",
                "role": "runs the crossings and publishes the tally",
                "representation_scale": "organization",
                "decides": False,
                "authority": "operates the sailings and keeps the crossing tally",
                "why_material": "its sailings are what the tally counts",
                "terminal_state_it_can_change": "its sailings produce the recorded crossings",
                "information_received": "its own boarding counts",
                "if_removed": "there are no crossings to count",
                "evidence_claim_ids": ["c-f1"],
            }
        ],
        "excluded_candidates": [
            {
                "name": "bay harbour authority",
                "why_immaterial": "it licenses the route but the licence is already issued "
                "for the whole quarter",
                "evidence_claim_ids": ["c-f1"],
            }
        ],
        "zero_actor_justification": {
            "no_material_decision": "the sailing timetable is fixed for the quarter and the "
            "record names no decision that could change it inside the window",
            "process_sufficiency": "the timetable and the seasonal demand factor determine "
            "the tally",
            "evidence_claim_ids": ["c-f1"],
        },
        "states": [
            {
                "name": "last season crossings",
                "owner": "world",
                "state_type": "quantity",
                "unit": "crossings",
                "initial": base,
                "why_material": "the cited base the tally is scaled from",
                "evidence_claim_ids": ["c-f1"],
            },
            {
                "name": "recorded crossings this quarter",
                "owner": "world",
                "state_type": "quantity",
                "unit": "crossings",
                "initial": "UNKNOWN",
                "why_material": "the terminal reads it",
                "not_a_stock_because": "the quarter's tally counts crossings that have "
                "already been made; it is a record rather than anything held at the "
                "terminal, and nothing can take a crossing back out of it",
                "evidence_claim_ids": [],
            },
            {
                "name": "seasonal crossing factor",
                "owner": "world",
                "state_type": "quantity",
                "unit": "ratio",
                "initial": "UNKNOWN",
                "why_material": "scales last season's crossings into this quarter's",
                "evidence_claim_ids": [],
            },
        ],
        "events": [],
        "affordances": [],
        "processes": [
            {
                "name": "quarterly crossing tally",
                "meaning": "the quarter's crossings are totalled at quarter end",
                "kind": "operational",
                "inputs": ["seasonal crossing factor"],
                "occurrences": [
                    {
                        "description": "the tally is computed",
                        "at": "2026-02-25T00:00:00+00:00",
                        "changes": [
                            {
                                "op": "set",
                                "target": "recorded crossings this quarter",
                                "value": {
                                    "kind": "product",
                                    "parts": [
                                        {"kind": "state", "state": "last season crossings"},
                                        {"kind": "state", "state": "seasonal crossing factor"},
                                    ],
                                },
                            }
                        ],
                    }
                ],
                "evidence_claim_ids": ["c-f1"],
            }
        ],
        "uncertainties": [
            {
                "name": "seasonal demand",
                "what_unknown": "how this quarter's demand compares with last season",
                "why_unknown": "the quarter has not finished",
                "affects_state": "seasonal crossing factor",
                "release_at": None,
                "alternatives": [
                    {
                        "value": value,
                        "weight": None,
                        "provenance": "symmetric_ignorance_assumption",
                        "grounding": f"demand runs at {value} of last season",
                        "meaning": f"this quarter's demand is {value} times last season's",
                        "why_unresolved": "the quarter is still running",
                        "changes": ["process_state"],
                        "terminal_sensitivity": "moves_the_terminal",
                        "evidence_claim_ids": factor_ids,
                    }
                    for value in factors
                ],
            }
        ],
        "terminal": {
            "form": "quantity_comparison",
            "state": "recorded crossings this quarter",
            "comparison": "greater_than",
            "threshold": threshold,
        },
        "terminal_producer_note": "The tally is computed from the cited last-season total "
        "scaled by the seasonal factor.",
        "world_facts": [],
    }
    if exemption:
        plan["single_multiplier_exemption"] = {
            "empirical_model": "the terminal's published operating model expresses a "
            "quarter's crossings as last season's total times a seasonal factor",
            "parameter_uncertainty": "the factor's range is the model's own published band",
            "evidence_claim_ids": ["c-f2"],
        }
    return plan


# ---------------------------------------------------------------------------
# CWF-4 / FD-3 — threshold straddling
# ---------------------------------------------------------------------------


def test_ungrounded_alternatives_either_side_of_the_threshold_are_refused() -> None:
    """The exact shape that published a probability made of two invented numbers.

    Neither factor is in the record and neither weight is supported, and the plan's own
    arithmetic puts one either side of the resolution threshold — so the answer is the
    choice of the two numbers. It is refused before anything is simulated.
    """

    errors = _validate(ferry_plan())
    straddle = [e for e in errors if e.startswith("THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS")]
    assert straddle, errors
    message = straddle[0]
    assert "seasonal demand" in message
    assert "1.1" in message and "1.4" in message
    # The correction boundary is named, and it is the alternatives — never the threshold.
    assert "Correction boundary: the alternatives' VALUES" in message
    assert "Do not move the threshold" in message


def test_the_break_even_is_derived_from_the_plans_own_arithmetic() -> None:
    """Not assumed, not hardcoded: bisected out of the world the plan describes.

    320000 x factor against 400000 breaks even at 1.25, and the refusal says so — which
    is how a planner learns that its two numbers were placed either side of a boundary it
    never wrote down. Change the base and the threshold and the same code finds the new
    boundary, which is the proof that no number lives in the gate.
    """

    (message,) = [
        e
        for e in _validate(ferry_plan())
        if e.startswith("THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS")
    ]
    assert "break-even draw ≈ 1.25" in message

    # A different domain scale entirely: base x7, threshold x5, factors either side of the
    # new break-even of 400000*5 / 320000*7 = 0.892857...
    scaled = ferry_plan(factors=(0.8, 1.05), base=320000 * 7, threshold=400000 * 5)
    (scaled_message,) = [
        e for e in _validate(scaled) if e.startswith("THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS")
    ]
    assert "break-even draw ≈ 0.892857" in scaled_message


def test_cited_alternatives_may_straddle_the_threshold_and_still_compile() -> None:
    """The honest shape of a real forecast is not the defect.

    The recharge world's driver decides the answer — 0.9 crosses, 0.6 does not — and that
    is exactly what makes the question open. What the gate refuses is *invented* values
    doing it. With the range taken from the record the world stands, and the weights stay
    labelled as the ignorance they are.
    """

    assert _validate(recharge_plan()) == []

    # The same world with the citation removed from the alternatives is refused — by the
    # straddling gate on the arithmetic, and by the filler gate on the plan's own
    # declaration that these uncited values decide the answer.
    errors = _validate(recharge_plan(cited_driver=False))
    assert "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS" in _defects(errors), errors
    assert _defects(errors) <= {
        "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS",
        "DEGENERATE_FILLER_ALTERNATIVE",
    }, errors


def test_ungrounded_equal_branches_cannot_become_a_calibrated_point() -> None:
    """Grounded values, unsupported split: bounds, never a calibrated number.

    Even when the plan is admissible, a 1/n split over alternatives the record does not
    rank must never be presented as a probability. The compiled branches carry the
    ungrounded provenance that keeps every downstream consumer honest.
    """

    compiled = _compile(recharge_plan())
    scenarios = compiled.scenario_set.scenarios
    assert len(scenarios) == 2
    assert {round(s.weight, 6) for s in scenarios} == {0.5}
    assert all(not weights_grounded(s) for s in scenarios)
    assert all(s.provenance in UNGROUNDED_PROVENANCES for s in scenarios)


# ---------------------------------------------------------------------------
# CWF-3 / FD-1 — one-step operational worlds
# ---------------------------------------------------------------------------


def test_one_step_reporting_cannot_masquerade_as_production() -> None:
    """One change, no intermediate state, one moment — and it is refused as reporting."""

    errors = _validate(ferry_plan(factors=(1.3, 1.4), cited_factors=True))
    assert _defects(errors) == {"ONE_STEP_OPERATIONAL_WORLD"}, errors
    (message,) = errors
    assert "recorded crossings this quarter" in message
    assert "quarterly crossing tally" in message
    assert "intermediate state" in message
    assert "single_multiplier_exemption" in message


def test_a_multi_stage_process_with_a_produced_intermediate_is_not_one_step() -> None:
    """The distinction is causal, not cosmetic: the recharge world's final change reads a
    volume the world itself produced one stage earlier, so it is a process, not a
    report."""

    assert _validate(recharge_plan()) == []
    assert _validate(recharge_plan(with_actor=True)) == []


def test_a_single_grounded_accumulation_is_thin_but_not_refused() -> None:
    """The gate is exactly as wide as the defect, and no wider.

    One scheduled transfer of a cited size is a thin world, not a false one: nothing is
    invented, no branch weight decides anything, and demanding an operating process would
    only extract detail the record does not contain. What is refused is the announced
    total and the uncertain multiplier — both of which put the answer somewhere other
    than the world.
    """

    plan = ferry_plan()
    plan["uncertainties"] = []
    plan["states"] = [s for s in plan["states"] if s["name"] != "seasonal crossing factor"]
    plan["processes"][0]["inputs"] = []
    plan["processes"][0]["occurrences"][0]["changes"] = [
        {
            "op": "increase",
            "target": "recorded crossings this quarter",
            "amount": {"kind": "state", "state": "last season crossings"},
        }
    ]
    plan["states"] = [
        dict(state, initial=0, evidence_claim_ids=["c-f1"])
        if state["name"] == "recorded crossings this quarter"
        else state
        for state in plan["states"]
    ]
    assert _validate(plan) == []

    # Turn the same single change back into an announced total and it is refused again.
    plan["processes"][0]["occurrences"][0]["changes"] = [
        {
            "op": "set",
            "target": "recorded crossings this quarter",
            "value": {"kind": "state", "state": "last season crossings"},
        }
    ]
    assert _defects(_validate(plan)) == {"ONE_STEP_OPERATIONAL_WORLD"}


def test_the_single_multiplier_exemption_must_be_claimed_and_cannot_launder_invention() -> None:
    """D4's escape hatch is real, recorded, and closed to invented parameters.

    With a documented, cited empirical model and a parameter range from that model, one
    multiplier may stand for the system — and the claim is visible in the plan, so a
    reviewer can see it was taken. The same exemption over uncited alternatives is
    refused: a documented model does not license numbers nobody published.
    """

    admitted = ferry_plan(factors=(1.3, 1.4), cited_factors=True, exemption=True)
    assert _validate(admitted) == []

    invented = ferry_plan(factors=(1.3, 1.4), cited_factors=False, exemption=True)
    errors = _validate(invented)
    assert _defects(errors) == {"SINGLE_DRIVER_EXEMPTION_UNGROUNDED"}, errors
    assert "seasonal demand" in errors[0]


# ---------------------------------------------------------------------------
# CWF-5 / FD-9 — who decides, and whether anyone has to
# ---------------------------------------------------------------------------


def test_a_world_with_no_deciding_entity_must_say_why_with_evidence() -> None:
    plan = recharge_plan()
    del plan["zero_actor_justification"]
    errors = _validate(plan)
    assert _defects(errors) == {"ZERO_ACTOR_WORLD_UNJUSTIFIED"}, errors
    assert "no material human or population decision" not in errors[0]  # it must be specific
    assert "zero_actor_justification" in errors[0]

    # A justification with no citation is an assertion, not evidence.
    plan["zero_actor_justification"] = {
        "no_material_decision": "nobody can change the weather",
        "process_sufficiency": "the storms decide it",
        "evidence_claim_ids": [],
    }
    assert _defects(_validate(plan)) == {"ZERO_ACTOR_WORLD_UNJUSTIFIED"}


def test_a_decorative_actor_is_refused() -> None:
    """An actor whose decisions cannot move anything the terminal reads is scenery.

    The press office really exists and really acts — it publishes a bulletin — but no
    bulletin has ever added an acre-foot. A world that includes it looks more alive and
    is not.
    """

    plan = recharge_plan(with_actor=True)
    plan["entities"].append(
        {
            "name": "Basin press office",
            "structural_type": "organization",
            "role": "publishes the district's recharge bulletin",
            "representation_scale": "organization",
            "decides": True,
            "authority": "publishes bulletins",
            "why_material": "it reports the recharge programme",
            "terminal_state_it_can_change": "none — it reports what the basins already did",
            "information_received": "the district's own log",
            "if_removed": "the bulletin is not published",
            "evidence_claim_ids": ["c-a1"],
        }
    )
    plan["states"].append(
        {
            "name": "published bulletins",
            "owner": "world",
            "state_type": "quantity",
            "unit": "bulletins",
            "initial": 0,
            "why_material": "counts the bulletins issued",
            "evidence_claim_ids": ["c-a1"],
        }
    )
    plan["affordances"].append(
        {
            "name": "publish the recharge bulletin",
            "meaning": "publish the district's recharge bulletin",
            "actor": "Basin press office",
            "visibility": "public",
            "changes": [
                {
                    "op": "increase",
                    "target": "published bulletins",
                    "amount": {"kind": "literal", "value": 1},
                }
            ],
            "evidence_claim_ids": ["c-a1"],
        }
    )
    plan["processes"][-1]["participants"].append("Basin press office")
    plan["processes"][-1]["allowed_affordances"].append("publish the recharge bulletin")

    errors = _validate(plan)
    assert _defects(errors) == {"DECORATIVE_ACTOR"}, errors
    assert "Basin press office" in errors[0]
    assert "excluded_candidates" in errors[0]

    # An actor who cannot move the terminal himself but informs one who can is NOT
    # decoration: influence through another decider is influence.
    plan["affordances"][-1]["changes"].append(
        {
            "op": "send",
            "target": "basin level briefing",
            "recipients": ["District water engineer"],
            "detail": "the basins are below the emergency trigger",
        }
    )
    assert _validate(plan) == []


def test_a_necessary_actor_cannot_be_omitted() -> None:
    """Delete the only decision-maker and the world must refuse, not quietly run.

    The record names an engineer whose release can decide a dry season. A plan that drops
    him keeps the question's participant count and then has nobody to meet it, and it
    cannot claim that no decision matters when its own resolution says one does.
    """

    plan = recharge_plan(with_actor=True)
    plan["entities"] = [e for e in plan["entities"] if e["name"] != "District water engineer"]
    plan["affordances"] = []
    plan["events"] = []
    plan["processes"] = [p for p in plan["processes"] if p["kind"] != "actor_moment"]

    errors = _validate(plan)
    assert "ZERO_ACTOR_WORLD_UNJUSTIFIED" in _defects(errors), errors
    assert "ZERO_ACTOR_CLAIM_CONTRADICTED" in _defects(errors), errors
    assert any("decision-relevant participants" in e for e in errors), errors

    # And the escape hatch is closed: a zero-actor justification cannot be pasted over a
    # question that declares the decisions it needs.
    plan["zero_actor_justification"] = {
        "no_material_decision": "nobody decides anything here",
        "process_sufficiency": "the storms do it all",
        "evidence_claim_ids": ["c-a1"],
    }
    assert "ZERO_ACTOR_CLAIM_CONTRADICTED" in _defects(_validate(plan))


def test_an_empty_justification_block_is_not_a_claim() -> None:
    """The schema shows both conditional blocks to every planner, so most plans echo them
    back empty. An empty block is silence, not a contradiction — otherwise every world
    with an actor in it would be refused for a claim nobody made."""

    plan = recharge_plan(with_actor=True)
    plan["zero_actor_justification"] = {
        "no_material_decision": "",
        "process_sufficiency": "",
        "evidence_claim_ids": [],
    }
    plan["single_multiplier_exemption"] = {
        "empirical_model": "",
        "parameter_uncertainty": "",
        "evidence_claim_ids": [],
    }
    assert _validate(plan) == []

    # And the actor-free twin still has to make its claim for real.
    bare = recharge_plan()
    bare["zero_actor_justification"] = {"no_material_decision": "", "process_sufficiency": ""}
    assert _defects(_validate(bare)) == {"ZERO_ACTOR_WORLD_UNJUSTIFIED"}


def test_a_world_that_has_actors_cannot_also_claim_none_are_material() -> None:
    plan = recharge_plan(with_actor=True)
    plan["zero_actor_justification"] = {
        "no_material_decision": "nothing anyone decides matters",
        "process_sufficiency": "the storms decide it",
        "evidence_claim_ids": ["c-a1"],
    }
    errors = _validate(plan)
    assert _defects(errors) == {"ZERO_ACTOR_CLAIM_CONTRADICTED"}, errors
    assert "District water engineer" in errors[0]


# ---------------------------------------------------------------------------
# CWF-1 — the representation-scale record
# ---------------------------------------------------------------------------


def test_an_entity_that_cannot_answer_the_representation_questions_is_refused() -> None:
    plan = recharge_plan()
    del plan["entities"][0]["if_removed"]
    del plan["entities"][0]["information_received"]
    errors = _validate(plan)
    assert _defects(errors) == {"REPRESENTATION_RECORD_INCOMPLETE"}, errors
    assert "if_removed" in errors[0] and "information_received" in errors[0]

    # The exclusion half is enforced too: a candidate named without a reason is an
    # omission nobody justified.
    plan = recharge_plan()
    plan["excluded_candidates"][0]["why_immaterial"] = ""
    assert _defects(_validate(plan)) == {"REPRESENTATION_RECORD_INCOMPLETE"}


def test_the_representation_record_is_an_artifact_section_of_the_compilation() -> None:
    """It lands in the compiled world, not in a paragraph nobody keeps."""

    compilation, _ = lower_plan(parse_semantic_plan(recharge_plan(with_actor=True)))
    record = compilation["representation_record"]

    included = {row["entity"]: row for row in record["included"]}
    assert set(included) == {"Cald Basin Recharge District", "District water engineer"}
    engineer = included["District water engineer"]
    assert engineer["decides"] is True
    assert engineer["runtime_id"] == "district_water_engineer"
    assert engineer["affordances"] == ["hold emergency storage", "release emergency storage"]
    for key in (
        "why_it_can_change_the_answer",
        "terminal_relevant_state_it_can_alter",
        "information_it_receives",
        "authority_it_holds",
        "if_removed",
    ):
        assert engineer[key], key

    (excluded,) = record["excluded"]
    assert excluded["candidate"] == "downstream orchard cooperative"
    assert excluded["why_removal_cannot_change_the_answer"]

    # What "terminal-relevant" means is recorded too, so the decorative-actor judgement
    # can be checked rather than taken on trust.
    assert set(record["terminal_relevant_states"]) == {
        "recorded recharge volume",
        "diverted storm runoff",
        "winter runoff fraction",
    }
    assert record["deciding_entities"] == ["District water engineer"]
    assert record["zero_actor_justification"] is None

    # The actor-free twin records the claim it is standing on.
    plain, _ = lower_plan(parse_semantic_plan(recharge_plan()))
    claim = plain["representation_record"]["zero_actor_justification"]
    assert claim["evidence_claim_ids"] == ["c-a1", "c-a2"]
    assert claim["no_material_decision"] and claim["process_sufficiency"]

    # A claimed exemption is visible in the same place.
    ferry, _ = lower_plan(
        parse_semantic_plan(ferry_plan(factors=(1.3, 1.4), cited_factors=True, exemption=True))
    )
    assert ferry["representation_record"]["single_multiplier_exemption"]["empirical_model"]


# ---------------------------------------------------------------------------
# FD-10 / FD-11 — alternatives carry meaning or they carry nothing
# ---------------------------------------------------------------------------


def test_an_alternative_that_cannot_say_what_it_means_is_refused() -> None:
    plan = recharge_plan()
    alt = plan["uncertainties"][0]["alternatives"][1]
    alt["meaning"] = ""
    alt["changes"] = []
    alt["terminal_sensitivity"] = ""
    errors = _validate(plan)
    assert "UNCERTAINTY_ALTERNATIVE_UNDESCRIBED" in _defects(errors), errors
    message = next(e for e in errors if e.startswith("UNCERTAINTY_ALTERNATIVE_UNDESCRIBED"))
    assert "meaning" in message and "changes" in message and "terminal_sensitivity" in message


def test_a_filler_alternative_may_not_be_the_one_that_decides() -> None:
    """The residual-label defect, in an invented domain.

    A live plan's decisive alternative was the label "other", whose own grounding said the
    evidence named no such alternative — and it carried half the mass and all of the YES
    mass. Here the same shape is a bare "some other regime": nothing cites it, nothing
    supports its weight, and the plan itself says it decides the answer.
    """

    plan = recharge_plan()
    plan["uncertainties"][0]["alternatives"][1] = {
        "value": "some other regime",
        "weight": None,
        "provenance": "symmetric_ignorance_assumption",
        "grounding": "no specific alternative in evidence",
        "meaning": "something else happens",
        "why_unresolved": "unknown",
        "changes": ["process_state"],
        "terminal_sensitivity": "decides_the_terminal",
        "evidence_claim_ids": [],
    }
    errors = _validate(plan)
    assert "DEGENERATE_FILLER_ALTERNATIVE" in _defects(errors), errors
    message = next(e for e in errors if e.startswith("DEGENERATE_FILLER_ALTERNATIVE"))
    assert "some other regime" in message


def test_a_sensitivity_declaration_is_checked_against_the_plans_own_arithmetic() -> None:
    """A planner cannot declare away a flip its own numbers produce."""

    plan = recharge_plan()
    for alt in plan["uncertainties"][0]["alternatives"]:
        alt["terminal_sensitivity"] = "immaterial_to_the_terminal"
    errors = _validate(plan)
    assert _defects(errors) == {"TERMINAL_SENSITIVITY_MISDECLARED"}, errors
    assert "recorded recharge volume" in errors[0]


# ---------------------------------------------------------------------------
# CWF-6 — the pre-rollout reviewer decides these for itself
# ---------------------------------------------------------------------------


def test_the_reviewer_rejects_a_one_step_world_without_asking_anyone() -> None:
    """The same attacks, computed from the executable — so they hold for both compiler
    modes and need no provider to decide them.

    That they also STOP a run when the reviewer is unreachable is the sibling test
    below (FD-34); this one proves only that they are computed.
    """

    findings = {f.key: f for f in mechanical_world_checks(_compile(ferry_plan()))}
    assert findings["terminal_set_in_one_step"].severity == "CRITICAL"
    assert findings["single_uncertain_multiplier_decides"].is_blocking
    assert findings["multiplier_lacks_evidence"].severity == "CRITICAL"
    assert findings["no_intermediate_production_state"].severity == "HIGH"
    assert findings["world_skips_the_causal_period"].severity == "HIGH"
    for finding in findings.values():
        assert finding.evidence_basis.startswith("computed from the compiled world")


def test_a_one_step_world_is_refused_even_when_the_reviewer_is_unreachable() -> None:
    """FD-34. These attacks survive an unreachable reviewer into the GATE, not merely
    into the record.

    A mechanical finding is a fact about the compiled world with no provider involved,
    so a provider outage must not launder it into an advisory note and let the run
    publish. The record built here is exactly what ``review_world`` returns when the
    model call fails: the mechanical findings, and an error saying no opinion was had.
    """

    from sworldmodel.api import ReviewRound, WorldReviewRecord
    from sworldmodel.world_review import _from_findings

    mechanical = mechanical_world_checks(_compile(ferry_plan()))
    review = _from_findings(mechanical).with_error("the review could not run: GatewayError")
    record = WorldReviewRecord((ReviewRound(0, "ferry", review),))

    assert record.model_opinion_obtained is False
    assert {f.key for f in record.surviving_mechanical_blocking} >= {
        "terminal_set_in_one_step",
        "multiplier_lacks_evidence",
    }
    assert record.blocks_publication is True
    assert record.causal_simulation_valid is False


def test_the_reviewer_passes_a_world_that_actually_operates() -> None:
    findings = {f.key: f for f in mechanical_world_checks(_compile(recharge_plan()))}
    assert [k for k, f in findings.items() if f.is_blocking] == []
    assert (
        "diverted storm runoff".replace(" ", "_")
        in findings["no_intermediate_production_state"].finding
    )
    assert findings["world_skips_the_causal_period"].severity == "PASS"


def test_a_draw_scaled_straight_into_the_total_is_seen_even_in_a_staged_world() -> None:
    """More stages do not hide a factor that decides the answer by itself.

    Here the world really does run in two stages, so it is not one-step — but the final
    stage multiplies the branch draw straight into the total, and the reviewer says so.
    Cited, it is a concern rather than a blocker; the observation is the same either way.
    """

    plan = recharge_plan()
    plan["processes"][0]["occurrences"][0]["changes"] = [
        {
            "op": "increase",
            "target": "recorded recharge volume",
            "amount": {"kind": "literal", "value": 100},
        }
    ]
    plan["processes"][0]["occurrences"][1]["changes"] = [
        {
            "op": "increase",
            "target": "recorded recharge volume",
            "amount": {
                "kind": "product",
                "parts": [
                    {"kind": "literal", "value": 800},
                    {"kind": "state", "state": "winter runoff fraction"},
                ],
            },
        }
    ]
    plan["states"] = [s for s in plan["states"] if s["name"] != "diverted storm runoff"]

    findings = {f.key: f for f in mechanical_world_checks(_compile(plan))}
    assert findings["terminal_set_in_one_step"].severity == "PASS"
    assert findings["single_uncertain_multiplier_decides"].severity == "MEDIUM"
    assert findings["multiplier_lacks_evidence"].severity == "PASS"


def test_a_cited_multiplier_is_a_concern_but_not_a_blocker() -> None:
    """Evidence changes the verdict, not the observation: the reviewer still says a single
    factor decides the result, and stops short of blocking a world whose factor the record
    supplies."""

    findings = {f.key: f for f in mechanical_world_checks(_compile(ferry_plan(cited_factors=True)))}
    assert findings["single_uncertain_multiplier_decides"].severity == "MEDIUM"
    assert findings["multiplier_lacks_evidence"].severity == "PASS"
    assert findings["terminal_set_in_one_step"].severity == "CRITICAL"


# ---------------------------------------------------------------------------
# CWF-8 — the process runs, with and without actors, through the real engine
# ---------------------------------------------------------------------------


def _bundle(plan: dict[str, Any]) -> Any:
    compilation, _ = lower_plan(parse_semantic_plan(plan))
    cited = sorted({cid for cid in CLAIMS if cid in _cited_ids(plan)})
    data = {
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
            dict(CLAIMS[cid], id=cid, published_at="2026-01-05T00:00:00+00:00") for cid in cited
        ],
    }
    return build_bundle(data)


def _cited_ids(plan: Any) -> set[str]:
    """Every claim id the plan mentions anywhere — the store it is entitled to."""

    out: set[str] = set()
    if isinstance(plan, dict):
        for key, value in plan.items():
            if key == "evidence_claim_ids" and isinstance(value, list):
                out |= {str(v) for v in value}
            else:
                out |= _cited_ids(value)
    elif isinstance(plan, list):
        for item in plan:
            out |= _cited_ids(item)
    return out


def _compile(plan: dict[str, Any], gateway: Any = None) -> Any:
    bundle = _bundle(plan)
    contract = ResolutionContract(
        question=str(plan["resolution"]["question"]),
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )
    gw = gateway or ProgrammableGateway({"reflect": {"beliefs_update": [], "new_memories": []}})
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gw,
        seed=0,
        max_branches=4,
    )


def _volumes(result: Any) -> dict[str, float]:
    return {
        branch: float(world.fields_dict().get("recorded_recharge_volume", 0))
        for branch, world in result.final_worlds.items()
    }


def test_a_genuine_operational_process_runs_without_any_actor() -> None:
    """No decider anywhere, and the world still produces its answer by operating.

    Both stages fire, the intermediate volume is produced and then infiltrated, and the
    two branches separate on the driver — the wet branch crosses, the dry one does not.
    Nothing is announced; the total is reached.
    """

    gw = ProgrammableGateway({"reflect": {"beliefs_update": [], "new_memories": []}})
    result = run(_compile(recharge_plan(), gw), gw, seed=0)

    assert not result.actor_decisions, "an actor-free world must invoke nobody"
    volumes = sorted(_volumes(result).values())
    assert volumes == [720.0, 960.0]
    outcomes = {o.branch_id: (o.resolved, o.outcome) for o in result.branch_outcomes}
    assert sorted(outcomes.values()) == [(True, "NO"), (True, "YES")]


def _at_the_meeting(choice: dict[str, Any]) -> Any:
    """An engineer who decides the allocation at the meeting and not before.

    The runoff reading wakes him first — that is the world telling him something — and a
    person who has been told the runoff does not thereby spend the reserve; he does that
    at the dated occasion his authority attaches to.
    """

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["why_you_are_deciding_now"]["trigger"] == "process_opportunity":
            return choice
        return wait_decision("the allocation is decided at the operations meeting")

    return decide


def _gateway_for(choice: dict[str, Any]) -> ProgrammableGateway:
    return ProgrammableGateway(
        {
            "actor_decision": _at_the_meeting(choice),
            "reflect": {"beliefs_update": [], "new_memories": []},
        }
    )


def test_the_same_process_runs_with_an_actor_whose_decision_moves_the_outcome() -> None:
    """The same world, plus the one person the record says can act — and it matters.

    Releasing the emergency allocation carries the dry branch over the threshold; holding
    it leaves the same branch short. The operating process is identical in both runs, so
    what moved the answer is the decision, not the arithmetic — which is the difference
    between a world with an actor in it and a world with an actor drawn on it.
    """

    plan = recharge_plan(with_actor=True)

    releasing = _gateway_for(act("release_emergency_storage"))
    released = run(_compile(plan, releasing), releasing, seed=0)

    # The engineer learns the runoff first and waits, then acts at his dated occasion.
    informed = [
        d
        for d in released.actor_decisions
        if d.wake_reason and "compiled_wake_rule" in d.wake_reason
    ]
    assert informed and all(d.validation_status == "wait" for d in informed)
    acted = [d for d in released.actor_decisions if d.intent.get("action_id")]
    assert acted and all(a.intent["action_id"] == "release_emergency_storage" for a in acted)

    assert sorted(_volumes(released).values()) == [920.0, 1160.0]
    assert all(o.resolved and o.outcome == "YES" for o in released.branch_outcomes)

    holding = _gateway_for(act("hold_emergency_storage"))
    held = run(_compile(copy.deepcopy(plan), holding), holding, seed=0)
    assert sorted(_volumes(held).values()) == [720.0, 960.0]
    assert sorted((o.resolved, o.outcome) for o in held.branch_outcomes) == [
        (True, "NO"),
        (True, "YES"),
    ]


def test_an_actor_who_waits_leaves_the_process_to_produce_the_answer() -> None:
    """Waiting is a real choice, and the world does not stall on it: the same two-stage
    process runs to the same totals as the actor-free twin."""

    waiting = ProgrammableGateway(
        {
            "actor_decision": wait_decision("the basins are not below the trigger"),
            "reflect": {"beliefs_update": [], "new_memories": []},
        }
    )
    result = run(_compile(recharge_plan(with_actor=True), waiting), waiting, seed=0)
    assert sorted(_volumes(result).values()) == [720.0, 960.0]
