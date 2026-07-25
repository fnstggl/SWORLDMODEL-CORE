"""The semantic compiler's contract, proven on invented domains.

Every world here is deliberately from a domain no acceptance question uses — a harbor,
an irrigation council, an observatory — because the design succeeds only if previously
unseen meanings compile without any scenario-specific code path. The tests prove the
PART-15 regressions that belong to the compiler boundary: code owns every symbol, the
model's syntax choices cannot exist, unknown stays unknown, unsupported meaning is a
named gap, and the lowered world clears the existing gates unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from sworldmodel.semantic_lowering import LoweringGap, build_symbols, lower_plan
from sworldmodel.semantic_plan import (
    SemanticPlanError,
    parse_semantic_plan,
    validate_semantic_plan,
)

AS_OF = datetime.fromisoformat("2026-01-10T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-03-01T23:59:59+00:00")


def harbor_plan() -> dict:
    """A named-person communication world: one decider, one dated occasion, an
    event-existence terminal. No hardcoded 'public_statement' mechanism anywhere."""

    return {
        "resolution": {
            "question": "Will the harbormaster publicly authorize night docking before March 1?",
            "yes_condition": "The harbormaster publicly communicates authorization of "
            "night docking before the horizon.",
            "subject_entity": "Harbormaster of Port Solent",
            "resolution_units": "a public authorization",
            "target_outcome": "night docking publicly authorized",
            "expected_participants": 1,
            "evidence_claim_ids": ["c-r1"],
        },
        "entities": [
            {
                "name": "Harbormaster of Port Solent",
                "structural_type": "person",
                "role": "harbormaster with sole authority over docking rules",
                "representation_scale": "individual",
                "represents_count": None,
                "decides": True,
                "authority": "may authorize or refuse night docking",
                "why_material": "the outcome is their own public act",
                "evidence_claim_ids": ["c-r1"],
            }
        ],
        "states": [],
        "events": [
            {
                "name": "night docking authorization",
                "meaning": "the harbormaster publicly communicates that night docking "
                "is authorized",
                "participants": {"speaker": "Harbormaster of Port Solent"},
                "visibility": "public",
                "information_created": "vessels may dock at night",
                "evidence_claim_ids": ["c-r1"],
            }
        ],
        "affordances": [
            {
                "name": "authorize night docking",
                "meaning": "publicly authorize night docking",
                "actor": "Harbormaster of Port Solent",
                "authority_required": "harbormaster docking authority",
                "visibility": "public",
                "duration_seconds": 0,
                "changes": [
                    {
                        "op": "record_event",
                        "target": "night docking authorization",
                        "detail": "authorized",
                    }
                ],
                "evidence_claim_ids": ["c-r1"],
            },
            {
                "name": "decline to authorize",
                "meaning": "publicly state that night docking stays prohibited",
                "actor": "Harbormaster of Port Solent",
                "authority_required": "harbormaster docking authority",
                "visibility": "public",
                "changes": [
                    {
                        "op": "send",
                        "target": "docking decision notice",
                        "recipients": ["Harbormaster of Port Solent"],
                        "detail": "night docking stays prohibited",
                    }
                ],
                "evidence_claim_ids": ["c-r1"],
            },
        ],
        "processes": [
            {
                "name": "harbor board session",
                "meaning": "the dated session at which docking rules are decided",
                "kind": "actor_moment",
                "participants": ["Harbormaster of Port Solent"],
                "allowed_affordances": ["authorize night docking", "decline to authorize"],
                "at": "2026-02-10T10:00:00+00:00",
                "deadline": "2026-02-28T23:59:59+00:00",
                "occurrences": [],
                "evidence_claim_ids": ["c-r1"],
            }
        ],
        "uncertainties": [],
        "terminal": {"form": "event_exists", "event": "night docking authorization"},
        "terminal_producer_note": "Only the harbormaster's own affordance records the "
        "authorization event; nothing initializes it and no uncertainty writes it.",
        "world_facts": [],
    }


def council_plan() -> dict:
    """A formal multi-member decision: five-of-nine stays five-of-nine, and an
    aggregate delegation represents seven members without becoming one."""

    return {
        "resolution": {
            "question": "Will the irrigation council adopt the drought measure with at "
            "least five of nine votes before March 1?",
            "yes_condition": "At least five adoption votes are recorded.",
            "subject_entity": "Upper Valley Irrigation Council",
            "resolution_units": "recorded adoption votes",
            "target_outcome": "the drought measure is adopted",
            "expected_participants": 9,
            "evidence_claim_ids": ["c-c1"],
        },
        "entities": [
            {
                "name": "Council chair",
                "structural_type": "person",
                "role": "chairs the council and votes",
                "representation_scale": "individual",
                "decides": True,
                "authority": "casts one vote",
                "why_material": "one of nine votes",
                "evidence_claim_ids": ["c-c1"],
            },
            {
                "name": "Council member for the east district",
                "structural_type": "person",
                "role": "council member",
                "representation_scale": "individual",
                "decides": True,
                "authority": "casts one vote",
                "why_material": "one of nine votes",
                "evidence_claim_ids": ["c-c1"],
            },
            {
                "name": "Delegation of seven basin towns",
                "structural_type": "coalition",
                "role": "seven town representatives who vote as a bloc",
                "representation_scale": "subunit",
                "represents_count": 7,
                "decides": True,
                "authority": "casts seven bloc votes",
                "why_material": "seven of nine votes",
                "evidence_claim_ids": ["c-c2"],
            },
        ],
        "states": [],
        "events": [
            {
                "name": "adoption vote",
                "meaning": "a council participant records a vote to adopt the measure",
                "participants": {},
                "visibility": "public",
                "evidence_claim_ids": ["c-c1"],
            }
        ],
        "affordances": [
            {
                "name": "chair votes to adopt",
                "meaning": "the chair records an adoption vote",
                "actor": "Council chair",
                "changes": [{"op": "record_event", "target": "adoption vote", "detail": "chair"}],
                "evidence_claim_ids": ["c-c1"],
            },
            {
                "name": "east member votes to adopt",
                "meaning": "the east-district member records an adoption vote",
                "actor": "Council member for the east district",
                "changes": [{"op": "record_event", "target": "adoption vote", "detail": "east"}],
                "evidence_claim_ids": ["c-c1"],
            },
            {
                "name": "delegation votes to adopt",
                "meaning": "the seven-town delegation records its bloc adoption votes",
                "actor": "Delegation of seven basin towns",
                "changes": [
                    {"op": "record_event", "target": "adoption vote", "detail": "bloc of seven"}
                ],
                "evidence_claim_ids": ["c-c2"],
            },
        ],
        "processes": [
            {
                "name": "council vote session",
                "meaning": "the dated session at which votes are recorded",
                "kind": "actor_moment",
                "participants": [
                    "Council chair",
                    "Council member for the east district",
                    "Delegation of seven basin towns",
                ],
                "at": "2026-02-15T09:00:00+00:00",
                "deadline": "2026-02-28T23:59:59+00:00",
                "occurrences": [],
                "evidence_claim_ids": ["c-c1"],
            }
        ],
        "uncertainties": [],
        "terminal": {
            "form": "record_count",
            "record_event": "adoption vote",
            "comparison": "greater_or_equal",
            "threshold": 5,
        },
        "terminal_producer_note": "Votes exist only as records the voters' own "
        "affordances append; the five-of-nine threshold is the real decision rule.",
        "world_facts": [],
    }


def observatory_plan() -> dict:
    """An operational aggregate: the total is produced by accumulation, uncertainty
    sits on the driver, and the report merely observes."""

    return {
        "resolution": {
            "question": "Will the observatory log more than 400 clear-sky hours this "
            "quarter?",
            "yes_condition": "Logged clear-sky hours exceed 400 at quarter end.",
            "subject_entity": "Mount Aster Observatory",
            "resolution_units": "clear-sky hours",
            "target_outcome": "more than 400 clear-sky hours logged",
            "expected_participants": None,
            "evidence_claim_ids": ["c-o1"],
        },
        "entities": [
            {
                "name": "Mount Aster Observatory",
                "structural_type": "institution",
                "role": "operates the sky log",
                "representation_scale": "organization",
                "decides": False,
                "authority": "maintains the official log",
                "why_material": "its log is the resolution source",
                "evidence_claim_ids": ["c-o1"],
            }
        ],
        "states": [
            {
                "name": "logged clear-sky hours",
                "owner": "world",
                "state_type": "quantity",
                "unit": "hours",
                "initial": 210,
                "why_material": "the terminal reads it",
                "evidence_claim_ids": ["c-o1"],
            },
            {
                "name": "february clear fraction",
                "owner": "world",
                "state_type": "quantity",
                "unit": "fraction",
                "initial": "UNKNOWN",
                "why_material": "drives how many hours February adds",
                "evidence_claim_ids": [],
            },
        ],
        "events": [],
        "affordances": [],
        "processes": [
            {
                "name": "february observing runs",
                "meaning": "February nights add clear-sky hours at the realised rate",
                "kind": "operational",
                "inputs": ["february clear fraction"],
                "occurrences": [
                    {
                        "description": "February's accumulation",
                        "at": "2026-02-28T23:00:00+00:00",
                        "changes": [
                            {
                                "op": "increase",
                                "target": "logged clear-sky hours",
                                "amount": {
                                    "kind": "product",
                                    "parts": [
                                        {"kind": "literal", "value": 240},
                                        {"kind": "state", "state": "february clear fraction"},
                                    ],
                                },
                            }
                        ],
                    }
                ],
                "evidence_claim_ids": ["c-o2"],
            }
        ],
        "uncertainties": [
            {
                "name": "february weather",
                "what_unknown": "the fraction of February hours that are clear",
                "why_unknown": "weather has not happened yet",
                "affects_state": "february clear fraction",
                "release_at": None,
                "alternatives": [
                    {
                        "value": 0.9,
                        "weight": None,
                        "provenance": "symmetric_ignorance_assumption",
                        "grounding": "clear winter scenario",
                    },
                    {
                        "value": 0.5,
                        "weight": None,
                        "provenance": "symmetric_ignorance_assumption",
                        "grounding": "cloudy winter scenario",
                    },
                ],
            }
        ],
        "terminal": {
            "form": "quantity_comparison",
            "state": "logged clear-sky hours",
            "comparison": "greater_than",
            "threshold": 400,
        },
        "terminal_producer_note": "The total starts at the cited 210 and is produced by "
        "the February accumulation at an uncertain rate; the uncertainty drives the "
        "rate, never the total.",
        "world_facts": [],
    }


def _valid(data: dict) -> list[str]:
    plan = parse_semantic_plan(data)
    return validate_semantic_plan(
        plan,
        as_of=AS_OF,
        horizon=HORIZON,
        known_claim_ids=frozenset({"c-r1", "c-c1", "c-c2", "c-o1", "c-o2"}),
    )


def test_three_unseen_domains_validate_and_lower_without_domain_code() -> None:
    for maker in (harbor_plan, council_plan, observatory_plan):
        data = maker()
        assert _valid(data) == []
        compilation, mapping = lower_plan(parse_semantic_plan(data))
        spec = compilation["world_spec"]
        assert spec["terminal"]["yes_when"]["op"] in (
            "greater_than",
            "greater_or_equal",
            "equals",
        )
        assert mapping, "the semantic→runtime mapping artifact must exist"


def test_the_model_never_chooses_between_field_and_field_id() -> None:
    """Every lowered effect keys its target as the runtime's canonical 'field' — the
    synonym drift that broke a live run cannot exist, because the model never writes
    the key at all."""

    compilation, _ = lower_plan(parse_semantic_plan(observatory_plan()))
    spec = compilation["world_spec"]
    effects = [
        eff
        for proc in spec["external_processes"]
        for occ in proc["occurrences"]
        for eff in occ["effects"]
    ]
    assert effects
    for eff in effects:
        if eff["op"] in ("set_field", "adjust_field"):
            assert "field" in eff and "field_id" not in eff


def test_terminal_needs_no_model_authored_expression_syntax() -> None:
    """The planner said 'event_exists' in ordinary terms; code wrote the AST."""

    compilation, _ = lower_plan(parse_semantic_plan(harbor_plan()))
    yes = compilation["world_spec"]["terminal"]["yes_when"]
    assert yes == {
        "op": "greater_than",
        "args": [{"op": "event_count", "args": ["night_docking_authorization"]}, 0],
    }


def test_an_aggregate_represents_seven_without_becoming_one_participant() -> None:
    compilation, _ = lower_plan(parse_semantic_plan(council_plan()))
    delegation = next(
        e
        for e in compilation["world_spec"]["entities"]
        if e["name"] == "Delegation of seven basin towns"
    )
    assert delegation["represents_count"] == 7
    assert delegation["is_actor"] is True


def test_five_of_nine_remains_five_of_nine() -> None:
    compilation, _ = lower_plan(parse_semantic_plan(council_plan()))
    yes = compilation["world_spec"]["terminal"]["yes_when"]
    assert yes["op"] == "greater_or_equal"
    assert yes["args"][1] == 5  # the real threshold, not the object count (3)


def test_a_terminal_without_a_producer_is_refused_before_lowering() -> None:
    data = harbor_plan()
    data["affordances"] = [data["affordances"][1]]  # keep only the non-recording act
    errors = _valid(data)
    assert any("no affordance or process" in e and "record" in e for e in errors)


def test_unknown_stays_unknown_in_the_lowered_world() -> None:
    """UNKNOWN never becomes zero or False: the lowered field simply has no initial,
    and the runtime reads it as honestly undetermined."""

    compilation, _ = lower_plan(parse_semantic_plan(observatory_plan()))
    fields = {f["field_id"]: f for f in compilation["world_spec"]["fields"]}
    assert "initial" not in fields["february_clear_fraction"]
    assert fields["logged_clear_sky_hours"]["initial"] == 210


def test_an_uncertainty_may_not_write_the_terminal_state() -> None:
    data = observatory_plan()
    data["uncertainties"][0]["affects_state"] = "logged clear-sky hours"
    errors = _valid(data)
    assert any("set directly by an uncertainty" in e for e in errors)


def test_a_bare_copy_of_an_uncertain_state_is_refused_at_the_semantic_level() -> None:
    data = observatory_plan()
    data["processes"][0]["occurrences"][0]["changes"] = [
        {
            "op": "set",
            "target": "logged clear-sky hours",
            "value": {"kind": "state", "state": "february clear fraction"},
        }
    ]
    errors = _valid(data)
    assert any("bare copy" in e for e in errors)


def test_unsupported_meaning_returns_a_named_lowering_gap() -> None:
    data = harbor_plan()
    data["affordances"][0]["changes"].append(
        {"op": "schedule", "target": "harbor board session"}
    )
    plan = parse_semantic_plan(data)
    with pytest.raises(LoweringGap) as exc:
        lower_plan(plan)
    details = exc.value.details
    assert details["failure"] == "lowering_gap"
    assert "schedule" in str(details["unsupported construct"])
    assert details["smallest missing universal capability"]


def test_lowering_is_deterministic_to_the_byte() -> None:
    a, _ = lower_plan(parse_semantic_plan(council_plan()))
    b, _ = lower_plan(parse_semantic_plan(council_plan()))
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def test_colliding_semantic_names_get_distinct_symbols() -> None:
    data = observatory_plan()
    data["states"].append(
        {
            "name": "Logged clear sky hours",  # slugs identically to the first state
            "owner": "world",
            "state_type": "quantity",
            "unit": "hours",
            "initial": "UNKNOWN",
            "why_material": "a second, distinct ledger",
            "evidence_claim_ids": [],
        }
    )
    plan = parse_semantic_plan(data)
    table = build_symbols(plan)
    ids = [r["runtime_id"] for r in table.records if r["namespace"] == "field"]
    assert len(ids) == len(set(ids)), "colliding names must mint distinct symbols"
    colliding = [i for i in ids if i.startswith("logged_clear_sky_hours")]
    assert len(colliding) == 2 and len(set(colliding)) == 2


def test_a_deciding_entity_with_no_affordance_is_a_semantic_error() -> None:
    data = harbor_plan()
    data["affordances"] = []
    errors = _valid(data)
    assert any("decides but has no affordance" in e for e in errors)


def test_an_actor_moment_without_a_date_is_a_semantic_error() -> None:
    data = harbor_plan()
    data["processes"][0]["at"] = None
    data["processes"][0]["deadline"] = None
    errors = _valid(data)
    assert any("dated occasion" in e for e in errors)


def test_unreadable_shapes_raise_precise_field_errors() -> None:
    with pytest.raises(SemanticPlanError) as exc:
        parse_semantic_plan({"resolution": {}, "terminal": {"form": "no_such_form"}})
    assert any("unknown terminal form" in e for e in exc.value.errors)


def test_the_lowered_world_clears_the_existing_gates_unchanged() -> None:
    """End to end through the REAL downstream path: assemble_bundle parses the lowered
    dict, compile_world runs every gate — reality, grounding, executability,
    outcome-production, pre-resolution, coverage — and the world stands. The gates are
    the same executor for both compiler modes; nothing semantic bypasses them."""

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _fakes import ProgrammableGateway, build_bundle
    from sworldmodel.models import ResolutionContract
    from sworldmodel.world_compiler import compile_world

    compilation, _ = lower_plan(parse_semantic_plan(harbor_plan()))
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
            {
                "id": "c-r1",
                "proposition": "The harbormaster of Port Solent holds sole authority "
                "over docking rules and has scheduled a decision session",
                "value": "sole authority",
                "entities": ["Harbormaster of Port Solent"],
                "published_at": "2026-01-05T00:00:00+00:00",
            }
        ],
    }
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="Will the harbormaster publicly authorize night docking before March 1?",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )
    gw = ProgrammableGateway({"reflect": {"beliefs_update": [], "new_memories": []}})
    compiled = compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gw,
        seed=0,
        max_branches=4,
    )
    assert compiled.spec.actions and compiled.spec.process.nodes
