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
            "question": "Will the observatory log more than 400 clear-sky hours this quarter?",
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
    assert any("uncertainty draws alone" in e for e in errors)


def test_unsupported_meaning_returns_a_named_lowering_gap() -> None:
    data = harbor_plan()
    data["affordances"][0]["changes"].append({"op": "schedule", "target": "harbor board session"})
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


def test_a_chained_operational_process_actually_fires_through_the_real_engine() -> None:
    """Dependency chains execute, and UNKNOWN stays unresolved — proven by running.

    The runtime schedules successors from `next_nodes` alone; an earlier lowering
    chained occurrences with `after_node` only, so every dependent occurrence was inert
    while still counting as a terminal producer — the compile gates passed a world whose
    deciding event could never happen. This lowers a two-stage accumulation (a dated
    first run, a dependent second run carrying the crossing amount), executes it through
    the REAL engine, and asserts the terminal resolved YES from the produced total. The
    same run proves the honest-unresolved contract: a second world whose only producer
    reads an UNKNOWN driver must report unresolved, never a manufactured NO.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _fakes import ProgrammableGateway, build_bundle
    from sworldmodel.engine import run
    from sworldmodel.models import ResolutionContract
    from sworldmodel.world_compiler import compile_world

    def chained(second_stage_changes: list[dict]) -> dict:
        return {
            "resolution": {
                "question": "Will the canal system move more than 400 barges this season?",
                "yes_condition": "Total barges moved exceeds 400.",
                "subject_entity": "Canal Authority",
                "resolution_units": "barges moved",
                "target_outcome": "more than 400 barges moved",
                "expected_participants": None,
                "evidence_claim_ids": ["c-k1"],
            },
            "entities": [
                {
                    "name": "Canal Authority",
                    "structural_type": "institution",
                    "role": "operates the locks",
                    "representation_scale": "organization",
                    "decides": False,
                    "authority": "operates the canal",
                    "why_material": "its operations produce the total",
                    "evidence_claim_ids": ["c-k1"],
                }
            ],
            "states": [
                {
                    "name": "barges moved",
                    "owner": "world",
                    "state_type": "quantity",
                    "unit": "barges",
                    "initial": 0,
                    "why_material": "the terminal reads it",
                    "evidence_claim_ids": ["c-k1"],
                }
            ],
            "events": [],
            "affordances": [],
            "processes": [
                {
                    "name": "early season locks",
                    "meaning": "the first lock cycle moves a verified count",
                    "kind": "operational",
                    "occurrences": [
                        {
                            "description": "first cycle",
                            "at": "2026-01-20T00:00:00+00:00",
                            "changes": [
                                {"op": "increase", "target": "barges moved", "amount": 150}
                            ],
                        }
                    ],
                    "evidence_claim_ids": ["c-k1"],
                },
                {
                    "name": "late season locks",
                    "meaning": "the second cycle follows the first",
                    "kind": "operational",
                    "occurrences": [
                        {
                            "description": "second cycle",
                            "after_process": "early season locks",
                            "delay_seconds": 3600,
                            "changes": second_stage_changes,
                        }
                    ],
                    "evidence_claim_ids": ["c-k1"],
                },
            ],
            "uncertainties": [],
            "terminal": {
                "form": "quantity_comparison",
                "state": "barges moved",
                "comparison": "greater_than",
                "threshold": 400,
            },
            "terminal_producer_note": "The total is produced by the two lock cycles; "
            "nothing initializes it above zero and no uncertainty writes it.",
            "world_facts": [],
        }

    def run_world(plan_dict: dict):
        compilation, _ = lower_plan(parse_semantic_plan(plan_dict))
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
                    "id": "c-k1",
                    "proposition": "The Canal Authority operates two lock cycles per "
                    "season, the first moving 150 barges",
                    "value": "150 first cycle",
                    "entities": ["Canal Authority"],
                    "published_at": "2026-01-05T00:00:00+00:00",
                }
            ],
        }
        bundle = build_bundle(data)
        contract = ResolutionContract(
            question=str(plan_dict["resolution"]["question"]),
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
        return run(compiled, gw, seed=0)

    # The dependent second cycle carries the crossing amount: if next_nodes chaining is
    # broken it never fires, the total stays 150, and the terminal would resolve NO.
    produced = run_world(chained([{"op": "increase", "target": "barges moved", "amount": 300}]))
    outcomes = list(produced.branch_outcomes)
    assert outcomes and all(o.resolved and o.outcome == "YES" for o in outcomes), outcomes

    # An UNKNOWN driver must surface as honest unresolved, never a manufactured NO: the
    # second cycle scales an UNKNOWN rate, so the total is undetermined at the terminal.
    unknown_plan = chained(
        [
            {
                "op": "increase",
                "target": "barges moved",
                "amount": {
                    "kind": "product",
                    "parts": [
                        {"kind": "literal", "value": 300},
                        {"kind": "state", "state": "late season throughput factor"},
                    ],
                },
            }
        ]
    )
    unknown_plan["states"].append(
        {
            "name": "late season throughput factor",
            "owner": "world",
            "state_type": "quantity",
            "unit": "fraction",
            "initial": "UNKNOWN",
            "why_material": "scales the second cycle",
            "evidence_claim_ids": [],
        }
    )
    undetermined = run_world(unknown_plan)
    outcomes = list(undetermined.branch_outcomes)
    assert outcomes and all(not o.resolved for o in outcomes), outcomes


def test_a_pure_factual_resolution_lowers_and_clears_the_gates() -> None:
    """The sanctioned 'already settled' encoding: one cited state, no mechanism.

    A live OPEC+ slice compiled exactly this — the record established the announcements
    had already been made — and the nothing_can_act gate refused it for having no
    actions, which is what a factual resolution deliberately has none of. The gate now
    recognizes the pure record shape: every terminal term evidence-established, nothing
    acting, no uncertainty near the terminal.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _fakes import ProgrammableGateway, build_bundle
    from sworldmodel.models import ResolutionContract
    from sworldmodel.world_compiler import compile_world

    plan = {
        "resolution": {
            "question": "Did the reservoir authority publish its annual level report "
            "before February 1?",
            "yes_condition": "The report is already published, per the cited record.",
            "subject_entity": "Reservoir Authority",
            "resolution_units": "a published report",
            "target_outcome": "the report is published",
            "expected_participants": None,
            "evidence_claim_ids": ["c-f1"],
        },
        "entities": [
            {
                "name": "Reservoir Authority",
                "structural_type": "institution",
                "role": "publishes the annual report",
                "representation_scale": "organization",
                "decides": False,
                "authority": "publishes official reports",
                "why_material": "its record settles the question",
                "evidence_claim_ids": ["c-f1"],
            }
        ],
        "states": [
            {
                "name": "annual report published",
                "owner": "world",
                "state_type": "boolean",
                "initial": True,
                "why_material": "the terminal reads it",
                "evidence_claim_ids": ["c-f1"],
            }
        ],
        "events": [],
        "affordances": [],
        "processes": [],
        "uncertainties": [],
        "terminal": {"form": "state_equals", "state": "annual report published", "value": True},
        "terminal_producer_note": "The cited record establishes publication before the "
        "window opened; nothing needs to act and nothing does.",
        "world_facts": [],
    }
    assert _valid_with(plan, {"c-f1"}) == []
    compilation, _ = lower_plan(parse_semantic_plan(plan))
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
                "id": "c-f1",
                "proposition": "The Reservoir Authority published its annual level "
                "report on January 12",
                "value": "published January 12",
                "entities": ["Reservoir Authority"],
                "published_at": "2026-01-12T00:00:00+00:00",
            }
        ],
    }
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question=str(plan["resolution"]["question"]),  # type: ignore[index]
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
    assert not compiled.spec.actions and not compiled.spec.external_processes


def _valid_with(data: dict, ids: set[str]) -> list[str]:
    return validate_semantic_plan(
        parse_semantic_plan(data), as_of=AS_OF, horizon=HORIZON, known_claim_ids=frozenset(ids)
    )


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


def test_wake_rules_are_derived_so_the_world_is_not_inert_between_moments() -> None:
    """H-3: the engine consults spec.wake_rules only after the directed/asked/revisit/
    communication checks, and field writes and appended records stay ambient unless a
    compiled rule says otherwise. A lowering that emitted wake_rules=[] made every
    semantic world inert between dated actor moments. The rules must be derived: a
    process input changing wakes the deciders that react to it, a declared event wakes
    its deciding participants, and the terminal's counted records wake everyone who can
    still act."""

    from sworldmodel.worldspec import parse_wake_rule

    data = harbor_plan()
    data["states"] = [
        {
            "name": "queued night arrivals",
            "owner": "world",
            "state_type": "quantity",
            "unit": "vessels",
            "initial": 4,
            "why_material": "pressure on the docking decision",
            "evidence_claim_ids": ["c-r1"],
        }
    ]
    data["processes"].append(
        {
            "name": "arrival ledger updates",
            "meaning": "the ledger accrues queued night arrivals through the window",
            "kind": "operational",
            "inputs": ["queued night arrivals"],
            "occurrences": [
                {
                    "description": "weekly ledger update",
                    "at": "2026-01-20T00:00:00+00:00",
                    "changes": [{"op": "increase", "target": "queued night arrivals", "amount": 2}],
                }
            ],
            "evidence_claim_ids": ["c-r1"],
        }
    )
    assert _valid(data) == []
    compilation, _ = lower_plan(parse_semantic_plan(data))
    spec = compilation["world_spec"]
    rules = spec["wake_rules"]
    assert rules, "a world with mechanisms and deciders must compile wake rules"

    entity_ids = {e["entity_id"] for e in spec["entities"]}
    field_ids = {f["field_id"] for f in spec["fields"]}
    for r in rules:
        parsed = parse_wake_rule(r)
        assert parsed.is_checkable(), f"rule {r['rule_id']!r} has no trigger"
        assert r["wakes"], f"rule {r['rule_id']!r} wakes nobody"
        assert set(r["wakes"]) <= entity_ids, f"rule {r['rule_id']!r} wakes unknown entities"

    # (a) the operational process's input changing wakes the decider that reads it.
    field_rules = [r for r in rules if r["on_field_change"]]
    assert field_rules and all(r["on_field_change"] in field_ids for r in field_rules)
    # (b) the declared event wakes its deciding participant, quoting the plan's meaning.
    event_rules = [r for r in rules if r["on_event_type"]]
    assert event_rules
    assert any(harbor_plan()["events"][0]["meaning"] in r["reason"] for r in event_rules)
    # (c) the terminal counts this event's records, so appending one wakes the deciders.
    record_rules = [r for r in rules if r["on_record_in"]]
    assert record_rules and all(set(r["wakes"]) == entity_ids for r in record_rules)


def test_a_private_event_reaches_its_own_participants() -> None:
    """H-4: the runtime's audience resolver reads only the effect's "to"/"audience"
    keys, so a private event whose participants rode only in `data` had an empty
    audience — visible_to returned False for everyone and the briefing reached nobody,
    including its own participants. Lowering must emit the participants where delivery
    actually looks."""

    from sworldmodel.effects import _audience

    data = harbor_plan()
    data["entities"].append(
        {
            "name": "Night pilots guild",
            "structural_type": "organization",
            "role": "guild whose pilots perform night approaches",
            "representation_scale": "organization",
            "decides": False,
            "authority": "operates night pilotage",
            "why_material": "the authorization is addressed to its pilots",
            "evidence_claim_ids": ["c-r1"],
        }
    )
    data["events"][0]["visibility"] = "private"
    data["events"][0]["participants"] = {
        "speaker": "Harbormaster of Port Solent",
        "briefed": "Night pilots guild",
    }
    assert _valid(data) == []
    compilation, _ = lower_plan(parse_semantic_plan(data))
    spec = compilation["world_spec"]
    id_by_name = {e["name"]: e["entity_id"] for e in spec["entities"]}
    both = {id_by_name["Harbormaster of Port Solent"], id_by_name["Night pilots guild"]}
    creates = [eff for a in spec["actions"] for eff in a["effects"] if eff["op"] == "create_event"]
    assert creates
    for eff in creates:
        assert set(eff["to"]) == both, "a private event's participants are its audience"
        # Proven against the runtime's own audience resolver, not a re-implementation.
        assert set(_audience("create_event", eff)) == both


def test_a_participantless_private_event_is_refused() -> None:
    data = harbor_plan()
    data["events"][0]["visibility"] = "private"
    data["events"][0]["participants"] = {}
    errors = _valid(data)
    assert any("private event with no participants" in e for e in errors)


def test_required_reality_facts_are_derived_from_cited_plan_content() -> None:
    """H-2: emitting required_reality_facts=[] made reality-gate check 4 vacuous in
    semantic mode only — nothing was ever checked against the evidence and
    evidence_coverage reported a meaningless 1.0. What the plan states as established
    (a cited world_fact, a citation-established initial state) must become facts the
    gate verifies against available evidence."""

    from sworldmodel.world_compiler import parse_required_facts

    data = observatory_plan()
    data["world_facts"] = [
        {
            "text": "The observatory's sky log is the official record",
            "evidence_claim_ids": ["c-o1"],
        }
    ]
    assert _valid(data) == []
    compilation, _ = lower_plan(parse_semantic_plan(data))
    facts = compilation["required_reality_facts"]
    assert facts, "cited plan content must yield required reality facts"
    for f in facts:
        assert f["key"] and f["description"]
        assert f["evidence_claim_ids"], f"fact {f['key']!r} cites no claims"

    descriptions = [f["description"] for f in facts]
    # The cited world_fact is required, citing its claim ids.
    fact = next(f for f in facts if "official record" in f["description"])
    assert fact["evidence_claim_ids"] == ["c-o1"]
    # The citation-established initial state is required; the UNKNOWN one must not be.
    state_fact = next(f for f in facts if "logged clear-sky hours" in f["description"])
    assert state_fact["evidence_claim_ids"] == ["c-o1"]
    assert not any("february clear fraction" in d for d in descriptions)
    # And the real downstream parser reads every entry, so check 4 has work to do.
    parsed = parse_required_facts(facts)
    assert len(parsed) == len(facts) and all(f.evidence_claim_ids for f in parsed)


def test_no_validated_field_vanishes_without_lowering_or_a_dropped_record() -> None:
    """M-2: SemanticProcess.inputs, information_produced, deadline on non-actor_moment
    processes and SemanticState.why_material were validated and then discarded with no
    trace. The silent-loss policy: everything the validator accepted must either lower
    into something a consumer reads, or leave an explicit "dropped" mapping record
    saying why it could not."""

    data = observatory_plan()
    data["processes"][0]["information_produced"] = "the realised february observing tally"
    data["processes"][0]["deadline"] = "2026-02-27T00:00:00+00:00"
    assert _valid(data) == []
    compilation, mapping = lower_plan(parse_semantic_plan(data))
    blob = json.dumps(compilation, default=str)

    # information_produced and why_material lower into descriptions consumers read.
    assert "the realised february observing tally" in blob
    assert "drives how many hours February adds" in blob

    # This world has no deciding entity, so the process input can wake nobody, and an
    # external process has no node to carry a deadline: both leave explicit dropped
    # records in the mapping artifact rather than vanishing.
    dropped = [r for r in mapping if r["namespace"] == "dropped"]
    assert dropped
    for r in dropped:
        assert r["runtime_id"] == ""
        assert str(r["lowering_rule"]).startswith("carried nowhere: ")
    assert any("february clear fraction" in r["semantic"] for r in dropped)
    assert any("deadline" in r["semantic"] for r in dropped)

    # The same deadline on a NODE process is not dropped: chained into the process
    # graph, every emitted node carries it — a deadline is a deadline whatever the
    # process kind, not an actor_moment privilege.
    data2 = observatory_plan()
    data2["processes"][0]["deadline"] = "2026-02-27T00:00:00+00:00"
    data2["processes"].append(
        {
            "name": "march calibration",
            "meaning": "the calibration pass follows the february runs",
            "kind": "operational",
            "occurrences": [
                {
                    "description": "calibration",
                    "after_process": "february observing runs",
                    "delay_seconds": 3600,
                    "changes": [
                        {"op": "increase", "target": "logged clear-sky hours", "amount": 1}
                    ],
                }
            ],
            "evidence_claim_ids": ["c-o2"],
        }
    )
    assert _valid(data2) == []
    comp2, mapping2 = lower_plan(parse_semantic_plan(data2))
    nodes = {n["node_id"]: n for n in comp2["world_spec"]["process"]["nodes"]}
    # The engine schedules deadline entries only for a node's participants; on a
    # participant-less operational node the field would be inert, so it is recorded as
    # honestly dropped rather than carried in name only.
    assert "deadline" not in nodes["february_observing_runs"]
    assert any(r["namespace"] == "dropped" and "deadline" in r["semantic"] for r in mapping2)


def test_must_refuse_false_gaps_skip_the_revision_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M-3: LoweringGap.must_refuse was written into the gap's details and never read
    — every gap took the one-revision-then-refuse path. must_refuse=False means "no
    rephrasing can help" (an unresolved reference is a validator defect, not a plan
    defect), so semantic_compile_live must refuse immediately without spending the
    revision round; the default True keeps the existing path. The gateway stub stands
    in ONLY for the external model provider; everything else is the production path."""

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import sworldmodel.semantic_compile as sc
    from _fakes import ProgrammableGateway, build_bundle
    from sworldmodel.semantic_plan import SemanticPlan

    compilation, _ = lower_plan(parse_semantic_plan(harbor_plan()))
    bundle = build_bundle(
        {
            "world_spec": compilation["world_spec"],
            "reality": {
                "subject_entity": compilation["subject_entity"],
                "resolution_units": compilation["resolution_units"],
                "target_outcome": compilation["target_outcome"],
                "as_of": AS_OF.isoformat(),
                "horizon": HORIZON.isoformat(),
            },
            "claims": [
                {
                    "id": "c-r1",
                    "proposition": "The harbormaster of Port Solent holds sole "
                    "authority over docking rules",
                    "value": "sole authority",
                    "entities": ["Harbormaster of Port Solent"],
                    "published_at": "2026-01-05T00:00:00+00:00",
                }
            ],
        }
    )
    view = bundle.evidence_store.view(AS_OF)

    def run_with(gap: LoweringGap) -> tuple[int, LoweringGap]:
        gw = ProgrammableGateway(
            {
                "semantic_plan": harbor_plan(),
                "semantic_review": {"verdict": "APPROVE", "reasons": [], "corrections": []},
            }
        )

        def raising_lower(
            plan: SemanticPlan, *, structure_id: str = "primary"
        ) -> tuple[dict, list]:
            raise gap

        monkeypatch.setattr(sc, "lower_plan", raising_lower)
        with pytest.raises(LoweringGap) as exc:
            sc.semantic_compile_live(
                gw,
                "Will the harbormaster publicly authorize night docking before March 1?",
                AS_OF,
                HORIZON,
                view,
            )
        planner_calls = sum(1 for r in gw.seen if r.task_kind == "semantic_plan")
        return planner_calls, exc.value

    stubborn = LoweringGap(
        "a construct a rephrased plan might avoid",
        why="the mapping cannot represent it as phrased",
        composable=True,
        smallest_missing="a universal mapping",
    )
    calls, err = run_with(stubborn)
    assert calls == 2, "must_refuse=True must keep the one-revision-then-refuse path"
    assert err is stubborn

    hopeless = LoweringGap(
        "an unresolved reference",
        why="no rephrasing can help; the validator should have refused this plan",
        composable=False,
        smallest_missing="nothing",
        must_refuse=False,
    )
    calls, err = run_with(hopeless)
    assert calls == 1, "must_refuse=False must skip the revision round entirely"
    assert err is hopeless

    # The flag is set where it belongs: the production unresolved-reference gap
    # (a validator defect, not a plan defect) refuses without a revision.
    table = build_symbols(parse_semantic_plan(harbor_plan()))
    with pytest.raises(LoweringGap) as ref:
        table.resolve("entity", "never declared anywhere")
    assert ref.value.must_refuse is False


def test_an_occurrence_at_or_before_the_cutoff_is_refused() -> None:
    """The simulation cannot re-perform history: a t0 occurrence that records the
    resolving event let a branch resolve YES off a re-enactment nobody produced."""

    data = harbor_plan()
    data["processes"].append(
        {
            "name": "pre-window recording",
            "meaning": "records the authorization at the cutoff",
            "kind": "operational",
            "occurrences": [
                {
                    "description": "t0 re-enactment",
                    "at": AS_OF.isoformat(),
                    "changes": [
                        {
                            "op": "record_event",
                            "target": "night docking authorization",
                            "detail": "already authorized",
                        }
                    ],
                }
            ],
            "evidence_claim_ids": ["c-r1"],
        }
    )
    errors = _valid(data)
    assert any("re-perform history" in e for e in errors)
