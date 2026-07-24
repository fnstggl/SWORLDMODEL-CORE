"""Shared cross-domain WorldSpec corpus builders for the tests.

Every builder returns a corpus dict in the *same* schema (evidence ``sources`` +
compiled ``world_spec`` + ``uncertainties`` + reality metadata). They differ only in
DATA — different entities, actions, process graphs and declarative terminals — and all
run through the one universal engine. There are no Banxico facts anywhere here; these
are synthetic worlds that exercise a committee decision, an individual response, a
negotiation, a population behavior, a geopolitical/organizational process, plus the
unknown-action and novel-action routes.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from sworldmodel import DeterministicGateway, ForecastConfig, MockResearchBackend, run_forecast

AS_OF = "2024-01-15T00:00:00+00:00"
HORIZON = "2024-02-15T12:00:00+00:00"
PUB = "2024-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Corpus assembly helpers
# ---------------------------------------------------------------------------


def _source(claims: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_id": "roster",
        "url": "https://example.org",
        "title": "Roster",
        "source_type": "official_institutional",
        "authority_level": 4,
        "published_at": PUB,
        "available_at": PUB,
        "lineage_event_id": "roster_event",
        "claims": claims,
    }


def _roster_claims(actor_ids: list[str]) -> list[dict[str, Any]]:
    claims = [
        {
            "id": f"r_{a}",
            "proposition": f"role: {a} participates",
            "normalized_value": "member",
            "entities": [a],
        }
        for a in actor_ids
    ]
    claims.append(
        {
            "id": "ctx",
            "proposition": "context: baseline conditions",
            "normalized_value": "calm",
            "entities": ["world"],
        }
    )
    return claims


def _corpus(
    world_spec: dict[str, Any],
    *,
    actor_ids: list[str],
    expected_participants: int | None,
    target: str,
    uncertainties: list[dict[str, Any]] | None = None,
    extra_claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    world_spec.setdefault("subject_entity", world_spec.get("title", "subject"))
    world_spec.setdefault("resolution_units", "the outcome")
    claims = _roster_claims(actor_ids) + list(extra_claims or [])
    return {
        "reality": {
            "as_of": AS_OF,
            "horizon": HORIZON,
            "subject_entity": world_spec["subject_entity"],
            "resolution_units": world_spec["resolution_units"],
            "target_outcome": target,
            "expected_participants": expected_participants,
            "authoritative_sources": ["roster"],
        },
        "world_spec": world_spec,
        "uncertainties": uncertainties or [],
        "world_facts": [
            {"text": "baseline conditions hold", "evidence_claim_ids": ["ctx"], "available_at": PUB}
        ],
        "required_reality_facts": [
            {
                "key": "roster",
                "description": "participant roster",
                "evidence_claim_ids": [f"r_{a}" for a in actor_ids],
            }
        ],
        "sources": [_source(claims)],
        "outcome": None,
    }


def _person(aid: str, authority: list[str], **attrs: Any) -> dict[str, Any]:
    ent: dict[str, Any] = {
        "entity_id": aid,
        "name": aid.upper(),
        "kind": "person",
        "is_actor": True,
        "role": "participant",
        "authority": authority,
        "evidence_claim_ids": [f"r_{aid}"],
    }
    if attrs:
        ent["attributes"] = attrs
    return ent


# ---------------------------------------------------------------------------
# 1. Committee decision (voting is just one compiled world, not the architecture)
# ---------------------------------------------------------------------------


def committee_world(priors: dict[str, str], *, target: str = "hold") -> dict[str, Any]:
    aids = list(priors)
    actors = []
    for aid, prior in priors.items():
        actors.append(
            {
                "entity_id": aid,
                "reasoning": f"{aid} prior was {prior}",
                "memory_seeds": [
                    {
                        "content": f"My prior position was {prior}.",
                        "kind": "episodic",
                        "importance": 0.7,
                        "evidence_claim_ids": [f"r_{aid}"],
                    }
                ],
                "policy": {
                    "default_action_id": "record_position",
                    "default_params": {"position": prior},
                    "rules": [
                        {
                            "when_field": "shock",
                            "op": "above",
                            "value": 0.5,
                            "action_id": "record_position",
                            "params": {"position": "cut"},
                        }
                    ],
                },
            }
        )
    spec = {
        "title": "policy council",
        "subject_entity": "the policy rate",
        "resolution_units": f"a unanimous {target} by {len(aids)} seats",
        "entities": [_person(a, ["decide"]) for a in aids],
        "actors": actors,
        "fields": [{"field_id": "shock", "value_type": "number", "initial": 0.0}],
        "actions": [
            {
                "action_id": "record_position",
                "meaning": "record your final position",
                "eligible_actors": ["*"],
                "required_authority": ["decide"],
                "parameters": [
                    {"name": "position", "type": "option", "choices": ["cut", "hold", "hike"]}
                ],
                "stages": ["decide"],
                "visibility": "private",
                "effects": [
                    {
                        "op": "append_record",
                        "collection": "votes",
                        "key": "$actor",
                        "value": "$param.position",
                        "visibility": "private",
                    }
                ],
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "briefing",
                    "stage": "brief",
                    "advance_seconds": 60,
                    "effects": [
                        {
                            "op": "deliver_information",
                            "text": "The decision is open.",
                            "visibility": "public",
                        }
                    ],
                },
                {
                    "node_id": "decision",
                    "stage": "decide",
                    "participants": ["*"],
                    "action_ids": ["record_position"],
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "equals",
                "args": [
                    {
                        "op": "count",
                        "args": [
                            "votes",
                            {"op": "equals", "args": [{"op": "item", "args": ["value"]}, target]},
                        ],
                    },
                    len(aids),
                ],
            },
            "unresolved_when": {
                "op": "less_than",
                "args": [{"op": "count", "args": ["votes"]}, len(aids)],
            },
            "description": f"YES iff all {len(aids)} recorded votes equal {target}",
        },
    }
    uncertainties = [
        {
            "variable": "shock",
            "why_unknown": "a future shock could force a cut",
            "reversal_capable": True,
            "constraining_evidence_ids": ["ctx"],
            "outcomes": [
                {
                    "value": "none",
                    "weight": 0.8,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["shock", 0.0]],
                    "description": "calm",
                },
                {
                    "value": "big",
                    "weight": 0.2,
                    "provenance": "explicit_model_distribution",
                    "field_effects": [["shock", 0.6]],
                    "description": "a decisive shock",
                },
            ],
        }
    ]
    return _corpus(
        spec,
        actor_ids=aids,
        expected_participants=len(aids),
        target=target,
        uncertainties=uncertainties,
    )


# ---------------------------------------------------------------------------
# 2. Individual response (a single person deciding whether to reply)
# ---------------------------------------------------------------------------


def individual_response_world() -> dict[str, Any]:
    spec = {
        "title": "reply decision",
        "subject_entity": "the director",
        "resolution_units": "a reply delivered before the deadline",
        "entities": [_person("director", ["reply"])],
        "actors": [
            {
                "entity_id": "director",
                "reasoning": "director triages by urgency",
                "memory_seeds": [
                    {
                        "content": "I received a message needing a decision.",
                        "kind": "episodic",
                        "importance": 0.6,
                        "evidence_claim_ids": ["r_director"],
                    }
                ],
                "policy": {
                    "default_action_id": "",
                    "rules": [
                        {
                            "when_field": "urgency",
                            "op": "above",
                            "value": 0.5,
                            "action_id": "send_reply",
                            "params": {},
                        }
                    ],
                },
            }
        ],
        "fields": [{"field_id": "urgency", "value_type": "number", "initial": 0.0}],
        "actions": [
            {
                "action_id": "send_reply",
                "meaning": "reply to the incoming message",
                "eligible_actors": ["director"],
                "required_authority": ["reply"],
                "stages": ["decide"],
                "visibility": "public",
                "effects": [
                    {
                        "op": "append_record",
                        "collection": "replies",
                        "key": "$actor",
                        "value": "sent",
                    },
                    {
                        "op": "create_event",
                        "event_type": "reply_delivered",
                        "text": "director replied",
                        "visibility": "public",
                    },
                ],
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "incoming",
                    "stage": "brief",
                    "effects": [
                        {
                            "op": "deliver_information",
                            "text": "A message arrived.",
                            "to": ["director"],
                        }
                    ],
                },
                {
                    "node_id": "triage",
                    "stage": "decide",
                    "participants": ["director"],
                    "action_ids": ["send_reply"],
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "greater_or_equal",
                "args": [{"op": "event_count", "args": ["reply_delivered"]}, 1],
            },
            "description": "YES iff a reply was delivered before the horizon",
        },
    }
    uncertainties = [
        {
            "variable": "urgency",
            "why_unknown": "whether the message reads as urgent is unknown",
            "reversal_capable": True,
            "constraining_evidence_ids": ["ctx"],
            "outcomes": [
                {
                    "value": "low",
                    "weight": 0.4,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["urgency", 0.0]],
                    "description": "reads as routine",
                },
                {
                    "value": "high",
                    "weight": 0.6,
                    "provenance": "explicit_model_distribution",
                    "field_effects": [["urgency", 0.8]],
                    "description": "reads as urgent",
                },
            ],
        }
    ]
    return _corpus(
        spec,
        actor_ids=["director"],
        expected_participants=1,
        target="a reply is delivered",
        uncertainties=uncertainties,
    )


# ---------------------------------------------------------------------------
# 3. Negotiation (offers/concessions -> agreement)
# ---------------------------------------------------------------------------


def negotiation_world() -> dict[str, Any]:
    def party(aid: str) -> dict[str, Any]:
        return {
            "entity_id": aid,
            "reasoning": f"{aid} concedes unless hostile",
            "memory_seeds": [
                {
                    "content": "We are far apart on price.",
                    "kind": "episodic",
                    "importance": 0.6,
                    "evidence_claim_ids": [f"r_{aid}"],
                }
            ],
            "policy": {
                "default_action_id": "concede",
                "rules": [
                    {
                        "when_field": "gap",
                        "op": "below",
                        "value": 0.001,
                        "action_id": "sign_deal",
                        "params": {},
                    },
                    {
                        "when_field": "hostility",
                        "op": "above",
                        "value": 0.5,
                        "action_id": "",
                        "params": {},
                    },
                ],
            },
        }

    spec = {
        "title": "price negotiation",
        "subject_entity": "the contract",
        "resolution_units": "a signed agreement before the deadline",
        "entities": [_person("buyer", ["offer"]), _person("seller", ["offer"])],
        "actors": [party("buyer"), party("seller")],
        "fields": [
            {"field_id": "gap", "value_type": "number", "initial": 20.0},
            {"field_id": "hostility", "value_type": "number", "initial": 0.0},
        ],
        "actions": [
            {
                "action_id": "concede",
                "meaning": "narrow the gap by conceding",
                "eligible_actors": ["*"],
                "required_authority": ["offer"],
                "stages": ["bargain"],
                "effects": [{"op": "adjust_field", "field": "gap", "delta": -6.0}],
                "evidence_claim_ids": ["ctx"],
            },
            {
                "action_id": "sign_deal",
                "meaning": "sign once the gap is closed",
                "eligible_actors": ["*"],
                "required_authority": ["offer"],
                "stages": ["bargain"],
                "preconditions": {
                    "op": "less_or_equal",
                    "args": [{"op": "field", "args": ["gap"]}, 0],
                },
                "effects": [
                    {
                        "op": "append_record",
                        "collection": "deal",
                        "key": "$actor",
                        "value": "signed",
                    }
                ],
                "evidence_claim_ids": ["ctx"],
            },
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "round1",
                    "stage": "bargain",
                    "participants": ["buyer", "seller"],
                    "action_ids": ["concede", "sign_deal"],
                    "rounds": 1,
                },
                {
                    "node_id": "round2",
                    "stage": "bargain",
                    "participants": ["buyer", "seller"],
                    "action_ids": ["concede", "sign_deal"],
                    "rounds": 1,
                },
                {
                    "node_id": "closing",
                    "stage": "bargain",
                    "participants": ["buyer", "seller"],
                    "action_ids": ["concede", "sign_deal"],
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {"op": "greater_or_equal", "args": [{"op": "count", "args": ["deal"]}, 1]},
            "description": "YES iff at least one party signed the deal",
        },
    }
    uncertainties = [
        {
            "variable": "hostility",
            "why_unknown": "whether talks turn hostile is unknown",
            "reversal_capable": True,
            "constraining_evidence_ids": ["ctx"],
            "outcomes": [
                {
                    "value": "cooperative",
                    "weight": 0.7,
                    "provenance": "explicit_model_distribution",
                    "field_effects": [["hostility", 0.0]],
                    "description": "talks stay cooperative",
                },
                {
                    "value": "hostile",
                    "weight": 0.3,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["hostility", 0.9]],
                    "description": "talks break down",
                },
            ],
        }
    ]
    return _corpus(
        spec,
        actor_ids=["buyer", "seller"],
        expected_participants=2,
        target="a deal is signed",
        uncertainties=uncertainties,
    )


# ---------------------------------------------------------------------------
# 4. Population behavior (weighted strata adoption; a weighted-sum terminal,
#    NOT a reused voting mechanism)
# ---------------------------------------------------------------------------


def population_world() -> dict[str, Any]:
    strata = {"core": (40, "adopt"), "leaners": (35, "reject"), "skeptics": (25, "reject")}
    entities = [_person(a, ["represent"], weight=w) for a, (w, _) in strata.items()]
    actors = []
    for a, (_w, base) in strata.items():
        rules = []
        if a == "leaners":
            rules = [
                {
                    "when_field": "appeal",
                    "op": "above",
                    "value": 0.5,
                    "action_id": "record_stance",
                    "params": {"stance": "adopt"},
                }
            ]
        actors.append(
            {
                "entity_id": a,
                "reasoning": f"{a} baseline stance is {base}",
                "memory_seeds": [
                    {
                        "content": f"My stratum tends to {base}.",
                        "kind": "semantic",
                        "importance": 0.6,
                        "evidence_claim_ids": [f"r_{a}"],
                    }
                ],
                "policy": {
                    "default_action_id": "record_stance",
                    "default_params": {"stance": base},
                    "rules": rules,
                },
            }
        )
    spec = {
        "title": "product adoption",
        "subject_entity": "the population",
        "resolution_units": "more than half the population, by weight, adopts",
        "entities": entities,
        "actors": actors,
        "fields": [{"field_id": "appeal", "value_type": "number", "initial": 0.0}],
        "actions": [
            {
                "action_id": "record_stance",
                "meaning": "the stratum settles on a stance",
                "eligible_actors": ["*"],
                "required_authority": ["represent"],
                "stages": ["settle"],
                "parameters": [
                    {"name": "stance", "type": "option", "choices": ["adopt", "reject"]}
                ],
                "effects": [
                    {
                        "op": "append_record",
                        "collection": "stances",
                        "key": "$actor",
                        "value": "$param.stance",
                        "extra": {"weight": "$self.weight"},
                    }
                ],
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "settle",
                    "stage": "settle",
                    "participants": ["*"],
                    "action_ids": ["record_stance"],
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "greater_than",
                "args": [
                    {
                        "op": "sum",
                        "args": [
                            "stances",
                            "weight",
                            {"op": "equals", "args": [{"op": "item", "args": ["value"]}, "adopt"]},
                        ],
                    },
                    50,
                ],
            },
            "description": "YES iff the adopting weight exceeds 50 of 100",
        },
    }
    uncertainties = [
        {
            "variable": "appeal",
            "why_unknown": "how the launch lands with leaners is unknown",
            "reversal_capable": True,
            "constraining_evidence_ids": ["ctx"],
            "outcomes": [
                {
                    "value": "weak",
                    "weight": 0.5,
                    "provenance": "market_or_survey_distribution",
                    "field_effects": [["appeal", 0.0]],
                    "description": "weak reception",
                },
                {
                    "value": "strong",
                    "weight": 0.5,
                    "provenance": "market_or_survey_distribution",
                    "field_effects": [["appeal", 0.8]],
                    "description": "strong reception",
                },
            ],
        }
    ]
    return _corpus(
        spec,
        actor_ids=list(strata),
        expected_participants=3,
        target="adopting weight > 50",
        uncertainties=uncertainties,
    )


# ---------------------------------------------------------------------------
# 5. Geopolitical / organizational process (state B signals, state A decides)
# ---------------------------------------------------------------------------


def geopolitical_world() -> dict[str, Any]:
    spec = {
        "title": "sanctions decision",
        "subject_entity": "state A",
        "resolution_units": "state A imposes sanctions before the horizon",
        "entities": [
            _person("state_a", ["decree"]),
            _person("state_b", ["signal"]),
            {"entity_id": "sanctions_bill", "name": "Sanctions Bill", "kind": "document"},
        ],
        "actors": [
            {
                "entity_id": "state_b",
                "reasoning": "B warns to deter",
                "policy": {"default_action_id": "send_warning", "default_params": {}},
            },
            {
                "entity_id": "state_a",
                "reasoning": "A acts only under high provocation",
                "policy": {
                    "default_action_id": "",
                    "rules": [
                        {
                            "when_field": "provocation",
                            "op": "above",
                            "value": 0.5,
                            "action_id": "impose_sanctions",
                            "params": {},
                        }
                    ],
                },
            },
        ],
        "fields": [{"field_id": "provocation", "value_type": "number", "initial": 0.0}],
        "actions": [
            {
                "action_id": "send_warning",
                "meaning": "B warns A off",
                "eligible_actors": ["state_b"],
                "required_authority": ["signal"],
                "stages": ["react"],
                "effects": [
                    {"op": "deliver_information", "text": "Do not escalate.", "to": ["state_a"]}
                ],
                "evidence_claim_ids": ["ctx"],
            },
            {
                "action_id": "impose_sanctions",
                "meaning": "A imposes sanctions",
                "eligible_actors": ["state_a"],
                "required_authority": ["decree"],
                "stages": ["decide"],
                "effects": [
                    {
                        "op": "create_or_update_document",
                        "document": "sanctions_bill",
                        "fields": {"status": "enacted"},
                    },
                    {
                        "op": "create_event",
                        "event_type": "sanctions_imposed",
                        "text": "A imposed sanctions",
                        "visibility": "public",
                    },
                ],
                "evidence_claim_ids": ["ctx"],
            },
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "react",
                    "stage": "react",
                    "participants": ["state_b"],
                    "action_ids": ["send_warning"],
                    "rounds": 1,
                },
                {
                    "node_id": "decide",
                    "stage": "decide",
                    "participants": ["state_a"],
                    "action_ids": ["impose_sanctions"],
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "greater_or_equal",
                "args": [{"op": "event_count", "args": ["sanctions_imposed"]}, 1],
            },
            "description": "YES iff state A imposed sanctions",
        },
    }
    uncertainties = [
        {
            "variable": "provocation",
            "why_unknown": "the level of provocation is unknown",
            "reversal_capable": True,
            "constraining_evidence_ids": ["ctx"],
            "outcomes": [
                {
                    "value": "restrained",
                    "weight": 0.6,
                    "provenance": "explicit_model_distribution",
                    "field_effects": [["provocation", 0.0]],
                    "description": "no major provocation",
                },
                {
                    "value": "severe",
                    "weight": 0.4,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["provocation", 0.9]],
                    "description": "a severe provocation",
                },
            ],
        }
    ]
    return _corpus(
        spec,
        actor_ids=["state_a", "state_b"],
        expected_participants=2,
        target="sanctions imposed",
        uncertainties=uncertainties,
    )


# ---------------------------------------------------------------------------
# Unknown-action world: an action whose name appears NOWHERE in the source code.
# ---------------------------------------------------------------------------


def unknown_action_world(action_name: str = "zorptcast_the_glyph") -> dict[str, Any]:
    spec = {
        "title": "invented process",
        "subject_entity": "the archivist",
        "resolution_units": "the invented action was performed",
        "entities": [_person("archivist", ["inscribe"])],
        "actors": [
            {
                "entity_id": "archivist",
                "reasoning": "always performs the invented action",
                "policy": {"default_action_id": action_name, "default_params": {}},
            }
        ],
        "fields": [],
        "actions": [
            {
                "action_id": action_name,
                "meaning": "an action name absent from the source code",
                "eligible_actors": ["archivist"],
                "required_authority": ["inscribe"],
                "stages": ["act"],
                "effects": [
                    {
                        "op": "append_record",
                        "collection": "ledger",
                        "key": "$actor",
                        "value": "done",
                    }
                ],
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "act",
                    "stage": "act",
                    "participants": ["archivist"],
                    "action_ids": [action_name],
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "greater_or_equal",
                "args": [{"op": "count", "args": ["ledger"]}, 1],
            },
            "description": "YES iff the invented action was recorded",
        },
    }
    return _corpus(
        spec, actor_ids=["archivist"], expected_participants=1, target="invented action performed"
    )


# ---------------------------------------------------------------------------
# Novel-action worlds: the actor proposes an action NOT in its compiled menu.
# ---------------------------------------------------------------------------


def novel_action_world(*, mode: str) -> dict[str, Any]:
    """``mode`` in {"accept", "reject_authority", "unrepresentable"}.

    The single compiled action is ``noop`` (records nothing decisive). The actor's
    disposition instead proposes a NOVEL action. Whether it succeeds depends only on
    authority + feasibility + representability — never on the compiled menu.
    """

    authority = ["publish"]
    novel_effects = [
        {"op": "append_record", "collection": "ledger", "key": "$actor", "value": "novel_done"}
    ]
    novel: dict[str, Any] = {
        "description": "improvise an action the compiler did not anticipate",
        "target": "",
        "intended_effect": "record a decisive entry",
        "parameters": {},
    }
    if mode == "accept":
        novel["parameters"] = {"effects": novel_effects, "required_authority": ["publish"]}
    elif mode == "reject_authority":
        novel["parameters"] = {"effects": novel_effects, "required_authority": ["dictate"]}
    elif mode == "unrepresentable":
        novel["parameters"] = {}  # no safe effect mapping -> interpreter refuses
    else:  # pragma: no cover
        raise ValueError(mode)

    spec = {
        "title": "novel action world",
        "subject_entity": "the agent",
        "resolution_units": "a decisive ledger entry exists",
        "entities": [_person("agent", authority)],
        "actors": [
            {
                "entity_id": "agent",
                "reasoning": "prefers an action outside the compiled menu",
                "policy": {"default_novel": novel},
            }
        ],
        "fields": [],
        "actions": [
            {
                "action_id": "noop",
                "meaning": "a harmless placeholder action",
                "eligible_actors": ["agent"],
                "required_authority": ["publish"],
                "stages": ["act"],
                "effects": [{"op": "create_event", "event_type": "noop", "text": "nothing"}],
                "evidence_claim_ids": ["ctx"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "act",
                    "stage": "act",
                    "participants": ["agent"],
                    "action_ids": ["noop"],
                    "allow_novel": True,
                    "rounds": 1,
                },
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "greater_or_equal",
                "args": [{"op": "count", "args": ["ledger"]}, 1],
            },
            "description": "YES iff a decisive ledger entry exists",
        },
    }
    return _corpus(
        spec, actor_ids=["agent"], expected_participants=1, target="decisive ledger entry exists"
    )


# ---------------------------------------------------------------------------
# Coverage fixtures: evidence names N people, the compiled world represents M
# ---------------------------------------------------------------------------

BODY = "the Governing Board"


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def named_body_world(
    people: list[str],
    represented: list[str],
    *,
    rule_kind: str = "majority",
    extra_claims: list[dict[str, Any]] | None = None,
    extra_required: list[dict[str, Any]] | None = None,
    body: str = BODY,
    org_is_actor: bool = False,
    doc_holder_claims: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A decision world whose *evidence* names ``people`` but whose compiled WorldSpec
    represents only ``represented`` — the exact shape of the motivating coverage
    failure. The compiled terminal states its own decision procedure in words, so a
    binding rule found in evidence (e.g. unanimity) that the world does not model is
    detectably absent.

    ``org_is_actor`` compiles the organization itself as the single acting entity (a
    world with no named individuals at all).
    """

    claims: list[dict[str, Any]] = [
        {
            "id": f"r_{_slug(p)}",
            "proposition": f"{p} is a voting member of {body} and voted to hold at the last meeting.",
            "normalized_value": "member",
            "entities": [p],
        }
        for p in people
    ]
    claims.append(
        {
            "id": "org_claim",
            "proposition": f"{body} is the body that decides the measure.",
            "normalized_value": "deciding_body",
            "entities": [body],
        }
    )
    claims += list(extra_claims or [])

    n = max(1, len(represented))
    threshold = n if rule_kind == "unanimous" else n // 2 + 1
    procedure = "unanimous consent of all members" if rule_kind == "unanimous" else "a majority"

    if org_is_actor:
        entities = [
            {
                "entity_id": "body",
                "name": body,
                "kind": "organization",
                "is_actor": True,
                "role": "deciding body",
                "authority": ["decide"],
                "evidence_claim_ids": ["org_claim"],
            }
        ]
        actor_ids = ["body"]
    else:
        entities = [
            {
                "entity_id": _slug(p),
                "name": p,
                "kind": "person",
                "is_actor": True,
                "role": "member",
                "authority": ["decide"],
                "evidence_claim_ids": [f"r_{_slug(p)}"],
            }
            for p in represented
        ]
        entities.append(
            {
                "entity_id": "body",
                "name": body,
                "kind": "organization",
                "is_actor": False,
                "evidence_claim_ids": ["org_claim"],
            }
        )
        actor_ids = [_slug(p) for p in represented]

    actors = []
    for i, aid in enumerate(actor_ids):
        seeds = [
            {
                "content": "My prior position was hold.",
                "kind": "episodic",
                "importance": 0.7,
                "evidence_claim_ids": [entities[i]["evidence_claim_ids"][0]],
            }
        ]
        if i == 0 and doc_holder_claims:
            seeds.append(
                {
                    "content": "I have read the staff briefing.",
                    "kind": "episodic",
                    "importance": 0.8,
                    "evidence_claim_ids": list(doc_holder_claims),
                }
            )
        actors.append(
            {
                "entity_id": aid,
                "reasoning": "prior position was hold",
                "memory_seeds": seeds,
                "policy": {
                    "default_action_id": "record_position",
                    "default_params": {"position": "hold"},
                },
            }
        )

    spec = {
        "title": body,
        "subject_entity": "the measure",
        "resolution_units": f"{rule_kind} of {n}",
        "entities": entities,
        "actors": actors,
        "fields": [],
        "actions": [
            {
                "action_id": "record_position",
                "meaning": f"record a position in the decision, which carries by {procedure}",
                "eligible_actors": ["*"],
                "required_authority": ["decide"],
                "parameters": [{"name": "position", "type": "option", "choices": ["cut", "hold"]}],
                "stages": ["decide"],
                "effects": [
                    {
                        "op": "append_record",
                        "collection": "votes",
                        "key": "$actor",
                        "value": "$param.position",
                    }
                ],
                "evidence_claim_ids": ["org_claim"],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "decision",
                    "stage": "decide",
                    "description": f"{body} decides the measure",
                    "participants": ["*"],
                    "action_ids": ["record_position"],
                }
            ]
        },
        "terminal": {
            "yes_when": {
                "op": "greater_or_equal",
                "args": [
                    {
                        "op": "count",
                        "args": [
                            "votes",
                            {"op": "equals", "args": [{"op": "item", "args": ["value"]}, "hold"]},
                        ],
                    },
                    threshold,
                ],
            },
            "description": (
                f"YES when the measure carries by {procedure}: at least {threshold} of {n} "
                "recorded positions are hold"
            ),
        },
    }

    required = [
        {
            "key": "roster",
            "description": "verified roster",
            "evidence_claim_ids": [e["evidence_claim_ids"][0] for e in entities[: len(actor_ids)]],
        }
    ]
    required += list(extra_required or [])

    corpus = _corpus(
        spec,
        actor_ids=actor_ids,
        expected_participants=len(actor_ids),
        target=f"the measure carries by {procedure}",
    )
    # Replace the generic roster claims with this fixture's named-people evidence.
    corpus["sources"] = [_source(claims)]
    corpus["required_reality_facts"] = required
    corpus["world_facts"] = []
    return corpus


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def dup(corpus: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(corpus)


def run_corpus(
    corpus: dict[str, Any], *, seed: int = 0, max_branches: int = 24, gateway: Any = None
):
    """Run the full pipeline on a corpus dict, using the corpus's own cutoff/horizon
    (so a 2026 Banxico corpus and a 2024 synthetic corpus both work). Returns (result, ctx)."""

    backend = MockResearchBackend(data=corpus)
    bundle = backend.research("q", datetime.fromisoformat(AS_OF), datetime.fromisoformat(HORIZON))
    as_of = bundle.as_of or datetime.fromisoformat(AS_OF)
    config = ForecastConfig(
        gateway=gateway or DeterministicGateway(),
        research_backend=backend,
        seed=seed,
        max_branches=max_branches,
    )
    return run_forecast("q", as_of, bundle.horizon, config)
