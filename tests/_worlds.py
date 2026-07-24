"""Authored compiled worlds for tests.

These are *compiled data*, exactly the shape the live compiler emits. They exist so
tests can drive the runtime deterministically. They are structurally unlike each other
on purpose: if the runtime needed to know what kind of question it was executing, these
would not both run through it unchanged.

Nothing here is imported by ``src/sworldmodel``.
"""

from __future__ import annotations

from typing import Any

AS_OF = "2026-05-14T23:59:59+00:00"
HORIZON = "2026-06-25T23:59:59+00:00"


def _claim(cid: str, proposition: str, value: Any = True) -> dict[str, Any]:
    return {
        "id": cid,
        "proposition": proposition,
        "value": value,
        "supporting_excerpt": proposition,
    }


# ---------------------------------------------------------------------------
# World A — several people with authority act on a scheduled date and their acts
# are counted. Nothing in the runtime knows this shape; it is entirely data.
# ---------------------------------------------------------------------------


def scheduled_multiparty_world(*, members: int = 5, threshold: int = 5) -> dict[str, Any]:
    ids = [f"member_{i}" for i in range(members)]
    entities = [
        {
            "entity_id": mid,
            "name": f"Member {i}",
            "kind": "person",
            "is_actor": True,
            "role": "member",
            "authority": ["record_position"],
            "representation_scale": "individual",
            "evidence_claim_ids": [f"c_{mid}"],
        }
        for i, mid in enumerate(ids)
    ]
    actors = [
        {
            "entity_id": mid,
            "reasoning": "leans toward continuity given the record",
            "goals": ["reach a defensible outcome"],
            "memory_seeds": [
                {
                    "content": f"I, Member {i}, recorded 'hold' at the previous session.",
                    "kind": "episodic",
                    "importance": 0.9,
                    "evidence_claim_ids": [f"c_{mid}"],
                }
            ],
        }
        for i, mid in enumerate(ids)
    ]
    return {
        "reality": {
            "as_of": AS_OF,
            "horizon": HORIZON,
            "subject_entity": "the reference level",
            "resolution_units": "recorded positions",
            "target_outcome": "every member records hold",
            "expected_participants": members,
        },
        "claims": [_claim(f"c_{mid}", f"{mid} recorded hold previously") for mid in ids]
        + [_claim("c_session", "the session is scheduled for 2026-06-25")],
        "world_spec": {
            "title": "scheduled multiparty determination",
            "subject_entity": "the reference level",
            "resolution_units": "recorded positions",
            "entities": entities,
            "actors": actors,
            "fields": [
                {"field_id": "external_signal", "value_type": "number", "initial": 3.5},
            ],
            "actions": [
                {
                    "action_id": "record_position",
                    "meaning": "record your position for the session",
                    "eligible_actors": ["role:member"],
                    "required_authority": ["record_position"],
                    "parameters": [
                        {
                            "name": "position",
                            "type": "option",
                            "required": True,
                            "choices": ["hold", "change"],
                        }
                    ],
                    "stages": ["session"],
                    "visibility": "public",
                    "effects": [
                        {
                            "op": "append_record",
                            "collection": "positions",
                            "key": "$actor",
                            "value": "$param.position",
                        }
                    ],
                    "evidence_claim_ids": ["c_session"],
                },
                {
                    "action_id": "circulate_note",
                    "meaning": "circulate a note to the others before the session",
                    "eligible_actors": ["role:member"],
                    "parameters": [{"name": "text", "type": "string", "required": True}],
                    "stages": ["preparation"],
                    "visibility": "public",
                    "effects": [
                        {"op": "deliver_information", "text": "$param.text"},
                        {
                            "op": "create_event",
                            "event_type": "note_circulated",
                            "text": "$param.text",
                        },
                    ],
                },
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "preparation",
                        "stage": "preparation",
                        "at": "2026-06-01T09:00:00+00:00",
                        "description": "material circulates",
                        "participants": ["*"],
                        "action_ids": ["circulate_note"],
                        "next_nodes": ["session"],
                    },
                    {
                        "node_id": "session",
                        "stage": "session",
                        "at": "2026-06-25T09:00:00+00:00",
                        "description": "the session at which positions are recorded",
                        "participants": ["*"],
                        "action_ids": ["record_position"],
                    },
                ]
            },
            "external_processes": [
                {
                    "process_id": "signal_release",
                    "description": "the scheduled external measurement",
                    "occurrences": [
                        {
                            "at": "2026-06-09T12:00:00+00:00",
                            "description": "measurement published",
                            "effects": [{"op": "release_data", "fields": {"external_signal": 3.5}}],
                        }
                    ],
                    "evidence_claim_ids": ["c_session"],
                }
            ],
            "wake_rules": [
                {
                    "rule_id": "a_member_speaking_matters",
                    "wakes": ["role:member"],
                    "reason": "another member has circulated something before the session",
                    "on_event_type": "note_circulated",
                },
                {
                    "rule_id": "signal_matters",
                    "wakes": ["role:member"],
                    "reason": "the published measurement bears on your position",
                    "on_field_change": "external_signal",
                },
            ],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [
                        {
                            "op": "count",
                            "args": [
                                "positions",
                                {
                                    "op": "equals",
                                    "args": [{"op": "item", "args": ["value"]}, "hold"],
                                },
                            ],
                        },
                        threshold,
                    ],
                },
                "unresolved_when": {
                    "op": "less_than",
                    "args": [{"op": "count", "args": ["positions"]}, members],
                },
                "description": f"YES when at least {threshold} of {members} positions are hold",
            },
        },
    }


# ---------------------------------------------------------------------------
# World B — one person receives something and may answer. No roles, no counting,
# no threshold, a completely different terminal shape.
# ---------------------------------------------------------------------------


def single_response_world(*, reply_deadline: str = "2026-06-20T00:00:00+00:00") -> dict[str, Any]:
    return {
        "reality": {
            "as_of": AS_OF,
            "horizon": HORIZON,
            "subject_entity": "the recipient",
            "resolution_units": "a reply",
            "target_outcome": "the recipient replies affirmatively",
            "expected_participants": 1,
        },
        "claims": [
            _claim("c_recipient", "the recipient's role and inbox are documented"),
            _claim("c_request", "the request was sent on 2026-05-20"),
        ],
        "world_spec": {
            "title": "single response",
            "subject_entity": "the recipient",
            "resolution_units": "a reply",
            "entities": [
                {
                    "entity_id": "recipient",
                    "name": "The Recipient",
                    "kind": "person",
                    "is_actor": True,
                    "role": "recipient",
                    "authority": ["reply"],
                    "representation_scale": "individual",
                    "evidence_claim_ids": ["c_recipient"],
                }
            ],
            "actors": [
                {
                    "entity_id": "recipient",
                    "reasoning": "responds to requests that name a deadline",
                    "memory_seeds": [
                        {
                            "content": "I answered a similar request last quarter.",
                            "kind": "episodic",
                            "importance": 0.7,
                            "evidence_claim_ids": ["c_recipient"],
                        }
                    ],
                }
            ],
            "fields": [{"field_id": "reply_sent", "value_type": "bool", "initial": False}],
            "actions": [
                {
                    "action_id": "send_reply",
                    "meaning": "send a reply",
                    "eligible_actors": ["recipient"],
                    "required_authority": ["reply"],
                    "parameters": [
                        {
                            "name": "answer",
                            "type": "option",
                            "required": True,
                            "choices": ["yes", "no"],
                        }
                    ],
                    "duration_seconds": 1800,
                    "delivery_delay_seconds": 60,
                    "visibility": "public",
                    "effects": [
                        {"op": "set_field", "field": "reply_sent", "value": True},
                        {
                            "op": "append_record",
                            "collection": "replies",
                            "key": "$actor",
                            "value": "$param.answer",
                        },
                    ],
                    "evidence_claim_ids": ["c_request"],
                }
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "request_arrives",
                        "stage": "open",
                        "at": "2026-05-20T10:00:00+00:00",
                        "description": "the request lands",
                        "effects": [
                            {
                                "op": "deliver_information",
                                "to": ["recipient"],
                                "text": "A request needing an answer has arrived.",
                            }
                        ],
                        "participants": ["recipient"],
                        "deadline": reply_deadline,
                    }
                ]
            },
            "wake_rules": [],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [
                        {
                            "op": "count",
                            "args": [
                                "replies",
                                {
                                    "op": "equals",
                                    "args": [{"op": "item", "args": ["value"]}, "yes"],
                                },
                            ],
                        },
                        1,
                    ],
                },
                "unresolved_when": {"op": "const", "args": [False]},
                "description": "YES when the recipient replies yes",
            },
        },
    }
