"""Test doubles for the live path: mocked HTTP + a prompt-reading fake LLM.

The fake LLM is NOT the DeterministicGateway (which is banned from production). It is a
mocked *provider*: it reads its prompt — the same rendered context a real model sees —
and returns realistic JSON, including citing the evidence claim ids that appear in the
prompt, and compiling a full universal WorldSpec. Combined with FakeTransport (canned
RSS + pages), this exercises the entire production code path (research loop,
verification, universal world compilation, event engine) with only the socket mocked.
"""

from __future__ import annotations

import re
import urllib.parse
from datetime import datetime

from sworldmodel.gateway import GatewayRequest, GatewayResponse, ModelGateway
from sworldmodel.http import FakeTransport, html_response
from sworldmodel.ids import canonical_json, prompt_hash

AS_OF = datetime.fromisoformat("2027-01-15T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2027-02-15T00:00:00+00:00")

ROSTER_URL = "https://widgetboard.example/board"
DECISION_URL = "https://news.example/widget-vote-2027-01-10"
OUTCOME_URL = "https://news.example/widget-final-2027-02-15"

ROSTER_PAGE = """[[ROSTER]] Widget Standards Board.
<meta property="article:published_time" content="2026-12-01T00:00:00+00:00">
The Widget Standards Board has three voting members: Ada Lovelace who serves as Chair,
Ben Carter, and Cara Diaz. Decisions are taken by a majority of the three members.
"""

DECISION_PAGE = """[[DECISION]] Widget Board recommends adoption.
<meta property="article:published_time" content="2027-01-10T00:00:00+00:00">
On January 10 2027 the Widget Standards Board voted to recommend adoption of the new
standard. Ada Lovelace supported adoption and Ben Carter supported adoption, while Cara
Diaz preferred to defer. The Board signaled that adoption is the expected outcome.
"""

OUTCOME_PAGE = """[[OUTCOME]] Widget standard adopted unanimously.
<meta property="article:published_time" content="2027-02-15T00:00:00+00:00">
On February 15 2027 the Board unanimously adopted the standard in a 3-0 vote.
"""


def _rss(links: list[str]) -> str:
    items = "".join(
        f"<item><title>Item {i}</title><link>{u}</link>"
        f"<source>example</source><pubDate>Tue, 12 Jan 2027 00:00:00 GMT</pubDate>"
        f"<description>d</description></item>"
        for i, u in enumerate(links)
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel>{items}</channel></rss>'


def _ddg(links: list[str]) -> str:
    anchors = "".join(
        f'<a rel="nofollow" href="//duckduckgo.com/l/?uddg={urllib.parse.quote(u, safe="")}">r</a>'
        for u in links
    )
    return f"<html><body>{anchors}</body></html>"


def widget_transport(*, include_outcome: bool = True) -> FakeTransport:
    t = FakeTransport()
    links = [ROSTER_URL, DECISION_URL] + ([OUTCOME_URL] if include_outcome else [])
    t.add(lambda u: "news.google.com/rss" in u, lambda m, u, b: html_response(u, _rss(links)))
    t.add(lambda u: "duckduckgo.com" in u, lambda m, u, b: html_response(u, _ddg(links)))
    t.add_url(ROSTER_URL, html_response(ROSTER_URL, ROSTER_PAGE))
    t.add_url(DECISION_URL, html_response(DECISION_URL, DECISION_PAGE))
    t.add_url(OUTCOME_URL, html_response(OUTCOME_URL, OUTCOME_PAGE))
    return t


_ID_RE = re.compile(r"\b[a-z]+-[0-9a-f]{12}\b")


def _id_for(prompt: str, needle: str) -> list[str]:
    """The claim id whose evidence line contains ``needle`` (the compiler cites real ids)."""

    for line in prompt.splitlines():
        if needle in line:
            m = re.match(r"\s*([a-z]+-[0-9a-f]{12})\s*\|", line)
            if m:
                return [m.group(1)]
    return []


class FakeLLM(ModelGateway):
    """A prompt-reading mock provider. is_live stays False (it is a test double)."""

    is_live = False

    @property
    def model_id(self) -> str:
        return "fake-llm"

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        handler = {
            "research_plan": self._plan,
            "followup_queries": lambda p: {"queries": []},
            "extract_claims": self._extract,
            "compile_world_spec": self._world_spec,
            "actor_decision": self._decision,
            "interpret_novel": lambda p: {
                "representable": False,
                "reason": "no novel action needed",
            },
            "reflect": lambda p: {"beliefs_update": [], "new_memories": [], "plan_note": ""},
        }[request.task_kind]
        data = handler(request.prompt)
        raw = canonical_json(data)
        return GatewayResponse(
            task_kind=request.task_kind,
            data=data,
            raw_text=raw,
            model=self.model_id,
            params={},
            seed=request.seed,
            prompt_hash=prompt_hash(request.prompt),
            tokens_in=len(request.prompt) // 4,
            tokens_out=len(raw) // 4,
            latency_ms=1,
        )

    def _plan(self, prompt: str) -> dict:
        return {
            "process_summary": "the Widget Standards Board decides whether to adopt a standard",
            "resolution_event": "the board's next standard vote",
            "deadline": HORIZON.isoformat(),
            "authoritative_sources": ["Widget Standards Board"],
            "decision_makers": ["Ada Lovelace", "Ben Carter", "Cara Diaz"],
            "rules": ["majority of three"],
            "prior_actions": ["the January 10 recommendation vote"],
            "scheduled_events": [],
            "causal_drivers": ["public comment"],
            "required_facts": ["roster of the board", "decision rule of the board"],
            "initial_queries": ["Widget Standards Board members", "Widget Standards Board vote"],
            "official_domains": ["widgetboard.example"],
        }

    def _extract(self, prompt: str) -> dict:
        if "[[ROSTER]]" in prompt:
            return {
                "claims": [
                    _c(
                        "roster: Ada Lovelace who serves as Chair",
                        "chair",
                        ["Ada Lovelace"],
                        "Ada Lovelace who serves as Chair",
                        4,
                    ),
                    _c(
                        "roster: Ben Carter is a voting member",
                        "member",
                        ["Ben Carter"],
                        "Ben Carter",
                        4,
                    ),
                    _c(
                        "roster: Cara Diaz is a voting member",
                        "member",
                        ["Cara Diaz"],
                        "Cara Diaz",
                        4,
                    ),
                    _c(
                        "rule: decisions are taken by a majority of the three members",
                        "majority_3",
                        ["Widget Standards Board"],
                        "majority of the three members",
                        4,
                    ),
                ]
            }
        if "[[DECISION]]" in prompt:
            return {
                "claims": [
                    _c(
                        "vote: Ada Lovelace supported adoption",
                        "adopt",
                        ["Ada Lovelace"],
                        "Ada Lovelace supported adoption",
                        3,
                    ),
                    _c(
                        "vote: Ben Carter supported adoption",
                        "adopt",
                        ["Ben Carter"],
                        "Ben Carter supported adoption",
                        3,
                    ),
                    _c(
                        "vote: Cara Diaz preferred to defer",
                        "defer",
                        ["Cara Diaz"],
                        "Cara Diaz preferred to defer",
                        3,
                    ),
                    _c(
                        "guidance: adoption is the expected outcome",
                        "adopt",
                        ["Widget Standards Board"],
                        "adoption is the expected outcome",
                        3,
                    ),
                    # An UNSUPPORTED claim (excerpt not in the page) — must be rejected.
                    _c(
                        "vote: the board unanimously rejected everything",
                        "reject",
                        ["Widget Standards Board"],
                        "the board rejected all proposals outright",
                        3,
                    ),
                ]
            }
        if "[[OUTCOME]]" in prompt:
            return {
                "claims": [
                    _c(
                        "outcome: the standard was adopted",
                        "adopt",
                        ["Widget Standards Board"],
                        "unanimously adopted the standard",
                        3,
                    )
                ]
            }
        return {"claims": []}

    def _world_spec(self, prompt: str) -> dict:
        ids = _ID_RE.findall(prompt)
        members = [
            ("ada_lovelace", "Ada Lovelace", "Chair", "adopt"),
            ("ben_carter", "Ben Carter", "Member", "adopt"),
            ("cara_diaz", "Cara Diaz", "Member", "defer"),
        ]
        rule_ids = _id_for(prompt, "majority of the three members")
        guidance_ids = _id_for(prompt, "adoption is the expected outcome")
        entities = [
            {
                "entity_id": aid,
                "name": name,
                "kind": "person",
                "is_actor": True,
                "role": role,
                "authority": ["decide"],
                "evidence_claim_ids": ids[:2],
            }
            for aid, name, role, _ in members
        ]
        entities.append(
            {
                "entity_id": "widget_standards_board",
                "name": "Widget Standards Board",
                "kind": "organization",
                "is_actor": False,
                "evidence_claim_ids": rule_ids + guidance_ids,
            }
        )
        actors = [
            {
                "entity_id": aid,
                "reasoning": f"prior stance {prior}",
                "memory_seeds": [
                    {
                        "content": f"My prior position was {prior}.",
                        "kind": "episodic",
                        "importance": 0.7,
                        "evidence_claim_ids": ids[:1],
                    }
                ],
                "policy": {
                    "default_action_id": "record_position",
                    "default_params": {"position": "adopt"},
                    "rules": [
                        {
                            "when_field": "opposition",
                            "op": "above",
                            "value": 0.5,
                            "action_id": "record_position",
                            "params": {"position": "reject"},
                        }
                    ],
                },
            }
            for aid, _, _, prior in members
        ]
        return {
            "subject_entity": "Widget standard",
            "resolution_units": "unanimity of three seats",
            "target_outcome": "a unanimous 3-0 adoption at the next meeting",
            "expected_participants": 3,
            "world_spec": {
                "title": "Widget Standards Board",
                "subject_entity": "Widget standard",
                "resolution_units": "unanimity of three seats",
                "entities": entities,
                "actors": actors,
                "fields": [{"field_id": "opposition", "value_type": "number", "initial": 0.0}],
                "actions": [
                    {
                        "action_id": "record_position",
                        "meaning": ("record the member's vote in the Board's majority decision"),
                        "eligible_actors": ["*"],
                        "required_authority": ["decide"],
                        "stages": ["decision"],
                        "visibility": "private",
                        "parameters": [
                            {
                                "name": "position",
                                "type": "option",
                                "choices": ["reject", "defer", "adopt"],
                            }
                        ],
                        "effects": [
                            {
                                "op": "append_record",
                                "collection": "votes",
                                "key": "$actor",
                                "value": "$param.position",
                                "visibility": "private",
                            }
                        ],
                        # The action exists under the Board's majority rule and is
                        # anchored by its published guidance: cite both.
                        "evidence_claim_ids": (rule_ids + guidance_ids) or ids[:1],
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
                                    "text": "The adoption decision is open.",
                                    "visibility": "public",
                                }
                            ],
                        },
                        {
                            "node_id": "decision",
                            "stage": "decision",
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
                                    {
                                        "op": "equals",
                                        "args": [{"op": "item", "args": ["value"]}, "adopt"],
                                    },
                                ],
                            },
                            3,
                        ],
                    },
                    "unresolved_when": {
                        "op": "less_than",
                        "args": [{"op": "count", "args": ["votes"]}, 3],
                    },
                    "description": "YES iff all three recorded votes are adopt",
                },
            },
            "uncertainties": [
                {
                    "variable": "opposition",
                    "why_unknown": "future public opposition is unknown",
                    "reversal_capable": True,
                    "constraining_evidence_ids": ids[:1],
                    "outcomes": [
                        {
                            "value": "quiet",
                            "weight": 0.75,
                            "provenance": "symmetric_ignorance_assumption",
                            "field_effects": [["opposition", 0.0]],
                            "description": "no opposition",
                        },
                        {
                            "value": "loud",
                            "weight": 0.25,
                            "provenance": "explicit_model_distribution",
                            "field_effects": [["opposition", 0.6]],
                            "description": "opposition mobilizes",
                        },
                    ],
                }
            ],
            "world_facts": [
                {
                    "text": "The board recommended adoption.",
                    "evidence_claim_ids": ids[:1],
                    "epistemic_type": "observation",
                }
            ],
            "required_reality_facts": [
                {
                    "key": "roster",
                    "description": "three-seat roster",
                    "evidence_claim_ids": ids[:3],
                },
                {
                    "key": "decision_rule",
                    "description": "majority of three",
                    "evidence_claim_ids": ids[:1],
                },
            ],
        }

    def _decision(self, prompt: str) -> dict:
        opposition = self._opposition(prompt)
        position = "reject" if opposition > 0.5 else "adopt"
        return {
            "action_mode": "compiled_action",
            "compiled_action_id": "record_position",
            "params": {"position": position},
            "target": "",
            "reasoning": f"I record my position given opposition={opposition}.",
            "referenced_memory_ids": [],
            "referenced_observation_ids": [],
        }

    @staticmethod
    def _opposition(prompt: str) -> float:
        m = re.search(r'"opposition":\s*([0-9.]+)', prompt)
        return float(m.group(1)) if m else 0.0


def _c(prop: str, value: str, entities: list[str], excerpt: str, auth: int) -> dict:
    return {
        "proposition": prop,
        "normalized_value": value,
        "entities": entities,
        "epistemic_type": "observation",
        "supporting_excerpt": excerpt,
        "authority_hint": auth,
    }
