"""Test doubles for the live path: mocked HTTP + a prompt-reading fake LLM.

The fake LLM is NOT the DeterministicGateway (which is banned from production). It is
a mocked *provider*: it reads its prompt — the same rendered context a real model
sees — and returns realistic JSON, including citing the evidence claim ids that appear
in the prompt. Combined with FakeTransport (canned RSS + pages), this exercises the
entire production code path (research loop, verification, universal compile, runtime)
with only the socket mocked.
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
Diaz preferred to defer. The Board signaled that adoption is the expected outcome at the
next meeting.
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
            "compile_reality": self._reality,
            "compile_uncertainty": self._frame,
            "compile_world": self._actors,
            "actor_decision": self._decision,
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
            "process_type": "committee_vote",
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

    def _reality(self, prompt: str) -> dict:
        ids = _ID_RE.findall(prompt)  # cite the claim ids present in the prompt
        return {
            "process_type": "committee_vote",
            "decision_body": "Widget Standards Board",
            "subject_entity": "Widget standard",
            "resolution_units": "unanimity of three seats",
            "institution_id": "widget_board",
            "institution_name": "Widget Standards Board",
            "decision_rule": {"kind": "majority", "total_seats": 3, "evidence_claim_ids": ids[:1]},
            "expected_voting_seats": 3,
            "target_option": "adopt",
            "terminal": {
                "mechanism": "committee_vote",
                "yes_condition": "unanimous_for_option",
                "target_option": "adopt",
                "k": None,
                "evidence_claim_ids": ids[:1],
            },
            "members": [
                _m("ada_lovelace", "Ada Lovelace", "Chair", "adopt", ids, chair=True),
                _m("ben_carter", "Ben Carter", "Member", "adopt", ids),
                _m("cara_diaz", "Cara Diaz", "Member", "defer", ids),
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

    def _frame(self, prompt: str) -> dict:
        ids = _ID_RE.findall(prompt)
        return {
            "options": ["reject", "defer", "adopt"],
            "signals": [
                {
                    "name": "opposition",
                    "baseline": 0.0,
                    "description": "public opposition",
                    "evidence_claim_ids": ids[:1],
                }
            ],
            "reaction_rules": [
                {
                    "trigger_signal": "opposition",
                    "direction": "above",
                    "threshold": 0.5,
                    "moves_to_option": "reject",
                    "rationale": "strong opposition",
                    "evidence_claim_ids": ids[:1],
                }
            ],
            "guidance_option": "adopt",
            "guidance_text": "adoption is the expected outcome",
            "guidance_evidence_ids": ids[:1],
            "acceptance_tolerance": 0.5,
            "uncertainty": [
                {
                    "signal": "opposition",
                    "why_unknown": "future opposition unknown",
                    "reversal_capable": True,
                    "constraining_evidence_ids": ids[:1],
                    "outcomes": [
                        {
                            "value": "quiet",
                            "weight": 0.75,
                            "provenance": "symmetric_ignorance_assumption",
                            "source_detail": "calm",
                            "signal_effects": [["opposition", 0.0]],
                            "description": "no opposition",
                        },
                        {
                            "value": "loud",
                            "weight": 0.25,
                            "provenance": "explicit_model_distribution",
                            "source_detail": "opposition",
                            "signal_effects": [["opposition", 0.6]],
                            "description": "opposition mobilizes",
                        },
                    ],
                },
            ],
        }

    def _actors(self, prompt: str) -> dict:
        ids = re.findall(r'"actor_id":\s*"([a-z_]+)"', prompt)
        return {
            "actors": [
                {
                    "actor_id": a,
                    "current_inclination": "adopt",
                    "reaction_rules": [],
                    "dissent_threshold": 0.5,
                    "reasoning": "grounded",
                }
                for a in ids
            ]
        }

    def _decision(self, prompt: str) -> dict:
        if (
            '"decision"' in prompt
            or "STAGE\ndecision" in prompt
            or '"stage": "decision"' in prompt.lower()
            or "decision" in _stage(prompt)
        ):
            return {
                "kind": "cast_vote",
                "vote_option": _proposal_option(prompt) or "adopt",
                "rationale": "I support the recommended option.",
                "expected_effect": "record vote",
                "referenced_memory_ids": [],
                "referenced_observation_ids": [],
            }
        if "positions" in _stage(prompt):
            return {
                "kind": "make_statement",
                "favored_option": "adopt",
                "statement_text": "I support adoption.",
                "info_signals": {},
                "rationale": "state position",
                "expected_effect": "colleagues learn my view",
                "referenced_memory_ids": [],
                "referenced_observation_ids": [],
            }
        return {
            "kind": "wait",
            "vote_option": "",
            "rationale": "await the proposal",
            "expected_effect": "wait",
            "pending_need": "await_proposal",
            "referenced_memory_ids": [],
            "referenced_observation_ids": [],
        }


def _stage(prompt: str) -> str:
    m = re.search(r"## STAGE\s*\n\s*\"?([a-z_]+)", prompt)
    return m.group(1) if m else ""


def _proposal_option(prompt: str) -> str:
    m = re.search(r'"option":\s*"([a-z_]+)"', prompt)
    return m.group(1) if m else ""


def _c(prop: str, value: str, entities: list[str], excerpt: str, auth: int) -> dict:
    return {
        "proposition": prop,
        "normalized_value": value,
        "entities": entities,
        "epistemic_type": "observation",
        "supporting_excerpt": excerpt,
        "authority_hint": auth,
    }


def _m(
    actor_id: str, name: str, role: str, prior: str, ids: list[str], *, chair: bool = False
) -> dict:
    auth = ["vote", "introduce_proposal", "chair"] if chair else ["vote"]
    return {
        "actor_id": actor_id,
        "name": name,
        "role": role,
        "is_voting_seat": True,
        "vote_power": 1,
        "prior_action": prior,
        "authority": auth,
        "evidence_claim_ids": ids[:2],
        "memory_seeds": [
            {
                "content": f"My prior position was {prior}.",
                "kind": "episodic",
                "importance": 0.7,
                "evidence_claim_ids": ids[:1],
            }
        ],
    }
