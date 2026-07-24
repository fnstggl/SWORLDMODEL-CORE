"""Shared test builders.

A tiny, controllable 3-seat committee corpus that individual tests mutate to craft
edge cases (missing member, duplicated seat, post-cutoff claim, contradiction, ...).
Everything here is synthetic — there are no Banxico facts in the test helpers.
"""

from __future__ import annotations

import copy
import json
import pathlib
from datetime import datetime
from typing import Any

from sworldmodel import DeterministicGateway, ForecastConfig, MockResearchBackend, run_forecast
from sworldmodel.api import _build_contract
from sworldmodel.compiler import compile_world
from sworldmodel.research import build_bundle_from_dict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BANXICO_CORPUS = REPO_ROOT / "evaluation" / "banxico" / "corpus" / "corpus.json"
SYNTHETIC_ROOT = REPO_ROOT / "evaluation" / "synthetic"

AS_OF = "2024-01-15T00:00:00+00:00"
HORIZON = "2024-02-15T12:00:00+00:00"
ROSTER_PUB = "2024-01-01T00:00:00+00:00"


def base_corpus() -> dict[str, Any]:
    """A minimal valid 3-seat committee: A/B hold, C cut; target unanimous hold."""

    def member(aid: str, name: str, prior: str, chair: bool = False) -> dict[str, Any]:
        auth = ["vote", "introduce_proposal", "chair"] if chair else ["vote"]
        return {
            "actor_id": aid,
            "name": name,
            "role": "Chair" if chair else "Member",
            "is_voting_seat": True,
            "vote_power": 1,
            "prior_action": prior,
            "authority": auth,
            "evidence_claim_ids": [f"r_{aid}"],
            "memory_seeds": [
                {
                    "content": f"My prior position was {prior}.",
                    "kind": "episodic",
                    "importance": 0.7,
                    "valid_time": ROSTER_PUB,
                    "evidence_claim_ids": [f"r_{aid}"],
                }
            ],
        }

    roster_claims = [
        {
            "id": f"r_{aid}",
            "proposition": f"office: {aid} is a member",
            "normalized_value": "member",
            "entities": [aid],
            "epistemic_type": "observation",
            "confidence": 0.99,
        }
        for aid in ("a", "b", "c")
    ]
    roster_claims += [
        {
            "id": "rule_c",
            "proposition": "rule: committee decides by majority of three",
            "normalized_value": "majority_3",
            "entities": ["committee"],
            "epistemic_type": "observation",
            "confidence": 0.98,
        },
        {
            "id": "guid_c",
            "proposition": "guidance: the common position is hold",
            "normalized_value": "hold",
            "entities": ["committee"],
            "epistemic_type": "observation",
            "confidence": 0.9,
        },
        {
            "id": "ctx_c",
            "proposition": "context: conditions are calm",
            "normalized_value": "calm",
            "entities": ["committee"],
            "epistemic_type": "observation",
            "confidence": 0.85,
        },
    ]

    return {
        "question_key": "test_committee",
        "reality": {
            "as_of": AS_OF,
            "horizon": HORIZON,
            "decision_body": "Test Committee",
            "subject_entity": "Test Measure",
            "resolution_units": "unanimity of three seats",
            "institution_id": "test_committee",
            "institution_name": "Test Committee",
            "decision_rule": {
                "kind": "majority",
                "total_seats": 3,
                "threshold": 2,
                "evidence_claim_ids": ["rule_c"],
            },
            "expected_voting_seats": 3,
            "target_option": "hold",
            "terminal": {
                "mechanism": "committee_vote",
                "yes_condition": "unanimous_for_option",
                "target_option": "hold",
            },
            "authoritative_sources": ["roster"],
            "members": [
                member("a", "Alice", "hold", chair=True),
                member("b", "Bob", "hold"),
                member("c", "Carol", "cut"),
            ],
        },
        "frame": {
            "options": ["cut", "hold", "hike"],
            "signals": [
                {
                    "name": "shock",
                    "baseline": 0.0,
                    "description": "a downside shock",
                    "evidence_claim_ids": ["ctx_c"],
                }
            ],
            "reaction_rules": [
                {
                    "trigger_signal": "shock",
                    "direction": "above",
                    "threshold": 0.5,
                    "moves_to_option": "cut",
                    "rationale": "a large shock forces a cut",
                    "evidence_claim_ids": ["ctx_c"],
                }
            ],
            "guidance_option": "hold",
            "guidance_text": "The common position is to hold.",
            "guidance_evidence_ids": ["guid_c"],
            "acceptance_tolerance": 0.5,
            "uncertainty": [
                {
                    "signal": "shock",
                    "why_unknown": "future shock unknown",
                    "reversal_capable": True,
                    "constraining_evidence_ids": ["ctx_c"],
                    "outcomes": [
                        {
                            "value": "none",
                            "weight": 0.8,
                            "provenance": "symmetric_ignorance_assumption",
                            "source_detail": "calm",
                            "signal_effects": [["shock", 0.0]],
                            "description": "no shock",
                        },
                        {
                            "value": "big",
                            "weight": 0.2,
                            "provenance": "explicit_model_distribution",
                            "source_detail": "a big shock",
                            "signal_effects": [["shock", 0.6]],
                            "description": "a decisive shock",
                        },
                    ],
                }
            ],
        },
        "world_facts": [
            {
                "text": "The common position is hold.",
                "evidence_claim_ids": ["guid_c"],
                "available_at": ROSTER_PUB,
                "epistemic_type": "observation",
            }
        ],
        "required_reality_facts": [
            {
                "key": "roster",
                "description": "three-seat roster",
                "evidence_claim_ids": ["r_a", "r_b", "r_c"],
            },
            {
                "key": "decision_rule",
                "description": "majority of three",
                "evidence_claim_ids": ["rule_c"],
            },
        ],
        "sources": [
            {
                "source_id": "roster",
                "url": "https://example.org",
                "title": "Roster",
                "source_type": "official_institutional",
                "authority_level": 4,
                "published_at": ROSTER_PUB,
                "available_at": ROSTER_PUB,
                "lineage_event_id": "roster_event",
                "claims": roster_claims,
            }
        ],
        "outcome": None,
    }


def dup(corpus: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(corpus)


def run_dict(corpus: dict[str, Any], *, seed: int = 0, max_branches: int = 24):
    """Run the full pipeline on a corpus dict. Returns (result, ctx, config)."""

    backend = MockResearchBackend(data=corpus)
    bundle = backend.research("q", datetime.fromisoformat(AS_OF), datetime.fromisoformat(HORIZON))
    as_of = bundle.as_of or datetime.fromisoformat(AS_OF)
    config = ForecastConfig(
        gateway=DeterministicGateway(),
        research_backend=backend,
        seed=seed,
        max_branches=max_branches,
    )
    result, ctx = run_forecast("q", as_of, bundle.horizon, config)
    return result, ctx, config


def compile_dict(corpus: dict[str, Any], *, gateway=None):
    """Compile a corpus into a CompiledWorld (runs the reality-integrity gate)."""

    backend = MockResearchBackend(data=corpus)
    bundle = backend.research("q", datetime.fromisoformat(AS_OF), datetime.fromisoformat(HORIZON))
    as_of = bundle.as_of or datetime.fromisoformat(AS_OF)
    evidence_view = bundle.evidence_store.view(as_of)
    contract = _build_contract("q", as_of, bundle.horizon, bundle)
    gw = gateway or DeterministicGateway()
    return compile_world(contract, evidence_view, bundle, gw, seed=0)


def bundle_of(corpus: dict[str, Any]):
    return build_bundle_from_dict(corpus)


def banxico_corpus() -> dict[str, Any]:
    return json.loads(BANXICO_CORPUS.read_text())


def synthetic_corpus(name: str) -> dict[str, Any]:
    return json.loads((SYNTHETIC_ROOT / name / "corpus.json").read_text())
