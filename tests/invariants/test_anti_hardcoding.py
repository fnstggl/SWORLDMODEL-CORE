"""Anti-hardcoding invariants: the core is a universal simulator, not a committee engine.

The core contains no scenario facts (Banxico names/constants), no mechanism-family
router (committee/vote/negotiation/... dispatch), and no hardcoded domain-action or
terminal-family vocabulary. Committee voting exists only as compiled DATA in a corpus.
"""

from __future__ import annotations

import pathlib

from _helpers import run_dict, synthetic_corpus

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "sworldmodel"
# The CLI is the evaluation harness (Banxico cutoff/paths allowed there). Core = the rest.
CORE_FILES = [p for p in SRC.glob("*.py") if p.name not in {"cli.py", "__main__.py"}]

BANXICO_NAMES = ["Rodriguez", "Heath", "Borja", "Mejia", "Cuadra", "Victoria", "Galia", "Omar"]
BANXICO_CONSTANTS = ["banxico", "6.50", "2026-06-25", "2026-05-14"]

# Mechanism-family routers, committee-specific types, and hardcoded domain actions /
# terminal families that must NOT exist anywhere in the core.
FORBIDDEN = [
    "committee_vote",
    "actor_action",
    "weighted_majority",
    "committee_protocol",
    "general_protocol",
    "evaluate_action_terminal",
    "is_voting_seat",
    "vote_power",
    "InstitutionSpec",
    "DecisionRule",
    "terminal_predicate",
    "TerminalSpec",
    "ScenarioFrame",
    "consensus_pull",
    "body_mean",
    "mean_field",
    'institution == "banxico"',
    "if institution ==",
]


def _core_text() -> str:
    return "\n".join(p.read_text() for p in CORE_FILES).lower()


def test_core_contains_no_banxico_member_names() -> None:
    blob = _core_text()
    for name in BANXICO_NAMES:
        assert name.lower() not in blob, f"core leaks Banxico name {name!r}"


def test_core_contains_no_banxico_constants() -> None:
    blob = _core_text()
    for token in BANXICO_CONSTANTS:
        assert token.lower() not in blob, f"core leaks Banxico constant {token!r}"


def test_core_contains_no_mechanism_family_router_or_types() -> None:
    blob = _core_text()
    for token in FORBIDDEN:
        assert token.lower() not in blob, f"core contains forbidden token {token!r}"


def test_supporting_a_new_domain_adds_no_source_only_data() -> None:
    # Five materially different domains run through the same code with no source change.
    for name in (
        "committee_decision",
        "individual_response",
        "negotiation",
        "population_behavior",
        "geopolitical_process",
    ):
        result, _ = run_dict(synthetic_corpus(name))
        assert result.probability_source == "weighted_simulated_trajectories"
        assert result.status.value == "resolved"
