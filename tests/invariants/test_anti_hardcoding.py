"""Anti-hardcoding invariants: the core is general; Banxico lives only in fixtures."""

from __future__ import annotations

import pathlib

from _helpers import base_corpus, dup, run_dict, synthetic_corpus

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "sworldmodel"
# The CLI is the *evaluation harness*; scenario constants (cutoff, corpus path) are
# allowed to live there per the architecture. The core is everything else.
CORE_FILES = [p for p in SRC.glob("*.py") if p.name not in {"cli.py", "__main__.py"}]

BANXICO_NAMES = [
    "Rodriguez",
    "Rodríguez",
    "Heath",
    "Borja",
    "Mejia",
    "Mejía",
    "Cuadra",
    "Victoria",
    "Galia",
    "Omar",
    "Jonathan",
    "Gabriel",
]
BANXICO_CONSTANTS = ["banxico", "6.50", "2026-06-25", "2026-05-14"]
FORBIDDEN_MECHANISMS = [
    "consensus_pull",
    "body_mean",
    "mean_field",
    "consensus_norm",
    "coalition_discipline",
    "leadership_floor",
    "if institution ==",
    'institution == "banxico"',
]


def _core_text() -> str:
    return "\n".join(p.read_text() for p in CORE_FILES).lower()


def test_core_contains_no_banxico_member_names() -> None:
    blob = _core_text()
    for name in BANXICO_NAMES:
        assert name.lower() not in blob, f"core leaks Banxico name {name!r}"


def test_core_contains_no_banxico_constants_or_special_branch() -> None:
    blob = _core_text()
    for token in BANXICO_CONSTANTS:
        assert token.lower() not in blob, f"core leaks Banxico constant {token!r}"


def test_core_contains_no_numeric_deliberation_mechanisms() -> None:
    blob = _core_text()
    for token in FORBIDDEN_MECHANISMS:
        assert token.lower() not in blob, f"core contains forbidden mechanism {token!r}"


def test_seven_member_committee_runs_through_the_same_code() -> None:
    result, _, _ = run_dict(synthetic_corpus("seven_member_council"))
    assert result.integrity_manifest.represented_voting_seats == 7
    resolved = [b for b in result.branch_outcomes if b.resolved]
    assert resolved
    for b in resolved:
        assert len(b.votes) == 7  # every seat casts a final vote


def test_different_proposal_and_option_set_runs_through_the_same_protocol() -> None:
    result, _, _ = run_dict(synthetic_corpus("data_shock_board"))
    assert result.status.value == "resolved"
    assert set(result.contract.outcome_space) == {"maintain", "change"}


def test_renaming_all_actors_does_not_change_mechanics() -> None:
    base = base_corpus()
    r_base, _, _ = run_dict(base)

    renamed = dup(base)
    remap = {"a": "zeta", "b": "yankee", "c": "xray"}
    members = renamed["reality"]["members"]
    for m in members:
        old = m["actor_id"]
        m["actor_id"] = remap[old]
        m["name"] = remap[old].upper()
        m["evidence_claim_ids"] = [f"r_{remap[old]}"]
        for s in m["memory_seeds"]:
            s["evidence_claim_ids"] = [f"r_{remap[old]}"]
    # rewrite roster claims + required facts to the new ids
    for src in renamed["sources"]:
        for c in src["claims"]:
            if c["id"].startswith("r_") and c["id"][2:] in remap:
                new = remap[c["id"][2:]]
                c["id"] = f"r_{new}"
                c["entities"] = [new]
                c["proposition"] = f"office: {new} is a member"
    for fact in renamed["required_reality_facts"]:
        if fact["key"] == "roster":
            fact["evidence_claim_ids"] = [f"r_{remap[a]}" for a in ("a", "b", "c")]

    r_renamed, _, _ = run_dict(renamed)
    assert r_base.simulation_probability == r_renamed.simulation_probability
    # the vote *distribution* per branch is identical up to renaming
    base_dist = sorted(sorted(o for _, o in b.votes) for b in r_base.branch_outcomes)
    renamed_dist = sorted(sorted(o for _, o in b.votes) for b in r_renamed.branch_outcomes)
    assert base_dist == renamed_dist
