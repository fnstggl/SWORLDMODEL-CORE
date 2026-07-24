from __future__ import annotations

from sworldmodel.mechanisms import carried_option, evaluate_terminal, tally_votes
from sworldmodel.models import DecisionRule, TerminalSpec

RULE = DecisionRule(kind="majority", total_seats=5, threshold=3)
UNANIMOUS_HOLD = TerminalSpec(
    mechanism="committee_vote", yes_condition="unanimous_for_option", target_option="hold"
)


def test_tally_is_deterministic_and_weighted() -> None:
    votes = {"a": "hold", "b": "hold", "c": "cut", "d": "hold", "e": "cut"}
    powers = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 1}
    t1 = tally_votes(votes, powers)
    t2 = tally_votes(votes, powers)
    assert t1 == t2 == {"hold": 4, "cut": 2}


def test_unanimous_yes() -> None:
    votes = dict.fromkeys("abcde", "hold")
    ev = evaluate_terminal(votes, dict.fromkeys("abcde", 1), 5, RULE, UNANIMOUS_HOLD)
    assert ev.resolved and ev.outcome == "YES"


def test_unanimous_no_when_one_dissents() -> None:
    votes = dict.fromkeys("abcd", "hold") | {"e": "cut"}
    ev = evaluate_terminal(votes, dict.fromkeys("abcde", 1), 5, RULE, UNANIMOUS_HOLD)
    assert ev.resolved and ev.outcome == "NO"


def test_incomplete_vote_is_unresolved_not_forced() -> None:
    votes = dict.fromkeys("abcd", "hold")  # only 4 of 5 voted
    ev = evaluate_terminal(votes, dict.fromkeys("abcde", 1), 5, RULE, UNANIMOUS_HOLD)
    assert not ev.resolved and ev.outcome is None
    assert "missing" in ev.reason


def test_at_least_k_and_majority_predicates() -> None:
    votes = dict.fromkeys("abc", "hold") | {"d": "cut", "e": "cut"}
    powers = dict.fromkeys("abcde", 1)
    k_spec = TerminalSpec(
        mechanism="committee_vote", yes_condition="at_least_k_for_option", target_option="hold", k=3
    )
    assert evaluate_terminal(votes, powers, 5, RULE, k_spec).outcome == "YES"
    maj_spec = TerminalSpec(
        mechanism="committee_vote", yes_condition="majority_for_option", target_option="hold"
    )
    assert evaluate_terminal(votes, powers, 5, RULE, maj_spec).outcome == "YES"
    assert carried_option({"hold": 3, "cut": 2}, RULE) == "hold"
