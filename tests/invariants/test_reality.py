"""Reality-integrity invariants: the simulator refuses a structurally false world."""

from __future__ import annotations

from datetime import datetime

import pytest

from _helpers import base_corpus, compile_dict, dup
from sworldmodel.actors import ActorState
from sworldmodel.errors import ContractMutationError, WorldIntegrityError
from sworldmodel.evidence import EvidenceStore
from sworldmodel.models import (
    ActorDefinition,
    ConditionalBehavior,
    DecisionRule,
    InstitutionSpec,
    ResolutionContract,
    TerminalSpec,
)
from sworldmodel.reality import verify_reality

T0 = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
AS_OF = datetime.fromisoformat("2024-01-15T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2024-02-15T00:00:00+00:00")


def _actor(aid: str, name: str, *, power: int = 1, voting: bool = True) -> ActorState:
    cb = ConditionalBehavior("hold", (), 0.5, "r")
    d = ActorDefinition(aid, name, "Member", ("vote",), voting, power, cb)
    return ActorState.from_definition(d, default_time=T0)


def _contract(
    expected: int, total: int, threshold: int, kind: str = "majority"
) -> ResolutionContract:
    return ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        outcome_space=("cut", "hold", "hike"),
        target_outcome="hold",
        subject_entity="s",
        decision_body="b",
        resolution_units="u",
        terminal_predicate=TerminalSpec("committee_vote", "unanimous_for_option", "hold"),
        decision_rule=DecisionRule(kind, total, threshold),
        expected_voting_seats=expected,
    )


def _view():
    return EvidenceStore().view(AS_OF)


# --- via the corpus + compiler (end of the reality gate) -------------------------


def test_nine_seat_board_represented_as_fewer_fails() -> None:
    corpus = dup(base_corpus())
    corpus["reality"]["expected_voting_seats"] = 9
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    assert "missing seats" in str(exc.value)


def test_missing_member_fails() -> None:
    corpus = dup(base_corpus())
    corpus["reality"]["members"] = corpus["reality"]["members"][:2]  # drop one, expect 3
    with pytest.raises(WorldIntegrityError):
        compile_dict(corpus)


def test_duplicated_seat_fails() -> None:
    corpus = dup(base_corpus())
    corpus["reality"]["members"][1]["name"] = corpus["reality"]["members"][0]["name"]
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    assert "duplicated" in str(exc.value).lower()


def test_threshold_inconsistent_with_rule_fails() -> None:
    corpus = dup(base_corpus())
    corpus["reality"]["decision_rule"]["threshold"] = 3  # majority of 3 must be 2
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    assert "threshold" in str(exc.value).lower()


def test_structurally_unknown_roster_blocks_rollout() -> None:
    corpus = dup(base_corpus())
    for fact in corpus["required_reality_facts"]:
        if fact["key"] == "roster":
            fact["evidence_claim_ids"] = []  # no evidence attaches the roster
    with pytest.raises(WorldIntegrityError):
        compile_dict(corpus)


# --- direct verify_reality (unit) ------------------------------------------------


def test_incorrect_vote_weights_fail() -> None:
    actors = {
        "a": _actor("a", "A", power=1),
        "b": _actor("b", "B", power=1),
        "c": _actor("c", "C", power=1),
    }
    # institution declares a DIFFERENT verified power for seat 'a'
    inst = InstitutionSpec(
        institution_id="i",
        name="I",
        member_actor_ids=("a", "b", "c"),
        decision_rule=DecisionRule("majority", 3, 2),
        seat_vote_powers=(("a", 2), ("b", 1), ("c", 1)),
    )
    with pytest.raises(WorldIntegrityError) as exc:
        verify_reality(_contract(3, 3, 2), _view(), actors, inst)
    assert "power" in str(exc.value).lower()


def test_actor_budget_cannot_remove_voting_seats() -> None:
    # A world that represents fewer seats than the contract expects must fail —
    # there is no path that trims a roster for speed.
    actors = {"a": _actor("a", "A"), "b": _actor("b", "B")}  # only 2 of expected 3
    with pytest.raises(WorldIntegrityError):
        verify_reality(_contract(3, 3, 2), _view(), actors, None)


def test_contract_load_bearing_fields_are_immutable() -> None:
    contract = _contract(3, 3, 2)
    with pytest.raises(ContractMutationError):
        contract.checked_replace(expected_voting_seats=5)
    with pytest.raises(ContractMutationError):
        contract.checked_replace(decision_rule=DecisionRule("majority", 5, 3))
    # the one permitted mutation (recording satisfied facts) is allowed
    assert contract.with_satisfied_facts(()) is not None
