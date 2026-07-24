"""Reality-integrity invariants: the simulator refuses a structurally false world.

Generic, not committee-specific: a claimed participant count must match the verified
roster (a nine-participant body can never become five modeled units), no duplicated
participant, required facts must be evidence-backed, and the contract's load-bearing
fields (including the declarative terminal) are immutable after verification.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from _helpers import base_corpus, compile_dict, dup
from sworldmodel.actors import ActorState
from sworldmodel.errors import ContractMutationError, WorldIntegrityError
from sworldmodel.evidence import EvidenceStore
from sworldmodel.models import ResolutionContract
from sworldmodel.reality import verify_reality
from sworldmodel.worldspec import ActorPolicy, ActorSpec, EntitySpec, Expr, TerminalExpression

AS_OF = datetime.fromisoformat("2024-01-15T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2024-02-15T00:00:00+00:00")
T0 = datetime.fromisoformat("2024-01-01T00:00:00+00:00")


def _actor(aid: str, name: str) -> ActorState:
    ent = EntitySpec(aid, name, "person", is_actor=True, role="member", authority=("decide",))
    return ActorState.from_spec(ent, ActorSpec(aid, ActorPolicy()), default_time=T0)


def _contract(expected: int | None) -> ResolutionContract:
    return ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity="s",
        resolution_units="u",
        terminal=TerminalExpression(Expr("const", (True,))),
        target_outcome="t",
        expected_participants=expected,
    )


def _view():
    return EvidenceStore().view(AS_OF)


# --- via the corpus + compiler (end of the reality gate) -------------------------


def test_claimed_participant_count_larger_than_roster_fails() -> None:
    corpus = dup(base_corpus())
    corpus["reality"]["expected_participants"] = 9  # only 3 are actually verified
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    assert "roster does not match" in str(exc.value)


def test_missing_participant_fails() -> None:
    corpus = dup(base_corpus())
    corpus["world_spec"]["actors"] = corpus["world_spec"]["actors"][:2]  # drop one, expect 3
    corpus["world_spec"]["entities"] = corpus["world_spec"]["entities"][:2]
    with pytest.raises(WorldIntegrityError):
        compile_dict(corpus)


def test_duplicated_participant_fails() -> None:
    corpus = dup(base_corpus())
    corpus["world_spec"]["entities"][1]["name"] = corpus["world_spec"]["entities"][0]["name"]
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    assert "duplicated" in str(exc.value).lower()


def test_structurally_unknown_roster_blocks_rollout() -> None:
    corpus = dup(base_corpus())
    for fact in corpus["required_reality_facts"]:
        if fact["key"] == "roster":
            fact["evidence_claim_ids"] = []  # no evidence attaches the roster
    with pytest.raises(WorldIntegrityError):
        compile_dict(corpus)


# --- direct verify_reality (unit) ------------------------------------------------


def test_represented_fewer_than_expected_fails() -> None:
    actors = {"a": _actor("a", "A"), "b": _actor("b", "B")}  # only 2 of expected 3
    with pytest.raises(WorldIntegrityError):
        verify_reality(_contract(3), _view(), actors)


def test_verified_when_counts_match() -> None:
    actors = {"a": _actor("a", "A"), "b": _actor("b", "B"), "c": _actor("c", "C")}
    manifest = verify_reality(_contract(3), _view(), actors)
    assert manifest.is_verified
    assert manifest.represented_participants == 3


def test_contract_load_bearing_fields_are_immutable() -> None:
    contract = _contract(3)
    with pytest.raises(ContractMutationError):
        contract.checked_replace(expected_participants=5)
    with pytest.raises(ContractMutationError):
        contract.checked_replace(terminal=TerminalExpression(Expr("const", (False,))))
    # the one permitted mutation (recording satisfied facts) is allowed
    assert contract.with_satisfied_facts(()) is not None
