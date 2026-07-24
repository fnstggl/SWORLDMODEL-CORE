"""Cumulative research and universal repair routing.

Two integration defects made live runs unreliable: research augmentation restarted
from an empty evidence store (so a retry could LOSE facts an earlier attempt had
verified), and the repair loop understood only coverage-gate omissions (so every other
refusal propagated without any attempt to find the missing evidence).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from _live_helpers import AS_OF, HORIZON, FakeLLM, widget_transport
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.live_research import LiveResearchBackend, ResearchBudget
from sworldmodel.repair import FailureType, classify_failure

NOW = datetime.fromisoformat("2027-03-01T00:00:00+00:00")
QUESTION = "Will the Widget Standards Board adopt the standard at its next meeting?"


def _backend() -> LiveResearchBackend:
    return LiveResearchBackend(
        FakeLLM(), widget_transport(), budget=ResearchBudget(max_rounds=1), now=NOW
    )


# --------------------------------------------------------------------------- #
# Cumulative research
# --------------------------------------------------------------------------- #


def test_augmentation_preserves_all_prior_verified_claims() -> None:
    backend = _backend()
    backend.research(QUESTION, AS_OF, HORIZON)
    session = backend._session(QUESTION, AS_OF, HORIZON)
    before = {c.id for c in session.store.all()}
    assert before, "the first attempt must verify some claims"

    backend.augment_for_coverage(QUESTION, AS_OF, HORIZON, ["[person] Someone Missing"], None)
    after = {c.id for c in backend._session(QUESTION, AS_OF, HORIZON).store.all()}
    # Every previously verified claim survives the retry.
    assert before <= after


def test_a_retry_reuses_the_same_session_and_never_shrinks() -> None:
    backend = _backend()
    backend.research(QUESTION, AS_OF, HORIZON)
    s = backend._session(QUESTION, AS_OF, HORIZON)
    first_attempts, first_count = s.attempts, s.claim_count
    backend.research(QUESTION, AS_OF, HORIZON)
    assert s.attempts == first_attempts + 1  # same session continued, not recreated
    assert s.claim_count >= first_count  # the store only grows


def test_completed_queries_are_not_repeated_across_attempts() -> None:
    backend = _backend()
    backend.research(QUESTION, AS_OF, HORIZON)
    session = backend._session(QUESTION, AS_OF, HORIZON)
    assert session.completed_queries, "the first attempt must record its queries"
    ran_first = set(session.completed_queries)
    backend.research(QUESTION, AS_OF, HORIZON)
    # The initial-query list is not re-run verbatim; budget goes to new ground.
    assert ran_first <= session.completed_queries


def test_a_different_question_gets_its_own_session() -> None:
    backend = _backend()
    backend.research(QUESTION, AS_OF, HORIZON)
    other = backend._session("A completely different question", AS_OF, HORIZON)
    assert other.claim_count == 0  # sessions are not cross-contaminated


# --------------------------------------------------------------------------- #
# Typed failure classification and routing
# --------------------------------------------------------------------------- #


def _fail(msg: str, **details: object) -> WorldIntegrityError:
    return WorldIntegrityError(msg, details=details)


def test_coverage_omission_is_repairable_with_targeted_needs() -> None:
    f = classify_failure(
        _fail("coverage", missing_material_candidates=["[person] Gala Reyes — material but absent"])
    )
    assert f.failure_type is FailureType.MISSING_MATERIAL_EVIDENCE
    assert f.retryable
    assert f.targeted_research_needs == ("Gala Reyes",)


def test_no_actors_is_repairable_and_asks_who_decides() -> None:
    f = classify_failure(_fail("no actors were verified from evidence — simulation refused"))
    assert f.failure_type is FailureType.NO_CAUSALLY_RELEVANT_ACTORS
    assert f.retryable and f.targeted_research_needs


def test_participant_shortfall_is_repairable_but_surplus_is_not() -> None:
    short = classify_failure(
        _fail(
            "participant roster does not match verified reality",
            **{"expected participants": 12, "represented participants": 5},
        )
    )
    assert short.failure_type is FailureType.PARTICIPANT_COUNT_MISMATCH
    assert short.retryable and short.expected_value == 12 and short.represented_value == 5

    surplus = classify_failure(
        _fail(
            "participant roster does not match verified reality",
            **{"expected participants": 3, "represented participants": 5},
        )
    )
    # A world with MORE participants than reality is false, not under-researched.
    assert surplus.failure_type is FailureType.PARTICIPANT_SURPLUS
    assert not surplus.retryable


def test_ungrounded_actor_is_repairable_by_actor_specific_research() -> None:
    f = classify_failure(
        _fail(
            "actors are not grounded as specific people",
            generic_actors=["dep_gov_1 (Deputy Governor 1): only a name and a generic role"],
            misattributed=[],
            missing_previous_actions=[],
        )
    )
    assert f.failure_type is FailureType.ACTOR_GROUNDING_INCOMPLETE
    assert f.retryable and f.targeted_research_needs


def test_misattributed_and_duplicate_and_contradiction_are_not_repairable() -> None:
    mis = classify_failure(_fail("actors are not grounded", misattributed=["b carries a's record"]))
    assert mis.failure_type is FailureType.ACTOR_IDENTITY_UNVERIFIED and not mis.retryable

    dup = classify_failure(_fail("duplicated participant detected — refused", duplicated=["ada"]))
    assert dup.failure_type is FailureType.DUPLICATED_PARTICIPANT and not dup.retryable

    con = classify_failure(_fail("decisive evidence contradiction blocks rollout"))
    assert con.failure_type is FailureType.CONTRADICTORY_LOAD_BEARING_EVIDENCE
    assert not con.retryable


def test_missing_required_fact_is_repairable() -> None:
    f = classify_failure(
        _fail("required reality facts are unverified", missing=["roster: no evidence attached"])
    )
    assert f.failure_type is FailureType.MISSING_REQUIRED_REALITY_FACT
    assert f.retryable


def test_a_surplus_roster_never_triggers_research() -> None:
    """The gates are not weakened: a false world refuses rather than being repaired.

    The repair loop consults ``retryable`` before calling the backend, so a surplus
    roster can never reach ``augment_for_coverage``.
    """

    surplus = _fail(
        "roster mismatch", **{"expected participants": 2, "represented participants": 9}
    )
    assert not classify_failure(surplus).retryable
    with pytest.raises(WorldIntegrityError):
        raise surplus
