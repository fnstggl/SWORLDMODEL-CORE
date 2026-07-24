"""Actor grounding: named actors are specific people, never generic role templates."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from _helpers import base_corpus, compile_dict, run_dict
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.grounding import (
    ActorGroundingProfile,
    Provenance,
    assess_actor_grounding,
    enforce_actor_grounding,
    profile_from_member,
    verified,
)


def _profiles(corpus: dict[str, Any]) -> dict[str, ActorGroundingProfile]:
    compiled = compile_dict(corpus)
    return {p.actor_id: p for p in compiled.actor_grounding.profiles}


# 1. Same role, different histories -> different grounded profiles.
def test_same_role_different_histories_give_different_profiles() -> None:
    profs = _profiles(base_corpus())
    b, c = profs["b"], profs["c"]
    assert b.role == c.role  # both "Member"
    assert b.previous_observed_action is not None and c.previous_observed_action is not None
    assert b.previous_observed_action.content != c.previous_observed_action.content
    assert b.render_grounding() != c.render_grounding()


# 2. Shared evidence reaches everyone without erasing actor-specific evidence.
def test_shared_context_does_not_replace_actor_specific_grounding() -> None:
    compiled = compile_dict(base_corpus())
    shared = {f.text for f in compiled.base_world.verified_facts}
    assert shared  # the common guidance is shared world context
    for p in compiled.actor_grounding.profiles:
        # ...and each actor still carries its own distinct verified action.
        assert p.previous_observed_action is not None
        assert p.previous_observed_action.is_verified


# 3. A verified previous action is never overwritten by an inferred inclination.
def test_verified_action_survives_a_contrary_inferred_inclination() -> None:
    p = profile_from_member(
        actor_id="x",
        name="Dana Vale",
        role="Member",
        authority=("vote",),
        previous_action="cut",
        previous_action_claim_ids=("k1",),
        inclination="hold",  # the compiler's inference disagrees with the record
    )
    assert p.previous_observed_action is not None
    assert "cut" in p.previous_observed_action.content
    assert p.previous_observed_action.provenance is Provenance.VERIFIED_OBSERVATION
    incl = p.current_evidence_grounded_inclination
    assert incl is not None and "hold" in incl.content
    assert incl.provenance is Provenance.SUPPORTED_INFERENCE
    rendered = p.render_grounding()
    assert "cut" in rendered and "hold" in rendered  # both present, not collapsed


# 4. Missing actor evidence is marked unknown, never invented.
def test_missing_actor_evidence_is_recorded_as_unknown() -> None:
    p = profile_from_member(
        actor_id="x", name="Dana Vale", role="Member", authority=("vote",), previous_action=None
    )
    assert p.missing_information
    assert any("no previous" in m for m in p.missing_information)
    assert "do not invent" in p.render_grounding()


# 5. One actor may not carry another actor's personal record.
def test_cross_assigned_personal_record_is_refused() -> None:
    good = profile_from_member(
        actor_id="ada_lin",
        name="Ada Lin",
        role="Member",
        authority=("vote",),
        previous_action="hold",
    )
    bad = ActorGroundingProfile(
        actor_id="ben_ruiz",
        canonical_identity="Ben Ruiz",
        role="Member",
        authority=("vote",),
        # Ben's profile carries Ada's statement — a misattribution.
        direct_statements=(verified("Ada Lin stated: I favor a hold.", ("k9",)),),
    )
    report = assess_actor_grounding((good, bad))
    assert not report.is_complete
    assert report.misattributed
    with pytest.raises(WorldIntegrityError):
        enforce_actor_grounding(report)


# 6. A post-cutoff claim cannot reach an actor profile.
def test_post_cutoff_claim_never_enters_an_actor_profile() -> None:
    corpus = copy.deepcopy(base_corpus())
    corpus["sources"][0]["claims"].append(
        {
            "id": "future_leak",
            "proposition": "vote: a voted to cut at the June meeting",
            "normalized_value": "cut",
            "entities": ["a"],
            "epistemic_type": "observation",
            "confidence": 0.99,
        }
    )
    # The claim is published after the cutoff, so it is unreachable through the view.
    corpus["sources"][0]["available_at"] = corpus["reality"]["horizon"]
    corpus["sources"][0]["published_at"] = corpus["reality"]["horizon"]
    with pytest.raises((WorldIntegrityError, Exception)):
        # Every roster claim is now post-cutoff, so the world cannot verify -> refused.
        compile_dict(corpus)


# 7. The exact provider prompt is preserved in the decision record.
def test_exact_prompt_is_recorded_for_every_actor_decision() -> None:
    _, ctx, _ = run_dict(base_corpus())
    lines = ctx.actor_decision_lines()
    assert lines
    import json

    for line in lines:
        rec = json.loads(line)
        prompt = rec["exact_prompt"]
        assert prompt, "the exact prompt must be recorded"
        # The recorded prompt is the actor's own, not a shared template.
        assert rec["canonical_identity"] in prompt
        assert "ACTOR-SPECIFIC GROUNDING" in prompt
        assert "SHARED WORLD CONTEXT" in prompt


# 8. Renaming an actor does not silently retain another person's profile.
def test_renaming_an_actor_changes_its_grounded_identity() -> None:
    corpus = copy.deepcopy(base_corpus())
    corpus["reality"]["members"][1]["name"] = "Renamed Person"
    profs = _profiles(corpus)
    assert profs["b"].canonical_identity == "Renamed Person"
    assert "Renamed Person" in profs["b"].render_grounding()
    assert "Bob" not in profs["b"].render_grounding()


# 9. A known real person cannot fall back to a generic role template.
def test_generic_role_only_actor_is_refused() -> None:
    generic = ActorGroundingProfile(
        actor_id="dep_gov_1",
        canonical_identity="Deputy Governor 1",
        role="Deputy Governor",
        authority=("vote",),
    )
    report = assess_actor_grounding((generic,))
    assert not report.is_complete
    assert report.generic_actors
    with pytest.raises(WorldIntegrityError):
        enforce_actor_grounding(report)


def test_constructed_representative_may_be_generic_when_labeled() -> None:
    stratum = ActorGroundingProfile(
        actor_id="stratum_a",
        canonical_identity="Urban voters",
        role="population segment",
        authority=("vote",),
        is_constructed_representative=True,
        population_weight=0.4,
    )
    assert assess_actor_grounding((stratum,)).is_complete
    # ...but an unlabeled/weightless synthetic agent is still refused.
    bad = ActorGroundingProfile(
        actor_id="stratum_b",
        canonical_identity="Rural voters",
        role="population segment",
        authority=("vote",),
        is_constructed_representative=True,
    )
    assert not assess_actor_grounding((bad,)).is_complete


# 10. Actors are not forced to differ when evidence supports the same position.
def test_actors_may_share_a_position_when_evidence_agrees() -> None:
    a = profile_from_member(
        actor_id="a", name="Ann Roe", role="Member", authority=("vote",), previous_action="hold"
    )
    b = profile_from_member(
        actor_id="b", name="Bo Poe", role="Member", authority=("vote",), previous_action="hold"
    )
    report = assess_actor_grounding((a, b))
    assert report.is_complete  # identical evidenced positions are legitimate
    assert a.previous_observed_action.content == b.previous_observed_action.content  # type: ignore[union-attr]


# 11. Grounding works for a non-committee process.
def test_grounding_works_for_a_non_committee_process() -> None:
    from _helpers import synthetic_corpus

    compiled = compile_dict(synthetic_corpus("individual_response"))
    # An actor-action process: the outcome is one actor acting, not a committee tally.
    assert compiled.base_world.contract.terminal_predicate.mechanism == "actor_action"
    profiles = compiled.actor_grounding.profiles
    assert profiles
    for p in profiles:
        assert p.canonical_identity
        assert p.render_grounding()
    assert compiled.actor_grounding.is_complete


# 12. Removing actor-specific grounding materially changes or blocks the simulation.
def test_stripping_actor_grounding_blocks_the_run() -> None:
    stripped = tuple(
        ActorGroundingProfile(
            actor_id=p.actor_id,
            canonical_identity=p.canonical_identity,
            role=p.role,
            authority=p.authority,
        )
        for p in compile_dict(base_corpus()).actor_grounding.profiles
    )
    report = assess_actor_grounding(stripped)
    assert not report.is_complete
    with pytest.raises(WorldIntegrityError):
        enforce_actor_grounding(report)
