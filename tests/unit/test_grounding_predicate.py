"""A citation supports the claim it is attached to, or it grounds nothing.

Six register entries — FD-29, FD-30, FD-35, FD-36, FD-38, FD-51 — are one defect wearing
six hats: a check that reads the PRESENCE of evidence as proof of SUPPORT. The widest of
them, FD-30, was a single predicate. ``semantic_plan.cited()`` returned ``True`` for any
id in the store, and four independent gates read that answer as "this value is
evidenced": D3's threshold straddle, D4's single-multiplier exemption, D5's zero-actor
justification and FD-11's filler alternative. Putting ``c-f1`` — *"The Kestrel Bay ferry
terminal recorded 320000 crossings"* — on both straddling alternatives of an **aquifer**
world's runoff fraction satisfied all four at once, and the world was accepted.

The precedent for the repair already existed in this file's own module. ``attest_profiles``
settled the same question for actors: a claim grounds an actor only when it actually names
that actor, whole names only, never parts — because an invented "Terminal operations
manager" shares the token "terminal" with any claim about a ferry terminal and would
attest itself. :class:`ClaimSupport` is that move applied to the other things a plan
cites, and the two rules that shape it are both proven here:

* **A gate that refuses correct worlds is worse than the hole it closes.** So the tests
  below that matter most are the negative ones: a claim recording a *range* supports every
  value inside it although it names none of them; a claim recording "3 percent" supports a
  1.03 multiplier; a categorical alternative is judged by one shared word, not by
  paraphrase; and the suite's three good fixtures still validate clean when the store's
  texts are put in front of the validator.
* **A check that could not run must never read as a check that passed.** Support is
  tri-state. Without claim texts every question answers UNDECIDABLE, the gates fall back
  to the old existence check, and they say so in the refusal they print.

Nothing here is domain code. The fixtures are the CWF suite's two invented worlds — an
aquifer recharge district and a bay ferry terminal — because the attack is on shape.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest
import test_causal_world_fidelity as T

from sworldmodel.grounding import (
    ClaimSupport,
    Support,
    cited_record,
    stated_quantities,
)
from sworldmodel.semantic_plan import parse_semantic_plan, validate_semantic_plan
from sworldmodel.world_review import mechanical_world_checks

# The support-aware validator is landing as a patch to semantic_plan.py, which another
# agent holds. The predicate, and the review's use of it, are here now; the gate tests
# below turn themselves on the moment the patch lands, and say why they are off until
# then rather than quietly reporting nothing.
_VALIDATOR_READS_THE_RECORD = (
    "claim_records" in inspect.signature(validate_semantic_plan).parameters
)
_needs_patch = pytest.mark.skipif(
    not _VALIDATOR_READS_THE_RECORD,
    reason="semantic_plan.py's claim_records patch is not applied yet (held by another agent)",
)

SUPPORT = ClaimSupport.from_claims(T.CLAIMS)


# ---------------------------------------------------------------------------
# The predicate: what a record has to say before it grounds a number
# ---------------------------------------------------------------------------


def test_a_record_that_states_the_range_supports_every_value_inside_it() -> None:
    """The legitimate shape, and the one a stricter test would have destroyed.

    ``c-a3`` records that winter runoff fractions "range from 0.6 to 0.9". It names
    neither alternative of the aquifer world's driver, and it supports both — which is
    exactly what D3's own correction boundary asks for ("a published range, a recorded
    distribution, a stated forecast"). Demanding that the record name the value would
    refuse the honest forecast this system exists to produce.
    """

    assert SUPPORT.supports_quantity(("c-a3",), 0.6) is Support.SUPPORTED
    assert SUPPORT.supports_quantity(("c-a3",), 0.9) is Support.SUPPORTED
    assert SUPPORT.supports_quantity(("c-a3",), 0.75) is Support.SUPPORTED


def test_a_record_about_something_else_supports_nothing_it_is_stapled_to() -> None:
    """FD-30's own demonstration, at the predicate.

    The ferry terminal's crossing count is a real claim, admissible at the cutoff, in the
    store — and it says nothing whatever about a runoff fraction. Existence was the whole
    of the old test.
    """

    assert SUPPORT.exists(("c-f1",)) is True
    assert SUPPORT.supports_quantity(("c-f1",), 0.9) is Support.UNSUPPORTED
    assert SUPPORT.supports_quantity(("c-f1",), 0.6) is Support.UNSUPPORTED
    # What it DOES support is the number it actually records.
    assert SUPPORT.supports_quantity(("c-f1",), 320000) is Support.SUPPORTED


def test_a_value_outside_the_cited_band_is_not_inside_it() -> None:
    """A published band is a band for the values in it, and for no others.

    ``c-f2`` publishes a seasonal factor "between 1.3 and 1.4". A plan citing it for a
    1.1 factor is citing a document that contradicts the number it is being used to
    support — which is the FD-29 shape at its sharpest, because the plan looks cited.
    """

    assert SUPPORT.supports_quantity(("c-f2",), 1.4) is Support.SUPPORTED
    assert SUPPORT.supports_quantity(("c-f2",), 1.35) is Support.SUPPORTED
    assert SUPPORT.supports_quantity(("c-f2",), 1.1) is Support.UNSUPPORTED


def test_a_percentage_supports_the_multiplier_a_modeller_would_write() -> None:
    """Legitimate indirect support that a naive numeric test would refuse.

    "Volumes rose 3 percent" is the ordinary way a record backs a 1.03 factor, and
    refusing it over a formatting difference would refuse correct worlds. The reading is
    still numeric — an adversary cannot satisfy it with an arbitrary claim, only with one
    whose own number stands in this relation to the value.
    """

    claims = {"c-p": {"proposition": "Crossings rose 3 percent on last season", "value": "3%"}}
    support = ClaimSupport.from_claims(claims)
    for value in (3.0, 0.03, 1.03, 0.97):
        assert support.supports_quantity(("c-p",), value) is Support.SUPPORTED, value
    assert support.supports_quantity(("c-p",), 1.4) is Support.UNSUPPORTED


def test_a_number_inside_an_identifier_is_not_a_quantity_the_record_states() -> None:
    """Otherwise every claim would support the digits in its own id or its version."""

    claims = {"c-x": {"proposition": "Report v2.1 for site c-a3 was filed", "value": ""}}
    support = ClaimSupport.from_claims(claims)
    assert support.supports_quantity(("c-x",), 3) is Support.UNSUPPORTED
    assert support.supports_quantity(("c-x",), 2.1) is Support.UNSUPPORTED


def test_stated_quantities_reads_ranges_but_not_two_separate_figures() -> None:
    """ "between A and B" is one interval; "A and B" is two numbers and no interval.

    The difference matters: reading every adjacent pair as a range would invent support
    for every value between any two figures a claim happens to mention.
    """

    _points, intervals = stated_quantities("between 0.6 and 0.9 of capacity")
    assert (0.6, 0.9) in intervals
    _points, intervals = stated_quantities("240 acre-feet and 800 acre-feet")
    assert intervals == ()


def test_a_filler_label_is_supported_by_nothing_and_a_real_one_by_a_word() -> None:
    """FD-11's shape, and the paraphrase that must survive it.

    A categorical alternative is judged far more loosely than a number — one significant
    word shared with the record is enough — because a plan legitimately paraphrases what
    a source says and no mechanical test can tell a paraphrase from an invention. What
    the loose test still refuses is the label that shares nothing at all.
    """

    assert SUPPORT.supports_value(("c-a1",), "some other regime") is Support.UNSUPPORTED
    assert SUPPORT.supports_value(("c-a1",), "managed recharge is logged") is Support.SUPPORTED


def test_a_record_attests_a_world_by_naming_it_and_never_by_a_part_of_a_name() -> None:
    """The zero-actor and exemption half, and the trap it has to avoid.

    Whole names only, exactly as :func:`attest_profiles` decided for actors: an invented
    "Terminal operations manager" shares the token "terminal" with a claim about a ferry
    terminal, and part-matching would let it attest itself out of a claim that is not
    about a person at all.
    """

    ferry = ["Kestrel Bay Ferry Terminal", "bay harbour authority"]
    aquifer = ["Cald Basin Recharge District"]
    assert SUPPORT.names_any_of(("c-f1",), ferry) is Support.SUPPORTED
    assert SUPPORT.names_any_of(("c-f1",), aquifer) is Support.UNSUPPORTED
    assert SUPPORT.names_any_of(("c-f1",), ["Terminal operations manager"]) is Support.UNSUPPORTED


def test_without_claim_texts_every_support_question_answers_undecidable() -> None:
    """The tri-state is the honesty. UNDECIDABLE is not a soft SUPPORTED.

    A caller holding only ids has not checked anything, and the gates read this and fall
    back to the existence check *while saying so*. Collapsing it to a pass here would
    reintroduce FD-30 one layer up.
    """

    blind = ClaimSupport.from_claims(None, known=frozenset(T.CLAIMS))
    assert blind.can_read_claims is False
    assert blind.exists(("c-f1",)) is True
    assert blind.supports_quantity(("c-f1",), 0.9) is Support.UNDECIDABLE
    assert blind.supports_value(("c-f1",), "anything") is Support.UNDECIDABLE
    assert blind.names_any_of(("c-f1",), ["Cald Basin Recharge District"]) is Support.UNDECIDABLE


def test_an_id_that_is_not_in_the_store_supports_nothing_and_says_so_separately() -> None:
    """Existence and support are different findings, and both are kept.

    A citation to a claim that does not exist is worse than one that exists and says
    something else, and the refusals differ; the predicate must not collapse them.
    """

    assert SUPPORT.exists(("c-no-such",)) is False
    assert SUPPORT.supports_quantity(("c-no-such",), 0.9) is Support.UNDECIDABLE


def test_a_claim_is_read_from_the_store_object_or_from_a_plain_mapping() -> None:
    """The validator is handed whatever the caller has; neither shape may be guessed at."""

    from dataclasses import dataclass

    @dataclass
    class StoreClaim:
        proposition: str
        normalized_value: str
        supporting_excerpt: str
        entities: tuple[str, ...]

    obj = cited_record("c1", StoreClaim("runoff ran 0.7", "0.7", "", ("Basin",)))
    mapping = cited_record("c1", {"proposition": "runoff ran 0.7", "value": "0.7"})
    assert obj.normalized_value == "0.7" and obj.entities == ("Basin",)
    assert mapping.normalized_value == "0.7"


# ---------------------------------------------------------------------------
# FD-51 — the compiled review asks the same question of the same record
# ---------------------------------------------------------------------------


def _review(plan: dict[str, Any]) -> dict[str, Any]:
    bundle = T._bundle(plan)
    view = bundle.evidence_store.view(T.AS_OF)
    return {f.key: f for f in mechanical_world_checks(T._compile(plan), view)}


def test_a_deciding_factor_whose_citation_does_not_state_its_value_is_refused() -> None:
    """FD-51. ``constraining_evidence_ids`` is a per-VARIABLE union, so one citation
    anywhere on an uncertainty made it non-empty and this check reported PASS — FD-29's
    shape surviving into the review, with no backstop if the static gate is evaded.

    Here the aquifer driver keeps its cited leg and its other leg is moved to 0.35, which
    the cited series (0.6 to 0.9) does not contain. The union is still non-empty; the
    branch the answer turns on is still a number somebody chose.
    """

    plan = T.recharge_plan(cited_driver=True)
    plan["uncertainties"][0]["alternatives"][1]["value"] = 0.35
    plan["uncertainties"][0]["alternatives"][1]["evidence_claim_ids"] = []

    finding = _review(plan)["multiplier_lacks_evidence"]
    assert finding.severity == "CRITICAL"
    assert finding.is_blocking
    assert "0.35" in finding.finding
    assert finding.evidence_basis.startswith("computed from the compiled world")


def test_a_deciding_factor_citing_an_unrelated_claim_is_refused_by_the_review() -> None:
    """FD-30 at the review: both legs cited, neither supported, the union non-empty."""

    plan = T.recharge_plan(cited_driver=True)
    for alt in plan["uncertainties"][0]["alternatives"]:
        alt["evidence_claim_ids"] = ["c-f1"]

    finding = _review(plan)["multiplier_lacks_evidence"]
    assert finding.severity == "CRITICAL"
    assert "0.9" in finding.finding and "0.6" in finding.finding


def test_a_straddle_the_cited_range_covers_still_passes_the_review() -> None:
    """The negative case, and the reason the fix is not "one citation per outcome".

    Requiring each alternative to carry its own id would refuse this world: the record
    states a range, both alternatives sit inside it, and which leg the planner stapled
    the id to is bookkeeping. What is checked is whether the record establishes each
    value the branch can take — so the honest forecast is admitted and FD-51's defect is
    still caught.
    """

    plan = T.recharge_plan(cited_driver=True)
    plan["uncertainties"][0]["alternatives"][0]["evidence_claim_ids"] = ["c-a3"]
    plan["uncertainties"][0]["alternatives"][1]["evidence_claim_ids"] = []

    findings = _review(plan)
    assert findings["multiplier_lacks_evidence"].severity == "PASS"
    # The observation is unchanged: one factor still decides the answer, and the review
    # says so. Evidence changes the verdict, never the observation.
    assert findings["single_uncertain_multiplier_decides"].severity == "MEDIUM"
    assert [k for k, f in findings.items() if f.is_blocking] == []


def test_without_an_evidence_view_the_review_reports_what_it_could_not_compare() -> None:
    """A caller with no store cannot decide support, and the finding must not imply it did."""

    findings = {f.key: f for f in mechanical_world_checks(T._compile(T.recharge_plan()))}
    finding = findings["multiplier_lacks_evidence"]
    assert finding.severity == "PASS"
    assert "presence only" in finding.finding
    assert "no store available" in finding.evidence_basis
    assert finding.evidence_basis.startswith("computed from the compiled world")


# ---------------------------------------------------------------------------
# The static gates, once the validator is handed the record (FD-29/30/35/36/38)
# ---------------------------------------------------------------------------


def _validate(plan: dict[str, Any]) -> list[str]:
    return validate_semantic_plan(
        parse_semantic_plan(plan),
        as_of=T.AS_OF,
        horizon=T.HORIZON,
        known_claim_ids=frozenset(T.CLAIMS),
        claim_records=T.CLAIMS,
    )


def _soften(plan: dict[str, Any]) -> dict[str, Any]:
    """Declare every alternative merely 'moves_the_terminal'.

    The adversary did this in every single attack, because it is what evades FD-36's two
    half-checks — so a test of the straddling gate that leaves the honest declaration in
    place is testing the filler gate instead.
    """

    for u in plan.get("uncertainties", []):
        for alt in u["alternatives"]:
            if alt.get("terminal_sensitivity") == "decides_the_terminal":
                alt["terminal_sensitivity"] = "moves_the_terminal"
    return plan


@_needs_patch
def test_the_three_good_fixtures_still_compile_when_the_record_is_consulted() -> None:
    """The constraint that decides the design, checked first.

    Every world the suite believes in is re-validated with the store's texts in front of
    the validator. If any of them starts failing, the support test is refusing correct
    worlds and is worse than the hole it closes.
    """

    assert _validate(T.recharge_plan()) == []
    assert _validate(T.recharge_plan(with_actor=True)) == []
    assert _validate(T.ferry_plan(factors=(1.3, 1.4), cited_factors=True, exemption=True)) == []


@_needs_patch
def test_an_arbitrary_claim_no_longer_grounds_a_straddling_alternative() -> None:
    """FD-30's headline repro: a ferry claim on both legs of an aquifer straddle."""

    plan = _soften(T.recharge_plan(cited_driver=False))
    for alt in plan["uncertainties"][0]["alternatives"]:
        alt["evidence_claim_ids"] = ["c-f1"]

    errors = _validate(plan)
    assert "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS" in T._defects(errors), errors
    assert "break-even draw" in errors[0]


@_needs_patch
def test_an_arbitrary_claim_no_longer_grounds_the_single_multiplier_exemption() -> None:
    """FD-30's second repro: D4's documented-model escape hatch, on unrelated claims."""

    plan = T.ferry_plan(factors=(1.3, 1.4), cited_factors=True, exemption=True)
    for alt in plan["uncertainties"][0]["alternatives"]:
        alt["evidence_claim_ids"] = ["c-a3"]

    assert T._defects(_validate(plan)) == {"SINGLE_DRIVER_EXEMPTION_UNGROUNDED"}


@_needs_patch
def test_an_arbitrary_claim_no_longer_grounds_a_zero_actor_justification() -> None:
    """FD-30 and FD-38 at D5. Two non-empty strings and any id in the store was the
    whole of the check that decides a world contains nobody who can change the answer."""

    plan = T.recharge_plan()
    plan["zero_actor_justification"]["evidence_claim_ids"] = ["c-f1"]
    errors = _validate(plan)
    assert T._defects(errors) == {"ZERO_ACTOR_WORLD_UNJUSTIFIED"}, errors
    assert "c-f1" in errors[0]
    assert "excluded_candidates" in errors[0]


@_needs_patch
def test_an_arbitrary_claim_no_longer_grounds_a_filler_alternative() -> None:
    """FD-30 at FD-11's gate: the label "other" with a citation stapled to it."""

    plan = T.recharge_plan()
    plan["uncertainties"][0]["alternatives"][1] = {
        "value": "some other regime",
        "weight": None,
        "provenance": "symmetric_ignorance_assumption",
        "grounding": "no specific alternative in evidence",
        "meaning": "something else happens",
        "why_unresolved": "unknown",
        "changes": ["process_state"],
        "terminal_sensitivity": "decides_the_terminal",
        "evidence_claim_ids": ["c-f1"],
    }
    errors = _validate(plan)
    assert "DEGENERATE_FILLER_ALTERNATIVE" in T._defects(errors), errors
    assert "some other regime" in errors[0] and "c-f1" in errors[0]


@_needs_patch
def test_citing_one_straddling_leg_no_longer_silences_the_gate() -> None:
    """FD-29. One cited leg beside one invented leg is the Tesla defect with a fig leaf.

    The gate required BOTH legs to be ungrounded, so this was accepted. The invented
    number still decides which side of the threshold its own branch lands on, and the
    refusal now names which side is the loose one.
    """

    plan = _soften(T.recharge_plan(cited_driver=False))
    plan["uncertainties"][0]["alternatives"][0]["evidence_claim_ids"] = ["c-a3"]

    errors = _validate(plan)
    (message,) = [e for e in errors if e.startswith("THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS")]
    assert "0.6 side is established by nothing" in message


@_needs_patch
def test_a_cited_leg_whose_value_the_record_contradicts_is_not_grounded() -> None:
    """Both legs carry the citation, and one of them sits outside the range it cites."""

    plan = _soften(T.recharge_plan(cited_driver=True))
    plan["uncertainties"][0]["alternatives"][1]["value"] = 0.35

    assert "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS" in T._defects(_validate(plan))


@_needs_patch
def test_a_straddle_cannot_be_hidden_behind_more_uncertainties() -> None:
    """FD-35. The same straddle, differing only in how many BACKGROUND channels exist.

    Five extra channels put the combination count past the cap and the enumeration fell
    back to *the first alternative of each* — which is whatever order the planner typed —
    so the driver's straddle never appeared. Three extra channels stayed under the cap
    and were refused. A planner escaped D3 by looking more thorough.
    """

    refused = {n: T._defects(_validate(_capped_plan(n))) for n in (3, 5)}
    assert refused[5] == refused[3] == {"THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS"}, refused


def _capped_plan(n_channels: int) -> dict[str, Any]:
    """The aquifer driver, plus ``n_channels`` ungauged side channels of pure background."""

    plan = _soften(T.recharge_plan(cited_driver=False))
    plan["uncertainties"][0]["alternatives"][0]["value"] = 0.60
    plan["uncertainties"][0]["alternatives"][1]["value"] = 0.80
    for i in range(n_channels):
        name = f"side channel {i} volume"
        plan["states"].append(
            {
                "name": name,
                "owner": "world",
                "state_type": "quantity",
                "unit": "acre-feet",
                "initial": "UNKNOWN",
                "why_material": "a minor channel that also drains into the recharge log",
                "evidence_claim_ids": [],
            }
        )
        plan["processes"][0]["occurrences"].append(
            {
                "description": f"side channel {i} drains into the log",
                "at": f"2026-02-2{i}T06:00:00+00:00",
                "changes": [
                    {
                        "op": "increase",
                        "target": "recorded recharge volume",
                        "amount": {"kind": "state", "state": name},
                    }
                ],
            }
        )
        plan["uncertainties"].append(
            {
                "name": f"side channel {i} yield",
                "what_unknown": "how much the side channel yields",
                "why_unknown": "the channel is ungauged",
                "affects_state": name,
                "release_at": None,
                "alternatives": [
                    {
                        "value": value,
                        "weight": None,
                        "provenance": "symmetric_ignorance_assumption",
                        "grounding": "ungauged",
                        "meaning": meaning,
                        "why_unresolved": "the channel is ungauged",
                        "changes": ["process_state"],
                        "terminal_sensitivity": "moves_the_terminal",
                        "evidence_claim_ids": [],
                    }
                    for value, meaning in ((0.0, "the channel stays dry"), (15.0, "it runs"))
                ],
            }
        )
    return plan


@_needs_patch
def test_declaring_the_middle_sensitivity_no_longer_evades_both_halves() -> None:
    """FD-36. ``terminal_sensitivity`` is self-declared and one of its three values was
    checked by nothing: MISDECLARED fired on `immaterial_to_the_terminal`, the filler
    gate fires on `decides_the_terminal`, and `moves_the_terminal` sat between them.

    The arithmetic has already resolved these alternatives on opposite sides of the
    threshold, which is `decides_the_terminal` and nothing weaker — a fact about the plan
    rather than an opinion about it.
    """

    plan = _soften(T.recharge_plan(cited_driver=True))
    errors = _validate(plan)
    assert T._defects(errors) == {"TERMINAL_SENSITIVITY_MISDECLARED"}, errors
    assert "moves_the_terminal" in errors[0]
    assert "decides_the_terminal" in errors[0]

    # And the honest declaration on the same world is still silent.
    assert _validate(T.recharge_plan(cited_driver=True)) == []


@_needs_patch
def test_the_zero_actor_gate_is_no_longer_satisfied_by_two_strings_and_any_id() -> None:
    """FD-38. The check was: two fields are non-empty, and some id exists."""

    plan = T.recharge_plan()
    plan["zero_actor_justification"] = {
        "no_material_decision": "x",
        "process_sufficiency": "y",
        "evidence_claim_ids": ["c-f1"],
    }
    assert T._defects(_validate(plan)) == {"ZERO_ACTOR_WORLD_UNJUSTIFIED"}

    # A justification resting on the district's own record is still admitted: the gate
    # refuses a citation about someone else, not a short one.
    plan["zero_actor_justification"]["evidence_claim_ids"] = ["c-a1", "c-a2"]
    assert _validate(plan) == []


@_needs_patch
def test_a_validator_with_no_claim_texts_behaves_exactly_as_it_did() -> None:
    """UNDECIDABLE falls back to the existence check — never to a refusal.

    A caller that hands over ids and no texts has not checked support, and must not have
    worlds refused on a comparison that never happened. What it does get is the note, in
    the refusal it prints, that nothing was compared.
    """

    plan = _soften(T.recharge_plan(cited_driver=False))
    for alt in plan["uncertainties"][0]["alternatives"]:
        alt["evidence_claim_ids"] = ["c-f1"]

    blind = validate_semantic_plan(
        parse_semantic_plan(copy.deepcopy(plan)),
        as_of=T.AS_OF,
        horizon=T.HORIZON,
        known_claim_ids=frozenset(T.CLAIMS),
    )
    assert "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS" not in T._defects(blind)
    assert "THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS" in T._defects(_validate(plan))

    # A world the blind validator DOES refuse says that support was never examined.
    bare = T.recharge_plan()
    bare["zero_actor_justification"]["evidence_claim_ids"] = []
    (message,) = validate_semantic_plan(
        parse_semantic_plan(bare),
        as_of=T.AS_OF,
        horizon=T.HORIZON,
        known_claim_ids=frozenset(T.CLAIMS),
    )
    assert "never checked" in message
