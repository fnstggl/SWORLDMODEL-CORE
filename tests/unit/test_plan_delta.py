"""The plan-delta merge contract: deterministic, byte-preserving, precise on refusal.

Delta repair exists so a revision round never regenerates the plan: the model returns
only corrected objects and code rebuilds the plan around them. The property that makes
this fidelity-preserving is proven here directly — objects a delta does not name come
through the merge as the SAME objects, byte-equivalent under canonical serialization,
so citations, weights, uncertainty structure and causal wiring cannot drift in a
revision that never mentioned them.
"""

from __future__ import annotations

import copy

import pytest

from sworldmodel.ids import canonical_json
from sworldmodel.semantic_plan import (
    PlanDeltaError,
    delta_is_empty,
    merge_plan_delta,
)


def base_plan() -> dict:
    return {
        "resolution": {"question": "q?", "yes_condition": "yes", "evidence_claim_ids": ["c-1"]},
        "entities": [
            {
                "name": "Registrar",
                "structural_type": "person",
                "decides": True,
                "evidence_claim_ids": ["c-1"],
            },
            {
                "name": "Applicants",
                "structural_type": "population",
                "decides": False,
                "evidence_claim_ids": ["c-2"],
            },
        ],
        "states": [
            {"name": "filings received", "state_type": "quantity", "initial": "UNKNOWN"},
        ],
        "events": [],
        "affordances": [
            {
                "name": "approve filing",
                "actor": "Registrar",
                "changes": [{"op": "increase", "target": "filings received", "amount": 1}],
            }
        ],
        "processes": [],
        "uncertainties": [
            {
                "name": "application volume",
                "affects_state": "filings received",
                "alternatives": [
                    {"value": 10, "weight": None, "provenance": "symmetric_ignorance_assumption"},
                    {"value": 20, "weight": None, "provenance": "symmetric_ignorance_assumption"},
                ],
            }
        ],
        "terminal": {"form": "quantity_comparison", "state": "filings received"},
        "terminal_producer_note": "the registrar's approvals produce the count",
        "world_facts": [{"text": "the office exists", "evidence_claim_ids": ["c-1"]}],
    }


def test_unnamed_objects_survive_byte_equivalent_and_identical() -> None:
    prior = base_plan()
    delta = {
        "revised": {
            "states": [{"name": "filings received", "state_type": "quantity", "initial": 4}]
        }
    }
    merged = merge_plan_delta(prior, delta)
    # The revised object was replaced in place.
    assert merged["states"][0]["initial"] == 4
    # Every object the delta did not name is the SAME object — not a re-serialization,
    # not a copy that could have drifted. Identity implies byte-equivalence.
    assert merged["entities"][0] is prior["entities"][0]
    assert merged["entities"][1] is prior["entities"][1]
    assert merged["affordances"][0] is prior["affordances"][0]
    assert merged["uncertainties"][0] is prior["uncertainties"][0]
    assert merged["terminal"] is prior["terminal"]
    assert merged["world_facts"] is prior["world_facts"]
    assert canonical_json(merged["uncertainties"]) == canonical_json(prior["uncertainties"])
    # Citations on untouched objects are untouched.
    assert merged["entities"][1]["evidence_claim_ids"] == ["c-2"]


def test_merge_is_deterministic_and_never_mutates_the_prior() -> None:
    prior = base_plan()
    snapshot = copy.deepcopy(prior)
    delta = {
        "revised": {
            "entities": [
                {"name": "Deputy registrar", "structural_type": "person", "decides": True},
                {"name": "Registrar", "structural_type": "person", "decides": True},
            ]
        },
        "removed": {"uncertainties": ["application volume"]},
    }
    a = merge_plan_delta(prior, delta)
    b = merge_plan_delta(prior, delta)
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(prior) == canonical_json(snapshot), "the prior plan was mutated"
    # Replacement stays in the prior plan's position; a new name appends after it, in
    # the delta's own order.
    assert [e["name"] for e in a["entities"]] == ["Registrar", "Applicants", "Deputy registrar"]
    assert a["uncertainties"] == []


def test_removing_an_undeclared_name_is_an_error_not_a_noop() -> None:
    with pytest.raises(PlanDeltaError) as exc:
        merge_plan_delta(base_plan(), {"removed": {"entities": ["Nobody at all"]}})
    assert any("Nobody at all" in e for e in exc.value.errors)
    assert any("not the one being revised" in e for e in exc.value.errors)


def test_revising_and_removing_the_same_name_is_an_error() -> None:
    with pytest.raises(PlanDeltaError) as exc:
        merge_plan_delta(
            base_plan(),
            {
                "revised": {"entities": [{"name": "Registrar", "structural_type": "person"}]},
                "removed": {"entities": ["Registrar"]},
            },
        )
    assert any("both revised and removed" in e for e in exc.value.errors)


def test_a_full_plan_reemission_is_rejected_with_the_exact_reason() -> None:
    """The failure mode delta repair exists to remove: the model returning the whole
    plan for a one-object correction. Accepting it would silently re-roll every
    unchanged object — so it is a named shape error, never merged."""

    with pytest.raises(PlanDeltaError) as exc:
        merge_plan_delta(base_plan(), base_plan())
    assert any("re-emitted a complete plan" in e for e in exc.value.errors)


def test_scalars_replace_wholesale_and_null_means_unchanged() -> None:
    prior = base_plan()
    merged = merge_plan_delta(
        prior,
        {
            "terminal": {"form": "event_exists", "event": "approval recorded"},
            "resolution": None,
            "terminal_producer_note": None,
        },
    )
    assert merged["terminal"] == {"form": "event_exists", "event": "approval recorded"}
    assert merged["resolution"] is prior["resolution"]
    assert merged["terminal_producer_note"] is prior["terminal_producer_note"]


def test_an_empty_delta_is_readable_and_changes_nothing() -> None:
    prior = base_plan()
    for empty in ({}, {"revised": {}, "removed": {}}):
        merged = merge_plan_delta(prior, empty)
        assert canonical_json(merged) == canonical_json(prior)
        assert delta_is_empty(empty)
    assert not delta_is_empty({"revised": {"states": [{"name": "x"}]}})
    assert not delta_is_empty({"terminal": {"form": "event_exists"}})


def test_objects_without_names_and_unknown_sections_are_named_errors() -> None:
    with pytest.raises(PlanDeltaError) as exc:
        merge_plan_delta(
            base_plan(),
            {"revised": {"states": [{"state_type": "quantity"}], "gadgets": []}},
        )
    errors = exc.value.errors
    assert any("needs its 'name'" in e for e in errors)
    assert any("unknown section" in e for e in errors)
