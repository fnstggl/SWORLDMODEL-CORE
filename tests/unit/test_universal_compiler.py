"""Normalization robustness: null/missing LLM fields never crash the compiler."""

from __future__ import annotations

from datetime import datetime

from sworldmodel.evidence import EvidenceStore
from sworldmodel.universal_compiler import _normalize_frame, _normalize_reality

AS_OF = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-02-01T00:00:00+00:00")


def _view():
    return EvidenceStore().view(AS_OF)


def test_null_heavy_reality_is_coerced_to_valid_structure() -> None:
    raw = {
        "decision_body": None,
        "institution_name": None,
        "subject_entity": None,
        "resolution_units": None,
        "target_option": None,
        "expected_voting_seats": None,
        "decision_rule": {},
        "terminal": {},
        "members": None,
        "authoritative_sources": None,
    }
    out = _normalize_reality(raw, _view(), AS_OF, HORIZON)
    # No string field is None (would crash the compiler); collections are lists.
    for key in (
        "decision_body",
        "institution_name",
        "institution_id",
        "subject_entity",
        "resolution_units",
        "target_option",
    ):
        assert isinstance(out[key], str) and out[key]
    assert isinstance(out["members"], list)
    assert isinstance(out["authoritative_sources"], list)
    rule = out["decision_rule"]
    assert isinstance(rule["kind"], str) and isinstance(rule["total_seats"], int)
    assert isinstance(out["terminal"]["mechanism"], str)


def test_frame_normalization_drops_bad_uncertainty_and_normalizes_weights() -> None:
    raw = {
        "options": None,
        "signals": None,
        "reaction_rules": None,
        "uncertainty": [
            {
                "signal": "x",
                "outcomes": [
                    {"value": "a", "weight": 3, "provenance": "bogus"},
                    {"value": "b", "weight": 1},
                ],
            },
            {"signal": "y", "outcomes": []},  # empty -> dropped
        ],
    }
    out = _normalize_frame(raw, _view())
    assert isinstance(out["options"], list)
    assert len(out["uncertainty"]) == 1  # the empty-outcome variable was dropped
    weights = [o["weight"] for o in out["uncertainty"][0]["outcomes"]]
    assert abs(sum(weights) - 1.0) < 1e-9  # per-variable conservation
    assert out["uncertainty"][0]["outcomes"][0]["provenance"] == "explicit_model_distribution"
