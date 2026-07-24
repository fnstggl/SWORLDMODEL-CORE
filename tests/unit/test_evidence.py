from __future__ import annotations

from datetime import datetime

import pytest

from sworldmodel.errors import CutoffViolationError
from sworldmodel.evidence import (
    EvidenceClaim,
    EvidenceStore,
    check_lineage_independence,
    mark_contradiction,
)
from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType

T = datetime.fromisoformat


def _claim(cid: str, *, available: str, lineage: str, value: str = "v") -> EvidenceClaim:
    pub = T(available)
    return EvidenceClaim(
        id=cid,
        proposition=f"prop {cid}",
        normalized_value=value,
        entities=("x",),
        valid_from=None,
        valid_until=None,
        published_at=pub,
        available_at=pub,
        source_id="s",
        source_url="",
        source_title="",
        source_type=SourceType.OFFICIAL_INSTITUTIONAL,
        authority_level=AuthorityLevel.AUTHORITATIVE,
        supporting_excerpt="",
        lineage_event_id=lineage,
        epistemic_type=EpistemicType.OBSERVATION,
        confidence=0.9,
        retrieved_at=pub,
    )


def _store() -> EvidenceStore:
    s = EvidenceStore()
    s.add(_claim("pre1", available="2024-01-01T00:00:00+00:00", lineage="e1"))
    s.add(_claim("pre2", available="2024-01-02T00:00:00+00:00", lineage="e1"))
    s.add(_claim("post", available="2024-03-01T00:00:00+00:00", lineage="e2"))
    return s


CUTOFF = T("2024-01-15T00:00:00+00:00")


def test_post_cutoff_claim_is_unavailable() -> None:
    view = _store().view(CUTOFF)
    ids = {c.id for c in view.available()}
    assert ids == {"pre1", "pre2"}
    with pytest.raises(CutoffViolationError):
        view.get("post")


def test_two_claims_from_one_event_share_lineage() -> None:
    store = _store()
    groups = store.lineage_groups()
    assert {c.id for c in groups["e1"]} == {"pre1", "pre2"}


def test_one_event_is_not_multiple_independent_cases() -> None:
    store = _store()
    claim_count, event_count = check_lineage_independence(store.all())
    assert claim_count == 3
    assert event_count == 2  # NOT 3


def test_token_limited_view_does_not_delete_store() -> None:
    store = _store()
    view = store.view(CUTOFF)
    text, ids = view.render_view("prop", limit=1)
    assert len(ids) == 1
    # canonical store still holds everything
    assert len(store.all()) == 3


def test_mark_contradiction_is_symmetric() -> None:
    a = _claim("a", available="2024-01-01T00:00:00+00:00", lineage="e", value="yes")
    b = _claim("b", available="2024-01-01T00:00:00+00:00", lineage="e", value="no")
    na, nb = mark_contradiction(a, b)
    assert "b" in na.contradiction_ids and "a" in nb.contradiction_ids
