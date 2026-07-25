"""Live-compilation normalization: ungrounded required facts must not block a run.

Merge regression found by a post-merge live run:

    question : "Will the Federal Reserve announce a reduction in the federal funds
               target range at its September 2026 FOMC meeting?"
    expected : a required-reality fact the model *names* but cannot cite is dropped —
               it is not a verified load-bearing fact — while the evidence-grounded
               gates continue to do the real work
    actual   : ``WorldIntegrityError: required reality facts are unverified — missing:
               ('fomc_has_authority: no evidence attached',)``
    cause    : rebuilt filtered these out in ``universal_compiler._normalize_reality``
               (commit 7720653). That module was deleted by the merge and the filter was
               never carried into ``world_compiler._normalize_compilation``.
    fix      : restore the filter in the canonical compiler. A fact that DOES cite
               evidence is untouched, so a citation unavailable by the cutoff still
               refuses the run — the gate is not weakened, only un-regressed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sworldmodel.evidence import EvidenceClaim, EvidenceStore
from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType
from sworldmodel.world_compiler import _normalize_compilation

AS_OF = datetime.fromisoformat("2026-07-24T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-09-30T00:00:00+00:00")
PUB = datetime.fromisoformat("2026-07-01T00:00:00+00:00")
POST = datetime.fromisoformat("2026-08-01T00:00:00+00:00")  # after the cutoff


def _store() -> EvidenceStore:
    store = EvidenceStore()
    for cid, available in (("c_ok", PUB), ("c_post", POST)):
        store.add(
            EvidenceClaim(
                id=cid,
                proposition=f"fact {cid}",
                normalized_value="v",
                entities=("body",),
                valid_from=None,
                valid_until=None,
                published_at=PUB,
                available_at=available,
                source_id="s",
                source_url="https://example.org",
                source_title="t",
                source_type=SourceType.OFFICIAL_INSTITUTIONAL,
                authority_level=AuthorityLevel.AUTHORITATIVE,
                supporting_excerpt="x",
                lineage_event_id="ev",
                epistemic_type=EpistemicType.OBSERVATION,
                confidence=0.9,
                retrieved_at=PUB,
            )
        )
    return store


def _normalize(required: list[dict[str, Any]]) -> list[dict[str, Any]]:
    data: dict[str, Any] = {"world_spec": {}, "required_reality_facts": required}
    out = _normalize_compilation(data, _store().view(AS_OF), AS_OF, HORIZON)
    return out["required_reality_facts"]


def test_uncited_required_fact_is_dropped_not_blocking() -> None:
    kept = _normalize(
        [
            {
                "key": "fomc_has_authority",
                "description": "the body may set the rate",
                "evidence_claim_ids": [],
            },
            {"key": "roster", "description": "the roster", "evidence_claim_ids": ["c_ok"]},
        ]
    )
    assert [f["key"] for f in kept] == ["roster"]


def test_fact_citing_only_post_cutoff_evidence_is_dropped_too() -> None:
    # The citation exists but is unreachable through the cutoff-bounded view, so the
    # model did not actually ground this fact.
    kept = _normalize(
        [{"key": "future", "description": "a post-cutoff fact", "evidence_claim_ids": ["c_post"]}]
    )
    assert kept == []


def test_grounded_fact_keeps_only_its_available_citations() -> None:
    kept = _normalize(
        [
            {
                "key": "mixed",
                "description": "cites one available and one post-cutoff claim",
                "evidence_claim_ids": ["c_ok", "c_post"],
            }
        ]
    )
    assert len(kept) == 1
    assert kept[0]["evidence_claim_ids"] == ["c_ok"]
