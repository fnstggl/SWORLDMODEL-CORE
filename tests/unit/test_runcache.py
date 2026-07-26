"""The record-and-replay caches: replay only, provenance intact, verification re-run.

A cache in this system may never do new work or weaken a gate: the source cache
replays a document with its ORIGINAL observation times (so cutoff admissibility is
judged exactly as the live path judged it), and the extraction cache replays the
recorded model output through the same verification a fresh call gets.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from _fakes import ProgrammableGateway
from sworldmodel.runcache import ExtractionCache, SourceCache, caching_disabled
from sworldmodel.source_extract import extract_claims
from sworldmodel.source_fetch import FetchedSource

AS_OF = datetime.fromisoformat("2027-01-15T00:00:00+00:00")
FETCHED_AT = datetime.fromisoformat("2027-01-10T00:00:00+00:00")


@pytest.fixture()
def caches_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("SWORLDMODEL_CACHE", "on")
    monkeypatch.setenv("SWORLDMODEL_SOURCE_CACHE_DIR", str(tmp_path / "sources"))
    monkeypatch.setenv("SWORLDMODEL_EXTRACT_CACHE_DIR", str(tmp_path / "extractions"))
    return tmp_path


def _source(text: str = "The registrar approved the filing on 5 January.") -> FetchedSource:
    return FetchedSource(
        url="https://registry.example/decisions",
        fetched_url="https://registry.example/decisions",
        final_url="https://registry.example/decisions",
        status=200,
        reachable=True,
        title="Decisions",
        text=text,
        publisher="registry.example",
        published_at=FETCHED_AT,
        archived_at=None,
        fetched_at=FETCHED_AT,
        content_hash="abc123",
        elapsed_ms=10,
    )


def test_the_kill_switch_disables_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWORLDMODEL_CACHE", "off")
    assert caching_disabled()
    cache = SourceCache()
    cache.store(_source(), archive_cutoff=None)
    assert cache.load("https://registry.example/decisions", archive_cutoff=None) is None


def test_source_replay_keeps_the_original_provenance(caches_on: Path) -> None:
    cache = SourceCache()
    cache.store(_source(), archive_cutoff=None)
    replayed = cache.load("https://registry.example/decisions", archive_cutoff=None)
    assert replayed is not None
    assert replayed.ok
    assert replayed.text == _source().text
    # The ORIGINAL fetch time survives: admissibility at a cutoff is judged against
    # when the content was demonstrably observed, not when it was replayed.
    assert replayed.fetched_at == FETCHED_AT
    assert replayed.evidence_time() == FETCHED_AT
    assert replayed.content_hash == "abc123"


def test_rejected_sources_are_never_stored(caches_on: Path) -> None:
    from dataclasses import replace

    cache = SourceCache()
    refused = replace(_source(), rejection_reason="no archived capture", reachable=False)
    cache.store(refused, archive_cutoff=None)
    assert cache.load(refused.url, archive_cutoff=None) is None


def test_retrieval_regimes_do_not_share_entries(caches_on: Path) -> None:
    """A live-fetched page must never satisfy a pastcast's archive-only lookup."""

    cache = SourceCache()
    cache.store(_source(), archive_cutoff=None)
    assert cache.load("https://registry.example/decisions", archive_cutoff=AS_OF) is None


def test_expired_entries_are_not_served(caches_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWORLDMODEL_SOURCE_CACHE_TTL", "0")
    cache = SourceCache()
    cache.store(_source(), archive_cutoff=None)
    import time

    time.sleep(0.01)
    assert cache.load("https://registry.example/decisions", archive_cutoff=None) is None


# ---------------------------------------------------------------------------
# Extraction cache through the real extract_claims path
# ---------------------------------------------------------------------------

_CLAIMS = {
    "claims": [
        {
            "proposition": "decision: the registrar approved the filing",
            "normalized_value": "approved",
            "entities": ["registrar"],
            "epistemic_type": "observation",
            "supporting_excerpt": "The registrar approved the filing on 5 January.",
            "authority_hint": 3,
        }
    ]
}


def test_extraction_replays_by_content_model_and_prompt(caches_on: Path) -> None:
    cache = ExtractionCache()
    gw = ProgrammableGateway({"extract_claims": _CLAIMS})
    first = extract_claims(gw, "was the filing approved?", _source(), AS_OF, cache=cache)
    assert not first.from_cache
    assert gw.call_count == 1
    assert first.claims[0].verified_in_text

    second = extract_claims(gw, "was the filing approved?", _source(), AS_OF, cache=cache)
    assert second.from_cache
    assert gw.call_count == 1, "the identical extraction must not re-call the provider"
    assert [c.proposition for c in second.claims] == [c.proposition for c in first.claims]

    # A different question is a different prompt — a different key, a fresh call.
    third = extract_claims(gw, "who is the registrar?", _source(), AS_OF, cache=cache)
    assert not third.from_cache
    assert gw.call_count == 2

    # Different content misses too.
    fourth = extract_claims(
        gw, "was the filing approved?", _source("Entirely different text."), AS_OF, cache=cache
    )
    assert not fourth.from_cache
    assert gw.call_count == 3


def test_a_cached_output_is_still_verified_not_trusted(caches_on: Path) -> None:
    """The cache stores the model's raw output; verification re-runs on replay. A
    recorded claim whose excerpt is not in the document is rejected on replay exactly
    as it would be live."""

    bogus = {
        "claims": [
            {
                "proposition": "decision: the registrar rejected everything",
                "normalized_value": "rejected",
                "entities": ["registrar"],
                "epistemic_type": "observation",
                "supporting_excerpt": "this sentence is not in the document",
                "authority_hint": 3,
            }
        ]
    }
    cache = ExtractionCache()
    gw = ProgrammableGateway({"extract_claims": bogus})
    live = extract_claims(gw, "q?", _source(), AS_OF, cache=cache)
    replay = extract_claims(gw, "q?", _source(), AS_OF, cache=cache)
    assert replay.from_cache
    assert not live.claims[0].verified_in_text
    assert not replay.claims[0].verified_in_text
    assert replay.claims[0].rejection_reason == live.claims[0].rejection_reason
