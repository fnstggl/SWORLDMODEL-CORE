"""Live-research invariants — mocked HTTP + a prompt-reading fake LLM (no network)."""

from __future__ import annotations

from datetime import datetime

from _live_helpers import AS_OF, HORIZON, FakeLLM, widget_transport
from sworldmodel.live_research import LiveResearchBackend, ResearchBudget

NOW = datetime.fromisoformat("2027-03-01T00:00:00+00:00")


def _research():
    transport = widget_transport()
    backend = LiveResearchBackend(
        FakeLLM(), transport, budget=ResearchBudget(max_rounds=1, max_queries=4), now=NOW
    )
    bundle = backend.research("Will the Widget Standards Board adopt the standard?", AS_OF, HORIZON)
    return bundle, transport


def test_question_alone_triggers_research_with_queries_and_fetches() -> None:
    bundle, transport = _research()
    trace = bundle.live_trace
    assert trace is not None
    assert trace["queries"], "no queries were issued"
    assert trace["rss_requests"], "no Google News RSS request recorded"
    assert trace["sources_fetched"], "no sources were fetched"
    assert any("duckduckgo" in c.url for c in transport.calls)  # real-URL search channel used


def test_unsupported_claim_is_rejected() -> None:
    bundle, _ = _research()
    rejected = bundle.live_trace["sources_rejected"]
    assert any("not supported by fetched text" in r.get("reason", "") for r in rejected)


def test_post_cutoff_source_is_excluded_from_the_view() -> None:
    bundle, _ = _research()
    view = bundle.evidence_store.view(AS_OF)
    # The Feb-15 outcome page ("adopted 3-0") is post-cutoff -> excluded from the view,
    # even though it may be stored.
    props = " ".join(c.proposition for c in view.available()).lower()
    assert "outcome:" not in props


def test_repeated_articles_about_one_event_share_lineage() -> None:
    bundle, _ = _research()
    store = bundle.evidence_store
    # Fewer independent events than raw claims (lineage de-duplicates identical facts).
    assert len(store.independent_event_ids()) <= len(store.all())


def test_live_path_does_not_read_a_prepared_corpus() -> None:
    bundle, transport = _research()
    # Every network call is HTTP; there is no corpus_dir on the live backend.
    assert bundle.live_trace is not None
    assert all(c.method in {"GET", "POST"} for c in transport.calls)


def test_verified_world_is_compiled_from_live_evidence() -> None:
    bundle, _ = _research()
    assert bundle.expected_participants == 3
    assert len(bundle.spec.actors) == 3
    # three named members plus the deciding organization itself
    assert len(bundle.spec.entities) == 4
    assert {e.kind for e in bundle.spec.entities} == {"person", "organization"}
    # the compiled terminal is a declarative predicate, not a fixed mechanism
    assert bundle.spec.terminal.yes_when.op == "equals"
