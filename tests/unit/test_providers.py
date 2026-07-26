"""The provider stack: health scoring, circuit breaking, and reply parsing.

Live behaviour observed 2026-07 and replayed here: Serper answers with organic
results; Jina answers 402 InsufficientBalanceError when the key has no tokens, and a
blocked key fails identically all run, so it must trip rather than being re-asked.
"""

from __future__ import annotations

import json

from _fakes import FakeTransport
from sworldmodel.http import HttpResponse
from sworldmodel.providers import (
    ProviderHealth,
    jina_reader,
    jina_title_search,
    serper_search,
)


def _json(url: str, payload: object, status: int = 200) -> HttpResponse:
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    return HttpResponse(url, url, status, {"content-type": "application/json"}, text, 1)


def test_serper_organic_results_are_parsed_best_first(monkeypatch) -> None:
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    t = FakeTransport().add(
        lambda u: "serper.dev" in u,
        _json(
            "s",
            {
                "organic": [
                    {"link": "https://a.example/1"},
                    {"link": "https://b.example/2"},
                    {"link": "https://a.example/1"},  # dupe collapses
                ],
                "answerBox": {"link": "https://c.example/answer"},
            },
        ),
    )
    h = ProviderHealth()
    urls = serper_search(t, "q", h)
    assert urls == ["https://a.example/1", "https://b.example/2", "https://c.example/answer"]
    assert h.stats["serper"].successes == 1


def test_a_blocked_provider_trips_and_is_not_asked_again(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    t = FakeTransport().add(
        lambda u: "r.jina.ai" in u,
        _json("r", {"code": 402, "name": "InsufficientBalanceError"}, status=402),
    )
    h = ProviderHealth()
    for _ in range(3):
        r = jina_reader(t, "https://pub.example/story", h)
        assert not r.ok
    assert h.stats["jina_reader"].tripped
    calls = len(t.calls)
    r = jina_reader(t, "https://pub.example/story", h)
    assert r.failure == "provider rested after repeated failures"
    assert len(t.calls) == calls  # rested: no request spent


def test_a_missing_key_is_a_named_condition_not_a_request(monkeypatch) -> None:
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    t = FakeTransport()
    r = jina_reader(t, "https://pub.example/story", ProviderHealth())
    assert r.failure == "no JINA_API_KEY configured"
    assert t.calls == []


def test_title_search_builds_the_exact_title_query(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    seen: list[str] = []

    def capture(method: str, url: str, body: object) -> HttpResponse:
        seen.append(url)
        return _json(url, {"data": [{"url": "https://pub.example/found"}]})

    t = FakeTransport().add(lambda u: "s.jina.ai" in u, capture)
    urls = jina_title_search(t, "OPEC Agrees Another Hike", "Reuters", ProviderHealth())
    assert urls == ["https://pub.example/found"]
    # The query is the exact headline plus the publisher, quoted — not a subject query.
    assert "%22OPEC%20Agrees%20Another%20Hike%22%20%22Reuters%22" in seen[0]


def test_no_secret_ever_reaches_a_result_or_the_call_log(monkeypatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "jina_SECRET_VALUE")
    t = FakeTransport().add(lambda u: "r.jina.ai" in u, _json("r", "# doc text " * 40))
    r = jina_reader(t, "https://pub.example/story", ProviderHealth())
    assert r.ok
    dump = json.dumps([c.__dict__ for c in t.calls]) + r.text + (r.failure or "")
    assert "SECRET_VALUE" not in dump
