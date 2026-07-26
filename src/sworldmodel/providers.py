"""Retrieval providers beyond direct fetch, and the health ledger that governs them.

The retrieval chain is ordered by directness, and each rung exists because the one
above it has a real failure mode observed live:

1. official feeds and direct publisher fetch (elsewhere: ``rss``, ``source_fetch``);
2. Google News RSS + the batchexecute resolver (``gnews_decode``);
3. **Jina Reader** — reads a *known* URL through ``r.jina.ai`` when the direct fetch
   is blocked, challenged, empty, JavaScript-only or truncated;
4. **Jina Search** — ``s.jina.ai``, used narrowly for exact-title recovery when the
   Google News resolver fails: ``"EXACT HEADLINE" "PUBLISHER"``, never a generic
   subject query;
5. **Serper** — Google results via ``google.serper.dev``, the last discovery
   fallback when the rungs above have not produced a needed source;
6. DuckDuckGo stays available as an optional low-priority extra (``search``), no
   longer load-bearing: its blocks were the direct cause of an empty-evidence pass.

Keys come from the environment (``JINA_API_KEY``, ``SERPER_API_KEY``) and are never
logged, traced or persisted; every diagnostic path here carries URLs and counts only.

:class:`ProviderHealth` is the circuit breaker. Every provider call is scored; a
provider that fails repeatedly is rested for the remainder of the run rather than
asked dozens more times — a live pass spent its whole discovery budget re-asking a
blocked engine and compiled an empty world from the silence.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from dataclasses import dataclass, field

from .http import HttpError, HttpResponse, HttpTransport

# Circuit-breaker threshold: consecutive clear failures before a provider is rested.
_TRIP_AFTER = 3


@dataclass
class ProviderStats:
    requests: int = 0
    successes: int = 0
    empties: int = 0
    blocks: int = 0
    timeouts: int = 0
    errors: int = 0
    consecutive_failures: int = 0
    tripped: bool = False

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "requests": self.requests,
            "successes": self.successes,
            "empties": self.empties,
            "blocks": self.blocks,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "consecutive_failures": self.consecutive_failures,
            "tripped": self.tripped,
        }


@dataclass
class ProviderHealth:
    """Per-provider scoreboard with circuit breaking, shared across one question run."""

    stats: dict[str, ProviderStats] = field(default_factory=dict)

    def _of(self, provider: str) -> ProviderStats:
        return self.stats.setdefault(provider, ProviderStats())

    def usable(self, provider: str) -> bool:
        return not self._of(provider).tripped

    def note_success(self, provider: str) -> None:
        s = self._of(provider)
        s.requests += 1
        s.successes += 1
        s.consecutive_failures = 0

    def note_empty(self, provider: str) -> None:
        # An empty result set is an answer, not a failure — a rare query legitimately
        # matches nothing — but a run of them from one engine is a block wearing an
        # empty result's clothes, so empties count toward the trip threshold.
        s = self._of(provider)
        s.requests += 1
        s.empties += 1
        s.consecutive_failures += 1
        self._maybe_trip(s)

    def note_block(self, provider: str) -> None:
        s = self._of(provider)
        s.requests += 1
        s.blocks += 1
        s.consecutive_failures += 1
        self._maybe_trip(s)

    def note_timeout(self, provider: str) -> None:
        s = self._of(provider)
        s.requests += 1
        s.timeouts += 1
        s.consecutive_failures += 1
        self._maybe_trip(s)

    def note_error(self, provider: str) -> None:
        s = self._of(provider)
        s.requests += 1
        s.errors += 1
        s.consecutive_failures += 1
        self._maybe_trip(s)

    def _maybe_trip(self, s: ProviderStats) -> None:
        if s.consecutive_failures >= _TRIP_AFTER:
            s.tripped = True

    def as_dict(self) -> dict[str, dict[str, int | bool]]:
        return {name: s.as_dict() for name, s in sorted(self.stats.items())}


# ---------------------------------------------------------------------------
# Jina Reader — read a KNOWN url as clean text
# ---------------------------------------------------------------------------

# Jina sits behind Cloudflare, which rejects the default urllib identity outright
# (error 1010) before Jina ever sees the request.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def jina_key() -> str:
    return os.environ.get("JINA_API_KEY", "")


def serper_key() -> str:
    return os.environ.get("SERPER_API_KEY", "")


@dataclass(frozen=True)
class ReaderResult:
    url: str
    text: str
    failure: str | None = None  # None on success

    @property
    def ok(self) -> bool:
        return self.failure is None and bool(self.text.strip())


def jina_reader(
    transport: HttpTransport,
    url: str,
    health: ProviderHealth,
    *,
    timeout: float = 45.0,
) -> ReaderResult:
    """Read ``url`` through r.jina.ai. For known URLs that direct fetch cannot read."""

    provider = "jina_reader"
    key = jina_key()
    if not key:
        return ReaderResult(url, "", "no JINA_API_KEY configured")
    if not health.usable(provider):
        return ReaderResult(url, "", "provider rested after repeated failures")
    try:
        resp = transport.get(
            f"https://r.jina.ai/{url}",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": _BROWSER_UA,
                "X-Return-Format": "markdown",
            },
            timeout=timeout,
        )
    except HttpError as exc:
        _score_exception(health, provider, exc)
        return ReaderResult(url, "", f"network: {exc}")
    failure = _score_response(health, provider, resp, min_useful_chars=200)
    if failure:
        return ReaderResult(url, "", failure)
    return ReaderResult(url, resp.text)


# ---------------------------------------------------------------------------
# Jina Search — exact-title recovery, not a general engine
# ---------------------------------------------------------------------------


def jina_title_search(
    transport: HttpTransport,
    headline: str,
    publisher: str,
    health: ProviderHealth,
    *,
    timeout: float = 45.0,
) -> list[str]:
    """URLs for ``"EXACT HEADLINE" "PUBLISHER"`` via s.jina.ai, best first.

    This is the recovery rung for a Google News item whose link would not decode: the
    headline and publisher are known exactly, so the query is exact. It is never used
    for subject queries — Search costs an order of magnitude more than Reader.
    """

    provider = "jina_search"
    key = jina_key()
    if not key or not health.usable(provider):
        return []
    query = urllib.parse.quote(f'"{headline}" "{publisher}"')
    try:
        resp = transport.get(
            f"https://s.jina.ai/{query}",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": _BROWSER_UA,
                "Accept": "application/json",
                "X-Respond-With": "no-content",
            },
            timeout=timeout,
        )
    except HttpError as exc:
        _score_exception(health, provider, exc)
        return []
    if _score_response(health, provider, resp, min_useful_chars=2):
        return []
    return _jina_search_urls(resp.text)


def _jina_search_urls(body: str) -> list[str]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            url = row.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                out.append(url)
    return out


# ---------------------------------------------------------------------------
# Serper — the last discovery fallback
# ---------------------------------------------------------------------------


def serper_search(
    transport: HttpTransport,
    query: str,
    health: ProviderHealth,
    *,
    limit: int = 8,
    timeout: float = 30.0,
) -> list[str]:
    """Organic result URLs from google.serper.dev, best first.

    Discovery only: every URL still goes through the canonical fetch → extract →
    verify pipeline; nothing a search engine says is evidence.
    """

    provider = "serper"
    key = serper_key()
    if not key or not health.usable(provider):
        return []
    try:
        resp = transport.post_json(
            "https://google.serper.dev/search",
            {"q": query, "num": max(limit, 10)},
            headers={"X-API-KEY": key},
            timeout=timeout,
        )
    except HttpError as exc:
        _score_exception(health, provider, exc)
        return []
    if _score_response(health, provider, resp, min_useful_chars=2):
        return []
    return _serper_urls(resp.text, limit)


def _serper_urls(body: str, limit: int) -> list[str]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for row in data.get("organic", []) if isinstance(data, dict) else []:
        if isinstance(row, dict):
            link = row.get("link")
            if isinstance(link, str) and link.startswith(("http://", "https://")):
                out.append(link)
    # An answer box or knowledge graph names a source too; keep organic first.
    if isinstance(data, dict):
        for extra_key in ("answerBox", "knowledgeGraph"):
            extra = data.get(extra_key)
            if isinstance(extra, dict):
                link = extra.get("link") or extra.get("website")
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    out.append(link)
    return list(dict.fromkeys(out))[:limit]


# ---------------------------------------------------------------------------
# Shared scoring
# ---------------------------------------------------------------------------


def _score_exception(health: ProviderHealth, provider: str, exc: Exception) -> None:
    if "timed out" in str(exc).lower():
        health.note_timeout(provider)
    else:
        health.note_error(provider)


def _score_response(
    health: ProviderHealth, provider: str, resp: HttpResponse, *, min_useful_chars: int
) -> str | None:
    """Score one reply; returns the failure description, or None when usable."""

    if resp.status in (401, 402, 403):
        # 402 is Jina's InsufficientBalanceError: the key exists but has no tokens.
        # That is an account state, not a moment — it will fail identically all run,
        # so it counts as a block and trips quickly.
        health.note_block(provider)
        return f"blocked (HTTP {resp.status}): {resp.text[:120]}"
    if resp.status == 429:
        health.note_block(provider)
        return "rate limited (HTTP 429)"
    if resp.status >= 400:
        health.note_error(provider)
        return f"HTTP {resp.status}"
    if len(resp.text.strip()) < min_useful_chars:
        health.note_empty(provider)
        return "empty response"
    health.note_success(provider)
    return None
