"""Web search that yields REAL publisher URLs.

DuckDuckGo's lite endpoint returns result URLs behind a ``uddg`` redirect parameter,
which decode to pages that can actually be fetched and verified. ``site:`` queries reach
official domains directly.

Search **fails closed**. A block page, a challenge, or an error response is not a result
set: it is an HTML page full of the search engine's own links. Harvesting hrefs from it
puts unrelated URLs into the research queue and the run then reports having "researched"
them, so a non-results response is reported as a failure and yields no URLs at all.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

from .http import HttpError, HttpTransport

DDG_LITE = "https://lite.duckduckgo.com/lite/"
_UDDG = re.compile(r"uddg=([^&\"']+)")


@dataclass(frozen=True)
class SearchOutcome:
    """The result of one search: URLs, or an explicit reason there are none."""

    query: str
    urls: tuple[str, ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def duckduckgo_search(transport: HttpTransport, query: str, *, limit: int = 8) -> SearchOutcome:
    """Run one search. On any non-results response the outcome carries ``error`` and no
    URLs — the caller must surface it rather than proceeding as if the search ran."""

    url = DDG_LITE + "?" + urllib.parse.urlencode({"q": query})
    try:
        resp = transport.get(url, timeout=25)
    except HttpError as exc:
        return SearchOutcome(query, error=f"search request failed: {exc}")
    if not resp.ok:
        return SearchOutcome(query, error=f"search returned HTTP {resp.status}")

    urls: list[str] = []
    seen: set[str] = set()
    for enc in _UDDG.findall(resp.text):
        real = urllib.parse.unquote(enc)
        if real.startswith(("http://", "https://")) and real not in seen:
            seen.add(real)
            urls.append(real)
    if not urls:
        # No result links at all. That is either a genuinely empty result set or a block
        # or challenge page, and the two are not distinguishable from here — so it is
        # reported, never papered over by scraping whatever links the page did contain.
        return SearchOutcome(
            query, error="search returned no result links (empty result set, block, or challenge)"
        )
    return SearchOutcome(query, urls=tuple(urls[:limit]))


def site_query(domain: str, terms: str) -> str:
    return f"site:{domain} {terms}".strip()
