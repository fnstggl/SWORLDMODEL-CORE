"""Web search that yields REAL publisher URLs.

Google News RSS links are obfuscated redirects that do not resolve to article text,
so RSS is used only for discovery signals; DuckDuckGo's lite endpoint returns real
result URLs (via the ``uddg`` redirect parameter) that can actually be fetched and
verified. ``site:`` queries reach official domains directly.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

from .http import HttpError, HttpTransport

DDG_LITE = "https://lite.duckduckgo.com/lite/"
_UDDG = re.compile(r"uddg=([^&\"']+)")
_DIRECT = re.compile(r'href="(https?://[^"]+)"')


@dataclass(frozen=True)
class SearchResult:
    url: str


def duckduckgo_search(transport: HttpTransport, query: str, *, limit: int = 8) -> list[str]:
    url = DDG_LITE + "?" + urllib.parse.urlencode({"q": query})
    try:
        resp = transport.get(url, timeout=25)
    except HttpError:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for enc in _UDDG.findall(resp.text):
        real = urllib.parse.unquote(enc)
        if real.startswith("http") and real not in seen:
            seen.add(real)
            urls.append(real)
    if not urls:  # fall back to any direct external href
        for href in _DIRECT.findall(resp.text):
            if "duckduckgo" not in href and href not in seen:
                seen.add(href)
                urls.append(href)
    return urls[:limit]


def site_query(domain: str, terms: str) -> str:
    return f"site:{domain} {terms}".strip()
