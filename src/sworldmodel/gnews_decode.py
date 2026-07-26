"""Resolve a Google News article link to the publisher URL it stands for.

Google News RSS is a real discovery channel — headlines, publishers and publication
times for exactly the query asked — but every item link is an opaque
``news.google.com/rss/articles/CBMi…`` reference. The id no longer encodes the
destination (the old base64 trick recovers nothing), and following the link yields a
JavaScript interstitial with no publisher URL in it. The working resolution is the
``batchexecute`` flow the Google News web app itself uses:

1. fetch the article's own Google News page and read the ``data-n-a-sg`` (signature)
   and ``data-n-a-ts`` (timestamp) attributes it carries;
2. POST a ``garturlreq`` envelope with the id, signature and timestamp to
   ``/_/DotsSplashUi/data/batchexecute``;
3. the reply's payload contains the publisher URL.

This module implements that flow over the project's own :class:`HttpTransport`, so the
fetch policy, size caps, redirect validation and per-call recording apply to these
requests exactly as to every other. The maintained ``googlenewsdecoder`` package
(MIT, audited 2026-07) was the reference for the protocol; it is not imported at
runtime because it performs raw ``requests`` calls with no timeout and no policy,
outside every safety property this transport enforces.

Failure is explicit and typed. A CAPTCHA or consent interstitial, a throttle, and a
protocol change (the reply shape moving under us) are each named in the result — the
resolver never invents a URL and never guesses. The caller decides what a failure
means; exact-title search recovery is the next rung, not this module's business.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass

from .http import HttpError, HttpTransport

_BATCH_ENDPOINT = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
# The browser identity Google News serves real attributes to. The default urllib
# identity receives a degraded page with no data-n-a-* attributes at all.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
_SIG = re.compile(r'data-n-a-sg="([^"]+)"')
_TS = re.compile(r'data-n-a-ts="([^"]+)"')
# Markers of Google's block/consent pages, which arrive with status 200.
_CHALLENGE_MARKERS = (
    "unusual traffic",
    "captcha",
    "consent.google.com",
    "enablejs",
    "detected unusual",
)


@dataclass(frozen=True)
class DecodeResult:
    """One resolution attempt, successful or not, with the reason preserved."""

    google_url: str
    article_id: str
    publisher_url: str | None
    failure: str | None = None  # None on success; else one of the kinds below
    # "not_a_gnews_url" | "challenge" | "throttled" | "protocol_change" | "network"

    @property
    def ok(self) -> bool:
        return self.publisher_url is not None


def article_id_of(url: str) -> str | None:
    """The opaque article id inside a Google News link, or None for other URLs."""

    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if (parts.hostname or "").lower() != "news.google.com":
        return None
    path = [p for p in parts.path.split("/") if p]
    if len(path) >= 2 and path[-2] in ("articles", "read"):
        return path[-1]
    return None


class GoogleNewsDecoder:
    """Stateful resolver: one signature fetch and one batchexecute POST per article,
    with an id-keyed cache so a headline that appears under several queries costs one
    resolution."""

    def __init__(self, transport: HttpTransport, *, interval_seconds: float = 0.2) -> None:
        self.transport = transport
        self.interval_seconds = interval_seconds
        self._cache: dict[str, DecodeResult] = {}
        self._last_request = 0.0

    def decode(self, google_url: str) -> DecodeResult:
        article_id = article_id_of(google_url)
        if article_id is None:
            return DecodeResult(google_url, "", None, "not_a_gnews_url")
        cached = self._cache.get(article_id)
        if cached is not None:
            return DecodeResult(google_url, article_id, cached.publisher_url, cached.failure)
        result = self._decode_fresh(google_url, article_id)
        # Challenges and throttles are moment-shaped; do not pin them to the id.
        if result.ok or result.failure == "protocol_change":
            self._cache[article_id] = result
        return result

    # -- the two-step protocol ------------------------------------------------

    def _decode_fresh(self, google_url: str, article_id: str) -> DecodeResult:
        self._pace()
        params = self._signature(article_id)
        if isinstance(params, str):
            return DecodeResult(google_url, article_id, None, params)
        signature, timestamp = params

        self._pace()
        payload = [
            "Fbv4je",
            '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,'
            'null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],'
            f'"{article_id}",{timestamp},"{signature}"]',
        ]
        form = "f.req=" + urllib.parse.quote(json.dumps([[payload]]))
        try:
            resp = self.transport.post_form(
                _BATCH_ENDPOINT,
                form,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "User-Agent": _UA,
                },
                timeout=20.0,
            )
        except HttpError as exc:
            return DecodeResult(google_url, article_id, None, f"network: {exc}")
        if resp.status == 429:
            return DecodeResult(google_url, article_id, None, "throttled")
        if not resp.ok or _challenged(resp.text):
            return DecodeResult(google_url, article_id, None, "challenge")
        url = _publisher_url_in(resp.text)
        if url is None:
            # The reply parsed as something other than the shape this flow relies on:
            # Google moved the protocol. Say so — this is the failure a maintainer
            # needs to hear about, and silently degrading would hide it.
            return DecodeResult(google_url, article_id, None, "protocol_change")
        return DecodeResult(google_url, article_id, url)

    def _signature(self, article_id: str) -> tuple[str, str] | str:
        """(signature, timestamp) from the article's own page, or a failure kind."""

        last = "network: no attempt made"
        for base in ("articles", "rss/articles"):
            try:
                resp = self.transport.get(
                    f"https://news.google.com/{base}/{article_id}",
                    headers={"User-Agent": _UA},
                    timeout=20.0,
                )
            except HttpError as exc:
                last = f"network: {exc}"
                continue
            if resp.status == 429:
                return "throttled"
            if _challenged(resp.text):
                return "challenge"
            sig, ts = _SIG.search(resp.text), _TS.search(resp.text)
            if sig and ts:
                return sig.group(1), ts.group(1)
            last = "protocol_change"
        return last

    def _pace(self) -> None:
        """Space requests out. Google throttles this endpoint, and a throttle costs
        every remaining resolution in the run."""

        wait = self.interval_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()


def _challenged(text: str) -> bool:
    low = text[:4000].lower()
    return any(m in low for m in _CHALLENGE_MARKERS)


def _publisher_url_in(reply: str) -> str | None:
    """The publisher URL inside a batchexecute reply, or None if the shape moved.

    The reply is anti-XSSI framed: a throwaway first line, then length-prefixed JSON
    chunks. The URL lives at ``json(parsed[0][2])[1]`` today; every step is checked so
    a protocol change surfaces as None rather than an exception or a wrong URL.
    """

    try:
        chunks = reply.split("\n\n")
        parsed = json.loads(chunks[1])[:-2]
        inner = json.loads(parsed[0][2])
        url = inner[1]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    return url
