"""Fetch a source page and extract readable text, title, publisher, and date.

Live verification: a claim can only be supported by a page that was actually fetched
and whose text actually contains supporting material. A generated URL is never trusted
on its own.

**Cutoff hygiene.** A pastcast researches a past cutoff today, so the fetch happens
long after ``as_of``. A page's self-declared publication date does not bound its
*content*: a page dated before the cutoff and edited after it serves post-cutoff text
under a pre-cutoff timestamp, and dating the source by that timestamp admits it. So
when the cutoff is in the past this module does not fetch the live page at all. It asks
the Wayback Machine for the last capture at or before the cutoff and fetches that
capture's original bytes (the ``id_`` modifier suppresses the archive's own banner and
link rewriting). The capture timestamp is a *demonstrated* observation time: the
content provably existed then, which is the property the cutoff actually needs. A URL
with no capture at or before the cutoff is not admissible for a pastcast and is
returned with a rejection reason rather than being fetched live.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from .http import HttpError, HttpTransport
from .ids import sha256_hex
from .pdf_text import looks_like_pdf, pdf_metadata_date, pdf_to_text

_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_MULTINL = re.compile(r"\n{3,}")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

# Page furniture. These elements exist on every site and carry no claim about anything:
# site navigation, headers and footers, sidebars, search forms, and the cookie and
# consent dialogs that now open most institutional pages.
_CHROME = re.compile(
    r"<(nav|header|footer|aside|form|noscript|svg|select|button)\b[^>]*>.*?</\1>"
    r"|<[a-z]+\b[^>]*\brole=[\"'](?:navigation|banner|contentinfo|search|dialog|menu"
    r"|menubar|complementary)[\"'][^>]*>.*?</[a-z]+>"
    r"|<[a-z]+\b[^>]*\b(?:id|class)=[\"'][^\"']*(?:cookie|consent|gdpr|onetrust|skip-link"
    r"|breadcrumb|site-nav|mega-menu|social-share)[^\"']*[\"'][^>]*>.*?</[a-z]+>",
    re.IGNORECASE | re.DOTALL,
)

# Where a document's own content lives, most specific first. Non-greedy so a wrapper
# does not swallow the footer, and checked for plausibility by the caller.
_MAIN_REGIONS = (
    re.compile(r"<main\b[^>]*>.*?</main>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<[a-z]+\b[^>]*\brole=[\"']main[\"'][^>]*>.*?</[a-z]+>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<article\b[^>]*>.*?</article>", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"<[a-z]+\b[^>]*\b(?:id|class)=[\"'][^\"']*(?:main-content|page-content|article-body"
        r"|content-block|rich-text)[^\"']*[\"'][^>]*>.*?</[a-z]+>",
        re.IGNORECASE | re.DOTALL,
    ),
)
_META_TIME = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|datePublished|pubdate|date)["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_TIME_TAG = re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE)
_JSONLD = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
WAYBACK_SNAPSHOT = "https://web.archive.org/web"
# Control characters a real text or HTML document never contains. Their presence means
# the body is binary, or was decoded with the wrong codec — either way it is not a
# document we can attribute claims to. U+FFFD is what a failed decode leaves behind.
_NON_TEXT = re.compile(r"[\x00-\x08\x0b\x0e-\x1f�]")


@dataclass(frozen=True)
class ArchiveCapture:
    """A Wayback capture: when the content was observed and where to read it back."""

    timestamp: datetime
    snapshot_url: str
    original_url: str


@dataclass(frozen=True)
class FetchedSource:
    url: str  # the publisher URL this source is attributed to
    fetched_url: str  # the URL actually requested (an archive snapshot for a pastcast)
    final_url: str  # where the request ended up after redirects
    status: int
    reachable: bool
    title: str
    text: str
    publisher: str
    published_at: datetime | None
    archived_at: datetime | None  # capture time, when read from an archive
    fetched_at: datetime
    content_hash: str
    elapsed_ms: int
    rejection_reason: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.reachable and self.status == 200 and bool(self.text) and not self.rejection_reason
        )

    @property
    def observed_at(self) -> datetime:
        """The instant at which *this exact content* is demonstrated to have existed.

        The archive capture time when the bytes came from an archive; otherwise the
        moment we fetched them. This — not the page's self-declared date — is what
        bounds a source for a cutoff.
        """

        return self.archived_at or self.fetched_at

    def evidence_time(self) -> datetime:
        """When the information in this source became available.

        The page's own publication date when it is consistent with ``observed_at``
        (a page cannot have been published after the capture that contains it),
        otherwise the demonstrated observation time. Never later than ``observed_at``,
        so a source can never be dated by a claim it makes about itself.
        """

        declared = self.published_at
        if declared is not None and declared <= self.observed_at:
            return declared
        return self.observed_at


def requires_archived_copy(as_of: datetime | None, now: datetime) -> bool:
    """True when the fetch would happen after the information cutoff.

    That is the whole condition: if we are reading a page later than the cutoff we are
    researching, the live page may have changed since, and only an archived capture at
    or before the cutoff demonstrates what it said. A nowcast (``as_of`` at or after the
    moment research runs, which is what "forecast from today" means) fetches live.

    ``now`` must be the moment the *run* started, not the moment this call happens — see
    :class:`RetrievalMode`.
    """

    return as_of is not None and as_of < now


@dataclass(frozen=True)
class RetrievalMode:
    """Nowcast or pastcast, decided once and carried for the whole run.

    Deciding this per fetch is a trap the previous acceptance run fell into from two
    directions. A cutoff a few hours in the past silently turned an intended nowcast
    into archive-only retrieval, and every un-archived official page was refused —
    which then looked like a research-recall problem rather than the mode error it was.
    And a cutoff set to "now" at launch flips to a pastcast the moment the clock passes
    it, so the same run could fetch live pages early and demand archives later.

    Fixing the comparison instant at process start removes both. It is not a tolerance
    window: a genuine pastcast is exactly as strict as before, since its cutoff is long
    past whenever the process happened to start.
    """

    as_of: datetime
    started_at: datetime
    archived_only: bool

    @classmethod
    def decide(cls, as_of: datetime, started_at: datetime) -> RetrievalMode:
        return cls(
            as_of=as_of,
            started_at=started_at,
            archived_only=requires_archived_copy(as_of, started_at),
        )

    @property
    def name(self) -> str:
        return "pastcast" if self.archived_only else "nowcast"

    @property
    def lag_seconds(self) -> float:
        return (self.started_at - self.as_of).total_seconds()

    @property
    def admissible_sources(self) -> str:
        if self.archived_only:
            return (
                "archived captures at or before the cutoff only; a URL with no such "
                "capture is refused and never fetched live"
            )
        return "current pages, fetched live; claims published after the cutoff stay inadmissible"

    def describe(self) -> str:
        lag = self.lag_seconds  # positive when the run started AFTER the cutoff
        when = (
            f"{abs(lag) / 3600:.1f}h {'after' if lag > 0 else 'before'} the cutoff"
            if abs(lag) >= 60
            else "at the cutoff"
        )
        return (
            f"RETRIEVAL MODE: {self.name.upper()} — cutoff {self.as_of.isoformat()}, "
            f"run started {self.started_at.isoformat()} ({when}). "
            f"Admissible: {self.admissible_sources}."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.name,
            "as_of": self.as_of.isoformat(),
            "process_started_at": self.started_at.isoformat(),
            "cutoff_lag_seconds": self.lag_seconds,
            "archived_captures_required": self.archived_only,
            "admissible_sources": self.admissible_sources,
        }


def fetch_source(
    transport: HttpTransport,
    url: str,
    *,
    now: datetime,
    as_of: datetime | None = None,
    timeout: float = 30.0,
) -> FetchedSource:
    """Fetch ``url`` as evidence for a question with cutoff ``as_of``.

    When the cutoff is in the past the archived capture at or before it is fetched
    instead of the live page; when no such capture exists the source is refused with a
    reason and *not* fetched live.
    """

    capture: ArchiveCapture | None = None
    target = url
    if requires_archived_copy(as_of, now):
        assert as_of is not None
        capture = archived_capture(transport, url, as_of, timeout=timeout)
        if capture is None:
            return _refused(
                url,
                now,
                f"no archived capture at or before the cutoff {as_of.isoformat()}; "
                "the live page cannot be shown to be the pre-cutoff content",
            )
        target = capture.snapshot_url

    try:
        resp = transport.get(target, timeout=timeout)
    except HttpError as exc:
        return _refused(url, now, f"fetch failed: {exc}", fetched_url=target)

    # A PDF (many authoritative minutes/filings) is read from its raw bytes; HTML from
    # its decoded text. A binary body that is neither yields no text.
    if looks_like_pdf(resp.content, content_type=resp.headers.get("content-type", ""), url=url):
        text = pdf_to_text(resp.content)
        title = ""
        published = pdf_metadata_date(resp.content)
    else:
        text = extract_text(resp.text) if _looks_textual(resp.text) else ""
        title = extract_title(resp.text)
        published = extract_published(resp.text)
    # Fall back to the server's Last-Modified header when the body carries no date, so a
    # stable, undated reference page (an official roster page) can still be dated and
    # used, rather than being treated as "now" and excluded from a pastcast.
    if published is None:
        published = _header_date(resp.headers.get("last-modified", ""))
    # A date parsed from a page body may be timezone-naive; make every source date aware
    # (assume UTC) so cutoff comparisons never mix naive and aware datetimes.
    published = _aware(published)

    observed = capture.timestamp if capture else now
    reason = ""
    if published is not None and published > observed:
        # The document claims to be newer than the moment we observed it. For an archive
        # capture that is self-contradictory; for a live fetch it is a future-dated page.
        # Either way its date cannot be used, and it cannot be dated any earlier.
        reason = (
            f"source declares a date ({published.isoformat()}) after the moment its "
            f"content was observed ({observed.isoformat()})"
        )
    return FetchedSource(
        url=url,
        fetched_url=target,
        final_url=resp.final_url,
        status=resp.status,
        reachable=True,
        title=title,
        text=text,
        publisher=_publisher(url if capture else resp.final_url),
        published_at=published,
        archived_at=capture.timestamp if capture else None,
        fetched_at=now,
        # The full digest of the exact text the claims were extracted from: a reader can
        # re-fetch the recorded URL and confirm they are looking at the same document.
        content_hash=sha256_hex(text) if text else "",
        elapsed_ms=resp.elapsed_ms,
        rejection_reason=reason,
    )


# ---------------------------------------------------------------------------
# Wayback Machine
# ---------------------------------------------------------------------------


def wayback_snapshot_url(original_url: str, timestamp: datetime) -> str:
    """The URL that returns a capture's *original* bytes.

    The ``id_`` suffix on the timestamp tells the Wayback Machine to serve the archived
    response as captured — no injected banner, no rewritten links — so the text we
    extract and the excerpts we verify are the publisher's own.
    """

    return f"{WAYBACK_SNAPSHOT}/{timestamp.strftime('%Y%m%d%H%M%S')}id_/{original_url}"


def archived_capture(
    transport: HttpTransport, url: str, as_of: datetime, *, timeout: float = 30.0
) -> ArchiveCapture | None:
    """The last successful Wayback capture of ``url`` at or before ``as_of``.

    Returns ``None`` when the archive has no such capture, when the index is
    unreachable, or when its answer cannot be parsed — in every one of those cases we
    have failed to demonstrate the pre-cutoff content, which is a refusal, not a
    licence to read the live page.
    """

    query = urllib.parse.urlencode(
        {
            "url": url,
            "to": as_of.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
            "limit": "-1",  # the LAST row at or before `to`: the closest capture
            "output": "json",
            "filter": "statuscode:200",
            "fl": "timestamp,original",
        }
    )
    try:
        resp = transport.get(f"{WAYBACK_CDX}?{query}", timeout=timeout)
    except HttpError:
        return None
    if not resp.ok:
        return None
    try:
        rows = json.loads(resp.text or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(rows, list) or len(rows) < 2:  # row 0 is the header
        return None
    last = rows[-1]
    if not isinstance(last, list) or len(last) < 2:
        return None
    stamp = _parse_wayback_timestamp(str(last[0]))
    if stamp is None or stamp > as_of:
        return None
    original = str(last[1]) or url
    return ArchiveCapture(
        timestamp=stamp, snapshot_url=wayback_snapshot_url(original, stamp), original_url=original
    )


def _parse_wayback_timestamp(value: str) -> datetime | None:
    digits = value.strip()
    if len(digits) != 14 or not digits.isdigit():
        return None
    try:
        return datetime.strptime(digits, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _refused(url: str, now: datetime, reason: str, *, fetched_url: str = "") -> FetchedSource:
    return FetchedSource(
        url=url,
        fetched_url=fetched_url or url,
        final_url=url,
        status=0,
        reachable=False,
        title="",
        text="",
        publisher=_publisher(url),
        published_at=None,
        archived_at=None,
        fetched_at=now,
        content_hash="",
        elapsed_ms=0,
        rejection_reason=reason,
    )


# ---------------------------------------------------------------------------
# HTML/text extraction
# ---------------------------------------------------------------------------


def extract_text(html: str) -> str:
    """The document's readable text, with the furniture removed and the body first.

    This used to strip tags and return whatever fell out, in source order. On a modern
    institutional site that means the first several thousand characters are a cookie
    banner, a skip-link list and a mega-menu — and since the extractor reads a bounded
    window, the model was handed navigation and asked what the page established. It
    answered, correctly, that it established nothing: in one acceptance run every single
    page from the Bank of England's site — the minutes, the Monetary Policy Report and
    three speeches — yielded zero claims.

    Two cheap, general steps fix it without a parser dependency. Chrome elements (nav,
    header, footer, aside, forms, cookie dialogs) are dropped by tag and by the ARIA
    roles that mark them. Then, if the markup labels its main content — ``<main>``,
    ``role="main"``, ``<article>``, or the near-universal ``id/class`` containing
    "content" — that region is hoisted to the front, so the window spends itself on the
    document rather than on the site around it.
    """

    body = _SCRIPT_STYLE.sub(" ", html)
    body = _CHROME.sub(" ", body)
    main = _main_region(body)
    if main:
        # Keep the rest: a date, a byline or a breadcrumb can sit outside the main
        # region, and this text is also what the verifier checks excerpts against.
        body = main + "\n\n" + body
    without_tags = _TAG.sub("\n", body)
    unescaped = _unescape(without_tags)
    lines = [_WS.sub(" ", line).strip() for line in unescaped.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return _MULTINL.sub("\n\n", joined).strip()


def _main_region(html: str) -> str:
    """The labeled main-content region, if the markup declares one."""

    for pattern in _MAIN_REGIONS:
        m = pattern.search(html)
        if m:
            region = m.group(0)
            # A wrapper that spans essentially the whole document has told us nothing.
            if len(region) < len(html) * 0.95:
                return region
    return ""


def extract_title(html: str) -> str:
    m = _TITLE.search(html)
    return _unescape(_TAG.sub("", m.group(1))).strip() if m else ""


def extract_published(html: str) -> datetime | None:
    m = _META_TIME.search(html)
    if m and (dt := _parse_dt(m.group(1))):
        return dt
    m = _TIME_TAG.search(html)
    if m and (dt := _parse_dt(m.group(1))):
        return dt
    for block in _JSONLD.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for obj in data if isinstance(data, list) else [data]:
            if (
                isinstance(obj, dict)
                and (val := obj.get("datePublished"))
                and (dt := _parse_dt(str(val)))
            ):
                return dt
    return None


def _parse_dt(value: str) -> datetime | None:
    value = value.strip()
    for candidate in (value, value.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        # Normalize to timezone-aware UTC so cutoff comparisons never mix naive/aware.
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    return None


def _header_date(value: str) -> datetime | None:
    """Parse an HTTP-date header (RFC 7231, e.g. 'Wed, 21 Oct 2026 07:28:00 GMT')."""

    if not value.strip():
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def _aware(dt: datetime | None) -> datetime | None:
    """Attach UTC to a timezone-naive datetime so cutoff comparisons stay consistent."""

    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _looks_textual(text: str) -> bool:
    """Reject undecodable/binary bodies (brotli garbage, challenge pages, images).

    A real text or HTML document contains no C0 control characters beyond whitespace and
    no U+FFFD replacement characters; a body that does was either binary or decoded with
    the wrong codec. This is a structural property of the bytes, not a tuned threshold.
    """

    return bool(text) and not _NON_TEXT.search(text)


def _publisher(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def _unescape(text: str) -> str:
    import html as html_module

    return html_module.unescape(text)
