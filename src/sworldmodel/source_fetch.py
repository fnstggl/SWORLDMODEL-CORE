"""Fetch a source page and extract readable text, title, publisher, and date.

Live verification: a claim can only be supported by a page that was actually fetched
and whose text actually contains supporting material. A generated URL is never
trusted on its own.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime

from .http import HttpError, HttpTransport
from .ids import sha256_hex
from .pdf_text import looks_like_pdf, pdf_metadata_date, pdf_to_text

_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_MULTINL = re.compile(r"\n{3,}")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_TIME = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|datePublished|pubdate|date)["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_TIME_TAG = re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE)
_JSONLD = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class FetchedSource:
    url: str
    final_url: str
    status: int
    reachable: bool
    title: str
    text: str
    publisher: str
    published_at: datetime | None
    fetched_at: datetime
    content_hash: str
    elapsed_ms: int

    @property
    def ok(self) -> bool:
        return self.reachable and self.status == 200 and bool(self.text)


def fetch_source(
    transport: HttpTransport, url: str, *, now: datetime, timeout: float = 30.0
) -> FetchedSource:
    try:
        resp = transport.get(url, timeout=timeout)
    except HttpError:
        return FetchedSource(url, url, 0, False, "", "", _publisher(url), None, now, "", 0)
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
    return FetchedSource(
        url=url,
        final_url=resp.final_url,
        status=resp.status,
        reachable=True,
        title=title,
        text=text,
        publisher=_publisher(resp.final_url),
        published_at=published,
        fetched_at=now,
        content_hash=sha256_hex(text)[:16] if text else "",
        elapsed_ms=resp.elapsed_ms,
    )


def extract_text(html: str) -> str:
    without_scripts = _SCRIPT_STYLE.sub(" ", html)
    without_tags = _TAG.sub("\n", without_scripts)
    unescaped = _unescape(without_tags)
    lines = [_WS.sub(" ", line).strip() for line in unescaped.splitlines()]
    joined = "\n".join(line for line in lines if line)
    return _MULTINL.sub("\n\n", joined).strip()


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
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _looks_textual(text: str) -> bool:
    """Reject undecodable/binary bodies (brotli garbage, challenge pages)."""

    if not text:
        return False
    sample = text[:2000]
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t ")
    return printable / len(sample) >= 0.85


def _publisher(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except ValueError:
        return ""


def _unescape(text: str) -> str:
    import html as html_module

    return html_module.unescape(text)
