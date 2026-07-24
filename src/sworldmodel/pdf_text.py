"""Dependency-free PDF text extraction.

Many authoritative primary records — central-bank minutes, official announcements,
court filings, regulatory decisions — are published as PDFs. A pipeline that can only
read HTML cannot verify a roster or a vote that lives in a minutes PDF, so this module
extracts readable text from a PDF using only the standard library (``zlib`` for the
usual FlateDecode streams). It is intentionally simple: it recovers the text-showing
operators from content streams, which covers digitally-generated text PDFs. It does not
OCR scanned images, and returns "" when it cannot recover meaningful text.
"""

from __future__ import annotations

import re
import zlib
from datetime import UTC, datetime, timedelta, timezone

_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_PDF_DATE = re.compile(rb"/(?:CreationDate|ModDate)\s*\(\s*D:(\d{8,14})([Zz+\-]\d{2}'?\d{2}'?)?")
_STR = re.compile(rb"\((?:[^()\\]|\\.)*\)")
_TJ = re.compile(rb"\((?:[^()\\]|\\.)*\)\s*Tj")
_TJ_ARRAY = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
_ESCAPED = re.compile(rb"\\([()\\])")


def looks_like_pdf(content: bytes, *, content_type: str = "", url: str = "") -> bool:
    """Detect a PDF by magic bytes, declared content type, or URL suffix."""

    if content[:5] == b"%PDF-":
        return True
    if "application/pdf" in content_type.lower():
        return True
    return url.split("?", 1)[0].lower().endswith(".pdf")


def pdf_to_text(content: bytes, *, max_chars: int = 200_000) -> str:
    """Extract readable text from PDF bytes. Returns "" if nothing usable is found."""

    if content[:5] != b"%PDF-":
        return ""
    parts: list[str] = []
    total = 0
    for match in _STREAM.finditer(content):
        raw = match.group(1)
        # FlateDecode is by far the most common; fall back to the raw bytes for an
        # uncompressed content stream (an image stream stays binary and matches nothing).
        chunk = _inflate(raw) or raw
        if not chunk:
            continue
        for tok in _TJ.finditer(chunk):
            parts.append(_decode_str(tok.group(0)))
        for arr in _TJ_ARRAY.finditer(chunk):
            parts.append("".join(_decode_str(s.group(0)) for s in _STR.finditer(arr.group(1))))
        total += len(chunk)
        if total > max_chars * 8:  # bound work on very large files
            break
    text = "\n".join(p for p in parts if p.strip())
    return _tidy(text)[:max_chars]


def pdf_metadata_date(content: bytes) -> datetime | None:
    """The document's own creation/modification date from its metadata (``/CreationDate``
    or ``/ModDate``). This is the reliable publication date used for cutoff enforcement;
    the *earliest* metadata date found is taken, since a document cannot predate it."""

    dates: list[datetime] = []
    for m in _PDF_DATE.finditer(content):
        dt = _parse_pdf_date(m.group(1), m.group(2))
        if dt is not None:
            dates.append(dt)
    return min(dates) if dates else None


def _parse_pdf_date(digits: bytes, offset: bytes | None) -> datetime | None:
    s = digits.decode("ascii", "ignore").ljust(14, "0")[:14]
    try:
        naive = datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    tz = UTC
    if offset and offset[:1] in (b"+", b"-"):
        try:
            o = offset.decode("ascii", "ignore").replace("'", "")
            sign = 1 if o[0] == "+" else -1
            tz = timezone(sign * timedelta(hours=int(o[1:3]), minutes=int(o[3:5] or "0")))
        except (ValueError, IndexError):
            tz = UTC
    return naive.replace(tzinfo=tz)


def _inflate(raw: bytes) -> bytes:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(raw, wbits)
        except zlib.error:
            continue
    return b""


def _decode_str(tok: bytes) -> str:
    start, end = tok.find(b"("), tok.rfind(b")")
    if start < 0 or end <= start:
        return ""
    body = tok[start + 1 : end]
    body = _ESCAPED.sub(rb"\1", body)
    body = body.replace(rb"\n", b"\n").replace(rb"\t", b"\t").replace(rb"\r", b"")
    try:
        return body.decode("latin-1")
    except (UnicodeDecodeError, ValueError):
        return ""


def _tidy(text: str) -> str:
    # PDFs frequently break words across text tokens; collapse runs of spaces and
    # stray single newlines while keeping paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
