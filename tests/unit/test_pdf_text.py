"""Dependency-free PDF text extraction."""

from __future__ import annotations

import zlib
from datetime import UTC

from sworldmodel.pdf_text import looks_like_pdf, pdf_metadata_date, pdf_to_text


def _pdf_with_stream(content_stream: bytes, *, compress: bool) -> bytes:
    body = zlib.compress(content_stream) if compress else content_stream
    return b"%PDF-1.7\n1 0 obj\n<< >>\nstream\n" + body + b"\nendstream\nendobj\n%%EOF"


def test_extracts_text_from_a_flate_stream() -> None:
    stream = b"BT /F1 12 Tf (Victoria Rodriguez, Governor) Tj ET"
    text = pdf_to_text(_pdf_with_stream(stream, compress=True))
    assert "Victoria Rodriguez, Governor" in text


def test_extracts_from_a_tj_array_and_uncompressed_stream() -> None:
    stream = b"BT [(the Board decided )-250(unanimously)] TJ ET"
    text = pdf_to_text(_pdf_with_stream(stream, compress=False))
    assert "the Board decided" in text
    assert "unanimously" in text


def test_non_pdf_bytes_yield_no_text() -> None:
    assert pdf_to_text(b"<html><body>not a pdf</body></html>") == ""


def test_metadata_date_is_the_latest_not_the_earliest() -> None:
    # A document is only as old as its most recent modification: the text extracted is
    # the text as it stands now. Dating this PDF by its creation date would admit a
    # document edited after an information cutoff into a pastcast under a pre-cutoff
    # timestamp, which is exactly the leak the latest-date rule closes.
    pdf = b"%PDF-1.7\n<< /CreationDate (D:20260415120000-06'00') /ModDate (D:20260420) >>\n%%EOF"
    dt = pdf_metadata_date(pdf)
    assert dt is not None
    assert (dt.year, dt.month, dt.day) == (2026, 4, 20)
    assert pdf_metadata_date(b"%PDF-1.7 no dates here") is None


def test_looks_like_pdf_detects_magic_type_and_suffix() -> None:
    assert looks_like_pdf(b"%PDF-1.4 ...")
    assert looks_like_pdf(b"", content_type="application/pdf; charset=binary")
    assert looks_like_pdf(b"", url="https://x.example/minutes.pdf?y=1")
    assert not looks_like_pdf(b"<html>", url="https://x.example/page.html")


def test_fetch_source_reads_a_pdf_body() -> None:
    from datetime import datetime

    from sworldmodel.http import FakeTransport, pdf_response
    from sworldmodel.source_fetch import fetch_source

    stream = b"BT (Galia Borja, Deputy governor) Tj ET"
    pdf = _pdf_with_stream(stream, compress=True)
    url = "https://bank.example/minutes.pdf"
    transport = FakeTransport().add_url(url, pdf_response(url, pdf))
    fs = fetch_source(transport, url, now=datetime.now(UTC))
    assert fs.ok
    assert "Galia Borja, Deputy governor" in fs.text
