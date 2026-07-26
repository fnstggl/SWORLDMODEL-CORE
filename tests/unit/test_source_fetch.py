"""Source fetching: date handling and cutoff hygiene."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from _fakes import FakeTransport
from sworldmodel.http import html_response
from sworldmodel.source_fetch import fetch_source

_NAIVE_DATE_PAGE = (
    '<html><head><meta property="article:published_time" content="2026-07-11T00:00:00">'
    "</head><body>content</body></html>"
)


def test_naive_body_date_is_made_timezone_aware() -> None:
    # A body date with no timezone must be coerced to aware so cutoff comparisons with
    # an aware as_of never raise "can't compare offset-naive and offset-aware".
    t = FakeTransport().add_url("u", html_response("u", _NAIVE_DATE_PAGE))
    fs = fetch_source(t, "u", now=datetime.now(UTC))
    assert fs.published_at is not None
    assert fs.published_at.tzinfo is not None
    as_of = datetime.fromisoformat("2026-05-14T23:59:59-06:00")
    assert fs.published_at > as_of  # must not raise


def test_last_modified_header_dates_an_undated_page() -> None:
    resp = html_response("u", "<html><body>stable reference page</body></html>")
    resp.headers["last-modified"] = "Wed, 21 Feb 2024 07:28:00 GMT"
    t = FakeTransport().add_url("u", resp)
    fs = fetch_source(t, "u", now=datetime.now(UTC))
    assert fs.published_at is not None
    assert fs.published_at.year == 2024 and fs.published_at.month == 2


def test_a_truncated_chunked_body_is_a_document_failure_not_a_dead_run() -> None:
    """A live EU-Mercosur run died twenty minutes in when one page's chunked response
    ended 15 bytes short: http.client.IncompleteRead is not an OSError, so the
    transport's catch missed it and the raw exception destroyed the run with no
    artifacts. One bad page is a rejected page."""

    import socket
    import threading

    from sworldmodel.http import FetchPolicy, HttpError, UrllibTransport

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve() -> None:
        conn, _ = srv.accept()
        conn.recv(4096)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            b"Transfer-Encoding: chunked\r\n\r\nf\r\ntruncated body!"
        )
        conn.close()  # closes mid-chunk -> IncompleteRead inside the body read

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    transport = UrllibTransport(policy=FetchPolicy(allow_private_addresses=True))
    try:
        with pytest.raises(HttpError, match="mid-body"):
            transport.get(f"http://127.0.0.1:{port}/doc")
    finally:
        thread.join(timeout=5)
        srv.close()
    # The failure is preserved in diagnostics, with the received byte count.
    assert transport.calls and "IncompleteRead" in (transport.calls[-1].error or "")
