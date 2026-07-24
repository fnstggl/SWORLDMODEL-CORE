"""Source fetching: date handling and cutoff hygiene."""

from __future__ import annotations

from datetime import UTC, datetime

from sworldmodel.http import FakeTransport, html_response
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
