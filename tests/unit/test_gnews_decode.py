"""The Google News resolver: recovers the publisher URL, or names why it cannot.

The live protocol was captured 2026-07: the article page carries data-n-a-sg and
data-n-a-ts, and batchexecute answers with an anti-XSSI framed reply whose payload
holds the URL. These tests replay that shape, plus each way it goes wrong.
"""

from __future__ import annotations

from _fakes import FakeTransport
from sworldmodel.gnews_decode import GoogleNewsDecoder, article_id_of
from sworldmodel.http import HttpResponse, html_response

GURL = "https://news.google.com/rss/articles/CBMiTESTID?oc=5"

_SIG_PAGE = '<c-wiz><div jscontroller="x" data-n-a-sg="SIGVALUE" data-n-a-ts="1234"></div></c-wiz>'
_BATCH_REPLY = (
    ")]}'\n\n"
    '[["wrb.fr","Fbv4je","[\\"garturlres\\",\\"https://publisher.example/story\\"]",null,null,null,"generic"],'
    '["di",59],["af.httprm",59,"123",7]]'
)


def _transport(reply: str = _BATCH_REPLY, sig_page: str = _SIG_PAGE) -> FakeTransport:
    t = FakeTransport()
    t.add(
        lambda u: "news.google.com/articles/" in u or "news.google.com/rss/articles/" in u,
        html_response(GURL, sig_page),
    )
    t.add(
        lambda u: "batchexecute" in u,
        HttpResponse("b", "b", 200, {"content-type": "application/json"}, reply, 1),
    )
    return t


def test_the_article_id_is_read_from_every_link_shape() -> None:
    assert article_id_of(GURL) == "CBMiTESTID"
    assert article_id_of("https://news.google.com/read/CBMiXYZ") == "CBMiXYZ"
    assert article_id_of("https://news.google.com/articles/CBMiABC?hl=en") == "CBMiABC"
    assert article_id_of("https://publisher.example/story") is None
    assert article_id_of("not a url at all") is None


def test_the_flow_recovers_the_publisher_url() -> None:
    d = GoogleNewsDecoder(_transport(), interval_seconds=0)
    result = d.decode(GURL)
    assert result.ok
    assert result.publisher_url == "https://publisher.example/story"
    assert result.article_id == "CBMiTESTID"


def test_a_successful_decode_is_cached_by_article_id() -> None:
    t = _transport()
    d = GoogleNewsDecoder(t, interval_seconds=0)
    assert d.decode(GURL).ok
    calls = len(t.calls)
    assert d.decode(GURL).ok
    assert len(t.calls) == calls  # no second network round


def test_a_challenge_page_is_named_not_parsed() -> None:
    d = GoogleNewsDecoder(
        _transport(sig_page="<html>Our systems have detected unusual traffic...</html>"),
        interval_seconds=0,
    )
    result = d.decode(GURL)
    assert not result.ok
    assert result.failure == "challenge"


def test_a_moved_protocol_is_named_never_guessed() -> None:
    """A reply that parses as something other than the expected shape means Google
    moved the protocol. The resolver must say exactly that — inventing a URL from a
    reply it does not understand would poison the evidence store at its root."""

    d = GoogleNewsDecoder(
        _transport(reply=')]}\'\n\n[["totally","different"]]'), interval_seconds=0
    )
    result = d.decode(GURL)
    assert not result.ok
    assert result.failure == "protocol_change"


def test_a_throttle_is_transient_and_not_cached() -> None:
    t = FakeTransport()
    t.add(
        lambda u: "news.google.com" in u,
        HttpResponse("x", "x", 429, {}, "slow down", 1),
    )
    d = GoogleNewsDecoder(t, interval_seconds=0)
    assert d.decode(GURL).failure == "throttled"
    # A throttle is about the moment, not the article: nothing pinned to the id.
    assert d._cache == {}


def test_a_non_gnews_url_is_refused_without_a_network_call() -> None:
    t = FakeTransport()
    d = GoogleNewsDecoder(t, interval_seconds=0)
    result = d.decode("https://publisher.example/direct")
    assert result.failure == "not_a_gnews_url"
    assert t.calls == []
