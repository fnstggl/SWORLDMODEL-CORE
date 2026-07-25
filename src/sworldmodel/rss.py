"""Google News RSS: query construction, parsing, and link resolution.

RSS is a real discovery *channel*, not the source of truth: every item points at a
publisher article that must be fetched and verified separately, exactly like a search
result. To make an item fetchable its link has to be turned back into the publisher's
own URL, because Google News wraps each article in a redirect.

:func:`resolve_item_url` recovers the publisher URL where the wrapper still permits it:
from a legacy ``url=`` parameter, from the base64 article id used by older feeds, or
from a non-Google link in the item description. Newer feeds use an opaque, server-side
article id that no offline transformation can invert; those items are reported as
unresolved rather than silently dropped, and the caller decides whether to spend a
request following the redirect.

An RSS feed is always fetched *live*, so its items are today's news. For a past cutoff
the item dates are what make it usable at all: :func:`RssItem.published` is what lets a
caller discard items that did not exist yet at ``as_of``.
"""

from __future__ import annotations

import base64
import binascii
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
_GOOGLE_HOSTS = ("news.google.com", "google.com", "www.google.com")
_URL_IN_TEXT = re.compile(r"https?://[^\s\"'<>]+")
_HREF = re.compile(r'href=["\'](https?://[^"\']+)["\']', re.IGNORECASE)


# Conventional feed locations. An institution that publishes decisions, minutes or press
# releases almost always exposes them at one of these, and the items carry the
# publisher's own article URLs — not a redirect that has to be inverted.
#
# This exists because the aggregator channel stopped yielding URLs at all: Google News
# now wraps every item in an opaque server-side id, and following it returns a
# JavaScript interstitial rather than a redirect. Measured on this machine, 0 of 100
# items from a live feed were resolvable. Going to the institution directly is both more
# reliable and a better source.
FEED_PATHS = (
    "/rss",
    "/feed",
    "/rss.xml",
    "/feed.xml",
    "/atom.xml",
    "/index.xml",
    "/news/rss",
    "/news/feed",
    "/en/rss",
    "/en/feed",
    "/press/rss",
    "/press-releases/rss",
    "/rss/news.xml",
    "/rss/news",
    "/news.xml",
)


_FEED_LINK = re.compile(
    r"<link\b[^>]*\btype=[\"']application/(?:rss\+xml|atom\+xml)[\"'][^>]*>",
    re.IGNORECASE,
)


def site_roots(domain: str) -> list[str]:
    """The domain's home page, with and without ``www``.

    Institutions differ on which one serves content and which one 404s, and guessing
    wrong costs the whole channel — the Bank of England's feed lives under ``www`` and
    the bare host does not resolve to it.
    """

    host = domain.strip().lower()
    if not host:
        return []
    host = host.removeprefix("https://").removeprefix("http://").split("/")[0]
    if not host or " " in host:
        return []
    hosts = [host] if host.startswith("www.") else [f"www.{host}", host]
    return [f"https://{h}/" for h in hosts]


def discover_feed_links(html: str, base_url: str) -> list[str]:
    """Feeds the page declares through standard autodiscovery.

    ``<link rel="alternate" type="application/rss+xml" href="...">`` is the convention
    every content system emits, and it is how a feed is found without knowing anything
    about the institution. Guessing conventional paths only works when the site happens
    to use one; asking the site works generally.
    """

    out: list[str] = []
    for tag in _FEED_LINK.findall(html or ""):
        m = _HREF.search(tag) or re.search(r'href=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        if not m:
            continue
        href = m.group(1).strip()
        if href:
            out.append(urllib.parse.urljoin(base_url, href))
    return list(dict.fromkeys(out))


def feed_urls_for(domain: str) -> list[str]:
    """Candidate feed URLs for an authoritative domain, most conventional first."""

    out: list[str] = []
    for root in site_roots(domain):
        out.extend(root.rstrip("/") + path for path in FEED_PATHS)
    return out


def google_news_rss_url(query: str, *, lang: str = "en-US", country: str = "US") -> str:
    params = {"q": query, "hl": lang, "gl": country, "ceid": f"{country}:{lang.split('-')[0]}"}
    return f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"


@dataclass(frozen=True)
class RssItem:
    title: str
    link: str
    source: str
    published: datetime | None
    description: str
    source_url: str = ""  # the publisher's site, from <source url="...">


def parse_rss(xml_text: str) -> list[RssItem]:
    items: list[RssItem] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return items
    for item in root.iter("item"):
        title = _text(item, "title")
        link = _text(item, "link")
        desc = _text(item, "description")
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        source_url = (source_el.get("url") or "").strip() if source_el is not None else ""
        published: datetime | None = None
        pub = _text(item, "pubDate")
        if pub:
            try:
                published = parsedate_to_datetime(pub)
            except (TypeError, ValueError):
                published = None
        items.append(
            RssItem(
                title=title,
                link=link,
                source=source,
                published=published,
                description=desc,
                source_url=source_url,
            )
        )
    return items


def is_google_redirect(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host.lower() in _GOOGLE_HOSTS


def resolve_item_url(item: RssItem) -> str | None:
    """The publisher URL behind an RSS item, or ``None`` if it cannot be recovered here.

    ``None`` means "not resolvable without a network round trip", not "no article".
    """

    if item.link and not is_google_redirect(item.link):
        return item.link
    for candidate in (
        _url_parameter(item.link),
        _decode_article_id(item.link),
        _publisher_link_in(item.description),
    ):
        if candidate and not is_google_redirect(candidate):
            return candidate
    return None


def _url_parameter(link: str) -> str | None:
    """Legacy wrappers carry the destination in a ``url=`` query parameter."""

    if not link:
        return None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query)
    for key in ("url", "u", "q"):
        for value in query.get(key, []):
            if value.startswith(("http://", "https://")):
                return value
    return None


def _decode_article_id(link: str) -> str | None:
    """Older Google News article ids base64-encode the destination URL.

    Newer ids encode a server-side reference instead, and no URL is recoverable from
    them; that case returns ``None`` rather than a guess.
    """

    if not link:
        return None
    segment = urllib.parse.urlsplit(link).path.rsplit("/", 1)[-1]
    if not segment:
        return None
    for padding in range(4):
        try:
            raw = base64.urlsafe_b64decode(segment + "=" * padding)
        except (binascii.Error, ValueError):
            continue
        match = _URL_IN_TEXT.search(raw.decode("latin-1"))
        if match:
            return match.group(0).rstrip("\\\x00")
    return None


def _publisher_link_in(description: str) -> str | None:
    for href in _HREF.findall(description or ""):
        if not is_google_redirect(str(href)):
            return str(href)
    return None


def _text(element: ElementTree.Element, tag: str) -> str:
    el = element.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""
