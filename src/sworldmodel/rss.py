"""Google News RSS: query construction and parsing.

RSS is one research *channel*, not the source of truth. Each RSS item points at a
publisher article that must be fetched and verified separately.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


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
        published: datetime | None = None
        pub = _text(item, "pubDate")
        if pub:
            try:
                published = parsedate_to_datetime(pub)
            except (TypeError, ValueError):
                published = None
        items.append(
            RssItem(title=title, link=link, source=source, published=published, description=desc)
        )
    return items


def _text(element: ElementTree.Element, tag: str) -> str:
    el = element.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""
