"""On-disk caches for retrieval and extraction — replay of recorded work, never new work.

Two caches, both strictly *record-and-replay*:

* **Accepted source text.** A page that was fetched, decoded, dated and accepted is
  stored with its full provenance (fetched URL, content hash, publication and
  observation times). A later run asking for the same URL under the same retrieval
  regime replays the recorded document instead of re-fetching it. This is
  fidelity-preserving by construction: the replayed source carries its ORIGINAL
  ``fetched_at``/``archived_at``, so admissibility at a cutoff is judged against when
  the content was demonstrably observed, exactly as the live path judges it. Only
  accepted (``ok``) sources are stored — a rejection is re-decided fresh every run.

* **Extracted claims.** An extraction call is identified by (model id, prompt sha256,
  seed) — the prompt hash covers the document content, the question, the window and
  the extraction contract's exact wording, so any change to content, schema, model or
  prompt version misses the cache. A hit replays the recorded model output through
  the same verification the live call uses; it never invents claims.

Both caches key on content identity, refuse to serve entries past their TTL, and can
be disabled entirely with ``SWORLDMODEL_CACHE=off``. Cache locations and TTLs come
from the environment (defaults under ``.cache/``, which is git-ignored):

* ``SWORLDMODEL_SOURCE_CACHE_DIR``  (default ``.cache/sources``)
* ``SWORLDMODEL_EXTRACT_CACHE_DIR`` (default ``.cache/extractions``)
* ``SWORLDMODEL_SOURCE_CACHE_TTL``  seconds (default 21600 — six hours — for live
  pages; archive captures at a fixed pre-cutoff instant are immutable and use
  ``SWORLDMODEL_ARCHIVE_CACHE_TTL``, default thirty days)
* ``SWORLDMODEL_EXTRACT_CACHE_TTL`` seconds (default 604800 — seven days; the key
  already changes whenever the content, model or prompt does)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .ids import sha256_hex
from .source_fetch import FetchedSource

_OFF_VALUES = frozenset({"off", "0", "false", "no"})


def caching_disabled() -> bool:
    return os.environ.get("SWORLDMODEL_CACHE", "").strip().lower() in _OFF_VALUES


def _dir(env: str, default: str) -> Path:
    return Path(os.environ.get(env) or default)


def _ttl(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, ""))
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, default=str) + "\n")
        tmp.replace(path)
    except OSError:
        # A cache that raised would be a new way to lose a run — the opposite of its
        # point. A failed write simply means the next run fetches live.
        pass


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class SourceCache:
    """Accepted source documents, keyed by URL and retrieval regime."""

    def __init__(self) -> None:
        self.dir = _dir("SWORLDMODEL_SOURCE_CACHE_DIR", ".cache/sources")
        self.live_ttl = _ttl("SWORLDMODEL_SOURCE_CACHE_TTL", 21600.0)
        self.archive_ttl = _ttl("SWORLDMODEL_ARCHIVE_CACHE_TTL", 30 * 86400.0)
        self.hits = 0
        self.misses = 0

    def _key(self, url: str, *, archive_cutoff: datetime | None) -> Path:
        regime = f"archive:{archive_cutoff.isoformat()}" if archive_cutoff else "live"
        digest = sha256_hex("\x00".join((regime, url)))
        return self.dir / f"{digest}.json"

    def load(self, url: str, *, archive_cutoff: datetime | None) -> FetchedSource | None:
        if caching_disabled():
            return None
        path = self._key(url, archive_cutoff=archive_cutoff)
        data = _read_json(path)
        if data is None or data.get("url") != url:
            self.misses += 1
            return None
        ttl = self.archive_ttl if archive_cutoff else self.live_ttl
        if time.time() - float(data.get("stored_at", 0)) > ttl:
            self.misses += 1
            return None
        fetched_at = _parse_dt(data.get("fetched_at"))
        if fetched_at is None:
            self.misses += 1
            return None
        self.hits += 1
        return FetchedSource(
            url=url,
            fetched_url=str(data.get("fetched_url") or url),
            final_url=str(data.get("final_url") or url),
            status=int(data.get("status") or 200),
            reachable=True,
            title=str(data.get("title") or ""),
            text=str(data.get("text") or ""),
            publisher=str(data.get("publisher") or ""),
            published_at=_parse_dt(data.get("published_at")),
            archived_at=_parse_dt(data.get("archived_at")),
            fetched_at=fetched_at,
            content_hash=str(data.get("content_hash") or ""),
            elapsed_ms=0,
        )

    def store(self, source: FetchedSource, *, archive_cutoff: datetime | None) -> None:
        if caching_disabled() or not source.ok:
            return
        _write_json(
            self._key(source.url, archive_cutoff=archive_cutoff),
            {
                "stored_at": time.time(),
                "url": source.url,
                "fetched_url": source.fetched_url,
                "final_url": source.final_url,
                "status": source.status,
                "title": source.title,
                "text": source.text,
                "publisher": source.publisher,
                "published_at": _iso(source.published_at),
                "archived_at": _iso(source.archived_at),
                "fetched_at": _iso(source.fetched_at),
                "content_hash": source.content_hash,
            },
        )


class ExtractionCache:
    """Recorded extraction-model outputs, keyed by content, schema, model and prompt.

    The stored value is the model's raw claim list BEFORE verification; the caller
    re-verifies against the same window exactly as the live path does, so a cache hit
    can never admit a claim the live call would have refused.
    """

    def __init__(self) -> None:
        self.dir = _dir("SWORLDMODEL_EXTRACT_CACHE_DIR", ".cache/extractions")
        self.ttl = _ttl("SWORLDMODEL_EXTRACT_CACHE_TTL", 7 * 86400.0)
        self.hits = 0
        self.misses = 0

    def _key(self, model_id: str, prompt_sha256: str, seed: int) -> Path:
        digest = sha256_hex("\x00".join((model_id, prompt_sha256, str(seed))))
        return self.dir / f"{digest}.json"

    def load(self, model_id: str, prompt_sha256: str, seed: int) -> dict[str, Any] | None:
        if caching_disabled():
            return None
        data = _read_json(self._key(model_id, prompt_sha256, seed))
        if data is None or data.get("prompt_sha256") != prompt_sha256:
            self.misses += 1
            return None
        if time.time() - float(data.get("stored_at", 0)) > self.ttl:
            self.misses += 1
            return None
        raw = data.get("response")
        if not isinstance(raw, dict):
            self.misses += 1
            return None
        self.hits += 1
        return raw

    def store(self, model_id: str, prompt_sha256: str, seed: int, response: dict[str, Any]) -> None:
        if caching_disabled():
            return
        _write_json(
            self._key(model_id, prompt_sha256, seed),
            {
                "stored_at": time.time(),
                "prompt_sha256": prompt_sha256,
                "model": model_id,
                "seed": seed,
                "response": response,
            },
        )
