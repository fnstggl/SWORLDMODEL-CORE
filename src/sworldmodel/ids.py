"""Deterministic identifiers, hashing, and canonical serialization.

The whole system must be replayable and reproducible: identical inputs must
produce identical IDs, prompt hashes, and serialized artifacts. We therefore never
use randomness or wall-clock time to mint IDs. Everything is content-addressed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to a stable, sorted-key JSON string.

    Deterministic serialization is a hard requirement: artifacts are hashed and
    compared across runs. ``datetime``/``date`` become ISO-8601 strings.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, frozenset | set):
        return sorted(value, key=str)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize object of type {type(value)!r}")


def sha256_hex(text: str) -> str:
    """Return the hex SHA-256 of ``text`` (UTF-8)."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_id(prefix: str, *parts: Any) -> str:
    """Return a short, deterministic content-addressed id: ``prefix-<12 hex>``.

    Two calls with equal ``parts`` always yield the same id; this is what makes
    branch-equivalence and decision-cache keys reproducible.
    """

    digest = sha256_hex(canonical_json(list(parts)))
    return f"{prefix}-{digest[:12]}"


def prompt_hash(prompt: str) -> str:
    """Full SHA-256 of a prompt string, recorded on every model call."""

    return sha256_hex(prompt)
