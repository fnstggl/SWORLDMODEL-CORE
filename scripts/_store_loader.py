#!/usr/bin/env python3
"""Shared harness support: frozen-store loading and run-directory preparation.

Both frozen-store harnesses (``scripts/frozen_forecast.py`` and
``scripts/semantic_slice.py``) reconstruct an :class:`EvidenceStore` from a prior run's
exported ``evidence_store.json``. That reconstruction is load-bearing for fidelity:
``render_evidence`` sorts claims by descending authority and truncates, so a loader
that flattens provenance (every claim MEDIUM ``contemporaneous_reporting``) presents an
official central-bank release and a blog post as equally authoritative and can change
which claims the compiler ever sees. This module therefore reads the FULL exported
provenance when the store carries it and falls back to the historical defaults ONLY for
legacy 8-field stores — loudly, because a defaulted store ranks evidence differently
than the live run did.

It also owns run-directory preparation: a harness output directory is stamped with the
run's identity and cleared of any prior run's pipeline artifacts, so a refusal can
never leave a previous run's ``forecast.json`` sitting beside this run's
``diagnosis.json`` for a downstream reader to score as fresh.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sworldmodel.evidence import EvidenceClaim, EvidenceStore  # noqa: E402
from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType  # noqa: E402

# The canonical artifact list and clearing rule live in src beside the writers, so the
# harness cannot drift behind what the pipeline actually writes. Re-exported here
# because both harnesses (and their tests) import them from this module.
from sworldmodel.rundir import PIPELINE_ARTIFACTS, prepare_run_dir  # noqa: E402,F401

LEGACY_STORE_WARNING = (
    "legacy store: provenance defaulted (authority ranking will differ from the live run)"
)

# The provenance fields a full-fidelity export carries per claim. A claim missing any
# of these came from a legacy 8-field export and gets the historical defaults.
_PROVENANCE_KEYS = (
    "authority_level",
    "source_type",
    "published_at",
    "valid_from",
    "valid_until",
    "source_id",
    "confidence",
    "retrieved_at",
    "lineage_event_id",
)
# Re-check provenance (contradiction_ids, retrieved_url, archived_at, content_sha256,
# extraction_prompt_sha256) is READ whenever present — dropping a recorded
# contradiction silently flipped the coverage gate's conflict check on replay — but its
# absence alone does not mark a store legacy: it does not affect authority ranking.


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value))


def _parse_authority(value: Any) -> AuthorityLevel:
    """Accept the export as an int (2), a digit string ("2") or a name ("MEDIUM")."""

    if isinstance(value, AuthorityLevel):
        return value
    if isinstance(value, int):
        return AuthorityLevel(value)
    text = str(value).strip()
    try:
        return AuthorityLevel[text.upper()]
    except KeyError:
        return AuthorityLevel(int(text))


def _parse_source_type(value: Any) -> SourceType:
    """Accept the export as a value ("official_institutional") or a name."""

    if isinstance(value, SourceType):
        return value
    text = str(value).strip()
    try:
        return SourceType(text.lower())
    except ValueError:
        return SourceType[text.upper()]


def load_store(path: Path) -> EvidenceStore:
    """Reconstruct the evidence store from an exported ``evidence_store.json``.

    Full-fidelity exports carry per-claim provenance (``authority_level``,
    ``source_type``, ``published_at``, ``valid_from``, ``valid_until``, ``source_id``,
    ``confidence``, ``retrieved_at``, ``lineage_event_id``) and every one of those real
    values is preserved, so authority ranking and temporal admissibility on the replay
    match the live run exactly. Each field falls back to the historical default only
    when its key is absent — a legacy 8-field store — and that fallback is announced
    with one loud warning line, because a defaulted store cannot reproduce the live
    run's claim ranking.
    """

    raw = json.loads(path.read_text())
    store = EvidenceStore()
    defaulted = False
    for c in raw:
        available = datetime.fromisoformat(c["available_at"])
        url = str(c.get("source_url") or "")
        host = urlparse(url).hostname or "unknown"
        if any(k not in c for k in _PROVENANCE_KEYS):
            defaulted = True
        store.add(
            EvidenceClaim(
                id=str(c["id"]),
                proposition=str(c["proposition"]),
                normalized_value=str(c.get("normalized_value") or ""),
                entities=tuple(c.get("entities") or ()),
                valid_from=_parse_dt(c["valid_from"]) if "valid_from" in c else available,
                valid_until=_parse_dt(c["valid_until"]) if "valid_until" in c else None,
                published_at=(_parse_dt(c["published_at"]) or available)
                if "published_at" in c
                else available,
                available_at=available,
                source_id=str(c["source_id"]) if "source_id" in c else host,
                source_url=url,
                source_title=str(c["source_title"]) if "source_title" in c else host,
                source_type=_parse_source_type(c["source_type"])
                if "source_type" in c
                else SourceType.CONTEMPORANEOUS_REPORTING,
                authority_level=_parse_authority(c["authority_level"])
                if "authority_level" in c
                else AuthorityLevel.MEDIUM,
                supporting_excerpt=str(c.get("supporting_excerpt") or ""),
                lineage_event_id=str(c["lineage_event_id"])
                if "lineage_event_id" in c
                else f"ev_{c['id']}",
                epistemic_type=EpistemicType(str(c.get("epistemic_type") or "observation")),
                confidence=float(c["confidence"]) if "confidence" in c else 0.8,
                retrieved_at=(_parse_dt(c["retrieved_at"]) or available)
                if "retrieved_at" in c
                else available,
                # Re-check provenance. Dropping contradiction_ids silently erased a
                # recorded decisive contradiction on replay, flipping the coverage
                # gate's conflict check for the same store.
                contradiction_ids=tuple(str(x) for x in (c.get("contradiction_ids") or ())),
                retrieved_url=str(c.get("retrieved_url") or ""),
                archived_at=_parse_dt(c["archived_at"])
                if c.get("archived_at") is not None
                else None,
                content_sha256=str(c.get("content_sha256") or ""),
                extraction_prompt_sha256=str(c.get("extraction_prompt_sha256") or ""),
            )
        )
    if defaulted:
        print(f"WARNING: {LEGACY_STORE_WARNING}", file=sys.stderr)
    return store
