"""Evidence research and world grounding.

More detailed simulation is harmful when the underlying information is false, so this
is one of the most important stages. A research backend produces the *complete*
structured evidence store (never a truncated string) plus a compiled
:class:`~sworldmodel.worldspec.WorldSpec` — the actual causal world required for the
question — the genuine uncertainties, and the load-bearing reality facts.

Two backends ship here:
* :class:`CorpusResearchBackend` — reads a document corpus of real, dated sources and
  materializes cited claims, runs event-level lineage de-duplication and contradiction
  detection. The corpus carries an authored ``world_spec`` (used offline / in tests).
* :class:`MockResearchBackend` — an in-memory bundle for unit tests.

The live web backend (:mod:`live_research`) implements the same ``research`` interface
and compiles the ``world_spec`` from the model instead of reading it from a corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .evidence import EvidenceClaim, EvidenceStore, mark_contradiction
from .models import (
    AuthorityLevel,
    EpistemicType,
    RequiredRealityFact,
    SourceType,
    UncertaintySpec,
)
from .world import WorldFact
from .world_compiler import (
    parse_required_facts,
    parse_uncertainties,
    parse_world_facts,
)
from .worldspec import WorldSpec, parse_world_spec


@dataclass(frozen=True)
class ResearchBundle:
    """The verified evidence store plus the compiled world and its uncertainties."""

    evidence_store: EvidenceStore
    spec: WorldSpec
    uncertainties: tuple[UncertaintySpec, ...]
    world_facts: tuple[WorldFact, ...]
    required_reality_facts: tuple[RequiredRealityFact, ...]
    subject_entity: str
    resolution_units: str
    target_outcome: str
    expected_participants: int | None
    authoritative_sources: tuple[str, ...]
    horizon: datetime
    research_plan: tuple[str, ...]
    as_of: datetime | None = None
    reference_class: dict[str, str] | None = None
    outcome: dict[str, Any] | None = None  # post-cutoff; never used by the forecast
    live_trace: dict[str, Any] | None = None
    # Typed record of every integrity failure repaired on the way to a valid world.
    repair_log: tuple[dict[str, Any], ...] = ()
    compile_responses: tuple[Any, ...] = ()


class ResearchBackend(Protocol):
    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle: ...


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


BACKWARD_PLAN = (
    "terminal condition over final world state",
    "<- the recorded actions/records that satisfy it",
    "<- the process events that produce those records",
    "<- what each actor can perceive and do",
    "<- the verified entities, roles, capabilities and resources",
    "<- the evidence-grounded initial world",
)


def build_bundle_from_dict(data: dict[str, Any]) -> ResearchBundle:
    """Materialize a :class:`ResearchBundle` from a corpus dict.

    Runs real lineage grouping and contradiction detection on the claims; both are
    behavior the forecast depends on, not decoration."""

    store = EvidenceStore()
    claim_keys: dict[str, str] = {}
    for src in data.get("sources", []):
        s_type = SourceType(src["source_type"])
        authority = AuthorityLevel(int(src["authority_level"]))
        published = _dt(src["published_at"])
        available = _dt(src.get("available_at") or src["published_at"])
        retrieved = _dt(src.get("retrieved_at") or src["published_at"])
        assert published is not None and available is not None and retrieved is not None
        lineage = src["lineage_event_id"]
        for claim in src["claims"]:
            if "claim_key" in claim:
                claim_keys[claim["id"]] = str(claim["claim_key"])
            store.add(
                EvidenceClaim(
                    id=claim["id"],
                    proposition=claim["proposition"],
                    normalized_value=str(claim["normalized_value"]),
                    entities=tuple(claim.get("entities", [])),
                    valid_from=_dt(claim.get("valid_from")),
                    valid_until=_dt(claim.get("valid_until")),
                    published_at=published,
                    available_at=available,
                    source_id=src["source_id"],
                    source_url=src.get("url", ""),
                    source_title=src.get("title", ""),
                    source_type=s_type,
                    authority_level=authority,
                    supporting_excerpt=claim.get("supporting_excerpt", ""),
                    lineage_event_id=lineage,
                    epistemic_type=EpistemicType(claim.get("epistemic_type", "observation")),
                    confidence=float(claim.get("confidence", 0.9)),
                    retrieved_at=retrieved,
                    contradiction_ids=tuple(claim.get("contradiction_ids", [])),
                )
            )
    _apply_contradictions(store, data.get("contradictions", []), claim_keys)
    return assemble_bundle(store, data)


def assemble_bundle(store: EvidenceStore, data: dict[str, Any]) -> ResearchBundle:
    """Assemble a :class:`ResearchBundle` from a materialized store + a compiled
    ``world_spec`` and its uncertainties/facts/reality metadata."""

    reality = data.get("reality", {})
    as_of = _dt(reality.get("as_of"))
    default_time = as_of or datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    horizon = _dt(reality.get("horizon"))
    assert horizon is not None, "corpus/live compilation must declare a horizon"

    available_ids = {c.id for c in store.all()}
    spec = parse_world_spec(data["world_spec"])
    return ResearchBundle(
        evidence_store=store,
        spec=spec,
        uncertainties=parse_uncertainties(data.get("uncertainties"), available_ids),
        world_facts=parse_world_facts(data.get("world_facts"), default_time),
        required_reality_facts=parse_required_facts(data.get("required_reality_facts")),
        subject_entity=str(reality.get("subject_entity") or spec.subject_entity or "the subject"),
        resolution_units=str(reality.get("resolution_units") or spec.resolution_units or "outcome"),
        target_outcome=str(reality.get("target_outcome", "")),
        expected_participants=(
            int(reality["expected_participants"])
            if reality.get("expected_participants") is not None
            else None
        ),
        authoritative_sources=tuple(reality.get("authoritative_sources", []) or []),
        horizon=horizon,
        research_plan=BACKWARD_PLAN,
        as_of=as_of,
        reference_class=data.get("reference_class"),
        outcome=data.get("outcome"),
        compile_responses=tuple(data.get("_compile_responses", []) or []),
    )


def _apply_contradictions(
    store: EvidenceStore, explicit: list[list[str]], claim_keys: dict[str, str]
) -> None:
    """Record decisive contradictions: (1) corpus-declared pairs and (2) claims that
    share an explicit ``claim_key`` but disagree on ``normalized_value``."""

    pairs: set[tuple[str, str]] = set()
    for pair in explicit:
        pairs.add((pair[0], pair[1]))
    by_key: dict[str, list[str]] = {}
    for cid, key in claim_keys.items():
        by_key.setdefault(key, []).append(cid)
    for ids in by_key.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = store.get(ids[i]), store.get(ids[j])
                if a.normalized_value != b.normalized_value:
                    pairs.add((a.id, b.id))
    for a_id, b_id in pairs:
        na, nb = mark_contradiction(store.get(a_id), store.get(b_id))
        store.claims[na.id] = na
        store.claims[nb.id] = nb


class CorpusResearchBackend:
    """Reads a corpus directory containing ``corpus.json``."""

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)

    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle:
        path = self.corpus_dir / "corpus.json"
        data = json.loads(path.read_text())
        return build_bundle_from_dict(data)


class MockResearchBackend:
    """Wraps a pre-built bundle (or a corpus dict) for deterministic unit tests."""

    def __init__(
        self, bundle: ResearchBundle | None = None, *, data: dict[str, Any] | None = None
    ) -> None:
        if bundle is None and data is None:
            raise ValueError("MockResearchBackend needs a bundle or data")
        self._bundle = bundle if bundle is not None else build_bundle_from_dict(data)  # type: ignore[arg-type]

    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle:
        return self._bundle
