"""The research contract and the bundle every backend must produce.

A research backend produces the *complete* structured evidence store — never a
truncated string — plus the compiled :class:`~sworldmodel.worldspec.WorldSpec` for this
question, its genuine uncertainties, and the load-bearing reality facts.

Exactly one backend ships in production: :mod:`live_research`, which starts from the
question alone, searches, fetches, verifies and dates real sources, and compiles the
world from the model. Corpus readers and in-memory fixtures live under ``tests/``,
where they cannot be reached from a live run — a prepared corpus that production code
can open is a prepared answer waiting to be found.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .errors import WorldIntegrityError
from .evidence import EvidenceStore
from .models import RequiredRealityFact, UncertaintySpec
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
    outcome: dict[str, Any] | None = None  # post-cutoff; never used by the forecast
    live_trace: dict[str, Any] | None = None
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


def assemble_bundle(store: EvidenceStore, data: dict[str, Any]) -> ResearchBundle:
    """Assemble a :class:`ResearchBundle` from a materialized store + a compiled
    ``world_spec`` and its uncertainties/facts/reality metadata."""

    reality = data.get("reality", {})
    as_of = _dt(reality.get("as_of"))
    default_time = as_of or datetime.fromisoformat("1970-01-01T00:00:00+00:00")
    horizon = _dt(reality.get("horizon"))
    if horizon is None:
        raise WorldIntegrityError(
            "a compilation must declare the horizon it was compiled for; without it the "
            "world has no end and the terminal cannot be evaluated",
            details={"reality_keys": sorted(reality)},
        )

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
        outcome=data.get("outcome"),
        compile_responses=tuple(data.get("_compile_responses", []) or []),
    )
