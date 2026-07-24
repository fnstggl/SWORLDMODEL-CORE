"""The canonical evidence store, cutoff enforcement, and lineage de-duplication.

This is the first persistent structure built after the resolution contract. It is
never truncated to a fixed character budget: the *whole* structured store is kept,
and task-specific, token-limited *views* are derived from it on demand — each view
still renders full claims with their ids.

Cutoff enforcement is mechanical, not prompt-wording: downstream code only ever
receives an :class:`EvidenceView` bound to an ``as_of``. A claim whose
``available_at`` is after ``as_of`` is physically unreachable through that view,
even though it still lives in the store (for later, sealed, post-outcome analysis).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime

from .errors import CutoffViolationError, EvidenceError
from .models import AuthorityLevel, EpistemicType, SourceType

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class EvidenceClaim:
    """A single, fully-attributed evidence claim.

    ``lineage_event_id`` ties claims that come from the *same underlying event*
    together. Multiple qualitative implications may be drawn from one meeting, but
    they share a lineage id and must never be counted as independent statistical
    cases.
    """

    id: str
    proposition: str
    normalized_value: str
    entities: tuple[str, ...]
    valid_from: datetime | None
    valid_until: datetime | None
    published_at: datetime
    available_at: datetime
    source_id: str
    source_url: str
    source_title: str
    source_type: SourceType
    authority_level: AuthorityLevel
    supporting_excerpt: str
    lineage_event_id: str
    epistemic_type: EpistemicType
    confidence: float
    retrieved_at: datetime
    contradiction_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.available_at < self.published_at:
            # A fact cannot be *available* before it was published.
            raise EvidenceError(
                f"Claim {self.id}: available_at ({self.available_at}) precedes "
                f"published_at ({self.published_at})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise EvidenceError(f"Claim {self.id}: confidence must be in [0,1]")

    def is_available_at(self, as_of: datetime) -> bool:
        return self.available_at <= as_of

    def render(self) -> str:
        """Full, id-tagged rendering for a prompt view."""

        tags = f"[{self.id} · {self.epistemic_type.value} · {self.source_type.value}]"
        return f"{tags} {self.proposition} (value={self.normalized_value})"

    def tokens(self) -> list[str]:
        return _tokens(self.proposition + " " + " ".join(self.entities))


@dataclass
class EvidenceStore:
    """The complete, mutable-at-build-time canonical store. After it is built it is
    only ever *read* (through views); the runtime never edits claims in place."""

    claims: dict[str, EvidenceClaim] = field(default_factory=dict)

    def add(self, claim: EvidenceClaim) -> None:
        if claim.id in self.claims:
            raise EvidenceError(f"Duplicate claim id {claim.id}")
        self.claims[claim.id] = claim

    def get(self, claim_id: str) -> EvidenceClaim:
        try:
            return self.claims[claim_id]
        except KeyError as exc:
            raise EvidenceError(f"Unknown claim id {claim_id}") from exc

    def all(self) -> list[EvidenceClaim]:
        return list(self.claims.values())

    def lineage_groups(self) -> dict[str, list[EvidenceClaim]]:
        """Group claims by their underlying event id."""

        groups: dict[str, list[EvidenceClaim]] = {}
        for claim in self.claims.values():
            groups.setdefault(claim.lineage_event_id, []).append(claim)
        return groups

    def independent_event_ids(self) -> set[str]:
        """The set of distinct underlying events. ``len()`` of this is the maximum
        number of *independent* observations available — not the claim count."""

        return {c.lineage_event_id for c in self.claims.values()}

    def contradictions(self) -> list[tuple[str, str]]:
        """Return decisive contradiction pairs (a_id, b_id) recorded on claims."""

        pairs: set[tuple[str, str]] = set()
        for claim in self.claims.values():
            for other in claim.contradiction_ids:
                pairs.add(tuple(sorted((claim.id, other))))  # type: ignore[arg-type]
        return sorted(pairs)

    def view(self, as_of: datetime) -> EvidenceView:
        """Return the cutoff-bounded, read-only view used by all downstream code."""

        return EvidenceView(store=self, as_of=as_of)


@dataclass(frozen=True)
class EvidenceView:
    """A read-only projection of the store bounded by ``as_of``.

    Post-cutoff claims are simply not returned by any accessor here. Attempting to
    fetch one by id raises :class:`CutoffViolationError` so a leak fails loudly
    instead of silently poisoning a pastcast.
    """

    store: EvidenceStore
    as_of: datetime

    def available(self) -> list[EvidenceClaim]:
        return [c for c in self.store.all() if c.is_available_at(self.as_of)]

    def get(self, claim_id: str) -> EvidenceClaim:
        claim = self.store.get(claim_id)
        if not claim.is_available_at(self.as_of):
            raise CutoffViolationError(
                f"Claim {claim_id} (available_at={claim.available_at.isoformat()}) is "
                f"after the cutoff {self.as_of.isoformat()} and cannot be used"
            )
        return claim

    def by_entity(self, entity: str) -> list[EvidenceClaim]:
        e = entity.lower()
        return [c for c in self.available() if any(e == x.lower() for x in c.entities)]

    def observations(self) -> list[EvidenceClaim]:
        return [c for c in self.available() if c.epistemic_type is EpistemicType.OBSERVATION]

    def independent_event_ids(self) -> set[str]:
        return {c.lineage_event_id for c in self.available()}

    def relevant(self, query: str, *, limit: int) -> list[EvidenceClaim]:
        """Rank available claims by lexical relevance to ``query`` and authority.

        This is an *operational* retrieval heuristic — token overlap plus authority —
        it encodes no social outcome. It never deletes anything: it selects the top
        ``limit`` claims and renders them fully, ids intact.
        """

        q = set(_tokens(query))
        scored: list[tuple[float, EvidenceClaim]] = []
        for claim in self.available():
            overlap = len(q & set(claim.tokens()))
            score = overlap + 0.1 * int(claim.authority_level)
            scored.append((score, claim))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [claim for _, claim in scored[:limit]]

    def render_view(self, query: str, *, limit: int) -> tuple[str, tuple[str, ...]]:
        """Render a token-limited, id-preserving evidence view for a prompt.

        Returns ``(text, claim_ids)``. The canonical store is untouched.
        """

        claims = self.relevant(query, limit=limit)
        text = "\n".join(c.render() for c in claims)
        return text, tuple(c.id for c in claims)


def check_lineage_independence(
    claims: list[EvidenceClaim],
) -> tuple[int, int]:
    """Return ``(claim_count, independent_event_count)``.

    A caller that wants to treat these claims as independent samples must use the
    *event* count, never the claim count.
    """

    return len(claims), len({c.lineage_event_id for c in claims})


def mark_contradiction(a: EvidenceClaim, b: EvidenceClaim) -> tuple[EvidenceClaim, EvidenceClaim]:
    """Return copies of ``a`` and ``b`` with each other recorded as a contradiction."""

    return (
        replace(a, contradiction_ids=tuple(sorted({*a.contradiction_ids, b.id}))),
        replace(b, contradiction_ids=tuple(sorted({*b.contradiction_ids, a.id}))),
    )
