"""Persistent actor memory.

Adapted *conceptually* from Generative Agents' associative memory (a newest-first
memory stream of concept nodes scored by recency × importance × relevance), but with
no spatial/grid machinery and with a deterministic lexical relevance in place of
neural embeddings, so the whole system runs offline and reproducibly.

Actors are never handed an empty memory: they are seeded from evidence-grounded
:class:`MemorySeed` records and accumulate episodic/semantic nodes as they perceive
and reflect. Retrieval records exactly which nodes were returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime

from .ids import content_id
from .models import MemorySeed

_WORD = re.compile(r"[a-z0-9]+")

# Effective retrieval weights (Generative Agents style: relevance-dominant).
W_RECENCY = 0.5
W_RELEVANCE = 3.0
W_IMPORTANCE = 2.0
RECENCY_DECAY = 0.99


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass(frozen=True)
class ConceptNode:
    node_id: str
    kind: str  # "episodic" | "semantic" | "thought"
    depth: int  # 0 for observations, >=1 for reflections
    content: str
    importance: float  # 0..1 poignancy
    created: datetime
    last_accessed: datetime
    evidence_claim_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def tokens(self) -> set[str]:
        return _tokens(self.content + " " + " ".join(self.tags))


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return dict.fromkeys(values, 0.5)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


@dataclass
class MemoryStream:
    """A persistent, newest-first stream of concept nodes owned by one actor."""

    nodes: list[ConceptNode] = field(default_factory=list)
    _index: dict[str, ConceptNode] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.nodes)

    def add(self, node: ConceptNode) -> ConceptNode:
        """Add a memory, or return the one already held if this is the same memory.

        "The same memory" is exactly what ``node_id`` already says it is: the same
        content, of the same kind, created at the same instant, carrying the same tags —
        and the tags include the source the actor heard it from. Inserting a second copy
        of that is not the actor learning something twice; it is one perception recorded
        twice, and it silently destroyed the retrieval window.

        The measured failure: with the branch clock pinned at one instant, an actor
        re-perceiving identical content produced an identical ``node_id`` inserted N
        times. ``retrieve`` then scored all N identically and returned k entries of the
        SAME node — and its refresh loop (``self.nodes.index(node)``) only ever found the
        first, so the duplicates were never even touched. ``retrieved_memory_ids`` was
        non-empty in 218 of 235 decision records, which reads as "memory works", while
        the actor was being shown six copies of one fact. The window was structurally
        unusable, not merely redundant.

        What this deliberately does NOT collapse is recurrence that carries information.
        The same fact from two different sources differs in its ``source:`` tag, and the
        same fact at two different times differs in ``created`` — both keep their own
        node, because hearing something twice from two people, or twice weeks apart, is
        corroboration and a real decider weighs it. Only one perception recorded twice
        is folded.
        """

        existing = self._index.get(node.node_id)
        if existing is not None:
            # Same content, same instant, same source: there is nothing new to record and
            # nothing to refresh — `last_accessed` on a node this instant created is
            # already this instant. Reordering the stream for a non-event would be worse.
            return existing
        self.nodes.insert(0, node)  # newest first
        self._index[node.node_id] = node
        return node

    def add_memory(
        self,
        content: str,
        *,
        kind: str,
        importance: float,
        created: datetime,
        evidence_claim_ids: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        depth: int = 0,
    ) -> ConceptNode:
        node = ConceptNode(
            node_id=content_id("mem", content, kind, created.isoformat(), tags),
            kind=kind,
            depth=depth,
            content=content,
            importance=max(0.0, min(1.0, importance)),
            created=created,
            last_accessed=created,
            evidence_claim_ids=evidence_claim_ids,
            tags=tags,
        )
        return self.add(node)

    def get(self, node_id: str) -> ConceptNode | None:
        return self._index.get(node_id)

    def retrieve(self, query: str, *, now: datetime, top_k: int) -> list[ConceptNode]:
        """Return the top-``k`` relevant nodes and refresh their last-accessed time.

        Score = w_recency·decay^rank + w_relevance·lexical + w_importance·poignancy,
        each component min-max normalized across candidates. This is an operational
        heuristic; it encodes no social outcome (no "consensus pull").

        Candidates are distinct memories. ``add`` is what keeps the stream free of
        duplicates, and this is the same invariant asserted where it is consumed: a
        window of ``top_k`` slots filled with k copies of one node is not a window, and
        it fails silently — every counter downstream reports k memories retrieved.
        """

        if not self.nodes:
            return []
        seen: set[str] = set()
        candidates: list[ConceptNode] = []
        for node in sorted(self.nodes, key=lambda n: n.last_accessed, reverse=True):
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            candidates.append(node)
        q = _tokens(query)

        recency: dict[str, float] = {}
        relevance: dict[str, float] = {}
        importance: dict[str, float] = {}
        for rank, node in enumerate(candidates):
            recency[node.node_id] = RECENCY_DECAY**rank
            nt = node.tokens()
            union = q | nt
            relevance[node.node_id] = (len(q & nt) / len(union)) if union else 0.0
            importance[node.node_id] = node.importance

        rec_n = _normalize(recency)
        rel_n = _normalize(relevance)
        imp_n = _normalize(importance)
        scored = [
            (
                W_RECENCY * rec_n[n.node_id]
                + W_RELEVANCE * rel_n[n.node_id]
                + W_IMPORTANCE * imp_n[n.node_id],
                n,
            )
            for n in candidates
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].node_id))
        chosen = [node for _, node in scored[:top_k]]

        # Retrieval refreshes recency, exactly as in the reference architecture.
        for node in chosen:
            refreshed = replace(node, last_accessed=now)
            self._index[node.node_id] = refreshed
            self.nodes[self.nodes.index(node)] = refreshed
        return [self._index[n.node_id] for n in chosen]

    def seed(self, seeds: tuple[MemorySeed, ...], *, default_time: datetime) -> None:
        for s in seeds:
            when = s.valid_time or default_time
            self.add_memory(
                s.content,
                kind=s.kind,
                importance=s.importance,
                created=when,
                evidence_claim_ids=s.evidence_claim_ids,
                tags=s.tags,
            )

    def snapshot(self) -> list[ConceptNode]:
        return list(self.nodes)
