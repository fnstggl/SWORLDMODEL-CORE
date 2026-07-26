#!/usr/bin/env python3
"""A read-only world surface built from a run's recorded ledger, for re-evaluation.

The forensic reconstruction re-evaluates a run's terminal against states it replays
from the recorded event ledger — including counterfactual states with some events
removed. To do that with the engine's OWN evaluator rather than a reimplementation of
it, the replayed state has to satisfy :class:`~sworldmodel.expressions.ExprContext`.

This is that adapter and nothing more. It holds replayed fields and event-type counts
and answers the evaluator's questions from them. It never invents a value: a field the
ledger never set reads as ``None``, which is exactly what the runtime's unresolved
guards test for.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class ForensicWorld:
    """Replayed state, presented as the read-only surface the evaluator expects."""

    def __init__(
        self,
        fields: dict[str, Any],
        counts: dict[str, int],
        *,
        now: datetime | None = None,
        horizon: datetime | None = None,
        as_of: datetime | None = None,
        documents: dict[str, dict[str, Any]] | None = None,
        resources: dict[tuple[str, str], float] | None = None,
        stage: str = "final",
    ) -> None:
        self._fields = dict(fields)
        self._counts = dict(counts)
        self._documents = documents or {}
        self._resources = resources or {}
        self._stage = stage
        default = datetime(1970, 1, 1, tzinfo=UTC)
        self._now = now or default
        self._horizon = horizon or default
        self._as_of = as_of or default

    def get_field(self, name: str) -> Any:
        return self._fields.get(name)

    def get_records(self, collection: str) -> list[dict[str, Any]]:
        # Only the cardinality is reconstructable from the ledger's append_record
        # payloads, and cardinality is what record_count terminals read. The
        # placeholders carry no fabricated content.
        return [{} for _ in range(self._counts.get(collection, 0))]

    def get_events(self, event_type: str) -> list[dict[str, Any]]:
        return [{} for _ in range(self._counts.get(event_type, 0))]

    def get_resource(self, resource_id: str, holder: str) -> float:
        return float(self._resources.get((resource_id, holder), 0.0))

    def get_document_field(self, document_id: str, field_name: str) -> Any:
        return (self._documents.get(document_id) or {}).get(field_name)

    def get_stage(self) -> str:
        return self._stage

    def get_now(self) -> datetime:
        return self._now

    def get_horizon(self) -> datetime:
        return self._horizon

    def get_as_of(self) -> datetime:
        return self._as_of
