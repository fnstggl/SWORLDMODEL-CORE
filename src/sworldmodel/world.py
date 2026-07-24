"""The one authoritative world state per branch.

There is exactly one :class:`WorldState` per branch. Everything an actor sees is a
*projection* of it (``view_for``). All mutation flows through :meth:`apply`, which
folds validated events into new state. The state is immutable: every update returns a
fresh instance, and branch clones deep-copy actor memory so worlds never share
mutable state.

The state is domain-free. It holds only universal containers — typed *fields*, named
*record* collections, *documents*, *resources*, *commitments*, a free-text *stage*,
persistent *actors*, and the event ledger — and interprets only the fixed universal
effect ops (``set_field``, ``append_record``, ``transfer_resource``, ...). It has no
concept of a vote, proposal, institution, or committee; a "vote" is merely a record in
some collection that a compiled terminal expression happens to count.

:class:`WorldState` also implements the read-only :class:`~sworldmodel.expressions.ExprContext`
surface, so the declarative evaluator can read fields/records/events/resources/documents
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .actors import ActorState, LocalView, Observation
from .evidence import EvidenceView
from .models import (
    BranchWeight,
    EpistemicType,
    Event,
    EventStatus,
    ResolutionContract,
    Visibility,
)
from .worldspec import EntitySpec

# Effect-op event kinds that carry observable content into an actor's view.
_OBSERVABLE_KINDS = frozenset(
    {
        "deliver_information",
        "release_data",
        "create_event",
        "append_record",
        "create_or_update_document",
        "update_commitment",
    }
)

# Payload keys that carry field-level information into an observation.
_INFO_KEYS = ("fields", "info_fields", "data", "levels")


@dataclass(frozen=True)
class Record:
    """A single entry appended to a named collection (a vote, an offer, a decision, a
    signature — the collection name and meaning are compiled, not hardcoded)."""

    collection: str
    key: str
    value: Any
    by: str
    time: datetime
    extra: tuple[tuple[str, Any], ...] = ()

    def as_item(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "key": self.key,
            "value": self.value,
            "by": self.by,
            "time": self.time,
        }
        item.update(dict(self.extra))
        return item


@dataclass(frozen=True)
class Commitment:
    by: str
    text: str
    tag: str
    time: datetime


@dataclass(frozen=True)
class WorldFact:
    fact_id: str
    text: str
    evidence_claim_ids: tuple[str, ...]
    available_at: datetime
    epistemic_type: EpistemicType = EpistemicType.OBSERVATION
    visibility: Visibility = Visibility.PUBLIC

    def render(self) -> str:
        cites = ",".join(self.evidence_claim_ids) if self.evidence_claim_ids else "no-citation"
        return f"{self.text} [{cites}]"


def _reskey(resource_id: str, holder: str) -> str:
    return f"{resource_id}@{holder}"


@dataclass(frozen=True)
class WorldState:
    branch_id: str
    parent_branch_id: str | None
    weight: BranchWeight
    time: datetime
    contract: ResolutionContract
    evidence: EvidenceView
    entities: tuple[EntitySpec, ...] = ()
    actors: dict[str, ActorState] = field(default_factory=dict)
    verified_facts: tuple[WorldFact, ...] = ()
    fields: tuple[tuple[str, Any], ...] = ()
    records: tuple[tuple[str, tuple[Record, ...]], ...] = ()
    documents: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    resources: tuple[tuple[str, float], ...] = ()
    commitments: tuple[Commitment, ...] = ()
    stage: str = "initial"
    pending_events: tuple[Event, ...] = ()
    event_history: tuple[Event, ...] = ()
    terminal_state: Any = None  # TerminalEvaluation | None (set by engine)

    # -- ExprContext surface (read-only accessors for the evaluator) ------------

    def get_field(self, name: str) -> Any:
        return dict(self.fields).get(name)

    def get_records(self, collection: str) -> list[dict[str, Any]]:
        for name, recs in self.records:
            if name == collection:
                return [r.as_item() for r in recs]
        return []

    def get_events(self, event_type: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ev in self.event_history:
            if ev.status is not EventStatus.APPLIED:
                continue
            data = ev.payload_dict
            if ev.kind == event_type or data.get("event_type") == event_type:
                item = {"type": event_type, "by": ev.actor_id or "environment", "time": ev.time}
                item.update(data)
                out.append(item)
        return out

    def get_resource(self, resource_id: str, holder: str) -> float:
        return float(dict(self.resources).get(_reskey(resource_id, holder), 0.0))

    def get_document_field(self, document_id: str, field_name: str) -> Any:
        for did, fields in self.documents:
            if did == document_id:
                return dict(fields).get(field_name)
        return None

    def get_stage(self) -> str:
        return self.stage

    def get_now(self) -> datetime:
        return self.time

    def get_horizon(self) -> datetime:
        return self.contract.horizon

    def get_as_of(self) -> datetime:
        return self.contract.as_of

    # -- roster helpers ---------------------------------------------------------

    def actor_ids(self) -> tuple[str, ...]:
        return tuple(self.actors.keys())

    def fields_dict(self) -> dict[str, Any]:
        return dict(self.fields)

    def records_dict(self) -> dict[str, tuple[Record, ...]]:
        return dict(self.records)

    # -- projection -------------------------------------------------------------

    def view_for(self, actor_id: str, trigger: Event | None = None) -> LocalView:
        """Derive the local view an actor could have received by ``self.time``."""

        actor = self.actors[actor_id]
        role = actor.role

        observations: list[Observation] = []
        for ev in self.event_history:
            if ev.status is not EventStatus.APPLIED or ev.time > self.time:
                continue
            if ev.kind not in _OBSERVABLE_KINDS:
                continue
            if not self._visible_to(ev, actor_id, role):
                continue
            if ev.event_id in actor.last_observed_event_ids:
                continue
            observations.append(_to_observation(ev))
        observations.sort(key=lambda o: (o.time, o.obs_id))

        public_facts = tuple(
            f.render()
            for f in self.verified_facts
            if f.visibility is Visibility.PUBLIC and f.available_at <= self.time
        )
        return LocalView(
            actor_id=actor_id,
            branch_time=self.time,
            role=role,
            authority=actor.authority,
            stage=self.stage,
            public_facts=public_facts,
            observations=tuple(observations),
            world_fields=self.fields,
            question=self.contract.question,
            subject=self.contract.subject_entity,
            trigger_obs_id=trigger.event_id if trigger else None,
        )

    @staticmethod
    def _visible_to(ev: Event, actor_id: str, role: str) -> bool:
        if ev.visibility is Visibility.PUBLIC:
            return True
        if ev.visibility is Visibility.PRIVATE:
            return actor_id in ev.audience or actor_id in ev.target_ids
        if ev.visibility is Visibility.ROLE:
            return role in ev.audience
        return False

    # -- state transitions (all return new instances) --------------------------

    def apply(self, events: list[Event] | tuple[Event, ...]) -> WorldState:
        fields = dict(self.fields)
        records = {name: list(recs) for name, recs in self.records}
        documents = {did: dict(f) for did, f in self.documents}
        resources = dict(self.resources)
        commitments = list(self.commitments)
        history = list(self.event_history)
        pending = list(self.pending_events)
        time = self.time

        for raw in events:
            ev = replace(raw, status=EventStatus.APPLIED)
            data = ev.payload_dict
            kind = ev.kind

            if kind == "set_field":
                fields[str(data["field"])] = data.get("value")
            elif kind == "adjust_field":
                cur = _num(fields.get(str(data["field"]), 0.0))
                fields[str(data["field"])] = cur + _num(data.get("delta", 0.0))
            elif kind == "release_data":
                for name, level in dict(data.get("fields", {})).items():
                    fields[name] = level
            elif kind == "append_record":
                coll = str(data["collection"])
                records.setdefault(coll, []).append(
                    Record(
                        collection=coll,
                        key=str(data.get("key", ev.actor_id or "")),
                        value=data.get("value"),
                        by=str(ev.actor_id or data.get("by", "environment")),
                        time=ev.time,
                        extra=tuple(sorted(dict(data.get("extra", {})).items())),
                    )
                )
            elif kind == "update_commitment":
                commitments.append(
                    Commitment(
                        by=str(ev.actor_id or data.get("by", "environment")),
                        text=str(data.get("text", "")),
                        tag=str(data.get("tag", "")),
                        time=ev.time,
                    )
                )
            elif kind == "transfer_resource":
                res = str(data["resource"])
                amt = _num(data.get("amount", 0.0))
                frm = _reskey(res, str(data["from"]))
                to = _reskey(res, str(data["to"]))
                resources[frm] = _num(resources.get(frm, 0.0)) - amt
                resources[to] = _num(resources.get(to, 0.0)) + amt
            elif kind == "consume_resource":
                key = _reskey(str(data["resource"]), str(data["holder"]))
                resources[key] = _num(resources.get(key, 0.0)) - _num(data.get("amount", 0.0))
            elif kind == "create_or_update_document":
                did = str(data["document"])
                documents.setdefault(did, {}).update(dict(data.get("fields", {})))

            history.append(ev)
            pending = [pe for pe in pending if pe.event_id != ev.event_id]
            time = max(time, ev.time)

        return replace(
            self,
            fields=tuple(sorted(fields.items())),
            records=tuple((k, tuple(v)) for k, v in sorted(records.items())),
            documents=tuple((k, tuple(sorted(v.items()))) for k, v in sorted(documents.items())),
            resources=tuple(sorted(resources.items())),
            commitments=tuple(commitments),
            event_history=tuple(history),
            pending_events=tuple(pending),
            time=time,
        )

    def with_stage(self, stage: str) -> WorldState:
        return replace(self, stage=stage)

    def with_time(self, time: datetime) -> WorldState:
        return replace(self, time=max(self.time, time))

    def with_actor(self, actor: ActorState) -> WorldState:
        actors = dict(self.actors)
        actors[actor.actor_id] = actor
        return replace(self, actors=actors)

    def set_terminal(self, evaluation: Any) -> WorldState:
        return replace(self, terminal_state=evaluation)

    def enqueue(self, events: list[Event] | tuple[Event, ...]) -> WorldState:
        return replace(self, pending_events=self.pending_events + tuple(events))

    def clone(self, new_branch_id: str, weight: BranchWeight) -> WorldState:
        new_actors = {aid: a.clone() for aid, a in self.actors.items()}
        return replace(
            self,
            branch_id=new_branch_id,
            parent_branch_id=self.branch_id,
            weight=weight,
            actors=new_actors,
        )


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_observation(ev: Event) -> Observation:
    data = ev.payload_dict
    info: dict[str, Any] = {}
    for key in _INFO_KEYS:
        sub = data.get(key)
        if isinstance(sub, dict):
            info.update(sub)
    return Observation(
        obs_id=ev.event_id,
        time=ev.time,
        kind=ev.kind,
        source=ev.actor_id or "environment",
        summary=_summarize(ev),
        info_fields=tuple(sorted(info.items())),
        evidence_claim_ids=ev.evidence_claim_ids,
    )


def _summarize(ev: Event) -> str:
    data = ev.payload_dict
    if ev.kind == "deliver_information":
        return f"Information from {ev.actor_id or 'environment'}: {data.get('text', '')}"
    if ev.kind == "release_data":
        return f"External data released: {dict(data.get('fields', {}))}"
    if ev.kind == "create_event":
        return f"Event: {data.get('event_type', 'happening')} {data.get('text', '')}".strip()
    if ev.kind == "append_record":
        return f"{ev.actor_id} recorded {data.get('collection')}={data.get('value')}"
    if ev.kind == "create_or_update_document":
        return f"Document {data.get('document')} updated"
    if ev.kind == "update_commitment":
        return f"{ev.actor_id} committed: {data.get('text', '')}"
    return f"{ev.kind} event"
