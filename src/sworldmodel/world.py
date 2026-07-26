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
from .ids import content_id
from .models import (
    BranchWeight,
    EpistemicType,
    Event,
    EventStatus,
    ResolutionContract,
    Visibility,
)
from .schedule import Schedule
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
class Delivery:
    """One step of the information lifecycle, for one actor and one event.

    Visibility, delivery, and noticing are three different things and each is recorded
    separately: an event may be *visible* to an actor, become *available* to it at some
    later time through some channel, and be *noticed* later still — or never. Only
    noticed information enters an actor's view, its memory, or its reasoning.
    """

    event_id: str
    actor_id: str
    available_at: datetime
    notice_at: datetime | None = None  # when it is scheduled to be noticed
    noticed_at: datetime | None = None  # when it actually was
    channel: str = ""

    @property
    def noticed(self) -> bool:
        return self.noticed_at is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "actor_id": self.actor_id,
            "available_at": self.available_at.isoformat(),
            "notice_at": self.notice_at.isoformat() if self.notice_at else None,
            "noticed_at": self.noticed_at.isoformat() if self.noticed_at else None,
            "channel": self.channel,
        }


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
    event_history: tuple[Event, ...] = ()
    terminal_state: Any = None  # TerminalEvaluation | None (set by engine)
    # The branch clock: everything that is still going to happen, in time order.
    schedule: Schedule = field(default_factory=Schedule)
    # The information lifecycle: who has been delivered what, and who noticed it.
    deliveries: tuple[Delivery, ...] = ()
    # Monotonic state version. An actor's intention records the version it saw, so an
    # intention formed against a world that has since moved can be caught and refused
    # instead of being applied to a world the actor never observed.
    version: int = 0

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
        """Derive the local view of an actor: strictly what it has **noticed**.

        Not what happened, not what was visible, not what was delivered — what this
        actor actually took in, by this branch time.
        """

        actor = self.actors[actor_id]
        by_id = {ev.event_id: ev for ev in self.event_history}

        observations: list[Observation] = []
        for d in self.deliveries:
            if d.actor_id != actor_id or d.noticed_at is None or d.noticed_at > self.time:
                continue
            ev = by_id.get(d.event_id)
            if ev is None or ev.status is not EventStatus.APPLIED:
                continue
            observations.append(_to_observation(ev, d))
        observations.sort(key=lambda o: (o.time, o.obs_id))

        public_facts = tuple(
            f.render()
            for f in self.verified_facts
            if f.visibility is Visibility.PUBLIC and f.available_at <= self.time
        )
        return LocalView(
            actor_id=actor_id,
            branch_time=self.time,
            role=actor.role,
            authority=actor.authority,
            stage=self.stage,
            public_facts=public_facts,
            observations=tuple(observations),
            world_fields=self.fields,
            question=self.contract.question,
            subject=self.contract.subject_entity,
            trigger_obs_id=trigger.event_id if trigger else None,
            world_version=self.version,
        )

    def observers_of(self, ev: Event) -> tuple[str, ...]:
        """Which actors *could* observe this event at all. Visibility only — becoming
        available and being noticed are separate, later steps."""

        out = []
        for aid, actor in self.actors.items():
            if aid == ev.actor_id and aid not in ev.audience:
                # An actor does not receive its own public act as news — it already
                # knows it acted. It *does* receive things addressed to it personally,
                # including the world's verdict on what it just attempted.
                continue
            if self.visible_to(ev, aid, actor.role):
                out.append(aid)
        return tuple(out)

    @staticmethod
    def visible_to(ev: Event, actor_id: str, role: str) -> bool:
        if ev.visibility is Visibility.PUBLIC:
            return True
        if ev.visibility is Visibility.PRIVATE:
            return actor_id in ev.audience or actor_id in ev.target_ids
        if ev.visibility is Visibility.ROLE:
            return role in ev.audience
        return False

    # -- information lifecycle --------------------------------------------------

    def deliver(self, deliveries: tuple[Delivery, ...]) -> WorldState:
        """Record that events became available to actors. Availability is not
        awareness: nothing enters an actor's view until it is noticed."""

        if not deliveries:
            return self
        known = {(d.event_id, d.actor_id) for d in self.deliveries}
        fresh = tuple(d for d in deliveries if (d.event_id, d.actor_id) not in known)
        if not fresh:
            return self
        return replace(self, deliveries=self.deliveries + fresh, version=self.version + 1)

    def mark_noticed(self, actor_id: str, event_ids: frozenset[str], at: datetime) -> WorldState:
        """Flip delivered-but-unseen information to noticed for one actor."""

        changed = False
        out: list[Delivery] = []
        for d in self.deliveries:
            if d.actor_id == actor_id and d.event_id in event_ids and d.noticed_at is None:
                out.append(replace(d, noticed_at=at))
                changed = True
            else:
                out.append(d)
        if not changed:
            return self
        return replace(self, deliveries=tuple(out), version=self.version + 1)

    def available_unnoticed(self, actor_id: str, *, by: datetime) -> tuple[Delivery, ...]:
        return tuple(
            d
            for d in self.deliveries
            if d.actor_id == actor_id and d.noticed_at is None and d.available_at <= by
        )

    def noticed_event_ids(self, actor_id: str) -> frozenset[str]:
        return frozenset(
            d.event_id for d in self.deliveries if d.actor_id == actor_id and d.noticed
        )

    # -- schedule ---------------------------------------------------------------

    def with_schedule(self, schedule: Schedule) -> WorldState:
        return replace(self, schedule=schedule)

    # -- state transitions (all return new instances) --------------------------

    def apply(self, events: list[Event] | tuple[Event, ...]) -> WorldState:
        fields = dict(self.fields)
        records = {name: list(recs) for name, recs in self.records}
        documents = {did: dict(f) for did, f in self.documents}
        resources = dict(self.resources)
        commitments = list(self.commitments)
        history = list(self.event_history)
        time = self.time

        for raw in events:
            ev = replace(raw, status=EventStatus.APPLIED)
            data = ev.payload_dict
            kind = ev.kind

            if kind == "set_field":
                value = data.get("value")
                # An effect whose value the world could not determine — an expression
                # over a field nothing has set yet — states nothing. Writing it would
                # erase whatever else had produced that field.
                if value is not None:
                    fields[str(data["field"])] = value
            elif kind == "adjust_field":
                delta = data.get("delta")
                # A delta the world could not determine adjusts nothing. Writing
                # cur + 0.0 would DETERMINE a previously-unset field — defeating any
                # equals(field, None) unresolved guard and turning an unknown into a
                # confident answer. The feasibility check refuses such actions up
                # front; this is the last line of defense for events built elsewhere
                # (e.g. deferred effects re-resolved at fire time).
                if delta is not None:
                    cur = _num(fields.get(str(data["field"]), 0.0))
                    fields[str(data["field"])] = cur + _num(delta)
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
                raw_amt = data.get("amount", 0.0)
                # An undetermined amount moves nothing: "transfer what I said" must
                # never silently become "transfer nothing, recorded as done".
                if raw_amt is not None:
                    amt = _num(raw_amt)
                    frm = _reskey(res, str(data["from"]))
                    to = _reskey(res, str(data["to"]))
                    resources[frm] = _num(resources.get(frm, 0.0)) - amt
                    resources[to] = _num(resources.get(to, 0.0)) + amt
            elif kind == "consume_resource":
                raw_amt = data.get("amount", 0.0)
                if raw_amt is not None:
                    key = _reskey(str(data["resource"]), str(data["holder"]))
                    resources[key] = _num(resources.get(key, 0.0)) - _num(raw_amt)
            elif kind == "create_or_update_document":
                did = str(data["document"])
                documents.setdefault(did, {}).update(dict(data.get("fields", {})))

            history.append(ev)
            time = max(time, ev.time)

        return replace(
            self,
            fields=tuple(sorted(fields.items())),
            records=tuple((k, tuple(v)) for k, v in sorted(records.items())),
            documents=tuple((k, tuple(sorted(v.items()))) for k, v in sorted(documents.items())),
            resources=tuple(sorted(resources.items())),
            commitments=tuple(commitments),
            event_history=tuple(history),
            time=time,
            version=self.version + 1,
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

    def clone(self, new_branch_id: str, weight: BranchWeight) -> WorldState:
        """Fork a branch. Actor memory, plans and commitments are copied, never shared:
        two possible worlds must be able to diverge completely."""

        new_actors = {aid: a.clone() for aid, a in self.actors.items()}
        return replace(
            self,
            branch_id=new_branch_id,
            parent_branch_id=self.branch_id,
            weight=weight,
            actors=new_actors,
        )

    def information_digest(self) -> str:
        """A content hash of *what the actors have been told*, keyed on content.

        The other half of progress. ``state_digest`` covers fields, records, documents,
        resources, commitments and stage — not knowledge — so a world in which people
        correspond without writing world state looks frozen. Three rounds of ordinary
        pre-meeting correspondence were enough to have a branch killed by the watchdog
        before it reached its own scheduled session.

        Counting deliveries instead would undo the fix it exists beside: a runaway
        cascade delivers constantly, and its whole problem is that it delivers *the same
        thing* — four hundred notices reading "action_rejected", one hundred and
        ninety-nine reading the same sentence. Keying on content rather than on event
        ids separates them exactly: novel information grows this set, repetition does
        not.
        """

        by_event = {e.event_id: e for e in self.event_history}
        signatures = set()
        for d in self.deliveries:
            ev = by_event.get(d.event_id)
            if ev is None:
                continue
            signatures.add(f"{d.actor_id}|{ev.kind}|{ev.payload!r}")
        return content_id("winfo", *sorted(signatures))

    def state_digest(self) -> str:
        """A content hash of the decision-relevant world state, used to detect that a
        trajectory has stopped making progress. Time is excluded on purpose: a world
        where only the clock moves has not changed."""

        return content_id(
            "wstate",
            repr(self.fields),
            repr(tuple((k, tuple((r.key, r.value, r.by) for r in v)) for k, v in self.records)),
            repr(self.documents),
            repr(self.resources),
            repr(tuple((c.by, c.text, c.tag) for c in self.commitments)),
            self.stage,
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


def _to_observation(ev: Event, delivery: Delivery) -> Observation:
    data = ev.payload_dict
    info: dict[str, Any] = {}
    for key in _INFO_KEYS:
        sub = data.get(key)
        if isinstance(sub, dict):
            info.update(sub)
    assert delivery.noticed_at is not None
    return Observation(
        obs_id=ev.event_id,
        time=delivery.noticed_at,
        kind=ev.kind,
        source=ev.actor_id or "environment",
        summary=_summarize(ev),
        info_fields=tuple(sorted(info.items())),
        evidence_claim_ids=ev.evidence_claim_ids,
        occurred_at=ev.time,
        available_at=delivery.available_at,
        channel=delivery.channel,
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
