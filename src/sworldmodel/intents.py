"""Intent validation and execution — the environment is authoritative.

Actors emit *intentions*; this module turns them into *consequences* (events). The
route is always:

    view = world.view_for(actor_id, trigger)
    intent = actor_runtime.step(actor_state, view)
    validated = environment.validate(intent, world)
    events = environment.execute(validated, world)
    next_world = world.apply(events)

The actor never mutates the world. Messages and statements keep their substantive
content — a communication is never flattened to "actor emitted statement".
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .errors import IntentValidationError
from .ids import content_id
from .models import (
    Event,
    EventKind,
    Intent,
    IntentKind,
    Visibility,
    make_payload,
)
from .world import WorldState


class Environment:
    """Validates intents against the world and executes them into events.

    Holds a monotonic sequence counter so event ids are unique and deterministic
    given the (deterministic) rollout order.
    """

    def __init__(self) -> None:
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- validation -------------------------------------------------------------

    def validate(self, intent: Intent, world: WorldState) -> Intent:
        if intent.kind not in IntentKind.ALL:
            raise IntentValidationError(f"Unknown intent kind {intent.kind!r}")
        actor = world.actors.get(intent.actor_id)
        if actor is None:
            raise IntentValidationError(f"Unknown actor {intent.actor_id!r}")
        auth = actor.definition.authority

        if intent.kind == IntentKind.CAST_VOTE:
            if not actor.definition.is_voting_seat or "vote" not in auth:
                raise IntentValidationError(f"Actor {intent.actor_id} is not authorized to vote")
            if world.protocol_stage != "decision":
                raise IntentValidationError(
                    f"Vote by {intent.actor_id} is out of order: stage is "
                    f"{world.protocol_stage!r}, not 'decision'"
                )
            option = intent.payload_dict.get("option")
            if option not in world.contract.outcome_space:
                raise IntentValidationError(f"Vote option {option!r} not in outcome space")
        elif intent.kind in (IntentKind.INTRODUCE_PROPOSAL, IntentKind.REVISE_PROPOSAL):
            if "introduce_proposal" not in auth and "chair" not in auth:
                raise IntentValidationError(
                    f"Actor {intent.actor_id} may not introduce/revise a proposal"
                )
        return intent

    # -- execution --------------------------------------------------------------

    def execute(self, intent: Intent, world: WorldState) -> list[Event]:
        """Turn a validated intent into environment events. Consequences originate
        here, never in the actor's own output."""

        data = intent.payload_dict
        t = world.time

        if intent.kind == IntentKind.CAST_VOTE:
            # A vote is a private ballot; other actors cannot observe it.
            return [
                self._event(
                    world,
                    kind=EventKind.VOTE_CAST,
                    actor_id=intent.actor_id,
                    payload={"option": data["option"], "rationale": intent.rationale},
                    visibility=Visibility.PRIVATE,
                    audience=(),
                    time=t,
                )
            ]

        if intent.kind == IntentKind.MAKE_STATEMENT:
            # The substantive statement content is delivered to all members.
            return [
                self._event(
                    world,
                    kind=EventKind.STATEMENT_MADE,
                    actor_id=intent.actor_id,
                    payload={
                        "statement_text": data.get("statement_text", ""),
                        "favored_option": data.get("favored_option", ""),
                        "info_signals": data.get("info_signals", {}),
                    },
                    visibility=Visibility.PUBLIC,
                    time=t,
                )
            ]

        if intent.kind in (IntentKind.INTRODUCE_PROPOSAL, IntentKind.REVISE_PROPOSAL):
            kind = (
                EventKind.PROPOSAL_INTRODUCED
                if intent.kind == IntentKind.INTRODUCE_PROPOSAL
                else EventKind.PROPOSAL_REVISED
            )
            pid = content_id("prop", world.branch_id, data.get("option"), self._next_seq())
            return [
                self._event(
                    world,
                    kind=kind,
                    actor_id=intent.actor_id,
                    payload={
                        "proposal_id": pid,
                        "option": data.get("option"),
                        "text": data.get("text", ""),
                        "revision_of": data.get("revision_of"),
                    },
                    visibility=Visibility.PUBLIC,
                    time=t,
                )
            ]

        if intent.kind == IntentKind.REQUEST_INFORMATION:
            return [
                self._event(
                    world,
                    kind=EventKind.INFORMATION_REQUESTED,
                    actor_id=intent.actor_id,
                    payload={"question": intent.rationale},
                    visibility=Visibility.PRIVATE,
                    audience=(intent.actor_id,),
                    time=t,
                )
            ]

        if intent.kind == IntentKind.MAKE_COMMITMENT:
            return [
                self._event(
                    world,
                    kind=EventKind.COMMITMENT_MADE,
                    actor_id=intent.actor_id,
                    payload={"text": data.get("text", intent.rationale)},
                    visibility=Visibility.PUBLIC,
                    time=t,
                )
            ]

        if intent.kind == IntentKind.SEND_MESSAGE:
            targets = tuple(str(x) for x in data.get("targets", ()) if str(x) in world.actors)
            text = str(data.get("text", ""))
            sent = self._event(
                world,
                kind=EventKind.MESSAGE_SENT,
                actor_id=intent.actor_id,
                payload={"text": text, "targets": list(targets)},
                visibility=Visibility.PRIVATE,
                audience=targets,
                time=t,
            )
            delivered = self._event(
                world,
                kind=EventKind.MESSAGE_DELIVERED,
                actor_id=intent.actor_id,
                payload={"text": text},
                visibility=Visibility.PRIVATE,
                audience=targets,
                time=t,
            )
            return [sent, delivered]

        if intent.kind == IntentKind.OPERATIONAL_ACTION:
            return [
                self._event(
                    world,
                    kind=EventKind.OPERATIONAL_ACTION,
                    actor_id=intent.actor_id,
                    payload={
                        "action": data.get("action", ""),
                        "text": data.get("text", intent.rationale),
                    },
                    visibility=Visibility.PUBLIC,
                    time=t,
                )
            ]

        # WAIT / PRESERVE_PLAN / SUPPORT / OPPOSE -> a recorded, low-consequence event.
        return [
            self._event(
                world,
                kind=EventKind.WAIT_RECORDED,
                actor_id=intent.actor_id,
                payload={"kind": intent.kind, "rationale": intent.rationale},
                visibility=Visibility.PRIVATE,
                audience=(intent.actor_id,),
                time=t,
            )
        ]

    # -- environment-originated events (briefings, data, procedure) -------------

    def environment_event(
        self,
        world: WorldState,
        *,
        kind: str,
        payload: dict[str, object],
        time: datetime,
        visibility: Visibility = Visibility.PUBLIC,
        audience: tuple[str, ...] = (),
        actor_id: str | None = None,
    ) -> Event:
        return self._event(
            world,
            kind=kind,
            actor_id=actor_id,
            payload=payload,
            visibility=visibility,
            audience=audience,
            time=time,
        )

    def _event(
        self,
        world: WorldState,
        *,
        kind: str,
        actor_id: str | None,
        payload: dict[str, object],
        visibility: Visibility,
        time: datetime,
        audience: tuple[str, ...] = (),
    ) -> Event:
        seq = self._next_seq()
        eid = content_id("ev", world.branch_id, kind, actor_id, seq)
        return Event(
            event_id=eid,
            branch_id=world.branch_id,
            time=time,
            kind=kind,
            actor_id=actor_id,
            target_ids=audience,
            payload=make_payload(payload),
            visibility=visibility,
            audience=audience,
        )


def tick(time: datetime, seconds: int = 60) -> datetime:
    return time + timedelta(seconds=seconds)
