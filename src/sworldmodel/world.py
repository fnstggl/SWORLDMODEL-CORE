"""The one authoritative world state per branch.

There is exactly one :class:`WorldState` per branch. Everything an actor sees is a
*projection* of it (``view_for``); there is no competing "knowledge packet",
"deliberation state", or "institution override" that owns facts. All mutation flows
through :meth:`WorldState.apply`, which folds validated events into new state. The
state is treated as immutable: every update returns a fresh instance, and branch
clones deep-copy actor memory so worlds never share mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from .actors import ActorState, LocalView, Observation
from .evidence import EvidenceView
from .mechanisms import TerminalEvaluation
from .models import (
    BranchWeight,
    Entity,
    EpistemicType,
    Event,
    EventKind,
    EventStatus,
    InstitutionSpec,
    Proposal,
    ResolutionContract,
    Visibility,
)

# Event kinds that carry observable content into an actor's view. A raw
# MESSAGE_SENT is deliberately absent: a sent message is not automatically noticed;
# only a MESSAGE_DELIVERED (produced by the environment) is visible.
_OBSERVABLE_KINDS = frozenset(
    {
        EventKind.BRIEFING_DISTRIBUTED,
        EventKind.EXTERNAL_DATA_RELEASED,
        EventKind.PROPOSAL_INTRODUCED,
        EventKind.PROPOSAL_REVISED,
        EventKind.STATEMENT_MADE,
        EventKind.MESSAGE_DELIVERED,
        EventKind.INFORMATION_PROVIDED,
        EventKind.DECISION_OPENED,
    }
)


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


@dataclass(frozen=True)
class WorldState:
    branch_id: str
    parent_branch_id: str | None
    weight: BranchWeight
    time: datetime
    contract: ResolutionContract
    evidence: EvidenceView
    entities: tuple[Entity, ...] = ()
    actors: dict[str, ActorState] = field(default_factory=dict)
    institution: InstitutionSpec | None = None
    verified_facts: tuple[WorldFact, ...] = ()
    uncertain_facts: tuple[WorldFact, ...] = ()
    proposals: tuple[Proposal, ...] = ()
    votes: tuple[tuple[str, str], ...] = ()  # (actor_id, option) final votes
    signals: tuple[tuple[str, float], ...] = ()  # current external signal levels
    resources: tuple[tuple[str, float], ...] = ()
    commitments: tuple[tuple[str, str], ...] = ()
    external_processes: tuple[str, ...] = ()
    pending_events: tuple[Event, ...] = ()
    event_history: tuple[Event, ...] = ()
    protocol_stage: str = "initial"
    terminal_state: TerminalEvaluation | None = None

    # -- roster helpers ---------------------------------------------------------

    def voting_actor_ids(self) -> tuple[str, ...]:
        return tuple(aid for aid, a in self.actors.items() if a.definition.is_voting_seat)

    def vote_powers(self) -> dict[str, int]:
        return {
            aid: a.definition.vote_power
            for aid, a in self.actors.items()
            if a.definition.is_voting_seat
        }

    def votes_dict(self) -> dict[str, str]:
        return dict(self.votes)

    def signals_dict(self) -> dict[str, float]:
        return dict(self.signals)

    # -- projection -------------------------------------------------------------

    def view_for(self, actor_id: str, trigger: Event | None = None) -> LocalView:
        """Derive the local view an actor could have received by ``self.time``.

        Reads only public facts, delivered/visible events, visible proposals, and the
        institutional stage. It never reads another actor's private state or memory.
        """

        actor = self.actors[actor_id]
        definition = actor.definition
        role = definition.role

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
        # Proposals within the institution are visible to its members.
        visible_proposals = tuple(p for p in self.proposals if p.introduced_at <= self.time)
        rule = self.contract.decision_rule
        rule_summary = f"{rule.kind}: {rule.threshold} of {rule.total_seats} seats"

        return LocalView(
            actor_id=actor_id,
            branch_time=self.time,
            role=role,
            authority=definition.authority,
            stage=self.protocol_stage,
            options=self.contract.outcome_space,
            decision_rule_summary=rule_summary,
            visible_proposals=visible_proposals,
            public_facts=public_facts,
            observations=tuple(observations),
            feasible_actions=_feasible_actions(self.protocol_stage, definition.is_voting_seat),
            public_votes=(),  # secret ballot during the meeting
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
        proposals = list(self.proposals)
        votes = dict(self.votes)
        signals = dict(self.signals)
        commitments = list(self.commitments)
        stage = self.protocol_stage
        history = list(self.event_history)
        pending = list(self.pending_events)
        time = self.time

        for raw in events:
            ev = replace(raw, status=EventStatus.APPLIED)
            data = ev.payload_dict
            if ev.kind in (EventKind.PROPOSAL_INTRODUCED, EventKind.PROPOSAL_REVISED):
                proposals.append(
                    Proposal(
                        proposal_id=str(data["proposal_id"]),
                        option=str(data["option"]),
                        text=str(data.get("text", "")),
                        introduced_by=str(ev.actor_id or "environment"),
                        introduced_at=ev.time,
                        revision_of=data.get("revision_of"),
                        evidence_claim_ids=ev.evidence_claim_ids,
                    )
                )
            elif ev.kind == EventKind.DECISION_OPENED:
                stage = "decision"
            elif ev.kind == EventKind.VOTE_CAST and ev.actor_id is not None:
                votes[ev.actor_id] = str(data["option"])
            elif ev.kind in (EventKind.EXTERNAL_DATA_RELEASED, EventKind.BRIEFING_DISTRIBUTED):
                for name, level in dict(data.get("signals", {})).items():
                    signals[name] = float(level)
            elif ev.kind == EventKind.COMMITMENT_MADE and ev.actor_id is not None:
                commitments.append((ev.actor_id, str(data.get("text", ""))))

            history.append(ev)
            pending = [pe for pe in pending if pe.event_id != ev.event_id]
            time = max(time, ev.time)

        return replace(
            self,
            proposals=tuple(proposals),
            votes=tuple(sorted(votes.items())),
            signals=tuple(sorted(signals.items())),
            commitments=tuple(commitments),
            protocol_stage=stage,
            event_history=tuple(history),
            pending_events=tuple(pending),
            time=time,
        )

    def with_stage(self, stage: str) -> WorldState:
        return replace(self, protocol_stage=stage)

    def with_time(self, time: datetime) -> WorldState:
        return replace(self, time=max(self.time, time))

    def with_actor(self, actor: ActorState) -> WorldState:
        actors = dict(self.actors)
        actors[actor.actor_id] = actor
        return replace(self, actors=actors)

    def set_terminal(self, evaluation: TerminalEvaluation) -> WorldState:
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


def _feasible_actions(stage: str, is_voting_seat: bool) -> tuple[str, ...]:
    if stage == "decision":
        return ("cast_vote",) if is_voting_seat else ("wait",)
    if stage == "positions":
        return ("make_statement", "wait")
    return ("wait", "request_information")


def _to_observation(ev: Event) -> Observation:
    data = ev.payload_dict
    info = {k: float(v) for k, v in dict(data.get("info_signals", {})).items()}
    # A briefing / data release carries its levels as info signals too.
    for k, v in dict(data.get("signals", {})).items():
        info[k] = float(v)
    return Observation(
        obs_id=ev.event_id,
        time=ev.time,
        kind=ev.kind,
        source=ev.actor_id or "environment",
        summary=_summarize(ev),
        info_signals=tuple(sorted(info.items())),
        evidence_claim_ids=ev.evidence_claim_ids,
    )


def _summarize(ev: Event) -> str:
    data = ev.payload_dict
    if ev.kind == EventKind.STATEMENT_MADE:
        return f"{ev.actor_id} stated: {data.get('statement_text', '')}"
    if ev.kind == EventKind.EXTERNAL_DATA_RELEASED:
        return f"External data released: {dict(data.get('signals', {}))}"
    if ev.kind == EventKind.BRIEFING_DISTRIBUTED:
        return f"Staff briefing distributed: {data.get('text', '')} {dict(data.get('signals', {}))}"
    if ev.kind in (EventKind.PROPOSAL_INTRODUCED, EventKind.PROPOSAL_REVISED):
        return f"Proposal on the table: {data.get('option')} — {data.get('text', '')}"
    if ev.kind == EventKind.DECISION_OPENED:
        return "The final decision is now open for voting."
    if ev.kind == EventKind.MESSAGE_DELIVERED:
        return f"Message from {ev.actor_id}: {data.get('text', '')}"
    if ev.kind == EventKind.INFORMATION_PROVIDED:
        return f"Information provided: {data.get('text', '')}"
    return f"{ev.kind} event"
