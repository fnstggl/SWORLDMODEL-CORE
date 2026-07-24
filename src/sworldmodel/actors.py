"""Persistent actors and their cognitive loop.

An actor owns durable memory and a plan; it is not reconstructed from scratch at
every call. It receives a :class:`LocalView` — a read-only projection of the world
it could actually have perceived — and returns a typed :class:`Intent`. It never
touches ``WorldState``: it has no reference to it, so it structurally cannot mutate
reality or see another actor's private state.

The loop mirrors Generative Agents: perceive → retrieve → plan/react → reflect →
emit intent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from .errors import IntentValidationError
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .memory import MemoryStream
from .models import (
    ActorDefinition,
    Intent,
    IntentKind,
    Proposal,
    make_payload,
)
from .prompts import render_decision_prompt, render_reflect_prompt

# Importance accumulated (sum of observation poignancy) that triggers a reflection.
REFLECTION_TRIGGER = 1.5


@dataclass(frozen=True)
class Observation:
    """What an actor perceived from a delivered/visible event."""

    obs_id: str  # equal to the source event id
    time: datetime
    kind: str
    source: str  # actor id or "environment"
    summary: str
    info_signals: tuple[tuple[str, float], ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "obs_id": self.obs_id,
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "info_signals": dict(self.info_signals),
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }


@dataclass(frozen=True)
class LocalView:
    """The read-only projection given to an actor. Built solely by
    ``world.view_for``; contains only what this actor could have received."""

    actor_id: str
    branch_time: datetime
    role: str
    authority: tuple[str, ...]
    stage: str
    options: tuple[str, ...]
    decision_rule_summary: str
    visible_proposals: tuple[Proposal, ...]
    public_facts: tuple[str, ...]
    observations: tuple[Observation, ...]
    feasible_actions: tuple[str, ...]
    public_votes: tuple[tuple[str, str], ...] = ()
    trigger_obs_id: str | None = None

    def active_proposal(self) -> Proposal | None:
        return self.visible_proposals[-1] if self.visible_proposals else None


@dataclass
class ActorState:
    """Persistent per-actor runtime state. Cloned per branch so memories stay
    isolated across worlds."""

    actor_id: str
    definition: ActorDefinition
    memory: MemoryStream
    beliefs: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    active_plan: str = ""
    pending_questions: tuple[str, ...] = ()
    last_observed_event_ids: frozenset[str] = frozenset()
    private_uncertainties: tuple[str, ...] = ()
    _importance_accum: float = 0.0

    @classmethod
    def from_definition(cls, definition: ActorDefinition, *, default_time: datetime) -> ActorState:
        mem = MemoryStream()
        mem.seed(definition.memory_seeds, default_time=default_time)
        return cls(
            actor_id=definition.actor_id,
            definition=definition,
            memory=mem,
            beliefs=(definition.conditional_behavior.reasoning,),
            goals=(),
            active_plan=f"Hold current inclination: {definition.conditional_behavior.current_inclination}",
        )

    def clone(self) -> ActorState:
        new_mem = MemoryStream()
        new_mem.nodes = list(self.memory.nodes)
        new_mem._index = dict(self.memory._index)
        return replace(self, memory=new_mem)

    def __deepcopy__(self, memo: dict[int, object]) -> ActorState:
        return self.clone()


class ActorRuntime:
    """Runs one actor step: perceive → retrieve → plan/react → reflect → emit."""

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    def step(
        self, actor: ActorState, view: LocalView, *, seed: int
    ) -> tuple[Intent, ActorState, list[GatewayResponse], dict[str, object]]:
        responses: list[GatewayResponse] = []
        now = view.branch_time

        # 1. perceive: form observations from newly-delivered/visible events.
        new_obs = [o for o in view.observations if o.obs_id not in actor.last_observed_event_ids]
        importance_gain = 0.0
        for obs in new_obs:
            poignancy = _poignancy(obs)
            importance_gain += poignancy
            actor.memory.add_memory(
                f"{obs.summary}",
                kind="episodic",
                importance=poignancy,
                created=obs.time,
                evidence_claim_ids=obs.evidence_claim_ids,
                tags=(f"source:{obs.source}", f"kind:{obs.kind}"),
            )
        observed_ids = actor.last_observed_event_ids | {o.obs_id for o in new_obs}
        accum = actor._importance_accum + importance_gain

        # 2. retrieve: pull relevant memories for the current trigger.
        proposal = view.active_proposal()
        query = " ".join(
            [
                actor.definition.conditional_behavior.current_inclination,
                proposal.option if proposal else "",
                " ".join(o.summary for o in new_obs),
                actor.active_plan,
            ]
        )
        retrieved = actor.memory.retrieve(query, now=now, top_k=6)

        # 4. reflect (before deciding) when accumulated importance crosses the trigger.
        beliefs = actor.beliefs
        if accum >= REFLECTION_TRIGGER and new_obs:
            reflect_ctx = {
                "actor_id": actor.actor_id,
                "beliefs": list(actor.beliefs),
                "observations": [o.as_dict() for o in new_obs],
            }
            rresp = self.gateway.generate(
                GatewayRequest(
                    task_kind="reflect",
                    prompt=render_reflect_prompt(reflect_ctx),
                    context=reflect_ctx,
                    seed=seed,
                )
            )
            responses.append(rresp)
            for mem in rresp.data.get("new_memories", []):
                actor.memory.add_memory(
                    mem["content"],
                    kind=mem.get("kind", "thought"),
                    importance=float(mem.get("importance", 0.5)),
                    created=now,
                    evidence_claim_ids=tuple(mem.get("evidence_claim_ids", [])),
                    depth=1,
                )
            beliefs = tuple(list(actor.beliefs) + list(rresp.data.get("beliefs_update", [])))
            accum = 0.0

        # 3. plan/react: ask the model for a typed intention.
        cb = actor.definition.conditional_behavior
        context = {
            "actor_id": actor.actor_id,
            "name": actor.definition.name,
            "role": actor.definition.role,
            "authority": list(actor.definition.authority),
            "stage": view.stage,
            "options": list(view.options),
            "current_inclination": cb.current_inclination,
            "dissent_threshold": cb.dissent_threshold,
            "acceptance_tolerance": cb.acceptance_tolerance,
            "reaction_rules": [_rule_dict(r) for r in cb.reaction_rules],
            "active_proposal": (
                {"option": proposal.option, "proposal_id": proposal.proposal_id}
                if proposal
                else None
            ),
            "observed_signals": {},
            "observations": [o.as_dict() for o in new_obs],
            "retrieved_memories": [
                {"id": n.node_id, "content": n.content, "kind": n.kind, "tags": list(n.tags)}
                for n in retrieved
            ],
            "public_facts": list(view.public_facts),
            "feasible_actions": list(view.feasible_actions),
        }
        dresp = self.gateway.generate(
            GatewayRequest(
                task_kind="actor_decision",
                prompt=render_decision_prompt(context),
                context=context,
                seed=seed,
            )
        )
        responses.append(dresp)
        intent = self._to_intent(actor.actor_id, dresp.data, view)

        # 5. update persistent state (still no external mutation).
        pending = actor.pending_questions
        need = dresp.data.get("pending_need")
        if intent.kind == IntentKind.WAIT and need:
            pending = tuple(sorted({*pending, str(need)}))
        elif intent.kind == IntentKind.CAST_VOTE:
            pending = ()  # resolved by voting

        new_actor = replace(
            actor,
            beliefs=beliefs,
            last_observed_event_ids=observed_ids,
            pending_questions=pending,
            _importance_accum=accum,
        )
        context["retrieved_memory_ids"] = [n.node_id for n in retrieved]
        return intent, new_actor, responses, context

    def _to_intent(self, actor_id: str, data: dict[str, object], view: LocalView) -> Intent:
        kind = str(data.get("kind", IntentKind.WAIT))
        if kind not in IntentKind.ALL:
            raise IntentValidationError(f"Actor {actor_id} emitted unknown intent kind {kind!r}")
        payload: dict[str, object] = {}
        if kind == IntentKind.CAST_VOTE:
            option = str(data.get("vote_option", "")).strip()
            if option not in view.options:
                raise IntentValidationError(
                    f"Actor {actor_id} voted for {option!r} not in options {view.options}"
                )
            payload["option"] = option
        if kind == IntentKind.MAKE_STATEMENT:
            payload["statement_text"] = str(data.get("statement_text", ""))
            payload["favored_option"] = str(data.get("favored_option", ""))
            payload["info_signals"] = _as_dict(data.get("info_signals"))
        if kind == IntentKind.SEND_MESSAGE:
            payload["text"] = str(data.get("text", data.get("message", "")))
            payload["targets"] = _as_str_tuple(data.get("targets"))
        if kind == IntentKind.OPERATIONAL_ACTION:
            payload["action"] = str(data.get("action", ""))
            payload["text"] = str(data.get("text", ""))
        if kind == IntentKind.MAKE_COMMITMENT:
            payload["text"] = str(data.get("text", data.get("statement_text", "")))
        return Intent(
            actor_id=actor_id,
            kind=kind,
            payload=make_payload(payload),
            rationale=str(data.get("rationale", "")),
            referenced_memory_ids=_as_str_tuple(data.get("referenced_memory_ids")),
            referenced_observation_ids=_as_str_tuple(data.get("referenced_observation_ids")),
            expected_effect=str(data.get("expected_effect", "")),
        )


def _as_dict(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _as_str_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(x) for x in value) if isinstance(value, list | tuple) else ()


def _poignancy(obs: Observation) -> float:
    """A small, deterministic importance heuristic (no model call)."""

    weighty = {
        "external_data_released",
        "proposal_introduced",
        "proposal_revised",
        "decision_opened",
    }
    if obs.kind in weighty or obs.info_signals:
        return 0.8
    if obs.kind in {"statement_made", "message_delivered", "briefing_distributed"}:
        return 0.5
    return 0.3


def _rule_dict(rule: object) -> dict[str, object]:
    from .models import ReactionRule

    assert isinstance(rule, ReactionRule)
    return {
        "trigger_signal": rule.trigger_signal,
        "direction": rule.direction,
        "threshold": rule.threshold,
        "moves_to_option": rule.moves_to_option,
        "rationale": rule.rationale,
    }
