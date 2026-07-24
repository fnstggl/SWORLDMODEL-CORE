"""Persistent actors and their cognitive loop.

An actor owns durable memory and a plan; it is not reconstructed from scratch at
every call. It receives a :class:`LocalView` — a read-only projection of the world it
could actually have perceived, plus the *feasible compiled actions* the engine offers
it at this moment and permission to propose a *novel* action — and returns a typed
:class:`ActionChoice`. It never touches ``WorldState``: it has no reference to it, so
it structurally cannot mutate reality or see another actor's private state.

The loop mirrors Generative Agents: perceive → retrieve → plan/react → reflect → emit.
The actor emits an *intention*; the environment (executor) decides the consequence.
Nothing here knows what a vote, a proposal, or a committee is — the action menu is
whatever the compiler produced for this world.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .memory import MemoryStream
from .prompts import render_decision_prompt, render_reflect_prompt
from .worldspec import ActionChoice, ActorSpec, EntitySpec

REFLECTION_TRIGGER = 1.5


@dataclass(frozen=True)
class Observation:
    """What an actor perceived from a delivered/visible event."""

    obs_id: str  # equal to the source event id
    time: datetime
    kind: str
    source: str  # actor id or "environment"
    summary: str
    info_fields: tuple[tuple[str, Any], ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "obs_id": self.obs_id,
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "info_fields": dict(self.info_fields),
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }


@dataclass(frozen=True)
class LocalView:
    """The read-only projection given to an actor. Built by ``world.view_for`` and
    then augmented by the engine with the feasible compiled actions for this moment."""

    actor_id: str
    branch_time: datetime
    role: str
    authority: tuple[str, ...]
    stage: str
    public_facts: tuple[str, ...]
    observations: tuple[Observation, ...]
    world_fields: tuple[tuple[str, Any], ...] = ()
    feasible_actions: tuple[dict[str, Any], ...] = ()
    allow_novel: bool = True
    question: str = ""
    subject: str = ""
    trigger_obs_id: str | None = None

    def observed_fields(self) -> dict[str, Any]:
        """Every field level this actor can currently read: the ambient world-field
        levels (a released signal, a negotiation gap — persistent shared state) plus
        anything carried in this turn's fresh observations."""

        fields: dict[str, Any] = dict(self.world_fields)
        for obs in self.observations:
            for name, level in obs.info_fields:
                fields[name] = level
        return fields


@dataclass
class ActorState:
    """Persistent per-actor runtime state. Cloned per branch so memories stay
    isolated across worlds."""

    entity_id: str
    entity: EntitySpec
    spec: ActorSpec
    memory: MemoryStream
    beliefs: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    active_plan: str = ""
    # Information the actor asked for and has not yet received. It persists across
    # invocations and is itself a reason the scheduler may bring the actor back.
    pending_questions: tuple[str, ...] = ()
    last_observed_event_ids: frozenset[str] = frozenset()
    _importance_accum: float = 0.0

    @property
    def actor_id(self) -> str:
        return self.entity_id

    @property
    def role(self) -> str:
        return self.entity.role

    @property
    def authority(self) -> tuple[str, ...]:
        return self.entity.authority

    @classmethod
    def from_spec(
        cls, entity: EntitySpec, spec: ActorSpec, *, default_time: datetime
    ) -> ActorState:
        mem = MemoryStream()
        mem.seed(_memory_seeds(spec), default_time=default_time)
        return cls(
            entity_id=entity.entity_id,
            entity=entity,
            spec=spec,
            memory=mem,
            beliefs=(spec.reasoning,) if spec.reasoning else (),
            active_plan=f"Act as {entity.role or entity.name} toward the world's outcome.",
        )

    def clone(self) -> ActorState:
        new_mem = MemoryStream()
        new_mem.nodes = list(self.memory.nodes)
        new_mem._index = dict(self.memory._index)
        return replace(self, memory=new_mem)

    def __deepcopy__(self, memo: dict[int, object]) -> ActorState:
        return self.clone()


def _memory_seeds(spec: ActorSpec) -> tuple[Any, ...]:
    from .models import MemorySeed

    seeds: list[MemorySeed] = []
    for raw in spec.memory_seeds:
        d = dict(raw)
        vt = d.get("valid_time")
        seeds.append(
            MemorySeed(
                content=str(d.get("content", "")),
                kind=str(d.get("kind", "episodic")),
                importance=float(d.get("importance", 0.6)),
                valid_time=datetime.fromisoformat(vt) if isinstance(vt, str) else None,
                evidence_claim_ids=tuple(d.get("evidence_claim_ids", []) or []),
                tags=tuple(d.get("tags", []) or []),
            )
        )
    return tuple(seeds)


class ActorRuntime:
    """Runs one actor step: perceive → retrieve → plan/react → reflect → emit."""

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    def step(
        self, actor: ActorState, view: LocalView, *, seed: int
    ) -> tuple[ActionChoice, ActorState, list[GatewayResponse], dict[str, object]]:
        responses: list[GatewayResponse] = []
        now = view.branch_time

        # 1. perceive
        new_obs = [o for o in view.observations if o.obs_id not in actor.last_observed_event_ids]
        importance_gain = 0.0
        for obs in new_obs:
            poignancy = _poignancy(obs)
            importance_gain += poignancy
            actor.memory.add_memory(
                obs.summary,
                kind="episodic",
                importance=poignancy,
                created=obs.time,
                evidence_claim_ids=obs.evidence_claim_ids,
                tags=(f"source:{obs.source}", f"kind:{obs.kind}"),
            )
        observed_ids = actor.last_observed_event_ids | {o.obs_id for o in new_obs}
        accum = actor._importance_accum + importance_gain

        # 2. retrieve
        query = " ".join([actor.active_plan, view.stage] + [o.summary for o in new_obs][:3])
        retrieved = actor.memory.retrieve(query, now=now, top_k=6)

        # 3. reflect (before deciding) when accumulated importance crosses the trigger.
        beliefs = actor.beliefs
        if accum >= REFLECTION_TRIGGER and new_obs:
            beliefs, rresp = self._reflect(actor, new_obs, now, seed)
            if rresp is not None:
                responses.append(rresp)
            accum = 0.0

        # 4. decide: ask the gateway for a typed ActionChoice over the feasible menu.
        context: dict[str, Any] = {
            "actor_id": actor.actor_id,
            "name": actor.entity.name,
            "role": actor.role,
            "authority": list(actor.authority),
            "attributes": actor.entity.attributes_dict,
            "stage": view.stage,
            "question": view.question,
            "subject": view.subject,
            "observations": [o.as_dict() for o in new_obs],
            "observed_fields": view.observed_fields(),
            "retrieved_memories": [
                {"id": n.node_id, "content": n.content, "kind": n.kind, "tags": list(n.tags)}
                for n in retrieved
            ],
            "public_facts": list(view.public_facts),
            "pending_information_needs": list(actor.pending_questions),
            "feasible_actions": list(view.feasible_actions),
            "allow_novel": view.allow_novel,
            "policy": _policy_dict(actor.spec),
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
        choice = self._to_choice(dresp.data, view)

        pending = actor.pending_questions
        need = dresp.data.get("pending_need")
        if choice.mode == "wait" and need:
            pending = tuple(sorted({*pending, str(need)}))
        elif choice.mode != "wait":
            pending = ()  # acting resolves the outstanding need

        new_actor = replace(
            actor,
            beliefs=beliefs,
            pending_questions=pending,
            last_observed_event_ids=observed_ids,
            _importance_accum=accum,
        )
        context["retrieved_memory_ids"] = [n.node_id for n in retrieved]
        return choice, new_actor, responses, context

    # -- helpers ----------------------------------------------------------------

    def _reflect(
        self, actor: ActorState, new_obs: list[Observation], now: datetime, seed: int
    ) -> tuple[tuple[str, ...], GatewayResponse | None]:
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
        beliefs = actor.beliefs
        for mem in _as_list(rresp.data.get("new_memories")):
            ev_ids: tuple[str, ...] = ()
            if isinstance(mem, str):
                content, kind, importance = mem, "thought", 0.5
            elif isinstance(mem, dict):
                content = str(mem.get("content", "")).strip()
                kind = str(mem.get("kind", "thought"))
                importance = _as_float(mem.get("importance"), 0.5)
                raw_ev = mem.get("evidence_claim_ids")
                ev_ids = tuple(str(x) for x in raw_ev) if isinstance(raw_ev, list) else ()
            else:
                continue
            if content:
                actor.memory.add_memory(
                    content,
                    kind=kind,
                    importance=importance,
                    created=now,
                    evidence_claim_ids=ev_ids,
                    depth=1,
                )
        raw_beliefs = rresp.data.get("beliefs_update")
        if isinstance(raw_beliefs, list):
            beliefs = tuple(list(actor.beliefs) + [str(x) for x in raw_beliefs])
        return beliefs, rresp

    def _to_choice(self, data: dict[str, Any], view: LocalView) -> ActionChoice:
        mode = str(data.get("action_mode", data.get("mode", "wait")))
        if mode not in ("compiled_action", "novel_action", "wait"):
            mode = "wait"
        raw_novel = data.get("novel_action")
        novel: dict[str, Any] = raw_novel if isinstance(raw_novel, dict) else {}
        return ActionChoice(
            mode=mode,
            action_id=str(data.get("compiled_action_id", data.get("action_id", ""))),
            params=_as_items(data.get("params")),
            target=str(data.get("target", "")),
            novel_description=str(novel.get("description", "")),
            novel_intended_effect=str(novel.get("intended_effect", "")),
            novel_target=str(novel.get("target", "")),
            novel_params=_as_items(novel.get("parameters", novel.get("params"))),
            rationale=str(data.get("reasoning", data.get("rationale", ""))),
            referenced_memory_ids=_as_str_tuple(data.get("referenced_memory_ids")),
            referenced_observation_ids=_as_str_tuple(data.get("referenced_observation_ids")),
        )


def _policy_dict(spec: ActorSpec) -> dict[str, Any]:
    p = spec.policy
    return {
        "default_action_id": p.default_action_id,
        "default_params": p.default_params_dict,
        "default_novel": p.default_novel_dict,
        "rules": [
            {
                "when_field": r.when_field,
                "op": r.op,
                "value": r.value,
                "action_id": r.action_id,
                "params": r.params_dict,
                "novel": r.novel_dict,
            }
            for r in p.rules
        ],
    }


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_items(value: object) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return ()


def _as_str_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(x) for x in value) if isinstance(value, (list, tuple)) else ()


def _poignancy(obs: Observation) -> float:
    if obs.info_fields or obs.kind in {"release_data", "create_event"}:
        return 0.8
    if obs.kind in {"deliver_information", "append_record", "create_or_update_document"}:
        return 0.5
    return 0.3
