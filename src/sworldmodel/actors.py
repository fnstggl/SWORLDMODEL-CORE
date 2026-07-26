"""Persistent actors: continuous cognition between events.

An actor is a *being that persists*, not a function called at protocol stages. Between
invocations it keeps its identity, memories, beliefs, goals, an **active plan**, an
**ongoing action**, **commitments**, **pending information needs** and **revisit
conditions**. It is invoked only when something in the world actually reaches it and
is material to that state (see :mod:`engine`), and when invoked its first question is
not "what should I do about the whole world" but "does what just happened change my
plan?".

The loop follows Generative Agents' persistent cognition — perceive → retrieve →
(reflect) → plan-continuation → emit — but without Smallville's fixed ticks, tile
maze, vision radius, attention bandwidth, retention count, hand-tuned retrieval
weights or importance threshold. Nothing schedules an actor here; nothing invents
activity here.

The actor emits an **intention**. It never touches :class:`~sworldmodel.world.WorldState`
— it has no reference to one — so it structurally cannot mutate reality, see another
actor's private state, or assert that its action succeeded. The environment decides
the consequence, and a malformed or unusable provider response is a provider failure,
never a fabricated decision to wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .errors import GatewayError
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .memory import MemoryStream
from .prompts import render_decision_prompt, render_reflect_prompt
from .worldspec import ActionChoice, ActorSpec, EntitySpec

# Plan lifecycle states. A plan keeps its identity across all of them, so the trace can
# show "the same plan" being paused and resumed rather than a new plan appearing.
PLAN_ACTIVE = "active"
PLAN_PAUSED = "paused"
PLAN_COMPLETED = "completed"
PLAN_ABANDONED = "abandoned"

# What an actor did with its plan on a given invocation. Recorded on every decision.
PLAN_DISPOSITIONS = frozenset(
    {"continue", "revise", "interrupt", "replace", "complete", "abandon", "none"}
)


@dataclass(frozen=True)
class Observation:
    """What an actor actually noticed. Reaching an actor's view means it passed
    visibility, was delivered, and was noticed — three separate transitions recorded
    on the event ledger, not one."""

    obs_id: str  # equal to the source event id
    time: datetime  # when the actor NOTICED it (not when it happened)
    kind: str
    source: str  # actor id or "environment"
    summary: str
    info_fields: tuple[tuple[str, Any], ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()
    occurred_at: datetime | None = None  # when the underlying event happened
    available_at: datetime | None = None  # when it became reachable by this actor
    channel: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "obs_id": self.obs_id,
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "info_fields": dict(self.info_fields),
            "evidence_claim_ids": list(self.evidence_claim_ids),
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
            "available_at": self.available_at.isoformat() if self.available_at else None,
            "noticed_at": self.time.isoformat(),
            "channel": self.channel,
        }


@dataclass(frozen=True)
class PlanStep:
    """One concrete step of an actor's plan. ``at`` is when the actor intends to do it;
    a step with a time schedules a real future opportunity in the world."""

    description: str
    intended_action_id: str = ""
    at: datetime | None = None
    status: str = "planned"  # planned | active | done | abandoned

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "intended_action_id": self.intended_action_id,
            "at": self.at.isoformat() if self.at else None,
            "status": self.status,
        }


@dataclass(frozen=True)
class Plan:
    """A persistent plan. It survives between invocations and is only changed by the
    actor itself; the runtime never rewrites one.

    ``basis`` records why this plan is admissible at all — a verified schedule, a role
    obligation, an existing commitment, or a step already begun in this trajectory. A
    plan with no basis is not an inference, it is invention, and the compiler/actor
    prompt requires one.
    """

    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...] = ()
    status: str = PLAN_ACTIVE
    created: datetime | None = None
    revised: datetime | None = None
    revision_count: int = 0
    interrupt_when: tuple[str, ...] = ()
    basis: str = ""
    evidence_claim_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.as_dict() for s in self.steps],
            "status": self.status,
            "created": self.created.isoformat() if self.created else None,
            "revised": self.revised.isoformat() if self.revised else None,
            "revision_count": self.revision_count,
            "interrupt_when": list(self.interrupt_when),
            "basis": self.basis,
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }


@dataclass(frozen=True)
class OngoingAction:
    """An action this actor has begun and not yet finished. While one is in progress the
    actor is *busy*: an immaterial observation is noticed and remembered but does not
    trigger a fresh decision, so the plan persists."""

    action_id: str
    description: str
    started: datetime
    expected_completion: datetime
    status: str = "in_progress"  # in_progress | completed | failed | interrupted
    world_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "description": self.description,
            "started": self.started.isoformat(),
            "expected_completion": self.expected_completion.isoformat(),
            "status": self.status,
            "world_version": self.world_version,
        }


@dataclass(frozen=True)
class PendingNeed:
    """Information the actor asked for and has not received. It persists, it is a
    reason the actor may be woken again, and when its deadline passes without an answer
    the actor is woken *for the failure* rather than left waiting forever."""

    question: str
    asked_at: datetime
    asked_of: str = ""
    deadline: datetime | None = None
    resolved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "asked_at": self.asked_at.isoformat(),
            "asked_of": self.asked_of,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class RevisitCondition:
    """A condition the actor itself named for coming back to a decision.

    Only mechanically checkable forms are accepted, so the runtime enforces them
    without a second model call and without an arbitrary relevance score:

    ``at``                    an exact branch datetime
    ``on_field_change``       a named world field changes
    ``on_information_from``   a named actor communicates
    ``on_record_in``          a record is appended to a named collection
    """

    description: str
    at: datetime | None = None
    on_field_change: str = ""
    on_information_from: str = ""
    on_record_in: str = ""

    def is_checkable(self) -> bool:
        return bool(
            self.at or self.on_field_change or self.on_information_from or self.on_record_in
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "at": self.at.isoformat() if self.at else None,
            "on_field_change": self.on_field_change,
            "on_information_from": self.on_information_from,
            "on_record_in": self.on_record_in,
        }


@dataclass(frozen=True)
class ActorCommitment:
    """Something the actor undertook to do, with a due time. When it falls due the
    actor is woken — a commitment is a real future cause, not decoration."""

    text: str
    made_at: datetime
    due: datetime | None = None
    to_actor: str = ""
    discharged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "made_at": self.made_at.isoformat(),
            "due": self.due.isoformat() if self.due else None,
            "to_actor": self.to_actor,
            "discharged": self.discharged,
        }


@dataclass(frozen=True)
class LocalView:
    """The read-only projection given to an actor: what it has *noticed*, what it knows,
    what it is currently doing, and which compiled actions are feasible for it right now.
    Built by ``world.view_for`` and completed by the engine with the feasible menu."""

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
    trigger_kind: str = ""
    trigger_detail: str = ""
    trigger_obs_id: str | None = None
    world_version: int = 0

    def observed_fields(self) -> dict[str, Any]:
        """Every field level this actor can currently read.

        Observations fold in the order they were noticed (the view keeps them sorted
        by notice time, then by event id), so two messages about the same field
        noticed at different times resolve to the later one; two noticed at the SAME
        instant resolve deterministically by event id — an arbitrary but stable
        tie-break, not a claim about which message "wins" in the world. The world's
        own *current* field levels are then laid over the top: an ambient level the
        world determines now is what the actor can read now, and a stale number
        quoted in an older message must not shadow it. A field the world has not
        determined stays at whatever the messages carried — the view never invents a
        level.
        """

        fields: dict[str, Any] = {}
        for obs in self.observations:
            for name, level in obs.info_fields:
                fields[name] = level
        for name, level in self.world_fields:
            if level is not None:
                fields[name] = level
        return fields


@dataclass
class ActorState:
    """Persistent per-actor state. Cloned per branch so memories, plans and commitments
    stay isolated across worlds."""

    entity_id: str
    entity: EntitySpec
    spec: ActorSpec
    memory: MemoryStream
    # The compiled grounding profile (grounding.ActorGroundingProfile). Typed as object
    # to keep the import graph acyclic; set by the world compiler for every actor.
    grounding: object | None = None
    beliefs: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    plan: Plan | None = None
    current_action: OngoingAction | None = None
    commitments: tuple[ActorCommitment, ...] = ()
    pending_needs: tuple[PendingNeed, ...] = ()
    revisit_conditions: tuple[RevisitCondition, ...] = ()
    relationships: tuple[tuple[str, str], ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    last_decision_time: datetime | None = None
    decision_count: int = 0
    noticed_event_ids: frozenset[str] = frozenset()
    # Events delivered to this actor and available, but not yet noticed.
    available_event_ids: frozenset[str] = frozenset()

    @property
    def actor_id(self) -> str:
        return self.entity_id

    @property
    def role(self) -> str:
        return self.entity.role

    @property
    def authority(self) -> tuple[str, ...]:
        return self.entity.authority

    @property
    def is_busy(self) -> bool:
        return self.current_action is not None and self.current_action.status == "in_progress"

    def open_needs(self) -> tuple[PendingNeed, ...]:
        return tuple(n for n in self.pending_needs if not n.resolved)

    def open_commitments(self) -> tuple[ActorCommitment, ...]:
        return tuple(c for c in self.commitments if not c.discharged)

    @classmethod
    def from_spec(
        cls, entity: EntitySpec, spec: ActorSpec, *, default_time: datetime
    ) -> ActorState:
        mem = MemoryStream()
        mem.seed(_memory_seeds(spec), default_time=default_time)
        plan: Plan | None = None
        if spec.initial_plan_goal:
            plan = Plan(
                plan_id=f"plan_{entity.entity_id}_0",
                goal=spec.initial_plan_goal,
                steps=tuple(
                    PlanStep(
                        description=str(d.get("description", "")),
                        intended_action_id=str(d.get("intended_action_id", "")),
                        at=_parse_dt(d.get("at")),
                    )
                    for d in spec.initial_plan_steps_dicts()
                ),
                created=default_time,
                revised=default_time,
                basis=spec.initial_plan_basis,
                evidence_claim_ids=spec.initial_plan_evidence_ids,
            )
        return cls(
            entity_id=entity.entity_id,
            entity=entity,
            spec=spec,
            memory=mem,
            beliefs=(spec.reasoning,) if spec.reasoning else (),
            goals=spec.goals,
            plan=plan,
            commitments=tuple(
                ActorCommitment(
                    text=str(d.get("text", "")),
                    made_at=_parse_dt(d.get("made_at")) or default_time,
                    due=_parse_dt(d.get("due")),
                    to_actor=str(d.get("to_actor", "")),
                )
                for d in spec.initial_commitment_dicts()
            ),
        )

    def clone(self) -> ActorState:
        new_mem = MemoryStream()
        new_mem.nodes = list(self.memory.nodes)
        new_mem._index = dict(self.memory._index)
        return replace(self, memory=new_mem)

    def __deepcopy__(self, memo: dict[int, object]) -> ActorState:
        return self.clone()

    def state_dict(self) -> dict[str, Any]:
        """The persistent state, for the trace. This is what "the actor before/after the
        call" means concretely."""

        return {
            "beliefs": list(self.beliefs),
            "goals": list(self.goals),
            "plan": self.plan.as_dict() if self.plan else None,
            "current_action": self.current_action.as_dict() if self.current_action else None,
            "commitments": [c.as_dict() for c in self.commitments],
            "pending_needs": [n.as_dict() for n in self.pending_needs],
            "revisit_conditions": [r.as_dict() for r in self.revisit_conditions],
            "relationships": dict(self.relationships),
            "unresolved_questions": list(self.unresolved_questions),
            "last_decision_time": (
                self.last_decision_time.isoformat() if self.last_decision_time else None
            ),
            "decision_count": self.decision_count,
        }


def _memory_seeds(spec: ActorSpec) -> tuple[Any, ...]:
    """The actor's starting memories — the *cited* ones only.

    An uncited seed is withheld from the grounding block, and it must be withheld here
    too: a memory is rendered to the actor under "RETRIEVED MEMORIES", where it reads
    exactly like something the actor knows. Excluding it from one prompt section and
    admitting it through another would put the same invented fact in front of the same
    actor by a different door.
    """

    from .models import MemorySeed

    seeds: list[MemorySeed] = []
    for raw in spec.memory_seeds:
        d = dict(raw)
        if not (d.get("evidence_claim_ids") or ()):
            continue
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


@dataclass
class DecisionResult:
    """Everything one actor invocation produced."""

    choice: ActionChoice
    actor: ActorState
    responses: list[GatewayResponse] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    plan_disposition: str = "none"
    plan_before: dict[str, Any] | None = None
    plan_after: dict[str, Any] | None = None
    retrieved_memory_ids: list[str] = field(default_factory=list)
    noticed_obs_ids: list[str] = field(default_factory=list)


class ActorRuntime:
    """Runs one actor invocation. It does not decide *whether* to run — the engine does,
    from real events — and it never fabricates a choice."""

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    def step(self, actor: ActorState, view: LocalView, *, seed: int) -> DecisionResult:
        responses: list[GatewayResponse] = []
        now = view.branch_time
        plan_before = actor.plan.as_dict() if actor.plan else None

        # 1. perceive — the view already contains only what this actor NOTICED. Newly
        #    noticed items enter episodic memory once.
        new_obs = [o for o in view.observations if o.obs_id not in actor.noticed_event_ids]
        for obs in new_obs:
            actor.memory.add_memory(
                obs.summary,
                kind="episodic",
                importance=_seed_importance(obs),
                created=obs.time,
                evidence_claim_ids=obs.evidence_claim_ids,
                tags=(f"source:{obs.source}", f"kind:{obs.kind}"),
            )
        noticed_ids = actor.noticed_event_ids | {o.obs_id for o in new_obs}

        # 2. retrieve — query is the actor's actual situation: its plan, what it is
        #    doing, what it is waiting for, and what just reached it.
        query = " ".join(
            [
                actor.plan.goal if actor.plan else "",
                actor.current_action.description if actor.current_action else "",
                *[n.question for n in actor.open_needs()][:2],
                *[o.summary for o in new_obs][:3],
            ]
        ).strip()
        retrieved = actor.memory.retrieve(query, now=now, top_k=6)

        # 3. decide — one call that covers plan continuation and the intention. The
        #    actor is told what it was already doing, so it is not reconsidering the
        #    world from scratch.
        context = self._context(actor, view, new_obs, retrieved)
        rendered_prompt = render_decision_prompt(context)
        dresp = self.gateway.generate(
            GatewayRequest(
                task_kind="actor_decision",
                prompt=rendered_prompt,
                context=context,
                seed=seed,
            )
        )
        responses.append(dresp)
        context["rendered_prompt"] = rendered_prompt
        context["provider_response"] = dict(dresp.data)

        choice = self._to_choice(dresp.data, actor)
        disposition = self._disposition(dresp.data)

        new_actor = self._apply_cognitive_updates(
            actor,
            dresp.data,
            disposition,
            now=now,
            noticed_ids=noticed_ids,
            view=view,
        )

        # 4. reflect — only when the actor itself says its understanding changed. There
        #    is no accumulated-importance threshold: an arbitrary constant deciding when
        #    a person reconsiders their beliefs is exactly the kind of invented social
        #    number this runtime refuses to carry.
        if bool(dresp.data.get("reflection_needed")) and new_obs:
            beliefs, rresp = self._reflect(new_actor, new_obs, now, seed)
            if rresp is not None:
                responses.append(rresp)
            new_actor = replace(new_actor, beliefs=beliefs)

        context["retrieved_memory_ids"] = [n.node_id for n in retrieved]
        return DecisionResult(
            choice=choice,
            actor=new_actor,
            responses=responses,
            context=context,
            plan_disposition=disposition,
            plan_before=plan_before,
            plan_after=new_actor.plan.as_dict() if new_actor.plan else None,
            retrieved_memory_ids=[n.node_id for n in retrieved],
            noticed_obs_ids=[o.obs_id for o in new_obs],
        )

    # -- context / prompt --------------------------------------------------------

    def _context(
        self,
        actor: ActorState,
        view: LocalView,
        new_obs: list[Observation],
        retrieved: list[Any],
    ) -> dict[str, Any]:
        profile = actor.grounding
        return {
            "actor_id": actor.actor_id,
            "name": actor.entity.name,
            "canonical_identity": getattr(profile, "canonical_identity", actor.entity.name),
            "role": actor.role,
            "authority": list(actor.authority),
            "attributes": actor.entity.attributes_dict,
            "actor_grounding": _render_grounding(profile),
            "actor_evidence_claim_ids": list(getattr(profile, "claim_ids", ())),
            "stage": view.stage,
            "branch_time": view.branch_time.isoformat(),
            "question": view.question,
            "subject": view.subject,
            "why_you_are_deciding_now": {
                "trigger": view.trigger_kind,
                "detail": view.trigger_detail,
                "about_observation": view.trigger_obs_id,
            },
            "observations": [o.as_dict() for o in new_obs],
            "observed_fields": view.observed_fields(),
            "retrieved_memories": [
                {"id": n.node_id, "content": n.content, "kind": n.kind, "tags": list(n.tags)}
                for n in retrieved
            ],
            "public_facts": list(view.public_facts),
            "active_plan": actor.plan.as_dict() if actor.plan else None,
            "current_action": (actor.current_action.as_dict() if actor.current_action else None),
            "your_commitments": [c.as_dict() for c in actor.open_commitments()],
            "pending_information_needs": [n.as_dict() for n in actor.open_needs()],
            "your_revisit_conditions": [r.as_dict() for r in actor.revisit_conditions],
            "unresolved_questions": list(actor.unresolved_questions),
            "goals": list(actor.goals),
            "beliefs": list(actor.beliefs),
            "relationships": dict(actor.relationships),
            "last_decision_time": (
                actor.last_decision_time.isoformat() if actor.last_decision_time else None
            ),
            "feasible_actions": list(view.feasible_actions),
            "allow_novel": view.allow_novel,
        }

    # -- response handling -------------------------------------------------------

    def _disposition(self, data: dict[str, Any]) -> str:
        raw = str(data.get("plan_disposition", "")).strip().lower()
        if raw in PLAN_DISPOSITIONS:
            return raw
        raise GatewayError(
            f"actor response has no usable plan_disposition (got {raw!r}); "
            f"expected one of {sorted(PLAN_DISPOSITIONS)}"
        )

    def _to_choice(self, data: dict[str, Any], actor: ActorState) -> ActionChoice:
        """Turn the provider's structured response into a typed intention.

        This is the *only* place a response becomes an intention, and it performs no
        semantic substitution: an unusable response raises, leaving the branch's mass
        explicitly unresolved. It never becomes "wait", never becomes the nearest
        compiled action, and never becomes a default.
        """

        mode = str(data.get("action_mode", data.get("mode", ""))).strip()
        if mode not in ("compiled_action", "novel_action", "wait"):
            raise GatewayError(
                f"actor {actor.actor_id!r} returned action_mode={mode!r}, which is not one of "
                "compiled_action|novel_action|wait; refusing to substitute an action"
            )
        raw_novel = data.get("novel_action")
        novel: dict[str, Any] = raw_novel if isinstance(raw_novel, dict) else {}
        if mode == "compiled_action" and not str(data.get("compiled_action_id", "")).strip():
            raise GatewayError(
                f"actor {actor.actor_id!r} chose compiled_action without naming one; "
                "refusing to pick an action on its behalf"
            )
        if mode == "novel_action" and not str(novel.get("description", "")).strip():
            raise GatewayError(
                f"actor {actor.actor_id!r} proposed a novel action with no description"
            )
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

    def _apply_cognitive_updates(
        self,
        actor: ActorState,
        data: dict[str, Any],
        disposition: str,
        *,
        now: datetime,
        noticed_ids: frozenset[str],
        view: LocalView,
    ) -> ActorState:
        plan = _updated_plan(actor, data, disposition, now=now)
        needs = _updated_needs(actor, data, now=now, view=view)
        commitments = _updated_commitments(actor, data, now=now)
        revisits = _parse_revisits(data.get("revisit_when"))
        unresolved = _as_str_tuple(data.get("unresolved_questions")) or actor.unresolved_questions
        return replace(
            actor,
            plan=plan,
            pending_needs=needs,
            commitments=commitments,
            revisit_conditions=revisits if revisits else actor.revisit_conditions,
            unresolved_questions=unresolved,
            noticed_event_ids=noticed_ids,
            last_decision_time=now,
            decision_count=actor.decision_count + 1,
        )

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


# ---------------------------------------------------------------------------
# Persistent-state updates (pure functions over the actor's own response)
# ---------------------------------------------------------------------------


def _updated_plan(
    actor: ActorState, data: dict[str, Any], disposition: str, *, now: datetime
) -> Plan | None:
    """Apply the actor's own plan decision. The runtime never authors a plan; it only
    records what the actor said it is doing with the one it has."""

    plan = actor.plan
    raw = data.get("plan_update")
    update: dict[str, Any] = raw if isinstance(raw, dict) else {}

    if disposition == "continue" or disposition == "none":
        return plan
    if disposition == "complete" and plan is not None:
        return replace(plan, status=PLAN_COMPLETED, revised=now)
    if disposition == "abandon" and plan is not None:
        return replace(plan, status=PLAN_ABANDONED, revised=now)
    if disposition == "interrupt" and plan is not None:
        # The plan keeps its identity and its steps; it is paused, not destroyed, so it
        # can be resumed later and the trace can show the resumption.
        return replace(plan, status=PLAN_PAUSED, revised=now)

    steps = tuple(
        PlanStep(
            description=str(s.get("description", "")),
            intended_action_id=str(s.get("intended_action_id", "")),
            at=_parse_dt(s.get("at")),
            status=str(s.get("status", "planned")),
        )
        for s in _as_list(update.get("steps"))
        if isinstance(s, dict) and str(s.get("description", "")).strip()
    )
    goal = str(update.get("goal", "")).strip()
    basis = str(update.get("basis", "")).strip()
    interrupt_when = _as_str_tuple(update.get("interrupt_when"))
    evidence = _as_str_tuple(update.get("evidence_claim_ids"))

    if disposition == "replace" or plan is None:
        if not goal:
            return plan
        n = (plan.revision_count + 1) if plan else 0
        return Plan(
            plan_id=f"plan_{actor.actor_id}_{n}",
            goal=goal,
            steps=steps,
            status=PLAN_ACTIVE,
            created=now,
            revised=now,
            revision_count=0,
            interrupt_when=interrupt_when,
            basis=basis,
            evidence_claim_ids=evidence,
        )
    # revise: same plan identity, updated content.
    return replace(
        plan,
        goal=goal or plan.goal,
        steps=steps or plan.steps,
        status=PLAN_ACTIVE,
        revised=now,
        revision_count=plan.revision_count + 1,
        interrupt_when=interrupt_when or plan.interrupt_when,
        basis=basis or plan.basis,
        evidence_claim_ids=evidence or plan.evidence_claim_ids,
    )


def _updated_needs(
    actor: ActorState, data: dict[str, Any], *, now: datetime, view: LocalView
) -> tuple[PendingNeed, ...]:
    """Resolve needs an observation answered, and add newly stated ones.

    A need is resolved only when something actually arrived from the party it was asked
    of (or the actor says so) — never merely because the actor acted.
    """

    arrived_from = {o.source for o in view.observations}
    resolved_texts = {str(x) for x in _as_list(data.get("resolved_information_needs"))}
    out: list[PendingNeed] = []
    for need in actor.pending_needs:
        if need.resolved:
            out.append(need)
            continue
        answered = need.question in resolved_texts or (
            need.asked_of and need.asked_of in arrived_from
        )
        out.append(replace(need, resolved=True) if answered else need)

    for raw in _as_list(data.get("information_needs")):
        if isinstance(raw, str):
            text, of, deadline = raw.strip(), "", None
        elif isinstance(raw, dict):
            text = str(raw.get("question", raw.get("need", ""))).strip()
            of = str(raw.get("asked_of", ""))
            deadline = _parse_dt(raw.get("deadline"))
        else:
            continue
        if text and not any(n.question == text and not n.resolved for n in out):
            out.append(PendingNeed(question=text, asked_at=now, asked_of=of, deadline=deadline))
    return tuple(out)


def _updated_commitments(
    actor: ActorState, data: dict[str, Any], *, now: datetime
) -> tuple[ActorCommitment, ...]:
    discharged = {str(x) for x in _as_list(data.get("discharged_commitments"))}
    out = [replace(c, discharged=True) if c.text in discharged else c for c in actor.commitments]
    for raw in _as_list(data.get("new_commitments")):
        if isinstance(raw, str):
            text, due, to = raw.strip(), None, ""
        elif isinstance(raw, dict):
            text = str(raw.get("text", "")).strip()
            due = _parse_dt(raw.get("due"))
            to = str(raw.get("to_actor", ""))
        else:
            continue
        if text and not any(c.text == text for c in out):
            out.append(ActorCommitment(text=text, made_at=now, due=due, to_actor=to))
    return tuple(out)


def _parse_revisits(raw: Any) -> tuple[RevisitCondition, ...]:
    out: list[RevisitCondition] = []
    for item in _as_list(raw):
        if not isinstance(item, dict):
            continue
        cond = RevisitCondition(
            description=str(item.get("description", "")),
            at=_parse_dt(item.get("at")),
            on_field_change=str(item.get("on_field_change", "")),
            on_information_from=str(item.get("on_information_from", "")),
            on_record_in=str(item.get("on_record_in", "")),
        )
        if cond.is_checkable():
            out.append(cond)
    return tuple(out)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _render_grounding(profile: object) -> str:
    render = getattr(profile, "render_grounding", None)
    return str(render()) if callable(render) else ""


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


def _seed_importance(obs: Observation) -> float:
    """Initial retrieval salience of a newly noticed observation.

    This is a *retrieval ranking* input, not a model of how much a person cares. It is
    deliberately flat: content the actor explicitly received privately or that carries
    typed data outranks ambient noise, and nothing else is asserted. No importance
    score here decides whether the actor acts — the engine's trigger rules do, and they
    are structural, not numeric.
    """

    if obs.info_fields:
        return 1.0
    if obs.kind == "deliver_information" and obs.source != "environment":
        return 1.0
    return 0.5
