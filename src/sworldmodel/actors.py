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
from .ids import content_id
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

# Event kinds the ENVIRONMENT mints *about* an actor's own attempt: the world's verdict
# on what it tried. They carry the acting actor's id, but they are not that actor's
# doing — a refusal or a failure is news to the person refused, and it is the one thing
# an actor learns about its own action that it did not already know. Named here, next to
# the self-echo rule that exempts them, because the two must never drift apart;
# :mod:`sworldmodel.executor` mints them from these same constants.
ACTION_REJECTED = "action_rejected"
ACTION_FAILED = "action_failed"
WORLD_VERDICT_KINDS = frozenset({ACTION_REJECTED, ACTION_FAILED})


def is_self_echo(actor_id: str, event_actor_id: str | None, event_kind: str) -> bool:
    """Is this event the world reflecting an actor's own act back at that actor?

    An agent learning of its own action is not news. Left unchecked it is a feedback
    loop with no damping — act, the act creates an event, the event is information, the
    information is delivered, the agent wakes, acts again — and it is what consumed a
    whole branch of ``artifacts/ab/individual_semantic``: 43 of 51 wakes were
    ``directed_information``, twenty of them producing the identical signal, because the
    compiled effect addressed the event to its own participants (see
    ``semantic_lowering``: ``create["to"] = the event's participants``) and the actor was
    a participant in its own act.

    The boundary matters and it is drawn narrowly, at the *author* of the event:

    * An actor is NOT woken by an event it produced itself. That is the bare echo.
    * An actor IS woken by another agent's reaction to its action — the event's author is
      somebody else, so it is not an echo at all. That is the social dynamic this
      runtime exists to simulate and nothing here touches it.
    * An actor IS woken by the world's verdict on its own attempt
      (:data:`WORLD_VERDICT_KINDS`). A refusal or a failure carries the one fact the
      actor could not already know: that what it set out to do did not happen.

    Nothing about noticing changes. The event is still delivered, still noticed, still
    remembered — only the *wake* is withheld, because visibility, delivery, notice and
    reconsideration are four separate transitions and this is a statement about the last
    one alone.
    """

    if not actor_id or event_actor_id != actor_id:
        return False
    return event_kind not in WORLD_VERDICT_KINDS


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
class ActedRecord:
    """One thing this actor did, and how it turned out.

    An actor's own acts do not reach it through the information lifecycle, and they must
    not: visibility, delivery and notice describe how a person comes to know something
    that happened *elsewhere*. You do not need to be told what you did.

    That is precisely why the slot was empty. ``WorldState.view_for`` builds a view's
    observations from ``deliveries`` where ``actor_id`` is this actor, and an actor is
    never the recipient of its own delivery — ``observers_of`` skips it deliberately. So
    an agent's own acts were structurally invisible to it, and the only trace of any of
    them anywhere in its view was the single, overwritten ``current_action`` slot.

    Measured, before this existed: ``member_0`` circulated a note **fifteen times**. At
    its sixteenth invocation its view held eight observations, none of them mentioning
    ``member_0``, and six retrieved memories, none of them mentioning ``member_0``. It
    was not being stubborn and it was not looping on a stale view — the view changed
    constantly, full of the other four members' notes. **It could not see that it had
    already acted.** Fifteen identical acts and no record of one of them.

    This is the record. It is derived from the ledger the actor's own attempts already
    wrote — nothing new is invented, and nothing is remembered that did not happen.
    """

    at: datetime
    action_id: str  # "" for a wait
    outcome: str  # in_progress | completed | failed | refused | waited
    params: tuple[tuple[str, Any], ...] = ()
    target: str = ""
    detail: str = ""  # the world's own stated reason, when it gave one
    event_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "at": self.at.isoformat(),
            "you_did": self.action_id or "(waited)",
            "outcome": self.outcome,
        }
        if self.params:
            out["params"] = dict(self.params)
        if self.target:
            out["target"] = self.target
        if self.detail:
            out["detail"] = self.detail
        return out


def act_counts(acts: tuple[ActedRecord, ...]) -> dict[str, int]:
    """How many times this actor has taken each action, by outcome.

    Pure arithmetic over :class:`ActedRecord`, invented nowhere: it exists because
    "you have already done this fourteen times" is the fact an agent needs at a glance,
    and it must not have to count a long list to find it.
    """

    counts: dict[str, int] = {}
    for a in acts:
        key = f"{a.action_id or '(waited)'}:{a.outcome}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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
    # What this actor has already done in this branch, oldest first, with the world's
    # verdict on each attempt. NOT an observation and NOT delivered — see
    # :class:`ActedRecord` for why the two must not be confused, and for the fifteen
    # identical acts an agent took because it could not see any of them.
    own_actions: tuple[ActedRecord, ...] = ()

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
    # The situation this actor last decided from and what it decided — the material for
    # recognising a re-decision. See :func:`situation_key` and :func:`intent_key`.
    last_situation_key: str = ""
    last_intent_key: str = ""
    repeat_decisions: int = 0

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
            "repeat_decisions": self.repeat_decisions,
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


# ---------------------------------------------------------------------------
# Convergence: recognising a re-decision, and reporting where the calls went
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Reduce an arbitrary world value to something ``canonical_json`` can serialize.

    World fields carry whatever the compiled world put in them. A key that raised on an
    unusual value would turn a diagnostic into a branch-killing exception, so unknown
    types degrade to their ``repr`` rather than escaping.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return repr(value)


def external_information(actor_id: str, observations: tuple[Observation, ...]) -> list[str]:
    """What this actor has learned from **outside itself**, keyed on content.

    Two deliberate choices, both of which decide where the convergence line falls:

    * Observations the actor itself produced are excluded. Its own act echoing back is
      not something it learned (:func:`is_self_echo` states the same boundary for wakes).
    * The key is the message's *content* — who said it, of what kind, saying what, with
      which typed fields — and never its event id. Information is what it says, not
      which envelope carried it, so the same reminder arriving twice under fresh ids
      does not read as two things learned. The §9 wake-novelty measurement already keys
      novelty this way; keying on ids instead would let a nagging cascade re-qualify as
      new information on every repetition.
    """

    seen: set[str] = set()
    for obs in observations:
        if obs.source == actor_id:
            continue
        seen.add(
            content_id("info", obs.source, obs.kind, obs.summary, _jsonable(dict(obs.info_fields)))
        )
    return sorted(seen)


def situation_key(actor: ActorState, view: LocalView) -> str:
    """A stable digest of the *material* situation this actor is deciding from.

    "Material" is doing real work here. The raw :class:`LocalView` can never repeat: it
    carries fresh event ids on every wake, which is exactly why 51 invocations of one
    branch produced 51 distinct prompt hashes while the actor was demonstrably deciding
    the same thing twenty times. ``prompt_hash`` therefore cannot detect a repeat, and
    this key is what can.

    What is in it is everything that could legitimately make the same intent a *new*
    decision — the clock, the stage, the readable world, the menu the world is offering,
    what reached the actor from anybody else, and the actor's own mind (its plan and how
    many times it has revised it, what it is carrying out, what it has undertaken, what
    it is waiting on, what it believes and wants). What is deliberately NOT in it is
    bookkeeping that turns over whether or not anything happened: event ids, the world's
    version counter (which the actor's own deliveries increment), and anything the actor
    itself produced — including its own act history.

    **Excluding the act history is the point, not an oversight.** ``own_actions`` grows
    by one every time the actor acts, so a key that included it could never match twice
    and the detector could never fire — which is exactly what a whole-``LocalView``
    comparison suffers from. The narrower question this key asks is: *has anything
    happened in the world, or reached this actor from anyone else, that could make the
    same act a new decision?* An agent that can see it has already circulated fourteen
    notes and circulates a fifteenth has decided something; this key does not call that
    a repeat, and nothing here suppresses it.

    **The clock is in the key, which makes the detector deliberately conservative.**
    Elapsed simulated time always counts as something having moved, so two identical acts
    separated by any interval are two decisions — a governor signalling support in June
    and again in September is never touched, because the world has aged around the second
    one and that ageing is itself information. Only a re-ask at the very same instant can
    match. This under-fires by construction, and that is the direction to err in: leaving
    some spin is recoverable, deleting a real decision is not.

    Two decisions with the same key were taken at the same instant, for the same stated
    cause, from a world that had not moved, by an actor that had not changed its mind,
    with nothing having reached it from anybody else in between.
    """

    plan = actor.plan
    ongoing = actor.current_action
    return content_id(
        "sit",
        view.actor_id,
        view.branch_time.isoformat(),
        view.stage,
        view.role,
        sorted(view.authority),
        # WHY the actor was called. A commitment falling due at the same instant as an
        # opportunity is a different cause, and a different cause is a real reason to
        # ask again even when nothing else has moved.
        view.trigger_kind,
        sorted(view.public_facts),
        _jsonable(view.observed_fields()),
        sorted(str(a.get("action_id", "")) for a in view.feasible_actions),
        bool(view.allow_novel),
        external_information(view.actor_id, view.observations),
        (
            [plan.plan_id, plan.goal, plan.status, plan.revision_count]
            + [s.description for s in plan.steps]
            if plan is not None
            else None
        ),
        [ongoing.action_id, ongoing.status] if ongoing is not None else None,
        sorted(c.text for c in actor.open_commitments()),
        sorted(n.question for n in actor.open_needs()),
        sorted(actor.beliefs),
        sorted(actor.goals),
        sorted(actor.unresolved_questions),
    )


def intent_key(choice: ActionChoice) -> str:
    """A stable digest of *what an actor decided to do* — the act, not the account of it.

    The rationale is excluded on purpose. Two invocations that take the identical action
    with differently-worded reasoning are the same decision, and keying on the prose
    would let a loop escape detection simply by varying how it explains itself.
    """

    return content_id(
        "int",
        choice.mode,
        choice.action_id,
        _jsonable(dict(choice.params)),
        choice.target,
        choice.novel_description.strip(),
        choice.novel_intended_effect.strip(),
        choice.novel_target,
        _jsonable(dict(choice.novel_params)),
    )


@dataclass(frozen=True)
class NonDecision:
    """A wake that cannot produce a decision, recognised *before* the model is called.

    Nothing about the actor's material situation differs from the one it last decided
    from: same instant, same cause, same readable world, same menu, same mind, and
    nothing has reached it from anybody else since. Asking again is not asking a person
    to reconsider; it is asking the same question twice at one timestamp, which the
    runtime already treats as a defect elsewhere (``Schedule.drop_matching`` and
    ``_merge_decisions`` both exist to stop exactly that).
    """

    situation_key: str
    prior_intent_key: str
    reason: str


def check_non_decision(actor: ActorState, view: LocalView) -> NonDecision | None:
    """Recognise a wake with no material cause, or return ``None``.

    **A genuine repetition always survives.** A central banker really can signal support
    twice and a negotiator really can repeat a demand; the line is not how many times an
    actor does something, it is whether anything happened in between. Because the clock,
    the readable world, the feasible menu, everything that reached the actor from anyone
    else, and the actor's own plan and beliefs are all in :func:`situation_key`, *any* of
    them moving makes the next identical act a new decision — however many times it has
    already been taken. Only a re-ask from a world that has not moved at all, at the very
    same instant, for the very same reason, is refused.
    """

    if not actor.last_intent_key:
        return None
    key = situation_key(actor, view)
    if key != actor.last_situation_key:
        return None
    return NonDecision(
        situation_key=key,
        prior_intent_key=actor.last_intent_key,
        reason=(
            f"nothing material has changed for {actor.actor_id!r} since it decided at "
            f"{view.branch_time.isoformat()}: same trigger ({view.trigger_kind or 'none'}), "
            "same readable world, same feasible actions, same plan and commitments, and "
            "nothing has reached it from anybody else — so this wake has no cause the "
            "actor has not already answered"
        ),
    )


@dataclass
class ConvergenceDiagnostics:
    """Where a branch's actor calls went, and which of them were the loop.

    Without this a run that spins reads exactly like a run that deliberates — the OPEC+
    branch and the reference multiparty world both just show "N invocations" — and no
    artifact tells them apart. Every number here is a count of a mechanism that actually
    fired, per actor, so a reader can see which participant was looping and what it cost.
    """

    self_echo_wakes_suppressed: int = 0
    non_decision_wakes_refused: int = 0
    repeat_decisions: int = 0
    by_actor: dict[str, dict[str, int]] = field(default_factory=dict)

    def _bump(self, actor_id: str, counter: str) -> None:
        row = self.by_actor.setdefault(
            actor_id,
            {
                "self_echo_wakes_suppressed": 0,
                "non_decision_wakes_refused": 0,
                "repeat_decisions": 0,
            },
        )
        row[counter] += 1

    def record_self_echo(self, actor_id: str) -> None:
        """A wake that would have fired *only* on what this actor produced itself."""

        self.self_echo_wakes_suppressed += 1
        self._bump(actor_id, "self_echo_wakes_suppressed")

    def record_non_decision(self, actor_id: str) -> None:
        """A wake refused before any model call, its situation unchanged since the last."""

        self.non_decision_wakes_refused += 1
        self._bump(actor_id, "non_decision_wakes_refused")

    def record_repeat(self, actor_id: str) -> None:
        """A call that was made and re-produced the previous decision unchanged."""

        self.repeat_decisions += 1
        self._bump(actor_id, "repeat_decisions")

    def summary(self) -> str:
        """One human-readable line, carried on the stop reason when a budget runs out.

        Exhaustion has to be loud about *why* the calls went: 80 calls spent deliberating
        and 80 spent re-deciding one thing are the same number and completely different
        runs.
        """

        return (
            f"{self.repeat_decisions} of the calls made were repeat decisions; "
            f"{self.self_echo_wakes_suppressed} wake(s) were withheld as self-echo and "
            f"{self.non_decision_wakes_refused} refused as non-decisions"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "self_echo_wakes_suppressed": self.self_echo_wakes_suppressed,
            "non_decision_wakes_refused": self.non_decision_wakes_refused,
            "repeat_decisions": self.repeat_decisions,
            "by_actor": {k: dict(v) for k, v in sorted(self.by_actor.items())},
        }


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
    # The situation this decision was taken from, what it decided, and whether that pair
    # exactly repeats the actor's previous decision.
    situation_key: str = ""
    intent_key: str = ""
    is_repeat: bool = False


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

        # Was this a decision, or the same decision again? Detected exactly, from the
        # situation the actor decided from and what it decided — never from the prompt
        # hash, which differs on every invocation because the view carries fresh event
        # ids. A repeat is recorded and its action still runs: the runtime does not
        # cancel an intention it disagrees with, and a monotone terminal that the repeat
        # satisfies must still be satisfied. What a repeat loses is momentum — it wakes
        # nobody by itself (see :func:`is_self_echo`), and the next wake from a situation
        # that still has not moved is refused (see :func:`check_non_decision`).
        sit_key = situation_key(actor, view)
        int_key = intent_key(choice)
        is_repeat = bool(
            actor.last_intent_key
            and sit_key == actor.last_situation_key
            and int_key == actor.last_intent_key
        )

        new_actor = self._apply_cognitive_updates(
            actor,
            dresp.data,
            disposition,
            now=now,
            noticed_ids=noticed_ids,
            view=view,
        )
        new_actor = replace(
            new_actor,
            last_situation_key=sit_key,
            last_intent_key=int_key,
            repeat_decisions=actor.repeat_decisions + (1 if is_repeat else 0),
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
        # Added AFTER the prompt was rendered, so the convergence record reaches the
        # trace (``local_view.convergence`` in actor_decisions.jsonl) without ever
        # reaching the actor: a person is not told they are repeating themselves.
        context["convergence"] = {
            "situation_key": sit_key,
            "intent_key": int_key,
            "repeat_of_previous_decision": is_repeat,
            "repeat_decisions_so_far": new_actor.repeat_decisions,
            "self_produced_observations_noticed": sum(
                1 for o in new_obs if o.source == actor.actor_id
            ),
        }
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
            situation_key=sit_key,
            intent_key=int_key,
            is_repeat=is_repeat,
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
            # What this actor has already done, and how often. Without these two the
            # agent is an amnesiac: its own acts reach it through no channel at all, and
            # `current_action` is one overwritten cell that says nothing about the
            # fourteen attempts before it.
            "your_actions_so_far": [a.as_dict() for a in view.own_actions],
            "your_action_counts": act_counts(view.own_actions),
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
