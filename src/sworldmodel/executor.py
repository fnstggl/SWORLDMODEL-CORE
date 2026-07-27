"""Action validation and execution — the environment is authoritative.

Actors emit an :class:`~sworldmodel.worldspec.ActionChoice`; this module decides what
actually happens. It is domain-free: it knows only how to check a compiled
:class:`~sworldmodel.worldspec.ActionDefinition` against the world (eligibility,
authority, stage, timing, targets, preconditions, resources, parameters) and to apply
its universal effects. A compiled action's *behavior is its effects*, so renaming the
action changes nothing.

Two rules govern everything here.

**Nothing is coerced.** An intention that fails validation is *rejected*, with the
exact reason written to the event ledger and returned to the actor. It is never turned
into a different action, never downgraded to waiting, never completed with a parameter
the runtime chose on the actor's behalf. Deciding what to do instead is the actor's
job; deciding what is possible is this module's.

**Acting is not succeeding.** An accepted action *starts*; its effects land at
completion, after ``duration_seconds`` of real time, and only if its
``completion_conditions`` and its feasibility still hold in the world as it is *then*.
An action begun against a world that has since moved fails visibly rather than being
applied to a world its actor never saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .actors import ACTION_FAILED, ACTION_REJECTED, ActorState, OngoingAction
from .effects import EffectExecutor
from .errors import UndeterminedExpressionError
from .expressions import evaluate
from .gateway import ModelGateway
from .models import Event, Visibility
from .novel import resolve_novel
from .schedule import ORIGIN_CONSEQUENCE, ScheduledEntry, make_entry
from .world import WorldState
from .worldspec import ActionChoice, ActionDefinition, Effect, ProcessNode, WorldSpec

_PARAM_REF = re.compile(r"\$param\.([A-Za-z0-9_]+)")

# Structural entry kinds this module produces / consumes.
KIND_ACTION_COMPLETION = "action_completion"


@dataclass
class TurnOutcome:
    """What one actor invocation produced.

    ``status`` is ``started`` (accepted, in flight), ``executed`` (accepted and already
    complete), ``rejected`` (refused, with a reason the actor receives), or ``wait``
    (the actor genuinely chose to wait — never a substitute for a failure).
    """

    events: list[Event]
    status: str
    reason: str
    mode: str
    action_id: str = ""
    scheduled: list[ScheduledEntry] = field(default_factory=list)
    # Effects this action stamped in the future: they are scheduled, not applied.
    deferred: list[tuple[datetime, Effect]] = field(default_factory=list)
    ongoing: OngoingAction | None = None
    gateway_responses: list[Any] = field(default_factory=list)


class ActionExecutor:
    def __init__(self, gateway: ModelGateway, effects: EffectExecutor) -> None:
        self.gateway = gateway
        self.effects = effects

    # -- feasibility (also used to build the actor's menu) ----------------------

    def feasible_actions(
        self, world: WorldState, node: ProcessNode | None, actor: ActorState, spec: WorldSpec
    ) -> list[ActionDefinition]:
        offered = node.action_ids if node is not None else ("*",)
        out: list[ActionDefinition] = []
        for action in spec.actions:
            if offered != ("*",) and action.action_id not in offered:
                continue
            if self.is_feasible(world, actor, action):
                out.append(action)
        return out

    def is_feasible(self, world: WorldState, actor: ActorState, action: ActionDefinition) -> bool:
        ok, _ = self._check_availability(world, actor, action)
        return ok

    def _check_availability(
        self, world: WorldState, actor: ActorState, action: ActionDefinition
    ) -> tuple[bool, str]:
        if not action.eligible(actor.role, actor.actor_id):
            return False, "actor not eligible for this action"
        missing = [a for a in action.required_authority if a not in actor.authority]
        if missing:
            return False, f"actor lacks authority {missing}"
        if action.stages and world.stage not in action.stages:
            return False, (
                f"action {action.action_id!r} is not available in stage {world.stage!r} "
                f"(permitted stages: {list(action.stages)})"
            )
        timing_ok, timing_reason = _timing_ok(world.time, action.not_before, action.not_after)
        if not timing_ok:
            return False, timing_reason
        for res, cost in action.resource_costs:
            if world.get_resource(res, actor.actor_id) < float(cost):
                return False, f"actor lacks resource {res!r}"
        try:
            satisfied = bool(evaluate(action.preconditions, world))
        except UndeterminedExpressionError as exc:
            # A precondition that reads a value the world never determined is not
            # satisfied — the action is simply unavailable right now. Letting the
            # exception escape would abort the whole branch over one unavailable action.
            return False, f"precondition undetermined: {exc}"
        if not satisfied:
            return False, "precondition not satisfied"
        return True, "ok"

    def action_card(self, action: ActionDefinition) -> dict[str, Any]:
        card: dict[str, Any] = {
            "action_id": action.action_id,
            "meaning": action.meaning,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.value_type,
                    "required": p.required,
                    "choices": list(p.choices),
                }
                for p in action.parameters
            ],
            "valid_targets": list(action.valid_targets),
        }
        if action.duration_seconds:
            card["takes_seconds"] = action.duration_seconds
        return card

    # -- execution: an intention becomes an attempt ------------------------------

    def execute(
        self,
        actor: ActorState,
        choice: ActionChoice,
        world: WorldState,
        spec: WorldSpec,
        seed: int,
        *,
        microstep: int = 0,
    ) -> TurnOutcome:
        """``microstep`` is the causal layer of the decision that produced this
        intention. Anything the action schedules is layered after it, so a zero-duration
        action completing at the same timestamp still completes *after* the decision to
        take it rather than sorting ahead of it."""

        if choice.mode == "novel_action":
            return self._execute_novel(actor, choice, world, spec, seed)
        if choice.mode == "compiled_action":
            return self._begin_compiled(actor, choice, world, spec, microstep=microstep)
        return TurnOutcome(
            events=[self._note(world, actor, "actor_waited", {"rationale": choice.rationale})],
            status="wait",
            reason="actor chose to wait",
            mode="wait",
        )

    def _begin_compiled(
        self,
        actor: ActorState,
        choice: ActionChoice,
        world: WorldState,
        spec: WorldSpec,
        *,
        microstep: int = 0,
    ) -> TurnOutcome:
        action = spec.action(choice.action_id)
        if action is None:
            return self._reject(
                world,
                actor,
                choice.action_id,
                f"no such action {choice.action_id!r} in this world",
            )

        ok, reason = self._check_availability(world, actor, action)
        if not ok:
            return self._reject(world, actor, action.action_id, reason)

        if action.valid_targets and not _target_ok(
            choice.target, action.valid_targets, actor, world
        ):
            return self._reject(world, actor, action.action_id, f"invalid target {choice.target!r}")

        params = dict(choice.params_dict)
        bad_param = _validate_params(action, params)
        if bad_param:
            return self._reject(world, actor, action.action_id, bad_param)

        # Dry-run the effects so an action that cannot land is refused before it starts.
        binding = _binding(actor, params, choice.target)
        effects = _cost_effects(action, actor.actor_id) + action.effects
        feasible, why = self.effects.can_apply(world, effects, binding)
        if not feasible:
            return self._reject(world, actor, action.action_id, why)

        started_at = world.time
        completes_at = started_at + timedelta(seconds=max(0, action.duration_seconds))
        note = self._note(
            world,
            actor,
            "action_started",
            {
                "action_id": action.action_id,
                "params": params,
                "target": choice.target,
                "completes_at": completes_at.isoformat(),
            },
        )
        entry = make_entry(
            at=completes_at,
            kind=KIND_ACTION_COMPLETION,
            actor_id=actor.actor_id,
            payload={
                "action_id": action.action_id,
                "params": params,
                "target": choice.target,
                "world_version": world.version,
                "started_at": started_at.isoformat(),
            },
            origin=ORIGIN_CONSEQUENCE,
            origin_detail=f"action:{action.action_id}",
            causal_parents=(note.event_id,),
            microstep=microstep + 1,
        )
        ongoing = OngoingAction(
            action_id=action.action_id,
            description=action.meaning or action.action_id,
            started=started_at,
            expected_completion=completes_at,
            world_version=world.version,
        )
        return TurnOutcome(
            events=[note],
            status="started",
            reason=f"action {action.action_id!r} begun; completes {completes_at.isoformat()}",
            mode="compiled_action",
            action_id=action.action_id,
            scheduled=[entry],
            ongoing=ongoing,
        )

    def complete(self, world: WorldState, entry: ScheduledEntry, spec: WorldSpec) -> TurnOutcome:
        """Finish an in-flight action, or fail it honestly.

        This is where "I did it" becomes "it happened" — and where it may not. The
        action is re-validated against the world *now*, not the world its actor saw.
        """

        p = entry.payload_dict
        actor = world.actors.get(str(entry.actor_id))
        action = spec.action(str(p.get("action_id", "")))
        if actor is None or action is None:
            return TurnOutcome(
                events=[], status="rejected", reason="actor or action no longer exists", mode=""
            )

        params = dict(p.get("params") or {})
        target = str(p.get("target", ""))
        binding = _binding(actor, params, target)

        ok, reason = self._check_availability(world, actor, action)
        if not ok:
            return self._fail(world, actor, action.action_id, f"no longer possible: {reason}")

        decided_version = int(p.get("world_version", world.version))
        if not bool(evaluate(action.completion_conditions, world)):
            return self._fail(
                world,
                actor,
                action.action_id,
                "completion condition no longer holds in the world at completion time "
                f"(intention formed at world version {decided_version}, now {world.version})",
            )

        effects = _cost_effects(action, actor.actor_id) + action.effects
        feasible, why = self.effects.can_apply(world, effects, binding)
        if not feasible:
            return self._fail(world, actor, action.action_id, f"no longer feasible: {why}")

        events, deferred = self.effects.build_events(world, effects, binding)
        return TurnOutcome(
            events=events,
            status="executed",
            reason=f"action {action.action_id!r} completed",
            mode="compiled_action",
            action_id=action.action_id,
            deferred=deferred,
        )

    def _execute_novel(
        self, actor: ActorState, choice: ActionChoice, world: WorldState, spec: WorldSpec, seed: int
    ) -> TurnOutcome:
        resolution, produced = resolve_novel(
            actor=actor,
            choice=choice,
            world=world,
            spec=spec,
            gateway=self.gateway,
            executor=self.effects,
            seed=seed,
        )
        gateway_responses = [x for x in produced if not isinstance(x, Event)]
        events = [x for x in produced if isinstance(x, Event)]
        if not resolution.executed:
            events.append(
                self._note(
                    world,
                    actor,
                    ACTION_REJECTED,
                    {
                        "mode": "novel_action",
                        "reason": resolution.reason,
                        "description": choice.novel_description,
                    },
                )
            )
            return TurnOutcome(
                events=events,
                status="rejected",
                reason=resolution.reason,
                mode="novel_action",
                gateway_responses=gateway_responses,
            )
        return TurnOutcome(
            events=events,
            status="executed",
            reason="novel action authorized and executed",
            mode="novel_action",
            gateway_responses=gateway_responses,
        )

    # -- refusals ---------------------------------------------------------------
    #
    # A refusal and a failure are the two things the environment says *about* an actor's
    # own attempt, and they are the only events carrying an actor's id that are genuinely
    # news to that actor: it did not know its intention would be turned down, or that the
    # world would move out from under an action already begun. They are minted from
    # ``actors.ACTION_REJECTED``/``ACTION_FAILED``, which is the same pair
    # ``actors.is_self_echo`` exempts from self-echo suppression — one definition, so the
    # verdict can never quietly become the echo the wake filter drops.

    def _reject(
        self, world: WorldState, actor: ActorState, action_id: str, reason: str
    ) -> TurnOutcome:
        ev = self._note(
            world,
            actor,
            ACTION_REJECTED,
            {"mode": "compiled_action", "action_id": action_id, "reason": reason},
        )
        return TurnOutcome(
            events=[ev],
            status="rejected",
            reason=reason,
            mode="compiled_action",
            action_id=action_id,
        )

    def _fail(
        self, world: WorldState, actor: ActorState, action_id: str, reason: str
    ) -> TurnOutcome:
        ev = self._note(world, actor, ACTION_FAILED, {"action_id": action_id, "reason": reason})
        return TurnOutcome(
            events=[ev], status="failed", reason=reason, mode="compiled_action", action_id=action_id
        )

    def _note(
        self, world: WorldState, actor: ActorState, kind: str, payload: dict[str, Any]
    ) -> Event:
        """A private record for the acting actor: what it attempted and what the world
        decided. It is how a refusal gets back to the person who was refused."""

        return self.effects.raw_event(
            world,
            kind=kind,
            actor_id=actor.actor_id,
            payload=payload,
            visibility=Visibility.PRIVATE,
            audience=(actor.actor_id,),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binding(actor: ActorState, params: dict[str, Any], target: str) -> dict[str, Any]:
    return {
        "actor": actor.actor_id,
        "self": actor.entity,
        "params": params,
        "target": target,
    }


def _timing_ok(now: datetime, not_before: Any, not_after: Any) -> tuple[bool, str]:
    """Check an action's time window.

    An unreadable bound is not an absent bound. Swallowing the parse error turned a gate
    the compiler wrote into a gate that always passes — the action becomes available at
    every moment, which is the opposite of what was compiled.
    """

    for label, raw, too_early in (
        ("not_before", not_before, True),
        ("not_after", not_after, False),
    ):
        if not isinstance(raw, str) or not raw:
            continue
        try:
            bound = datetime.fromisoformat(raw)
        except ValueError:
            return False, f"action has an unreadable {label} bound {raw!r}"
        if (now < bound) if too_early else (now > bound):
            return False, f"action is outside its permitted time window ({label}={raw})"
    return True, "ok"


def _target_ok(
    target: str, valid_targets: tuple[str, ...], actor: ActorState, world: WorldState
) -> bool:
    if "*" in valid_targets:
        return bool(target)
    if target in valid_targets:
        return True
    ent = next((e for e in world.entities if e.entity_id == target), None)
    if ent is not None:
        for sel in valid_targets:
            if sel.startswith("role:") and sel[5:] == ent.role:
                return True
    return False


def _validate_params(action: ActionDefinition, params: dict[str, Any]) -> str:
    """Check the actor supplied what the action needs.

    A missing parameter is *never* filled in. Choosing "the first declared option" on
    an actor's behalf would be the runtime casting its vote, picking its offer, or
    selecting its answer — the single most consequential form of fabricated behavior
    available to a simulator.
    """

    for p in action.parameters:
        if p.required and p.name not in params:
            return f"missing required parameter {p.name!r}"
        if p.choices and p.name in params and params[p.name] not in p.choices:
            return (
                f"parameter {p.name!r}={params[p.name]!r} is not one of {list(p.choices)}; "
                "state one of the listed values"
            )
    referenced = {
        name
        for eff in action.effects
        for value in eff.params_dict.values()
        for name in _PARAM_REF.findall(str(value))
    }
    unsupplied = sorted(n for n in referenced if n not in params)
    if unsupplied:
        return (
            f"action {action.action_id!r} needs parameter(s) {unsupplied} to have any effect; "
            "supply them explicitly"
        )
    return ""


def _cost_effects(action: ActionDefinition, actor_id: str) -> tuple[Effect, ...]:
    return tuple(
        Effect(
            op="consume_resource",
            params=(("amount", float(cost)), ("holder", actor_id), ("resource", res)),
        )
        for res, cost in action.resource_costs
    )
