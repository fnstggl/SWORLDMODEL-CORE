"""Action validation and execution — the environment is authoritative.

Actors emit an :class:`~sworldmodel.worldspec.ActionChoice`; this module turns it into
consequences (events) or an explicit rejection. It is domain-free: it knows only how
to check a compiled :class:`~sworldmodel.worldspec.ActionDefinition` against the world
(eligibility, authority, stage/timing, targets, preconditions, resources) and to apply
its universal effects. A compiled action's *behavior is its effects*, so renaming the
action changes nothing. Novel actions are delegated to :mod:`novel`.

The same feasibility logic that gates execution also builds the action menu offered to
each actor, so an actor is only ever shown actions currently feasible in its local
world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .actors import ActorState
from .effects import EffectExecutor
from .expressions import evaluate
from .gateway import ModelGateway
from .models import Event, Visibility
from .novel import resolve_novel
from .world import WorldState
from .worldspec import ActionChoice, ActionDefinition, Effect, ProcessNode, WorldSpec


@dataclass
class TurnOutcome:
    events: list[Event]
    status: str  # "executed" | "rejected" | "wait"
    reason: str
    mode: str
    action_id: str = ""
    gateway_responses: list[Any] = field(default_factory=list)


class ActionExecutor:
    def __init__(self, gateway: ModelGateway, effects: EffectExecutor) -> None:
        self.gateway = gateway
        self.effects = effects

    # -- feasibility (also used to build the actor's menu) ----------------------

    def feasible_actions(
        self, world: WorldState, node: ProcessNode, actor: ActorState, spec: WorldSpec
    ) -> list[ActionDefinition]:
        offered = node.action_ids
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
            return False, f"action not available in stage {world.stage!r}"
        if not _timing_ok(world.time, action.not_before, action.not_after):
            return False, "action is outside its permitted time window"
        for res, cost in action.resource_costs:
            if world.get_resource(res, actor.actor_id) < float(cost):
                return False, f"actor lacks resource {res!r}"
        if not bool(evaluate(action.preconditions, world)):
            return False, "precondition not satisfied"
        return True, "ok"

    def action_card(self, action: ActionDefinition) -> dict[str, Any]:
        return {
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

    # -- execution --------------------------------------------------------------

    def execute(
        self, actor: ActorState, choice: ActionChoice, world: WorldState, spec: WorldSpec, seed: int
    ) -> TurnOutcome:
        if choice.mode == "novel_action":
            return self._execute_novel(actor, choice, world, spec, seed)
        if choice.mode == "compiled_action":
            return self._execute_compiled(actor, choice, world, spec)
        return TurnOutcome(
            events=[self._note(world, actor, "wait", {"rationale": choice.rationale})],
            status="wait",
            reason="actor waited",
            mode="wait",
        )

    def _execute_compiled(
        self, actor: ActorState, choice: ActionChoice, world: WorldState, spec: WorldSpec
    ) -> TurnOutcome:
        action = spec.action(choice.action_id)
        if action is None:
            return self._reject(
                world, actor, choice.action_id, f"no such action {choice.action_id!r}"
            )

        ok, reason = self._check_availability(world, actor, action)
        if not ok:
            return self._reject(world, actor, action.action_id, reason)

        if action.valid_targets and not _target_ok(
            choice.target, action.valid_targets, actor, world
        ):
            return self._reject(world, actor, action.action_id, f"invalid target {choice.target!r}")

        params = _fill_params(action, choice.params_dict)
        bad_param = _validate_params(action, params)
        if bad_param:
            return self._reject(world, actor, action.action_id, bad_param)

        binding = {
            "actor": actor.actor_id,
            "self": actor.entity,
            "params": params,
            "target": choice.target,
        }
        effects = _cost_effects(action, actor.actor_id) + action.effects
        feasible, why = self.effects.can_apply(world, effects, binding)
        if not feasible:
            return self._reject(world, actor, action.action_id, why)

        events = self.effects.build_events(world, effects, binding)
        return TurnOutcome(
            events=events,
            status="executed",
            reason="compiled action executed",
            mode="compiled_action",
            action_id=action.action_id,
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
                    "action_rejected",
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

    def _reject(
        self, world: WorldState, actor: ActorState, action_id: str, reason: str
    ) -> TurnOutcome:
        ev = self._note(
            world,
            actor,
            "action_rejected",
            {"mode": "compiled_action", "action_id": action_id, "reason": reason},
        )
        return TurnOutcome(
            events=[ev],
            status="rejected",
            reason=reason,
            mode="compiled_action",
            action_id=action_id,
        )

    def _note(
        self, world: WorldState, actor: ActorState, kind: str, payload: dict[str, Any]
    ) -> Event:
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


def _timing_ok(now: datetime, not_before: Any, not_after: Any) -> bool:
    if isinstance(not_before, str) and not_before:
        try:
            if now < datetime.fromisoformat(not_before):
                return False
        except ValueError:
            pass
    if isinstance(not_after, str) and not_after:
        try:
            if now > datetime.fromisoformat(not_after):
                return False
        except ValueError:
            pass
    return True


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


def _fill_params(action: ActionDefinition, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for p in action.parameters:
        if p.name not in out and p.choices:
            out[p.name] = p.choices[0]  # default to the first declared choice
    return out


def _validate_params(action: ActionDefinition, params: dict[str, Any]) -> str:
    for p in action.parameters:
        if p.required and p.name not in params:
            return f"missing required parameter {p.name!r}"
        if p.choices and p.name in params and params[p.name] not in p.choices:
            return f"parameter {p.name!r}={params[p.name]!r} not in {list(p.choices)}"
    return ""


def _cost_effects(action: ActionDefinition, actor_id: str) -> tuple[Effect, ...]:
    return tuple(
        Effect(
            op="consume_resource",
            params=(("amount", float(cost)), ("holder", actor_id), ("resource", res)),
        )
        for res, cost in action.resource_costs
    )
