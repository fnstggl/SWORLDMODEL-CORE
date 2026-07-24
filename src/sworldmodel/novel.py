"""The novel-action route.

Compiled action menus are guidance, not a closed list. On any turn an actor may
propose a *novel* action the compiler did not anticipate. A novel action never
executes directly: the actor states an intention and the external world decides the
consequence, through this pipeline::

    novel intention
      -> semantic interpretation   (map free description -> universal effect ops)
      -> authority validation      (does this actor hold the required capability?)
      -> feasibility validation    (do the ops apply safely? resources/targets?)
      -> resource / timing validation
      -> translation into safe world operations
      -> execute  OR  explicit rejection

If the proposal cannot be represented with the safe universal operations, it is
rejected (or the branch left unresolved) — it is **never** silently converted into
the nearest known action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actors import ActorState
from .effects import UNIVERSAL_OPS, EffectExecutor
from .gateway import GatewayRequest, ModelGateway
from .prompts import render_novel_interpret_prompt
from .world import WorldState
from .worldspec import ActionChoice, Effect, WorldSpec


@dataclass(frozen=True)
class NovelResolution:
    executed: bool
    reason: str
    effects: tuple[Effect, ...] = ()
    required_authority: tuple[str, ...] = ()


def resolve_novel(
    *,
    actor: ActorState,
    choice: ActionChoice,
    world: WorldState,
    spec: WorldSpec,
    gateway: ModelGateway,
    executor: EffectExecutor,
    seed: int,
) -> tuple[NovelResolution, list[Any]]:
    """Interpret and validate a novel action. Returns the resolution and the events to
    apply (empty if rejected). Never mutates the world."""

    # 1. semantic interpretation: free intention -> candidate universal effects.
    interp_ctx: dict[str, Any] = {
        "actor_id": actor.actor_id,
        "authority": list(actor.authority),
        "description": choice.novel_description,
        "target": choice.novel_target,
        "intended_effect": choice.novel_intended_effect,
        "parameters": choice.novel_params_dict,
        "rationale": choice.rationale,
        "universal_ops": sorted(UNIVERSAL_OPS),
        "world_fields": [f for f, _ in world.fields] or [f.field_id for f in spec.fields],
        "entities": [e.entity_id for e in spec.entities],
        "resources": [r.resource_id for r in spec.resources],
    }
    resp = gateway.generate(
        GatewayRequest(
            task_kind="interpret_novel",
            prompt=render_novel_interpret_prompt(interp_ctx),
            context=interp_ctx,
            seed=seed,
        )
    )
    data = resp.data
    if not data.get("representable"):
        return (
            NovelResolution(False, f"unrepresentable: {data.get('reason', 'no safe mapping')}"),
            [resp],
        )

    effects = _parse_effects(data.get("effects"))
    if not effects:
        return NovelResolution(False, "interpreter produced no safe effects"), [resp]
    bad = [e.op for e in effects if e.op not in UNIVERSAL_OPS]
    if bad:
        return NovelResolution(False, f"non-universal ops proposed: {bad}"), [resp]

    required = tuple(str(a) for a in (data.get("required_authority") or []))

    # 2. authority validation.
    missing = [a for a in required if a not in actor.authority]
    if missing:
        return (
            NovelResolution(False, f"actor lacks authority {missing}", effects, required),
            [resp],
        )

    # 3-4. feasibility / resource / timing validation.
    binding = {
        "actor": actor.actor_id,
        "self": actor.entity,
        "params": choice.novel_params_dict,
        "target": choice.novel_target,
    }
    ok, reason = executor.can_apply(world, effects, binding)
    if not ok:
        return NovelResolution(False, f"infeasible: {reason}", effects, required), [resp]

    # 5. translation into safe world operations -> execute.
    events = executor.build_events(world, effects, binding)
    return NovelResolution(True, "novel action authorized and executed", effects, required), [
        resp,
        *events,
    ]


def _parse_effects(raw: Any) -> tuple[Effect, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[Effect] = []
    for item in raw:
        if not isinstance(item, dict) or "op" not in item:
            continue
        params = {k: v for k, v in item.items() if k != "op"}
        out.append(Effect(op=str(item["op"]), params=tuple(sorted(params.items()))))
    return tuple(out)
