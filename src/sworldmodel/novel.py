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
from .worldspec import (
    ActionChoice,
    ActionDefinition,
    Effect,
    WorldSpec,
    display_term,
    effect_terms,
    expression_terms,
)

# What the compiled world says about who may produce one piece of world state. Every
# term an effect writes lands in exactly one of these, and each is a *distinct* answer:
#
#   ACTOR        compiled actions produce it, so somebody has standing — the question is
#                whether it is this actor.
#   ENVIRONMENT  only process nodes / external-process occurrences produce it. Nobody has
#                standing, because it is not the kind of thing an actor does at all.
#   UNPRODUCED   nothing in this compiled world produces it.
#
# They are named rather than left implicit because the defect this closes was exactly a
# missing distinction: the old check computed a gatekeeper list and wrote
# ``if not gatekeepers: continue``, so "the world says only the weather does this" and
# "the world has never heard of this" and "I did not manage to look" were one silence,
# and that silence read as *permitted*. It is the shape of FD-34 (a provider outage
# laundering mechanical facts), FD-42 (blank records evaluating as an answer) and FD-45
# (a crashed check reading as an approved one): a "found nothing" that cannot be told
# apart from a "did not look". An empty producer set means *nobody has standing*, never
# *nobody objects*.
STANDING_ACTOR = "actor"
STANDING_ENVIRONMENT = "environment"
STANDING_UNPRODUCED = "unproduced"


@dataclass(frozen=True)
class TermStanding:
    """Who, in the compiled world, may write one term — and whether this actor is one."""

    term: str
    standing: str
    actor_producers: tuple[str, ...] = ()
    environment_producers: tuple[str, ...] = ()
    # The compiled action giving this actor a legitimate route to the term, if any.
    satisfied_by: str = ""
    # The authority the world demands for that route: the satisfied action's tokens when
    # there is one, otherwise everything any producing action requires.
    required_authority: tuple[str, ...] = ()
    decides_terminal: bool = False

    @property
    def permitted(self) -> bool:
        if self.standing == STANDING_ACTOR:
            return bool(self.satisfied_by)
        if self.standing == STANDING_UNPRODUCED:
            # Genuinely new ground: no action, no process, no schedule writes this. That
            # is what the novel route is FOR, and refusing it would close the system's
            # only escape from pre-enumeration. It is refused in exactly one case —
            # when the thing nobody produces is the thing the terminal reads, because
            # then "invent an action" and "write the answer" are the same move.
            return not self.decides_terminal
        return False

    def refusal(self) -> str:
        if self.permitted:
            return ""
        what = display_term(self.term)
        if self.standing == STANDING_ACTOR:
            return (
                f"this would write {what}, which in this world requires standing the actor "
                f"does not hold (compiled actions producing it: "
                f"{list(self.actor_producers)}, requiring {list(self.required_authority)}); "
                "a novel action is not a way around authority"
            )
        if self.standing == STANDING_ENVIRONMENT:
            return (
                f"this would write {what}, which in this world only the environment "
                f"produces ({list(self.environment_producers)}). An effect no actor "
                "produces is not one any actor has standing to perform: the actor may "
                "state an intention, and the world remains the thing that does it"
            )
        return (
            f"this would write {what}, which the terminal condition reads and which "
            "nothing in this compiled world produces. A novel action may not settle the "
            "outcome by writing it directly"
        )


@dataclass(frozen=True)
class NovelResolution:
    executed: bool
    reason: str
    effects: tuple[Effect, ...] = ()
    required_authority: tuple[str, ...] = ()
    # What the world said about each term the proposal would write. Kept on the
    # resolution so a permitted novel action records the standing it acted on, not only
    # a refused one records the standing it lacked.
    standing: tuple[TermStanding, ...] = ()


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

    # 2. authority validation — against the world, not only against the interpreter.
    #    The interpreter is a model, and a model asked "what authority does this need?"
    #    can answer "none". That would make a novel action a way to do, without
    #    standing, exactly what the compiled world says requires standing. So the
    #    binding check is what the *compiled world* demands of anyone producing this
    #    effect; the interpreter's answer is an additional constraint on top, never a
    #    replacement for it.
    blocked = _blocked_by_world_authority(spec, effects, actor)
    if blocked:
        return NovelResolution(False, blocked, effects, required), [resp]

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
    events, _deferred = executor.build_events(world, effects, binding)
    return NovelResolution(True, "novel action authorized and executed", effects, required), [
        resp,
        *events,
    ]


def _effect_signature(eff: Effect) -> tuple[str, str]:
    """What an effect *does*, ignoring its values: the op plus the thing it writes to.

    Two effects with the same signature change the same part of the world, whatever the
    action producing them is called.
    """

    p = eff.params_dict
    target = str(
        p.get("collection") or p.get("field") or p.get("document") or p.get("resource") or ""
    )
    return (eff.op, target)


def _blocked_by_world_authority(
    spec: WorldSpec, effects: tuple[Effect, ...], actor: ActorState
) -> str:
    """Refuse a novel action that reaches an effect the compiled world gates.

    If every compiled action that produces this same effect requires standing this actor
    does not hold, the actor cannot reach that effect by renaming the route to it.
    Effects the compiled world has no action for are not covered here — those are
    genuinely novel, and the interpreter's declared authority governs them.
    """

    for eff in effects:
        sig = _effect_signature(eff)
        gatekeepers = [
            a for a in spec.actions if any(_effect_signature(e) == sig for e in a.effects)
        ]
        if not gatekeepers:
            continue
        if any(
            all(token in actor.authority for token in a.required_authority)
            and a.eligible(actor.role, actor.actor_id)
            for a in gatekeepers
        ):
            continue
        needed = sorted({t for a in gatekeepers for t in a.required_authority})
        return (
            f"this would {eff.op} {sig[1] or 'world state'}, which in this world requires "
            f"standing the actor does not hold (compiled actions producing it require "
            f"{needed}); a novel action is not a way around authority"
        )
    return ""


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
