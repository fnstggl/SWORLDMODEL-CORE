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
from datetime import datetime
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
    parse_effect,
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
    # Effects the proposal stamped in the future. They are scheduled, not applied, and
    # they must reach the caller: dropping them reported an action as executed that
    # produced nothing at all.
    deferred: tuple[tuple[datetime, Effect], ...] = ()


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

    declared = tuple(str(a) for a in (data.get("required_authority") or []))

    # 2. authority validation — against the world, not only against the interpreter.
    #    ``required_authority`` is written by the same model that just proposed the
    #    action, so it is a free parameter deciding whether a gate fires: asked "what
    #    authority does this need?", a model can answer "none", and that answer must not
    #    be able to unlock anything. The world's verdict is derived from what else
    #    produces this state and what standing the compiled actors hold; the declaration
    #    may only ADD to it.
    standing = world_standing(spec, effects, actor)
    required = tuple(sorted({*declared, *(t for s in standing for t in s.required_authority)}))

    #    The declaration is checked first only because it is the proposal failing its own
    #    stated requirement — the cheaper, more specific answer to give back to an actor.
    #    Both checks are unconditional and neither can be reached around: what follows
    #    refuses on the world's verdict whatever the declaration said, including when it
    #    said nothing at all.
    missing = [a for a in declared if a not in actor.authority]
    if missing:
        return (
            NovelResolution(False, f"actor lacks authority {missing}", effects, required, standing),
            [resp],
        )

    refusal = next((s.refusal() for s in standing if not s.permitted), "")
    if refusal:
        return NovelResolution(False, refusal, effects, required, standing), [resp]

    #    There is deliberately no third token check here. The world's own requirement is
    #    enforced inside ``world_standing``: a term is permitted only when this actor
    #    holds a compiled route to it, and ``_satisfies`` grants a route only to an actor
    #    already holding that action's authority. Re-checking ``required`` at this point
    #    would be a gate that can never fire — which is the very thing being fixed, one
    #    layer up. ``required`` travels on the resolution as the record of what standing
    #    was actually relied on.

    # 3-4. feasibility / resource / timing validation.
    binding = {
        "actor": actor.actor_id,
        "self": actor.entity,
        "params": choice.novel_params_dict,
        "target": choice.novel_target,
    }
    ok, reason = executor.can_apply(world, effects, binding)
    if not ok:
        return NovelResolution(False, f"infeasible: {reason}", effects, required, standing), [resp]

    # 5. translation into safe world operations -> execute.
    #    An effect stamped in the future has not happened; it is scheduled. The deferred
    #    list was discarded here, so a novel `schedule_event` produced zero events and
    #    was still reported "authorized and executed" — the actor told its intention had
    #    become a consequence when nothing whatever had occurred.
    events, deferred = executor.build_events(world, effects, binding)
    return NovelResolution(
        True,
        "novel action authorized and executed",
        effects,
        required,
        standing,
        tuple(deferred),
    ), [resp, *events]


def terminal_terms(spec: WorldSpec) -> frozenset[str]:
    """Everything the compiled terminal reads.

    Both legs: ``unresolved_when`` decides the answer exactly as much as ``yes_when``
    does — writing it turns a determined branch into an undetermined one.
    """

    return expression_terms(spec.terminal.yes_when) | expression_terms(
        spec.terminal.unresolved_when
    )


def _environment_producers(spec: WorldSpec) -> dict[str, list[str]]:
    """Which non-agent parts of the compiled world write which terms.

    A process node's own ``effects`` and an external process's occurrences are the world
    happening, not anybody acting: they fire on the calendar, on their entry condition,
    with no actor and no intention behind them. Labels match ``world_compiler``'s
    producer lineage so a compile-time refusal and this one name the same producer.
    """

    out: dict[str, list[str]] = {}
    for node in spec.process.nodes:
        for eff in node.effects:
            for term in effect_terms(eff):
                out.setdefault(term, []).append(f"process_node:{node.node_id}")
        if node.stage:
            out.setdefault("stage:", []).append(f"process_node:{node.node_id}")
    for proc in spec.external_processes:
        for i, occ in enumerate(proc.occurrences):
            for eff in occ.effects:
                for term in effect_terms(eff):
                    out.setdefault(term, []).append(f"external_process:{proc.process_id}#{i}")
    return out


def _satisfies(action: ActionDefinition, actor: ActorState) -> bool:
    """Whether this actor could take this compiled action at all — who it is, not when.

    Standing is a property of the actor and the action, so stage windows and
    preconditions are deliberately not consulted: they say *when* a route is open, and
    a route that is shut today is still the actor's route.
    """

    return action.eligible(actor.role, actor.actor_id) and all(
        token in actor.authority for token in action.required_authority
    )


def world_standing(
    spec: WorldSpec, effects: tuple[Effect, ...], actor: ActorState
) -> tuple[TermStanding, ...]:
    """What the compiled world says about this actor producing each of these effects.

    One :class:`TermStanding` per term the proposal would write, in effect order, with
    no term left unclassified. The compiled actions are the only producers that confer
    standing; process nodes and external processes are producers that confer none, which
    is precisely why an effect they alone produce must be refused rather than waved
    through. Rain is not an action. A scheduled release is not an action. A quarterly
    production run is not an action.
    """

    environment = _environment_producers(spec)
    decides = terminal_terms(spec)
    out: list[TermStanding] = []
    seen: set[str] = set()

    for eff in effects:
        for term in sorted(effect_terms(eff)):
            if term in seen:
                continue
            seen.add(term)
            producers = [a for a in spec.actions if term in _action_terms(a)]
            env = tuple(dict.fromkeys(environment.get(term, ())))
            if producers:
                satisfied = next((a for a in producers if _satisfies(a, actor)), None)
                needed = (
                    tuple(satisfied.required_authority)
                    if satisfied is not None
                    else tuple(sorted({t for a in producers for t in a.required_authority}))
                )
                out.append(
                    TermStanding(
                        term=term,
                        standing=STANDING_ACTOR,
                        actor_producers=tuple(a.action_id for a in producers),
                        environment_producers=env,
                        satisfied_by=satisfied.action_id if satisfied is not None else "",
                        required_authority=needed,
                        decides_terminal=term in decides,
                    )
                )
                continue
            out.append(
                TermStanding(
                    term=term,
                    standing=STANDING_ENVIRONMENT if env else STANDING_UNPRODUCED,
                    environment_producers=env,
                    decides_terminal=term in decides,
                )
            )
    return tuple(out)


def _action_terms(action: ActionDefinition) -> frozenset[str]:
    out: frozenset[str] = frozenset()
    for eff in action.effects:
        out |= effect_terms(eff)
    return out


def _parse_effects(raw: Any) -> tuple[Effect, ...]:
    """Read the interpreter's effects exactly as the compiled path reads a compiler's.

    ``parse_effect`` is used rather than a local construction so the *same JSON* names
    the same world state whichever side wrote it. It was a local construction, and the
    divergence was the thing to be afraid of: the compiled side renames a model's
    ``field_id`` to ``field`` and the novel side did not, so an authority check reading
    ``field`` and an executor reading ``field`` would have seen an effect that, on this
    path alone, named neither. A gate and the thing it gates must read one object.
    """

    if not isinstance(raw, list):
        return ()
    return tuple(
        parse_effect(item) for item in raw if isinstance(item, dict) and str(item.get("op", ""))
    )
