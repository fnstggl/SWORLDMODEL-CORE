"""The one universal event runtime.

For each genuine-uncertainty scenario we clone the verified world, release the
scenario's uncertain future data, and drive the *compiled* process graph — advancing
time, applying environment effects, and letting each participant perceive, plan, and
act by choosing among its feasible compiled actions or proposing a novel one. The
terminal is then evaluated by the deterministic declarative evaluator from the actual
world state.

There is exactly one runtime. It does **not** branch on the kind of question: a
committee vote, an individual response, a negotiation, a population behavior, and a
geopolitical process are all just different compiled :class:`WorldSpec` programs
executed here. Adding a new kind of question adds compiled data, never a runtime
branch. Delete the actor calls and the records disappear — there is no second, hidden
model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .actors import ActorRuntime
from .compiled import CompiledWorld
from .effects import EffectExecutor
from .errors import GatewayError
from .executor import ActionExecutor
from .expressions import evaluate
from .gateway import ModelGateway
from .models import BranchOutcome, BranchWeight, Event, TrajectorySummary, Visibility
from .uncertainty import Scenario
from .world import WorldState
from .worldspec import ProcessNode, TerminalExpression, WorldSpec


@dataclass(frozen=True)
class TerminalEvaluation:
    resolved: bool
    outcome: str | None  # "YES" | "NO" | None
    reason: str
    highlights: tuple[tuple[str, str], ...] = ()


@dataclass
class ActorDecisionRecord:
    branch_id: str
    actor_id: str
    stage: str
    trigger: str
    decision_context: dict[str, object]
    retrieved_memory_ids: list[str]
    choice: dict[str, object]
    status: str
    reason: str
    event_ids: list[str]
    prompt_hash: str
    model: str
    tokens_out: int


@dataclass
class RunResult:
    branch_outcomes: tuple[BranchOutcome, ...]
    trajectory_summaries: tuple[TrajectorySummary, ...]
    event_ledger: list[Event]
    actor_decisions: list[ActorDecisionRecord]
    final_worlds: dict[str, WorldState]
    truncated_mass: float
    truncated_reason: str


def run(compiled: CompiledWorld, gateway: ModelGateway, *, seed: int) -> RunResult:
    effects = EffectExecutor()
    action_exec = ActionExecutor(gateway, effects)
    actor_runtime = ActorRuntime(gateway)

    branch_outcomes: list[BranchOutcome] = []
    summaries: list[TrajectorySummary] = []
    ledger: list[Event] = []
    decisions: list[ActorDecisionRecord] = []
    final_worlds: dict[str, WorldState] = {}

    for scenario in compiled.scenario_set.scenarios:
        weight = BranchWeight(scenario.weight, scenario.provenance, scenario.provenance_detail)
        world = compiled.base_world.clone(new_branch_id=scenario.scenario_id, weight=weight)
        try:
            world = _release_scenario_data(world, scenario, effects, ledger)
            world = _run_graph(
                world, compiled.spec, effects, action_exec, actor_runtime, seed, decisions, ledger
            )
            world = _finalize(world, compiled.spec.terminal, effects, ledger)
        except GatewayError as exc:
            final_worlds[scenario.scenario_id] = world
            branch_outcomes.append(_unresolved_outcome(scenario, f"provider_failure: {exc}"))
            summaries.append(_unresolved_summary(scenario, "provider failure"))
            continue
        final_worlds[scenario.scenario_id] = world
        branch_outcomes.append(_branch_outcome(world, scenario))
        summaries.append(_summary(world, scenario))

    return RunResult(
        branch_outcomes=tuple(branch_outcomes),
        trajectory_summaries=tuple(summaries),
        event_ledger=ledger,
        actor_decisions=decisions,
        final_worlds=final_worlds,
        truncated_mass=compiled.scenario_set.truncated_mass,
        truncated_reason=compiled.scenario_set.truncated_reason,
    )


def _release_scenario_data(
    world: WorldState, scenario: Scenario, effects: EffectExecutor, ledger: list[Event]
) -> WorldState:
    """Deliver the scenario's uncertain future field levels as an observable data
    release (a simulated branch hypothesis, not a post-cutoff fact)."""

    if not scenario.field_levels:
        return world
    as_of = world.contract.as_of
    horizon = world.contract.horizon
    t_data = as_of + (horizon - as_of) / 2
    ev = effects.raw_event(
        world.with_time(t_data),
        kind="release_data",
        actor_id=None,
        payload={"fields": dict(scenario.field_levels), "epistemic_type": "hypothesis"},
        visibility=Visibility.PUBLIC,
    )
    world = world.apply([ev])
    ledger.append(_last(world))
    return world


@dataclass(frozen=True)
class _Activation:
    """A justified reason to invoke one actor now. An actor is never invoked on a
    fixed schedule: it is invoked because something in the world actually affects it."""

    actor_id: str
    reason: str


def _run_graph(
    world: WorldState,
    spec: WorldSpec,
    effects: EffectExecutor,
    action_exec: ActionExecutor,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
) -> WorldState:
    for node in spec.process.nodes:
        if not bool(evaluate(node.condition, world)):
            continue
        world = _advance(world, node)
        if node.stage:
            world = world.with_stage(node.stage)
        # 1. environment effects for the node (briefings, data releases, scheduling).
        if node.effects:
            env_events = effects.build_events(world, node.effects, {"actor": None, "self": None})
            world = world.apply(env_events)
            for ev in env_events:
                ledger.append(_find(world, ev.event_id))
        # 2. interaction: event-driven actor invocation (never a fixed turn order).
        world = _drive_activations(
            world, spec, node, effects, action_exec, actor_runtime, seed, decisions, ledger
        )
    return world


def _drive_activations(
    world: WorldState,
    spec: WorldSpec,
    node: ProcessNode,
    effects: EffectExecutor,
    action_exec: ActionExecutor,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
) -> WorldState:
    """Invoke actors from events until nothing further affects anyone.

    An actor enters the queue only with a concrete trigger: information it has not yet
    perceived, an opportunity (a feasible compiled action in this node), an unmet
    information need it recorded earlier, or a consequence of another actor's completed
    action that reaches it. An actor with no trigger is simply not invoked — inertia is
    preserved rather than manufacturing decorative activity.

    ``node.rounds`` is a *budget* (the most times one actor may be re-invoked in this
    node), not a schedule: it bounds cascades without forcing anyone to act.
    """

    participants = _participants(world, spec, node)
    if not participants:
        return world
    budget = max(1, node.rounds)
    turns: dict[str, int] = dict.fromkeys(participants, 0)

    queue: deque[_Activation] = deque()
    for aid in participants:
        reason = _trigger_for(world, spec, node, action_exec, aid)
        if reason:
            queue.append(_Activation(aid, reason))

    while queue:
        activation = queue.popleft()
        aid = activation.actor_id
        if aid not in world.actors or turns.get(aid, 0) >= budget:
            continue
        # Re-check at pop time: the world moved since this activation was queued.
        if not _trigger_for(world, spec, node, action_exec, aid):
            continue
        turns[aid] = turns.get(aid, 0) + 1
        before = len(world.event_history)
        world = _actor_turn(
            world,
            aid,
            node,
            spec,
            effects,
            action_exec,
            actor_runtime,
            seed,
            decisions,
            ledger,
            trigger=activation.reason,
        )
        new_events = world.event_history[before:]
        # A completed action can reach other actors (information, a response, or a
        # newly-feasible option). Those are the only follow-up invocations.
        for other in participants:
            if other == aid or turns.get(other, 0) >= budget:
                continue
            reason = _consequence_trigger(world, spec, node, action_exec, other, new_events)
            if reason:
                queue.append(_Activation(other, reason))
    return world


def _trigger_for(
    world: WorldState,
    spec: WorldSpec,
    node: ProcessNode,
    action_exec: ActionExecutor,
    actor_id: str,
) -> str:
    """Why this actor should act now, or '' when nothing affects it."""

    actor = world.actors.get(actor_id)
    if actor is None:
        return ""
    reasons: list[str] = []
    if world.view_for(actor_id).observations:
        reasons.append("information")
    if action_exec.feasible_actions(world, node, actor, spec):
        reasons.append("opportunity")
    if actor.pending_questions:
        reasons.append("pending_need")
    if not reasons and node.allow_novel and world.view_for(actor_id).public_facts:
        # An actor with no compiled option may still propose a novel action, but only
        # when it has something to go on; otherwise it stays inert.
        return ""
    return "+".join(reasons)


def _consequence_trigger(
    world: WorldState,
    spec: WorldSpec,
    node: ProcessNode,
    action_exec: ActionExecutor,
    actor_id: str,
    new_events: tuple[Event, ...],
) -> str:
    """Whether another actor's completed action actually reaches this actor."""

    if not new_events:
        return ""
    actor = world.actors.get(actor_id)
    if actor is None:
        return ""
    view = world.view_for(actor_id)
    fresh = {o.obs_id for o in view.observations}
    if any(ev.event_id in fresh for ev in new_events):
        return "response_to_completed_action"
    if actor.pending_questions and view.observations:
        return "pending_need_met"
    return ""


def _actor_turn(
    world: WorldState,
    actor_id: str,
    node: ProcessNode,
    spec: WorldSpec,
    effects: EffectExecutor,
    action_exec: ActionExecutor,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
    *,
    trigger: str = "",
) -> WorldState:
    actor = world.actors[actor_id]
    feasible = action_exec.feasible_actions(world, node, actor, spec)
    base_view = world.view_for(actor_id, None)
    view = _with_menu(base_view, feasible, node, action_exec)

    choice, new_actor, responses, context = actor_runtime.step(actor, view, seed=seed)
    world = world.with_actor(new_actor)

    outcome = action_exec.execute(actor, choice, world, spec, seed)
    responses.extend(_gw_responses(outcome.gateway_responses))
    world = world.apply(outcome.events)
    for ev in outcome.events:
        ledger.append(_find(world, ev.event_id))

    rmi = context.get("retrieved_memory_ids")
    retrieved_ids = [str(x) for x in rmi] if isinstance(rmi, list) else []
    decisions.append(
        ActorDecisionRecord(
            branch_id=world.branch_id,
            actor_id=actor_id,
            stage=view.stage,
            trigger=trigger,
            decision_context=context,
            retrieved_memory_ids=retrieved_ids,
            choice={
                "mode": choice.mode,
                "action_id": choice.action_id,
                "params": dict(choice.params),
                "target": choice.target,
                "novel_description": choice.novel_description,
                "rationale": choice.rationale,
            },
            status=outcome.status,
            reason=outcome.reason,
            event_ids=[e.event_id for e in outcome.events],
            prompt_hash=responses[-1].prompt_hash if responses else "",
            model=responses[-1].model if responses else "",
            tokens_out=responses[-1].tokens_out if responses else 0,
        )
    )
    return world


def _finalize(
    world: WorldState, terminal: TerminalExpression, effects: EffectExecutor, ledger: list[Event]
) -> WorldState:
    world = world.with_time(world.contract.horizon)
    evaluation = evaluate_terminal(world, terminal)
    world = world.set_terminal(evaluation)
    ev = effects.raw_event(
        world,
        kind="create_event",
        actor_id=None,
        payload={
            "event_type": "result_recorded",
            "text": evaluation.reason,
            "data": {"outcome": evaluation.outcome or "unresolved"},
        },
        visibility=Visibility.PUBLIC,
    )
    world = world.apply([ev])
    ledger.append(_last(world))
    return world


def evaluate_terminal(world: WorldState, terminal: TerminalExpression) -> TerminalEvaluation:
    """The single place YES/NO/unresolved is decided — deterministic, from world state,
    via the universal operators only. No LLM, no fixed mechanism family."""

    if bool(evaluate(terminal.unresolved_when, world)):
        return TerminalEvaluation(
            resolved=False,
            outcome=None,
            reason=terminal.description or "terminal condition not determinable",
            highlights=_highlights(world),
        )
    yes = bool(evaluate(terminal.yes_when, world))
    return TerminalEvaluation(
        resolved=True,
        outcome="YES" if yes else "NO",
        reason=terminal.description or ("yes_when satisfied" if yes else "yes_when not satisfied"),
        highlights=_highlights(world),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _advance(world: WorldState, node: ProcessNode) -> WorldState:
    if isinstance(node.at, str) and node.at:
        try:
            from datetime import datetime

            world = world.with_time(datetime.fromisoformat(node.at))
        except ValueError:
            pass
    if node.advance_seconds:
        from datetime import timedelta

        world = world.with_time(world.time + timedelta(seconds=node.advance_seconds))
    return world


def _participants(world: WorldState, spec: WorldSpec, node: ProcessNode) -> tuple[str, ...]:
    if not node.participants:
        return ()  # an environment-only node: no actor acts here
    if node.participants == ("*",):
        return tuple(a.entity_id for a in spec.actors if a.entity_id in world.actors)
    out: list[str] = []
    for sel in node.participants:
        if sel == "*":
            out.extend(a.entity_id for a in spec.actors if a.entity_id in world.actors)
        elif sel.startswith("role:"):
            role = sel[5:]
            out.extend(aid for aid, a in world.actors.items() if a.role == role)
        elif sel in world.actors:
            out.append(sel)
    # preserve declared actor order, de-duplicated
    order = {a.entity_id: i for i, a in enumerate(spec.actors)}
    return tuple(sorted(dict.fromkeys(out), key=lambda x: order.get(x, 1 << 30)))


def _with_menu(
    view: Any, feasible: list[Any], node: ProcessNode, action_exec: ActionExecutor
) -> Any:
    from dataclasses import replace

    cards = tuple(action_exec.action_card(a) for a in feasible)
    return replace(view, feasible_actions=cards, allow_novel=node.allow_novel)


def _gw_responses(items: list[Any]) -> list[Any]:
    return list(items)


def _highlights(world: WorldState) -> tuple[tuple[str, str], ...]:
    """A compact, human-readable snapshot of the decisive world state for the report."""

    out: list[tuple[str, str]] = []
    for coll, recs in world.records:
        for r in recs:
            out.append((f"{coll}:{r.key}", str(r.value)))
    for name, value in world.fields:
        out.append((f"field:{name}", str(value)))
    return tuple(out)


def _last(world: WorldState) -> Event:
    return world.event_history[-1]


def _find(world: WorldState, event_id: str) -> Event:
    for ev in reversed(world.event_history):
        if ev.event_id == event_id:
            return ev
    raise KeyError(event_id)


def _branch_outcome(world: WorldState, scenario: Scenario) -> BranchOutcome:
    term = world.terminal_state
    resolved = bool(term and term.resolved)
    outcome = term.outcome if (term and term.resolved) else None
    reason = None if resolved else (term.reason if term else "no terminal reached")
    return BranchOutcome(
        branch_id=scenario.scenario_id,
        parent_lineage=("root",),
        weight=scenario.weight,
        resolved=resolved,
        outcome=outcome,
        unresolved_reason=reason,
        truncated=False,
        key_conditions=scenario.conditions,
        records=term.highlights if term else (),
        event_count=len(world.event_history),
    )


def _unresolved_outcome(scenario: Scenario, reason: str) -> BranchOutcome:
    return BranchOutcome(
        branch_id=scenario.scenario_id,
        parent_lineage=("root",),
        weight=scenario.weight,
        resolved=False,
        outcome=None,
        unresolved_reason=reason,
        truncated=False,
        key_conditions=scenario.conditions,
        records=(),
        event_count=0,
    )


def _unresolved_summary(scenario: Scenario, reason: str) -> TrajectorySummary:
    return TrajectorySummary(
        branch_id=scenario.scenario_id,
        weight=scenario.weight,
        outcome=None,
        narrative=f"Branch {scenario.scenario_id} left unresolved: {reason}.",
        records=(),
        key_conditions=scenario.conditions,
    )


def _summary(world: WorldState, scenario: Scenario) -> TrajectorySummary:
    term = world.terminal_state
    conds = ", ".join(f"{k}={v}" for k, v in scenario.conditions) or "baseline"
    if term and term.resolved:
        rec = ", ".join(f"{k}={v}" for k, v in term.highlights[:8])
        narrative = f"Under [{conds}], world state [{rec}]; {term.reason} -> {term.outcome}."
        outcome = term.outcome
    else:
        narrative = (
            f"Under [{conds}], the branch did not resolve: {term.reason if term else 'n/a'}."
        )
        outcome = None
    return TrajectorySummary(
        branch_id=scenario.scenario_id,
        weight=scenario.weight,
        outcome=outcome,
        narrative=narrative,
        records=term.highlights if term else (),
        key_conditions=scenario.conditions,
    )
