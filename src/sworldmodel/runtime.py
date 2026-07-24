"""The branch runtime: one causal route from actors and events to the terminal.

For each genuine-uncertainty scenario we clone the verified world, release the
scenario's uncertain future data, and drive the protocol graph — briefing, proposal,
statements (delivered and reacted to), decision, votes — recording every actor
decision. The terminal is evaluated by deterministic code from the *actual* votes.
There is no second, hidden model: delete the actor calls and the votes disappear.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .actors import ActorRuntime
from .compiler import CompiledWorld
from .errors import GatewayError
from .gateway import ModelGateway
from .ids import content_id
from .intents import Environment
from .mechanisms import evaluate_action_terminal, evaluate_terminal
from .models import (
    BranchOutcome,
    BranchWeight,
    Event,
    EventKind,
    TrajectorySummary,
    Visibility,
)
from .protocols import ProtocolGraph, StepKind
from .uncertainty import Scenario
from .world import WorldState


@dataclass
class ActorDecisionRecord:
    branch_id: str
    actor_id: str
    stage: str
    decision_context: dict[str, object]
    retrieved_memory_ids: list[str]
    intent: dict[str, object]
    validation: str
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
    env = Environment()
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
            world = _release_scenario_data(world, scenario, env)
            world = _run_protocol(
                world, compiled.protocol, env, actor_runtime, seed, decisions, ledger
            )
        except GatewayError as exc:
            # A provider failure leaves this branch's mass genuinely unresolved. It is
            # NEVER converted into a prior or a default vote.
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


def _release_scenario_data(world: WorldState, scenario: Scenario, env: Environment) -> WorldState:
    """Deliver the scenario's uncertain future data as an observable event.

    This is a *simulated* future (a branch hypothesis), not post-cutoff evidence: the
    levels come from the uncertainty model, not from a fact published after ``as_of``.
    """

    as_of = world.contract.as_of
    horizon = world.contract.horizon
    t_data = as_of + (horizon - as_of) / 2
    ev = env.environment_event(
        world,
        kind=EventKind.EXTERNAL_DATA_RELEASED,
        payload={
            "signals": dict(scenario.signal_levels),
            "conditions": dict(scenario.conditions),
            "epistemic_type": "hypothesis",
        },
        time=t_data,
        visibility=Visibility.PUBLIC,
    )
    ledger_world = world.apply([ev])
    return ledger_world


def _run_protocol(
    world: WorldState,
    protocol: ProtocolGraph,
    env: Environment,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
) -> WorldState:
    contract = world.contract
    for step in protocol.steps:
        kind = step.kind
        if kind == StepKind.DISTRIBUTE_BRIEFING:
            world = world.with_time(contract.horizon)
            ev = env.environment_event(
                world,
                kind=EventKind.BRIEFING_DISTRIBUTED,
                payload={"text": step.get("text", ""), "signals": world.signals_dict()},
                time=contract.horizon,
            )
            world = world.apply([ev])
            ledger.append(_last(world))
        elif kind == StepKind.INTRODUCE_PROPOSAL:
            pid = content_id("prop", world.branch_id, step.get("option"))
            ev = env.environment_event(
                world,
                kind=EventKind.PROPOSAL_INTRODUCED,
                actor_id=step.get("chair"),
                payload={
                    "proposal_id": pid,
                    "option": step.get("option"),
                    "text": step.get("text", ""),
                    "revision_of": None,
                },
                time=world.time,
            )
            world = world.apply([ev]).with_stage("positions")
            ledger.append(_last(world))
        elif kind == StepKind.REQUEST_STATEMENTS:
            world = world.with_stage("positions")
            for aid in step.get("members", ()):
                world = _actor_act(world, aid, env, actor_runtime, seed, decisions, ledger)
        elif kind == StepKind.DELIVER_STATEMENTS:
            world = world.with_time(world.time + timedelta(minutes=5))
        elif kind == StepKind.REVISE_PROPOSAL:
            world = world.with_time(world.time + timedelta(minutes=1))
        elif kind == StepKind.OPEN_DECISION:
            world = world.with_time(world.time + timedelta(minutes=5))
            ev = env.environment_event(
                world, kind=EventKind.DECISION_OPENED, payload={}, time=world.time
            )
            world = world.apply([ev])
            ledger.append(_last(world))
        elif kind == StepKind.CAST_VOTES:
            world = world.with_stage("decision")
            for aid in step.get("members", ()):
                world = _actor_act(world, aid, env, actor_runtime, seed, decisions, ledger)
        elif kind == StepKind.ACTOR_TURN:
            world = world.with_stage(str(step.get("stage", "act")))
            aid = step.get("actor")
            if aid in world.actors:
                world = _actor_act(world, str(aid), env, actor_runtime, seed, decisions, ledger)
        elif kind in (StepKind.TALLY, StepKind.EVALUATE):
            world = _evaluate_and_record(world, env, ledger)
        elif kind == StepKind.PUBLISH:
            outcome = world.terminal_state.outcome if world.terminal_state else "unresolved"
            ev = env.environment_event(
                world,
                kind=EventKind.RESULT_PUBLISHED,
                payload={"outcome": outcome or "unresolved"},
                time=world.time,
            )
            world = world.apply([ev])
            ledger.append(_last(world))
    return world


def _evaluate_and_record(world: WorldState, env: Environment, ledger: list[Event]) -> WorldState:
    """Dispatch the terminal predicate: committee tally or actor-action, by mechanism."""

    contract = world.contract
    spec = contract.terminal_predicate
    if spec.mechanism == "actor_action":
        evaluation = evaluate_action_terminal(world.event_history, spec)
    else:
        evaluation = evaluate_terminal(
            world.votes_dict(),
            world.vote_powers(),
            contract.expected_voting_seats or len(world.voting_actor_ids()),
            contract.decision_rule,
            spec,
        )
    world = world.set_terminal(evaluation)
    ev = env.environment_event(
        world,
        kind=EventKind.TALLY_COMPUTED,
        payload={
            "tally": dict(evaluation.tally),
            "carried": evaluation.carried or "none",
            "outcome": evaluation.outcome or "unresolved",
            "reason": evaluation.reason,
        },
        time=world.time,
    )
    world = world.apply([ev])
    ledger.append(_last(world))
    return world


def _actor_act(
    world: WorldState,
    actor_id: str,
    env: Environment,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
) -> WorldState:
    actor = world.actors[actor_id]
    view = world.view_for(actor_id, None)
    intent, new_actor, responses, context = actor_runtime.step(actor, view, seed=seed)
    world = world.with_actor(new_actor)

    validated = env.validate(intent, world)
    events = env.execute(validated, world)
    world = world.apply(events)
    for ev in events:
        ledger.append(_find(world, ev.event_id))

    rmi = context.get("retrieved_memory_ids")
    retrieved_ids = [str(x) for x in rmi] if isinstance(rmi, list) else []
    decisions.append(
        ActorDecisionRecord(
            branch_id=world.branch_id,
            actor_id=actor_id,
            stage=view.stage,
            decision_context=context,
            retrieved_memory_ids=retrieved_ids,
            intent={
                "kind": intent.kind,
                "payload": dict(intent.payload),
                "rationale": intent.rationale,
                "expected_effect": intent.expected_effect,
                "referenced_memory_ids": list(intent.referenced_memory_ids),
                "referenced_observation_ids": list(intent.referenced_observation_ids),
            },
            validation="ok",
            event_ids=[e.event_id for e in events],
            prompt_hash=responses[-1].prompt_hash,
            model=responses[-1].model,
            tokens_out=responses[-1].tokens_out,
        )
    )
    return world


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
        votes=tuple(sorted(world.votes)),
        final_tally=term.tally if term else (),
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
        votes=(),
        final_tally=(),
        event_count=0,
    )


def _unresolved_summary(scenario: Scenario, reason: str) -> TrajectorySummary:
    return TrajectorySummary(
        branch_id=scenario.scenario_id,
        weight=scenario.weight,
        outcome=None,
        narrative=f"Branch {scenario.scenario_id} left unresolved: {reason}.",
        votes=(),
        key_conditions=scenario.conditions,
    )


def _summary(world: WorldState, scenario: Scenario) -> TrajectorySummary:
    term = world.terminal_state
    votes = tuple(sorted(world.votes))
    if term and term.resolved:
        vote_str = ", ".join(f"{a}:{o}" for a, o in votes)
        narrative = (
            f"Under conditions [{_cond_str(scenario)}], the board voted {vote_str}; "
            f"{term.reason} -> {term.outcome}."
        )
        outcome = term.outcome
    else:
        narrative = (
            f"Under conditions [{_cond_str(scenario)}], the branch did not resolve: "
            f"{term.reason if term else 'no terminal'}."
        )
        outcome = None
    return TrajectorySummary(
        branch_id=scenario.scenario_id,
        weight=scenario.weight,
        outcome=outcome,
        narrative=narrative,
        votes=votes,
        key_conditions=scenario.conditions,
    )


def _cond_str(scenario: Scenario) -> str:
    return ", ".join(f"{k}={v}" for k, v in scenario.conditions) or "baseline"
