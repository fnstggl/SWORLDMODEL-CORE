"""The one universal runtime: an event-driven clock over compiled worlds.

The loop is the same for every question::

    next scheduled thing on the branch calendar
      -> apply it to the external world
      -> work out who could see the result, when it reaches them, when they notice
      -> wake ONLY the actors it actually affects, and record why each was woken
      -> the actor retrieves memories, inspects its plan, and continues / revises /
         interrupts / replaces it, then emits an intention
      -> the environment validates the intention and either starts it or refuses it
      -> consequences enter the queue at their real times
      -> advance to the next real event

There is no tick, no round, no "one turn per actor", no per-stage actor schedule. How
many times an actor is invoked is an *output* of the trajectory, not a constant: an
actor that nothing reaches is never invoked, and an actor in a busy exchange may be
invoked many times. Two branches routinely invoke the same actor a different number of
times because different things happen in them.

The runtime does not branch on the kind of question. A committee decision, a single
message reply, a negotiation, a population process and a multi-state process are all
compiled :class:`~sworldmodel.worldspec.WorldSpec` programs executed here. Supporting a
new kind of question adds compiled *data*; it never adds a branch in this file.

Nothing here can save a run. When the model cannot be reached the branch's mass stays
unresolved; when a trajectory stops making progress it stops, and says so.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from .actors import (
    PLAN_ACTIVE,
    ActorRuntime,
    ActorState,
)
from .compiled import CompiledWorld
from .effects import UNIVERSAL_OPS, EffectExecutor
from .errors import GatewayError, UndeterminedExpressionError
from .executor import KIND_ACTION_COMPLETION, ActionExecutor
from .expressions import evaluate
from .gateway import ModelGateway
from .models import BranchOutcome, BranchWeight, Event, EventStatus, TrajectorySummary, Visibility
from .schedule import (
    KIND_SCENARIO_RELEASE,
    ORIGIN_ACTOR_PLAN,
    ORIGIN_CONSEQUENCE,
    ORIGIN_EXTERNAL,
    ORIGIN_PROCESS,
    ScheduledEntry,
    make_entry,
)
from .uncertainty import Scenario, weights_grounded
from .world import Delivery, WorldState
from .worldspec import (
    MONOTONE_DOWN,
    MONOTONE_FIXED,
    MONOTONE_UP,
    Effect,
    ProcessNode,
    TerminalExpression,
    WakeRule,
    WorldSpec,
    expression_monotonicity,
    monotone_sources,
)

# Structural schedule-entry kinds. These are runtime mechanics, not domain event types:
# what a "message" or a "vote" is lives entirely in compiled data.
KIND_PROCESS_NODE = "process_node"
KIND_EXTERNAL = "external_occurrence"
KIND_NOTICE = "information_noticed"
KIND_DECISION = "actor_decision"
KIND_COMMITMENT_DUE = "commitment_due"
KIND_REVISIT = "revisit_condition"
KIND_NEED_DEADLINE = "information_need_deadline"
KIND_PLAN_STEP = "plan_step_due"
KIND_DEADLINE = "process_deadline"
KIND_DEFERRED_EFFECT = "deferred_effect"

# Why an actor was woken. Every invocation carries exactly one of these plus a detail.
WAKE_OPPORTUNITY = "process_opportunity"
WAKE_DIRECTED = "directed_information"
WAKE_COMMUNICATION = "communication_from_another_actor"
WAKE_RULE = "compiled_wake_rule"
WAKE_REVISIT = "own_revisit_condition"
WAKE_NEED_MET = "pending_need_answered"
WAKE_NEED_FAILED = "pending_need_unanswered_at_deadline"
WAKE_COMMITMENT = "commitment_due"
WAKE_OWN_ACTION = "own_action_resolved"
WAKE_PLAN_STEP = "own_plan_step_due"
WAKE_DEADLINE = "deadline_reached"

# Reasons that interrupt an actor who is in the middle of doing something. A mere
# opportunity does not: an actor with a plan in progress keeps working on it.
_INTERRUPTING = frozenset(
    {
        WAKE_DIRECTED,
        WAKE_COMMUNICATION,
        WAKE_RULE,
        WAKE_REVISIT,
        WAKE_NEED_MET,
        WAKE_NEED_FAILED,
        WAKE_COMMITMENT,
        WAKE_OWN_ACTION,
        WAKE_DEADLINE,
    }
)

# What KIND of cause brought the actor back. The distinction is not cosmetic: it is
# what makes the ACT-8 "repeated without new information" measurement readable.
#
# A CALENDAR-driven wake fires because a moment arrived. Arriving at a deadline or a
# scheduled opportunity with nothing new to read is normal life — a person turning up
# to a meeting has not learned anything on the way. Volume matters; each instance does
# not.
#
# An INFORMATION-driven wake fires because the world claims something reached this
# actor. One of those carrying nothing materially new is a defect every time: the
# runtime asserted an arrival that the actor's own noticed set does not contain.
#
# Reporting both under one headline made ACT-8 flag correct behaviour about half the
# time on a live run, which is how a reader learns to ignore it — and how FD-7 survived
# its first fix.
CALENDAR_WAKE_REASONS = frozenset(
    {
        WAKE_OPPORTUNITY,
        WAKE_COMMITMENT,
        WAKE_PLAN_STEP,
        WAKE_DEADLINE,
        WAKE_REVISIT,
        WAKE_NEED_FAILED,
    }
)
INFORMATION_WAKE_REASONS = frozenset(
    {
        WAKE_DIRECTED,
        WAKE_COMMUNICATION,
        WAKE_RULE,
        WAKE_NEED_MET,
        WAKE_OWN_ACTION,
    }
)

WAKE_CAUSE_CALENDAR = "calendar_driven"
WAKE_CAUSE_INFORMATION = "information_driven"
WAKE_CAUSE_UNCLASSIFIED = "unclassified"


def wake_cause_class(wake_reason: str) -> str:
    """Classify one recorded ``wake_reason`` as calendar- or information-driven.

    Several causes landing on one actor at one instant are merged into a single wake
    whose reason is the causes joined by ``+`` (see :func:`_merge_decision_entries`).
    Such a wake counts as information-driven if ANY of its causes was: the world stated
    that something reached this actor, and that statement is the one worth checking
    against what the actor actually noticed.
    """

    parts = [p for p in wake_reason.split("+") if p]
    if any(p in INFORMATION_WAKE_REASONS for p in parts):
        return WAKE_CAUSE_INFORMATION
    if any(p in CALENDAR_WAKE_REASONS for p in parts):
        return WAKE_CAUSE_CALENDAR
    return WAKE_CAUSE_UNCLASSIFIED


# A branch either ran until the world had nothing left to do inside the window, or WE
# stopped it. The second is not uncertainty about the world; it is a fact about this
# simulator, and it must be priced as one everywhere downstream.
EXECUTION_COMPLETE = "complete"
EXECUTION_INCOMPLETE = "EXECUTION_INCOMPLETE"

# Why a branch's mass never reached an answer. ``WORLD_UNDETERMINED`` is honest
# uncertainty — the world ran to the end of its own schedule and the terminal still
# could not be read. ``EXECUTION_INCOMPLETE`` is a failed run.
UNRESOLVED_WORLD_UNDETERMINED = "WORLD_UNDETERMINED"

# Budget scaling (see RunBudget.for_world).
_ACTOR_CALLS_FLOOR = 80
_ACTOR_CALLS_CEILING = 400
_ROUNDS_PER_ACTOR_PER_WEEK = 3
_ACTOR_RESPONSE_ALLOWANCE = 8
_EVENTS_PER_ACTOR_CALL = 5
_BATCHES_PER_ACTOR_CALL = 6


@dataclass(frozen=True)
class RunBudget:
    """Hard stops that bound a trajectory. Reaching one *ends* the branch and marks it
    unresolved with the reason; it never forces a decision, inserts a default action or
    resolves a terminal that the world did not reach.

    The defaults are the FLOOR, not the budget. A flat ceiling is a statement that every
    world costs the same to simulate, and it is false in the one direction that matters:
    a nine-participant world over ten weeks got the same 80 actor calls as a
    two-participant world over two weeks, so the budget — not the world — decided who
    got to act. Callers that do not pin a budget get :meth:`for_world`.
    """

    max_events: int = 600
    max_actor_calls: int = _ACTOR_CALLS_FLOOR
    max_batches: int = 400
    no_progress_batches: int = 8

    @classmethod
    def for_world(cls, *, participants: int, horizon_days: float) -> RunBudget:
        """A budget sized to the world, not to the last world anybody looked at.

        The formula, and why each term is there:

        ``participants x (ROUNDS_PER_WEEK x weeks + RESPONSE_ALLOWANCE)``

        * **per participant**, because an actor that never gets an invocation is a
          participant the run has silently deleted. The budget must be able to reach
          every one of them, and the roster is known before the run starts.
        * **rounds per week x weeks**, because occasions to act arrive with the calendar:
          a ten-week window genuinely offers more of them than a two-week window. Three
          per week is deliberately below any plausible real cadence — it is a budget,
          not a schedule, and nothing forces those invocations to be used.
        * **a flat response allowance**, because acting is not the only reason to be
          woken: an actor also has to be able to answer what reaches it. This is the
          term that a purely calendar-derived budget lacks, and its absence is what made
          a busy exchange starve the participants who had not spoken yet.

        The result is clamped to ``[80, 400]``. The floor is the old flat value, so no
        world is ever given LESS than it had. The ceiling is not a safety margin — it is
        the point past which "we ran out of calls" is the correct diagnosis and buying
        more of them would only hide it. Exhaustion is reported as
        ``EXECUTION_INCOMPLETE``, and a run carrying material incomplete mass is a
        failed run rather than a forecast.

        ``max_events`` and ``max_batches`` scale with the call budget for one reason:
        raising the actor-call ceiling alone moves the stop from one budget to the next
        and changes nothing a reader can see. Measured on the multiparty run, an
        invocation that acts costs about two ledger events (start, completion, its
        effects, the notices) — five is headroom, six batches likewise.
        """

        weeks = max(1.0, horizon_days / 7.0)
        per_actor = _ROUNDS_PER_ACTOR_PER_WEEK * weeks + _ACTOR_RESPONSE_ALLOWANCE
        calls = int(max(1, participants) * per_actor)
        calls = max(_ACTOR_CALLS_FLOOR, min(_ACTOR_CALLS_CEILING, calls))
        defaults = cls()
        return cls(
            max_events=max(defaults.max_events, calls * _EVENTS_PER_ACTOR_CALL),
            max_actor_calls=calls,
            max_batches=max(defaults.max_batches, calls * _BATCHES_PER_ACTOR_CALL),
            no_progress_batches=defaults.no_progress_batches,
        )


@dataclass(frozen=True)
class TerminalEvaluation:
    resolved: bool
    outcome: str | None  # "YES" | "NO" | None
    reason: str
    highlights: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BankedTerminal:
    """A monotone terminal, recorded at the instant the world satisfied it.

    A banked YES is final. The claim it makes is narrow and provable: the terminal was
    TRUE at ``at``, its expression is structurally monotone in the satisfying direction
    (see :func:`sworldmodel.worldspec.expression_monotonicity`), and the terms it grew
    out of are append-only — so nothing the rest of the simulation could have done, and
    nothing that stopping early prevented, can make it false again.

    It carries its own cause because a bank that cannot say what made it true is
    indistinguishable from an assertion: ``caused_by_event_ids`` are the ledger events
    that wrote the append-only terms in ``sources``.
    """

    at: str
    outcome: str  # always "YES"
    sources: tuple[str, ...]
    caused_by_event_ids: tuple[str, ...]
    cause_count: int
    world_version: int
    batches_elapsed: int
    actor_calls_elapsed: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "outcome": self.outcome,
            "sources": list(self.sources),
            "caused_by_event_ids": list(self.caused_by_event_ids),
            "cause_count": self.cause_count,
            "world_version": self.world_version,
            "batches_elapsed": self.batches_elapsed,
            "actor_calls_elapsed": self.actor_calls_elapsed,
            "reason": self.reason,
        }


@dataclass
class ActorDecisionRecord:
    """The complete record of one actor invocation — enough to replay it and enough to
    show that the actor, not the runtime, decided."""

    branch_id: str
    actor_id: str
    branch_time: str
    stage: str
    wake_reason: str
    wake_detail: str
    trigger_event_ids: list[str]
    delivered_observation_ids: list[str]
    noticed_observation_ids: list[str]
    retrieved_memory_ids: list[str]
    plan_before: dict[str, object] | None
    plan_after: dict[str, object] | None
    plan_disposition: str
    state_before: dict[str, object]
    state_after: dict[str, object]
    decision_context: dict[str, object]
    intent: dict[str, object]
    validation_status: str
    validation_reason: str
    event_ids: list[str]
    world_version_at_decision: int
    prompt_hash: str
    model: str
    tokens_out: int


@dataclass
class BranchDiagnostics:
    actor_call_counts: dict[str, int] = field(default_factory=dict)
    batches: int = 0
    events: int = 0
    stop_reason: str = "schedule exhausted"
    pending_beyond_horizon: list[dict[str, Any]] = field(default_factory=list)
    unfired_in_horizon: int = 0
    # W2. ``execution_status`` is EXECUTION_COMPLETE only when the branch ran until the
    # world had nothing further scheduled inside the window. Every other stop — any
    # budget, the no-progress guard, a provider failure — is EXECUTION_INCOMPLETE and
    # names the limit we hit in ``execution_limit``. ``unresolved_class`` is the seam
    # aggregation prices: "" when the branch resolved, otherwise EXECUTION_INCOMPLETE
    # (our failure) or UNRESOLVED_WORLD_UNDETERMINED (genuine uncertainty).
    execution_status: str = EXECUTION_COMPLETE
    execution_limit: str = ""
    unresolved_class: str = ""
    # W1. The monotone terminal this branch banked, if any, and the record of a bank
    # that the horizon evaluation went on to contradict — which is a defect in the
    # monotonicity analysis and must never be silent.
    banked_terminal: BankedTerminal | None = None
    bank_contradicted: str = ""


@dataclass
class RunResult:
    branch_outcomes: tuple[BranchOutcome, ...]
    trajectory_summaries: tuple[TrajectorySummary, ...]
    event_ledger: list[Event]
    actor_decisions: list[ActorDecisionRecord]
    final_worlds: dict[str, WorldState]
    truncated_mass: float
    truncated_reason: str
    diagnostics: dict[str, BranchDiagnostics] = field(default_factory=dict)
    # The branch weight this run could not simulate to the end of its own schedule. This
    # is not unresolved mass and it must never be averaged into one: it is the share of
    # the world we did not manage to run. A run carrying material incomplete mass is a
    # failed run, not a forecast; where the threshold sits is the caller's decision.
    #
    # The two numbers are different questions and a gate wants both. ``..._mass`` is
    # "how much of this world did we fail to run" — a reliability statement, true even
    # of a branch whose answer is nonetheless known because a monotone terminal banked
    # it. ``..._unresolved_mass`` is the subset that has no answer AND has none because
    # of us: mass that today flows into bounds and suppression dressed as uncertainty
    # about reality, when it is nothing of the kind.
    execution_incomplete_mass: float = 0.0
    execution_incomplete_unresolved_mass: float = 0.0
    execution_incomplete_branches: tuple[tuple[str, str], ...] = ()


@dataclass
class _BranchRun:
    """One branch's complete result, produced independently of every other branch."""

    scenario: Scenario
    world: WorldState
    ledger: list[Event]
    decisions: list[ActorDecisionRecord]
    diagnostics: BranchDiagnostics
    failure: str = ""
    # The terminal's answer for the initialized world, before anything simulated ran.
    pre_resolved: bool = False
    pre_outcome: str | None = None


def run(
    compiled: CompiledWorld,
    gateway: ModelGateway,
    *,
    seed: int,
    budget: RunBudget | None = None,
    max_concurrent_branches: int = 4,
) -> RunResult:
    """Simulate every branch of the compiled world.

    Branches are genuinely independent possible worlds — they share no state, and each
    carries its own clock, its own actors and its own effect sequence — so they are run
    concurrently. This is a wall-clock decision only: nothing about a trajectory depends
    on which other branches were running at the time, and results are assembled in
    scenario order so the output is identical either way.

    A caller that does not pin a ``budget`` gets one sized to this world's roster and
    window (:meth:`RunBudget.for_world`) rather than a flat constant. A pinned budget is
    honoured exactly as given.
    """

    budget = budget or RunBudget.for_world(
        participants=len(compiled.base_world.actors),
        horizon_days=(
            compiled.base_world.contract.horizon - compiled.base_world.contract.as_of
        ).total_seconds()
        / 86400.0,
    )
    scenarios = list(compiled.scenario_set.scenarios)
    workers = max(1, min(max_concurrent_branches, len(scenarios)))

    def run_one(scenario: Scenario) -> _BranchRun:
        # Per-branch executor: the effect sequence must not interleave across branches,
        # or two independent worlds would produce each other's event ids.
        effects = EffectExecutor()
        action_exec = ActionExecutor(gateway, effects)
        actor_runtime = ActorRuntime(gateway)
        ledger: list[Event] = []
        decisions: list[ActorDecisionRecord] = []
        diag = BranchDiagnostics()
        weight = BranchWeight(scenario.weight, scenario.provenance, scenario.provenance_detail)
        world = compiled.base_world.clone(new_branch_id=scenario.scenario_id, weight=weight)
        pre_resolved = False
        pre_outcome: str | None = None
        try:
            world = _seed_branch(world, compiled.spec, scenario, effects, ledger)
            # PRE-SIMULATION OUTCOME. This is the exact point where the branch world is
            # initialized but nothing has run. ``_seed_branch`` has (a) pushed the
            # compiled calendar — process entry nodes, external occurrences, plan and
            # commitment entries — onto the schedule *without firing any of it* (those
            # only execute inside ``_event_loop``), and (b) established this branch's
            # hypothesis about its uncertain values: a standing condition (no known
            # release date, or one already public at the cutoff) is applied to world
            # state right there, while every value whose public release lies in the
            # future is only *scheduled*, each at its own moment, so the branch state
            # honestly lacks it until its release fires inside the loop (TMP-4). This
            # evaluation must still answer "what does the terminal say under this
            # branch's conditions if nothing further happens before the horizon?", so
            # the deferred releases are overlaid onto a throwaway copy for this one
            # evaluation — the world the loop runs is untouched and still learns each
            # value only at its own release time. The copy's clock is moved to the
            # horizon so the evaluation answers the same question ``_finalize`` will
            # answer instead of tripping time-window guards at ``as_of``.
            pre_world = world
            if scenario.dated_releases(world.contract.as_of):
                pre_world = replace(
                    world,
                    fields=tuple(
                        sorted({**dict(world.fields), **dict(scenario.field_levels)}.items())
                    ),
                )
            pre_eval = evaluate_terminal(
                pre_world.with_time(world.contract.horizon), compiled.spec.terminal
            )
            pre_resolved = pre_eval.resolved
            pre_outcome = pre_eval.outcome if pre_eval.resolved else None
            world = _event_loop(
                world,
                compiled.spec,
                effects,
                action_exec,
                actor_runtime,
                seed,
                decisions,
                ledger,
                budget,
                diag,
            )
            world = _finalize(world, compiled.spec.terminal, effects, ledger, diag)
        except GatewayError as exc:
            diag.stop_reason = f"provider_failure: {exc}"
            # A branch we could not finish because the provider was unreachable is the
            # same statement as a branch we could not finish because we ran out of
            # calls: the world did not fail to determine its answer, we failed to ask.
            diag.execution_status = EXECUTION_INCOMPLETE
            diag.execution_limit = "provider unreachable"
            return _BranchRun(
                scenario,
                world,
                ledger,
                decisions,
                diag,
                f"provider_failure: {exc}",
                pre_resolved=pre_resolved,
                pre_outcome=pre_outcome,
            )
        return _BranchRun(
            scenario,
            world,
            ledger,
            decisions,
            diag,
            pre_resolved=pre_resolved,
            pre_outcome=pre_outcome,
        )

    if workers == 1:
        runs = [run_one(s) for s in scenarios]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(run_one, scenarios))

    branch_outcomes: list[BranchOutcome] = []
    summaries: list[TrajectorySummary] = []
    ledger: list[Event] = []
    decisions: list[ActorDecisionRecord] = []
    final_worlds: dict[str, WorldState] = {}
    diagnostics: dict[str, BranchDiagnostics] = {}

    for br in runs:  # scenario order, so the assembled result is deterministic
        diagnostics[br.scenario.scenario_id] = br.diagnostics
        final_worlds[br.scenario.scenario_id] = br.world
        ledger.extend(br.ledger)
        decisions.extend(br.decisions)
        if br.failure:
            branch_outcomes.append(
                _unresolved_outcome(
                    br.scenario,
                    br.failure,
                    pre_resolved=br.pre_resolved,
                    pre_outcome=br.pre_outcome,
                )
            )
            summaries.append(_unresolved_summary(br.scenario, br.failure))
        else:
            branch_outcomes.append(
                _branch_outcome(
                    br.world,
                    br.scenario,
                    pre_resolved=br.pre_resolved,
                    pre_outcome=br.pre_outcome,
                    diag=br.diagnostics,
                )
            )
            summaries.append(_summary(br.world, br.scenario))

    # W2 — the two kinds of "no answer" are priced apart here, once, from the branch
    # table and its diagnostics. ``unresolved_class`` is per-branch and is the seam
    # aggregation reads; ``execution_incomplete_mass`` is the run-level fact that makes
    # a run with too much of it a failed run rather than a wide forecast.
    incomplete: list[tuple[str, str]] = []
    incomplete_mass = 0.0
    incomplete_unresolved_mass = 0.0
    for br, outcome in zip(runs, branch_outcomes, strict=True):
        diag = br.diagnostics
        if diag.execution_status == EXECUTION_INCOMPLETE:
            incomplete.append((br.scenario.scenario_id, diag.execution_limit or diag.stop_reason))
            incomplete_mass += br.scenario.weight
        if outcome.resolved:
            diag.unresolved_class = ""
        elif diag.execution_status == EXECUTION_INCOMPLETE:
            diag.unresolved_class = EXECUTION_INCOMPLETE
            incomplete_unresolved_mass += br.scenario.weight
        else:
            diag.unresolved_class = UNRESOLVED_WORLD_UNDETERMINED

    return RunResult(
        branch_outcomes=tuple(branch_outcomes),
        trajectory_summaries=tuple(summaries),
        event_ledger=ledger,
        actor_decisions=decisions,
        final_worlds=final_worlds,
        truncated_mass=compiled.scenario_set.truncated_mass,
        truncated_reason=compiled.scenario_set.truncated_reason,
        diagnostics=diagnostics,
        execution_incomplete_mass=incomplete_mass,
        execution_incomplete_unresolved_mass=incomplete_unresolved_mass,
        execution_incomplete_branches=tuple(incomplete),
    )


# ---------------------------------------------------------------------------
# Seeding the branch calendar
# ---------------------------------------------------------------------------


def _seed_branch(
    world: WorldState,
    spec: WorldSpec,
    scenario: Scenario,
    effects: EffectExecutor,
    ledger: list[Event],
) -> WorldState:
    """Put the compiled world's *own* calendar into the queue, plus this branch's
    hypothesis about the uncertain future.

    The queue is seeded from three real sources — the compiled process's entry nodes,
    the compiled external (non-agent) processes, and the actors' grounded initial plan
    steps and commitments. Nothing is scheduled for the sake of giving anyone a turn.
    """

    entries: list[ScheduledEntry] = []
    as_of = world.contract.as_of

    for node in _entry_nodes(spec):
        at = _node_time(node, as_of)
        entries.append(
            make_entry(
                at=at,
                kind=KIND_PROCESS_NODE,
                payload={"node_id": node.node_id},
                origin=ORIGIN_PROCESS,
                origin_detail=node.node_id,
            )
        )

    for proc in spec.external_processes:
        for i, occ in enumerate(proc.occurrences):
            at = _parse_dt(occ.at) or as_of
            entries.append(
                make_entry(
                    at=at,
                    kind=KIND_EXTERNAL,
                    payload={"process_id": proc.process_id, "index": i},
                    origin=ORIGIN_EXTERNAL,
                    origin_detail=proc.process_id,
                )
            )

    for actor in world.actors.values():
        entries.extend(_plan_entries(actor))
        entries.extend(_commitment_entries(actor))

    world = world.with_schedule(world.schedule.push(*entries))

    # This branch's hypothesis about its uncertain future values — ONE SCHEDULED ENTRY
    # PER RELEASE MOMENT, never one merged announcement for the whole branch. Where the
    # compiler knows when a value becomes public, that value is released THEN, as an
    # entry the event loop fires at its real time, so the branch state before it
    # honestly lacks the value and everything compiled in between still happens first
    # (TMP-4/FD-7: the BoE branch collapsed every dated release to t0 and decided the
    # outcome at the first invocation). Merging a branch's releases into one moment
    # reproduced that defect on the merged side: two uncertainties dated nineteen days
    # apart both fired on the earlier date, and a value whose timing the evidence never
    # established was published on its neighbour's date. A value with no known release
    # date — or one already public at the cutoff — is a standing condition of the
    # branch from the start. It is never dropped at an invented midpoint of the
    # forecast window just to give the world something to react to.
    standing = scenario.standing_levels(as_of)
    if standing:
        ev = effects.raw_event(
            world,
            kind="release_data",
            actor_id=None,
            payload={
                "fields": dict(standing),
                "epistemic_type": "hypothesis",
                "branch_conditions": dict(scenario.conditions),
            },
            visibility=Visibility.PUBLIC,
        )
        world = world.apply([ev])
        applied = world.event_history[-1]
        ledger.append(applied)
        world = _propagate(world, spec, [applied])
    for rel in scenario.dated_releases(as_of):
        assert rel.at is not None  # dated_releases filters on it
        world = world.with_schedule(
            world.schedule.push(
                make_entry(
                    at=rel.at,
                    # The dedicated schedule kind: when the compiled world models the
                    # release event itself — as an occurrence at the same instant OR as
                    # a deferred at-stamped placeholder — this branch's hypothesis
                    # states what that release *revealed*, so it must land strictly
                    # after everything else scheduled at its instant. The schedule's
                    # ordering class guarantees that structurally; a plain deferred
                    # entry used to tie with the placeholder and the winner fell to
                    # entry-id hash order.
                    kind=KIND_SCENARIO_RELEASE,
                    payload={
                        "fields": dict(rel.field_levels),
                        "branch_conditions": dict(rel.conditions),
                    },
                    origin=ORIGIN_EXTERNAL,
                    origin_detail=f"scenario_release:{scenario.scenario_id}@{rel.at.isoformat()}",
                )
            )
        )
    return world


def _entry_nodes(spec: WorldSpec) -> tuple[ProcessNode, ...]:
    """The nodes the branch calendar is seeded with.

    A node is seeded unless something else enters it. A node whose ``after_node`` names
    a node that does not exist is *not* entered by anything, so seeding it is what keeps
    a compiled step from vanishing between compilation and simulation because of one bad
    reference.
    """

    known = {n.node_id for n in spec.process.nodes}
    entered = {nid for n in spec.process.nodes for nid in n.next_nodes if nid in known}
    return tuple(
        n
        for n in spec.process.nodes
        if n.node_id not in entered and (not n.after_node or n.after_node not in known)
    )


def _plan_entries(actor: ActorState) -> list[ScheduledEntry]:
    """An actor's own planned future steps are real future causes: they put an
    opportunity on the calendar rather than waiting for the world to poke the actor."""

    out: list[ScheduledEntry] = []
    if actor.plan is None or actor.plan.status != PLAN_ACTIVE:
        return out
    for step in actor.plan.steps:
        if step.at is None or step.status in ("done", "abandoned"):
            continue
        out.append(
            make_entry(
                at=step.at,
                kind=KIND_PLAN_STEP,
                actor_id=actor.actor_id,
                payload={"plan_id": actor.plan.plan_id, "step": step.description},
                origin=ORIGIN_ACTOR_PLAN,
                origin_detail=f"{actor.actor_id}:{actor.plan.plan_id}",
            )
        )
    return out


def _commitment_entries(actor: ActorState) -> list[ScheduledEntry]:
    out: list[ScheduledEntry] = []
    for c in actor.open_commitments():
        if c.due is None:
            continue
        out.append(
            make_entry(
                at=c.due,
                kind=KIND_COMMITMENT_DUE,
                actor_id=actor.actor_id,
                payload={"commitment": c.text},
                origin=ORIGIN_ACTOR_PLAN,
                origin_detail=f"{actor.actor_id}:commitment",
            )
        )
    return out


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _event_loop(
    world: WorldState,
    spec: WorldSpec,
    effects: EffectExecutor,
    action_exec: ActionExecutor,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
    budget: RunBudget,
    diag: BranchDiagnostics,
) -> WorldState:
    horizon = world.contract.horizon
    stale_batches = 0
    last_digest = (world.state_digest(), world.time, world.information_digest())
    last_queued_here = world.schedule.pending_at(world.time)

    # W1. The terminal is watched CONTINUOUSLY, not once at the end. The structural
    # analysis runs once here — it depends only on the compiled expression — so a
    # terminal that can never be banked costs the loop nothing at all.
    watch = _bankable(spec.terminal)
    _bank_monotone_terminal(world, spec.terminal, watch, diag)

    while True:
        if diag.batches >= budget.max_batches:
            _stop_on_budget(diag, f"batch budget exhausted ({budget.max_batches})", "max_batches")
            break
        if diag.events >= budget.max_events:
            _stop_on_budget(diag, f"event budget exhausted ({budget.max_events})", "max_events")
            break
        if sum(diag.actor_call_counts.values()) >= budget.max_actor_calls:
            _stop_on_budget(
                diag,
                f"actor-call budget exhausted ({budget.max_actor_calls})",
                "max_actor_calls",
            )
            break

        schedule, batch = world.schedule.pop_batch(horizon=horizon)
        if not batch:
            diag.stop_reason = "schedule exhausted"
            break
        world = world.with_schedule(schedule)
        world = world.with_time(batch[0].at)
        diag.batches += 1

        produced: list[Event] = []
        for entry in _merge_decisions(batch):
            world, evs = _dispatch(
                world,
                spec,
                entry,
                effects,
                action_exec,
                actor_runtime,
                seed,
                decisions,
                ledger,
                budget,
                diag,
            )
            produced.extend(evs)
        diag.events += len(produced)

        # The moment the world satisfies a monotone terminal, that fact is banked. It
        # is checked here — after the batch has been applied — so the instant recorded
        # is the instant the world reached it, not the instant we happened to stop.
        _bank_monotone_terminal(world, spec.terminal, watch, diag)

        # No progress means the world stops *changing*, not that it stops emitting.
        #
        # The old condition also required `not produced`, and a cascade always produces
        # something — that is what makes it a cascade. Measured on the Bank of England
        # run: 798 batches, 400 actor calls, and exactly ONE distinct world digest
        # throughout. The guard armed 398 times and its longest consecutive run was 1,
        # because every other batch emitted an event that reset it. It could not fire.
        #
        # Comparing the digest alone is the whole fix, and it must be the digest rather
        # than the event count: honest runs do reach streaks of nine or ten batches that
        # emit nothing while time advances, and they are distinguished by the state
        # having changed, not by their silence.
        # Progress has three faces, and a run is stale only when none of them moves:
        # the world state changed, the clock advanced, or somebody learned something
        # they had not already been told. The third was missing, and without it a world
        # whose actors correspond without writing world state looks frozen — three
        # rounds of ordinary pre-meeting correspondence were enough to kill a branch
        # before it reached its own scheduled session. Counting deliveries would undo
        # the cascade fix; keying on their *content* does not, because the cascade's
        # defining property is that it delivers the same thing four hundred times.
        # A finite burst of simultaneous work is not a stall. Everyone reading the
        # notes that were just circulated produces batch after batch that changes no
        # world state and teaches nobody anything new — and it *drains*, strictly, until
        # the clock moves on to the next real event. A cascade does the opposite: every
        # item it handles schedules another at the same instant, so the queue at that
        # instant holds or grows. Requiring the queue to have stopped draining is what
        # separates them, and without it the guard killed branches four rounds of
        # ordinary correspondence before their own scheduled session.
        queued_here = world.schedule.pending_at(world.time)
        digest = (world.state_digest(), world.time, world.information_digest())
        draining = queued_here < last_queued_here
        last_queued_here = queued_here
        if digest == last_digest and not draining:
            stale_batches += 1
            if stale_batches >= budget.no_progress_batches:
                _stop_on_budget(
                    diag,
                    f"no progress: {stale_batches} consecutive batches left the world "
                    f"state unchanged ({len(produced)} event(s) in the last batch, none "
                    "of which altered anything the world records)",
                    "no_progress_batches",
                )
                break
        else:
            stale_batches = 0
            last_digest = digest

    diag.pending_beyond_horizon = [
        e.as_dict() for e in world.schedule.beyond_horizon(horizon=horizon)[:50]
    ]
    diag.unfired_in_horizon = world.schedule.pending_count(horizon=horizon)
    if diag.unfired_in_horizon > 0 and diag.execution_status != EXECUTION_INCOMPLETE:
        # Leaving in-horizon work unfired is the definition of not finishing, whatever
        # the loop believed its reason for stopping was.
        _stop_on_budget(diag, diag.stop_reason, "unfired in-horizon entries")
    return world


def _stop_on_budget(diag: BranchDiagnostics, reason: str, limit: str) -> None:
    """Record a stop that WE imposed.

    Exhaustion used to be one more ``stop_reason`` string among several, indistinguishable
    at a glance from the schedule running out — which is the one stop that means the
    branch actually finished. The two say opposite things about the run, and only one of
    them is about the world, so they are recorded as different things here rather than
    left for a reader to tell apart by reading prose.
    """

    diag.stop_reason = reason
    diag.execution_status = EXECUTION_INCOMPLETE
    diag.execution_limit = limit


def _merge_decisions(batch: tuple[ScheduledEntry, ...]) -> tuple[ScheduledEntry, ...]:
    """Collapse simultaneous wake-ups of the same actor into one invocation.

    Two independent things can reach the same person at the same instant. That is one
    moment of their attention, not two: they are told both reasons and decide once.
    """

    merged: dict[tuple[str, str], ScheduledEntry] = {}
    out: list[ScheduledEntry] = []
    for entry in batch:
        if entry.kind != KIND_DECISION or not entry.actor_id:
            out.append(entry)
            continue
        key = (entry.actor_id, entry.at.isoformat())
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            continue
        p, q = existing.payload_dict, entry.payload_dict
        combined = dict(p)
        combined["wake_reason"] = f"{p.get('wake_reason')}+{q.get('wake_reason')}"
        combined["wake_detail"] = f"{p.get('wake_detail')}; {q.get('wake_detail')}"
        combined["observation_ids"] = list(
            dict.fromkeys(
                list(p.get("observation_ids") or []) + list(q.get("observation_ids") or [])
            )
        )
        combined["node_id"] = p.get("node_id") or q.get("node_id") or ""
        merged[key] = make_entry(
            at=entry.at,
            kind=KIND_DECISION,
            actor_id=entry.actor_id,
            payload=combined,
            origin=existing.origin,
            origin_detail=existing.origin_detail,
            causal_parents=tuple(dict.fromkeys(existing.causal_parents + entry.causal_parents)),
            microstep=min(existing.microstep, entry.microstep),
        )
    out.extend(merged.values())
    return tuple(sorted(out, key=lambda e: e.sort_key()))


def _dispatch(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    effects: EffectExecutor,
    action_exec: ActionExecutor,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
    budget: RunBudget,
    diag: BranchDiagnostics,
) -> tuple[WorldState, list[Event]]:
    kind = entry.kind
    if kind == KIND_PROCESS_NODE:
        return _fire_process_node(world, spec, entry, effects, ledger)
    if kind == KIND_EXTERNAL:
        return _fire_external(world, spec, entry, effects, ledger)
    if kind == KIND_DEFERRED_EFFECT:
        return _fire_deferred(world, spec, entry, effects, ledger)
    if kind == KIND_SCENARIO_RELEASE:
        return _fire_scenario_release(world, spec, entry, effects, ledger)
    if kind == KIND_ACTION_COMPLETION:
        return _complete_action(world, spec, entry, action_exec, ledger)
    if kind == KIND_NOTICE:
        return _notice(world, spec, entry)
    if kind == KIND_DECISION:
        return _invoke_actor(
            world,
            spec,
            entry,
            effects,
            action_exec,
            actor_runtime,
            seed,
            decisions,
            ledger,
            budget,
            diag,
        )
    if kind in (
        KIND_COMMITMENT_DUE,
        KIND_REVISIT,
        KIND_NEED_DEADLINE,
        KIND_PLAN_STEP,
        KIND_DEADLINE,
    ):
        return _wake_from_entry(world, spec, entry, action_exec)
    return world, []


# -- world-side entries ------------------------------------------------------


def _fire_process_node(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    effects: EffectExecutor,
    ledger: list[Event],
) -> tuple[WorldState, list[Event]]:
    node = spec.process.node(str(entry.payload_dict.get("node_id", "")))
    if node is None:
        return world, []
    if not bool(evaluate(node.entry_condition, world)):
        # The node's moment came and its precondition was not met. That is a real
        # outcome, not a reason to retry until it is.
        ev = effects.raw_event(
            world,
            kind="create_event",
            actor_id=None,
            payload={
                "event_type": "process_node_skipped",
                "text": f"{node.node_id}: entry condition not met",
            },
            visibility=Visibility.PUBLIC,
        )
        world = world.apply([ev])
        ledger.append(world.event_history[-1])
        return world, [world.event_history[-1]]

    if node.stage:
        world = world.with_stage(node.stage)

    produced: list[Event] = []
    follow: list[ScheduledEntry] = []
    if node.effects:
        env_events, deferred = effects.build_events(
            world, node.effects, {"actor": None, "self": None}
        )
        world = world.apply(env_events)
        for ev in env_events:
            applied = _find(world, ev.event_id)
            ledger.append(applied)
            produced.append(applied)
        world = _propagate(world, spec, produced, microstep=entry.microstep)
        follow.extend(
            _deferred_entries(
                deferred,
                origin=ORIGIN_PROCESS,
                detail=node.node_id,
                parents=tuple(e.event_id for e in produced),
                microstep=entry.microstep + 1,
            )
        )
    for nxt_id in node.next_nodes:
        nxt = spec.process.node(nxt_id)
        if nxt is None:
            continue
        at = _node_time(nxt, world.time, after=world.time)
        follow.append(
            make_entry(
                at=at,
                kind=KIND_PROCESS_NODE,
                payload={"node_id": nxt.node_id},
                origin=ORIGIN_PROCESS,
                origin_detail=f"{node.node_id}->{nxt.node_id}",
                causal_parents=tuple(e.event_id for e in produced),
                microstep=entry.microstep + 1,
            )
        )
    deadline = _parse_dt(node.deadline)
    if deadline is not None:
        for aid in _participants(world, spec, node):
            follow.append(
                make_entry(
                    at=deadline,
                    kind=KIND_DEADLINE,
                    actor_id=aid,
                    payload={"node_id": node.node_id, "detail": f"deadline for {node.node_id}"},
                    origin=ORIGIN_PROCESS,
                    origin_detail=f"{node.node_id}:deadline",
                )
            )

    # Participants gain an *opportunity*: a reason to be woken, not a scheduled turn.
    for aid in _participants(world, spec, node):
        follow.append(
            _decision_entry(
                at=world.time,
                actor_id=aid,
                reason=WAKE_OPPORTUNITY,
                detail=f"{node.node_id}: {node.description or node.stage or 'opportunity to act'}",
                node_id=node.node_id,
                causal_parents=tuple(e.event_id for e in produced),
                # +2, not +1: whatever this node delivered is noticed at +1, so the
                # actor arrives at its opportunity already knowing what just happened
                # rather than being asked to act on a world it has not seen.
                microstep=entry.microstep + 2,
            )
        )
    if follow:
        world = world.with_schedule(world.schedule.push(*follow))
    return world, produced


def _deferred_entries(
    deferred: list[tuple[datetime, Effect]],
    *,
    origin: str,
    detail: str,
    parents: tuple[str, ...] = (),
    microstep: int = 1,
) -> list[ScheduledEntry]:
    """Turn effects stamped in the future into real entries on the branch calendar.

    This is what makes ``schedule_event`` schedule. Before, a future-stamped effect was
    applied on the spot and pulled the clock along with it, so everything genuinely due
    in between was skipped.
    """

    return [
        make_entry(
            at=when,
            kind=KIND_DEFERRED_EFFECT,
            payload={"op": eff.op, "params": dict(eff.params)},
            origin=origin,
            origin_detail=detail,
            causal_parents=parents,
            microstep=microstep,
        )
        for when, eff in deferred
    ]


def _fire_deferred(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    effects: EffectExecutor,
    ledger: list[Event],
) -> tuple[WorldState, list[Event]]:
    """Apply an effect whose scheduled moment has now arrived."""

    p = entry.payload_dict
    eff = Effect(op=str(p.get("op", "")), params=tuple(sorted(dict(p.get("params") or {}).items())))
    if eff.op not in UNIVERSAL_OPS:
        return world, []
    # `at`/`after_seconds` already fired by being scheduled; strip them so the effect
    # lands now rather than re-deferring itself forever.
    eff = Effect(
        op=eff.op,
        params=tuple((k, v) for k, v in eff.params if k not in ("at", "after_seconds")),
    )
    evs, _ = effects.build_events(world, (eff,), {"actor": entry.actor_id, "self": None})
    world = world.apply(evs)
    produced = [_find(world, e.event_id) for e in evs]
    for ev in produced:
        ledger.append(ev)
    return _propagate(world, spec, produced, microstep=entry.microstep), produced


def _fire_scenario_release(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    effects: EffectExecutor,
    ledger: list[Event],
) -> tuple[WorldState, list[Event]]:
    """Publish one part of this branch's hypothesis, at its own release moment.

    Deliberately not routed through ``_fire_deferred``: a scenario release is branch
    *construction* arriving on time, not a compiled process moving the world, and the
    event it writes says so. Carrying the same ``branch_conditions`` /
    ``epistemic_type`` stamp the standing seed carries is what lets every downstream
    reader — the §9 temporal report, the replay core's scenario-condition and
    branch-initial-state reconstruction — tell a branch hypothesis from a process
    update. Without it a run reported one phantom "process update" per branch, and the
    forensic replay could not name the conditions of a branch whose hypothesis was
    dated. ``deferred_release`` distinguishes the two: this value was NOT part of the
    branch's initial state; the branch learned it here.
    """

    p = entry.payload_dict
    fields = dict(p.get("fields") or {})
    if not fields:
        return world, []
    ev = effects.raw_event(
        world,
        kind="release_data",
        actor_id=None,
        payload={
            "fields": fields,
            "epistemic_type": "hypothesis",
            "branch_conditions": dict(p.get("branch_conditions") or {}),
            "deferred_release": True,
        },
        visibility=Visibility.PUBLIC,
    )
    world = world.apply([ev])
    applied = world.event_history[-1]
    ledger.append(applied)
    world = _propagate(world, spec, [applied], microstep=entry.microstep)
    return world, [applied]


def _fire_external(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    effects: EffectExecutor,
    ledger: list[Event],
) -> tuple[WorldState, list[Event]]:
    p = entry.payload_dict
    proc = next(
        (x for x in spec.external_processes if x.process_id == str(p.get("process_id"))), None
    )
    if proc is None:
        return world, []
    idx = int(p.get("index", 0))
    if idx >= len(proc.occurrences):
        return world, []
    occ = proc.occurrences[idx]
    if not bool(evaluate(occ.condition, world)):
        return world, []
    produced: list[Event] = []
    if occ.effects:
        evs, deferred = effects.build_events(world, occ.effects, {"actor": None, "self": None})
        world = world.apply(evs)
        for ev in evs:
            applied = _find(world, ev.event_id)
            ledger.append(applied)
            produced.append(applied)
        world = _propagate(world, spec, produced, microstep=entry.microstep)
        if deferred:
            world = world.with_schedule(
                world.schedule.push(
                    *_deferred_entries(
                        deferred,
                        origin=ORIGIN_EXTERNAL,
                        detail=proc.process_id,
                        parents=tuple(e.event_id for e in produced),
                        microstep=entry.microstep + 1,
                    )
                )
            )
    return world, produced


def _complete_action(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    action_exec: ActionExecutor,
    ledger: list[Event],
) -> tuple[WorldState, list[Event]]:
    """Finish an in-flight action. Its effects land now, in the world as it is now."""

    aid = str(entry.actor_id)
    actor = world.actors.get(aid)
    outcome = action_exec.complete(world, entry, spec)
    world = world.apply(outcome.events)
    produced = [_find(world, e.event_id) for e in outcome.events]
    for ev in produced:
        ledger.append(ev)
    world = _propagate(
        world, spec, produced, source_action_id=outcome.action_id, microstep=entry.microstep
    )
    if outcome.deferred:
        world = world.with_schedule(
            world.schedule.push(
                *_deferred_entries(
                    outcome.deferred,
                    origin=ORIGIN_CONSEQUENCE,
                    detail=f"action:{outcome.action_id}",
                    parents=tuple(e.event_id for e in produced),
                    microstep=entry.microstep + 1,
                )
            )
        )

    if actor is not None:
        status = "completed" if outcome.status == "executed" else "failed"
        ongoing = actor.current_action
        updated = (
            replace(ongoing, status=status)
            if ongoing is not None and ongoing.action_id == outcome.action_id
            else None
        )
        world = world.with_actor(replace(world.actors[aid], current_action=updated))
        if status == "completed":
            # Acting is not succeeding, so succeeding gets its own ledger line: the
            # completion instant is a real recorded moment (the §9 temporal report
            # counts completions from it, next to started/failed/rejected). It is
            # private bookkeeping and NOT an observable kind — a success is not news,
            # and it wakes nobody (see below).
            done = action_exec.effects.raw_event(
                world,
                kind="action_completed",
                actor_id=aid,
                payload={
                    "action_id": outcome.action_id,
                    "started_at": str(entry.payload_dict.get("started_at", "")),
                    "reason": outcome.reason,
                },
                visibility=Visibility.PRIVATE,
                audience=(aid,),
            )
            world = world.apply([done])
            applied_done = world.event_history[-1]
            ledger.append(applied_done)
            produced.append(applied_done)
        if status == "failed":
            # A failure is news the actor needs: what it set out to do did not happen,
            # and it may now do something else. A *success* is not news — the actor
            # already knows it acted, and waking it here would make every completed
            # action immediately prompt another one, which is a treadmill, not a life.
            # After a success the actor comes back only when its own plan, a commitment,
            # or something in the world brings it back.
            world = world.with_schedule(
                world.schedule.push(
                    _decision_entry(
                        at=world.time,
                        actor_id=aid,
                        reason=WAKE_OWN_ACTION,
                        detail=f"your action {outcome.action_id!r} failed: {outcome.reason}",
                        causal_parents=tuple(e.event_id for e in produced),
                        microstep=entry.microstep + 1,
                    )
                )
            )
    return world, produced


def _notice(
    world: WorldState, spec: WorldSpec, entry: ScheduledEntry
) -> tuple[WorldState, list[Event]]:
    """An actor's attention arrives: everything available to it by now becomes noticed.

    Noticing is where information enters the actor's world — not delivery. Whether it
    then *acts* is a separate question answered below, and the answer is usually no.
    """

    aid = str(entry.actor_id)
    actor = world.actors.get(aid)
    if actor is None:
        return world, []
    available = world.available_unnoticed(aid, by=world.time)
    if not available:
        return world, []
    ids = frozenset(d.event_id for d in available)
    # The world records that the actor noticed these; the actor records that it has
    # folded them into memory, which happens when it is next invoked. Setting the
    # actor-side marker here would make the information vanish before it was ever put
    # in front of the actor.
    world = world.mark_noticed(aid, ids, world.time)
    world = world.with_actor(
        replace(
            world.actors[aid],
            available_event_ids=world.actors[aid].available_event_ids - ids,
        )
    )

    reason, detail = _relevance(world, spec, world.actors[aid], ids)
    if reason:
        world = world.with_schedule(
            world.schedule.push(
                _decision_entry(
                    at=world.time,
                    actor_id=aid,
                    reason=reason,
                    detail=detail,
                    observation_ids=tuple(sorted(ids)),
                    causal_parents=tuple(sorted(ids)),
                    microstep=entry.microstep + 1,
                )
            )
        )
    return world, []


def _wake_from_entry(
    world: WorldState, spec: WorldSpec, entry: ScheduledEntry, action_exec: ActionExecutor
) -> tuple[WorldState, list[Event]]:
    """A time-based cause fires: a commitment falls due, a self-set revisit arrives, a
    requested answer definitively failed to come, a planned step's moment arrives, or a
    real deadline is reached."""

    aid = str(entry.actor_id or "")
    actor = world.actors.get(aid)
    if actor is None:
        return world, []
    p = entry.payload_dict
    mapping = {
        KIND_COMMITMENT_DUE: (
            WAKE_COMMITMENT,
            f"your commitment is due: {p.get('commitment', '')}",
        ),
        KIND_REVISIT: (WAKE_REVISIT, f"the condition you set has arrived: {p.get('detail', '')}"),
        KIND_NEED_DEADLINE: (
            WAKE_NEED_FAILED,
            f"you are still waiting on: {p.get('question', '')} — it has not arrived",
        ),
        KIND_PLAN_STEP: (WAKE_PLAN_STEP, f"your planned step is due: {p.get('step', '')}"),
        KIND_DEADLINE: (WAKE_DEADLINE, str(p.get("detail", "a deadline has been reached"))),
    }
    reason, detail = mapping[entry.kind]

    if entry.kind == KIND_NEED_DEADLINE:
        # Only wake for a need that is genuinely still open.
        q = str(p.get("question", ""))
        if not any(n.question == q and not n.resolved for n in actor.pending_needs):
            return world, []

    world = world.with_schedule(
        world.schedule.push(
            _decision_entry(
                at=world.time,
                actor_id=aid,
                reason=reason,
                detail=detail,
                causal_parents=entry.causal_parents,
                microstep=entry.microstep + 1,
            )
        )
    )
    return world, []


# -- the actor invocation ----------------------------------------------------

# The validation_status recorded when a wake finds nothing the actor could do. It is a
# statement about the world's offer, not about a decision — no model was called.
NO_FEASIBLE_ACTION = "no_feasible_action"


def _no_feasible_action_record(
    world: WorldState,
    spec: WorldSpec,
    node: ProcessNode | None,
    actor: ActorState,
    action_exec: ActionExecutor,
    entry: ScheduledEntry,
) -> ActorDecisionRecord:
    """The record of a wake at which nothing was feasible and nothing novel allowed.

    Without it, the turn vanished: actor_decisions.jsonl showed no trace of the wake,
    and a world in which the only offered action's authority token mismatched looked
    inert for no stated reason. The record names each offered action and the exact
    reason it was infeasible, using the same availability check that refused them.
    """

    p = entry.payload_dict
    offered = node.action_ids if node is not None else ("*",)
    candidates = [
        a for a in spec.actions if offered == ("*",) or not offered or a.action_id in offered
    ]
    # The private check is the SAME one feasible_actions used to exclude these
    # actions; asking it again is what makes the recorded reason the true reason.
    reasons = [
        f"{a.action_id}: {action_exec._check_availability(world, actor, a)[1]}" for a in candidates
    ]
    why = (
        "; ".join(reasons)
        if reasons
        else "this node offers no actions at all, and novel actions are not allowed"
    )
    state = actor.state_dict()
    return ActorDecisionRecord(
        branch_id=world.branch_id,
        actor_id=actor.actor_id,
        branch_time=world.time.isoformat(),
        stage=world.stage,
        wake_reason=str(p.get("wake_reason", "")),
        wake_detail=str(p.get("wake_detail", "")),
        trigger_event_ids=list(entry.causal_parents),
        delivered_observation_ids=[
            d.event_id for d in world.deliveries if d.actor_id == actor.actor_id
        ],
        noticed_observation_ids=[],
        retrieved_memory_ids=[],
        plan_before=None,
        plan_after=None,
        plan_disposition="not consulted: the wake offered nothing to decide",
        state_before=state,
        state_after=state,
        decision_context={},
        intent={},
        validation_status=NO_FEASIBLE_ACTION,
        validation_reason=f"no feasible action and novel actions not allowed here — {why}",
        event_ids=[],
        world_version_at_decision=world.version,
        prompt_hash="",
        model="",
        tokens_out=0,
    )


# An intention the environment accepted and put into the world, as opposed to a wait,
# a refusal or a failure. Only these mean the actor has actually taken the opportunity.
_ACTED_STATUSES = frozenset({"started", "executed"})


def _cancel_answered_deadline(
    world: WorldState,
    entry: ScheduledEntry,
    reason: str,
    status: str,
    before: ActorState,
    after: ActorState,
) -> WorldState:
    """Drop a node's deadline wake for a participant that already exercised it.

    Firing a process node queues two things per participant: the *opportunity* to act
    now, and a wake when the node's deadline arrives. A deadline exists to catch the
    participant who has not yet done the thing. Once an actor has done it — its
    intention was accepted and is in the world — the deadline wake asks the same actor
    the same question about the same node on the same evidence, and a live run's audit
    duly flagged exactly that: a second ``deadline_reached`` invocation the same day,
    after the actor had already acted, with no state change between the two.

    Deliberately narrow. An actor that *declined* the opportunity keeps its deadline: a
    closing window is real news to someone who has not acted, and "it is now or never"
    is a cause to reconsider, not a repeat. An actor that asked to be brought back keeps
    it too — its own revisit condition is honoured over this cancellation. And nothing
    information-driven is ever cancelled here: a message, a directed answer or a
    compiled wake rule reaches it regardless. Only the redundant calendar copy of an
    opportunity already taken is removed.
    """

    if status not in _ACTED_STATUSES:
        return world
    if WAKE_OPPORTUNITY not in reason.split("+"):
        return world
    node_id = str(entry.payload_dict.get("node_id", ""))
    if not node_id:
        return world
    if after.revisit_conditions != before.revisit_conditions:
        return world
    aid = after.actor_id
    return world.with_schedule(
        world.schedule.drop(
            lambda e: (
                e.kind == KIND_DEADLINE
                and e.actor_id == aid
                and str(e.payload_dict.get("node_id", "")) == node_id
            )
        )
    )


def _invoke_actor(
    world: WorldState,
    spec: WorldSpec,
    entry: ScheduledEntry,
    effects: EffectExecutor,
    action_exec: ActionExecutor,
    actor_runtime: ActorRuntime,
    seed: int,
    decisions: list[ActorDecisionRecord],
    ledger: list[Event],
    budget: RunBudget,
    diag: BranchDiagnostics,
) -> tuple[WorldState, list[Event]]:
    aid = str(entry.actor_id)
    actor = world.actors.get(aid)
    if actor is None:
        return world, []
    p = entry.payload_dict
    reason = str(p.get("wake_reason", ""))
    detail = str(p.get("wake_detail", ""))

    # An actor busy with its own unfinished action is not interrupted by a mere
    # opportunity. This is what makes a plan persist across events.
    if actor.is_busy and reason not in _INTERRUPTING:
        return world, []

    node = spec.process.node(str(p.get("node_id", ""))) if p.get("node_id") else None
    feasible = action_exec.feasible_actions(world, node, actor, spec)
    allow_novel = node.allow_novel if node is not None else True
    if not feasible and not allow_novel:
        # A wake the actor could do nothing with is still a wake, and it goes on the
        # record: skipping it silently made the world look inert for no stated reason.
        # The record carries WHY each offered action was infeasible, so the ledger
        # shows "the officer woke and lacked the authority", not nothing at all.
        decisions.append(_no_feasible_action_record(world, spec, node, actor, action_exec, entry))
        return world, []

    base_view = world.view_for(aid)
    view = replace(
        base_view,
        feasible_actions=tuple(action_exec.action_card(a) for a in feasible),
        allow_novel=allow_novel,
        trigger_kind=reason,
        trigger_detail=detail,
        trigger_obs_id=(list(p.get("observation_ids") or []) or [None])[0],
    )
    world = world.with_schedule(
        world.schedule.drop_matching(kind=KIND_DECISION, actor_id=aid, at=world.time)
    )
    state_before = actor.state_dict()
    version_at_decision = world.version
    # What had been delivered to this actor AT the moment it decided. Snapshotted
    # here, before the decision's own events apply: reading it after `world.apply`
    # quietly credited the decision with its own consequences — the record claimed
    # the actor had been handed things that only existed because it acted.
    delivered_at_decision = [d.event_id for d in world.deliveries if d.actor_id == aid]

    result = actor_runtime.step(actor, view, seed=seed)
    world = world.with_actor(result.actor)
    diag.actor_call_counts[aid] = diag.actor_call_counts.get(aid, 0) + 1

    outcome = action_exec.execute(
        result.actor, result.choice, world, spec, seed, microstep=entry.microstep
    )
    world = world.apply(outcome.events)
    produced = [_find(world, e.event_id) for e in outcome.events]
    for ev in produced:
        ledger.append(ev)
    world = _propagate(world, spec, produced, microstep=entry.microstep)

    updated = world.actors[aid]
    if outcome.ongoing is not None:
        updated = replace(updated, current_action=outcome.ongoing)
    elif outcome.status in ("rejected", "wait", "failed"):
        updated = replace(updated, current_action=None)
    world = world.with_actor(updated)

    follow: list[ScheduledEntry] = list(outcome.scheduled)
    follow.extend(
        _deferred_entries(
            outcome.deferred,
            origin=ORIGIN_CONSEQUENCE,
            detail=f"action:{outcome.action_id}",
            parents=tuple(e.event_id for e in produced),
            microstep=entry.microstep + 1,
        )
    )
    follow.extend(_plan_entries(updated))
    follow.extend(_commitment_entries(updated))
    follow.extend(_revisit_entries(updated, world))
    follow.extend(_need_deadline_entries(updated))
    if outcome.status == "rejected":
        # A refusal is information the actor receives. It may then choose something
        # else — a genuine new decision, not a rewrite of the one it made.
        follow.append(
            _decision_entry(
                at=world.time,
                actor_id=aid,
                reason=WAKE_OWN_ACTION,
                detail=f"your attempt was refused: {outcome.reason}",
                causal_parents=tuple(e.event_id for e in produced),
                microstep=entry.microstep + 2,
            )
        )
    if follow:
        world = world.with_schedule(world.schedule.push(*follow))
    world = _cancel_answered_deadline(
        world, entry, reason, outcome.status, actor, world.actors[aid]
    )

    responses = result.responses + list(outcome.gateway_responses)
    gw = [r for r in responses if hasattr(r, "prompt_hash")]
    decisions.append(
        ActorDecisionRecord(
            branch_id=world.branch_id,
            actor_id=aid,
            branch_time=view.branch_time.isoformat(),
            stage=view.stage,
            wake_reason=reason,
            wake_detail=detail,
            trigger_event_ids=list(entry.causal_parents),
            delivered_observation_ids=delivered_at_decision,
            noticed_observation_ids=result.noticed_obs_ids,
            retrieved_memory_ids=result.retrieved_memory_ids,
            plan_before=result.plan_before,
            plan_after=result.plan_after,
            plan_disposition=result.plan_disposition,
            state_before=state_before,
            state_after=world.actors[aid].state_dict(),
            decision_context=result.context,
            intent={
                "mode": result.choice.mode,
                "action_id": result.choice.action_id,
                "params": dict(result.choice.params),
                "target": result.choice.target,
                "novel_description": result.choice.novel_description,
                "novel_intended_effect": result.choice.novel_intended_effect,
                "rationale": result.choice.rationale,
            },
            validation_status=outcome.status,
            validation_reason=outcome.reason,
            event_ids=[e.event_id for e in produced],
            world_version_at_decision=version_at_decision,
            prompt_hash=gw[-1].prompt_hash if gw else "",
            model=gw[-1].model if gw else "",
            tokens_out=gw[-1].tokens_out if gw else 0,
        )
    )
    return world, produced


# ---------------------------------------------------------------------------
# Information propagation and relevance
# ---------------------------------------------------------------------------


def _propagate(
    world: WorldState,
    spec: WorldSpec,
    events: list[Event],
    *,
    source_action_id: str = "",
    microstep: int = 0,
) -> WorldState:
    """Turn events into *deliveries* and schedule the moments they may be noticed.

    Visibility says who could ever see it. Delivery says when it reached them. Notice
    says when they took it in. Each is a separate recorded transition with its own
    timestamp, because collapsing them is how simulators accidentally give everyone
    perfect, instant, universal awareness.

    ``microstep`` is the causal layer of whatever produced these events; the notices go
    one layer *after* it. Stamping them at a fixed layer instead is what made the Bank of
    England run non-terminating: a notice at layer 1 woke an actor whose action landed at
    layer 2, whose notices went back to layer 1, and ``pop_batch`` always takes the
    lowest layer present — so the loop oscillated 2→1→2→1 forever at a single instant.
    Nearly 800 batches fired at one timestamp, with 400 actor calls all seeing the same
    clock. A causal layer must be monotone or it is not an ordering.
    """

    action = spec.action(source_action_id) if source_action_id else None
    deliver_delay = timedelta(seconds=action.delivery_delay_seconds if action else 0)
    notice_delay = timedelta(seconds=action.notice_delay_seconds if action else 0)

    deliveries: list[Delivery] = []
    entries: list[ScheduledEntry] = []
    for ev in events:
        if ev.kind not in _OBSERVABLE_EVENT_KINDS:
            continue
        for aid in world.observers_of(ev):
            available_at = ev.time + deliver_delay
            notice_at = available_at + notice_delay
            deliveries.append(
                Delivery(
                    event_id=ev.event_id,
                    actor_id=aid,
                    available_at=available_at,
                    notice_at=notice_at,
                    channel=str(ev.payload_dict.get("channel", "")),
                )
            )
            entries.append(
                make_entry(
                    at=notice_at,
                    kind=KIND_NOTICE,
                    actor_id=aid,
                    payload={"about": ev.event_id},
                    origin=ORIGIN_CONSEQUENCE,
                    origin_detail=f"delivery:{ev.event_id}",
                    causal_parents=(ev.event_id,),
                    microstep=microstep + 1,
                )
            )
    if not deliveries:
        return world
    world = world.deliver(tuple(deliveries))
    for aid in {d.actor_id for d in deliveries}:
        world = world.with_actor(
            replace(
                world.actors[aid],
                available_event_ids=world.actors[aid].available_event_ids
                | {d.event_id for d in deliveries if d.actor_id == aid},
            )
        )
    return world.with_schedule(world.schedule.push(*entries))


# Event kinds that are a *person saying or undertaking something*, as opposed to the
# world changing. Only these make another participant's act inherently worth noticing.
#
# This is the runtime's own definition of a communication, and it is the one the §9
# temporal report counts, because this module is what decides it: these kinds are what
# make ``_relevance`` wake a recipient, and the delivery records any message counter
# joins against are written here. Counting a different set produced a report that
# disagreed with the mechanism it was describing — most visibly the terminal's own
# ``create_event`` result line, which is not addressed to anyone, is never propagated,
# and inflated "messages sent" by exactly one per branch on every run.
COMMUNICATION_KINDS = frozenset({"deliver_information", "update_commitment"})

# Event kinds that carry observable content. Bookkeeping kinds (an action starting, an
# attempt being refused) are private to the acting actor and are handled separately.
_OBSERVABLE_EVENT_KINDS = frozenset(
    {
        "deliver_information",
        "release_data",
        "create_event",
        "schedule_event",
        "append_record",
        "create_or_update_document",
        "update_commitment",
        "action_rejected",
        "action_failed",
    }
)


def _relevance(
    world: WorldState, spec: WorldSpec, actor: ActorState, noticed: frozenset[str]
) -> tuple[str, str]:
    """Is what this actor just noticed a reason to reconsider?

    Being able to see something is not a reason to act on it. An actor is brought back
    only for a *stated* cause, checked from most specific to least: something addressed
    to it personally, an answer to a question it actually asked, a condition it named
    itself, another participant saying or undertaking something, or a rule the compiler
    wrote for this particular world. Otherwise it remembers what it saw and carries on —
    which is what people do, and what keeps this from becoming a machine that consults
    everyone about everything.
    """

    by_id = {e.event_id: e for e in world.event_history}
    events = [by_id[i] for i in sorted(noticed) if i in by_id]

    for ev in events:
        if actor.actor_id in ev.audience or actor.actor_id in ev.target_ids:
            return WAKE_DIRECTED, f"{ev.kind} addressed to you from {ev.actor_id or 'environment'}"

    for need in actor.open_needs():
        for ev in events:
            if need.asked_of and (ev.actor_id or "") == need.asked_of:
                return WAKE_NEED_MET, f"an answer arrived from {need.asked_of}: {need.question}"

    for ev in events:
        for cond in actor.revisit_conditions:
            if cond.on_information_from and cond.on_information_from == (ev.actor_id or ""):
                return WAKE_REVISIT, cond.description
            if (
                cond.on_record_in
                and ev.kind == "append_record"
                and str(ev.payload_dict.get("collection", "")) == cond.on_record_in
            ):
                return WAKE_REVISIT, cond.description
            if cond.on_field_change and ev.kind in ("set_field", "release_data", "adjust_field"):
                data = ev.payload_dict
                names = set(dict(data.get("fields", {})).keys()) | {str(data.get("field", ""))}
                if cond.on_field_change in names:
                    return WAKE_REVISIT, cond.description

    # A person communicating is not ambient noise. Something another participant said or
    # undertook, reaching you, is a reason to reconsider even when it was said to
    # everyone — which is not true of a data release or a record being filed. Those stay
    # ambient unless a compiled wake rule says otherwise.
    for ev in events:
        if ev.actor_id and ev.actor_id in world.actors and ev.kind in COMMUNICATION_KINDS:
            return (
                WAKE_COMMUNICATION,
                f"{ev.actor_id} communicated: {str(ev.payload_dict.get('text', ''))[:120]}",
            )

    for ev in events:
        for rule in spec.wake_rules:
            if _rule_matches(rule, ev) and _selects(world, spec, rule.wakes, actor.actor_id):
                return WAKE_RULE, rule.reason or rule.rule_id

    return "", ""


def _rule_matches(rule: WakeRule, ev: Event) -> bool:
    data = ev.payload_dict
    if rule.on_record_in and ev.kind == "append_record":
        return str(data.get("collection", "")) == rule.on_record_in
    if rule.on_event_type and ev.kind in ("create_event", "schedule_event"):
        return str(data.get("event_type", "")) == rule.on_event_type
    if rule.on_field_change and ev.kind in ("set_field", "adjust_field", "release_data"):
        names = set(dict(data.get("fields", {})).keys()) | {str(data.get("field", ""))}
        return rule.on_field_change in names
    if rule.on_information_from:
        return (ev.actor_id or "") == rule.on_information_from
    return False


def _revisit_entries(actor: ActorState, world: WorldState) -> list[ScheduledEntry]:
    """A revisit the actor set for a specific time becomes a real future wake-up."""

    out: list[ScheduledEntry] = []
    for cond in actor.revisit_conditions:
        if cond.at is None or cond.at <= world.time:
            continue
        out.append(
            make_entry(
                at=cond.at,
                kind=KIND_REVISIT,
                actor_id=actor.actor_id,
                payload={"detail": cond.description},
                origin=ORIGIN_ACTOR_PLAN,
                origin_detail=f"{actor.actor_id}:revisit",
            )
        )
    return out


def _need_deadline_entries(actor: ActorState) -> list[ScheduledEntry]:
    """Waiting for something that never comes is itself an event."""

    out: list[ScheduledEntry] = []
    for need in actor.open_needs():
        if need.deadline is None:
            continue
        out.append(
            make_entry(
                at=need.deadline,
                kind=KIND_NEED_DEADLINE,
                actor_id=actor.actor_id,
                payload={"question": need.question},
                origin=ORIGIN_ACTOR_PLAN,
                origin_detail=f"{actor.actor_id}:need",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def _bankable(terminal: TerminalExpression) -> tuple[str, ...] | None:
    """The append-only terms this terminal could be banked on, or None if it cannot be.

    Three structural conditions, all computed from the compiled expression and none of
    them declarable by whoever wrote it:

    1. ``yes_when`` is monotone NON-DECREASING — once true, no further evolution of the
       world can make it false. ``event_count(X) > 0`` qualifies; ``event_count(X) < 3``
       is monotone the other way and must never be banked as a YES; ``field(x) > 4e5``
       is not monotone at all and keeps the conservative behaviour.
    2. ``unresolved_when`` is monotone NON-INCREASING — once the world is determinate,
       it stays determinate. Without this a branch could bank while determinate and
       finish indeterminate, and the bank would be asserting an answer over a world
       that had gone back to not having one.
    3. It rests on at least one append-only term. An expression that is monotone only
       because it is constant has nothing to point at, and a bank that cannot name its
       own cause is an assertion.
    """

    if expression_monotonicity(terminal.yes_when) != MONOTONE_UP:
        return None
    if expression_monotonicity(terminal.unresolved_when) not in (MONOTONE_DOWN, MONOTONE_FIXED):
        return None
    return monotone_sources(terminal.yes_when) or None


def _bank_monotone_terminal(
    world: WorldState,
    terminal: TerminalExpression,
    sources: tuple[str, ...] | None,
    diag: BranchDiagnostics,
) -> None:
    """Record a monotone terminal at the instant the world satisfied it.

    G1, in one branch of one real run: the agent chose the satisfying action on twenty
    separate invocations, nineteen of which landed before the actor-call budget ran out
    three scheduled events short of the horizon, and the branch reported
    ``resolved=False``. The run published 0.0 from the two branches that happened to
    finish. The system watched the answer happen nineteen times and published its
    opposite, because the terminal was read once, at the end, on a trajectory that never
    got there.

    A banked YES is final. Nothing here overrides the conservative behaviour for
    anything else: the bank is only reachable through :func:`_bankable`, and it is only
    taken when the world can also produce the events that caused it.
    """

    if sources is None or diag.banked_terminal is not None:
        return
    try:
        if bool(evaluate(terminal.unresolved_when, world)):
            return
        if not bool(evaluate(terminal.yes_when, world)):
            return
    except UndeterminedExpressionError:
        return

    causes = _monotone_witnesses(world, sources)
    if not causes:
        # Satisfied, monotone — and nothing in this branch's ledger produced it. That is
        # an initial condition, which ``pre_resolved`` already reports; it is not
        # something this trajectory watched happen, so it is not banked.
        return

    named = ", ".join(sources)
    diag.banked_terminal = BankedTerminal(
        at=world.time.isoformat(),
        outcome="YES",
        sources=sources,
        caused_by_event_ids=tuple(e.event_id for e in causes),
        cause_count=len(causes),
        world_version=world.version,
        batches_elapsed=diag.batches,
        actor_calls_elapsed=sum(diag.actor_call_counts.values()),
        reason=(
            f"the terminal was satisfied at {world.time.isoformat()} by {len(causes)} "
            f"event(s) on append-only term(s) [{named}], and its expression cannot "
            "decrease; no later truncation can un-happen them"
        ),
    )


def _monotone_witnesses(world: WorldState, sources: tuple[str, ...]) -> tuple[Event, ...]:
    """The ledger events that wrote the append-only terms a bank rests on.

    Matching mirrors :meth:`WorldState.get_events` and the ``append_record`` path in
    :meth:`WorldState.apply` exactly. If the two ever disagreed, a bank could name a
    cause the evaluator never counted, or count one it could not name.
    """

    want_events = {t.split(":", 1)[1] for t in sources if t.startswith("event:")}
    want_collections = {t.split(":", 1)[1] for t in sources if t.startswith("collection:")}
    out: list[Event] = []
    for ev in world.event_history:
        if ev.status is not EventStatus.APPLIED:
            continue
        data = ev.payload_dict
        wrote_event = ev.kind in want_events or str(data.get("event_type", "")) in want_events
        wrote_record = (
            ev.kind == "append_record" and str(data.get("collection", "")) in want_collections
        )
        if wrote_event or wrote_record:
            out.append(ev)
    return tuple(out)


def _finalize(
    world: WorldState,
    terminal: TerminalExpression,
    effects: EffectExecutor,
    ledger: list[Event],
    diag: BranchDiagnostics,
) -> WorldState:
    # Nothing further is scheduled inside the window, so the branch's clock reaches the
    # horizon. This is the end of the question's window, not a jump over live events.
    world = world.with_time(world.contract.horizon)
    evaluation = evaluate_terminal(world, terminal)
    banked = diag.banked_terminal

    # A branch that stopped with things still due to happen did not reach its end; it
    # was cut short. Reporting a resolved outcome for it would claim we watched the
    # process finish when we stopped watching, which is forced completion wearing the
    # terminal evaluator's clothes. The mass stays unresolved and widens the bounds.
    #
    # That reasoning is exactly right for `field('deliveries') > 400000`: more
    # simulation genuinely could move the number, so an evaluation taken where we
    # stopped is a claim about a process we did not watch. It is exactly WRONG for
    # `event_count(X) > 0`. Truncation cannot un-happen nineteen events that already
    # happened, and refusing the answer there does not withhold a claim — it publishes
    # the opposite one. The distinction is monotonicity, and it is computed from the
    # expression's own structure (:func:`_bankable`), never declared by whoever wrote
    # the world. A banked YES was watched happening, with the causing events named.
    if diag.unfired_in_horizon > 0 and evaluation.resolved and banked is None:
        evaluation = TerminalEvaluation(
            resolved=False,
            outcome=None,
            reason=(
                f"trajectory cut short with {diag.unfired_in_horizon} scheduled events "
                f"still due before the horizon ({diag.stop_reason}); the process did not "
                "run to its end, so its outcome is not known — the simulator stopped "
                "before the world did, which is a failure of this run and not "
                "uncertainty about the world"
            ),
            highlights=evaluation.highlights,
        )

    if banked is not None:
        if evaluation.resolved and evaluation.outcome == "YES":
            evaluation = TerminalEvaluation(
                resolved=True,
                outcome="YES",
                reason=f"{evaluation.reason}; banked at {banked.at}: {banked.reason}",
                highlights=evaluation.highlights,
            )
        else:
            # The monotonicity analysis said this could not come undone, and the world
            # at the horizon disagrees. One of the two is wrong and it is not safe to
            # guess which, so the branch resolves nothing and says precisely that. A
            # check that can be wrong silently is a check that reads like an approval.
            diag.bank_contradicted = (
                f"banked YES at {banked.at} on [{', '.join(banked.sources)}] but the "
                f"horizon evaluation says "
                f"{evaluation.outcome if evaluation.resolved else 'unresolved'}"
            )
            evaluation = TerminalEvaluation(
                resolved=False,
                outcome=None,
                reason=(
                    "a monotone terminal banked at "
                    f"{banked.at} is contradicted by the evaluation at the horizon "
                    f"({evaluation.reason}); the monotonicity analysis and the world "
                    "disagree, so this branch resolves nothing"
                ),
                highlights=evaluation.highlights,
            )

    world = world.set_terminal(evaluation)
    ev = effects.raw_event(
        world,
        kind="create_event",
        actor_id=None,
        payload={
            "event_type": "result_recorded",
            "text": evaluation.reason,
            # The banked fact belongs in the trace, not only in the diagnostics: a
            # forensic replay has to be able to see WHEN the answer became true and
            # WHAT made it true without re-deriving either.
            "data": {
                "outcome": evaluation.outcome or "unresolved",
                "stop": diag.stop_reason,
                "execution_status": diag.execution_status,
                "banked_terminal": banked.as_dict() if banked else None,
            },
        },
        visibility=Visibility.PUBLIC,
    )
    world = world.apply([ev])
    ledger.append(world.event_history[-1])
    return world


def terminal_lineage(
    world: WorldState, terminal: TerminalExpression, spec: WorldSpec | None = None
) -> tuple[dict[str, object], ...]:
    """For each term the terminal reads, what actually wrote it in *this* trajectory.

    The compile-time gate asks whether something *could* produce each term. This is the
    other half, and the one that cannot be satisfied by a plausible-looking world spec:
    it walks the branch's own event ledger and names the event, the actor and the causal
    parents behind every terminal term. A term with no writer here was not produced by
    anything that happened — whatever the spec promised.
    """

    from .world_compiler import _expr_terms

    # The same namespaced terms the compile-time gate reads, so the two halves of
    # producer lineage cannot disagree about what the terminal even reads. Walking only
    # bare field and collection names left a terminal built on a document field, a
    # resource or an event count with *no* lineage at all — vacuously clean, on the
    # question where "what produced this" matters most.
    terms = sorted(_expr_terms(terminal.yes_when))
    writers: dict[str, list[Event]] = {t: [] for t in terms}
    for ev in world.event_history:
        for term in _event_writes(ev) & set(terms):
            writers[term].append(ev)

    # A term the record established before the window opened has no writer in this
    # ledger and never will: the world produced it, months earlier, and the compiler
    # cited the claims that say so. Reporting it as unproduced would read as the defect
    # this function exists to expose, so its lineage is the citation.
    established = _established_terms(spec)

    out: list[dict[str, object]] = []
    for term in terms:
        evs = writers[term]
        cites = established.get(term, ())
        out.append(
            {
                "terminal_term": term,
                "written_by": [
                    {
                        "event_id": e.event_id,
                        "kind": e.kind,
                        "at": e.time.isoformat(),
                        "actor_id": e.actor_id,
                        "caused_by": list(e.parent_event_ids),
                        "evidence_claim_ids": list(e.evidence_claim_ids),
                    }
                    for e in evs
                ],
                "writer_count": len(evs),
                "produced_by_an_actor": any(e.actor_id for e in evs),
                "established_by_evidence": list(cites),
                "unproduced": not evs and not cites,
            }
        )
    return tuple(out)


def _event_writes(ev: Event) -> set[str]:
    """The namespaced terms one applied event actually wrote.

    The mirror of ``world_compiler._effect_produces``, which asks the same question of a
    compiled effect before anything runs. Both must name a term the same way or a term
    the gate cleared would look unproduced here.
    """

    payload = ev.payload_dict
    out: set[str] = set()
    name = payload.get("field")
    if ev.kind in ("set_field", "adjust_field") and isinstance(name, str):
        out.add(f"field:{name}")
    coll = payload.get("collection")
    if ev.kind == "append_record" and isinstance(coll, str):
        out.add(f"collection:{coll}")
    if ev.kind == "create_or_update_document":
        doc = payload.get("document")
        sub = payload.get("fields")
        if isinstance(doc, str) and isinstance(sub, dict):
            out |= {f"document:{doc}.{k}" for k in sub}
    if ev.kind in ("create_event", "schedule_event"):
        etype = payload.get("event_type")
        if isinstance(etype, str):
            out.add(f"event:{etype}")
    if ev.kind in ("transfer_resource", "consume_resource"):
        res = payload.get("resource")
        if isinstance(res, str):
            out.add(f"resource:{res}")
    if ev.kind == "release_data":
        sub = payload.get("fields")
        if isinstance(sub, dict):
            out |= {f"field:{k}" for k in sub}
    if ev.kind == "deliver_information":
        sub = payload.get("info_fields")
        if isinstance(sub, dict):
            out |= {f"field:{k}" for k in sub}
    return out


def _established_terms(spec: WorldSpec | None) -> dict[str, tuple[str, ...]]:
    """Terminal terms whose initial value the compiled world cites evidence for."""

    if spec is None:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for f in spec.fields:
        if f.initial is not None and f.evidence_claim_ids:
            out[f"field:{f.field_id}"] = f.evidence_claim_ids
    for doc in spec.documents:
        if not doc.evidence_claim_ids:
            continue
        for name, value in doc.fields:
            if value is not None:
                out[f"document:{doc.document_id}.{name}"] = doc.evidence_claim_ids
    return out


def evaluate_terminal(world: WorldState, terminal: TerminalExpression) -> TerminalEvaluation:
    """The single place YES/NO/unresolved is decided — deterministic, from world state,
    through the universal operators only. No LLM, no mechanism family, no default."""

    try:
        if bool(evaluate(terminal.unresolved_when, world)):
            return TerminalEvaluation(
                resolved=False,
                outcome=None,
                reason=terminal.description or "terminal condition not determinable",
                highlights=_highlights(world),
            )
        yes = bool(evaluate(terminal.yes_when, world))
    except UndeterminedExpressionError as exc:
        # The terminal reads something this branch never determined. That is an honest
        # unresolved outcome — not a NO, and not a crashed run.
        return TerminalEvaluation(
            resolved=False,
            outcome=None,
            reason=f"terminal depends on a value the world never determined: {exc}",
            highlights=_highlights(world),
        )
    return TerminalEvaluation(
        resolved=True,
        outcome="YES" if yes else "NO",
        reason=terminal.description or ("yes_when satisfied" if yes else "yes_when not satisfied"),
        highlights=_highlights(world),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision_entry(
    *,
    at: datetime,
    actor_id: str,
    reason: str,
    detail: str,
    node_id: str = "",
    observation_ids: tuple[str, ...] = (),
    causal_parents: tuple[str, ...] = (),
    microstep: int = 1,
) -> ScheduledEntry:
    return make_entry(
        at=at,
        kind=KIND_DECISION,
        actor_id=actor_id,
        payload={
            "wake_reason": reason,
            "wake_detail": detail,
            "node_id": node_id,
            "observation_ids": list(observation_ids),
        },
        origin=ORIGIN_CONSEQUENCE,
        origin_detail=f"wake:{reason}",
        causal_parents=causal_parents,
        microstep=microstep,
    )


def _node_time(node: ProcessNode, default: datetime, *, after: datetime | None = None) -> datetime:
    """When this node happens.

    A node with an explicit ``at`` happens then. A node without one happens relative to
    what entered it. An ``at`` the runtime cannot read is a compilation defect, not a
    licence to invent a time — but it is also not worth ending a run over, so the node
    falls back to its relative timing and the unreadable value is left in the compiled
    spec where the trace shows it.
    """

    at = _parse_dt(node.at)
    if at is not None:
        return at
    base = after if after is not None else default
    return base + timedelta(seconds=max(0, node.delay_seconds))


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _participants(world: WorldState, spec: WorldSpec, node: ProcessNode) -> tuple[str, ...]:
    if not node.participants:
        return ()  # an environment-only node: nobody acts here
    return _select(world, spec, node.participants)


def _select(world: WorldState, spec: WorldSpec, selectors: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for sel in selectors:
        if sel == "*":
            out.extend(a.entity_id for a in spec.actors if a.entity_id in world.actors)
        elif sel.startswith("role:"):
            role = sel[5:]
            out.extend(aid for aid, a in world.actors.items() if a.role == role)
        elif sel in world.actors:
            out.append(sel)
    order = {a.entity_id: i for i, a in enumerate(spec.actors)}
    return tuple(sorted(dict.fromkeys(out), key=lambda x: order.get(x, 1 << 30)))


def _selects(world: WorldState, spec: WorldSpec, selectors: tuple[str, ...], actor_id: str) -> bool:
    return actor_id in _select(world, spec, selectors)


def _highlights(world: WorldState) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    for coll, recs in world.records:
        for r in recs:
            out.append((f"{coll}:{r.key}", str(r.value)))
    for name, value in world.fields:
        out.append((f"field:{name}", str(value)))
    return tuple(out)


def _find(world: WorldState, event_id: str) -> Event:
    for ev in reversed(world.event_history):
        if ev.event_id == event_id:
            return ev
    raise KeyError(event_id)


def _branch_outcome(
    world: WorldState,
    scenario: Scenario,
    *,
    pre_resolved: bool,
    pre_outcome: str | None,
    diag: BranchDiagnostics,
) -> BranchOutcome:
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
        pre_outcome=pre_outcome,
        pre_resolved=pre_resolved,
        weight_grounded=weights_grounded(scenario),
        unresolved_class=diag.unresolved_class,
        banked_at=diag.banked_terminal.at if diag.banked_terminal else None,
        banked_by_event_ids=(
            diag.banked_terminal.caused_by_event_ids if diag.banked_terminal else ()
        ),
    )


def _unresolved_outcome(
    scenario: Scenario,
    reason: str,
    *,
    pre_resolved: bool = False,
    pre_outcome: str | None = None,
) -> BranchOutcome:
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
        pre_outcome=pre_outcome,
        pre_resolved=pre_resolved,
        weight_grounded=weights_grounded(scenario),
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
