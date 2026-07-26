"""The per-run temporal fidelity report (§9; TMP-2, FD-13).

Computed mechanically from the run's own record — the event ledger, the actor
decision records, the branch outcomes and the final per-branch worlds — with no
model call and no re-simulation. Every counter answers one of the §9 questions:
how many distinct moments each branch actually lived through; where its largest
silent jump was; which actions took no time; which communications were noticed
the same instant they were sent; which wake-ups carried materially new
information and which repeated an actor without any (the ACT-8 measurement);
what became of every action (started / completed / failed / rejected); what
became of every message (sent / delivered / undelivered / noticed / missed);
how often non-actor processes moved the world; how often the branch's own
hypothesis entered it; and how often the terminal was checked.

Every counter names a real unit and holds to it. The message block counts
(message, recipient) pairs throughout — counting sends per *event* against
deliveries per *recipient* once produced "16 of 5 messages delivered" — and it
counts the kinds the runtime itself calls communications
(:data:`sworldmodel.engine.COMMUNICATION_KINDS`), never the terminal's own
result line. A branch's hypothesis is counted as branch construction rather
than as a process moving the world, wherever in the window it lands.

Wake novelty is judged on what the actor actually *noticed* at the wake — never
on what merely sat delivered-unread in front of it, and never on the decision's
own consequences — and it is keyed on information *content* (the
``information_digest`` approach in :mod:`sworldmodel.world`), so byte-identical
material re-delivered under a fresh event id is not "new". Flagged repeats are
reported split by wake cause, because a calendar arriving with nothing new and
the world claiming an arrival that never happened are different findings.

Nothing here decides anything. The report *measures*; a frozen clock, an
instant conversation or a causally inert re-invocation is put on the record for
the realism adversary to reject, never smoothed over.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .engine import (
    COMMUNICATION_KINDS,
    WAKE_CAUSE_CALENDAR,
    WAKE_CAUSE_INFORMATION,
    WAKE_CAUSE_UNCLASSIFIED,
    ActorDecisionRecord,
    RunResult,
    wake_cause_class,
)
from .ids import canonical_json
from .models import BranchOutcome, Event
from .world import WorldState

TEMPORAL_REPORT_FILENAME = "temporal_report.json"

# The event kinds counted as communications: the runtime's OWN definition
# (:data:`sworldmodel.engine.COMMUNICATION_KINDS`), imported rather than restated.
#
# The engine is the module that decides what a communication is — these kinds are what
# make a recipient's ``_relevance`` fire, and the delivery records this block joins
# against are written by the same module. Aligning instead with the replay core's
# wider artifact-extraction set misaligned the report from that mechanism and made it
# measurably wrong: ``create_event`` covers the terminal's own ``result_recorded``
# line, which :func:`sworldmodel.engine._finalize` emits once per branch, addresses to
# nobody and never propagates, so "messages sent" was inflated by exactly the branch
# count on every run (a live four-branch run reported 4 messages sent when zero were),
# while ``update_commitment`` — which the engine does treat as a communication and
# does generate deliveries for — was dropped, so a genuinely missed commitment
# disappeared from ``missed`` altogether.
_COMM_KINDS = COMMUNICATION_KINDS

# The terminal's own line. It is a recorded result, not something anyone said, and it
# is excluded explicitly as well as by kind so widening the kind set can never let it
# back in unnoticed.
_TERMINAL_EVENT_TYPE = "result_recorded"

# Limits of the content key ACT-8 novelty rests on (:func:`_content_signature`),
# recorded with every report because the measurement is only as strong as this key.
# None of these is repaired silently: an over-count here shows up as a wake credited
# with "new" information, which is the direction that flatters the runtime, so the
# limits travel with the number.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "wake novelty is keyed on repr() of the event payload: two payloads that differ "
    "only in nested dict insertion order read as different content, so re-delivered "
    "material can be counted as new",
    "the key is not numerically canonical: 1 and 1.0, or 0.30 and 0.3, read as different content",
    "the key is exact, not semantic: near-identical text ('please reply' vs 'Please "
    "reply.') reads as new information",
    "the key resolves ids against this branch's ledger only; an unresolvable "
    "observation id falls back to the id itself and is never assumed to repeat",
)


def write_temporal_report(out_dir: Path, result: RunResult) -> None:
    """Emit ``temporal_report.json`` beside the other run artifacts."""

    payload = compute_temporal_report(result)
    (out_dir / TEMPORAL_REPORT_FILENAME).write_text(canonical_json(payload) + "\n")


def compute_temporal_report(result: RunResult) -> dict[str, Any]:
    """Every §9 counter, per branch and totalled across the run."""

    events_by_branch: dict[str, list[Event]] = {}
    for ev in result.event_ledger:
        events_by_branch.setdefault(ev.branch_id, []).append(ev)
    decisions_by_branch: dict[str, list[ActorDecisionRecord]] = {}
    for d in result.actor_decisions:
        decisions_by_branch.setdefault(d.branch_id, []).append(d)
    outcomes_by_branch: dict[str, BranchOutcome] = {b.branch_id: b for b in result.branch_outcomes}

    branch_ids = sorted(set(events_by_branch) | set(decisions_by_branch) | set(result.final_worlds))
    branches = {
        bid: _branch_report(
            events_by_branch.get(bid, []),
            decisions_by_branch.get(bid, []),
            result.final_worlds.get(bid),
            outcomes_by_branch.get(bid),
        )
        for bid in branch_ids
    }
    return {
        "branches": branches,
        "run": _run_totals(branches),
        # The ACT-8 measurement is exactly as strong as the content key it rests on,
        # so the key's limits ship with every report rather than in a document the
        # reader of this artifact may never open.
        "known_limitations": {"wake_novelty_content_key": list(KNOWN_LIMITATIONS)},
    }


# ---------------------------------------------------------------------------
# Per-branch counters
# ---------------------------------------------------------------------------


def _branch_report(
    events: list[Event],
    decisions: list[ActorDecisionRecord],
    world: WorldState | None,
    outcome: BranchOutcome | None,
) -> dict[str, Any]:
    timestamps = _distinct_timestamps(events, decisions)
    return {
        "distinct_timestamps": len(timestamps),
        "first_timestamp": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
        "largest_jump": _largest_jump(timestamps),
        "zero_duration_actions": _zero_duration_actions(events),
        "same_timestamp_communications": _same_timestamp_communications(events, world),
        "wake_ups": _wake_ups(decisions, events),
        "actions": _actions(events),
        "messages": _messages(events, world),
        "process_updates": _process_updates(events),
        "scenario_releases": _scenario_releases(events),
        "terminal_checks": _terminal_checks(events, outcome),
    }


def _distinct_timestamps(
    events: list[Event], decisions: list[ActorDecisionRecord]
) -> list[datetime]:
    """Every distinct moment this branch actually lived through, in order: the
    moments its events happened plus the moments its actors decided."""

    moments = {ev.time for ev in events}
    for d in decisions:
        at = _parse_dt(d.branch_time)
        if at is not None:
            moments.add(at)
    return sorted(moments)


def _largest_jump(timestamps: list[datetime]) -> dict[str, Any]:
    """The largest gap between consecutive recorded moments — where the branch
    clock moved furthest with nothing on the record in between."""

    if len(timestamps) < 2:
        return {"seconds": 0.0, "from": None, "to": None}
    best_from, best_to = timestamps[0], timestamps[1]
    for a, b in zip(timestamps, timestamps[1:], strict=False):
        if (b - a) > (best_to - best_from):
            best_from, best_to = a, b
    return {
        "seconds": (best_to - best_from).total_seconds(),
        "from": best_from.isoformat(),
        "to": best_to.isoformat(),
    }


def _zero_duration_actions(events: list[Event]) -> int:
    """Actions that started and were due to complete at the same instant."""

    count = 0
    for ev in events:
        if ev.kind != "action_started":
            continue
        completes = _parse_dt(str(ev.payload_dict.get("completes_at", "")))
        if completes is not None and completes == ev.time:
            count += 1
    return count


def _is_communication(ev: Event) -> bool:
    """Whether this event is a person saying or undertaking something.

    The runtime's own kind set, minus the terminal's ``result_recorded`` line, which
    records what the branch turned out to be rather than anything anyone said.
    """

    if ev.kind not in _COMM_KINDS:
        return False
    return str(ev.payload_dict.get("event_type", "")) != _TERMINAL_EVENT_TYPE


def _same_timestamp_communications(events: list[Event], world: WorldState | None) -> int:
    """Communications noticed by some recipient at the very instant they were sent —
    the instant-conversation shape the realism adversary looks for.

    This is a TMP-3 realism check, so it is kept strictly to things a participant
    actually said. Counting bookkeeping events here manufactured "instant
    conversations" out of records nobody spoke — four of them on a multiparty probe
    world — and a check that cries wolf is a check that gets ignored.
    """

    if world is None:
        return 0
    sent_at = {ev.event_id: ev.time for ev in events if _is_communication(ev)}
    instant = {
        d.event_id
        for d in world.deliveries
        if d.event_id in sent_at
        and d.noticed_at is not None
        and d.noticed_at == sent_at[d.event_id]
    }
    return len(instant)


def _content_signature(ev: Event | None, obs_id: str) -> str:
    """The content identity of one noticed observation.

    The same keying as :meth:`sworldmodel.world.WorldState.information_digest`:
    kind plus payload, NOT the event id — a cascade's defining property is that it
    delivers the same sentence under a hundred fresh ids. An id the ledger cannot
    resolve falls back to the id itself: unresolvable content is never assumed to
    repeat anything.
    """

    if ev is None:
        return f"id:{obs_id}"
    return f"{ev.kind}|{ev.payload!r}"


def _wake_ups(decisions: list[ActorDecisionRecord], events: list[Event]) -> dict[str, Any]:
    """Wake-ups with and without materially new information (ACT-8 measurement).

    A wake carries materially new information when the actor *noticed* something
    at it whose content no prior wake of the same actor in the same branch had
    already put before it. Delivered-but-unread material does not count — a wake
    that happens while a message sits unread is not credited with that message,
    and the later wake at which the actor actually reads it is. A wake after the
    actor's first that notices nothing new is counted and *flagged* — measured,
    never suppressed. The per-wake trail makes every verdict auditable.

    The flagged repeats are **split by what caused the wake**, because the two causes
    mean opposite things. An information-driven repeat (a communication, a directed
    answer, a compiled wake rule) is a defect every time: the runtime said something
    reached this actor and the actor's own noticed set is empty of it. A calendar-driven
    repeat (a scheduled opportunity, a commitment falling due, a deadline arriving) is
    often exactly right — turning up to a meeting having learned nothing on the way is
    not a bug — and matters as a volume, not as an incident. A live run reported "8 of
    12 wakes repeated without new information" under one headline; the adversary
    decomposed it into 4 legitimate opportunities and 4 genuinely redundant deadline
    wakes. A number that is half false alarms is a number readers learn to skip, and
    a skipped ACT-8 is how the FD-7 class survived its first fix.
    """

    by_id = {ev.event_id: ev for ev in events}
    seen: dict[str, set[str]] = {}
    wake_index: dict[str, int] = {}
    with_new = 0
    without_new = 0
    repeated_without_new = 0
    by_cause = {WAKE_CAUSE_CALENDAR: 0, WAKE_CAUSE_INFORMATION: 0, WAKE_CAUSE_UNCLASSIFIED: 0}
    flagged: list[dict[str, Any]] = []
    trail: list[dict[str, Any]] = []
    for d in decisions:
        prior = seen.setdefault(d.actor_id, set())
        n = wake_index.get(d.actor_id, 0)
        carried = {_content_signature(by_id.get(i), i) for i in d.noticed_observation_ids}
        novel = carried - prior
        cause = wake_cause_class(d.wake_reason)
        if novel:
            with_new += 1
        else:
            without_new += 1
            if n > 0:
                # A repeat invocation at which the actor took in nothing it had
                # not already been shown — the FD-7 "re-signaled 3×" shape.
                repeated_without_new += 1
                by_cause[cause] = by_cause.get(cause, 0) + 1
                flagged.append(
                    {
                        "actor_id": d.actor_id,
                        "branch_time": d.branch_time,
                        "wake_reason": d.wake_reason,
                        "wake_detail": d.wake_detail,
                        "cause_class": cause,
                    }
                )
        trail.append(
            {
                "actor_id": d.actor_id,
                "branch_time": d.branch_time,
                "wake_reason": d.wake_reason,
                "cause_class": cause,
                "materially_new_information": bool(novel),
                "new_content_count": len(novel),
                "noticed_event_ids": list(d.noticed_observation_ids),
            }
        )
        prior |= carried
        wake_index[d.actor_id] = n + 1
    return {
        "total": len(decisions),
        "with_new_information": with_new,
        "without_new_information": without_new,
        "repeated_without_new_information": repeated_without_new,
        "repeated_without_new_information_by_cause": by_cause,
        "flagged_repeats": flagged,
        "trail": trail,
    }


def _actions(events: list[Event]) -> dict[str, int]:
    kinds = [ev.kind for ev in events]
    return {
        "started": kinds.count("action_started"),
        "completed": kinds.count("action_completed"),
        "failed": kinds.count("action_failed"),
        "rejected": kinds.count("action_rejected"),
    }


def _messages(events: list[Event], world: WorldState | None) -> dict[str, int]:
    """What became of every message, in ONE unit: the (message, recipient) pair.

    ``delivered``, ``noticed`` and ``missed`` have always been per-recipient — they are
    counts of delivery records — so counting ``sent`` per *event* put a numerator and a
    denominator from different universes side by side, and a probe world duly reported
    "16 of 5 messages delivered". ``sent`` is therefore every (message, recipient) pair
    the branch produced: everyone a message was addressed to, plus everyone the world
    actually put it in front of. ``send_events`` keeps the message count itself, named
    for what it is, so nothing is lost.

    ``undelivered`` is an addressee the world never delivered to; ``missed`` is
    delivered-but-never-noticed by the end of the branch — a message that reached
    someone who never took it in. The two are different failures and stay apart.
    """

    comm = [ev for ev in events if _is_communication(ev)]
    comm_ids = {ev.event_id for ev in comm}
    deliveries = (
        [d for d in world.deliveries if d.event_id in comm_ids] if world is not None else []
    )
    delivered_to: dict[str, set[str]] = {}
    for d in deliveries:
        delivered_to.setdefault(d.event_id, set()).add(d.actor_id)
    sent = sum(len(_recipients(ev, delivered_to)) for ev in comm)
    delivered = len(deliveries)
    noticed = sum(1 for d in deliveries if d.noticed_at is not None)
    return {
        "send_events": len(comm),
        "sent": sent,
        "delivered": delivered,
        "undelivered": sent - delivered,
        "noticed": noticed,
        "missed": delivered - noticed,
    }


def _recipients(ev: Event, delivered_to: dict[str, set[str]]) -> set[str]:
    """Who this message was for: everyone it named, plus everyone it reached.

    A message names its addressees (``audience``/``target_ids``); a broadcast names
    nobody and its recipients are whoever the world's visibility rules let receive it.
    Taking the union keeps ``sent`` a superset of ``delivered`` by construction, so
    ``undelivered`` counts real gaps — an addressee the world never delivered to —
    instead of going negative on a public message that reached more people than it
    named.
    """

    return set(ev.audience) | set(ev.target_ids) | delivered_to.get(ev.event_id, set())


def _is_branch_hypothesis(ev: Event) -> bool:
    """This branch's own hypothesis entering the world — its standing seed, or a
    dated part of it arriving at its release moment. Both carry ``branch_conditions``:
    the runtime stamps the deferred release the same way it stamps the seed, so a
    hypothesis is recognisable wherever in the window it lands."""

    return ev.kind == "release_data" and "branch_conditions" in ev.payload_dict


def _process_updates(events: list[Event]) -> int:
    """Non-actor events that moved the world: external occurrences, process-node
    effects and fired data releases.

    The branch's own hypothesis is branch *construction*, not a process moving the
    world, and is counted separately as ``scenario_releases``. Only the standing seed
    used to be excluded, because only the seed carried ``branch_conditions``: a branch
    whose release was dated announced itself as a process update, so a live run
    reported four process updates when its processes had moved nothing at all. The
    terminal's ``result_recorded`` line is counted as a terminal check instead.
    """

    count = 0
    for ev in events:
        if ev.actor_id:
            continue
        if _is_branch_hypothesis(ev):
            continue
        if str(ev.payload_dict.get("event_type", "")) == _TERMINAL_EVENT_TYPE:
            continue
        count += 1
    return count


def _scenario_releases(events: list[Event]) -> int:
    """How many times this branch's own hypothesis entered the world.

    Never hidden — just not counted as something the world did. One per branch means
    the whole hypothesis was standing from t0; more than one means the branch learned
    its uncertain values at their own separate moments, which is the shape TMP-4
    requires when the evidence dates them differently.
    """

    return sum(1 for ev in events if _is_branch_hypothesis(ev))


def _terminal_checks(events: list[Event], outcome: BranchOutcome | None) -> dict[str, int]:
    """How often the terminal was consulted: once before anything ran (recorded on
    the branch outcome as ``pre_resolved``/``pre_outcome``) and once per
    ``result_recorded`` event at finalization."""

    final = sum(
        1 for ev in events if str(ev.payload_dict.get("event_type", "")) == "result_recorded"
    )
    return {
        "pre_simulation": 1 if outcome is not None else 0,
        "final_recorded": final,
    }


# ---------------------------------------------------------------------------
# Run totals
# ---------------------------------------------------------------------------


def _run_totals(branches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sums across branches; the largest jump is the largest anywhere, because each
    branch carries its own clock."""

    totals: dict[str, Any] = {
        "branch_count": len(branches),
        "distinct_timestamps": 0,
        "largest_jump_seconds": 0.0,
        "zero_duration_actions": 0,
        "same_timestamp_communications": 0,
        "wake_ups": {
            "total": 0,
            "with_new_information": 0,
            "without_new_information": 0,
            "repeated_without_new_information": 0,
            "repeated_without_new_information_by_cause": {
                WAKE_CAUSE_CALENDAR: 0,
                WAKE_CAUSE_INFORMATION: 0,
                WAKE_CAUSE_UNCLASSIFIED: 0,
            },
        },
        "actions": {"started": 0, "completed": 0, "failed": 0, "rejected": 0},
        "messages": {
            "send_events": 0,
            "sent": 0,
            "delivered": 0,
            "undelivered": 0,
            "noticed": 0,
            "missed": 0,
        },
        "process_updates": 0,
        "scenario_releases": 0,
        "terminal_checks": {"pre_simulation": 0, "final_recorded": 0},
    }
    for report in branches.values():
        totals["distinct_timestamps"] += int(report["distinct_timestamps"])
        totals["largest_jump_seconds"] = max(
            float(totals["largest_jump_seconds"]), float(report["largest_jump"]["seconds"])
        )
        totals["zero_duration_actions"] += int(report["zero_duration_actions"])
        totals["same_timestamp_communications"] += int(report["same_timestamp_communications"])
        for key in (
            "total",
            "with_new_information",
            "without_new_information",
            "repeated_without_new_information",
        ):
            totals["wake_ups"][key] += int(report["wake_ups"][key])
        for cause, count in report["wake_ups"]["repeated_without_new_information_by_cause"].items():
            causes = totals["wake_ups"]["repeated_without_new_information_by_cause"]
            causes[cause] = causes.get(cause, 0) + int(count)
        for key in ("started", "completed", "failed", "rejected"):
            totals["actions"][key] += int(report["actions"][key])
        for key in ("send_events", "sent", "delivered", "undelivered", "noticed", "missed"):
            totals["messages"][key] += int(report["messages"][key])
        totals["process_updates"] += int(report["process_updates"])
        totals["scenario_releases"] += int(report["scenario_releases"])
        for key in ("pre_simulation", "final_recorded"):
            totals["terminal_checks"][key] += int(report["terminal_checks"][key])
    return totals


def _parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
