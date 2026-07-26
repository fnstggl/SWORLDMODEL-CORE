"""The per-run temporal fidelity report (§9; TMP-2, FD-13).

Computed mechanically from the run's own record — the event ledger, the actor
decision records, the branch outcomes and the final per-branch worlds — with no
model call and no re-simulation. Every counter answers one of the §9 questions:
how many distinct moments each branch actually lived through; where its largest
silent jump was; which actions took no time; which communications were noticed
the same instant they were sent; which wake-ups carried materially new
information and which repeated an actor without any (the ACT-8 measurement);
what became of every action (started / completed / failed / rejected); what
became of every message (sent / delivered / noticed / missed); how often
non-actor processes moved the world; and how often the terminal was checked.

Wake novelty is judged on what the actor actually *noticed* at the wake — never
on what merely sat delivered-unread in front of it, and never on the decision's
own consequences — and it is keyed on information *content* (the
``information_digest`` approach in :mod:`sworldmodel.world`), so byte-identical
material re-delivered under a fresh event id is not "new".

Nothing here decides anything. The report *measures*; a frozen clock, an
instant conversation or a causally inert re-invocation is put on the record for
the realism adversary to reject, never smoothed over.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .engine import ActorDecisionRecord, RunResult
from .ids import canonical_json
from .models import BranchOutcome, Event
from .replaycore import COMMUNICATION_EVENT_KINDS
from .world import WorldState

TEMPORAL_REPORT_FILENAME = "temporal_report.json"

# The event kinds counted as communications. This is deliberately the SAME set the
# replay core's ``extract_communications`` uses to build ``communications.jsonl``,
# so the report's ``messages`` block reconciles row-for-row with that artifact
# instead of quietly counting a different universe of events.
_COMM_KINDS = frozenset(COMMUNICATION_EVENT_KINDS)


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
    return {"branches": branches, "run": _run_totals(branches)}


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


def _same_timestamp_communications(events: list[Event], world: WorldState | None) -> int:
    """Communications noticed by some recipient at the very instant they were sent —
    the instant-conversation shape the realism adversary looks for."""

    if world is None:
        return 0
    sent_at = {ev.event_id: ev.time for ev in events if ev.kind in _COMM_KINDS}
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
    """

    by_id = {ev.event_id: ev for ev in events}
    seen: dict[str, set[str]] = {}
    wake_index: dict[str, int] = {}
    with_new = 0
    without_new = 0
    repeated_without_new = 0
    flagged: list[dict[str, Any]] = []
    trail: list[dict[str, Any]] = []
    for d in decisions:
        prior = seen.setdefault(d.actor_id, set())
        n = wake_index.get(d.actor_id, 0)
        carried = {_content_signature(by_id.get(i), i) for i in d.noticed_observation_ids}
        novel = carried - prior
        if novel:
            with_new += 1
        else:
            without_new += 1
            if n > 0:
                # A repeat invocation at which the actor took in nothing it had
                # not already been shown — the FD-7 "re-signaled 3×" shape.
                repeated_without_new += 1
                flagged.append(
                    {
                        "actor_id": d.actor_id,
                        "branch_time": d.branch_time,
                        "wake_reason": d.wake_reason,
                        "wake_detail": d.wake_detail,
                    }
                )
        trail.append(
            {
                "actor_id": d.actor_id,
                "branch_time": d.branch_time,
                "wake_reason": d.wake_reason,
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
    """Sent / delivered / noticed / missed, joined against the branch's own
    delivery records. ``missed`` is delivered-but-never-noticed by the end of the
    branch — a message that reached someone who never took it in. ``sent`` counts
    exactly the events ``communications.jsonl`` extracts for this branch
    (``replaycore.COMMUNICATION_EVENT_KINDS``), so the two artifacts reconcile
    row-for-row."""

    comm_ids = {ev.event_id for ev in events if ev.kind in _COMM_KINDS}
    deliveries = (
        [d for d in world.deliveries if d.event_id in comm_ids] if world is not None else []
    )
    delivered = len(deliveries)
    noticed = sum(1 for d in deliveries if d.noticed_at is not None)
    return {
        "sent": len(comm_ids),
        "delivered": delivered,
        "noticed": noticed,
        "missed": delivered - noticed,
    }


def _process_updates(events: list[Event]) -> int:
    """Non-actor events that moved the world: external occurrences, process-node
    effects and fired data releases. The branch's own seed hypothesis (the
    ``release_data`` event carrying ``branch_conditions``) is initialization, not
    a process update, and the terminal's ``result_recorded`` line is counted as a
    terminal check instead."""

    count = 0
    for ev in events:
        if ev.actor_id:
            continue
        payload = ev.payload_dict
        if ev.kind == "release_data" and "branch_conditions" in payload:
            continue
        if str(payload.get("event_type", "")) == "result_recorded":
            continue
        count += 1
    return count


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
        },
        "actions": {"started": 0, "completed": 0, "failed": 0, "rejected": 0},
        "messages": {"sent": 0, "delivered": 0, "noticed": 0, "missed": 0},
        "process_updates": 0,
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
        for key in ("started", "completed", "failed", "rejected"):
            totals["actions"][key] += int(report["actions"][key])
        for key in ("sent", "delivered", "noticed", "missed"):
            totals["messages"][key] += int(report["messages"][key])
        totals["process_updates"] += int(report["process_updates"])
        for key in ("pre_simulation", "final_recorded"):
            totals["terminal_checks"][key] += int(report["terminal_checks"][key])
    return totals


def _parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
