"""The branch clock: a time-ordered queue of everything that is going to happen.

The runtime advances **event to event on the real calendar**. There is no tick, no
round, no "T1/T2", and no fixed per-stage actor turn. Every entry carries the exact
branch ``datetime`` at which it occurs and a provenance saying *why it is scheduled*.

Entries come from exactly four places, and nowhere else:

``compiled_process``      a node of the compiled process graph, at its compiled time
``compiled_external``     an external (non-agent) process: a data release, a deadline,
                          a publication, an administrative step
``consequence``           a consequence of something that already happened: an action
                          completing, information becoming available/noticed, a
                          commitment falling due, a follow-up an action scheduled
``actor_plan``            an actor's own planned future action or self-set revisit

Ordering is **insertion-order invariant**: entries are ordered by
``(at, microstep, content_key)``. Two runs that queue the same set of things in a
different order execute them in the same order, which is what makes the trace
replayable. ``pop_batch`` returns *every* entry sharing the earliest timestamp, so
simultaneity is modeled as simultaneity — actors in the same batch act without having
seen each other.

Entries past the horizon are never executed; they are retained and reported, so a
branch that ran out of world before it ran out of question says so instead of being
quietly completed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .errors import WorldIntegrityError
from .ids import content_id

# Why an entry exists. Provenance is recorded on every entry and surfaced in the trace.
ORIGIN_PROCESS = "compiled_process"
ORIGIN_EXTERNAL = "compiled_external"
ORIGIN_CONSEQUENCE = "consequence"
ORIGIN_ACTOR_PLAN = "actor_plan"

VALID_ORIGINS = frozenset({ORIGIN_PROCESS, ORIGIN_EXTERNAL, ORIGIN_CONSEQUENCE, ORIGIN_ACTOR_PLAN})

# The one entry kind with a dedicated ordering class: a branch's own scenario release
# (its hypothesis about what an uncertain value turns out to be when it becomes
# public). It fires strictly AFTER every other entry scheduled at its instant — see
# ``pop_batch`` — so the branch's defining condition can never be overwritten by a
# same-instant compiled placeholder. Before this class existed, a deferred
# ``at``-stamped placeholder release tied with the hypothesis at the same
# ``(at, microstep)`` and the winner fell to content-hash order of entry ids: two
# spelled-differently-but-equivalent compiled worlds produced opposite branch states.
KIND_SCENARIO_RELEASE = "scenario_release_due"


# ---------------------------------------------------------------------------
# How many actor calls a world of this size, over a window of this length, may spend
# ---------------------------------------------------------------------------
#
# The bound lives here because it is a property of the branch's *window*: the runtime
# advances event to event over a real calendar, and how much can legitimately happen in
# that window scales with how long it is and how many people are in it. A flat constant
# gave a nine-participant world over ten weeks exactly the budget of a two-participant
# world over two weeks (`artifacts/ab/individual_semantic` and
# `artifacts/phase2/geopolitical2` are both nine- and two-participant worlds that hit
# the same 80), which is not a bound on the world, it is a bound on nothing in
# particular.
#
# These are POLICY, not a model of behaviour. Nothing here predicts how often a person
# decides — invocation count is an output of the trajectory, and an actor that nothing
# reaches is never invoked at all. They are the point past which we would rather report
# an incomplete run than keep spending model calls, and reaching one is a statement
# about the simulator, never about the world.
ACTOR_CALLS_PER_ACTOR_PER_WEEK = 4

# The floor is the historical flat budget. Keeping it as the floor rather than replacing
# it means no world gets *less* than it had: the change can only give a bigger world
# more room, never take room from a small one.
ACTOR_CALLS_FLOOR = 80

# The outermost stop. A compile that would ask for tens of thousands of model calls per
# branch has a defect the budget should not fund.
ACTOR_CALLS_CEILING = 1000


def actor_call_budget(*, participants: int, horizon_days: float) -> int:
    """The actor-call bound for a world of ``participants`` people over ``horizon_days``.

    Scales with participants × horizon, floored at the historical flat budget and capped.
    A window shorter than a week still counts as a week: a two-day question with nine
    participants is a dense two days, not a fractional one.
    """

    weeks = max(1.0, horizon_days / 7.0)
    scaled = math.ceil(max(0, participants) * weeks * ACTOR_CALLS_PER_ACTOR_PER_WEEK)
    return max(ACTOR_CALLS_FLOOR, min(ACTOR_CALLS_CEILING, int(scaled)))


def _ordering_class(entry: ScheduledEntry) -> int:
    """0 for ordinary entries, 1 for the scenario release — the outermost tie level
    within one instant, ahead of ``microstep``."""

    return 1 if entry.kind == KIND_SCENARIO_RELEASE else 0


@dataclass(frozen=True)
class ScheduledEntry:
    """One thing that will happen at one exact branch time.

    ``kind`` is structural (what the runtime must do), not a domain event type — the
    domain meaning lives in the compiled payload. ``microstep`` layers strictly causal
    dependencies that share a timestamp: a delivery caused by an action at time T is
    microstep 1 relative to that action's microstep 0, so it is ordered after it
    without needing a later clock reading.
    """

    at: datetime
    kind: str
    payload: tuple[tuple[str, Any], ...] = ()
    actor_id: str | None = None
    origin: str = ORIGIN_CONSEQUENCE
    origin_detail: str = ""
    causal_parents: tuple[str, ...] = ()
    microstep: int = 0

    @property
    def payload_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def entry_id(self) -> str:
        return content_id(
            "sched",
            self.at.isoformat(),
            self.kind,
            self.actor_id,
            self.origin,
            self.origin_detail,
            repr(self.payload),
        )

    def sort_key(self) -> tuple[Any, ...]:
        """Content-derived and insertion-order invariant. The ordering class sits
        ahead of ``microstep`` so a scenario release sorts — and fires — after every
        ordinary entry at its instant, never on an entry-id hash tie."""

        return (
            self.at,
            _ordering_class(self),
            self.microstep,
            self.kind,
            self.actor_id or "",
            self.entry_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "at": self.at.isoformat(),
            "kind": self.kind,
            "actor_id": self.actor_id,
            "origin": self.origin,
            "origin_detail": self.origin_detail,
            "causal_parents": list(self.causal_parents),
            "microstep": self.microstep,
            "payload": self.payload_dict,
        }


def make_entry(
    *,
    at: datetime,
    kind: str,
    payload: dict[str, Any] | None = None,
    actor_id: str | None = None,
    origin: str = ORIGIN_CONSEQUENCE,
    origin_detail: str = "",
    causal_parents: tuple[str, ...] = (),
    microstep: int = 0,
) -> ScheduledEntry:
    if origin not in VALID_ORIGINS:
        raise ValueError(f"unknown schedule origin {origin!r}")
    return ScheduledEntry(
        at=at,
        kind=kind,
        payload=tuple(sorted((payload or {}).items())),
        actor_id=actor_id,
        origin=origin,
        origin_detail=origin_detail,
        causal_parents=causal_parents,
        microstep=microstep,
    )


def _refuse_ambiguous_release_instant(entries: tuple[ScheduledEntry, ...]) -> None:
    """Refuse two scenario releases queued at one instant.

    The ordering class puts the scenario release last within its instant, which settles
    hypothesis-vs-compiled-placeholder collisions structurally. It does NOT order two
    hypothesis releases against *each other*: they share ``(at, class, microstep,
    kind)`` and fall through to entry-id hash order, so which of two announcements
    wrote the last value would be decided by a content hash — the exact defect the
    ordering class was introduced to remove, one refactor away from returning.

    The branch builder groups a scenario's releases by moment precisely so this cannot
    arise (:meth:`sworldmodel.uncertainty.Scenario.dated_releases`). If it ever does,
    the run stops here rather than publishing a hash-decided world.
    """

    seen: dict[datetime, ScheduledEntry] = {}
    for e in entries:
        if e.kind != KIND_SCENARIO_RELEASE:
            continue
        other = seen.get(e.at)
        if other is not None:
            raise WorldIntegrityError(
                "two scenario releases are queued at the same instant "
                f"({e.at.isoformat()}); which one writes the branch's final value "
                "would be decided by entry-id hash order, not by the world. Group a "
                "branch's releases by release moment before scheduling them.",
                details={
                    "at": e.at.isoformat(),
                    "first": other.origin_detail,
                    "second": e.origin_detail,
                },
            )
        seen[e.at] = e


@dataclass(frozen=True)
class Schedule:
    """An immutable, deterministically ordered set of pending entries.

    Immutability matters: a branch clone must not share a mutable heap with its
    parent, and the schedule is part of the branch state that the trace serializes.
    """

    entries: tuple[ScheduledEntry, ...] = ()
    fired: frozenset[str] = field(default_factory=frozenset)

    def push(self, *new: ScheduledEntry) -> Schedule:
        """Add entries, ignoring exact duplicates and anything already fired.

        De-duplication is what stops two independent causes from waking the same actor
        twice for the same reason at the same instant; it never merges entries that
        differ in time, trigger or payload.
        """

        pending = {e.entry_id for e in self.entries}
        added = [e for e in new if e.entry_id not in pending and e.entry_id not in self.fired]
        if not added:
            return self
        merged = tuple(sorted(self.entries + tuple(added), key=lambda e: e.sort_key()))
        if any(e.kind == KIND_SCENARIO_RELEASE for e in added):
            _refuse_ambiguous_release_instant(merged)
        return replace(self, entries=merged)

    def drop(self, predicate: Any) -> Schedule:
        """Remove pending entries matching ``predicate`` (used when a plan is abandoned
        so its self-scheduled revisits do not fire for a plan that no longer exists)."""

        kept = tuple(e for e in self.entries if not predicate(e))
        if len(kept) == len(self.entries):
            return self
        return replace(self, entries=kept)

    def next_time(self, *, horizon: datetime) -> datetime | None:
        for e in self.entries:
            if e.at <= horizon:
                return e.at
        return None

    def pop_batch(self, *, horizon: datetime) -> tuple[Schedule, tuple[ScheduledEntry, ...]]:
        """Pop the next causal layer: the earliest in-horizon timestamp, and within it
        the earliest ``(ordering class, microstep)``.

        Entries in a returned batch are genuinely *simultaneous and independent* — an
        actor in the batch has not seen the others' results. Entries that causally
        depend on something at the same timestamp carry a higher microstep and are held
        back to the next layer, so a message delivered by an action at time T is
        noticed strictly after that action, without needing the clock to move.

        The scenario release (``KIND_SCENARIO_RELEASE``) is held back until every
        ordinary entry at its instant has fired: the branch's hypothesis states what
        the released value *turned out to be*, so nothing else scheduled at that same
        moment — in particular a compiled placeholder release of the same fields — may
        land after it and overwrite the branch's defining condition. Without the
        dedicated class the collision was decided by entry-id hash order.
        """

        t = self.next_time(horizon=horizon)
        if t is None:
            return self, ()
        at_t = [e for e in self.entries if e.at == t]
        layer = min((_ordering_class(e), e.microstep) for e in at_t)
        batch = tuple(e for e in at_t if (_ordering_class(e), e.microstep) == layer)
        taken = {e.entry_id for e in batch}
        rest = tuple(e for e in self.entries if e.entry_id not in taken)
        return replace(self, entries=rest, fired=self.fired | taken), batch

    def drop_matching(self, *, kind: str, actor_id: str, at: datetime) -> Schedule:
        """Discard pending entries of one kind for one actor at one instant.

        Used when several independent causes ask for the same actor at the same moment:
        the actor decides once, having been told every reason, rather than being asked
        the same question twice at the same timestamp.
        """

        return self.drop(lambda e: e.kind == kind and e.actor_id == actor_id and e.at == at)

    def beyond_horizon(self, *, horizon: datetime) -> tuple[ScheduledEntry, ...]:
        """Entries that will never run because the question's window closed first."""

        return tuple(e for e in self.entries if e.at > horizon)

    def pending_count(self, *, horizon: datetime) -> int:
        return sum(1 for e in self.entries if e.at <= horizon)

    def pending_at(self, moment: datetime) -> int:
        """How much is still queued at one exact instant.

        The difference between a queue that is draining and one that is refilling
        itself. A finite burst of simultaneous work — everyone reading the notes that
        were just circulated — drives this strictly down. A runaway cascade holds it up
        or grows it, because every item it handles schedules another at the same instant.
        """

        return sum(1 for e in self.entries if e.at == moment)

    def __len__(self) -> int:
        return len(self.entries)
