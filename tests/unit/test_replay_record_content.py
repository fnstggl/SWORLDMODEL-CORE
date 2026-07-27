"""FD-42: the replay reconstructs record CONTENT, not merely how many records exist.

``ReplayWorld`` used to answer :func:`~sworldmodel.expressions.evaluate`'s
``get_records`` with N *empty dicts* — the cardinality was right and every field of
every record was blank. A cardinality terminal (``count('positions') >= 5``) replayed
exactly. A content-predicated one — ``count('positions', equals(item('value'),
'hold')) >= 5``, which is the shape every actor world in this repo uses — evaluated
over blanks, counted zero, and answered NO with full confidence.

Two things rested on that, and both are tested here end to end rather than at the unit
level, because a unit test of ``get_records`` would have passed in both worlds:

* **the forensic tool** (``scripts/forensics.py``) calls the same
  :func:`~sworldmodel.replaycore.reconstruct_run`, so a sound run whose terminal reads
  record content reconstructed as NO and was reported as a CRITICAL
  ``simulation_probability`` / ``resolved_yes_mass`` mismatch, verdict
  ``FORENSICALLY_INVALID`` — a false invalid against a run that was fine;
* **the D6 responsibility gate** cannot evaluate its mandatory deletion
  counterfactuals over blank records, so it fails closed on every such terminal.

The load-bearing test is :func:`test_a_content_predicated_terminal_replays_to_the_live_answer`:
run a real world, replay its own ledger, and require the replayed terminal to equal the
answer the live engine produced — for every branch, including the genuine YES.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import scheduled_multiparty_world
from sworldmodel import replaycore
from sworldmodel.compiled import CompiledWorld
from sworldmodel.engine import RunResult, run
from sworldmodel.models import ResolutionContract
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")

# The terminal every world below resolves through. Reproduced from
# `_worlds.scheduled_multiparty_world` so the reader can see the shape under test:
#
#   yes_when:        count('positions', equals(item('value'), 'hold')) >= 5
#   unresolved_when: count('positions') < 5
#
# The first argument is a collection name and the second is a where-predicate over each
# record. `item('value')` is exactly what a blank placeholder cannot answer.


# --------------------------------------------------------------------------- #
# A world whose answer genuinely depends on what the actors recorded
# --------------------------------------------------------------------------- #


def _split_world() -> dict[str, Any]:
    """The multiparty world, plus one uncertainty that splits it into a YES and a NO.

    The external measurement is released after the cutoff, so the run enumerates a
    high branch and a low branch. The actors read the released level and record
    ``change`` or ``hold`` accordingly, which is what makes one branch answer YES and
    the other NO — the answer is in the record VALUES, not in how many there are.
    """

    data = scheduled_multiparty_world()
    data["uncertainties"] = [
        {
            "variable": "external_signal",
            "why_unknown": "the measurement is published after the cutoff",
            "reversal_capable": True,
            "release_at": "2026-06-09T12:00:00+00:00",
            "outcomes": [
                {
                    "value": "high",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 9.0]],
                },
                {
                    "value": "low",
                    "weight": 0.5,
                    "provenance": "symmetric_ignorance_assumption",
                    "field_effects": [["external_signal", 1.0]],
                },
            ],
        }
    ]
    return data


def _signal_sensitive(ctx: dict[str, Any]) -> dict[str, Any]:
    if ctx["stage"] != "session":
        return wait_decision()
    signal = float(ctx.get("observed_fields", {}).get("external_signal", 0) or 0)
    return act("record_position", {"position": "hold" if signal < 5 else "change"})


def _run_live(data: dict[str, Any]) -> tuple[CompiledWorld, RunResult]:
    """Compile and execute the world for real, through the production engine."""

    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="Will every member record hold?",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )
    gateway = ProgrammableGateway(
        {
            "actor_decision": _signal_sensitive,
            "reflect": {"beliefs_update": [], "new_memories": []},
        }
    )
    compiled = compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=8,
    )
    return compiled, run(compiled, gateway, seed=0)


def _live_answer(branch: Any) -> str:
    return str(branch.outcome) if branch.resolved else "UNRESOLVED"


# --------------------------------------------------------------------------- #
# The property: a replayed terminal equals the terminal the live run evaluated
# --------------------------------------------------------------------------- #


def test_a_content_predicated_terminal_replays_to_the_live_answer() -> None:
    """Run it, replay it, and require the same answer — per branch, YES branch included.

    Before FD-42 was fixed this failed on exactly one branch: the low branch, where
    five members each recorded ``hold``, answered YES live and NO on replay. The high
    branch agreed even with the bug, because its honest answer is also NO — which is
    precisely why a NO-vs-NO test would have proven nothing.
    """

    compiled, result = _run_live(_split_world())
    terminal = replaycore.terminal_ast_from_world(compiled.spec)
    initials = replaycore.initial_fields_from_world(compiled.spec) or {}
    by_branch = replaycore.group_events_by_branch(result.event_ledger)

    # The world is not degenerate: the actors really did split it.
    live = {b.branch_id: _live_answer(b) for b in result.branch_outcomes}
    assert set(live.values()) == {"YES", "NO"}

    replayed: dict[str, str | None] = {}
    for branch in result.branch_outcomes:
        events = by_branch.get(replaycore.branch_key(branch.branch_id), [])
        replayed[branch.branch_id] = replaycore.counterfactual_outcome(
            events,
            lambda _e: True,
            initial=replaycore.branch_initial_state(initials, events),
            terminal=terminal,
            rendered={},
        )
    assert replayed == live


def test_the_replay_reads_the_recorded_values_not_a_count_of_blanks() -> None:
    """The YES branch is decided by the values; the same cardinality with other values
    must answer NO. This is the difference the blank placeholders erased."""

    compiled, result = _run_live(_split_world())
    terminal = replaycore.terminal_ast_from_world(compiled.spec)
    initials = replaycore.initial_fields_from_world(compiled.spec) or {}
    by_branch = replaycore.group_events_by_branch(result.event_ledger)

    yes_branch = next(b for b in result.branch_outcomes if b.outcome == "YES")
    no_branch = next(b for b in result.branch_outcomes if b.outcome == "NO")
    yes_events = by_branch[replaycore.branch_key(yes_branch.branch_id)]
    no_events = by_branch[replaycore.branch_key(no_branch.branch_id)]

    def appended(events: list[Any]) -> list[Any]:
        return [
            replaycore.event_payload(e).get("value")
            for e in events
            if replaycore.event_kind(e) == "append_record"
        ]

    # Same number of records in both branches — only the values differ.
    assert len(appended(yes_events)) == len(appended(no_events)) == 5
    assert set(appended(yes_events)) == {"hold"}
    assert set(appended(no_events)) == {"change"}

    def replay(events: list[Any]) -> str | None:
        return replaycore.counterfactual_outcome(
            events,
            lambda _e: True,
            initial=replaycore.branch_initial_state(initials, events),
            terminal=terminal,
            rendered={},
        )

    assert replay(yes_events) == "YES"
    assert replay(no_events) == "NO"


def test_get_records_returns_the_recorded_key_value_author_and_time() -> None:
    """Every attribute ``item(<attr>)`` can read comes from the recorded payload."""

    _compiled, result = _run_live(_split_world())
    events = replaycore.group_events_by_branch(result.event_ledger)
    yes = next(b for b in result.branch_outcomes if b.outcome == "YES")
    state = replaycore.replay(events[replaycore.branch_key(yes.branch_id)])

    records = state.records["positions"]
    assert len(records) == 5
    assert [r["value"] for r in records] == ["hold"] * 5
    assert sorted(r["key"] for r in records) == [f"member_{i}" for i in range(5)]
    # `by` is the actor the ledger attributed the event to, and `time` is when it
    # happened — both read straight from the record, neither invented.
    assert sorted(r["by"] for r in records) == [f"member_{i}" for i in range(5)]
    assert all(r["time"] for r in records)
    # A collection nothing ever appended to is empty, not a fabricated placeholder.
    assert state.records.get("collection_that_never_existed") is None

    world = replaycore.ReplayWorld(state.fields, state.counts, records=state.records)
    assert world.get_records("positions") == records
    assert world.get_records("collection_that_never_existed") == []


def test_a_deletion_counterfactual_still_removes_the_records_it_deletes() -> None:
    """Records follow the keep-predicate, so the counterfactual machine still works.

    A reconstruction that returned the full record set regardless of ``keep`` would
    make every deletion counterfactual answer YES and quietly destroy the
    responsibility classification. Deleting the actors' output must flip the answer.
    """

    compiled, result = _run_live(_split_world())
    terminal = replaycore.terminal_ast_from_world(compiled.spec)
    initials = replaycore.initial_fields_from_world(compiled.spec) or {}
    by_branch = replaycore.group_events_by_branch(result.event_ledger)
    yes = next(b for b in result.branch_outcomes if b.outcome == "YES")
    events = by_branch[replaycore.branch_key(yes.branch_id)]
    initial = replaycore.branch_initial_state(initials, events)

    def outcome(keep: replaycore.KeepPredicate) -> str | None:
        return replaycore.counterfactual_outcome(
            events, keep, initial=initial, terminal=terminal, rendered={}
        )

    assert outcome(lambda _e: True) == "YES"
    # With every actor-produced event deleted no position is recorded at all, so the
    # compiled unresolved_when (`count('positions') < 5`) holds: the branch does not
    # answer NO, it stops having an answer.
    assert outcome(lambda e: not replaycore.event_actor_id(e)) == "UNRESOLVED"
    assert outcome(lambda _e: False) == "UNRESOLVED"

    # And the records really are gone from the counterfactual state.
    without_actors = replaycore.replay(events, lambda e: not replaycore.event_actor_id(e))
    assert without_actors.records.get("positions") is None


# --------------------------------------------------------------------------- #
# The consequence: the forensic tool no longer reports a false invalid
# --------------------------------------------------------------------------- #


def _forecast_payload(result: RunResult) -> dict[str, Any]:
    """The run's published forecast, in the shape ``forecast.json`` records it."""

    branches = [
        {
            "branch_id": b.branch_id,
            "weight": b.weight,
            "resolved": b.resolved,
            "outcome": b.outcome,
            "conditions": dict(b.key_conditions),
            "world_state": dict(b.records),
            "event_count": b.event_count,
        }
        for b in result.branch_outcomes
    ]
    yes = sum(float(b["weight"]) for b in branches if b["outcome"] == "YES")
    no = sum(float(b["weight"]) for b in branches if b["outcome"] == "NO")
    resolved_mass = yes + no
    return {
        "question": "Will every member record hold?",
        "simulation_probability": (yes / resolved_mass) if resolved_mass else None,
        "resolved_yes_mass": yes,
        "resolved_no_mass": no,
        "unresolved_mass": 1.0 - resolved_mass,
        "branches": branches,
    }


def test_forensic_reconstruction_of_a_content_predicated_run_is_not_a_false_invalid() -> None:
    """The exact path ``scripts/forensics.py`` takes, over a run that is entirely sound.

    ``scripts/forensics.py`` is a consumer of :func:`replaycore.reconstruct_run` (D7:
    one ledger-replay implementation). Before the fix this reconstruction turned the
    genuine YES branch into a NO, which made ``resolved_yes_mass`` and
    ``simulation_probability`` disagree with what the run published — both CRITICAL,
    both ``invalidates_result`` — and the tool declared a sound run
    ``FORENSICALLY_INVALID``.
    """

    compiled, result = _run_live(_split_world())
    record = replaycore.RunRecord(
        label="content_predicated",
        run_dir="(test)",
        forecast=_forecast_payload(result),
        events=list(result.event_ledger),
        decisions=list(result.actor_decisions),
        world=compiled.spec,
    )

    reconstruction = replaycore.reconstruct_run(record)

    assert reconstruction["verdict"] == "RECONSTRUCTED"
    assert [m for m in reconstruction["mismatches"] if m["invalidates_result"]] == []
    assert reconstruction["responsibility"]["classification"] != "INVALID_TRACE"
    assert reconstruction["reconstruction_notes"] == []
    # Every branch reproduces, and the published probability recomputes exactly.
    assert all(b["matches_published"] for b in reconstruction["branches"])
    assert reconstruction["recomputed"]["probability"] == reconstruction["published"]["probability"]
    # The run is not trivially reproducible: the actors' output decided an answer.
    assert reconstruction["responsibility"]["any_actor_output_changed_an_outcome"]


def test_the_reconstruction_reads_the_run_back_from_plain_json_identically() -> None:
    """Years later there are no live objects — only the artifact dicts on disk."""

    compiled, result = _run_live(_split_world())
    ledger = [
        {
            "event_id": e.event_id,
            "branch_id": e.branch_id,
            "time": e.time.isoformat(),
            "kind": e.kind,
            "actor_id": e.actor_id,
            "payload": dict(e.payload_dict),
            "visibility": e.visibility.value,
            "evidence_claim_ids": list(e.evidence_claim_ids),
        }
        for e in result.event_ledger
    ]
    forecast = _forecast_payload(result)

    live = replaycore.reconstruct_run(
        replaycore.RunRecord(
            label="live",
            run_dir="(test)",
            forecast=forecast,
            events=list(result.event_ledger),
            world=compiled.spec,
        )
    )
    # `_split_world()` is itself the compiled-world dict shape that a run persists as
    # compiled_world.json, so this is the on-disk reader end to end.
    from_disk = replaycore.reconstruct_run(
        replaycore.RunRecord(
            label="disk",
            run_dir="(test)",
            forecast=forecast,
            events=ledger,
            world=_split_world(),
        )
    )

    assert from_disk["verdict"] == live["verdict"] == "RECONSTRUCTED"
    assert [b["recomputed_outcome"] for b in from_disk["branches"]] == [
        b["recomputed_outcome"] for b in live["branches"]
    ]
    assert [b["counterfactuals"] for b in from_disk["branches"]] == [
        b["counterfactuals"] for b in live["branches"]
    ]


# --------------------------------------------------------------------------- #
# The rendered-string fallback, for runs that persisted no executable world
# --------------------------------------------------------------------------- #


_RENDERED = {
    "yes_when": "greater_or_equal(count('positions', equals(item('value'), 'hold')), 5)",
    "unresolved_when": "less_than(count('positions'), 5)",
}


def test_the_rendered_terminal_grammar_evaluates_a_where_predicate() -> None:
    """A run that persisted no ``compiled_world.json`` replays through the declared
    subset of the rendered grammar, and it reads record content there too."""

    _compiled, result = _run_live(_split_world())
    by_branch = replaycore.group_events_by_branch(result.event_ledger)
    for branch in result.branch_outcomes:
        state = replaycore.replay(by_branch[replaycore.branch_key(branch.branch_id)])
        resolved, outcome, how = replaycore.reevaluate_terminal(
            None, _RENDERED, state.fields, state.counts, records=state.records
        )
        assert resolved
        assert outcome == _live_answer(branch)
        assert "rendered terminal string" in how


def test_the_rendered_grammar_refuses_a_predicate_it_has_no_records_for() -> None:
    """Silently dropping the predicate would return the raw cardinality and call a NO
    a YES. Without the records the honest answer is that it cannot be reconstructed."""

    try:
        replaycore.reevaluate_terminal(None, _RENDERED, {}, {"positions": 5})
    except replaycore.Unreconstructable as exc:
        assert "record content" in str(exc)
    else:  # pragma: no cover - the assertion is the point of the test
        raise AssertionError("a where-predicate with no reconstructed records must refuse")
