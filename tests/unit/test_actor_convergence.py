"""W3 — convergence: an agent that can see what it did, and does not wake on its echo.

**These tests require patches to `engine.py`, `prompts.py` and `cli.py` that are handed
to the CTO separately** (those files were owned by other agents while this work was
done). Until those land, the integration tests here FAIL. They are red on purpose: they
were written before the fix, they reproduce the defect from
`artifacts/ab/individual_semantic` in a compiled world rather than by replaying an
artifact, and the correct response to a failure is to apply the patch — never to relax
an assertion. Every threshold below is stated as the property it protects.

The defect, measured on this repository before the change: one branch, Andrew Bailey
invoked 51 times, 43 of those wakes `directed_information`, twenty producing the
identical signal, the entire 80-call budget consumed and the branch reported
`resolved=False` — while the terminal it was asked about had been satisfied twenty
times over.

Two independent causes, and both are needed:

1. **The agent was woken by its own act.** A compiled effect addresses an event to its
   own participants, the acting agent is a participant in its own act, and the runtime's
   existing guard (`WorldState.observers_of` skips an actor for its own public act) is
   defeated by that audience. Act -> event -> delivered to self -> noticed -> wake -> act.
2. **The agent could not see that it had already acted.** A view's observations come
   from deliveries, and an actor is never the recipient of its own delivery, so an
   agent's own acts reached it through no channel at all. The only trace of any of them
   was one overwritten `current_action` slot. It was not being stubborn; it was blind.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import AS_OF as AS_OF_S
from _worlds import HORIZON as HORIZON_S
from sworldmodel.actors import (
    ACTION_FAILED,
    ACTION_REJECTED,
    ActedRecord,
    ActorState,
    LocalView,
    act_counts,
    check_non_decision,
    intent_key,
    is_self_echo,
    situation_key,
)
from sworldmodel.compiled import CompiledWorld
from sworldmodel.engine import RunResult, run
from sworldmodel.memory import MemoryStream
from sworldmodel.models import ResolutionContract
from sworldmodel.world_compiler import compile_world
from sworldmodel.worldspec import ActionChoice

AS_OF = datetime.fromisoformat(AS_OF_S)
HORIZON = datetime.fromisoformat(HORIZON_S)

WINDOW_1 = "2026-06-01T09:00:00+00:00"
WINDOW_2 = "2026-06-10T09:00:00+00:00"


# ---------------------------------------------------------------------------
# One compiled world, exercised by different scripted behaviours
# ---------------------------------------------------------------------------


def _claim(cid: str, proposition: str) -> dict[str, Any]:
    return {"id": cid, "proposition": proposition, "value": True, "supporting_excerpt": proposition}


def convergence_world(*, second_window: bool = True) -> dict[str, Any]:
    """A governor who can speak publicly, and a deputy who can answer.

    Two things about this world matter and both are copied from what the real compiler
    emits, not invented for the test:

    * ``signal_support``'s ``create_event`` is addressed **to the governor itself**
      (``to: ["governor"]``). That is what `semantic_lowering` produces — it sets an
      event's audience to the event's own participants, and the actor taking the action
      is one of its participants. It is the exact shape that consumed the Bailey branch.
    * the wake rule wakes the governor *and* the deputy on that event, which is what the
      compiler's "an event wakes its deciding participants" rule produces. So the echo
      arrives by two independent routes — the audience and a compiled rule — and a fix
      that only closed one of them would look like it worked.

    ``knock`` exists to produce a rejection cascade: its precondition never holds, so the
    world refuses it every time, and each refusal is real news that legitimately wakes the
    governor. That cascade is not self-echo, which is what makes it the right probe for
    the non-decision check.
    """

    nodes = [
        {
            "node_id": "window_1",
            "stage": "window",
            "at": WINDOW_1,
            "description": "the first window in which the governor may speak",
            "participants": ["governor"],
            "action_ids": ["signal_support", "knock"],
        }
    ]
    if second_window:
        nodes.append(
            {
                "node_id": "window_2",
                "stage": "window",
                "at": WINDOW_2,
                "description": "a second, separate window nine days later",
                "participants": ["governor"],
                "action_ids": ["signal_support"],
            }
        )
    return {
        "reality": {
            "as_of": AS_OF_S,
            "horizon": HORIZON_S,
            "subject_entity": "the governor",
            "resolution_units": "public signals",
            "target_outcome": "the governor signals support",
            "expected_participants": 2,
        },
        "claims": [
            _claim("c_gov", "the governor speaks publicly on policy"),
            _claim("c_dep", "the deputy answers the governor in public"),
            _claim("c_window", "the policy windows are scheduled"),
        ],
        "world_spec": {
            "title": "convergence probe",
            "subject_entity": "the governor",
            "resolution_units": "public signals",
            "entities": [
                {
                    "entity_id": "governor",
                    "name": "The Governor",
                    "kind": "person",
                    "is_actor": True,
                    "role": "governor",
                    "authority": ["signal"],
                    "representation_scale": "individual",
                    "evidence_claim_ids": ["c_gov"],
                },
                {
                    "entity_id": "deputy",
                    "name": "The Deputy",
                    "kind": "person",
                    "is_actor": True,
                    "role": "deputy",
                    "authority": ["respond"],
                    "representation_scale": "individual",
                    "evidence_claim_ids": ["c_dep"],
                },
            ],
            "actors": [
                {
                    "entity_id": "governor",
                    "reasoning": "states a view when a window opens",
                    "memory_seeds": [
                        {
                            "content": "I have spoken at previous windows.",
                            "kind": "episodic",
                            "importance": 0.7,
                            "evidence_claim_ids": ["c_gov"],
                        }
                    ],
                },
                {
                    "entity_id": "deputy",
                    "reasoning": "answers the governor in public",
                    "memory_seeds": [
                        {
                            "content": "I have answered publicly at previous windows.",
                            "kind": "episodic",
                            "importance": 0.7,
                            "evidence_claim_ids": ["c_dep"],
                        }
                    ],
                },
            ],
            "fields": [{"field_id": "door_open", "value_type": "bool", "initial": False}],
            "actions": [
                {
                    "action_id": "signal_support",
                    "meaning": "publicly signal support",
                    "eligible_actors": ["governor"],
                    "required_authority": ["signal"],
                    "visibility": "public",
                    "effects": [
                        {
                            "op": "create_event",
                            "event_type": "governor_signals_support",
                            "text": "the governor signals support",
                            "to": ["governor"],
                        },
                        {
                            "op": "append_record",
                            "collection": "signals",
                            "key": "$actor",
                            "value": "support",
                        },
                    ],
                    "evidence_claim_ids": ["c_gov"],
                },
                {
                    "action_id": "comment",
                    "meaning": "say something back to the governor",
                    "eligible_actors": ["deputy"],
                    "required_authority": ["respond"],
                    "parameters": [{"name": "text", "type": "string", "required": True}],
                    "visibility": "public",
                    "effects": [
                        {"op": "deliver_information", "to": ["governor"], "text": "$param.text"}
                    ],
                    "evidence_claim_ids": ["c_dep"],
                },
                {
                    "action_id": "knock",
                    "meaning": "try a door that is shut",
                    "eligible_actors": ["governor"],
                    "required_authority": ["signal"],
                    "visibility": "public",
                    "preconditions": {
                        "op": "equals",
                        "args": [{"op": "field", "args": ["door_open"]}, True],
                    },
                    "effects": [
                        {"op": "create_event", "event_type": "door_used", "text": "went through"}
                    ],
                    "evidence_claim_ids": ["c_gov"],
                },
            ],
            "process": {"nodes": nodes},
            "wake_rules": [
                {
                    "rule_id": "the_room_hears_the_governor",
                    "wakes": ["governor", "role:deputy"],
                    "reason": "the governor has spoken",
                    "on_event_type": "governor_signals_support",
                }
            ],
            "terminal": {
                "yes_when": {
                    "op": "greater_than",
                    "args": [{"op": "event_count", "args": ["governor_signals_support"]}, 0],
                },
                "unresolved_when": {"op": "const", "args": [False]},
                "description": "YES when the governor signals support at least once",
            },
        },
    }


def _gateway(decision: Any) -> ProgrammableGateway:
    return ProgrammableGateway(
        {"actor_decision": decision, "reflect": {"beliefs_update": [], "new_memories": []}}
    )


def _compile(data: dict[str, Any], gateway: ProgrammableGateway) -> CompiledWorld:
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="does the governor signal support?",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        required_reality_facts=bundle.required_reality_facts,
        expected_participants=bundle.expected_participants,
    )
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=4,
    )


def _run(data: dict[str, Any], decide: Any) -> RunResult:
    gw = _gateway(decide)
    return run(_compile(data, gw), gw, seed=0)


def _diag(result: RunResult) -> Any:
    return next(iter(result.diagnostics.values()))


def _calls(result: RunResult) -> int:
    return sum(_diag(result).actor_call_counts.values())


# ---------------------------------------------------------------------------
# 1. The spin itself: bounded invocations, and the branch reaches its terminal
# ---------------------------------------------------------------------------


def test_an_actor_is_not_driven_by_the_consequences_of_its_own_action() -> None:
    """The `artifacts/ab/individual_semantic` defect, reproduced and closed.

    A governor who signals whenever it is asked. Its signal creates an event addressed to
    itself, and a compiled rule names it too — so before the fix the act woke its own
    author twice over, and the author acted again. Measured on this exact world before
    the change: **80 actor calls, 79 of them `directed_information`, one
    `process_opportunity`, 239 batches, stop reason "actor-call budget exhausted (80)",
    one scheduled entry still due, `resolved=False`.** The answer the terminal asks about
    had happened on the very first call.

    Both halves are asserted, because either alone is a smaller failure rather than a
    fix: the invocations must be bounded, AND the branch must reach its terminal. A
    quiet branch that still reports "we stopped watching" is not convergence.
    """

    result = _run(
        convergence_world(),
        lambda ctx: (
            act("signal_support") if ctx["actor_id"] == "governor" else wait_decision("not mine")
        ),
    )
    diag = _diag(result)

    # Two compiled windows, one governor, one deputy who never speaks. Anything near the
    # old 80 is the loop; the honest number is a small multiple of the real occasions.
    assert _calls(result) <= 10, (
        f"the branch is still spinning: {diag.actor_call_counts} — {diag.stop_reason}"
    )
    assert diag.stop_reason == "schedule exhausted", (
        f"the branch did not run out of world, it ran out of us: {diag.stop_reason}"
    )
    assert diag.unfired_in_horizon == 0, "the branch left in-horizon work unfired"

    (outcome,) = result.branch_outcomes
    assert outcome.resolved and outcome.outcome == "YES", (
        "the branch did not reach its terminal — bounded invocations with an unresolved "
        f"branch is a smaller failure, not a fixed one ({outcome.unresolved_reason})"
    )

    # Not one wake was caused by the governor's own act...
    for d in result.actor_decisions:
        own = [
            e
            for e in result.event_ledger
            if e.event_id in d.trigger_event_ids and e.actor_id == d.actor_id
        ]
        assert not own, f"{d.actor_id} was woken by its own event: {[e.kind for e in own]}"

    # ...and the suppression is of the WAKE alone. The governor was still delivered its
    # own event and still noticed it: visibility, delivery, notice and reconsideration
    # are four transitions, and only the last one is withheld.
    world = next(iter(result.final_worlds.values()))
    (signal,) = [
        e
        for e in world.event_history
        if str(e.payload_dict.get("event_type", "")) == "governor_signals_support"
    ][:1]
    mine = [
        d
        for d in world.deliveries
        if d.event_id == signal.event_id and d.actor_id == signal.actor_id
    ]
    assert mine and all(d.noticed_at is not None for d in mine), (
        "the actor's own act was hidden from it entirely; only the wake should be withheld"
    )


def test_the_run_diagnostics_tell_a_spinning_run_from_a_deliberating_one() -> None:
    """ "51 invocations" is the same number for a loop and for a deliberation.

    Whatever the runtime detects has to be visible, per branch and per actor, or a reader
    cannot tell the two apart — which is the state the audit found the artifacts in.
    """

    result = _run(
        convergence_world(),
        lambda ctx: (
            act("signal_support") if ctx["actor_id"] == "governor" else wait_decision("not mine")
        ),
    )
    conv = _diag(result).convergence.as_dict()

    assert conv["self_echo_wakes_suppressed"] > 0, (
        "the run withheld no self-echo wakes, so this world no longer probes the defect"
    )
    assert conv["by_actor"]["governor"]["self_echo_wakes_suppressed"] > 0, (
        "the diagnostics do not say WHICH participant was looping"
    )
    assert set(conv) == {
        "self_echo_wakes_suppressed",
        "non_decision_wakes_refused",
        "repeat_decisions",
        "by_actor",
    }


# ---------------------------------------------------------------------------
# 2. A genuine repetition survives
# ---------------------------------------------------------------------------


def test_the_same_act_twice_with_something_real_in_between_is_two_decisions() -> None:
    """A central banker really can signal support twice.

    Two separate compiled windows nine days apart, and between them the deputy answers.
    Both signals must stand, both decisions must be recorded, and neither may be counted
    as a repeat or refused. The line is not how many times an actor does something — it
    is whether anything happened in between.

    Each participant speaks once per occasion here, deliberately: an actor scripted to
    answer every wake with the identical sentence at one instant is a ping-pong loop, and
    a loop is what the OTHER tests probe. This one has to be a world where the repetition
    is genuine, or it proves nothing about genuine repetition.
    """

    spoke: set[str] = set()

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        occasion = f"{ctx['actor_id']}@{ctx['branch_time']}"
        if occasion in spoke:
            return wait_decision("I have said my piece for now")
        if ctx["actor_id"] == "governor":
            spoke.add(occasion)
            return act("signal_support")
        if any("governor" in str(o.get("summary", "")) for o in ctx["observations"]):
            spoke.add(occasion)
            return act("comment", {"text": "noted, governor"})
        return wait_decision("nothing to answer")

    result = _run(convergence_world(), decide)
    diag = _diag(result)

    signals = [
        e
        for e in result.event_ledger
        if str(e.payload_dict.get("event_type", "")) == "governor_signals_support"
    ]
    assert len(signals) >= 2, (
        f"a genuine repetition was suppressed: only {len(signals)} signal(s) survived"
    )
    assert len({e.time for e in signals}) >= 2, "probe shape lost: both signals fell at one instant"
    assert any(
        e.actor_id == "deputy" for e in result.event_ledger if e.kind == "deliver_information"
    ), "probe shape lost: nothing real happened between the two signals"

    decided = [
        d
        for d in result.actor_decisions
        if d.actor_id == "governor" and d.intent.get("action_id") == "signal_support"
    ]
    assert len(decided) >= 2, "the second signal was never asked for"
    assert diag.convergence.repeat_decisions == 0, (
        "an act taken nine days later, after another participant spoke, was counted as a "
        "repeat of the first"
    )
    assert diag.convergence.non_decision_wakes_refused == 0, (
        "a genuine second occasion to act was refused as a non-decision"
    )
    for d in decided:
        conv = d.decision_context.get("convergence") or {}
        assert conv.get("repeat_of_previous_decision") is False


def test_a_refusal_can_never_straddle_two_instants() -> None:
    """The structural guarantee behind "a genuine repetition survives".

    Elapsed simulated time is always something having moved: a governor who signals in
    June and again in September has plausibly decided twice, and the second act means
    something different because the world aged around it. The situation key carries the
    clock precisely so that can never be mistaken for a loop — so a refusal can only ever
    duplicate a decision the same actor already took **at the same instant**.

    Asserted across every world in this module, because the property is about the
    detector, not about one probe.
    """

    for data, decide in (
        (
            convergence_world(),
            lambda ctx: (
                act("signal_support")
                if ctx["actor_id"] == "governor"
                else act("comment", {"text": "noted, governor"})
            ),
        ),
        (
            convergence_world(second_window=False),
            lambda ctx: act("knock") if ctx["actor_id"] == "governor" else wait_decision("no"),
        ),
    ):
        result = _run(data, decide)
        decided_at: dict[str, set[str]] = {}
        for d in result.actor_decisions:
            if d.validation_status == "non_decision":
                assert d.branch_time in decided_at.get(d.actor_id, set()), (
                    f"{d.actor_id} was refused at {d.branch_time} without having decided at "
                    "that instant — the detector reached across time, and time is information"
                )
            else:
                decided_at.setdefault(d.actor_id, set()).add(d.branch_time)


def test_a_repeat_from_a_world_that_has_not_moved_at_all_is_refused() -> None:
    """The other side of the same line, and the reason it is a line and not a ban.

    `knock` can never succeed: its precondition never holds. Each refusal is genuine news
    — the world's verdict on the attempt — so it legitimately wakes the governor, which
    is why this is NOT self-echo and why it is the right probe. The governor knocks
    again, is refused again, and is woken again: same instant, same stated cause, same
    readable world, same menu, same mind, nothing from anybody else.

    A second attempt still stands. What is refused is a further re-ask from a world that
    has not moved at all — and it is refused before any model call, so it costs nothing
    and invents nothing.
    """

    result = _run(
        convergence_world(second_window=False),
        lambda ctx: act("knock") if ctx["actor_id"] == "governor" else wait_decision("not mine"),
    )
    diag = _diag(result)

    attempts = [d for d in result.actor_decisions if d.intent.get("action_id") == "knock"]
    assert len(attempts) >= 2, (
        f"the governor was not allowed to try twice ({len(attempts)}); a second attempt "
        "at a closed door is a real act, not a loop"
    )
    assert diag.convergence.non_decision_wakes_refused >= 1, (
        "an unbounded rejection cascade ran to a budget or a watchdog instead of being "
        f"cut at its cause: {diag.stop_reason}"
    )
    refused = [d for d in result.actor_decisions if d.validation_status == "non_decision"]
    assert refused, "the refused wake left no record; a wake that vanishes is not honest"
    assert refused[0].intent == {}, "a refused wake must not carry a fabricated intention"
    assert refused[0].prompt_hash == "", "a refused wake must not have called the model"
    assert "nothing material has changed" in refused[0].validation_reason

    # And the branch survives it: the cascade is cut at its cause, not by the blunt
    # watchdog that kills a branch for looking frozen.
    assert "no progress" not in diag.stop_reason, diag.stop_reason


# ---------------------------------------------------------------------------
# 3. Another agent's reaction still wakes the actor
# ---------------------------------------------------------------------------


def test_an_agent_still_wakes_on_another_agents_reaction_to_its_own_act() -> None:
    """The boundary that makes this damping rather than deafening.

    The governor speaks; the deputy hears it and answers. That answer is a consequence
    of the governor's own action — it exists only because the governor acted — and it
    must still wake the governor, because its author is somebody else. Suppressing it
    would remove the social dynamic the runtime exists to simulate.
    """

    spoken: list[str] = []

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["actor_id"] == "governor":
            if not spoken:
                spoken.append("x")
                return act("signal_support")
            return wait_decision("said my piece")
        if any("governor" in str(o.get("summary", "")) for o in ctx["observations"]):
            return act("comment", {"text": "I disagree, governor"})
        return wait_decision("nothing to answer")

    result = _run(convergence_world(second_window=False), decide)

    (reply,) = [
        e for e in result.event_ledger if e.kind == "deliver_information" and e.actor_id == "deputy"
    ]
    woken = [
        d
        for d in result.actor_decisions
        if d.actor_id == "governor" and reply.event_id in d.trigger_event_ids
    ]
    assert woken, (
        "the governor was not woken by the deputy's reaction to its own act — the "
        "self-echo guard is deafening the agent instead of damping a loop"
    )
    assert "directed_information" in woken[0].wake_reason
    assert reply.event_id in woken[0].noticed_observation_ids


# ---------------------------------------------------------------------------
# 4. The agent can see what it has already done
# ---------------------------------------------------------------------------


def test_an_actor_sees_its_own_act_history_and_can_act_on_it() -> None:
    """The amnesia regression, and the reason the loop existed at all.

    Measured before this existed: `member_0` circulated a note fifteen times, and at its
    sixteenth invocation its view held eight observations mentioning it zero times and
    six memories mentioning it zero times. Its own acts reached it through no channel.

    Here the governor's rule is "if I have already signalled, wait" — a rule it can only
    follow if it can see what it did. Under the old view it could not, and it would
    signal at every window forever.
    """

    seen: list[dict[str, Any]] = []

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["actor_id"] != "governor":
            return wait_decision("not mine")
        history = ctx["your_actions_so_far"]
        seen.append({"n": len(history), "counts": dict(ctx["your_action_counts"])})
        if any(h["you_did"] == "signal_support" for h in history):
            return wait_decision("I have already signalled")
        return act("signal_support")

    result = _run(convergence_world(), decide)

    assert seen[0]["n"] == 0, "the first invocation was handed a history it had not made"
    later = [s for s in seen if s["n"] > 0]
    assert later, (
        "the actor never saw a single one of its own acts — its own record reaches it "
        "through no channel, which is the defect this exists to close"
    )
    assert any("signal_support" in k for s in later for k in s["counts"]), (
        "the act history does not name the action that was taken"
    )

    signals = [
        e
        for e in result.event_ledger
        if str(e.payload_dict.get("event_type", "")) == "governor_signals_support"
    ]
    assert len(signals) == 1, (
        f"the governor signalled {len(signals)} times while following a rule that depends "
        "on seeing its own history — the history is not reaching the decision"
    )


def test_the_act_history_is_in_the_prompt_the_provider_actually_received() -> None:
    """Half-landing this is the dangerous failure, so it gets its own test.

    A scripted gateway reads the context dict, so every other test here would pass on a
    build where the history is computed, recorded in the trace, and never rendered into
    the prompt. A real provider sees only the prompt. That build would look correct to an
    auditor and ship an agent that is still blind — giving an agent its own act history
    and then not printing it is the exact defect this closes.

    Asserted on `rendered_prompt`, which is the byte-exact string that was sent.
    """

    result = _run(
        convergence_world(),
        lambda ctx: (
            act("signal_support") if ctx["actor_id"] == "governor" else wait_decision("not mine")
        ),
    )
    prompts = [
        (d, str(d.decision_context.get("rendered_prompt", "")))
        for d in result.actor_decisions
        if d.actor_id == "governor"
    ]
    assert prompts, "probe shape lost: the governor was never asked anything"
    assert all("WHAT YOU HAVE ALREADY DONE" in p for _, p in prompts), (
        "the act history is not rendered into the decision prompt — it reaches the trace "
        "and never reaches the actor"
    )
    later = [p for d, p in prompts if d.decision_context.get("your_actions_so_far")]
    assert later, "probe shape lost: no invocation had a history to show"
    assert any(
        "signal_support" in p.split("WHAT YOU HAVE ALREADY DONE", 1)[1][:800] for p in later
    ), "the block is printed but the acts are not in it"


def test_the_act_history_records_the_worlds_verdict_not_an_assumed_success() -> None:
    """A history that reported every attempt as done would be worse than none."""

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx["actor_id"] != "governor":
            return wait_decision("not mine")
        return act("knock")

    result = _run(convergence_world(second_window=False), decide)
    histories = [
        d.decision_context.get("your_actions_so_far") or []
        for d in result.actor_decisions
        if d.actor_id == "governor"
    ]
    with_history = [h for h in histories if h]
    assert with_history, "no invocation saw any history of the refused attempts"
    assert all(entry["outcome"] == "refused" for h in with_history for entry in h), (
        f"a refused attempt is recorded as something other than refused: {with_history[0]}"
    )


def test_a_completed_attempt_is_paired_with_its_completion() -> None:
    """An attempt and the world's verdict on it are one thing that happened."""

    result = _run(
        convergence_world(second_window=False),
        lambda ctx: (
            act("signal_support") if ctx["actor_id"] == "governor" else wait_decision("not mine")
        ),
    )
    world = next(iter(result.final_worlds.values()))
    acts = world.own_actions("governor")
    signals = [a for a in acts if a.action_id == "signal_support"]
    assert signals, "the governor's own act is missing from its history"
    assert all(a.outcome == "completed" for a in signals), (
        f"a completed act was not paired with its completion: {[a.outcome for a in signals]}"
    )
    assert act_counts(acts)["signal_support:completed"] == len(signals)


# ---------------------------------------------------------------------------
# 5. Memory: a retrieval window of k copies of one node is not a window
# ---------------------------------------------------------------------------


def test_identical_content_at_one_instant_is_one_memory_not_n() -> None:
    """`add` inserted unconditionally, and `node_id` includes `created` — so with the
    branch clock pinned at one instant, re-perceiving identical content produced N nodes
    with the SAME id. `retrieve` then scored them identically and filled its whole window
    with copies of one of them, while `retrieved_memory_ids` reported a full window."""

    now = datetime.fromisoformat(WINDOW_1)
    mem = MemoryStream()
    for _ in range(20):
        mem.add_memory(
            "the governor signalled support",
            kind="episodic",
            importance=1.0,
            created=now,
            tags=("source:governor",),
        )
    assert len(mem) == 1, f"one perception recorded twenty times became {len(mem)} memories"

    for i in range(5):
        mem.add_memory(f"a distinct thing number {i}", kind="episodic", importance=0.5, created=now)
    got = mem.retrieve("the governor signalled support", now=now, top_k=6)
    assert len({n.node_id for n in got}) == len(got), (
        f"the retrieval window is {len(got)} slots holding {len({n.node_id for n in got})} "
        "distinct memories"
    )


def test_recurrence_that_carries_information_is_never_collapsed() -> None:
    """Hearing the same thing from two people, or twice weeks apart, is corroboration.

    Only one perception recorded twice is folded — never the same content arriving twice
    for different reasons.
    """

    t1 = datetime.fromisoformat(WINDOW_1)
    t2 = datetime.fromisoformat(WINDOW_2)
    mem = MemoryStream()
    mem.add_memory(
        "rates will be cut", kind="episodic", importance=0.8, created=t1, tags=("source:governor",)
    )
    mem.add_memory(
        "rates will be cut", kind="episodic", importance=0.8, created=t1, tags=("source:deputy",)
    )
    assert len(mem) == 2, "the same fact from two different sources was collapsed into one"
    mem.add_memory(
        "rates will be cut", kind="episodic", importance=0.8, created=t2, tags=("source:governor",)
    )
    assert len(mem) == 3, "the same fact heard again nine days later was collapsed into one"


# ---------------------------------------------------------------------------
# 6. The predicates themselves, at their boundaries
# ---------------------------------------------------------------------------


def test_is_self_echo_draws_the_boundary_at_the_author() -> None:
    assert is_self_echo("a", "a", "create_event")
    assert is_self_echo("a", "a", "append_record")
    # Another agent's act — including its reaction to yours — is never an echo.
    assert not is_self_echo("a", "b", "create_event")
    assert not is_self_echo("a", "b", "deliver_information")
    # The environment's own events have no author.
    assert not is_self_echo("a", None, "release_data")
    # The world's verdict on your attempt is the one thing you did not already know.
    assert not is_self_echo("a", "a", ACTION_REJECTED)
    assert not is_self_echo("a", "a", ACTION_FAILED)


def _view(**over: Any) -> LocalView:
    base: dict[str, Any] = {
        "actor_id": "a",
        "branch_time": datetime.fromisoformat(WINDOW_1),
        "role": "governor",
        "authority": ("signal",),
        "stage": "window",
        "public_facts": (),
        "observations": (),
        "trigger_kind": "directed_information",
        "feasible_actions": ({"action_id": "signal_support"},),
    }
    base.update(over)
    return LocalView(**base)


def _actor(**over: Any) -> ActorState:
    from sworldmodel.worldspec import ActorSpec, EntitySpec

    state = ActorState(
        entity_id="a",
        entity=EntitySpec(
            entity_id="a", name="A", kind="person", role="governor", authority=("signal",)
        ),
        spec=ActorSpec(entity_id="a"),
        memory=MemoryStream(),
    )
    for k, v in over.items():
        setattr(state, k, v)
    return state


def test_the_situation_key_ignores_the_actors_own_act_history() -> None:
    """Deliberate, and the reason the detector can fire at all.

    `own_actions` grows by one every time the actor acts, so a key that included it could
    never match twice — which is precisely why a whole-`LocalView` comparison is unable
    to detect anything. The key asks the narrower question: has anything happened in the
    world, or reached this actor from anyone else, that could make the same act a new
    decision?
    """

    actor = _actor()
    bare = _view()
    acted = _view(
        own_actions=(
            ActedRecord(at=bare.branch_time, action_id="signal_support", outcome="completed"),
        )
    )
    assert situation_key(actor, bare) == situation_key(actor, acted)


def test_the_situation_key_moves_when_the_world_or_anybody_else_does() -> None:
    actor = _actor()
    base = situation_key(actor, _view())

    from sworldmodel.actors import Observation

    later = _view(branch_time=datetime.fromisoformat(WINDOW_2))
    assert situation_key(actor, later) != base, (
        "elapsed simulated time reads as 'nothing moved' — a repeat months later would "
        "be suppressed, and time itself is information"
    )
    assert situation_key(actor, _view(stage="closed")) != base
    assert situation_key(actor, _view(world_fields=(("door_open", True),))) != base
    assert situation_key(actor, _view(feasible_actions=())) != base
    assert situation_key(actor, _view(trigger_kind="commitment_due")) != base
    heard = _view(
        observations=(
            Observation(
                obs_id="ev-1",
                time=_view().branch_time,
                kind="deliver_information",
                source="b",
                summary="I disagree",
            ),
        )
    )
    assert situation_key(actor, heard) != base, "what another participant said did not register"
    # ...but the actor's OWN message, echoed back, is not something it learned.
    echo = _view(
        observations=(
            Observation(
                obs_id="ev-2",
                time=_view().branch_time,
                kind="create_event",
                source="a",
                summary="the governor signals support",
            ),
        )
    )
    assert situation_key(actor, echo) == base


def test_intent_key_is_the_act_not_the_account_of_it() -> None:
    """A loop that varies how it explains itself is still a loop."""

    a = ActionChoice(mode="compiled_action", action_id="signal_support", rationale="because X")
    b = ActionChoice(mode="compiled_action", action_id="signal_support", rationale="because Y")
    assert intent_key(a) == intent_key(b)
    c = ActionChoice(mode="compiled_action", action_id="signal_support", params=(("tone", "firm"),))
    assert intent_key(c) != intent_key(a)


def test_check_non_decision_needs_a_prior_decision_to_compare_against() -> None:
    view = _view()
    assert check_non_decision(_actor(), view) is None, "a first wake can never be a non-decision"
    actor = _actor(last_situation_key=situation_key(_actor(), view), last_intent_key="int-whatever")
    found = check_non_decision(actor, view)
    assert found is not None and found.prior_intent_key == "int-whatever"
    assert check_non_decision(actor, _view(trigger_kind="commitment_due")) is None
