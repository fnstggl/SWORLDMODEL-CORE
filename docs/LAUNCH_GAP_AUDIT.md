# LAUNCH GAP AUDIT — the simulation against the product vision

Audited directly from artifacts and code, not from reports. Every claim below names the
file, line or artifact it came from, and every mechanism was reproduced.

## The vision, stated as four testable properties

1. **Digital twin of an arbitrary natural-language situation** — the compiled world contains
   the participants and forces the situation actually contains.
2. **Evidence finds what is *acting*** — retrieval establishes not only who exists but what
   each one can do and would do.
3. **People simulated as AI agents, played forward** — agents decide, act, inform each other,
   react, and change their minds over time.
4. **Their behaviour IS the answer** — the outcome is produced by what the agents did, not by
   arithmetic over branch weights.

## The one-sentence state

**We have built an excellent instrument and it is correctly reporting that the simulation is
not yet happening.** The runtime can do everything the vision needs — proven, not asserted.
The compiler does not ask it to, the loop does not converge, and one honest-looking guard
throws away answers the system already watched being produced.

---

## What genuinely works (verified, not assumed)

**The agent runtime is real and it is the strongest part of the system.** Executed live at
`b8bcf4a` against `_worlds.scheduled_multiparty_world`:

```
ledger: action_started 5, action_completed 5, append_record 5, actor_waited 10,
        release_data 1, create_event 1
branch baseline: resolved=True outcome=YES
```

Five agents deliberated, recorded positions, one circulated a note that reached the other
four, and **the terminal resolved YES out of what the agents did**. Communication delivery
and notice are counted and correct. This is property 3 and property 4, working.

**The per-decision record is genuinely forensic.** `ActorDecisionRecord` carries
`exact_prompt`, `prompt_hash`, `provider_response`, `feasible_actions`, `intent`,
`plan_before`/`plan_after`, `local_view`, `retrieved_memory_ids`, `wake_reason`,
`state_before`/`state_after`, `validation_status`. Sampled from
`artifacts/phase2/geopolitical2`: the prompt is a real first-person brief with trigger,
active plan, commitments, pending information and revisit conditions. 51 invocations in one
branch produced 51 distinct prompt hashes — no two decisions were asked the same question.

**Retrieval finds the real participants.** `artifacts/phase2/geopolitical2` compiled nine:
OPEC+, Saudi Arabia, Russia, Iraq, Kuwait, Algeria, Kazakhstan, Oman, UAE.

**The honesty layer works and is now ahead of everything else.** All three headline numbers
this programme was convened over now refuse to publish.

---

## G1 · CRITICAL · The system watched the answer happen twenty times and published 0.0

The most important defect in the repository. `artifacts/ab/individual_semantic`:

- **Terminal:** `greater_than(event_count('bailey_signals_support_for_further_cut'), 0)`
- **What the agent did:** chose `signal_support_for_further_cut` on **20 separate invocations**
- **What the branch reported:** `resolved=False`, *"trajectory cut short with 3 scheduled
  events still due before the horizon (actor-call budget exhausted (80))"*
- **What was published:** `0.0`, labelled `weighted_simulated_trajectories`

The two branches that finished said NO. The branch where the answer was YES — demonstrably,
twenty times over — was dropped as unresolved, and the published number is the average of
the survivors.

Three separate causes, all real:

1. **The terminal is evaluated once, at the end.** `engine.py:1700-1712` (`_finalize`):
   the world is advanced to the horizon and `evaluate_terminal` runs a single time. A
   terminal satisfied at step 3 is never recorded as satisfied if step 80 truncates.
2. **A cut-short branch discards a resolved evaluation.** `engine.py:1713-1725`: when
   `unfired_in_horizon > 0`, a resolved evaluation is overwritten with unresolved. The
   comment defending it is *right in general* — *"Reporting a resolved outcome for it would
   claim we watched the process finish when we stopped watching"* — and **wrong for a
   monotone terminal**. `event_count(X) > 0` cannot become false again. Truncation cannot
   un-happen twenty events that already happened.
3. **Execution failure is priced as uncertainty.** Budget exhaustion produces unresolved
   mass, which flows into bounds and suppression as though reality were uncertain. It is
   not: our simulator ran out of calls.

## G2 · CRITICAL · Compiled worlds are scenery, not societies

`artifacts/phase2/geopolitical2` — nine entities, and the **entire world contains one action**:

```json
{"action_id": "adjust_quotas",
 "effects": [{"field": "quota_increase_announced", "op": "set_field", "value": true}]}
```

The terminal reads exactly that field. So Saudi Arabia and Russia — whose disagreement is
what actually determines OPEC+ quota policy — have **no affordances at all**. They cannot
advocate, resist, defect, stall, or bargain. Eight of nine participants are set dressing,
and the question resolves on whether one aggregate actor flips one boolean.

This is the actor-side twin of the Tesla defect (a cited base times an invented multiplier).
Same shape: the world has the *form* of a multi-party situation and the *arithmetic* of a
single switch. Across every run: `entities 2–11`, `actions 0–2`.

## G3 · HIGH · Agents do not talk to each other

`deliver_information` is in `UNIVERSAL_OPS` (`effects.py:45-60`), defaults to PRIVATE
visibility, and is **proven working** — the test world sends one note that reaches four
recipients, all delivered and noticed. Across every real artifact in the repository:
**6 `deliver_information` events out of 637**, and zero in any headline run. No coalition
forms, no position propagates, nobody persuades anybody.

A world model of a social situation in which the participants cannot communicate is not a
social world model.

## G4 · HIGH · The outcome is not produced by the agents

Every published number came from `release_data` (branch-weight construction) or from
arithmetic. The forensic audit already reached this verdict independently
(`BRANCH_WEIGHTS_DOMINATED` ×2, `FACTUALLY_RESOLVED` ×1). D6 now *classifies* it after the
fact; nothing yet prevents *compiling* a world whose answer no agent can influence.

## G5 · MEDIUM · The actor loop does not converge

In the branch above, **43 of 51 wakes were `directed_information`** — the agent being woken
by the consequences of its own actions and re-deciding the same thing. 20 identical signals,
23 waits. `max_actor_calls` is a flat **80** (`engine.py:169`) regardless of world size, so
a nine-participant world over ten weeks gets the same budget as a two-participant world over
two weeks.

---

## MINIMUM WORKING IMPLEMENTATIONS

Ordered by leverage. Each must be a working mechanism proven on a real run, not a gate.

### W1 — Bank a monotone terminal the moment it is satisfied
Evaluate the terminal continuously. When it becomes true through a **monotone** expression
(`event_count`, `count`, `exists`, append-only records — anything that cannot un-happen),
record the satisfying instant and the causing event id, and make it final: later truncation
cannot unresolve it. Non-monotone terminals (`field > threshold`) keep today's conservative
behaviour, because more simulation genuinely could move them. **This alone turns G1's run
from a wrong 0.0 into a right answer.**

### W2 — Truncation is an execution failure, not uncertainty
A branch that stops on budget is `EXECUTION_INCOMPLETE`, reported separately from genuine
unresolved mass, and never silently averaged into a headline. A run with material
execution-incomplete mass is a **failed run**, not a forecast — that is a reliability
statement about us, and it must read as one.

### W3 — Convergence
An actor that re-decides the same action from an unchanged local view is looping, not
deciding: detect it, and stop waking an actor on information it produced itself. Scale the
call budget to participants × horizon rather than a flat 80. Exhaustion must be loud.

### W4 — Affordance synthesis, then a gate
Every material participant gets at least one evidence-grounded action with real
preconditions and effects — derived from what the evidence says that participant actually
does. Then refuse a world where the terminal is reachable by one actor's single write while
other included participants hold no affordances. Build the synthesis first; the gate is
worthless with nothing to admit.

### W5 — Communication as a compiled requirement
A multi-participant world must express at least one real information channel between
participants, using the `deliver_information` machinery that already works. Compile it, then
gate it.

### W6 — Acceptance on the preregistered pastcasts
The ten sealed cases in `docs/acceptance/ACTOR_FIDELITY_SUITE.md` actually run, under the
sealed protocol. The bar is **simulation fidelity** — did agent behaviour produce the
outcome — not calibration alone. A run that gets the right number for the wrong reason fails.

---

## The ordering principle for this phase

Every phase so far made it harder to publish a bad answer. **None made it more likely to
produce a good world.** The gates are now well ahead of the generator, which is why the
honest reading of this repository is a suppressed headline on every run.

This phase inverts that: build the generator up to the gates. No new gate ships in this
phase without the mechanism that lets a correct world pass it.
