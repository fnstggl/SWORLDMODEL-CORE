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

## CORRECTIONS — adversarial review of this audit (vision-adversary)

Every claim below was reproduced independently. Two of this audit's judgements did not
survive, and one of its rankings was wrong.

### The central inference was WRONG: the fixture's convergence came from the test, not the runtime

This audit claimed *"the runtime can do everything the vision needs — proven, not asserted"*
on the strength of `_worlds.scheduled_multiparty_world`. The adversary read that world's
driver (`tests/invariants/test_scheduling.py:95-100`) and found the convergence is
hand-written:

```python
if ctx["actor_id"] == "member_0" and ctx["stage"] == "preparation":
    return act("circulate_note", {"text": "a note for the others"})
```

`ctx["actor_id"] == "member_0"` hardcodes that **exactly one named agent ever
communicates, exactly once**. Removing that single `if` collapses the same world:

```
V3  audit's exact driver, 5 members   -> actor_calls 23  resolved=True  outcome=YES
V3b audit's exact driver, 9 members   -> actor_calls 43  resolved=True  outcome=YES
V1  per-actor guard REMOVED, 5 members-> actor_calls 80  BUDGET EXHAUSTED  resolved=False
V2b any may speak, self-limited, 9    -> actor_calls 81  BUDGET EXHAUSTED  resolved=False
```

**Scale is not the problem** — nine participants resolve fine *with* the guard. The
runtime has never been shown to converge without a human-written decision function. So
"the compiler is the bottleneck and the runtime is ready" is not established, and the
ordering principle below must not be read as "the runtime is fine."

### G1 is CONFIRMED — build on it

Reconstructed from the branch's own ledger, running the real evaluator:

```
event_count('bailey_signals_support_for_further_cut') = 19
unresolved_when -> False   yes_when -> True
evaluate_terminal WOULD return: resolved=True outcome=YES
forecast.json ACTUALLY reported: resolved=False outcome=None
published simulation_probability: 0.0
```

The intents did produce the events — 20 started, none rejected, 19 landed (one in flight at
truncation). This audit said "twenty events"; it is nineteen. Immaterial to the argument.

**But W1's ranking was wrong.** Across all 28 runs / 59 branches, the truncation guard has
discarded a resolved evaluation **exactly once** — this branch. Only 4 of 28 terminals use
`event_count`; **zero** use `count` or `exists`; 24 of 28 are `equals(field(...), True)`,
which W1 deliberately leaves alone. W1 is a one-branch fix in the corpus as it stands. It
becomes common once worlds get bigger, so it still ships — but W2 has corpus-wide reach and
should carry the weight.

### W1 has a real wrong-banking case: `unresolved_when` is never checked

Both legs monotone, both built from the "safe" ops:

```
yes_when        : event_count('board_approves_merger') > 0     -> non_decreasing
unresolved_when : count('formal_challenges') > 0               -> non_decreasing
  t1 board approves      <-- banked YES
  t2 challenge filed     -> terminal says UNRESOLVED
```

Banking must also require `unresolved_when` to be non-increasing or fixed: once
determinable, always determinable.

### One claim in this audit was itself wrong, and I withdraw it

The corpus-wide `action_started 89 / action_completed 3` ratio is **not** evidence that
actions fail to complete. `action_completed` was added in `db8929e` (Phase 2); the runs
showing zero were made at `f30ff03`, before it existed. Their effects (`create_event`,
`append_record`, `set_field`) all land in the ledger immediately after each start. Actions
complete. Anyone re-deriving that statistic across commits will reach the same false
conclusion — it is a dating artifact, not a defect.

---

## WHAT THIS AUDIT MISSED — ranked by leverage (vision-adversary)

**M1 · Agents are blind to their own actions, and this invalidates W3's proposed
mechanism.** `WorldState.view_for` (`world.py:223-261`) builds observations only from
`self.deliveries` where `d.actor_id == actor_id`, and an actor is never a recipient of its
own delivery. Measured: `member_0` circulated a note **15 times**; on invocation #16 its
view held `0 observations mentioning member_0`, `0 such memories`, one overwritten
`current_action` slot, no commitments, no plan. W3 was told to detect "an actor re-deciding
from an unchanged local view" — the view changes every time (other members' notes,
`world_version` 0 → 350). **The agent repeats itself because it cannot see that it already
acted.** Fix the view, not the loop.

**M2 · No theory of mind at all.** `view_for` never reads `self.actors`, so other agents'
`beliefs`, `goals` and `plan` are in scope and never touched; `relationships` is `{}` in
every record and never reaches a prompt. An agent can see what another *has done* and has no
representation that others exist, want anything, or will do anything. Five soliloquies, not
a society. No W-item builds this.

**M3 · The `decides=false` escape hatch is G2's actual root cause.**
`semantic_plan.py:2655-2661`'s own error text offers the way out — *"give it the actions its
role affords, or mark it decides=false"* — and the same model writes both the affordances and
the flag. `phase2/geopolitical2`: 9 entities, 1 `decides:true`, 1 affordance,
`integrity_verdict: "verified"`. `prompts.py:272` already declares scenery actors a refusal;
`decides=false` makes the entity invisible to the only check that exists.

**M4 · Memory collapses to one duplicated item in exactly the runs that loop.**
`MemoryStream.add` (`memory.py:71-73`) inserts with no dedup, and `node_id` (`memory.py:88`)
includes `created` — constant when the clock is pinned. Identical content at one instant
yields an identical id inserted N times; `retrieve(top_k=6)` returned **six copies of one
node**. `retrieved_memory_ids` is non-empty in 218/235 records, which reads as "memory
works"; the content is one fact repeated.

**M5 · Inter-actor communication has never once happened.** The six `deliver_information`
events are junk — four have `actor_id = None` (environment drops), two have `text = ""`.
Genuine inter-actor communications in the entire artifact tree: **zero**. Upstream cause:
across all 18 semantic plans the affordance `changes` use exactly one op, `{'set': 5}` — the
planner has **never emitted a `send`**, though `semantic_lowering.py:501-509` maps it
correctly. W5 is therefore not "gate that communication exists" but "make the planner able to
express a channel at all."

**M6 · Evidence certifies contradictory stores as verified.** `individual_semantic`'s 13
claims flatly contradict each other; `0 of 13` carry `contradiction_ids`,
`unresolved_conflicts: []`, `evidence_coverage: 1.0`, `integrity_verdict: "verified"`.
`EvidenceStore.contradictions()` (`evidence.py:151-163`) warns in its own docstring that
reporting "no conflicts" from an empty result requires detection to have run. It did not.
Separately, that world's `verified_roles` appear in **none** of the 13 claims and the MPC — a
compiled participant holding an affordance — appears in **zero** evidence. Retrieval is
lexical overlap (`evidence.py:219-234`) with no axis for disposition, incentive or reaction.

**M7 · `allow_novel` is hardcoded `false` on all 40 compiled process nodes**
(`semantic_lowering.py:1061,1089`) against a spec default of `True`. The novel path has still
fired 22 times because `engine.py:1532` only enforces the flag when there are zero feasible
compiled actions — so escape from pre-enumeration works *in spite of* the compiler, and the
flag meant to govern it does not govern it.

## The ordering principle for this phase

Every phase so far made it harder to publish a bad answer. **None made it more likely to
produce a good world.** The gates are now well ahead of the generator, which is why the
honest reading of this repository is a suppressed headline on every run.

This phase inverts that: build the generator up to the gates. No new gate ships in this
phase without the mechanism that lets a correct world pass it.

---

## STATUS — W1..W5 landed (551 passing)

| item | state | evidence |
| --- | --- | --- |
| **W1** monotone banking | **DONE** | G1's reproducer: `FAILED — cut short with 20 due` → `resolved=True outcome=YES`, same unfired entries, same stop |
| **W2** truncation is ours | **DONE** | `execution_incomplete_mass` split from `execution_incomplete_unresolved_mass`; execution status on the report line beside the stop reason |
| **W3** convergence | **DONE** | same world by AST: `81 calls / budget exhausted / 2 unfired / history [0,0,…0]` → `4 calls / schedule exhausted / 0 unfired / history [0,1] / YES`. Adversary's collapse case re-verified independently: `80 / exhausted / unresolved` → `25 / schedule exhausted / YES` |
| **W4** affordance gate | **GATE DONE, GENERATOR NOT** | `INERT_PARTICIPANT` fires on `geopolitical2` (naming all 8 inert countries) and `geopolitical3`; silent on the three legitimately single-decider worlds |
| **W5** communication | **DONE** | a party's `deliver_information` is delivered **and noticed** by other parties and not by its author — **the first inter-actor communication in the artifact tree's history** |

### The two honest negatives

**The OPEC+ recompile did not produce a society.** Six live runs on the recorded store. The
baseline reproduces the defect exactly and **all gates pass it** — 5 entities, 1 action,
admitted. After the change it is refused instead of published, but **no post-change run
reached the simulator**, and the new gate never fired live: the planner stopped producing
scenery and produced *over-compression* instead, which two pre-existing gates (coverage,
actor-grounding) independently reject —

```
v3  planner put all seven countries at the meeting  -> refused on the `decides` flag
v4  1 coalition, Iraq excluded                       -> coverage_incomplete:
      "Iraq is urging a reassessment — exclusion rejected by independent review"
v5  1 coalition                                      -> actors_ungrounded:
      "constructed representative without a weight"
```

The generator is closer and is not there. Untested hypothesis worth trying first:
`participant_brief` orders names by claim count, which puts **OPEC+ (16 claims) at the top**
and each member below — that ordering may itself bias toward the aggregate. Invert it. The
other untried path is the full `run_forecast` repair loop, which feeds gate refusals back;
the slice harness stops at the first refusal.

**`allow_novel` must stay closed, and the reason is a real hole.** Checked before opening, as
required. A novel action **can write a terminal term directly** whenever the terminal's only
producer is a process node: `_blocked_by_world_authority` finds `gatekeepers` empty, returns
`""`, and the sole remaining gate is the interpreter model's self-declared
`required_authority`, which can be `[]`. Opening the flag today re-opens every gate built
this month. Fix: treat `spec.process.nodes[].effects` and external-process occurrences as
gatekeepers, since an effect only the environment produces is not something an actor has
standing to perform. Derivable rule for *where* to open it once safe: `allow_novel=True`
exactly at `actor_moment` nodes, `False` at `operational`/`scheduled_release` — read off
`SemanticProcess.kind`, never a free parameter.

### Residual, deliberately not guessed

A plan with `expected_participants` of 1/None **and** no `represents_count` can still carry
inert parties (`phase2/geopolitical` has this shape). There is no evidence-side signal
separating it from a legitimately non-acting institution — the recharge district and the Bank
of England are the same shape — and the stricter first rule produced real false positives on
both. Left to the prompt rather than guessed, on the standing rule that a gate refusing
correct worlds is worse than the hole it closes.

---

## CORRECTION — the ordering experiment showed no difference, and I reported one

Commit `3e52001` and its report state that the claim-count-descending brief produced
**"7 entities, 7 deciders — the member countries"**. **No artifact supports that.** The three
probe files still on disk, written within six minutes of each other and matching the described
three-arm design, hold:

| arm | entities | `decides:true` | affordances | parties holding acts |
| --- | --- | --- | --- | --- |
| claim-count desc | **3** | **1** | 1 | **1** |
| alphabetical | 1 | 1 | 1 | **1** |
| no brief (control) | — | — | — | **1** |

The claim-count arm's three entities are `seven_opec_countries`, `strait_of_hormuz`, `OPEC+`.
It did not surface the member countries: it produced **the same invented coalition**
(`seven_opec_countries` is the actor `actors_ungrounded` later refused for having no claim that
mentions it) plus a strait, which is not a party at all. A search of every JSON artifact in the
session scratchpad finds **no plan anywhere with ≥5 entities carrying a `decides` flag.**

So all three arms produced `parties_holding_acts = 1`. The experiment distinguished nothing,
and the conclusion drawn from it — that heading order changes which parties the planner
reaches for — is unsupported by the record. The revert of the alphabetical ordering was
therefore also made on no evidence; it is harmless either way, since neither arm differed.

This is the defect class this programme exists to stop — a claim the record does not support —
and it was made in the CTO's own commit message and relayed to the user. It was caught by an
instrument built specifically because single-run readings of a stochastic compiler are not
evidence, which is the argument for the instrument.

Two further problems with that eleven-run record, from the same review:

- **Its arms used different questions** (`7472de75…` vs `c62b0f63…`), so they were never
  comparable in the first place. `society_bench --compare` now refuses such a pair before
  printing any statistic.
- The recorded 0→11 entity swing that motivated the noise argument is not reproduced by the
  bench, which observes a range of 0..2 on the same store. Either the compiler has become far
  more compressed, or that spread pooled configurations. The bench cannot say which.

**The standing correction:** `parties_holding_acts` has been 1 in every measured world, in
every arm, under every prompt variation tried so far. Nothing has moved it. That was the
finding all along, and the ordering result was noise dressed as a difference.

---

## ADJUDICATED — the eight-party world's coverage refusal was CORRECT

The first world this repository ever measured above two acting parties (eight entities, each
of the seven OPEC+ countries holding its own affordance with a `send`) was refused by the
coverage gate. The obvious temptation was that the gate was in the way. It was not.

**The world genuinely omitted the claim, and it stays refused.** Run 06 cites 11 of the
store's 19 verified claims. `c-cb1acde4e1e8` — *"the plan includes reviving the remaining
one-third of a 1.65 million bpd supply cutback (roughly 550,000 bpd) in three monthly
stages"* — appears in no compiled object and is not in `accessible_claim_ids`, so **not one of
the eight actors could perceive it.** Eight parties deciding whether to agree a further
increase, none of whom knows about the staged increase already scheduled. And the terminal
reads *"beyond already announced adjustments"* — so what counts as already-announced is the
hinge of the question, not background colour.

**It was also cheap to satisfy, which is what makes "the planner must include it" a fair
demand rather than a wish.** Three of the five worlds in the same arm — same frozen store,
same compiler sha, same bench seed — did include it; run 00 wired it as a `world_facts` entry
with `actor_view`, as did the recorded live run in `artifacts/phase2/geopolitical2`. One entry
closes it. Runs 06 and 07 were the only two that dropped it, and they cited the fewest claims
overall (11 and 9 of 19).

So: **gate right, world wrong, and the planner has demonstrably done it correctly before on
this exact store.**

### What WAS wrong is separate, and it made the gate inconsistent in the permissive direction

The same gate certified **six other claims in that same world as covered while they were
absent in precisely the same sense** — no object, no accessibility. Their propositions contain
the string "OPEC" and the world had an entity named "OPEC+", and `_match_objects` matched an
object's name inside a candidate's whole proposition:

```
"OPEC+ has begun gradually restoring reduced production levels, with full resumption
 of voluntary cuts scheduled for the coming months"
      -> CERTIFIED COVERED, "represented by 'OPEC+' via actor"

"The plan includes reviving the remaining one-third of a 1.65 million bpd supply
 cutback (roughly 550,000 bpd) in three monthly stages"
      -> EXCLUDED, challenged, refused the world
```

**Same fact, same absence, opposite verdicts, decided by a substring.** `_semantic_match`
already refuses exactly this one-sided match and says why in its own docstring; the rule had
simply never reached the name-containment branch.

The fix is a **tightening** — it converts false passes into exclusions. On the five recorded
worlds it flips no deterministic verdict, so it refuses nothing previously admitted; its live
effect is that four more items per affected world now reach the exclusion reviewer.

**The lesson for this phase:** the gate was simultaneously too permissive (six substring
passes) and correctly strict (one real exclusion). "The gate is blocking our good world" was
the wrong frame; the gate was right about the thing it refused and wrong about six things it
waved through.

### REFUTED — coverage does not punish multi-party worlds. It rewards them.

I sent the coverage investigation with a named hypothesis: *"A world with more parties has
more surface to cover, so a coverage rule that scales badly with party count would
systematically punish exactly the worlds we want."* I flagged it as a hypothesis rather than a
finding, and the evidence went the other way.

- The **8-entity world passed** the deterministic coverage gate.
- The only deterministic coverage refusal in that arm was a **1-entity world**.
- The 8-party world **covered** a claim — Iraq's demand for a reassessment — that **all three
  1-entity worlds failed to cover**, because it had Iraq as an entity carrying that claim.

So parties are not coverage *burden*; parties are what **carries** claims. A world that
represents the seven countries separately has somewhere to attach the seven countries'
positions, and a world that compresses them into one aggregate has nowhere to put them. The
compression this phase has been fighting makes coverage *harder to satisfy*, not easier.

That reverses the framing I gave: coverage is not an obstacle standing between the compiler
and a multi-party world. It is, if anything, an argument for one.
