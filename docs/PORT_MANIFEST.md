# Port manifest

Every semantic adapted from a donor repository, with its origin, its destination, what
changed on the way, and what was deliberately left behind. No directory was copied; no
donor module is imported at runtime; `tests/invariants/test_universality.py`
(`test_no_legacy_repository_is_imported`) asserts that mechanically over the AST.

---

## From SWORLDMODEL (legacy)

### 1. Event queue with real timestamps and simultaneity batching

- **origin** `swm/world_model_v2/events.py:118 EventQueue`, `:55 Event`
- **destination** `src/sworldmodel/schedule.py` — `Schedule`, `ScheduledEntry`
- **semantics taken**
  - the clock is the queue: the world advances to the next scheduled thing, never by a
    tick or a round;
  - `pop_batch` returns *every* entry at the earliest timestamp, so simultaneous events
    are simultaneous and an actor in the batch has not seen the others' results;
  - ordering is derived from content, not insertion (legacy's `content_key` comment:
    *"insertion order must not decide reality"*), which is what makes a run replayable;
  - entries past the horizon are retained and reported rather than executed, so a branch
    that ran out of window says so.
- **changed**
  - `float` unix timestamps → `datetime`, because the product's whole point is real
    calendar time;
  - a mutable `heapq` → an immutable sorted tuple, because branches clone and must not
    share a queue, and because the schedule is serialized into the trace;
  - added `microstep` layering to `pop_batch`: same-timestamp entries that causally
    depend on each other execute in dependency order without the clock moving. Legacy
    carried `microstep` on the event but its queue popped the whole timestamp at once
    and layered afterwards in a separate runtime;
  - added `origin` provenance on every entry (`compiled_process` / `compiled_external` /
    `consequence` / `actor_plan`), so the trace can answer "why was this scheduled".
- **left behind** `StochasticHazard` (samples event times from invented rates —
  Monte Carlo over a `"broad prior"`); the global `_EVENT_TYPES` registry pre-seeded with
  `collective_vote` / `message_delivered` / `election` types (a mechanism-family list).

### 2. Decision triggers

- **origin** `swm/world_model_v2/temporal_model.py:232 DecisionTrigger`,
  `temporal_runtime.py:403 make_trigger`
- **destination** `src/sworldmodel/engine.py` — the `WAKE_*` constants, `_relevance`,
  `_merge_decisions`, and `ActorDecisionRecord.wake_reason` / `wake_detail`
- **semantics taken** the invariant stated in the legacy docstring — *"No trigger → no
  decision event → no actor call"* — plus recording the causal parents and *why now* on
  every invocation.
- **changed** legacy's trigger was a free-text advisory record produced alongside the
  call. Here the trigger is the *cause* of the call: an actor with no stated reason is
  not invoked at all, the reasons are a closed structural set, and simultaneous reasons
  for the same actor merge into one invocation carrying all of them (a person facing two
  things at once has one moment of attention, not two).
- **left behind** `provenance="scenario_generated"` free-text triggers and the parallel
  `ctrl_attention` control-event machinery.

### 3. Information lifecycle: delivered ≠ seen

- **origin** `swm/world_model_v2/information.py:30 Exposure` (`observed: bool`,
  commented *"False = delivered but not yet seen (inbox != read)"*);
  `temporal_runtime.py:364 collect_attention_bundle` (*"Items enter the actor's
  information set HERE, not at delivery"*)
- **destination** `src/sworldmodel/world.py` — `Delivery`, `deliver`, `mark_noticed`,
  `available_unnoticed`, and a `view_for` that returns only noticed items
- **semantics taken** visibility, availability and noticing are three separate,
  separately-timestamped transitions; only noticed information enters an actor's view,
  memory or reasoning.
- **changed** `salience` and its `half_life_days = 10.0` exponential decay were dropped:
  a decay constant on memory is an invented number about how people forget. Retrieval
  recency in `memory.py` already covers the useful part without asserting a half-life.
- **left behind** `InformationItem.credibility = 0.6` defaults and the
  `misinformation` / `latent_unknown` item taxonomy (compiled evidence already carries
  authority and epistemic type).

### 4. What was examined and *rejected* as weaker than the core

- `swm/world_model_v2/institutions_v2/authority.py:17 ACTION_REQUIRES_AUTHORITY` — a
  hardcoded map from `approve` / `veto` / `vote` / `certify` / `moderate` / `reinstate`
  to authority types. The core's per-action compiled `required_authority` needs no such
  vocabulary. **Core kept unchanged.**
- `InformationBoundary` / `INFO_CLASSES` — a fixed institutional information taxonomy
  (`sealed`, `privileged`, `ex_parte`, …). The core's `Visibility` + `audience` is
  universal. **Core kept unchanged.**
- `swm/world_model_v2/lean_v2/` (runtime profiles), `phase4_completion.py`, `phase13/`
  (numbered phase pipelines), forecast recovery, prior/simulation combiners,
  deadline-forced decisions, roster caps, `experiments/replay_vault*` — rejected in
  full, per the consolidation brief.

---

## From generative_agents

See `docs/GENERATIVE_AGENTS_EXTRACTION.md` for the component-level record. Summary of
what actually moved:

### 5. Ongoing action with a start and a duration

- **origin** `persona/memory_structures/scratch.py` — `act_start_time`, `act_duration`,
  `act_description`, `act_address`; `scratch.py:533 act_check_finished()`;
  `cognitive_modules/plan.py:521 _determine_action`
- **destination** `src/sworldmodel/actors.py OngoingAction`;
  `src/sworldmodel/executor.py` (start → scheduled completion);
  `src/sworldmodel/engine.py` (`actor.is_busy` and `_INTERRUPTING`)
- **semantics taken** an agent with an unfinished action does not plan a new one. This
  single mechanism is what makes a plan persist across events, and it is the thing that
  a "call each actor once per stage" loop structurally cannot express.
- **changed** GA compares wall-clock strings against a fixed step; here completion is a
  scheduled queue entry at `started + duration_seconds`, and the action is *re-validated
  against the world at that moment* — GA has no notion of an action failing because the
  world moved.
- **left behind** the `"sleep"` / `"bed"` string tests in `determine_decomp`, tile
  addresses, and the whole `f_daily_schedule` hourly/minute decomposition, which
  presumes a day-structured life that most forecast questions do not have.

### 6. Interruption as a decision, not an erasure

- **origin** `cognitive_modules/plan.py:699 _should_react`, `:806 _create_react`
- **destination** `actors.Plan` + `plan_disposition`
  (`continue` / `revise` / `interrupt` / `replace` / `complete` / `abandon` / `none`)
- **semantics taken** a new event does not destroy the current plan; the agent decides
  what happens to it, and an interrupted plan keeps its identity so it can resume.
- **changed** GA's reaction rules are hardcoded (`lets_talk` refuses if
  `curr_time.hour == 23`, if either party's description contains `"sleeping"`, or if a
  `chatting_with_buffer` counter is positive). Here the *actor* states the disposition
  and the runtime records it; the only structural rule is that an actor busy with its
  own unfinished action is not interrupted by a mere opportunity.
- **left behind** every one of those hardcoded conditions, and the `chat` / `wait`
  reaction dichotomy.

### 7. Perception as a filter, not a firehose

- **origin** `cognitive_modules/perceive.py`
- **destination** `world.view_for` + `engine._relevance`
- **semantics taken** an agent perceives a *subset* of what exists, and what it perceives
  enters memory whether or not it acts on it.
- **changed** GA filters by `vision_r`, `att_bandwidth` and `retention` — three integers
  on a tile grid. Here the filter is structural and evidence-grounded: visibility →
  delivery → noticing → a *stated* reason to reconsider (addressed to you, a compiled
  wake rule for this world, a condition you set yourself, an answer to something you
  asked). No numeric attention score exists anywhere in the runtime.
- **left behind** `vision_r`, `att_bandwidth`, `retention`, the spatial memory tree.

### 8. Retrieval and reflection

- **origin** `cognitive_modules/retrieve.py` (`[0.5, 3, 2]` weights, whose own comment
  says *"We currently use hard coded weights… we could learn optimal weights"*),
  `cognitive_modules/reflect.py` (`importance_trigger_max = 150`)
- **decision** **REJECT both constants.** Core's existing `memory.py` retrieval was kept
  as-is. The core's own `REFLECTION_TRIGGER = 1.5` was deleted: reflection now happens
  when the actor's own response says its understanding changed. An arbitrary number
  deciding when a person revises their beliefs is the same defect in both repositories.

---

## Numeric constants: what was removed

Every hardcoded number in the runtime that made a claim about human behavior:

| constant | was | now |
| --- | --- | --- |
| `actors.REFLECTION_TRIGGER = 1.5` | accumulated importance at which an actor reflects | deleted — the actor states `reflection_needed` |
| `actors._poignancy` → `0.8 / 0.5 / 0.3` | per-event-kind importance | reduced to a flat retrieval-ranking input (`1.0` for directed or data-bearing content, `0.5` otherwise) that decides *ordering only* and never whether the actor acts |
| `ProcessNode.rounds` | how many times each participant acts per node | deleted — the field does not exist |
| `_release_scenario_data` midpoint | when uncertain future data arrives | compiled `release_at`, or a standing branch condition; never an invented date |
| `executor._fill_params` → `choices[0]` | the actor's unstated option | deleted — the action is refused |
| GA `[0.5, 3, 2]`, `150`, `vision_r`, `att_bandwidth`, `retention`, `sec_per_step` | — | never ported |
| legacy `half_life_days = 10.0`, `credibility = 0.6`, hazard rates | — | never ported |

The numbers that remain in the runtime are budgets (`RunBudget`), and they can only
*stop* a trajectory and mark it unresolved. None of them can cause an action, choose an
option, resolve a terminal, or move a probability.

---

## Completion run: donor code adapted, and what stayed rejected

A second audit of the legacy repository, run specifically against the question *why did
the old system reach simulation on real questions when this one refuses*, corrected the
premise. The legacy default compiler (`lean_v2`) imports no retrieval module at all — its
reported score came from being handed a benchmark's own frozen background paragraph. The
compiler that did research live (`full_fidelity`) reached simulation partly through
machinery that must never be ported.

### Adapted

| capability | legacy origin | destination | what changed |
| --- | --- | --- | --- |
| truncated-JSON salvage | `swm/world_model_v2/compiler.py::_salvage_json` (`:201-268`) | `jsonsalvage.py` | Rewritten to the core's types and made to return `None` rather than `{}` for "unsalvageable", so an unusable reply can never be mistaken for a successfully parsed empty world. The legacy forensics identified truncation as the cause of exactly the zero-actors symptom seen here; the gateway previously answered truncation by doubling `max_tokens` and discarding the prefix. |
| targeted missing-element repair | `swm/world_model_v2/evidence_recompile.py::recompile_with_evidence` | `repair.py` + `api._compile_with_repair` | The legacy version reconciles an evidence inventory against a plan diff. Here each gate emits a machine-readable failure code and the planner maps it to *specific* queries and a *specific* instruction. Termination is progress rather than an attempt count. |

### Examined and rejected

| item | legacy origin | why |
| --- | --- | --- |
| model-knowledge roster construction | `compiler.py:46-52` — *"Use REAL NAMES … and your world knowledge about them"*, with no citation field | This is the recall the old system had, and it is unsafe invention. Every entity here carries claim ids or it does not exist. |
| fidelity critic that adds named people | `fidelity.py:18-100` | Adds participants from model knowledge, which is precisely what the coverage and reality gates exist to prevent. |
| `generic_outcome_prior` terminal fallback | `compiler.py:565-569` | Resolves a hollow world from a broad prior. A world that produces no outcome must report unresolved. |
| keyword → scenario-family → canned base rate | `family_hazards.py:17-24` | A live router mapping `"beat"` to sports and `"sign"` to deals, stamping a fitted base rate onto the terminal. The single clearest violation of the no-mechanism-families rule. |
| canned institution registry | `institutions_v2/build.py`, forced on every question by `compiler.py:707` | Institutions must be discovered from evidence per question. |
| invented populations, relations, nine-member panels | `activation_synthesis.py` | Fabricated structure. |
| canned personas | `qualitative_actor.py:275` | Actors are grounded in cited evidence or refused. |
| regex → preset-prior table | `phase3b_reference_priors.py:25` | Dead in the donor, and must never revive. |

### Notable non-findings

The donor has **no** official-domain query reserve (the core's `authoritative_query_reserve`
already exceeds it), **no** PDF, table, JSON-LD or pagination reader, and **no** role
attachment; its contradiction detector records a graph that is hashed and never read, and
it verifies spans against a 400-character RSS blurb rather than the article. Nothing was
taken from those paths because there was nothing there to take.
