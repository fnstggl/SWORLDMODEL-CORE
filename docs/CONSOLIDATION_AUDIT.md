# Consolidation audit

Three repositories were traced by executing and reading their real code paths, not by
reading filenames or docstrings. That distinction is load-bearing: **the single most
common defect found in all three repositories is a docstring asserting a property the
code does not have.** Several are quoted below.

| repository | role | path |
| --- | --- | --- |
| SWORLDMODEL-CORE | the only repository modified; destination | `src/sworldmodel/` |
| SWORLDMODEL (legacy) | read-only parts donor | `swm/` (989 Python files) |
| generative_agents | read-only parts donor | `reverie/backend_server/` |

---

## 1. What the core actually did, before this change

Traced from `cli.cmd_forecast` to the final probability:

```
cli.cmd_forecast
  -> ForecastConfig.live            config.py:52   (DeepSeekGateway + LiveResearchBackend)
  -> api.run_forecast               api.py:118
  -> backend.research(...)          live_research.py:163  -> compile_world_spec_live
  -> world_compiler.compile_world   world_compiler.py:108 (reality gate, coverage gate,
                                                            grounding gate, scenarios)
  -> engine.run                     engine.py:73
  -> outcomes.aggregate             outcomes.py
  -> tracing.TraceContext.write     tracing.py
```

The route was already single and already trajectory-only — that part of the previous
work is sound and was kept. What was not sound was almost everything about *how the
world advanced inside `engine.run`*, and several integrity gates that reported passes
they had not performed.

### 1.1 The runtime was a protocol walker, not an event loop

`engine._run_graph` iterated `spec.process.nodes` as a **static list**, and inside each
node `_drive_activations` invoked participants up to `node.rounds` times. Compiled
worlds set `rounds: 1`. The observable consequence was the one that prompted this work:
every actor was invoked about once per stage, so a two-stage process produced almost
exactly two calls per actor regardless of what happened to anyone.

Other consequences of the same design:

- `_release_scenario_data` (engine.py:127) delivered the branch's uncertain future data
  at `as_of + (horizon - as_of) / 2` — **an invented midpoint**, chosen so the world had
  something to react to.
- `_finalize` jumped the clock straight to the horizon.
- `world.apply` set `time = max(time, ev.time)`, so `schedule_event` — which computed a
  future timestamp — **applied immediately and dragged the clock forward with it**.
  Scheduling did not schedule.
- `WorldState.pending_events` existed, was appended to by `enqueue`, and was never read.

### 1.2 Actor intentions were rewritten

Three separate places changed what an actor decided:

- `actors._to_choice`: any `action_mode` the runtime could not parse silently became
  `"wait"` — an unreadable provider response was recorded as a deliberate decision to
  do nothing.
- `executor._fill_params`: a missing action parameter was filled with `p.choices[0]`.
  For an action whose parameter *is* the decision — which way you go, what you answer,
  how much you offer — this is the runtime casting the actor's vote and recording it
  under the actor's name.
- `worldspec.ActorPolicy.default_action_id` + `ActorPolicyRule`: the compiler was asked
  to emit a rule table and a default action per actor, which `DeterministicGateway`
  executed. The prompt schema requested it (`prompts._WORLD_SCHEMA`), so it existed on
  the live path too, where it was rendered into the actor's own grounding block as its
  "current inclination".

### 1.3 Actors were thin

`ActorState` carried `active_plan: str` — one string, seeded with the template
`f"Act as {role} toward the world's outcome."` and never updated. There were no
commitments, no ongoing action, no revisit conditions, no plan lifecycle. Reflection
fired on `REFLECTION_TRIGGER = 1.5` accumulated from `_poignancy` returning 0.8 / 0.5 /
0.3 by event kind — four unsupported numbers deciding when a person reconsiders their
beliefs.

`world.view_for` returned *every* visible event the actor had not yet seen. Visibility
was awareness: no delivery, no delay, no noticing, no possibility of missing something.

### 1.4 Integrity gates that passed without checking

Found by adversarial trace of the compiler and research paths:

- **Uncited memory seeds were stamped `VERIFIED_OBSERVATION`.** `_normalize_compilation`
  strips every `evidence_claim_id` not present in the store; `actor_grounding_profile`
  kept the seed anyway, and `render_grounding` marked it verified. Live actors were
  therefore told invented facts *labeled as established*.
- **The participant-count "reality check" compared the LLM's `expected_participants`
  against the count of actors in the same response.** Nothing anchored either number to
  evidence. The docstring claimed it prevented a nine-seat body becoming five units; it
  could only detect an LLM contradicting itself.
- **An actor with no matching entity was silently materialized** as a synthetic
  `EntitySpec` (`_default_actor_entity`), participated in the simulation, inflated the
  verified-participant count, and was invisible to the coverage gate, which enumerates
  `spec.entities` only.
- **The coverage gate could be satisfied by vocabulary**: a scheduled-event candidate
  whose description contained "decision", "vote", "meeting", "ruling" or "verdict" was
  matched to the always-present terminal object and recorded `INCLUDED` without being
  represented anywhere.
- **The checklist and the gate disagreed by construction**: `evidence_checklist` built
  the inventory with an empty focal-identity set, so organization and population
  candidates were never shown to the compiler, while the gate built it *with* focal
  identities, where the same candidates became material and were demanded.
- **Unlabeled branch weights were promoted.** An outcome whose `provenance` string was
  missing or unrecognized became `EXPLICIT_MODEL` — a *stronger* identification than
  the honest default — which then won the weakest-provenance ordering.

### 1.5 Research and evidence

- **Coverage repair destroyed the evidence store.** `augment_for_coverage` ignored its
  `prior` bundle and called `research()` again, building a brand-new `EvidenceStore`;
  `api._compile_with_repair` then replaced the good bundle with the unrelated one, up
  to twice per run.
- **The cutoff was enforced against a page's self-declared publication date while the
  fetch happened at wall-clock now.** A page dated before the cutoff but edited after it
  delivered post-cutoff text under a pre-cutoff timestamp. PDFs took the *earliest*
  metadata date, so a document created before and modified after the cutoff was admitted
  with its creation date and its post-cutoff content extracted.
- **Fetched page text was interpolated raw into an extraction prompt** with no delimiter
  defense, and the HTTP transport had no scheme, SSRF, size or content-type protection
  while fetching URLs scraped off third-party HTML.
- **Google News RSS was fetched and discarded** — `parse_rss` results were used only for
  a count; no RSS URL ever reached the fetcher.
- **Authoritative-domain queries were enqueued last** and starved by the global query
  cap; `plan.authoritative_sources` was never queried at all.
- **Contradiction detection did not exist on the live path**: nothing set
  `contradiction_ids`, so the trace field, the follow-up-query input and the reality
  gate's conflict check all read a permanently empty set as "no conflicts".
- `--corpus` on the `forecast` command swapped in `DeterministicGateway` *before* the
  live gate, and `_live_audit` wrote `"prepared_corpus_read": False` as a **literal**
  into `acceptance.json` as a proof line.

---

## 2. What the legacy repository actually contains

989 Python files, multiple overlapping runtimes (`swm/worlds/`, `swm/simulation/`,
`swm/world_model_v2/`, `swm/world_model_v2/lean_v2/`), numbered phase pipelines, and a
large volume of dated experiment artifacts. Most of it is superseded. Three components
are genuinely stronger than anything the core had, and were adapted.

### Adopted

**`swm/world_model_v2/events.py:118 EventQueue`** — a heap keyed on exact timestamps,
with `pop_batch` returning *all* events sharing the earliest timestamp so simultaneity
is modeled as simultaneity, a horizon cutoff, and `peek_pending` for honest truncation
reporting. Its `Event.content_key` comment states the principle the core needed and
lacked: *"insertion order must not decide reality"*.

**`swm/world_model_v2/temporal_model.py:232 DecisionTrigger`** — every actor decision
event must carry one, with `trigger_type`, `causal_parent_events`, `observed`,
`decision_relevance` and `why_now`. Its docstring states the invariant exactly: *"No
trigger → no decision event → no actor call."* This is the rule the core's rewritten
scheduler now enforces.

**`swm/world_model_v2/information.py:30 Exposure`** — carries `observed: bool`, commented
*"False = delivered but not yet seen (inbox != read)"*, plus arrival time and channel.
`temporal_runtime.collect_attention_bundle:364` makes items enter the information set at
*attention*, not at delivery. This is the information lifecycle the core collapsed into
one step.

### Rejected

**`swm/world_model_v2/institutions_v2/authority.py:17 ACTION_REQUIRES_AUTHORITY`** — a
hardcoded map from domain action names (`approve`, `veto`, `vote`, `certify`, `moderate`,
`reinstate`, …) to authority types. This is precisely the hardcoded social-action
vocabulary the architecture forbids, and the core's compiled per-action
`required_authority` is strictly more general. **Legacy is weaker here; core kept.**

**`InformationBoundary.INFO_CLASSES`** — a fixed institutional taxonomy (`sealed`,
`privileged`, `ex_parte`, `internal_deliberation`, …). The core's `Visibility` +
`audience` is universal and does not presume a legal-institutional world.

**`events.py:100 StochasticHazard`** — samples event times from `rng.expovariate(rate)`
with rates defaulting to `prov_note="broad prior"`. Monte-Carlo sampling from invented
rates is exactly the arbitrary numeric social assumption this consolidation removes;
timing uncertainty must be branched or reported, not sampled.

**`events.py:38 _EVENT_TYPES`** — a global registry pre-seeded with `collective_vote`,
`message_delivered`, `election`-shaped types. A mechanism-family list by another name.

Also rejected wholesale, and confirmed present in legacy: multiple runtime profiles and
Lean/full-fidelity routing (`swm/world_model_v2/lean_v2/`), numbered phase pipelines
(`phase4_completion.py`, `phase13/`), forecast recovery, prior/simulation combiners,
deadline-forced decisions, roster caps, and `experiments/replay_vault{,_v2,_v3}/`
artifacts.

---

## 3. What Generative Agents actually contains

See `docs/GENERATIVE_AGENTS_EXTRACTION.md` for the component-by-component record.

The essential finding: the value is **not** the perceive → retrieve → plan → reflect
naming, which is easy to imitate. It is `Scratch`'s persistent action state —
`act_start_time`, `act_duration`, `act_description`, `chatting_with`,
`chatting_end_time` — combined with `act_check_finished()` (scratch.py:533), which is
what makes an agent *keep doing what it is doing* until it is actually finished. That
mechanism was ported. Smallville's fixed `sec_per_step` tick, tile maze, vision radius,
attention bandwidth, retention count, the `[0.5, 3, 2]` retrieval weights (whose own
comment says they were manually chosen) and the `importance_trigger_max = 150`
reflection threshold were all rejected.

---

## 4. Capability matrix

Decision key: **KEEP** core as-is · **PORT/ADAPT** donor semantics into core types ·
**REWRITE** small and correct · **DELETE** entirely.

| capability | core (before) | legacy | generative agents | core strength | donor strength | observed failure risk | decision | reason | tests | final owner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single forecast route | `api.run_forecast` | many runtimes/profiles | n/a | one readable path | — | — | **KEEP** | already correct and the reason this repo exists | `test_universality` | `api.py` |
| product interface | `cli` + `--corpus`/banxico/synthetic | scripts | n/a | question-only exists | — | corpus path reachable from `forecast` | **REWRITE** | one command, question-only, no fixture escape | `test_the_cli_takes_only_the_question_and_its_window` | `cli.py` |
| live research loop | `live_research.py` | `swm/retrieval/asof_store` | n/a | real fetch+verify | as-of store | RSS discarded, official queries starved | **ADAPT** | keep the loop, fix discovery and ordering | `tests/unit` | `live_research.py` |
| cutoff enforcement | published-date only | as-of retrieval layer | n/a | mechanical filter | physically cannot return future items | post-cutoff edits leak | **REWRITE** | archived retrieval at the cutoff | `tests/unit` | `source_fetch.py` |
| evidence store + lineage | `evidence.py` | `world_model_v2/evidence.py` | n/a | claim ids, lineage | — | repair discarded the store | **REWRITE (repair)** | follow-up must extend, never replace | `tests/unit` | `live_research.py` |
| source safety | none | none | n/a | — | — | SSRF + prompt injection | **REWRITE** | untrusted input must be data | `tests/unit` | `http.py`, `source_extract.py` |
| contradiction detection | inert | present | n/a | — | real | a check that always passed | **REWRITE** | run it or delete the plumbing | `tests/unit` | `live_research.py` |
| reality-integrity gate | `reality.py` | phase gates | n/a | refuses to simulate | — | self-referential participant count | **REWRITE** | anchor the count to evidence | `tests/invariants` | `reality.py` |
| coverage gate | `coverage.py` | — | n/a | evidence→world mapping | — | keyword satisfaction; checklist ≠ gate | **REWRITE** | same inputs both sides; no word matching | `tests/invariants` | `coverage.py` |
| actor grounding | `grounding.py` | — | `Scratch` identity | per-actor evidence block | — | uncited seeds marked VERIFIED | **REWRITE** | marks must track surviving citations | `tests/invariants` | `grounding.py` |
| WorldSpecification | `worldspec.py` | scenario models | n/a | domain-free, data-only | — | policy/default action embedded | **ADAPT** | delete policy; add scale, timing, externals, wake rules | `test_universality` | `worldspec.py` |
| representation scale | absent | implicit | n/a | — | — | orgs silently individuals | **REWRITE** | compiler must choose and record | `test_universality` | `worldspec.py` |
| structural world uncertainty | absent (one world) | ensembles | n/a | — | ensemble idea | one confident structure | **REWRITE (partial)** | `structure_id`/`structure_rationale` carried; see limitations | — | `worldspec.py` |
| dynamic action compilation | `ActionDefinition` | templates | n/a | effects-defined actions | — | no duration/delivery/failure | **ADAPT** | add lifecycle fields | `test_universality` | `worldspec.py` |
| novel-action interpretation | `novel.py` | — | n/a | refuses rather than approximates | — | — | **KEEP** | already correct | `test_no_coercion` | `novel.py` |
| process graph | node list + `rounds` | institutional stages | fixed tick | declarative conditions | entry conditions, durations, deadlines | a script, not a calendar | **REWRITE** | nodes are dated moments in a graph | `test_scheduling` | `worldspec.py` |
| external (non-agent) processes | absent | continuous processes | n/a | — | interval evolution | fake actors for the weather | **PORT** | `ExternalProcess` | `test_scheduling` | `worldspec.py` |
| event queue + real clock | none | `EventQueue` (heap, batch, horizon) | fixed `sec_per_step` | — | insertion-order invariance, batching | protocol walking, invented midpoints | **PORT** | the central fix | `test_scheduling` | `schedule.py` |
| decision triggers | `_trigger_for` string | `DecisionTrigger` | `_should_react` | — | typed cause, `why_now` | actors woken by stage | **PORT** | no trigger, no call | `test_scheduling` | `engine.py` |
| information lifecycle | visibility = awareness | `Exposure.observed` | perceive bandwidth | — | delivered ≠ seen | omniscient actors | **PORT** | `Delivery` with three timestamps | `test_scheduling` | `world.py` |
| relevance / interruption | none | attention bundles | `_should_react` (hardcoded) | — | reconsideration triggers | everyone consulted about everything | **REWRITE** | structural reasons only, no scores | `test_scheduling` | `engine.py` |
| persistent actor state | `active_plan: str` | qualitative state | `Scratch` (full) | memory stream | ongoing action + duration | actors reset each call | **PORT** | `Plan`, `OngoingAction`, commitments, needs, revisits | `test_scheduling` | `actors.py` |
| plan persistence | none | — | `act_check_finished` | — | the right mechanism | plans re-derived每 call | **PORT** | busy actors are not interrupted by opportunity | `test_scheduling` | `engine.py`, `actors.py` |
| episodic/semantic memory | `memory.py` | — | `AssociativeMemory` | already close | node kinds | — | **KEEP** | core version is smaller and adequate | `tests/unit` | `memory.py` |
| retrieval | recency/relevance | — | `[0.5, 3, 2]` weights | — | — | hand-tuned weights | **KEEP core** | donor's own comment calls its weights arbitrary | `tests/unit` | `memory.py` |
| reflection trigger | `1.5` constant | — | `150` constant | — | — | invented number decides belief change | **REWRITE** | the actor states when it needs to reflect | `tests/invariants` | `actors.py` |
| authority checks | compiled tokens | action→authority map | n/a | fully general | — | — | **KEEP** | core is strictly more general than legacy | `test_no_coercion` | `executor.py` |
| feasibility / preconditions | declarative `Expr` | stage permits | n/a | universal operators | — | — | **KEEP** | — | `test_no_coercion` | `executor.py` |
| intent vs consequence | separated | separated | — | actor never asserts effect | — | effects were instantaneous | **ADAPT** | add start/complete/fail lifecycle | `test_no_coercion` | `executor.py` |
| stale-intention revalidation | none | read/write sets | n/a | — | declared read sets | acts applied to a world never seen | **REWRITE** | re-validate at completion; record world version | `test_no_coercion` | `executor.py` |
| parameter handling | first-choice default | — | — | — | — | **runtime cast the actor's vote** | **DELETE** | refuse instead | `test_no_coercion` | `executor.py` |
| actor policy / default action | `ActorPolicy` | default votes | hardcoded reactions | — | — | deterministic behavior wearing an actor's name | **DELETE** | there is no default action | `test_no_coercion` | — |
| branch construction | Cartesian product | ensembles | n/a | mass conservation + disclosure | — | dependent unknowns crossed as independent | **REWRITE** | refuse declared dependence | `tests/unit` | `uncertainty.py` |
| branch cloning / lineage | `WorldState.clone` | lineage ids | n/a | deep-copies actor memory | — | — | **KEEP** | — | `test_scheduling` | `world.py` |
| loop / no-progress guards | none | budgets | n/a | — | — | open-ended cascades | **REWRITE** | budgets stop, never force | `test_scheduling` | `engine.py` |
| terminal expression | declarative | mechanism families | n/a | universal operators only | — | — | **KEEP** | already correct | `test_universality` | `expressions.py` |
| trajectory aggregation | `outcomes.aggregate` | prior combiners | n/a | trajectory-only | — | reference-class diagnostic present | **KEEP + DELETE prior surface** | remove where a prior could enter | `test_forecast` | `outcomes.py` |
| trace / replay contract | decisions + ledger | replay vaults | — | exact prompts recorded | — | no trigger, plan, or state fields | **ADAPT** | full invocation contract | `test_replay` | `tracing.py` |
| model-call instrumentation | full | prompt tracing | — | tokens, latency, retries, stages | — | — | **KEEP** | — | `tests/unit` | `gateway.py` |
| deterministic gateway | in package | profiles | — | — | — | reachable stand-in for actors | **DELETE from package** | tests only | `test_universality` | `tests/_fakes.py` |
| prepared corpora | in package + repo | fixtures | — | — | — | reachable from `forecast` | **DELETE** | — | `test_universality` | — |

---

## 5. Production files deleted

| path | why |
| --- | --- |
| `evaluation/banxico/corpus/corpus.json` | prepared corpus reachable from the product command |
| `evaluation/synthetic/*/corpus.json` (5) | same |
| `gateway.DeterministicGateway` | deterministic stand-in for live actors, inside the shipped package |
| `gateway.ScriptedGateway` | test fixture in the shipped package |
| `research.CorpusResearchBackend`, `research.MockResearchBackend`, `research.build_bundle_from_dict`, `research._apply_contradictions` | corpus readers in the shipped package |
| `config.ForecastConfig.offline` | a configuration that produces forecasts without touching reality |
| `cli.cmd_banxico_run`, `cmd_banxico_evaluate`, `cmd_synthetic_run`, `cmd_banxico_live`, `forecast --corpus` | scenario-specific commands and the fixture escape hatch |
| `worldspec.ActorPolicy`, `worldspec.ActorPolicyRule` | compiled default actions |
| `world_compiler._default_actor_entity` | silently invented actors |
| `grounding.with_simulated_hypothesis` | unreachable |
| `WorldState.pending_events`, `WorldState.enqueue` | written, never read; superseded by `Schedule` |
| `ResearchBundle.reference_class`, `ForecastConfig.include_reference_class_diagnostic` | a route by which a statistical prior could reach the report |
| `tests/_worlds.py`, `tests/_helpers.py`, `tests/_live_helpers.py`, `tests/acceptance/`, `tests/integration/`, 8 invariant/unit test modules | tested the policy-driven, fixed-round architecture that no longer exists |

Nothing was quarantined. Every remaining file in `src/sworldmodel/` is on the live path.

---

## 6. Architecture before and after

**Before**

```
forecast -> research -> compile -> for node in nodes:
                                      set stage
                                      apply node effects
                                      for round in range(node.rounds):
                                          for participant in node.participants:
                                              call actor            <- ~1 call/actor/stage
                                              apply effects instantly
                                   jump clock to horizon -> evaluate
```

**After**

```
forecast -> research -> compile -> seed the branch calendar from
                                     compiled process roots
                                     compiled external processes
                                     grounded actor plans and commitments
                                   while the queue has in-horizon entries:
                                       pop the next causal layer (time, microstep)
                                       apply world events at their real times
                                       deliver -> schedule noticing
                                       wake only actors with a stated cause
                                       actor: continue/revise/interrupt/replace plan
                                       validate intention -> start or refuse
                                       schedule completion; re-validate on completion
                                   evaluate the terminal expression
```

---

## 7. Honest status

What this audit did **not** establish:

- Whether the resulting forecasts are *accurate*. Architecture and calibration are
  different claims; only the first is addressed here.
- Full structural-world uncertainty. `structure_id` and `structure_rationale` are carried
  through the spec and the trace, but the compiler still emits one world per run;
  competing causal structures are not yet enumerated and weighted.
- The legacy repository was traced through its `world_model_v2` runtime and its event,
  information and authority modules. Its aggregate/individual prediction machinery
  (`swm/worlds/`, `swm/variables/`, `swm/transition/`) was reviewed at the interface
  level and judged out of scope: it predicts response distributions statistically, which
  is the thing this architecture replaces with simulated trajectories.
