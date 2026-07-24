# Architecture

One package, one canonical runtime path, an acyclic import graph. A developer can
trace the whole causal route from `forecast()` to terminal aggregation without
navigating phase adapters.

## The causal route

```
api.forecast / api.run_forecast
  │
  ├─ research.ResearchBackend.research  ──►  ResearchBundle
  │     • complete EvidenceStore (never a truncated string)
  │     • evidence-grounded ScenarioFrame (options, signals, reaction rules,
  │       guidance, joint uncertainty with provenance)
  │     • verified reality inputs (roster, roles, prior actions, rule, deadline)
  │
  ├─ evidence.EvidenceStore.view(as_of) ──►  EvidenceView   (cutoff-bounded)
  │
  ├─ models.ResolutionContract          ──►  immutable question definition
  │
  ├─ compiler.compile_world
  │     • gateway "compile_world": conditional behavior per verified member
  │     • deterministic validation into typed ActorDefinition / InstitutionSpec
  │     • reality.verify_reality  ── RAISES WorldIntegrityError if unfaithful
  │     • protocols.committee_protocol, uncertainty.enumerate_scenarios,
  │       causal graph                     ──►  CompiledWorld
  │
  ├─ runtime.run  (per uncertain-future scenario)
  │     world = base_world.clone(branch)
  │     release scenario data (a simulated future, not post-cutoff evidence)
  │     for each protocol step:
  │        view   = world.view_for(actor, trigger)          # local view
  │        intent = actor_runtime.step(actor_state, view)    # perceive→…→emit
  │        valid  = environment.validate(intent, world)
  │        events = environment.execute(valid, world)        # consequences here
  │        world  = world.apply(events)
  │     terminal = mechanisms.evaluate_terminal(votes, …)    # deterministic
  │                                            ──►  BranchOutcome per branch
  │
  ├─ outcomes.aggregate  ──►  ForecastResult   (weighted YES frequency + bounds)
  └─ tracing.TraceContext.write ──►  sealed, replayable artifacts + report
```

## Module map

| module | responsibility |
|---|---|
| `models.py` | pure typed domain records (contract, evidence enums, world building-blocks, events, intents, forecast result, scenario frame) |
| `errors.py` | explicit exceptions that stop a run instead of being repaired |
| `ids.py` | deterministic ids, hashing, canonical JSON serialization |
| `evidence.py` | canonical `EvidenceStore`, cutoff-bounded `EvidenceView`, lineage dedup |
| `research.py` | `ResearchBackend` (corpus + mock), `ResearchBundle`, backward research plan |
| `reality.py` | `verify_reality` — the reality-integrity gate |
| `compiler.py` | constrained structured generation + deterministic validation → `CompiledWorld` |
| `world.py` | the one authoritative `WorldState`; `view_for`; validated `apply`; branch clone |
| `actors.py` | persistent `ActorState`, `LocalView`, `ActorRuntime` (the cognitive loop) |
| `memory.py` | persistent `MemoryStream` (recency×importance×relevance retrieval) |
| `protocols.py` | declarative `ProtocolGraph` of generic institutional primitives |
| `intents.py` | `Environment` — intent validation + execution into events |
| `mechanisms.py` | deterministic tally and terminal-predicate evaluation (no LLM) |
| `uncertainty.py` | joint scenario enumeration, mass conservation, disclosed truncation |
| `gateway.py` | `ModelGateway` boundary; `DeterministicGateway`, `ScriptedGateway` |
| `prompts.py` | prompt rendering from the same context the model reasons over |
| `runtime.py` | the branch rollout engine + protocol interpreter |
| `outcomes.py` | trajectory-only aggregation and bounds |
| `tracing.py` | replayable ledger, manifests, under-the-hood report |
| `api.py` | the single `forecast()` entry point |
| `cli.py` | the *evaluation harness* (Banxico/synthetic constants live here, not in core) |
| `deepseek_gateway.py` | live DeepSeek `ModelGateway` (real HTTP, retries, real tokens) |
| `http.py` | HTTP transport (real `UrllibTransport` + `FakeTransport` for tests) |
| `search.py`, `rss.py` | discovery: DuckDuckGo real URLs; Google News RSS signal |
| `source_fetch.py`, `source_extract.py` | fetch + verify pages; LLM claim extraction |
| `research_planner.py` | LLM research plan (backward from the outcome) |
| `live_research.py` | `LiveResearchBackend`: iterative research → `ResearchBundle` |
| `universal_compiler.py` | LLM reality + uncertainty compilation from live evidence |

The **live** path (`live_research` + `deepseek_gateway`) and the **corpus** path
(`CorpusResearchBackend` + `DeterministicGateway`, tests only) are two front-ends that
feed the *same* `compile_world` → `runtime.run` → `outcomes.aggregate`. The terminal is
dispatched by mechanism (`committee_vote` incl. weighted strata, or `actor_action`), so
one runtime handles committees, populations, and single-actor/negotiation questions.
See `docs/LIVE_PRODUCT.md`.

## Import direction (acyclic)

```
ids, errors                → leaves
models                     → ids, errors
evidence                   → models
memory                     → models, ids
gateway, prompts           → models, ids, errors
actors                     → models, memory, gateway, prompts
mechanisms                 → models
world                      → models, evidence, actors, mechanisms
reality                    → models, evidence, actors
protocols, uncertainty     → models
intents                    → world, models
research                   → models, evidence, world
compiler                   → world, reality, protocols, uncertainty, research, gateway
runtime                    → compiler, intents, mechanisms, actors, world
outcomes, tracing          → models (+ compiler/research/runtime for tracing)
api                        → compiler, runtime, outcomes, tracing, research, config
```

There are **no imports from any reference repository** and no compatibility shims.

## Why one authoritative state

Each branch owns exactly one `WorldState`. Everything an actor sees is a *projection*
(`view_for`) of it. There is no separate "knowledge packet", "deliberation state", or
"institution override" that owns competing facts. Actors never receive a `WorldState`
reference — they receive a read-only `LocalView` — so they structurally cannot mutate
reality or read another actor's private state.
