# Architecture — one universal world simulator

SWORLDMODEL-CORE is **one universal world simulator**. For every arbitrary question the
LLM compiles the *actual causal world* required to answer it — the entities and actors,
their roles/authority/capabilities, the objects/documents/resources/channels, the real
process that can produce the outcome, the scenario-specific actions each actor may take,
the genuine uncertainties, and the exact declarative condition that makes the answer YES.
One runtime then executes that compiled program on a real calendar and reads the
outcome from world state.

For the audit that produced this architecture, the capability-by-capability decisions,
and the record of what was adapted from where, see `CONSOLIDATION_AUDIT.md` and
`PORT_MANIFEST.md`. For the actor's persistent state and the complete set of reasons it
can be invoked, see `ACTOR_RUNTIME.md`.

It is **not** a committee simulator, a router between predefined scenario types, or a
collection of hardcoded action families. A committee vote, an individual response, a
negotiation, a population behavior, and a geopolitical process are all just different
compiled `WorldSpec` programs executed by the same engine. Supporting a new kind of
question adds compiled **data**, never a new runtime branch, terminal enum, protocol
class, or action handler.

The governing rule:

> **Hardcode only the universal laws by which a world changes. Never hardcode what
> world, process, or actions a question must contain.**

## The one canonical path

```
forecast(question, as_of, horizon, config)          # api.py — the single entry point
  → live research       live_research.py  (rss/search/source_fetch/pdf_text/source_extract)
                        → evidence.py (canonical store, lineage, cutoff)
  → compile (semantic, the default)                                        # api.compile_for_mode
        semantic plan   semantic_plan.py       # LLM authors causal MEANING only
        reality review  semantic_compile.py    # independent LLM judgement
        validation      validate_semantic_plan # code, zero model calls
        delta repair    merge_plan_delta       # revisions merge; no full regeneration
        lowering        semantic_lowering.py → worldspec.py (schema)  # code mints all symbols
        (the direct compiler, world_compiler.compile_world_spec_live, remains ONLY
         behind the explicit --compiler direct diagnostic flag)
                        (handed coverage.evidence_checklist so nothing verified is forgotten)
  → contract            models.ResolutionContract (locks the declarative terminal)
  → reality gate        reality.py                                          # refuse a false world
  → coverage gate       coverage.py  vs world_compiler.world_spec_view      # nothing lost
  → repair loop         api._compile_with_repair → backend.augment_for_coverage
  → uncertainty         uncertainty.py                                      # genuine branches
  → event runtime       engine.py                                           # one universal loop
        per branch:  world.py (state) · actors.py (perceive→plan→intend)
                     engine._drive_activations (event-driven invocation)
                     executor.py (authority/feasibility/timing/target/resource)
                     novel.py (novel-action route) · effects.py (universal ops)
                     expressions.py (declarative terminal + preconditions)
  → aggregate           outcomes.py                                         # weighted YES frequency
  → report              tracing.py (ledger, coverage report, report.md)
```

There is exactly one normal path and it does **not** branch on the kind of question.
No profiles, no phase pipelines, no mechanism families, no fallbacks, no prior/simulation
combiner, no hidden institution model.

## What is hardcoded (the universal laws) vs compiled (the world)

**Hardcoded — the execution language, fixed forever (`effects.py`):**
`create_event`, `schedule_event`, `deliver_information`, `release_data`, `set_field`,
`adjust_field`, `append_record`, `update_commitment`, `transfer_resource`,
`consume_resource`, `create_or_update_document`, `advance_time`. Plus the universal
operators in `expressions.py` (`equals`, `greater_than`, `count`, `sum`, `all`, `any`,
`before`, `after`, `duration`, …). These are the language; they never grow when a new
question appears.

**Compiled per question — the program (`worldspec.py`, produced by `world_compiler.py`):**
the entities/actors, `ActionDefinition` records (each mapping to the universal effects),
the process graph, the genuine uncertainties, and the declarative `TerminalExpression`.
Nothing here is assumed to be a vote, committee, negotiation, election, or population
before compilation.

An action's behavior **is** its compiled effects — renaming an action changes nothing.
Every actor may also propose a **novel action** the compiler did not anticipate; it is
routed through interpretation → authority → feasibility → resource/timing → safe-effect
compilation → execute-or-reject, and never auto-succeeds or is silently coerced into a
known action.

## Responsibility table — one owner per function

| Function                                   | Sole owner            |
|--------------------------------------------|-----------------------|
| Public `forecast()` entry + orchestration  | `api.py`              |
| Live question-only research orchestration  | `live_research.py`    |
| Source discovery / fetching / PDF text     | `rss.py`, `search.py`, `source_fetch.py`, `pdf_text.py` |
| Claim extraction + excerpt verification    | `source_extract.py`   |
| Shared research types + corpus/mock (tests)| `research.py`         |
| Evidence store, cutoff, lineage            | `evidence.py`         |
| Candidate inventory + coverage integrity + exclusion challenge | `coverage.py` |
| Compiled world schema (WorldSpec, ActionDefinition, process graph, terminal) | `worldspec.py` |
| Evidence-grounded world compilation        | `world_compiler.py`   |
| Reality-integrity gate                     | `reality.py`          |
| Authoritative WorldState + actor-local views | `world.py`          |
| Persistent actors (memory/plan/reflect/intent) | `actors.py`       |
| Persistent actor memory stream             | `memory.py`           |
| The one universal event runtime (+ event-driven actor scheduling) | `engine.py` |
| Action authority/feasibility/timing/target/resource validation + execution | `executor.py` |
| Universal world-effect execution language  | `effects.py`          |
| Novel-action interpretation + validation   | `novel.py`            |
| Declarative terminal + condition evaluation| `expressions.py`      |
| Uncertainty branching + mass accounting    | `uncertainty.py`      |
| Trajectory aggregation (the forecast)      | `outcomes.py`         |
| Provider-independent gateway + test gateways | `gateway.py`        |
| Real production DeepSeek calls             | `deepseek_gateway.py` |
| Compiled-world container                   | `compiled.py`         |
| Tracing / under-the-hood report            | `tracing.py`          |

## Production dependency graph from `forecast()`

A runtime import audit (import `sworldmodel.api`, run a forecast, inspect
`sys.modules`) confirms that execution beginning at `api.forecast()` loads only the
modules above and **cannot reach any superseded module**. There is exactly one path to
the simulated result.

## Deletion list — superseded modules removed in the universal rebuild

These older files implemented mechanism-family routing / committee assumptions and were
deleted (their still-useful universal logic folded into the canonical modules above).
No compatibility wrappers, re-export shims, or dormant runtimes remain.

| Deleted file            | Replaced by                          | Why removed |
|-------------------------|--------------------------------------|-------------|
| `mechanisms.py`         | `expressions.py` + `engine.py`       | Two hardcoded terminal families (`committee_vote`, `actor_action`) → one declarative evaluator |
| `protocols.py`          | `worldspec.ProcessGraph` + `engine.py` | Two fixed protocols (`committee_protocol`, `general_protocol`) → compiled process graphs |
| `compiler.py`           | `world_compiler.py`                  | Committee-shaped compiler (always built an institution) → universal WorldSpec compiler |
| `universal_compiler.py` | `world_compiler.py`                  | A "universal" wrapper that still emitted `terminal.mechanism="committee_vote"` |
| `runtime.py`            | `engine.py`                          | Protocol-step interpreter with mechanism dispatch → one universal event loop |
| `intents.py`            | `executor.py` + `effects.py`         | Fixed `IntentKind`→`EventKind` mapping → compiled actions over universal effects |

Removed from `models.py`: `DecisionRule`, `InstitutionSpec`, `TerminalSpec` (mechanism),
`ReactionRule`, `ConditionalBehavior`, committee `ActorDefinition`, `SignalDef`,
`ScenarioFrame`, `Proposal`, `CausalGraph`, `IntentKind`, and the committee `EventKind`
ontology. The research planner's fixed `process_type` enum (which included
`committee_vote`) became a free-text `process_summary`.

## Invariants (each has a test)

- The forecast is `weighted_simulated_trajectories` — never a prior or a separate model.
- A claimed participant count can never exceed the verified roster (a nine-participant
  body can never become five modeled units); the reality gate refuses.
- Actors emit *intentions*; the environment (executor) produces *consequences*.
- A compiled action's behavior is its effects: renaming it changes nothing.
- A novel action never auto-succeeds; if it can't be represented safely it is rejected,
  never coerced into the nearest known action.
- No fact available after `as_of` can affect a pastcast (mechanically enforced).
- The production runtime contains no routing on a question family.
- Deleting the actor calls changes/kills the forecast; the full run replays from the ledger.

See `docs/REBUILT_UNIVERSAL_MERGE.md` for how the live-research and universal-simulator
lines were fused, and `docs/` for the reality-integrity gate, evidence-to-world coverage
integrity, the evidence/cutoff model, the actor runtime, forecast semantics, the live
product, and the Banxico evaluation.
