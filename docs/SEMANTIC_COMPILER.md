# The semantic compiler (the default and canonical path)

The semantic compiler is the DEFAULT: the CLI, the API, both frozen-store harnesses,
the acceptance scripts and every example select it when no compiler is named. It began
as a test of one hypothesis — that asking a single model call to *understand the
causal world* and simultaneously *author a large internally consistent executable
program* is a major source of compilation failures — and is now the canonical
engineering path all development, testing, instrumentation and optimization target.

The direct compiler (one call authors the WorldSpec) survives only behind the explicit
diagnostic flag `--compiler direct`, for controlled comparison, regression diagnosis
and removal planning. Nothing ever falls back from semantic to direct; a semantic
refusal is the run's result; the compiler is never chosen by question type; and resume
rejects artifacts recorded under the other mode (`compiler_mode_mismatch`). Every run
records its compiler mode. The governing principle:

> THE LLM SUPPLIES CAUSAL MEANING. CODE SUPPLIES INTERNAL SYMBOLS AND EXECUTABLE SYNTAX.

## The path

```
question → research (unchanged)
        → semantic causal-world plan          one model call, meaning only
        → independent reality review          one model call, separate context
        → static semantic validation          code, zero model calls
        → deterministic two-pass lowering     code, zero model calls
        → the existing WorldSpec              same gates, same runtime, same auditors
```

Both compiler modes emit the identical compilation dict, so `assemble_bundle`, every
`compile_world` gate, the engine, the outcome aggregation, the auditors and the replay
viewer are byte-for-byte the same executor. The mode is recorded in the research trace
(`semantic_compilation` carries the plan, the review verdict and the mapping artifact).

## The semantic plan (`semantic_plan.py`)

Universal structural types only — entity, state, event, process, action affordance,
uncertainty, terminal query — with every real-world meaning inside them open-ended
natural language. It contains **no** runtime IDs, field namespaces, effect-operation
names, expression ASTs, node IDs or binding strings. Universal change operations:
`set`, `increase`, `decrease`, `record_event`, `send` (+ `schedule`, parsed but
currently a declared LOWERING_GAP). Universal terminal forms: `event_exists`,
`state_equals`, `quantity_comparison`, `record_count`, `all_of`, `any_of`, `not`.

`UNKNOWN` is the single sentinel for a value the evidence does not establish; it lowers
to an absent initial, which the runtime reads as honestly unresolved — never zero,
never False.

The static validator (`validate_semantic_plan`, zero model calls) enforces: every
reference resolves; every deciding entity cites evidence, has an affordance, and is a
participant of at least one dated `actor_moment` (an affordance nobody is ever woken to
use is an inert world); non-agent processes carry no participants; precise initial
numbers cite claims or are UNKNOWN; no uncertainty writes the terminal state and no
producer sets it to a bare copy of an uncertain state; weights are grounded or
explicitly symmetric-ignorance; declared participant counts reconcile with
`represents_count`.

## The independent review (`semantic_compile.py`)

A separate model call receives the question, the evidence and the plan — none of the
planner's reasoning — and returns exactly one verdict: APPROVE, REVISE (with exact
semantic corrections) or ABSTAIN. One targeted revision is allowed; one further
validator-only round may fix mechanical inconsistencies the revision introduced; then
the refusal is real (`semantic_plan_invalid`, with every unresolved reason).

## Delta revisions (`merge_plan_delta`)

The initial plan is the ONLY full generation of a compile cycle. Every revision —
validator fix, reviewer correction, lowering gap, repair carried across cycles — is a
`semantic_plan_delta` call: the prompt appends the accepted prior plan and the exact
corrections after the unchanged full-plan prompt (so the shared prefix is priced by
the provider's context cache), and the reply contains only the corrected objects,
upserted/removed BY NAME. `merge_plan_delta` rebuilds the plan deterministically:
objects the delta does not name are carried through as the same objects —
byte-equivalent citations, weights, uncertainty structure and causal wiring — which is
what makes a revision unable to silently re-roll unchanged structure (the failure that
once deleted a cited downside alternative). A full-plan re-emission is a named shape
error re-asked once as a delta; an unusable delta refuses as
`semantic_plan_invalid`. Each `_semantic` record carries `delta_rounds` so a trace
reader can verify no full plan was regenerated for a local defect.

## Deterministic lowering (`semantic_lowering.py`)

Pass 1 mints every runtime symbol (entity/field/action/node/external/event/authority)
into a `SymbolTable` keyed by (namespace, semantic name) — deterministic slugs,
collision-suffixed in plan order, so identical plans lower to byte-identical
executables. Pass 2 resolves every reference through the table and emits the WorldSpec:

- `set` → `set_field`; `increase`/`decrease` → `adjust_field` (negated delta);
- `record_event` → `create_event` (typed) + `append_record` (record collection), so
  both `event_exists` and `record_count` terminals count real produced occurrences;
- `send` → `deliver_information` through the existing delivery/notice runtime;
- `actor_moment` → a process node with participants, date and allowed actions;
  dependency-chained operational processes → nodes with `after_node`; dated operational
  processes → `external_processes`;
- terminal forms → the existing expression operators, written by code alone.

A meaning the mapping cannot represent raises `LOWERING_GAP` with the unsupported
construct, why it cannot be represented, whether it composes from current primitives,
and the smallest genuinely universal missing capability. It is never approximated and
never silently dropped.

## What this does not change

The runtime contract, the gates, the research stage, the auditors, the acceptance
protocol. No domain event list exists anywhere: "Bailey publicly communicates
conditional support for a further cut" is an event *meaning* declared by the plan, and
a previously unseen domain (a harbor authority, an irrigation council, an observatory —
the regression suite's invented worlds) compiles through exactly the same code path.

## Status

The FULL canonical route now runs from frozen evidence stores:
`scripts/frozen_forecast.py` executes `api.forecast()` itself — repair loop,
pre-rollout world review, structural alternatives, the event runtime, aggregation and
both auditors — with the only substitution at the retrieval boundary (claims come from
a prior run's exported store). `scripts/semantic_slice.py` remains the fast
compile-only inner loop (plan → review → validate → lower → gates, no simulation). On
both, `--mode semantic|direct` on the same store is the A/B switch. Both harnesses
share `scripts/_store_loader.py`: a full-fidelity export keeps every claim's real
provenance (authority level, source type, publication/validity dates, source id,
confidence) so authority ranking matches the live run; a legacy 8-field store falls
back to defaulted provenance with a loud warning, because a defaulted store ranks
evidence differently than the live run did. Each harness run stamps its output
directory (`run_stamp.json`) and clears any prior run's pipeline artifacts first, so a
refusal can never leave a stale `forecast.json` to be scored as fresh.

The static validator additionally enforces the **re-perform-history rule**
(`validate_semantic_plan` in `semantic_plan.py`): a process occurrence or actor_moment
dated at or before the cutoff is refused — the simulation window opens at the cutoff
and the world may not re-enact history. An outcome the record already establishes
belongs in a cited initial state value or `world_facts`, never in a scheduled
occurrence; a live run once resolved YES off a simulated re-enactment of the very
event the question asked about, which this rule now makes impossible.

Two lowering capabilities are the design contract of work landing in parallel; until
it lands the lowerer emits them empty, which is a known fidelity gap versus the
direct route, not parity:

- **Wake-rule derivation.** The lowered WorldSpec currently carries
  `wake_rules: []`. The contract: wake rules are derived deterministically from the
  plan's declared meanings so semantic worlds get the same event-driven "wakes only
  the actors it affects" behavior the direct compiler's specs express, instead of an
  empty rule set.
- **Required-reality-facts lowering.** The compilation currently carries
  `required_reality_facts: []`. The contract: they are lowered from the plan's cited
  reality (not emitted empty) so the reality-integrity gate can check the compiled
  world against the facts it is REQUIRED to contain on the semantic route exactly as
  it does on the direct route.

### The A/B benchmark procedure

Promotion is decided on frozen stores so retrieval variance is zero by construction:
every question in the frozen acceptance set (`artifacts/acceptance/questions.json`)
plus the pre-registered holdouts (`artifacts_holdouts.json`) is run through
`scripts/frozen_forecast.py` twice from the SAME store — once `--mode semantic`, once
`--mode direct`. The per-question outcome matrix (mode, outcome, failure stage, wall
seconds, model calls, probability) is accumulated in `artifacts/ab_matrix.json` and
the human-readable comparison is written to `artifacts/ab/RESULTS.md`. The semantic
mode becomes the default only if the PART-12 promotion standard is met on that
matrix; until then `direct` remains the default. Neither artifact exists yet — the
benchmark has not been run.
