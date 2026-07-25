# The semantic compiler (experimental mode)

`--compiler semantic` tests one hypothesis: that asking a single model call to
*understand the causal world* and simultaneously *author a large internally consistent
executable program* is a major source of compilation failures. The governing principle
of the alternative path:

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

Vertical slice proven on frozen evidence stores; `scripts/semantic_slice.py` is the
inner-loop harness (`--mode semantic|direct` on the same store is the A/B switch). The
mode becomes the default only if the PART-12 promotion standard is met on the frozen
acceptance set plus pre-registered holdouts (`artifacts_holdouts.json`); until then
`direct` remains the default.
