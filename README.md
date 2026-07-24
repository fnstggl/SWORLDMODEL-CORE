# SWORLDMODEL-CORE

A general, **reality-first** social world model. The reported forecast comes *only*
from simulated actor trajectories inside a verified, persistent world. World
completion is secondary; reality fidelity is primary.

The governing question is not *"how do we make every generated world run to
completion?"* It is *"is this actually the right world?"*

## The product: question only, live

```bash
export DEEPSEEK_API_KEY=...        # provider credentials (never committed)
sworldmodel forecast \
  --question "Will [social/institutional event] happen by [date]?" \
  --as-of  "2026-05-14T23:59:59-04:00" \
  --horizon "2026-06-25T23:59:59-04:00"
```

No corpus, roster, decision rule, protocol, thresholds, uncertainties, weights,
memory seeds, or source URLs are supplied. The system runs **live research** (real web
retrieval + verification), a **live DeepSeek** world compiler and actors, simulates,
and aggregates. See `docs/LIVE_PRODUCT.md`. The deterministic reasoner and prepared
corpora are confined to tests and the historical-evaluation fixture; the live command
refuses to run "live" without a live gateway + live research.

This is **one universal world simulator**, not a committee engine or a router between
scenario types. For every arbitrary question the LLM compiles the *actual causal world*
— entities, actors, capabilities, objects/resources, the real process, the
scenario-specific actions, the genuine uncertainties, and the exact declarative
condition that makes the answer YES — and one runtime executes it. A committee vote is
only one world the compiler may produce; an individual response, negotiation, population
behavior, or geopolitical process compiles an entirely different world with **no source
change**.

> Hardcode only the universal laws by which a world changes. Never hardcode what world,
> process, or actions a question must contain.

## The one canonical path

```
forecast(question, as_of, horizon, config)
   → research        (cited evidence store, built live from the question)
   → compile         (LLM compiles a WorldSpec: entities, actions→universal effects,
                      process graph, uncertainties, declarative terminal)
   → contract        (immutable question definition; locks the declarative terminal)
   → integrity gate  (refuse a structurally false world)
   → uncertainty     (genuine, weighted, provenance-tagged branches)
   → event runtime   (one universal loop: perceive → plan → intend → environment executes)
   → terminal eval   (deterministic declarative predicate over world state; code, not an LLM)
   → aggregation     (weighted frequency of YES trajectories)
   → report          (fully auditable; probability reconstructable by hand)
```

There is exactly one runtime path and it does **not** branch on the kind of question.
No profiles, no phase pipelines, no mechanism families, no fallbacks, no prior/simulation
combiner, no hidden institution model. The runtime hardcodes only a small, fixed
*effect language* (create event, deliver information, set field, append record, transfer
resource, …) and universal *operators* (equals, count, sum, all, before, …); the LLM
compiles the scenario-specific program from those primitives. Actors may also propose
**novel actions** the compiler did not anticipate, validated (authority → feasibility →
safe effects) before the world decides their consequence.

## Install & run

```bash
make install          # editable install (stdlib-only runtime; no network needed)
make test             # pytest — acceptance + invariant + unit + integration tests
make lint             # ruff check + format
make typecheck        # mypy --strict
make banxico          # run the Banxico pastcast; write SEALED pre-outcome artifacts
make banxico-eval     # compare the sealed forecast to the known result
make synthetic        # cross-domain proofs (committee, response, negotiation, population, geopolitical)
```

The kernel has **zero runtime dependencies** and runs offline and deterministically
via a transparent `DeterministicGateway` (a calibrated-behavior stand-in for a live
LLM). A live provider gateway implements the same interface.

## Repository layout

```
src/sworldmodel/      the canonical kernel (see docs/ARCHITECTURE.md)
tests/                unit / invariants / integration / fixtures
evaluation/           Banxico + synthetic corpora (scenario data, never in core)
docs/                 architecture and semantics
artifacts/            generated run artifacts (git-ignored)
```

## Key invariants (each has a test)

- The forecast is `weighted_simulated_trajectories` — never a prior or a separate model.
- The production runtime contains **no** routing on a question family (committee /
  negotiation / election / population / geopolitical / response).
- A claimed participant count can never exceed the verified roster (a nine-participant
  body can never become five modeled units).
- Every materially relevant verified evidence item is represented and causally wired
  into the compiled WorldSpec that is actually simulated, or explicitly excluded with a
  recorded reason — nothing important is silently dropped between research and
  simulation (`docs/COVERAGE_INTEGRITY.md`).
- Actors emit *intentions*; the environment produces *consequences*.
- A compiled action's behavior is its effects — renaming it changes nothing.
- A novel action never auto-succeeds; if it cannot be represented safely it is rejected,
  never coerced into the nearest known action.
- No fact available after `as_of` can affect a pastcast (mechanically enforced).
- Deleting the actor calls changes/kills the forecast; the full run replays from the ledger.

The mandatory acceptance tests live in `tests/acceptance/test_universal.py`. See
`docs/ARCHITECTURE.md` for the responsibility table, the production dependency graph
and the deletion list; `docs/REBUILT_UNIVERSAL_MERGE.md` for how the live-research and
universal-simulator lines were merged; and `docs/` for the reality-integrity gate,
evidence-to-world coverage integrity, the evidence/cutoff model, the actor runtime,
forecast semantics, and the Banxico evaluation.
