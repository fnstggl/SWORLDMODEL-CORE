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

## The one canonical path

```
forecast(question, as_of, horizon, config)
   → research        (cited evidence store + evidence-grounded scenario frame)
   → contract        (immutable question definition)
   → integrity gate  (refuse a structurally false world)
   → compile         (actors with conditional behavior, institution, protocol, uncertainty)
   → initialize      (one authoritative WorldState per branch)
   → event runtime   (perceive → retrieve → plan/react → reflect → intent → environment)
   → terminal eval   (deterministic vote tally; code, never an LLM)
   → aggregation     (weighted frequency of YES trajectories)
   → report          (fully auditable; probability reconstructable by hand)
```

There is exactly one runtime path. No profiles, no phase pipelines, no fallbacks,
no prior/simulation combiner, no hidden institution model.

## Install & run

```bash
make install          # editable install (stdlib-only runtime; no network needed)
make test             # pytest — 60 invariant + unit + integration tests
make lint             # ruff check + format
make typecheck        # mypy --strict
make banxico          # run the Banxico pastcast; write SEALED pre-outcome artifacts
make banxico-eval     # compare the sealed forecast to the known result
make synthetic        # generalization proofs (7-seat council, data-shock board)
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
- A nine-seat board can never become five modeled units; thresholds are never rescaled.
- Actors emit *intentions*; the environment produces *consequences*.
- No fact available after `as_of` can affect a pastcast (mechanically enforced).
- Deleting the actor calls changes/kills the forecast.
- The full run replays from the event ledger.

See `docs/` for the reality-integrity gate, evidence/cutoff model, actor runtime,
forecast semantics, the legacy-extraction record, and the Banxico evaluation.
