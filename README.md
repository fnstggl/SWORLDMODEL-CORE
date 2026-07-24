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
memory seeds, or source URLs are supplied — because none may be *needed*. The system
runs **live research** (real web retrieval, archived at the cutoff for a pastcast,
verified against the fetched page), a **live DeepSeek** world compiler and actors,
simulates, and aggregates.

There is no `--corpus` flag, no offline mode and no deterministic gateway in the
shipped package. Test stand-ins live under `tests/`, where the product cannot reach
them. If the live path cannot answer a question faithfully, that is a fact about the
system and it is reported — not routed around.

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
   → structures      (is this even the right world? competing causal structures, each
                      compiled from the same evidence and simulated)
   → uncertainty     (genuine, weighted, provenance-tagged branches)
   → event runtime   (a real branch calendar: the next scheduled thing happens, is
                      delivered, is noticed, and wakes only the actors it affects)
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
make install          # editable install (stdlib-only runtime)
make test             # pytest — invariants + unit
make lint             # ruff check + format
make typecheck        # mypy --strict
make check            # all three
```

Inspect a written run — what woke each actor, what it was sent, what it returned:

```bash
sworldmodel inspect artifacts/<run> --summary        # invocations per actor, and why
sworldmodel inspect artifacts/<run> --actor <id> --prompts
```

The kernel has **zero runtime dependencies**. Producing a forecast requires a
`DEEPSEEK_API_KEY` and network access, by design.

## Repository layout

```
src/sworldmodel/      the canonical kernel — every file is on the live path
tests/                invariants + unit, plus the test-only fakes and worlds
docs/                 audit, port manifest, architecture and semantics
artifacts/            generated run artifacts (git-ignored)
```

## Key invariants (each has a test)

- The forecast is `weighted_simulated_trajectories` — never a prior or a separate model.
- The production runtime contains **no** routing on a question family (committee /
  negotiation / election / population / geopolitical / response).
- The expected participant roster is derived from **evidence**, not from the compiler's
  own claim about itself, and a person the evidence names as decision-relevant cannot be
  missing from the simulated world.
- An actor is invoked because something reached it. Invocation counts differ between
  actors and between branches; nothing is called on a schedule.
- Noticing something is not acting on it: an actor with an unfinished action is not
  interrupted by a mere opportunity, and its plan persists.
- Nothing uncited is ever shown to an actor as an established fact.
- Every materially relevant verified evidence item is represented and causally wired
  into the compiled WorldSpec that is actually simulated, or explicitly excluded with a
  recorded reason — nothing important is silently dropped between research and
  simulation (`docs/COVERAGE_INTEGRITY.md`).
- Actors emit *intentions*; the environment produces *consequences*.
- A compiled action's behavior is its effects — renaming it changes nothing.
- A novel action never auto-succeeds; if it cannot be represented safely it is rejected,
  never coerced into the nearest known action.
- An invalid intention is refused with its reason and the reason goes back to the actor.
  Nothing is rewritten into a different action, downgraded to waiting, or completed with
  a parameter the runtime chose on the actor's behalf.
- Acting is not succeeding: an action started in a world that has since moved fails
  visibly rather than landing in a world its actor never saw.
- No fact available after `as_of` can affect a pastcast (mechanically enforced).
- Deleting the actor calls changes/kills the forecast; the full run replays from the ledger.

The load-bearing tests are `tests/invariants/`: `test_scheduling.py` (actor calls are
caused, never scheduled), `test_no_coercion.py` (the environment decides what is
possible, the actor decides what it wants, and neither does the other's job),
`test_universality.py` (no question-family routing, proven over the AST) and
`test_forecast_and_replay.py` (the number is the trajectories, and the trajectories
replay).

`docs/CONSOLIDATION_AUDIT.md` records what each of the three repositories actually did
and the KEEP / PORT / REWRITE / DELETE decision for every capability;
`docs/PORT_MANIFEST.md` records each adapted semantic and every behavioral constant
removed; `docs/GENERATIVE_AGENTS_EXTRACTION.md` records what was taken from the
reference implementation and what was rejected.
