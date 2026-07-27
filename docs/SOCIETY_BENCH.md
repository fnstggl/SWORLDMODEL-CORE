# SOCIETY BENCH — measuring the compiler instead of eyeballing it

`scripts/society_bench.py` runs the compile stage N times against a frozen evidence
store and reports the **distribution** of the society metrics, plus a plain statement of
what that N can and cannot distinguish.

## Why it exists

From `docs/LAUNCH_GAP_AUDIT.md`, after eleven live compiles on one frozen OPEC+ store:

> *n=1 per arm, and this store's entity count has swung 0→11 across runs. My prompt
> edits are inside that noise, which is itself the finding — I should not have shipped
> either change on a single read.*

A prompt change was shipped on one sample and reverted on one sample. Both were noise.
Every compiler change before this one was judged the same way. The same eleven runs left
one stable signal:

> **`parties_holding_acts` has never once exceeded 1.** Entity count swings wildly; the
> number of parties that can ACT has never moved.

Entity count is noise. Acting parties is the product. This harness makes that
measurable.

## Running it

```bash
PYTHONPATH=src python3 scripts/society_bench.py \
    --store artifacts/phase2/geopolitical2/evidence_store.json \
    --question "Will OPEC+ announce an increase in crude oil production quotas at or before its next scheduled ministerial meeting in 2026?" \
    --as-of 2026-07-25T20:37:43+00:00 --horizon 2026-10-01T23:59:59+00:00 \
    --runs 10 --concurrency 4 --label opec_baseline --bench-seed 20260727 \
    --out /tmp/bench/baseline
```

Writes `bench.json` (full machine-readable record, including every run), `report.txt`
(the rendered report) and `run_NN/` per run (the semantic plan and the lowered
compilation, so any number in the report can be traced back to the world it came from).

To compare two configurations that were each benched separately:

```bash
python3 scripts/society_bench.py --compare /tmp/bench/a/bench.json /tmp/bench/b/bench.json
```

### Freeze the compiler first

This branch has many agents editing `src/`. A ten-run bench takes ~10 minutes of wall
clock, and a commit landing in the middle of it produces an arm that is two different
compilers averaged together. **Copy `src/` and point the bench at the copy:**

```bash
cp -r src /tmp/frozen_src
PYTHONPATH=/tmp/frozen_src python3 scripts/society_bench.py ...
```

The bench records a SHA-256 over the `sworldmodel` package it actually imported, per
run. If the runs in one arm disagree on that hash, the report refuses to pool them and
says so in capitals. This is not hypothetical: while this harness was being built,
`src/` changed between a smoke run (`07e27f20…`) and the baseline twenty minutes later
(`8308e11e…`).

### Options that change what is measured

| flag | default | effect |
| --- | --- | --- |
| `--runs N` | 5 | samples per configuration |
| `--concurrency K` | 3 | runs in flight at once; each gets its own gateway |
| `--mode semantic\|direct` | `semantic` | which compiler is under test |
| `--no-gates` | off | stop after lowering. Saves the exclusion-challenge calls; costs the refusal code, and the world measured may be one no gate would admit |
| `--seed-mode vary\|production` | `vary` | see below |
| `--bench-seed N` | 0 | makes one bench invocation replayable |
| `--max-calls N` | 40 | per-run provider ceiling; a runaway repair loop stops instead of spending the whole bench's budget |

## The metrics

Read off the **lowered world spec** — what was actually built, not what the planner said
it was building:

| metric | definition |
| --- | --- |
| **`parties_holding_acts`** | **HEADLINE.** Distinct entity ids named in at least one action's `eligible_actors`. Deliberately not "entities marked `is_actor`" and not `len(actors)`: the question is not who was labelled a decider but who was handed something to do. An action with no eligible actor is environment-driven and contributes nobody. |
| `entities` | `len(world_spec.entities)`. The number that swings 0→11 and means nothing on its own. |
| `declared_actors` | `len(world_spec.actors)`. `phase2/geopolitical2` had 9 entities and 1 of these. |
| `affordances` | `len(world_spec.actions)`. |
| `send_effects` | Effects whose op is `deliver_information` — the lowered form of a plan-level `send`. Counted from the effect the lowerer produced, never from the planner's word, so a plan that says "send" and lowers to nothing cannot be scored as communication. |
| `channels` | `len(world_spec.channels)`. |
| `process_nodes` | `len(world_spec.process.nodes)`. |

Read off the **semantic plan**, before lowering, because that is what a reader
inspecting a plan by hand sees, and the two do not always agree — a plan can declare
seven deciders and lower to one party holding every act:

`plan_entities`, `plan_deciders` (entities with `decides: true`), `plan_affordances`,
`plan_send_changes` (changes with `op: "send"`).

### Outcomes

| outcome | meaning |
| --- | --- |
| `compiled` | every gate passed |
| `refused_gates` | a world WAS built and a gate rejected it. Its metrics are still in the distribution — this is the most informative case there is |
| `refused_plan` | no world was produced; nothing to measure |
| `harness_error` | **our** infrastructure failed (provider outage, exhausted ceiling). Never counted as a compiler result. If a world had already been built when the failure hit, that world stays in the distribution — the outage cost the gate verdict, not the world |

`GatewayError` subclasses `SWorldModelError`, so the obvious `except` ordering files an
unreachable endpoint in the same column as `coverage_incomplete`. The harness catches it
first, and `tests/unit/test_society_bench.py` pins that.

The **refusal code** is `details["failure"]` from the pipeline's own exception —
`coverage_incomplete`, `actors_ungrounded`, `semantic_plan_invalid`, and so on. The
report shows the distribution of those codes, because *which* gate refuses is itself a
measurement of where the generator is stuck.

## Seeds: what `--seed-mode vary` does and why

`semantic_compile.py` derives its planner seed from the question text:

```python
seed=int(prompt_hash(f"semantic{question}{attempt}")[:8], 16)
```

That is right for a production run — one question, one draw — and wrong for a bench,
which needs N draws from one configuration. Under `--seed-mode vary` (the default) the
harness wraps the gateway and re-derives each request's seed from `(bench_seed,
run_index, task_kind, original seed)`. The wrapper changes nothing inside the compiler;
it substitutes a seed at the boundary the architecture already provides, and it is
derived rather than random so one `--bench-seed` reproduces one bench.

`--seed-mode production` leaves the compiler's seeds alone. The two measure different
things and the report always names which ran.

**Honest caveat:** it is not established that this provider honours `seed` at all. The
recorded evidence points the other way — eleven production runs of one question, all
with the same derived seed, produced wildly different worlds. `vary` is therefore best
read as a guarantee that the bench is not *accidentally* pinning a draw, not as a claim
that production would have been deterministic without it.

## What is cut, and what that costs

Scope is the **compile stage only**: the real planner and reviewer prompts, the real
parser, validator and lowerer, `assemble_bundle`, and every `compile_world` gate — the
same functions `scripts/semantic_slice.py` exercises, on the same frozen-store loader
(`scripts/_store_loader.py`). No mock stands in for any of them.

Cut:

* **Research.** The store is frozen and identical across every run and every arm. That
  is what makes an A/B an A/B; it also means the bench says nothing about retrieval.
* **The simulation.** A full `run_forecast` cost a peer 698 seconds and 17 provider
  calls. At that price N=10 is unaffordable, and an unaffordable instrument is an unused
  one.

The fidelity that costs: **this bench measures what was BUILT, not what happened.** It
cannot see whether the parties holding affordances ever use them, whether a compiled
channel carries a message, or whether the terminal resolves. `parties_holding_acts` is
an upper bound on the parties that can matter — never a claim that they did. Moving it
is necessary for a society and is not sufficient for one; the runtime evidence for
sufficiency has to come from somewhere else.

## Reading the NOISE section

Everything statistical here is exact and stdlib-only (`math.comb`), so a reader can
check it by hand at the N a bench can afford. No stats library is imported.

* **"any rate up to X is consistent with this result"** — the one-sided 95%
  Clopper–Pearson upper limit, found by bisection on the exact binomial tail. For k=0
  the closed form is `1 − 0.05**(1/n)`: **0.45 at N=5, 0.26 at N=10**. Zero out of ten
  does not mean impossible; it means a compiler that built the thing a quarter of the
  time would show this result about one time in twenty.
* **"a comparison arm must show ≥ M of N"** — the smallest split that reaches p≤0.05 on
  the two-sided Fisher exact test against this arm. **At N=5 per arm that is 4 of 5; at
  N=10 it is 5 of 10; at N=3 no split exists at all**, and the harness says
  `NOT ENOUGH RUNS TO COMPARE ANYTHING` rather than printing two numbers.
* **Count metrics** in a comparison get an exact two-sided permutation test on the
  difference of means — every split of the pooled values enumerated, no distributional
  assumption. Beyond a cap it returns nothing rather than an approximation nobody named.
* **Multiplicity is not corrected.** Eleven metrics are printed; at p≤0.05 each, about
  one in twenty looks different by chance. Only `parties_holding_acts` is the
  pre-registered result. Every other number is descriptive.

A comparison also refuses to be read as an A/B when the two arms disagree on question,
store, cutoff, horizon, mode or gate setting — it leads with
`THESE ARMS DO NOT SHARE THEIR INPUTS` instead of a p-value. (The eleven-run episode
changed the question halfway through: runs 1–2 carry question hash `7472de75…`, runs
3–5 carry `c62b0f63…`. Those arms were never comparable.)

<!-- BASELINE -->

## What this harness deliberately does not do

* It does not modify anything under `src/sworldmodel/`. A bench that adjusted the
  compiler to get a cleaner reading would be worthless.
* It does not judge world *quality* — only shape. A three-party world with fabricated
  affordances scores the same as a three-party world with grounded ones. The grounding
  gates are what judge that, and the bench reports their verdict rather than replacing
  it.
* It does not aggregate across stores or questions. One bench invocation is one
  configuration on one store.
