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

**Honest caveat, and the measurement that would close it:** whether this provider
honours `seed` at all is **untested**. The earlier live runs cannot settle it — the ones
that shared a question were made at different commits, so their divergence has a second
explanation. `vary` is therefore best read as a guarantee that the bench is not
*accidentally* pinning a draw, not as a claim about what production would have done. The
experiment that answers it is one `--seed-mode production` arm at N≥5 against a frozen
`src`: if the distribution collapses to a single value, the provider honours seeds and
`vary` is measuring a spread production would never show; if it does not, `vary` and
`production` are sampling the same thing and the flag is a formality. **That arm has not
been run** — roughly 6 calls and 3 minutes per sample, so ~30 calls buys the answer.

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

## RECORDED BASELINE — OPEC+, N=10

This is the number the phase has to move. Reproduce with the exact invocation at the
top of this document.

| | |
| --- | --- |
| store | `artifacts/phase2/geopolitical2/evidence_store.json` — sha256 `e72653aae39d6afe…`, 19 claims, 19 admissible at the cutoff |
| question | sha256 `7472de756a997ab8…` (the acceptance question, `scripts/acceptance.sh:46`) |
| as_of → horizon | `2026-07-25T20:37:43+00:00` → `2026-10-01T23:59:59+00:00` |
| compiler | sha256 `8308e11e7e44888c…` over `src/sworldmodel` (55 files), frozen for the whole run; identical across all 10 |
| mode | `semantic`, gates ON, `--seed-mode vary`, `--bench-seed 20260727` |
| runs | 10 attempted, 10 completed, 0 harness errors |

**Outcome**

```
compiled                3 / 10
refused_gates           4 / 10     coverage_incomplete 3, actors_ungrounded 1
refused_plan            3 / 10     semantic_plan_invalid 3
worlds produced         7 / 10
```

**`parties_holding_acts` — the headline**

```
 0 | ###########                  2      (both FACTUAL RESOLUTION worlds: 0 entities by design)
 1 | ############################ 5
 2 |                              0
```

**`parties_holding_acts` >= 2 in 0 of 7 worlds.** Never once above 1 — which is what the
eleven earlier runs said, now with a denominator and an error bar.

**Every other distribution, over the same 7 worlds**

| metric | distribution |
| --- | --- |
| `entities` | 0×2, 1×4, 2×1 |
| `declared_actors` | 0×2, 1×5 |
| `affordances` | 0×2, 1×3, 2×1, 3×1 |
| `send_effects` | **0 in all 7** |
| `channels` | **0 in all 7** |
| `process_nodes` | 0×2, 1×4, 2×1 |
| `plan_send_changes` | **0 in all 7** |

**What this establishes, and what it does not**

* `parties_holding_acts` never exceeded 1 in 7 worlds. That does **not** mean the
  compiler cannot build a two-party world. The exact one-sided 95% bound is **0.35**: a
  configuration that built one 35% of the time could still show 0 of 7. Rates above 0.35
  are ruled out; nothing below is.
* **To claim an improvement over this baseline, a comparison arm of the same size must
  show `parties_holding_acts` >= 2 in at least 5 of its 7 worlds.** 4 of 7 or fewer is
  inside the noise and must not be reported as a win. Plan for the attrition: **10 runs
  bought 7 worlds**, and the headline denominator is worlds, not runs.
* `LAUNCH_GAP_AUDIT` M5 — *"the planner has never emitted a `send`"* — is now measured
  rather than sampled: 0 send effects and 0 plan-level `send` changes across 7 worlds,
  and 0 channels. The upper bound on the per-world send rate at this N is the same 0.35.
* **A mode nobody had named.** 2 of the 7 worlds were compiled as a *factual resolution*:
  the planner judged the pre-cutoff record to have already settled the question, set
  `expected_participants: 0` with a written justification, and produced a world with **no
  entities and no acts at all**. Every gate admitted one of them. That is the documented
  `_cited_factual_resolution` path (`world_compiler.py:1689`), not a new defect — but it
  means the compiler is bimodal in a second way, and any average taken across both modes
  describes neither. The bench flags these separately for exactly that reason.
* **Entity count did not swing 0→11 here.** The observed range was 0..2. Either the
  compiler at `8308e11e` is materially more compressed than the one that produced the
  recorded 0→11 spread, or that spread pooled runs from more than one configuration.
  This bench cannot say which; it can say that at this configuration, at N=10, the
  spread is 0..2.

**Cost of this baseline**

```
10 runs · 63 provider calls · 631,588 tokens in / 240,087 out
722 s wall at concurrency 4 (2,188 s of compile time summed)
median 180 s and 6 calls per run
```

The run's own `bench.json`, `report.txt` and the ten `run_NN/` world directories were
left in the measuring agent's scratch at
`.../scratchpad/society-measurement/baseline_n10/`. Everything needed to reproduce them
is in this section; nothing above was transcribed by hand from a run that is not on disk.

Roughly **6 calls and 3 minutes per sample**. An N=10 arm is ~12 minutes of wall clock;
an A/B of two N=10 arms is ~25 minutes and ~130 calls. That is cheap enough to run before
shipping a prompt change, which was the design constraint — a full `run_forecast` costs
698 s and 17 calls for a single sample, and at that price nobody would run N=10.

### A note on the earlier eleven-run record

Two things in that record do not survive a re-read, and both are the same failure this
harness exists to end:

1. **The arms used different questions.** Runs v1–v2 carry question hash `7472de75…`;
   runs v3–v5 carry `c62b0f63…` ("Will the seven participating OPEC+ countries agree a
   further increase … at their meeting on 2 August 2026?"). Those arms were never
   comparable. `--compare` now refuses such a pair with
   `THESE ARMS DO NOT SHARE THEIR INPUTS` before printing any p-value.
2. **The ordering probe's recorded artifact does not match its summary.** Commit
   `3e52001` describes the claim-count-desc arm as *"7 entities, 7 deciders, the member
   countries"*. The probe artifact for that arm holds **3 entities**
   (`seven_opec_countries`, `strait_of_hormuz`, `OPEC+`), **1** with `decides: true`, and
   **1** distinct affordance actor — i.e. `parties_holding_acts` = 1, the same as every
   other arm. Flagged, not adjudicated: the artifact may not be the run the message
   describes. Either way it does not support a difference, and at n=1 it could not have.



## What this harness deliberately does not do

* It does not modify anything under `src/sworldmodel/`. A bench that adjusted the
  compiler to get a cleaner reading would be worthless.
* It does not judge world *quality* — only shape. A three-party world with fabricated
  affordances scores the same as a three-party world with grounded ones. The grounding
  gates are what judge that, and the bench reports their verdict rather than replacing
  it.
* It does not aggregate across stores or questions. One bench invocation is one
  configuration on one store.
