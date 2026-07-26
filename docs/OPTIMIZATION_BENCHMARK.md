# Optimization benchmark — frozen eight-case A/B

Benchmark commit **`2a489a3`** — one unchanged commit for all sixteen runs. Five
acceptance cases plus three holdouts, both compiler modes, each pair reading the
*identical* frozen evidence store (`artifacts/stores/<case>/evidence_store.json`,
frozen by `scripts/freeze_stores.py` in one live retrieval pass per case). Per-run
rows: `artifacts/ab/ab_results.jsonl`; regenerate this table with
`python3 scripts/ab_report.py`.

The equivalence regressions (delta-merge byte-equivalence, serial-versus-parallel
replay identity, memo in-flight dedup, cache replay re-verification) were proven
**before** this benchmark ran — see `tests/unit/test_plan_delta.py`,
`tests/unit/test_semantic_delta_compile.py`,
`tests/invariants/test_parallel_equivalence.py`, `tests/unit/test_runcache.py`.

## Headline

| mode | n | wall median | wall p90 | calls med | tokens in med | tokens out med | cache hit rate | memo reuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **semantic** | 8 | **8.1 min** | 15.7 min | 20 | 74,288 | 47,200 | **30.8%** | 0 |
| direct | 8 | 3.7 min | 7.0 min | 9 | 29,481 | 22,988 | 6.5% | 36 |

Totals across the eight runs of each mode: semantic 643,779 in / 427,074 out with
226,048 prompt tokens (35.1%) served from the provider's context cache; direct
260,402 in / 263,226 out with 49,664 cached (19.1%).

**Cost.** Reported as token totals. No provider list price is asserted here —
`scripts/ab_report.py` prices the run only when `DEEPSEEK_PRICE_IN` /
`DEEPSEEK_PRICE_OUT` / `DEEPSEEK_PRICE_CACHED` are supplied, so no invented number
enters the record. On any per-token pricing, semantic's cached share (35.1% of input)
is billed at the cache-hit rate rather than the full input rate.

**1–5 minute consumer target: NOT reached.** Median supported semantic runtime is
8.1 minutes (p90 15.7). One case in the set — `committee` — is a zero-claim store
whose semantic run burned the entire 900-second compile budget; excluding it the
median is 7.6 minutes and the p90 11.4. That case is now fixed (see *After the
benchmark*), but the target remains unreached and is **not** claimed.

## Per-case

| case | semantic | p | wall | calls | direct | p | wall | calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| committee | refused (compilation) | — | 15.7 min | 22 | refused (compilation) | — | 1.1 min | 3 |
| geopolitical | resolved | 0.00 | 8.6 min | 17 | resolved | 1.00 | 13.5 min | 28 |
| holdout_science_institution | resolved | 1.00 | 4.5 min | 18 | resolved | 1.00 | 3.9 min | 9 |
| holdout_space_operations | partially_resolved | 0.75 | 17.4 min | 32 | refused (compilation) | — | 2.5 min | 2 |
| holdout_sports_governance | resolved | 0.70 | 11.4 min | 34 | resolved | 0.00 | 3.3 min | 5 |
| individual | refused (compilation) | — | 7.6 min | 13 | resolved | 0.50 | 5.4 min | 10 |
| negotiation | refused (compilation) | — | 3.5 min | 6 | refused (compilation) | — | 3.5 min | 25 |
| population | resolved | 1.00 | 7.4 min | 21 | partially_resolved | 1.00 | 7.0 min | 9 |

Outcome counts: semantic 4 produced a number (3 resolved, 1 partially resolved) and 4
refused; direct 5 produced a number (4 resolved, 1 partially resolved) and 3 refused.

## Did the optimizations do what they were built to do?

**Yes, each is visible in the instrumentation.**

- **Delta repair.** `semantic_plan` median **1** against `semantic_plan_delta` median
  **8**: exactly one full plan generation per compile cycle, every revision a delta.
  No local validator or reviewer defect regenerated a plan in any of the sixteen runs.
- **Prompt-prefix reuse.** 30.8% median (35.1% aggregate) of semantic input tokens
  served from the provider's context cache, against 6.5% median for direct — the
  difference the exact-prefix revision prompts buy.
- **Identical-request memoization.** 36 reuses on the direct path (its
  `exclusion_challenge` stage repeats prompts; geopolitical alone reused 19, which
  would otherwise have been 47 calls instead of 28). Zero on semantic, which is the
  correct result: the semantic path does not re-issue byte-identical prompts.
- **Bounded parallelism.** Branch, structure and review/assessment concurrency ran on
  every case with deterministic artifact ordering; correctness is pinned by
  `test_parallel_equivalence.py`, not by these runs.

## Where the remaining time actually goes

Provider generation time as a share of wall clock, per semantic run (>100% means
calls overlapped, i.e. parallelism was working):

| case | wall | provider time | share | plan+delta calls |
| --- | --- | --- | --- | --- |
| holdout_space_operations | 1043 s | 894 s | 86% | 15 |
| committee | 939 s | 812 s | 86% | 22 |
| holdout_sports_governance | 681 s | 643 s | 94% | 11 |
| geopolitical | 518 s | 442 s | 85% | 8 |
| individual | 457 s | 404 s | 89% | 9 |
| population | 446 s | 431 s | 97% | 9 |
| holdout_science_institution | 273 s | 284 s | 104% | 8 |
| negotiation | 208 s | 208 s | 100% | 5 |

**85–104% of wall is the provider generating tokens.** Local computation, scheduling,
gates, lowering and the runtime are collectively negligible. The remaining lever is
therefore not efficiency of the code but the number of *serially dependent* model
calls, because a revision cannot start before the plan it revises exists.

The serial chain is driven by compile cycles, and every run paid 2–4 of them:

| case | compile cycles | reviewer verdicts |
| --- | --- | --- |
| geopolitical | 2 | REVISE, REVISE |
| population | 2 | APPROVE, REVISE |
| holdout_sports_governance | 3 | REVISE ×3 |
| holdout_space_operations | 4 | REVISE ×3 (+1) |

A second cycle is triggered by a *world gate* refusing after compilation, and each new
cycle re-runs the independent review. Inspecting the reviewer's findings across cycles
shows they are **genuinely different each time** — cycle 1 of geopolitical attacks the
already-occurred announcement, cycle 2 attacks `represents_count` and two duplicate
processes — so this is not re-litigation that could be suppressed, and the review is
doing real work. Shortening it would mean removing review, which is out of bounds.

## Forecast and world differences

- **geopolitical — semantic 0.00 vs direct 1.00.** The largest divergence in the set,
  and it is a *world* difference: the reviewer's first-cycle finding records that the
  evidence shows seven OPEC+ countries already decided a production increase on
  2026-07-24, and the semantic plan was revised away from treating the announcement as
  a future uncertainty. The two modes are answering from materially different causal
  worlds on the same store; this needs adjudicating against the record before either
  number is trusted. (The prior wave had both modes at 1.00 on a *different* store.)
- **holdout_sports_governance — semantic 0.70 vs direct 0.00.**
- **holdout_space_operations — semantic 0.75 (partially resolved) vs direct refused**
  at compilation after 2 calls: semantic represented a world direct could not build.
- **individual — semantic refused, direct 0.50.** Semantic's coverage gate refused
  with 3 of 17 material candidates "represented but not causally wired" and only 2
  included. This is a real semantic defect, not a fidelity win, and `repair_attempts:
  0` shows the coverage failure never reached a repair round.
- **population** — both 1.00, but semantic reached `resolved` with 6 actor-decision
  calls where direct returned `partially_resolved`.
- **committee / negotiation** — both modes refuse; agreement on the refusal, different
  named causes.

## After the benchmark: the largest measured avoidable time, fixed

`committee` set the p90. Its store holds zero claims (the Banxico pastcast found 40
candidate URLs, none with an archived capture at or before the May-2026 cutoff), and
the semantic path spent the entire 900-second compile-and-repair budget across 22
model calls discovering that nothing can cite nothing — then refused anyway. Direct
refused the same store in 68 seconds.

That is avoidable by construction: an empty admissible view cannot ground any world,
so every plan is uncitable and every repair round re-derives the same impossibility.
`compile_for_mode` now refuses immediately for both modes at zero model calls, and the
diagnosis names the archive gap rather than the compiler (commit `ce84aa8`,
regressions in `tests/unit/test_compiler_default.py`). This is the same refusal 900
seconds earlier with a truer cause — no fidelity is traded.

Projected effect on this set, holding everything else constant: p90 15.7 → 11.4 min,
median 8.1 → 7.6 min. **Still short of the 5-minute target**, and the projection is
arithmetic on the existing rows, not a re-measurement — the next A/B will measure it.

## Honest status

- The 1–5 minute consumer target is **not** reached and consumer readiness is **not**
  claimed.
- The remaining floor is *largely* external model-generation latency (85–104% of wall),
  but it is not yet *demonstrated* to be irreducible: the serial compile chain is 5–22
  calls where a clean run needs ~4, and the second compile cycle exists because world
  gates fire after compilation rather than during it. Moving the mechanical gate checks
  inside the compile cycle — so a gate defect is fixed by a delta in the same cycle
  instead of triggering a fresh cycle with a fresh review — is the next structural
  lever, and it preserves review, evidence, uncertainty and causal processes.
- Semantic's remaining functional defects on this set are the coverage-gate refusals
  (`individual`, `negotiation`) and the `repair_attempts: 0` path that lets a coverage
  failure refuse without a repair round. Semantic being the default means these are the
  failures to fix, not evidence that the default is wrong.
