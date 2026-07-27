# Optimization benchmark — frozen eight-case A/B, two waves

Two full sixteen-run waves over the *same* frozen evidence stores, each at one
unchanged commit:

| wave | commit | what it measures |
| --- | --- | --- |
| 1 | `2a489a3` | the optimizations as first landed |
| 2 | `3606b20` | wave 1 plus the two fixes wave 1's profiling identified |

Five acceptance cases plus three holdouts, both compiler modes, each pair reading the
identical store (`artifacts/stores/<case>/evidence_store.json`, one live retrieval pass
per case via `scripts/freeze_stores.py`). Per-run rows:
`artifacts_benchmark/ab_results.jsonl` (wave 2) — regenerate with
`python3 scripts/ab_report.py`.

Equivalence was proven **before** either wave ran: delta-merge byte-equivalence,
serial-versus-parallel replay identity, memo in-flight dedup, cache replay
re-verification (`tests/unit/test_plan_delta.py`,
`tests/unit/test_semantic_delta_compile.py`,
`tests/invariants/test_parallel_equivalence.py`, `tests/unit/test_runcache.py`).

## Headline — wave 1 → wave 2

| metric | semantic w1 | semantic w2 | direct w1 | direct w2 |
| --- | --- | --- | --- | --- |
| wall median | 8.1 min | **6.7 min** | 3.7 min | 3.5 min |
| wall p90 | 15.7 min | **13.9 min** | 7.0 min | 5.6 min |
| calls median | 20 | **14** | 9 | 7 |
| tokens in (8 runs) | 643,779 | 539,140 | 260,402 | 194,909 |
| tokens out (8 runs) | 427,074 | 356,622 | 263,226 | 190,641 |
| prompt-cache hit rate | 35.1% | 39.8% | 19.1% | 37.4% |
| **cases producing a number** | 5 / 8 | **7 / 8** | 5 / 8 | 4 / 8 |

**The 1–5 minute consumer target is NOT reached. Median supported semantic runtime is
6.7 minutes. Consumer readiness is not claimed.**

**Cost** is reported as token totals. No provider list price is asserted anywhere;
`scripts/ab_report.py` prices a wave only when `DEEPSEEK_PRICE_IN` /
`DEEPSEEK_PRICE_OUT` / `DEEPSEEK_PRICE_CACHED` are supplied, so no invented number
enters the record. Semantic's cached share (39.8% of input) bills at the cache-hit rate.

### A confound, stated plainly

Wave 2 ran after wave 1 against the same stores and questions, so DeepSeek's
context cache was **warmer**. Direct's hit rate nearly doubled (19.1% → 37.4%) on
code paths this branch did not touch, and its wall improved too. So wave-2 wall-clock
gains are **partly the fixes and partly cache warmth plus ordinary run-to-run
variance**. The trustworthy signals are the structural ones — call counts, compile
cycles, and outcomes — not wall time alone.

## Per-case, wave 1 → wave 2 (semantic)

| case | wall | plan+delta calls | status |
| --- | --- | --- | --- |
| committee | 939 s → **0 s** | 22 → **0** | refused → refused (instantly) |
| geopolitical | 518 → 436 s | 8 → **4** | resolved 0.00 → resolved 1.00 |
| individual | 457 → 833 s | 9 → 19 | **refused → partially_resolved 0.50** |
| negotiation | 208 → **125 s** | 5 → **3** | **refused → resolved** |
| population | 447 → 548 s | 9 → **6** | resolved → partially_resolved |
| holdout_science_institution | 273 → 365 s | 8 → 7 | resolved 1.00 → resolved 1.00 |
| holdout_space_operations | 1043 → 1233 s | 15 → 17 | partially_resolved (both) |
| holdout_sports_governance | 681 → 245 s | 11 → **4** | resolved (both) |

`plan+delta` — the serial compile chain — fell in six of eight cases, median 8 → 6.
Two cases rose, and both are explicable: `individual` now *completes* where it
previously refused (finishing costs more calls than refusing), and
`holdout_space_operations` remains the hardest world in the set.

## Did the optimizations do what they were built to do?

- **Delta repair.** `semantic_plan` median **1** against `semantic_plan_delta` median
  5 in wave 2 — exactly one full plan generation per compile cycle across all sixteen
  runs of both waves. No local validator or reviewer defect ever regenerated a plan.
- **Prompt-prefix reuse.** 39.8% of semantic input tokens served from the provider's
  context cache (see the warmth confound above).
- **Identical-request memoization.** 36 reuses in wave 1 on the direct path
  (geopolitical alone reused 19, which would otherwise have been 47 calls rather than
  28). Zero on semantic — correct, since the semantic path does not re-issue
  byte-identical prompts.
- **Bounded parallelism.** Ran on every case with deterministic ordering; correctness
  is pinned by `test_parallel_equivalence.py`, not by these runs. The >100% provider
  shares below are the direct evidence that calls genuinely overlapped.

## The two fixes wave 1's profiling produced

1. **Empty admissible view refuses at once** (`ce84aa8`). The Banxico pastcast store
   holds zero claims (40 candidate URLs, none with an archived capture at or before
   the cutoff). Wave 1 spent the entire 900-second compile budget over 22 model calls
   discovering that nothing can cite nothing, then refused anyway — it set the p90.
   **Measured in wave 2: 939 s / 22 calls → 0.0 s / 0 calls**, both modes, refusing
   with the archive gap named rather than the compiler blamed. The same refusal, 900
   seconds earlier.
2. **Inert states refused by the free validator** (`3606b20`). A state nothing reads,
   writes or draws previously survived the plan validator and was refused later by the
   *coverage* gate — a refusal that costs a whole extra compile cycle. Now caught
   mechanically at zero model cost. **Measured: two former coverage-gate refusals
   (`individual`, `negotiation`) now produce numbers**, and the compile chain shortened
   in six of eight cases. Deliberately confined to states: the same rule over entities
   refused three invented-domain regression worlds, a stricter standard than the gate
   it anticipates.

## Where the remaining time goes

Provider generation as a share of wall, wave 2 (>100% means calls overlapped):

| case | wall | provider | share | plan+delta |
| --- | --- | --- | --- | --- |
| holdout_space_operations | 1233 s | 1041 s | 84% | 17 |
| individual | 832 s | 734 s | 88% | 19 |
| population | 548 s | 559 s | 102% | 6 |
| geopolitical | 436 s | 310 s | 71% | 4 |
| holdout_science_institution | 365 s | 313 s | 86% | 7 |
| holdout_sports_governance | 245 s | 249 s | 102% | 4 |
| negotiation | 125 s | 130 s | 104% | 3 |

Median provider share **88%** (range 71–104%). Local computation — gates, lowering,
scheduling, the runtime, aggregation, both auditors — is a single-digit percentage of
wall in every run.

## Honest status

- **Target not reached.** Median 6.7 min against a 1–5 min target. Consumer readiness
  is **not** claimed.
- **The floor is not yet demonstrated irreducible.** Provider generation is 88% of
  wall at the median, so the remaining lever is the count of *serially dependent*
  calls, not code efficiency. The compile chain is still 3–19 calls where a clean run
  needs about 4, because a world gate that refuses *after* compilation forces an
  entire new cycle including a fresh independent review. Moving the mechanical gate
  checks inside the compile cycle — so a gate defect is repaired by an in-cycle delta —
  is the next structural lever, and it preserves review, evidence, uncertainty and
  causal processes. The two fixes here are instances of exactly that principle
  (catch it where it is free, not where it is expensive), and both worked.
- **Fidelity moved the right way**: semantic went from 5/8 to 7/8 cases producing a
  number, with no gate, review, actor or audit weakened — the improvement came from
  refusing bad *plans* earlier, not from accepting bad worlds.
- **Remaining semantic defects**: `holdout_space_operations` still takes 17 compile
  calls and 20 minutes; the `repair_attempts: 0` path lets a coverage failure refuse
  without a repair round; and the geopolitical world difference (wave 1 semantic 0.00
  vs direct 1.00; wave 2 semantic 1.00) shows plan-level nondeterminism that deserves
  its own investigation.
