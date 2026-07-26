# Efficiency report — measured, semantic route

Measurements first; every number below is from a real run's own record
(`llm_calls.jsonl`, run summaries). Frozen-store replays of the acceptance stores,
DeepSeek `deepseek-v4-flash`, one process per run. Updated 2026-07-26 during the
launch-candidate slice wave; the final table will be regenerated from the 8-case A/B
at the unchanged launch candidate.

## Per-run totals (completed slice-wave runs)

| case | outcome | wall | provider calls | tokens in/out | actor calls | plan calls |
| --- | --- | --- | --- | --- | --- | --- |
| individual (BoE) | COMPLETED 0.25, genuine actor simulation, 4 branches | 540 s | 26 | 97,929 / 68,954 | 14 | 7 |
| geopolitical (OPEC+) | COMPLETED 1.00, factual resolution | 426 s | 12 | 62,939 / 53,829 | 0 | 6 |

## Per-stage breakdown

individual (26 calls):

| stage | calls | tokens in | tokens out |
| --- | --- | --- | --- |
| semantic_plan | 7 | 43,692 | 47,285 |
| actor_decision | 14 | 36,574 | 9,652 |
| semantic_review | 2 | 8,719 | 4,416 |
| trajectory_audit | 1 | 5,017 | 3,007 |
| world_review | 1 | 2,382 | 3,620 |
| assess_structure | 1 | 1,545 | 974 |

geopolitical (12 calls):

| stage | calls | tokens in | tokens out |
| --- | --- | --- | --- |
| semantic_plan | 6 | 46,022 | 42,207 |
| semantic_review | 2 | 10,326 | 4,910 |
| world_review | 1 | 3,373 | 4,155 |
| trajectory_audit | 1 | 666 | 1,240 |
| assess_structure | 1 | 2,066 | 890 |
| exclusion_challenge | 1 | 486 | 427 |

## The three largest time costs

1. **Semantic plan rounds.** 6–7 `semantic_plan` calls per run, each regenerating the
   COMPLETE plan (~7k output tokens/call). Output-token generation dominates wall
   time; plan rounds alone account for the majority of both runs' walls. The rounds
   are: initial plan, up to two validator-error revisions, then the same trio again
   when the pre-rollout review forces a recompile.
2. **Actor decisions.** 14 calls on the branching run (4 scenario branches × 3–4
   moments). Serial today; branches are independent by construction.
3. **Reviews and audits.** semantic_review + world_review + trajectory_audit +
   assess_structure ≈ 5 calls/run. Individually small; together a fixed ~1–2 minute
   tail. (The trajectory-audit call on factual resolutions is already removed —
   commit 2c2a654 — which deletes one call from every settled-record run.)

## The three largest LLM-call costs

Same ranking by tokens: plan rounds (~90k combined tokens/run), actor decisions
(~46k on the branching run), the review tail (~20k).

## Duplicated work

- Every plan revision re-emits the full plan for a validator error that names one
  defect. A revision that returns only the corrected sections, merged
  deterministically, would cut most plan output tokens. (Contract change; measured
  candidate, not implemented.)
- The review-forced recompile repeats the full plan+validator trio from scratch with
  one instruction line added. Same delta opportunity.

## Cacheable work

- The plan prompt's fixed prefix (schema + universal rules + consistency
  requirements) is identical across all rounds and all questions; provider-side
  context caching would price 6–7 rounds close to 1. Infra-level; no fidelity
  impact.

## Parallelizable work

- Scenario branches are independent: the 14 actor calls on the individual run can
  run per-branch concurrently (bounded by provider rate limits), cutting the
  simulation phase roughly by the branch count.
- world_review and assess_structure read the same compiled world and are
  independent of each other.

## Skippable work (fidelity-preserving only)

- Trajectory-audit model pass on factual resolutions — already skipped (2c2a654).
- Nothing else is skippable without weakening research, review, gates, actor
  invocations, or audits; none of those are candidates.

## Minimum call plan preserving exact reality fidelity

For a completing genuine-simulation run: 1 plan + ≤2 revisions + 1 semantic review
+ 1 world review + 1 structure assessment + (branches × moments) actor calls +
1 trajectory audit. For a settled-record run: 1 plan + ≤2 revisions + 1 semantic
review + 1 world review + 1 structure assessment, and zero actor/audit-model calls.
The measured runs are already near these floors per call-count; the gap to the 1–5
minute consumer target is output-token volume in plan rounds (delta revisions +
prefix caching) and serial branches (parallel actor batches), not extra calls.

## Target

Consumer mode 1–5 minutes median: current measured walls are 7.1–9.0 minutes.
The two structural levers above (delta plan revisions, parallel branches) are the
path; both preserve the exact research, gate, and audit behavior. A deeper mode
(more branches, more structures) remains explicitly selected, not default.
