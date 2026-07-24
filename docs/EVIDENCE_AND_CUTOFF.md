# Evidence and cutoff

More detailed simulation is harmful when the underlying information is false.
Relevant, verified information is one of the few proven ways to improve simulation
quality — so evidence is a first-class, persistent structure, and the information
cutoff is enforced mechanically.

## The canonical evidence store

`EvidenceStore` holds the *complete* set of `EvidenceClaim`s. It is never reduced to
an arbitrary fixed character count. Each claim retains: id, proposition, normalized
value, entities, valid/publication/availability times, source id/url/title/type,
authority level, supporting excerpt, lineage event id, epistemic type
(observation/inference/hypothesis), confidence, retrieval time, and contradiction ids.

Token-limited **views** are derived on demand by relevance (`render_view`), and each
view still renders full claims with their ids. The store is never mutated to build a
view (`test_token_limited_view_never_deletes_the_canonical_store`).

## Cutoff enforcement (mechanical, not prompt wording)

Downstream code only ever receives an `EvidenceView` bound to `as_of`. A claim whose
`available_at` is after `as_of` is unreachable through that view: `available()`
excludes it and `get()` raises `CutoffViolationError`. The claim still lives in the
store for later, sealed, post-outcome analysis.

In the Banxico corpus the June 25 decision claim (`available_at = 2026-06-25`) is
present in the store but excluded from every pre-outcome view
(`test_post_cutoff_evidence_is_not_available_to_the_simulation`).

Note: the *simulated* future data a branch delivers (an uncertain inflation surprise,
say) is **not** post-cutoff evidence — it is a branch hypothesis produced by the
uncertainty model, carried as an `epistemic_type: hypothesis` event, never a fact
published after `as_of`.

## Lineage de-duplication

Claims derived from the same underlying event share a `lineage_event_id`. One meeting
may yield several qualitative implications, but it is one event, not many independent
statistical cases. `check_lineage_independence` returns
`(claim_count, independent_event_count)`; callers must use the *event* count.

The Banxico May-7 decision yields seven claims (the decision, five individual votes,
the guidance) that all share `banxico_2026_05_07_decision` — 17 total claims collapse
to 6 independent events (`test_lineage_deduplicates_one_event_into_shared_id`).

## Contradictions

Detection is deliberately *precise*: explicit corpus-declared pairs plus exact
`claim_key` matches. Fuzzy topic/prefix heuristics are avoided because they cause
false refusals (e.g. two officers sharing an "office:" namespace). A decisive
contradiction among available load-bearing claims blocks rollout with `EvidenceError`
(`test_contradictory_decisive_claims_block_rollout`).

## Facts, inferences, hypotheses

Every world fact cites evidence claim ids or is explicitly marked non-observation
(`test_every_world_fact_has_evidence_or_is_marked_non_observation`). The compiler may
generate inferences and hypotheses, but never presents them as observations.
