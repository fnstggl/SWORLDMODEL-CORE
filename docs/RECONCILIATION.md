# Reconciling PR #4 into the consolidation branch

Two branches descended from the merged base `claude/sworldmodel-core-rebuild-lcfpq1`:

* `claude/sworldmodel-consolidation-hyhovj` — this branch, where the live diagnoses and
  fixes continued;
* `codex/complete-the-sworldmodel-consolidation-run` (PR #4) — a single squashed commit,
  `f30e449`.

PR #4 turned out to be a **snapshot of this branch's own work** at an earlier point: it
contains `epistemics.py`, `jsonsalvage.py`, `world_review.py`, `repair.py`,
`diagnosis.py`, `scripts/acceptance.sh`, `scripts/acceptance_report.py` and
`docs/EPISTEMIC_POLICY.md`, all created here. So the reconciliation was not two rival
architectures but a diff against a fork point, and this branch is ahead almost
everywhere — `world_compiler` by 321 lines, `source_extract` by 88, `expressions` by 61,
`repair` by 77.

Five things in PR #4 were genuinely unique and better. All five were taken. Nothing was
merged wholesale in either direction, and no duplicate implementation was kept.

## Reconciliation table

| capability | this branch | PR #4 | relation | final | reason |
| --- | --- | --- | --- | --- | --- |
| undetermined expression values | crashed the branch | `UndeterminedExpressionError`, caught at the terminal and at preconditions | **PR #4 only** | **PR #4** | An undetermined value is an honest *unresolved* branch, never a crash and never a rounded-down NO. Complements this branch's compile-time operator check: that catches operators the evaluator lacks, this catches values the world never set. |
| fourth epistemic class in the core enum | `EpistemicClass.UNSUPPORTED` in `epistemics` only | also `EpistemicType.UNSUPPORTED` in `models` | **PR #4 more complete** | **PR #4** | The four classes must stay distinct in the evidence store too, not only in the grounding layer. |
| research planning prompt | "outcome ← terminal decision/action ← actor choices" | "outcome ← terminal-producing events ← causal producers… may be people, organizations, populations, markets, production/logistics systems… Do not assume a person decides an aggregate quantity" | **PR #4 better** | **PR #4** | Research must look for the real producers from the first query, not just for people. Exactly the Tesla failure, one stage earlier. |
| uncitable required reality facts | none — a fact the model named but could not cite refused the run | dropped in `_normalize_compilation`; cited facts still gate | **PR #4 only** | **PR #4** | A live Federal Reserve run died on `fomc_has_authority: no evidence attached`. Refusing for the compiler's own rhetoric is unnecessary refusal. A fact that *does* cite evidence is untouched, so an unavailable citation still refuses. |
| `test_live_normalization.py` | absent | present | **PR #4 only** | **PR #4** | Pins the regression above. |
| retrieval-mode reporting | `RetrievalMode` decided once at run start, pinned, printed, recorded in the trace | recomputed `datetime.now()` in the CLI and printed | conflicting | **this branch** | Deciding per call is the bug: a run begun as a nowcast flips to a pastcast when the clock passes its own cutoff. Pinning the instant removes it and the mode reaches the diagnosis. |
| page-content filter | `nav/footer/aside/noscript/svg/select/button` only, plus fallback to unfiltered text | also `form`, `header`, and role/class patterns closing on `</[a-z]+>` | conflicting | **this branch** | Measured: `form` under DOTALL extracts an ASP.NET page to the empty string; `header` inside `article` deletes the dateline the verifier needs; the class patterns let a cookie wrapper swallow the document. |
| main-content hoist | moves the region (`body.replace(main, " ", 1)`), `<main>`/`<article>` only | copies it, and includes role/class regions | conflicting | **this branch** | Copying fills more than half the bounded extraction window with the same words twice; the role/class regions select one paragraph. |
| appositive name split | splits on comma only when the head is already a full name | always splits on the comma | conflicting | **this branch** | "Smith, John" is one name written backwards; truncating it to "Smith" loses the person. |
| research trace across repair | carried through a compiler-only recompile | dropped | conflicting | **this branch** | Without it a completed run reported "0 queries, 0 sources fetched, 0 claims" beside an audit showing 38 extractions and 399 HTTP requests. |
| producer-lineage gate | terminal terms namespaced by kind; refuses a terminal that reads nothing; refuses an uncertainty that writes a terminal term even when an action writes it too; refuses environment presets; accepts actors reaching the outcome through a producer's gate | earlier form: fields and collections only, orphan check only | this branch later and stricter | **this branch** | Each addition came from a live failure: `const(true)` passing vacuously, and a Bank of England run reporting 1.0000 because an uncertainty named `bailey_choice_to_signal` wrote the terminal field. |
| no-progress detection | state digest + clock + information digest, and requires the same-instant queue to have stopped draining | state digest + clock | this branch later | **this branch** | Bounded correspondence must reach its own scheduled session; a same-instant spin must still stop. |
| parser shape tolerance | 16 drift shapes coerced, pinned by tests | subset | this branch later | **this branch** | Each shape was a live run that died in a parser. |
| contradiction handling | future-vs-fact distinguished; decisive contradictions repairable before fatal | blocks on any decisive contradiction | this branch later | **this branch** | OPEC+ was refused over a report of a decision versus a projection about what follows it. |
| everything else | — | identical | identical | either | Same code; PR #4 is a snapshot of it. |

## One implementation per capability

Verified by search after reconciliation:

| capability | module |
| --- | --- |
| forecast entry point | `api.run_forecast` |
| live research backend | `live_research.LiveResearchBackend` (`research.ResearchBackend` is a Protocol, not a second implementation) |
| evidence store | `evidence.EvidenceStore` |
| repair system | `repair.plan_repair` + `api._compile_with_repair` |
| world compiler | `world_compiler.compile_world` |
| actor grounding | `grounding.assess_actor_grounding` |
| producer-lineage gate | `world_compiler.enforce_outcome_is_produced` |
| event-driven engine | `engine.run` |
| terminal evaluator | `engine.evaluate_terminal` |
| tracing and diagnosis | `tracing.TraceContext`, `diagnosis.RunDiagnosis` |

No module matching a legacy, V2, alternate-runtime, mechanism-family or scenario-family
pattern exists in `src/sworldmodel/`; no `corpus.json` path and no scripted or
deterministic gateway is reachable from production. Nothing was kept "for
compatibility", and no third combined architecture was created.
