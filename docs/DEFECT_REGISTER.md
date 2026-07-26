# DEFECT REGISTER — ranked launch blockers

Every defect: observed failure → earliest incorrect pipeline stage → one owner → one
independent reviewer → smallest reproduction → universal fix → regression → real-run
proof → adversary sign-off. A builder never closes their own defect. Severity is
CRITICAL (invalidates answers or honesty), HIGH (materially wrong world/trace),
MEDIUM, LOW.

## Open

| ID | Sev | Defect (observed, with artifact) | Earliest stage | Owner / Reviewer | Req |
| --- | --- | --- | --- | --- | --- |
| FD-17 | CRITICAL | Live run artifacts/phase2/geopolitical2 (commit bc20aa0) published `probability: 1.0000` labeled `weighted_simulated_trajectories` with only 0.25 of mass resolved (1 branch YES, 3 UNRESOLVED, 0.75 unresolved). The point estimate is conditional on a MINORITY of branch mass over ungrounded symmetric-ignorance weights; bounds [0,1] and status partially_resolved are honest but the headline number is not. FI-2's "ungrounded weights that DISAGREE" test does not fire when the others are unresolved rather than opposed. | aggregation/publication | forecast-integrity / forensic auditor | FI-2, FI-3 |
| FD-18 | HIGH | Same run: 8 of 12 actor wake-ups flagged repeated_without_new_information (67%) — the ACT-8 measurement now works and is reporting a real behavioral defect: actors re-woken with nothing new to read. | scheduling/wake rules | runtime-impl / realism-adversary | ACT-8, TMP-3 |
| FD-1 | CRITICAL | Tesla world = cited Q2 × invented factor, zero actors, one set_field; published 0.50 was branch construction (artifacts/forensics/population) | semantic planning | world-compiler-impl / causal-adversary | CWF-2,3 |
| FD-2 | CRITICAL | Ungrounded disagreeing weights published their average as headline (BoE 0.25, Tesla 0.50) | aggregation/publication | forecast-integrity / forensic auditor | FI-2 |
| FD-3 | CRITICAL | Threshold-straddling invented alternatives (0.8/1.05 vs break-even 0.8331) undetected end-to-end | semantic validation | world-compiler-impl / causal-adversary | CWF-4, FI-5 |
| FD-4 | CRITICAL | Responsibility classification exists only in the offline forensic script; publication is ungated (a BRANCH_WEIGHTS_DOMINATED run published normally) | forecast assembly | forecast-integrity / forensic auditor | FI-3,4 |
| FD-5 (CLOSED 127f8a7, obs-reviewer) | HIGH | No per-branch initial state, state diffs, communications, or process transitions in the production trace (forensics had to derive them) | trace write | obs-trace / obs-reviewer | OBS-1..4,7 |
| FD-6 (CLOSED 127f8a7, obs-reviewer) | HIGH | Replay viewer does not render the forensic/observability artifact set | viewer | obs-viewer / obs-reviewer | OBS-6 |
| FD-7 | HIGH | BoE branch: `release_at: null` collapsed all data releases to t0; outcome decided at first invocation; 10/14 invocations causally inert; actor re-signaled 3× (run's own audit: HIGH) | semantic planning + scheduling | runtime-impl + world-compiler-impl / realism-adversary | TMP-4, ACT-8 |
| FD-8 | HIGH | OPEC+ resolution-scope ambiguity (group-wide quota vs sub-group unwinding) silently resolved; store contained both readings; published 1.00 with [1,1] bounds | resolution contract | world-compiler-impl / retrieval-adversary | CWF-7 |
| FD-9 | HIGH | Zero-actor worlds admissible without evidence that no human decision matters (Tesla passed compile with no actors) | compile gates | world-compiler-impl / causal-adversary | CWF-5 |
| FD-10 | HIGH | Uncertainty alternatives carry no meaning/sensitivity/structure-vs-state fields; planner may invent numeric alternatives around thresholds (§15 unimplemented) | semantic schema | world-compiler-impl / forecast-integrity | FI-6 |
| FD-11 | MEDIUM | `"other"`-style degenerate filler alternative carried the entire YES mass in BoE (grounding: "No specific alternative in evidence") | semantic validation | world-compiler-impl / causal-adversary | FI-6 |
| FD-12 | MEDIUM | Same-commit plan variance unmeasured/unstabilized (population: 0.50 world vs unresolved world on identical inputs) | planning determinism | world-compiler-impl / forecast-integrity | REL-3,4 |
| FD-13 | MEDIUM | Temporal metrics (§9) not computed or reported on any run | runtime reporting | runtime-impl / realism-adversary | TMP-2 |
| FD-14 | MEDIUM | branch_schedule "events" counts in-loop events only; artifact counters unreconciled with ledger (uniform +2/branch) | diagnostics | reliability-impl / chaos-adversary | REL-2 |
| FD-15 | LOW | report.md boilerplate limitation ("deleting actor calls deletes the forecast") emitted for zero-actor runs | reporting | reliability-impl / release-manager | REL-2 |
| FD-16 | LOW | branch-id `primary/` prefix inconsistency needs a joined-view helper for readers (namespacing itself fixed in b2c10ed) | trace format | obs-trace / obs-reviewer | OBS-7 |

## Closed this run (verification: 304-test suite + forensic re-run at e17f189)

| ID | Was | Fixed | Proof |
| --- | --- | --- | --- |
| FC-1 | Ledger not namespaced; two mechanical audit checks structurally vacuous, false PASS published | b2c10ed | regression + forensics re-run |
| FC-2 | Citations validated on entities only; transposed-id claim grounded a NO branch; coverage said complete | b2c10ed | fabricated-citation regression |
| FC-3 | Single-variable-only initialization check missed conjunctions (BoE) | 189c88d | conjunction regression |
| FC-4 | ForecastIntegrity computed but persisted nowhere | 189c88d | forecast.json integrity block |
| FC-5 | Call/token counters snapshotted before audit call; sealed forecast disagreed with own log | 189c88d | counter regression |
| FC-6 | Factual-resolution runs labeled weighted_simulated_trajectories | 71393bb | source-label regression |
| FC-7 | No executable world persisted; replay required an LLM | 71393bb | compiled_world.json |
| FC-8 | llm_calls carried hash-only prompts, no chronology | 71393bb | stamped prompts/timestamps |
| FC-9 | Report line crashed completed run's trace write (intent={}) | 19f2a67 | totality regression |
