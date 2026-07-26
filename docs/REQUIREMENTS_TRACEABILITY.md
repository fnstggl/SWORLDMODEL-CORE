# REQUIREMENTS TRACEABILITY

Stable IDs for every requirement cluster in the completion directive. Status values:
OPEN → IMPL (code+regression exists) → RUN-PROVEN (real production-path run exercised
it; artifact linked) → CLOSED (independent reviewer approved the artifacts). A row is
never CLOSED by its implementer. Columns: owner / reviewer / affected files /
completion test / proving artifact / commit.

Phase-0 baseline (2026-07-26, commit e17f189): 304 tests green, ruff+format clean,
mypy --strict clean (52 files); forensic verdicts RECONSTRUCTED ×3
(artifacts/forensics/*); A/B medians semantic ≈801s/23.5 calls, direct ≈386s/9.5.

## OBS — Observability (Phase 1)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| OBS-1 | Per-branch complete initial state persisted (post-scenario, pre-event) | obs-trace / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-2 | Every state diff persisted with trigger, authority, evidence lineage, terminal relevance (§12) | obs-trace / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-3 | Communications persisted with §11 fields (sender, recipient, channel, send/deliver, notice, interpretation) | obs-trace / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-4 | Process transitions persisted (§ process-by-process trace) | obs-trace / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-5 | Actor calls + model calls persisted exact and complete (prompts, raw responses, chronology) | done pre-run (71393bb, 189c88d) / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-6 | Replay viewer reconstructs the real run: timeline, actors, comms, processes, state, calls, branches, weights, lineage, cost, audit views; real artifacts only | obs-viewer / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-7 | Initial state + ordered diffs reconstruct the world with zero LLM calls (D7 replay core, shared by trace writer, publication gate, forensics) | obs-trace / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |
| OBS-8 | Complete chronological dossier (§13, 26 sections) emitted per real run | obs-trace / obs-reviewer | CLOSED (obs-reviewer APPROVE, 127f8a7) |

## ACT — Actor fidelity (Phase 2)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| ACT-1 | Persistent identity/role/authority/beliefs/memory/commitments/plan/revisit conditions per actor (§10) | runtime-impl / realism-adversary | OPEN (partial exists) |
| ACT-2 | Wake causes restricted to the §10 list; every wake records its cause | runtime-impl / realism-adversary | OPEN (partial exists) |
| ACT-3 | Prompt separates verified facts / visible state / memory / interpretation / private assumptions / feasible actions / unavailable info | runtime-impl / realism-adversary | OPEN (partial) |
| ACT-4 | Environment validates authority, feasibility, timing, resources, consistency; actor never narrates final reality | runtime-impl / realism-adversary | RUN-PROVEN (existing gates) |
| ACT-5 | Complete information→notice→interpretation→action lifecycle harness (production functions, mocked providers) | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |
| ACT-6 | Multi-actor communication chain harness: send→deliver→notice→interpret→respond→consequence | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |
| ACT-7 | Rejected/failed action followed by actor reconsideration harness | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |
| ACT-8 | No repeated actor calls without materially new information (measured; flagged) | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |

## TMP — Temporal fidelity (Phase 2)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| TMP-1 | Time from verified dates, grounded delays, schedules, durations, dependencies, deadlines — no arbitrary rounds/microsteps | runtime-impl / realism-adversary | OPEN (partial) |
| TMP-2 | Per-run temporal report: all §9 counters (timestamps, jumps, zero-duration actions, wake novelty, comms, process updates, terminal checks) | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |
| TMP-3 | Realism adversary rejects: frozen-timestamp repeats, unexplained instant action/comms, jumps over material events, early termination, fixed turn counts | realism-adversary / CTO | OPEN |
| TMP-4 | Known scheduled material events fire inside the window (BoE t0-collapse class fixed) | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |

## COM — Communication fidelity (Phase 2)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| COM-1 | Sending ≠ delivery ≠ notice ≠ agreement ≠ execution preserved and visible | runtime-impl / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |
| COM-2 | Every communication carries §11 fields end-to-end into artifacts | obs-trace / realism-adversary | CLOSED (realism-adversary APPROVE, 89e04c0) |

## CWF — Causal-world fidelity (Phase 3)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| CWF-1 | Representation-scale record: per included entity why/what-state/what-info/what-authority/what-if-removed; per excluded candidate why exclusion is immaterial (§7) | world-compiler-impl / causal-adversary | OPEN |
| CWF-2 | Operational aggregates modeled as multi-stage systems (demand/production/logistics-class causal questions as PLANNING questions, never runtime types) (§8) | world-compiler-impl / causal-adversary | OPEN |
| CWF-3 | D4: one-step operational worlds refused (single set_field / arbitrary multiplier) | world-compiler-impl / causal-adversary | OPEN |
| CWF-4 | D3 static gate: THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS refused at semantic validation | world-compiler-impl / causal-adversary | OPEN |
| CWF-5 | D5: zero-actor admissibility rule; decorative actors rejected; necessary actors cannot be omitted | world-compiler-impl / causal-adversary | OPEN |
| CWF-6 | Reviewer rejects: one-step terminal, ungrounded multiplier decides result, no intermediate state, skipped causal period (§ immediate-6) | world-compiler-impl / causal-adversary | OPEN |
| CWF-7 | Resolution-rule ambiguity surfaced and adjudicated (OPEC+ scope class) instead of silently resolved | world-compiler-impl / retrieval-adversary | OPEN |
| CWF-8 | Cross-domain regressions: straddling rejected; ungrounded equal branches ≠ calibrated point; genuine operational process runs with/without actors; decorative actors rejected; necessary actors required; one-step reporting ≠ production | world-compiler-impl+forecast-integrity / CTO | OPEN |

## FI — Forecast integrity (Phase 4)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| FI-1 | D1 validity triple in every forecast artifact | forecast-integrity / CTO+forensics | OPEN |
| FI-2 | D2 headline suppression: ungrounded disagreeing weights → no point estimate; bounds + reason published; average in diagnostics only | forecast-integrity / forensic auditor | OPEN |
| FI-3 | D6 responsibility classification (8 values) computed in-run as publication gate; only ACTOR_/PROCESS_/ACTOR_AND_PROCESS_CAUSED/FACTUALLY_RESOLVED publish | forecast-integrity / forensic auditor | OPEN |
| FI-4 | Mandatory deletion counterfactuals + weight sensitivity + ungrounded-numeric perturbation before publication (§14) | forecast-integrity / forensic auditor | OPEN |
| FI-5 | D3 aggregation backstop flag | forecast-integrity / forensic auditor | OPEN |
| FI-6 | §15 weight rules: grounded-weight evidence classes; equal ≠ probability; no invented threshold-adjacent alternatives | forecast-integrity+world-compiler-impl / causal-adversary | OPEN |
| FI-7 | Before-vs-after simulation comparison + terminal lineage persisted (done 189c88d) and consumed by gates | forecast-integrity / forensic auditor | IMPL |
| FI-8 | Prior results reclassified per D9 in all docs/PR | CTO / release-manager | OPEN |

## REL — Reliability & repeatability (Phase 5)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| REL-1 | Chaos matrix: interruption after each of the 13 stages preserves work + diagnosis | chaos-adversary / reliability-impl | OPEN |
| REL-2 | Malformed responses, timeouts, disk failures, stale artifacts, mismatched commit/question/evidence/mode all diagnosed | chaos-adversary / reliability-impl | OPEN (partial) |
| REL-3 | Same-commit ×2 repeatability per final case; structural disagreement surfaced, not averaged | forecast-integrity / release-manager | OPEN |
| REL-4 | Smallest fidelity-preserving stabilization (normalization / targeted consensus / reviewer arbitration) | world-compiler-impl / CTO | OPEN |

## EVAL — Sealed evaluation (Phases 6–7)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| EVAL-1 | Actor-fidelity suite preregistered: 8 pastcasts meeting §6 composition + 1–2 operational controls; frozen wording/rule/cutoff/horizon/store/scoring | CTO+forecast-integrity / user veto window | OPEN |
| EVAL-2 | D8 sealing: content hashes for all frozen inputs; outcomes as salted commitments; plaintext only in artifacts/sealed/, forecast-integrity-only, post-seal | forecast-integrity / release-manager | OPEN |
| EVAL-3 | Leakage rule enforced: any post-cutoff access invalidates the case; invalidated cases never count | forecast-integrity / release-manager | OPEN |
| EVAL-4 | Existing 5 acceptance cases kept as regressions but not counted as actor-simulation evidence | release-manager / CTO | OPEN |

## EFF — Efficiency (Phase 6)

| ID | Requirement | Owner / Reviewer | Status |
| --- | --- | --- | --- |
| EFF-1 | Delta plan repairs + deterministic patch merging | world-compiler-impl / CTO | OPEN |
| EFF-2 | Caching: prompt-prefix, source text, extracted claims, query/URL dedup | retrieval-impl / retrieval-adversary | OPEN (partial) |
| EFF-3 | Bounded parallel branches + parallel independent reviews; actor-context reuse only on materially identical context | runtime-impl / realism-adversary | OPEN |
| EFF-4 | Stage-level wall/calls/tokens/cache/cost measured; consumer 1–5 min median or proven external floor; explicit deep-mode | release-manager / CTO | OPEN |

## RG — Release gates (§20, Phase 7)

RG-1 architecture single-route · RG-2 causal-world fidelity · RG-3 actor fidelity ·
RG-4 temporal fidelity · RG-5 forecast integrity · RG-6 reliability ·
RG-7 observability · RG-8 evaluation · RG-9 performance · RG-10 product surface.
Each MET only on real artifacts, judged by release-manager, who may return NOT READY.

## DEL — Deliverables

DEL-1..28 map 1:1 to §21 of the directive; tracked in DEFECT_REGISTER as they land.
Final report answers the 30 questions of §22. PR #6 updated; never merged.
