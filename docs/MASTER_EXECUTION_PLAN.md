# MASTER EXECUTION PLAN — reality-fidelity completion run

Mission: make the question-only path reliably construct the right causal world,
simulate it at the appropriate level of detail, and let a reviewer inspect every
material step that produced the answer. Honest readiness only. Nothing may improve
completion, speed, or scores by making the simulated world less real.

Governing documents: this plan, `DEFECT_REGISTER.md`, `REQUIREMENTS_TRACEABILITY.md`,
`CONTINUATION.md` (session recovery). A requirement is complete only when the
production implementation exists, a focused regression passes, a real production-path
run exercises it, and its independent reviewer approves the actual artifacts.

## Organization (single implicit team; ≤4 write-enabled agents concurrent)

| Role | Agent name | Write? | Owns (files) |
| --- | --- | --- | --- |
| Chief Architect / CTO / integration lead | main session | yes (docs, integration, arbitration) | canonical branch; models.py arbitration; merges |
| Retrieval implementer | `retrieval-impl` | yes | live_research, research_planner, search, source_fetch, source_extract, http, rss, gnews_decode, providers, pdf_text, evidence |
| Retrieval adversarial reviewer | `retrieval-adversary` | no | — |
| Retrieval failure tester | `retrieval-chaos` | tests only | tests/unit/test_retrieval_failures* |
| Semantic-world compiler implementer | `world-compiler-impl` | yes | semantic_plan, semantic_compile, semantic_lowering, world_compiler, coverage, world_review, structures |
| Causal-world adversary | `causal-adversary` | no (blocking authority pre-simulation) | — |
| Runtime & actor implementer | `runtime-impl` | yes | engine, actors, executor, effects, world, memory, schedule, expressions, worldspec, temporal_report (new) |
| Behavioral/temporal realism adversary | `realism-adversary` | no | — |
| Reliability implementer | `obs-trace` / `reliability-impl` | yes | tracing, rundir, diagnosis, gateway, deepseek_gateway, repair, replaycore (new), scripts/ |
| Reliability & chaos adversary | `chaos-adversary` | tests only | tests/unit/test_chaos* |
| Forecast integrity & evaluation | `forecast-integrity` | yes | outcomes, uncertainty, trajectory_audit, responsibility (new), models.py forecast section; SOLE access to sealed outcomes |
| Integration engineer 1 (pipeline seams) | `integration-1` | yes | api.py, research.py, cli.py |
| Integration engineer 2 (trace→viewer) | `obs-viewer` / `integration-2` | yes | viz/ |
| Release manager | `release-manager` | no (reports) | docs/RELEASE_READINESS report |

Working agreements: one implementation owner + one independent reviewer per defect;
a builder may not close their own defect; adversaries never edit production code;
no two agents rewrite the same subsystem; implementers work in isolated worktrees on
`fix/*` branches; the CTO merges into `claude/sworldmodel-consolidation-hyhovj` and
personally inspects real-run artifacts before closing anything.

## Standing architectural decisions (binding on all agents)

- **D1 Validity triple.** Every forecast separately reports `trace_reproducible`,
  `causal_simulation_valid`, `point_estimate_calibrated`. RECONSTRUCTED means only
  that arithmetic reproduces; it is never presented as trustworthy.
- **D2 Headline suppression.** When ungrounded branch weights disagree, the published
  answer is: point estimate unavailable; honest scenario bounds; the reason
  ("scenario probabilities are not grounded"). The scenario average remains visible
  inside diagnostics only. Applies universally, no question-family code.
- **D3 Threshold-straddling gate.** `THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS`:
  unsupported numeric alternatives that directly occupy opposite sides of the
  resolution threshold must not produce a calibrated point forecast. Enforced twice:
  statically in the semantic validator (earliest stage, recompilable) and as an
  aggregation-time backstop.
- **D4 One-step operational worlds are rejected.** A terminal quantity produced only
  by one final set_field or one arbitrary multiplier is not an operational
  simulation. A single multiplier may stand in for a system only with a documented
  empirical model, grounded parameter uncertainty, reviewer approval, and no
  threshold-straddling sensitivity.
- **D5 Zero actors** are permitted only when evidence shows no material human or
  population decision can change the answer, the non-actor process is causally
  sufficient and detailed, and the causal-world adversary approves. Decorative
  actors are equally rejected.
- **D6 Responsibility classification is a publication gate.** Deterministic deletion
  counterfactuals + weight sensitivity run on every forecast before writing.
  Vocabulary: ACTOR_CAUSED, PROCESS_CAUSED, ACTOR_AND_PROCESS_CAUSED,
  FACTUALLY_RESOLVED, INITIAL_ASSUMPTIONS_DOMINATED, BRANCH_WEIGHTS_DOMINATED,
  UNRESOLVED, INVALID. Only the first four publish an answer; actor/process-caused
  results still need grounded weights for a point estimate.
- **D7 Single replay core.** The ledger-replay/state-diff/counterfactual machinery
  lives in `src/sworldmodel/replaycore.py`; the production trace writer, the
  publication gate, and `scripts/forensics.py` all consume it. Initial state plus
  ordered diffs must reconstruct every branch without an LLM call.
- **D8 Sealed pastcast evaluation.** Frozen question/rule/cutoff/horizon/store with
  content hashes. Implementation agents never see outcomes or post-cutoff sources.
  Outcomes live as salted SHA-256 commitments in the committed suite doc; plaintext
  only in `artifacts/sealed/` (gitignored, hash-committed), read only by
  `forecast-integrity` after the run directory is sealed. Honest caveat, recorded:
  model weights know famous past events; the mechanical seal is cutoff-enforced
  stores, universality invariants (no case-specific code), and preregistered scoring.
- **D10 A closing deadline is real news; the measurement adapts, not the behavior.**
  CTO ruling on FD-18(1). A node's deadline entry is cancelled only for a participant
  that already EXERCISED the opportunity (its intention was accepted into the world),
  never merely for one that answered and declined. Waking an actor because its window
  is closing is legitimate causal information — "now or never" is a real reason to
  reconsider — and three standing invariants (test_scheduling: a deadline wakes the
  participants, plan identity survives revision; test_no_coercion: a failed completion
  takes no effect) encode that contract deliberately. Those invariants are NOT
  retargeted. The honest fix for the noise the adversary measured is FD-18(2): the
  repeat counter splits calendar-driven wakes from information-driven ones, so
  "the clock moved" is never reported as "someone re-signaled and taught nothing".
  Rule of thumb this instantiates: when a measurement calls correct behavior a defect,
  fix the measurement — never bend the world to satisfy it.

- **D11 Suppression empties the field, not just the label.** CTO ruling, Phase 4. When
  D2 suppresses a point estimate, `ForecastResult.simulation_probability` is set to
  None — the field that IS the published headline. A label beside a retained number
  does not suppress it: the forensic audit found readers taking the headline while
  `probability_source` and [0,1] bounds sat next to it saying otherwise. The type
  already allows it (`float | None`, models.py:418) and the CLI already renders None.
  The figure survives as diagnostics-only `scenario_average` plus
  `integrity.probability_after_simulation`; a stale `point_estimate_is_calibrated:
  true` is as misleading as the number and must fall with it. Suppression fires when
  ungrounded weights DISAGREE **or** when resolved mass is a MINORITY of total branch
  mass (FD-17 proved the disagreement test alone cannot fire when the other branches
  are unresolved rather than opposed). Tests that pin the pre-D2 contract are
  retargeted, never deleted: each keeps proving its arithmetic against
  `scenario_average` and gains the suppression fact.

- **D9 Reclassified prior results.** Tesla 50% = invalid forecast
  (BRANCH_WEIGHTS_DOMINATED, one-step world). BoE 25% = valid causal pathway,
  invalid calibrated number. OPEC+ 100% = factual lookup with an unadjudicated
  wording ambiguity. None is presented as a successful simulation-generated
  forecast anywhere.

## Phases (strict order; no phase advance while a CRITICAL/HIGH defect is open in it)

- **PHASE 0 — Establish truth.** Verify branch/canonical path, forensic fixes, suite
  green, forensic verdicts, baseline metrics. Exit: baseline recorded in
  CONTINUATION.md.
- **PHASE 1 — Observability.** Persist complete per-branch initial state, every
  state diff, communications, process transitions (actor calls + model calls already
  full-fidelity). Replay viewer reconstructs the real run from real artifacts.
  Owners: `obs-trace`, `obs-viewer`. Exit: fresh real run re-verified by a read-only
  observability reviewer; viewer renders it.
- **PHASE 2 — Actor & temporal vertical slices.** Production-path harnesses (mocked
  providers only): one complete information→notice→interpretation→action lifecycle;
  one multi-actor communication chain (send→deliver→notice→interpret→respond→
  consequence); one rejected/failed action followed by reconsideration; event-driven
  time with the §9 temporal report computed per run. Owner: `runtime-impl`;
  reviewer: `realism-adversary`.
- **PHASE 3 — Causal-world fidelity.** Representation-scale record (why included /
  what-if-removed per entity; why-excluded per candidate); operational worlds with
  real intermediate progression; D3 static gate; D4 one-step rejection; D5
  zero-actor rule; strengthened reviewer; resolution-rule ambiguity surfacing
  (OPEC+ class). Owner: `world-compiler-impl`; reviewer: `causal-adversary`.
- **PHASE 4 — Forecast integrity.** D1/D2/D6 in production: validity triple,
  headline suppression, responsibility gates, before-vs-after comparison, terminal
  lineage, weight-provenance rules (§15). Owner: `forecast-integrity`.
- **PHASE 5 — Reliability & repeatability.** Chaos matrix (interrupt after every
  stage), resume, stale-artifact and mode-crossing protection, same-commit
  repeatability measurement and smallest fidelity-preserving stabilization.
  Owners: `reliability-impl`, `chaos-adversary`, `integration-1`.
- **PHASE 6 — Efficiency.** Delta plan repairs, prompt-prefix/source/claim caching,
  dedup, bounded parallel branches, actor-context reuse rules, consumer/deep modes,
  measured stage costs. Never by removing material reality.
- **PHASE 7 — Frozen release candidate.** Preregistered actor-fidelity suite (8
  pastcasts: ≥6 actor-requiring, ≥4 multi-actor, ≥2 in-simulation information
  response, ≥1 failed-action reconsideration, ≥1 full communication chain) + 1–2
  operational controls + sealed holdouts + repeat runs + clean install + CLI/API/
  replay + final independent audits + release report (NOT READY is an acceptable
  verdict). Owner: `release-manager` + `forecast-integrity`.

## Rerun obligations

After Phase 3+4 integrate: recompile Tesla from the same frozen store (expect a real
operational world or an honest refusal/bounds-only result — never 1/n over invented
factors); rerun BoE (expect bounds + uncalibrated label unless weights become
grounded); run one unseen operational aggregate. After Phase 7: full frozen suite on
one unchanged commit, two repetitions per case.

## Release gates

The §20 gate list of the directive maps to REQUIREMENTS_TRACEABILITY IDs RG-1..RG-10.
The release manager may declare readiness only with every gate MET on real artifacts.
