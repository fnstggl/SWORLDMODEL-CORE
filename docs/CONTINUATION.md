# CONTINUATION — session recovery file

A new session begins by reading MASTER_EXECUTION_PLAN.md, DEFECT_REGISTER.md,
REQUIREMENTS_TRACEABILITY.md, then this file. Do not repeat completed work because a
context window ended.

## State at last checkpoint (2026-07-26, Phase 0 → Phase 1 launch)

- Branch: `claude/sworldmodel-consolidation-hyhovj` (repo fnstggl/SWORLDMODEL-CORE,
  base `claude/sworldmodel-core-rebuild-lcfpq1`). Working tree clean at checkpoint.
- Commit: see `git log` — plan docs committed on top of e17f189.
- Canonical path: semantic compiler default-for-engineering behind `--compiler`;
  direct retained for comparison only; mode stamped everywhere. No silent fallback.
- Phase 0 (establish truth): VERIFIED — 304 tests green, ruff+format clean,
  `mypy --strict` clean over 52 files at e17f189; forensic reconstruction verdicts
  RECONSTRUCTED for individual/population/geopolitical
  (`artifacts/forensics/*/forensic_verdict.json`); traces internally consistent
  after FC-1..FC-9. Baseline metrics: A/B medians semantic ≈801 s / 23.5 calls /
  ≈181k tokens; direct ≈386 s / 9.5 calls / ≈61k tokens (artifacts/ab/RESULTS.md).
- Prior-result reclassification (binding, D9): Tesla 50% invalid forecast; BoE 25%
  valid pathway, uncalibrated number; OPEC+ 100% factual lookup with unresolved
  wording ambiguity.

## Active org

- CTO/integration: main session. Write-agent cap 4 (current: obs-trace, obs-viewer).
- Phase 1 in flight: `obs-trace` (worktree, branch `fix/observability-trace`;
  replaycore + trace artifacts OBS-1..4,7,8) and `obs-viewer` (worktree, branch
  `fix/observability-viewer`; OBS-6). Reviewer to spawn after integration:
  read-only observability reviewer on a fresh real run.

## Frozen runs / artifacts

- Frozen acceptance stores: artifacts/acceptance/<case>/run_trace/evidence_store.json
  (5 cases); holdout seed stores artifacts/ab/seed_holdout_*/ (3).
- Forensic sets: artifacts/forensics/{individual,population,geopolitical}/ (13
  artifacts each incl. run_dossier.html).
- A/B record: artifacts/ab/ab_results.jsonl + RESULTS.md (commit f30ff03).

## Phase 1 status (2026-07-26 ~19:10)

Integrated at 04a5395 (obs-trace replaycore + trace artifacts; obs-viewer dossier
views). CTO-verified: 313 tests, mypy --strict 53 files, forensic verdicts
byte-identical. Real gate run artifacts/phase1/individual/ completed at the
integrated commit with the FULL observability set; it took the settled-record path
and published established_before_simulation_from_cited_record (new honest label
working in production). Read-only reviewer `obs-reviewer` is auditing it now —
Phase-1 exit REJECTED by obs-reviewer:
trace side sound (OBS-1..5,7,8 closeable — every artifact recomputes byte-identically,
LLM-free replay confirms YES at t0), but OBS-6 CRITICAL: viz resolve_forensics
basename fallback attached slice92's forensics (0.25 BRANCH_WEIGHTS_DOMINATED) to the
phase1 run (1.0 FACTUALLY_RESOLVED); plus HIGH (_llm prefers stale llm_calls_full) and
MEDIUM (state view ignores branch_initial_state.json). obs-viewer respawned with the
three findings (identity-match rule, own-calls preference, initial-state artifact);
after its fix: CTO verify, commit, obs-reviewer re-checks items 4+8 only, then OBS
rows close and Phase 2 launches (runtime-impl + realism-adversary per plan). Note for FD-12: same frozen store has now produced
three structurally different plans across runs (0.25 four-branch, 0.50-class, 1.0
factual resolution) — repeatability work is Phase 5.

## PHASE 1 CLOSED (obs-reviewer APPROVE at 127f8a7; OBS-1..8 CLOSED; FD-5, FD-6 closed)

Phase 2 launched: runtime-impl building ACT-5..7 production-path harnesses + TMP-2
temporal report; realism-adversary reviews on completion; then Phase 3 per plan.

## OPERATING LESSON (2026-07-26 20:35, CTO)

Subagent-launched long runs DIE when the agent's turn ends. The realism-adversary
launched artifacts/phase2/geopolitical2 at 20:20:59 (commit bc20aa0); it wrote only
its research checkpoint, left no process and no diagnosis, and the agent then blocked
waiting on a dead poll. RULE: every real production run (frozen slice, acceptance
case, benchmark) is launched by the CTO/main session in background and its artifact
path handed to the reviewing agent. Subagents verify artifacts; they do not own
long-lived processes. Also: check transcript size + child-process liveness, not just
elapsed time, when judging whether an agent is progressing.

## PHASE 2 CLOSED (realism-adversary APPROVE at 89e04c0)

ACT-5..8, TMP-2, TMP-4, COM-1..2 CLOSED; FD-7, FD-13, FD-18..21 closed. Verified by
adversarial probe, not assertion: 8 collapse attacks on per-date releases all failed;
message identities sent=delivered+undelivered and delivered=noticed+missed hold on a
world where both diverge (7=6+1, 6=4+2); D10's deadline cancellation verified on a
five-participant probe (acted->cancelled; declined/rejected/waited/own-revisit->kept);
the new harness proven to FAIL under pre-fix rules (12 decisions read an unpublished
figure 19 days early). New non-blocking: FD-22 (replaycore comm kinds not narrowed),
FD-23 (LOW notes). Phase 3 launches: world-compiler-impl (CWF-1..8) + causal-adversary.

## Next exact steps

1. Integrate `fix/observability-trace` then `fix/observability-viewer` (in that
   order; both touch nothing shared except viz reads artifacts).
2. Rerun one frozen slice (geopolitical or individual) at the integrated commit to
   produce Phase-1 artifacts on a real run; spawn read-only observability reviewer.
3. Phase 1 exit → spawn Phase 2 (`runtime-impl` harnesses ACT-5..7, TMP-2 report;
   `realism-adversary` review).
4. Phase 3 (`world-compiler-impl`: CWF-1..7) → Phase 4 (`forecast-integrity`:
   FI-1..6) → Tesla/BoE reruns + one unseen operational aggregate.
5. Preregister EVAL-1 suite (draft in progress by CTO; user veto window before
   freeze).

## Blockers / user decisions pending

- None hard-blocking. EVAL-1 suite DRAFT-FROZEN in
  docs/acceptance/ACTOR_FIDELITY_SUITE.md (10 cases, outcome commitments sealed in
  artifacts/sealed/outcomes.json, sha256 committed). User veto window announced
  in-session; store freeze happens in one commit before any scoring run.

## Session checkpoint — Phase 3 BLOCKED, Phase 4 landed (commit eef48e9)

**Branch** `claude/sworldmodel-consolidation-hyhovj`, HEAD `eef48e9`. Tree carries uncommitted
work from four concurrent agents (api.py, cli.py, diagnosis.py, models.py, outcomes.py,
semantic_*.py, uncertainty.py, world_review.py, replaycore.py, scripts/forensics.py, several test
files) plus untracked `responsibility.py`, `tests/unit/test_publication_gate.py`,
`test_responsibility.py`, `test_stock_flow.py`. **Do not assume the tree is clean.**

### Where the program actually stands

- **Phase 4 (forecast integrity, FI-1..FI-6) is DONE and verified against live artifacts**, not
  fixtures. All three headline numbers this program was convened over now publish as `None` with
  `point_estimate_is_calibrated: false`: OPEC+ 1.0 (0.25 resolved mass, minority rule), BoE 0.25
  (ungrounded weights disagree), Tesla 0.50 (disagree + FI-5 backstop names
  `seasonal_factor_q3_over_q2` off the real branch table).
- **Phase 3 (causal-world fidelity) is BLOCKED by adversary verdict.** No CWF row may close. The
  one-sentence finding: the gates fire on the Tesla failure's *syntax*, not its *shape*. All 25 CWF
  regressions are green and every attack passes them.
- **FD-42 blocks activation of the D6 responsibility gate** and is a live false-invalid in
  `scripts/forensics.py`. Until it lands, `aggregate(responsibility=...)` must NOT be wired;
  `world_review_blocking=` is unaffected.

### Live agents (four write-enabled, the cap)

| agent | owns | assignment |
| --- | --- | --- |
| world-compiler-impl-3 | semantic_plan/compile/lowering, uncertainty, test_semantic_compiler, test_causal_world_fidelity | FD-39 units/dimensions (ranked first), FD-24 clamp-with-shortfall at the state-transition boundary, FD-25 recurrence, FD-32 |
| world-review-impl | world_review.py, test_review_evasion (new) | FD-28 lineage closure (the highest-value single lever), FD-31, FD-33, FD-40 evasion suite |
| integration-1 | api.py, cli.py, test_forecast_and_replay, test_publication_gate | FD-34 ruling (mechanical findings block regardless of provider outage), FD-41 |
| reliability-impl | replaycore.py, scripts/forensics.py, test_replay_record_content (new) | FD-42 |

`forecast-integrity` is READ-ONLY this round (fifth agent; cap is four) — diagnosing whether a
suppressed run is legible to a reader or merely declines.

### Standing rules confirmed this session

- **Long runs belong to the CTO.** A subagent-launched run dies with the agent.
- **Agents do not commit.** Disjoint file ownership; contended files are handed over as verbatim
  patches and applied at integration.
- **Demotion only, never promotion**, for weight provenance (FI-6). A mechanical check may strip a
  weight's claimed grounding; nothing may automatically grant it.
- **A builder never closes their own defect.**
- A fix that makes a previously-green regression red is a FINDING, not a thing to tune around. The
  shipped "good" recharge fixture is expected to fail the lineage-closure fix.

### Next exact steps

1. Integrate the four agents' returns; apply verbatim patches for contended files.
2. Re-dispatch the semantic_plan.py grounding family — FD-29 (one cited leg silences D3), FD-30
   (`cited()` is an existence check — load-bearing for D3, D5, D4 and FD-11 simultaneously),
   FD-35 (16-combination cap), FD-36, FD-38 — which is QUEUED behind world-compiler-impl-3 on file
   ownership, not on priority. FD-30 is the widest single hole in the static gates.
3. CWF-7 per the adjudicated design now in REQUIREMENTS_TRACEABILITY.md.
4. Re-run the adversary against the integrated tree. Phase 3 does not close on regressions.
