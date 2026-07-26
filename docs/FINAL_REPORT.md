# Final report — semantic-compiler experiment and launch-readiness run

Branch `claude/sworldmodel-consolidation-hyhovj`, final commit `19f2a67`.
Benchmark commit `f30ff03` (the only later change, `19f2a67`, fixes a trace-write
crash the benchmark itself exposed; it alters no compile or runtime behavior).
Full matrix: `artifacts/ab_matrix.json`; per-run rows: `artifacts/ab/ab_results.jsonl`;
narrative table: `artifacts/ab/RESULTS.md`.

## The twenty answers

**1. Did the semantic compiler finish end to end?**
Yes, repeatedly, on real runs: frozen-store acceptance cases (BoE 0.25 across four
scenario branches; OPEC+ 1.00 as a cited factual resolution; Tesla 0.50 across two
threshold-spanning branches) and live holdout seed runs (Nobel, SpaceX, FIFA), each
through plan → independent review → validation → deterministic lowering → the
unchanged gates, engine, aggregation, both auditors, and the trace.

**2. Did Tesla run after the timezone fix?**
Yes. After `e5eab72` (lowering emits only timezone-aware timestamps) the population
case ran E2E. Its subsequent runs exposed and then survived the review-legitimacy
fixes; the verified slice outcome is 0.50 with the cited downside alternative intact
and bounds exposed. In the final matrix the same store produced an honest
full-unresolved (see Q19 on plan variance).

**3. Did OPEC+ clear the evidence-coverage and runtime gates?**
Yes. It first refused twice for general reasons that became fixes (an uninformed
exclusion challenge — `613533a`; a plan-stage refusal that never reached its
registered repair — `dd0af7a`), then completed as a cited factual resolution, 1.00,
classification `factual_resolution`, in both the slice wave and the final matrix
(semantic: 26 calls / 734 s vs direct: 88 calls / 1672 s).

**4. Did Bank of England run through the complete trajectory rather than merely compile?**
Yes. Genuine actor simulation: four scenario branches (job market × inflation
outlook), 101 actor invocations in the matrix run, resolved masses 0 / 0.667 / 0.333
(YES/NO/unresolved) with the unresolved mass exposed; the slice run produced 0.25.
The pre-rollout review legitimately refuted a factual-resolution reading by naming
the precise specificity gap, and the repair modeled the conditional process — the
designed behavior after `b3f15e3`.

**5. Did all three vertical slices pass?**
Yes — individual COMPLETED (0.25, genuine actor simulation), geopolitical COMPLETED
(1.00, factual resolution), population COMPLETED (0.50, operational process
simulation, threshold-spanning branches, honest bounds).

**6. Did both compiler modes run on all five acceptance cases and three holdouts?**
Yes — 16 runs, one unchanged commit, identical frozen stores and contracts. 14
produced an outcome; committee (the Banxico pastcast) refused in both modes.

**7. What failed in each mode?**
Direct: committee refused after 12 non-converging repair attempts; individual
completed its forecast then crashed the trace write on a `KeyError('mode')` from a
no-feasible-action record (fixed, `19f2a67`). Semantic: committee refused with three
named validator defects (uncited deciding actor, undeclared change target, an
uncertainty-laundering pattern); population declined to produce a number (honest
unresolved). No other failures.

**8. Did semantic mode materially reduce failures?**
Not at the final commit — because the universal hardening the experiment surfaced
fixed BOTH modes. Direct went from its historical 0/5 (five distinct compile-boundary
deaths across five commits) to 6/8 here. The hypothesis's diagnosis (the
compile boundary was the dominant failure source) is confirmed; the effective cure
was the general fixes, not the mode switch. At `f30ff03` the modes tie 1–1 on
compile refusals, same case.

**9. Did it preserve or improve causal-world fidelity?**
Improved, on the evidence of this matrix. Where the modes disagree, semantic is the
more honest surfacer: BoE — direct reported a flat certain NO against a store
containing the subject's own dovish signals, semantic exposed a third of the mass as
unresolved; Tesla — direct's "0.00" is a conditional number on 32% resolved mass,
semantic refused to number it; EU-Mercosur — semantic's confident NO tracks the
cited Court-of-Justice ratification freeze while direct spread 0.70 unresolved. Its
refusals name exact defects where direct's say "repair did not converge", and every
fidelity defect this run fixed was found through the semantic path's legible
artifacts.

**10. Did it preserve or improve known-outcome accuracy?**
Preserved where the record decides (OPEC+ 1.00 in both modes via the cited pre-cutoff
announcement — semantic classifies it `factual_resolution` rather than simulating a
re-enactment). The holdouts resolve inside the window, so ground truth is not yet
observable; both modes gave 1.00 on all three from the same thin stores (1–4 claims),
which measures shared research quality, not a mode difference.

**11. Did it reduce compiler calls and wall time?**
No in the median (calls 23.5 vs 9.5; wall ≈ 13.4 min vs ≈ 6.4 min); yes exactly where
direct thrashed (geopolitical 26 vs 88 calls, 734 s vs 1672 s, and committee's
bounded refusal vs 12 repair attempts). The p90s converge (~19 min both). Cost
levers that close the gap without touching fidelity are measured in
`docs/EFFICIENCY_REPORT.md` (delta plan revisions, prompt-prefix caching, parallel
branch simulation).

**12. Did it introduce false completions?**
None found. Every semantic completion's number reconstructs from its trajectory or
cited record (Q15); the closest call — negotiation 0.00 — was adversarially checked
against the store and tracks the recorded ratification freeze.

**13. Did it introduce false refusals?**
One class existed and was fixed mid-run: the plan reviewer abstaining over
labeled-ignorance structure the system itself defines as legal (`1676f32`), and the
world review destroying legitimate worlds (settled records `b3f15e3`; labeled
ignorance `83d2a79` + `744ea23`). After those fixes the only semantic refusal on the
matrix is committee, whose named defects are real properties of the plan, and the
only honest-unresolved is population, which is a disclosure, not a refusal.

**14. Did any unseen domain require scenario-specific hardcoding?**
No. The vocabulary remains universal (entities/states/events/processes/affordances/
uncertainties/terminal queries; set/increase/decrease/record_event/send). The
invented-domain regressions (harbor, irrigation council, observatory, canal) pin
this, and the holdout domains (a prize committee, a sports federation, a launch
provider) compiled through the identical code path with no new branches.

**15. Did every accepted forecast come from the trajectory?**
Yes. Sources on the matrix are `weighted_simulated_trajectories` and
`scenario_enumeration_ungrounded_weights` only; preresolved-without-citation, launder,
and unset-guard gates refuse the alternatives, and the counterfactual audit confirms
production (e.g., the Nobel run: deleting actor/process outputs flips 1.00 to 0.00).
The one legitimate non-simulated basis is the cited factual resolution, which is
exactly the OPEC+ case.

**16. Were all CRITICAL and HIGH findings fixed and independently rechecked?**
All CRITICAL/HIGH findings from the audit waves were fixed with focused regressions
and re-verified (13 independently re-reviewed CLOSED in the earlier wave; this
wave's twelve fixes — timestamps, informed exclusion reviewer, initial-compile
replan, settled-record trio, review legitimacy, prior-plan repairs, uncertainty
citations visible, plan-error escape, research-record attach, archive-gap diagnosis,
canonical artifact list, report-line totality — each carry a regression and were
re-exercised by real runs). Remaining open findings are MEDIUM/LOW or recorded
limitations (Q17/Q20).

**17. Median and p90 runtime, calls, tokens, estimated cost?**
Across the 8 final-matrix runs per mode — semantic: wall median ≈ 801 s, p90 ≈ 1142 s;
calls median 23.5, p90 ≈ 55; tokens (in+out) median ≈ 181k, max 541k (BoE, 101 actor
calls). Direct: wall median ≈ 386 s, p90 ≈ 1131 s; calls median 9.5, p90 ≈ 37; tokens
median ≈ 61k, max 345k. At flash-tier list prices (fractions of a dollar per million
tokens) that is single-digit cents per typical run and tens of cents at the extreme;
exact cost depends on the account's contracted rate.

**18. Is semantic compilation now the production default?**
No. The PART-12 standard (simultaneously fewer compile failures, fewer repair calls,
lower wall, no wrong-world acceptances, no new false refusals) is not met on cost:
compile failures tie and semantic is slower in the median. Direct remains the
default; `--compiler semantic` remains fully supported, and the decision record in
`artifacts/ab/RESULTS.md` states the re-run condition (close the wall gap via the
measured efficiency levers, then re-benchmark).

**19. Is the system genuinely ready for an honest alpha launch?**
For an alpha whose contract is honesty-first — real trajectories or cited records
only, labeled ignorance, bounds, named refusals, complete artifacts — the machinery
now demonstrably delivers that contract on 7 of 8 matrix cases per mode. Three
things keep this short of an unqualified READY: (a) run-to-run compile variance at
temperature 0 (the same frozen store produced a threshold-spanning 0.50 world and an
honest-unresolved world on the same code) — provider-side nondeterminism both modes
share; (b) wall time (median 6–13 min vs the stated 1–5 min consumer target); (c)
thin-store holdout research (1–4 claims yielding uniform 1.00s — the binding
constraint is retrieval depth, not compilation). None of these makes the system
dishonest; all three make it slower, more variable, and more often
honestly-unresolved than a consumer product should be.

**20. If not, what exact blocker remains?**
No internal launch-blocking defect remains fixable-but-unfixed: everything found by
this run's loop is fixed, regression-pinned, and re-exercised. The three residuals
in Q19 are the exact blockers, and each is external or infrastructural in nature:
provider nondeterminism at temperature 0 (external to this codebase; mitigable with
plan-level majority/consensus at ~2–3× compile cost), provider generation throughput
(the wall-time floor is output-token latency; the fidelity-preserving levers are
specified and measured in `docs/EFFICIENCY_REPORT.md`), and retrieval depth on
sparse topics (bounded by source availability and the extract budget at the
provider boundary). Verdict: **honest-alpha READY with the three stated
qualifications; consumer-target NOT READY until the wall and variance residuals are
closed.**

## Per-case appendix (final matrix, commit f30ff03)

| case | mode | outcome | p | Y/N/U | calls | wall |
| --- | --- | --- | --- | --- | --- | --- |
| individual | direct | resolved, then trace-write crash (fixed) | 0.00 | 0/1.0/0 | 15 | 559 s† |
| individual | semantic | partially_resolved | 0.00 | 0/0.667/0.333 | 112 | 868 s |
| geopolitical | direct | resolved | 1.00 | 1.0/0/0 | 88 | 1672 s |
| geopolitical | semantic | resolved (factual resolution) | 1.00 | 1.0/0/0 | 26 | 734 s |
| population | direct | partially_resolved | 0.00‡ | 0/0.32/0.68 | 6 | 316 s |
| population | semantic | unresolved | — | 0/0/1.0 | 15 | 501 s |
| negotiation | direct | partially_resolved | 0.50 | 0.15/0.15/0.70 | 13 | 899 s |
| negotiation | semantic | resolved | 0.00 | 0/1.0/0 | 30 | 871 s |
| committee | direct | REFUSED (12 repair attempts) | — | — | 13 | 342 s |
| committee | semantic | REFUSED (3 named validator defects) | — | — | 25 | 1274 s |
| holdout_science | direct | resolved | 1.00 | 1.0/0/0 | 6 | 194 s |
| holdout_science | semantic | resolved | 1.00 | 1.0/0/0 | 12 | 400 s |
| holdout_sports | direct | resolved | 1.00 | 1.0/0/0 | 5 | 103 s |
| holdout_sports | semantic | resolved | 1.00 | 1.0/0/0 | 14 | 294 s |
| holdout_space | direct | resolved | 1.00 | 1.0/0/0 | 6 | 430 s |
| holdout_space | semantic | partially_resolved | 1.00‡ | 0.444/0/0.556 | 22 | 1109 s |

† crash after the forecast was written; `19f2a67` makes the report line total.
‡ conditional on resolved mass; the unresolved mass and bounds are disclosed in the
forecast artifact.
