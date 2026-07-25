# Forecast integrity

A forecast has value only when the simulation added something the initialization did not
already contain. Both false completion and unnecessary refusal are failures.

## No arbitrary 50/50

An uncertainty's branch weights may be used as a calibrated probability only when
something grounds them: an empirical frequency, a documented base rate, a verified
reference case, market or survey evidence, or observed current-state evidence — named in
the provenance and citing claims. When nothing grounds a split, the compiler uses
`symmetric_ignorance_assumption`, which is a declaration that the weights are arbitrary.

The mathematical value 0.5000 is not forbidden — a grounded simulation may genuinely
produce it. What is forbidden is a point estimate that merely repeats an ungrounded
initialization. When resolved branches carrying ungrounded weights disagree,
`aggregate()` reports `probability_source = "scenario_enumeration_ungrounded_weights"`,
sets `point_estimate_is_calibrated = false`, and the honest answer is the scenario bounds
`[grounded YES mass, 1 − grounded NO mass]` rather than the arbitrary average.

## Before and after

Every completed forecast records, in `ForecastResult.integrity`:

* `probability_before_simulation` — the terminal evaluated on each branch's initial world,
  after the scenario conditions are applied but before any actor or process runs;
* `probability_after_simulation` — the actual result;
* `simulation_shift` — the difference;
* `counterfactual_note` — whether deleting the actor and process outputs would change the
  answer. When every branch's pre-outcome equals its post-outcome, deletion changes
  nothing and the note says so.

The Bank of England failure — 0.5000 identical to the prior — now reports `p_before=0.0`,
`p_after=0.5`, ungrounded weights, uncalibrated: the headline number can no longer pass as
a simulated finding.

## Producer lineage

Every terminal term must have a real producer: a completed action, an accumulated
process, a scheduled external occurrence, or a verified pre-cutoff fact (factual
resolution). Forbidden and refused before simulation: an uncertainty filling the terminal,
initialization pre-writing it, a reporting event creating the figure it reports, a
decorative process, a hardcoded literal, a default becoming NO, an undetermined value
silently rounded. After simulation, `engine.terminal_lineage` names the exact event that
wrote each terminal field per branch; a YES on an unproduced term is a CRITICAL trajectory
finding.

## The two adversaries

* **Pre-simulation reality auditor** (`world_review`): 14 questions with severities. A
  CRITICAL or HIGH finding — production modeled as reporting, a decorative actor, an
  uncertainty that is the answer, an arbitrary weight, an ungrounded number, a terminal
  that could already resolve — blocks and routes to repair. A finding with no evidence
  basis is demoted to LOW.

* **Post-simulation trajectory auditor** (`trajectory_audit`): mechanical checks
  (repeated equivalent calls, a YES on an unproduced term, a forecast that repeats its
  initialization, a frozen clock) then one model pass, then a mechanical classification:
  `genuine_actor_simulation`, `operational_process_simulation`, `factual_resolution`,
  `unresolved`, or `invalid`.
