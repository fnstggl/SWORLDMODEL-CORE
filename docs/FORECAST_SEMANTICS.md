# Forecast semantics

The simulation forecast is computed **solely** from world trajectories. No historical
prior, generic outcome prior, persistence prior, numerical consensus model, or
forecast combiner may override completed simulated trajectories.

## The computation

For a binary outcome, aggregating over terminal branches (`outcomes.aggregate`):

```
resolved_yes  = Σ weight  over resolved branches with outcome YES
resolved_no   = Σ weight  over resolved branches with outcome NO
unresolved    = Σ weight  over unresolved branches (+ disclosed truncated mass)
resolved      = resolved_yes + resolved_no

simulation_probability = resolved_yes / resolved        (conditional on resolved)
                       = None  if resolved == 0          (no fabricated point estimate)

lower_bound = resolved_yes / total_initial_mass
upper_bound = (resolved_yes + unresolved) / total_initial_mass
```

`probability_source` is always `weighted_simulated_trajectories`. Mass is conserved:
`resolved + unresolved == 1` (± ε), else `MassConservationError`.

The result clearly distinguishes simulation probability, resolved mass, unresolved
mass, lower/upper bounds, and the per-branch outcomes — so the headline is
reconstructable by hand from the branch table
(`test_no_hidden_override_probability_is_reconstructable`).

## No hidden override

There is one causal route from actors and external events to the terminal. Deleting
the actor decisions changes or kills the forecast: with actor decisions disabled the
status becomes `UNRESOLVED` and the probability becomes `None`, never a default vote
or a prior (`test_provider_failure_becomes_unresolved_not_a_default_vote`,
`test_deleting_actor_decisions_changes_the_forecast`).

A reference-class estimate may appear in `diagnostics` as a separately labeled
comparator, but it never contributes to the simulation probability.

## Branch weights carry provenance

Every branch weight names its source: `direct_empirical`, `market_or_survey`,
`calibrated_behavior`, `explicit_model`, `symmetric_ignorance`, or `sensitivity_only`.
Uncertainty over future external data is genuine epistemic uncertainty, weighted with
`symmetric_ignorance` / `explicit_model` provenance; where weakly identified, the
reported bounds expose the sensitivity rather than concealing it behind precise
decimals. Precision is never invented from one example, a qualitative label, or the
number of hypotheses a model happened to produce.

## Branching discipline

Branches represent genuine unknown realities, not decorative personality variants
(`uncertainty.enumerate_scenarios`). We build a small joint scenario set — not the
Cartesian product of every generated uncertainty — normalize per-variable weights,
cap the total, and **disclose** any truncated mass rather than silently
renormalizing. Two runs of the same seed and model reproduce the same trace, which is
the safe basis for merging genuinely-identical decision contexts; distinct conditions
are never merged (`test_branch_merging_only_for_equivalent_states`).

## Replay

Every terminal outcome is reconstructable from the event ledger: the votes recorded as
`VOTE_CAST` events re-tally to the same outcome
(`test_branch_replay_reconstructs_the_same_outcome`).
