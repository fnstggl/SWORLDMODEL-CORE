# Legacy extraction record

**Summary: no legacy files were copied.** Every utility here was reimplemented from
scratch after inspecting the legacy repository (`../SWORLDMODEL`) as forensic
reference. This repository imports no runtime code from the legacy tree, contains no
Lean V1/V2, no PR #133 recovery ladders, no PR #134 numerical deliberation, and no
compatibility fallback that could reactivate a legacy path.

The Generative Agents repository (`../generative_agents-reference`) was used only as
*conceptual* reference for the persistent-actor loop; none of its Smallville/grid/
filesystem/prompt code was imported or copied.

## Inspected, then reimplemented fresh

Each of these concepts existed in the legacy tree but was found coupled to a broad
runtime; each was rewritten cleanly here with focused tests. Nothing was assumed
correct; behavior was re-derived.

| concept | legacy location inspected | clean reimplementation | tests |
|---|---|---|---|
| strict `as_of` cutoff filtering | `swm/retrieval/asof_store.py` (the clean gate) | `evidence.EvidenceView` (physical exclusion + `CutoffViolationError`) | `test_evidence.py`, `test_evidence_invariants.py` |
| model/provider gateway abstraction | `swm/world_model_v2/lean_v2/gateway.py` | `gateway.ModelGateway` + call/token ledger | `test_actors.py`, `test_runtime.py` |
| prompt/response tracing | `swm/world_model_v2/lean_v2/traces.py` | `tracing.TraceContext` (jsonl ledgers + report) | Banxico integration |
| deterministic vote counting | `swm/world_model_v2/institutions_v2/decisions.py` | `mechanisms.tally_votes` / `evaluate_terminal` | `test_mechanisms.py` |
| probability-mass accounting | `swm/world_model_v2/lean_v2/worlds.py` (`WeightedBranchCoalescer`) | `uncertainty` + `outcomes` (conserve + disclose truncation) | `test_uncertainty.py`, `test_runtime.py` |
| canonical option normalization | `swm/world_model_v2/mode_graph.py` | discrete option handling in `mechanisms` / `gateway` | `test_interaction.py` |
| immutable branch cloning | `swm/world_model_v2/lean_v2/worlds.py` (`clone`) | `world.WorldState.clone` (deep-copies actor memory) | `test_runtime.py` |
| event-driven clock | `swm/world_model_v2/institutions_v2/procedure.py` | time advance inside `runtime` / `world.apply` | `test_runtime.py` |

## Inspected and explicitly rejected

These legacy behaviors are the reason for the rebuild. They are **not** present here.

| rejected behavior | legacy location | why rejected |
|---|---|---|
| numeric scalar-convergence "deliberation" | `swm/simulation/agent_society.py` (`_deliberate_step`, `consensus_pull`) | deliberation as a moving-mean pull is not a social process |
| institution probability override | the `consensus_pull` term dragging actors to the body mean | a hidden model must never overwrite completed trajectories |
| action-baseline actor policy | `swm/world_model_v2/phase4_policy.py`, `phase4_llm_baselines.py` | actor behavior became a statistical distribution, not simulated choice |
| deadline-forced voting / recovery ladders | `swm/world_model_v2/lean_v2/engine.py` (deadline force-default), `unified_runtime.py` (`_apply_result_guards`) | forcing a vote fabricates the information the actor rationally required |
| threshold rescaling / roster compression | `swm/world_model_v2/actor_selection.py` + `decisions.py` denominator | compressing a roster silently rescales the real threshold |
| forecast recovery / prior-simulation combiner | `swm/world_model_v2/forecast_recovery.py` | a prior standing in for unsimulated branches overrides the simulation |
| multiple execution profiles | Lean V1 / Lean V2 / full-fidelity routing | one product needs one authoritative semantics |
| generated n=1 latent-state frequency tables | grounding/"counted reference class" path | one event cannot become several independent rates |

## Verification that no legacy path can activate

- `pyproject.toml` declares zero runtime dependencies; the package imports only the
  standard library and its own modules.
- `tests/invariants/test_anti_hardcoding.py` greps the core for the forbidden
  mechanism names above (`consensus_pull`, `body_mean`, `mean_field`,
  `coalition_discipline`, `leadership_floor`, `if institution ==`, …) and fails if any
  reappears.
