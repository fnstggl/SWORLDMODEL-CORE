# Launch readiness

Status of each launch gate, in three honest states:

- **MET** — enforced by code that is on the real path and by a test that passes in the
  current suite (289 tests green as of 2026-07-26, branch
  `claude/sworldmodel-consolidation-hyhovj`), or by an artifact that exists in the
  repository. Nothing is marked MET on intention.
- **PENDING-VERIFICATION** — implemented, locally tested or smoke-tested, but not yet
  independently verified (integration review, or a real live/frozen run exercising it).
- **OPEN** — not done, or the evidence that would prove it does not exist yet.

A gate is met only when every row under it is MET.

## Architecture

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| One universal runtime path; no routing on question family | MET | `tests/invariants/test_universality.py` (proven over the AST) |
| Forecast is weighted simulated trajectories only; replays from the ledger | MET | `tests/invariants/test_forecast_and_replay.py` |
| Actor invocation is caused, never scheduled | MET | `tests/invariants/test_scheduling.py` |
| Environment/actor separation (no coercion, no rewriting intentions) | MET | `tests/invariants/test_no_coercion.py` |
| Grounding gates refuse ungrounded worlds | MET | `tests/invariants/test_grounding_gates.py` |
| stdlib-only kernel, zero runtime dependencies | MET | `pyproject.toml` (`dependencies = []`); suite runs offline |

## Reliability

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Every ending owes a diagnosis (refusal and crash both write `diagnosis.json`) | MET | `cli.py` / `scripts/frozen_forecast.py` catch-all handlers; exercised by real refused frozen runs (slice92 wave); classification unit-tested (`tests/unit/test_diagnosis_classification.py`) |
| Compile-stage refusal keeps the research record (live + frozen routes) | MET | `api._checkpoint_partial`; exercised by real refused runs on both routes (geopolitical slices, holdout seed run) |
| A recompilable initial-compile refusal gets its registered repair | MET | `api._replan_initial_compile`; regression in `tests/unit/test_runtime_honesty.py`; exercised by the geopolitical slice (semantic_plan_invalid → repaired → completed) |
| Run output dirs cannot inherit stale artifacts (harnesses AND live CLI) | MET | One canonical list in `sworldmodel/rundir.py` used by `cli.py` and `scripts/_store_loader.py`; drift-scan regression in `tests/unit/test_harness_loader.py` |
| Completion exports the bundle that was actually simulated | MET | `api.forecast` rewrites `research_trace.json`/`evidence_store.json` from the final bundle; regression in `tests/unit/test_runtime_honesty.py` |

## Compiler

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Direct compiler on the canonical path with all `compile_world` gates | MET | `tests/unit/test_compilation_shapes.py`, gate tests in the suite |
| Semantic mode: plan → independent review → static validation → deterministic lowering | MET | Unit-tested (`tests/unit/test_semantic_compiler.py`); exercised end-to-end by real frozen runs (individual 0.25 genuine actor simulation; geopolitical 1.0 factual resolution) |
| Re-perform-history validator rule (no occurrence dated at/before the cutoff) | MET | `semantic_plan.py` `validate_semantic_plan`; covered in `tests/unit/test_semantic_compiler.py` |
| Wake-rule derivation in the semantic lowerer | MET | `semantic_lowering.py` `_wake_rules` (inputs→on_field_change, events→on_event_type, terminal-counted→on_record_in); unit-tested |
| Lowered executables carry only timezone-aware timestamps | MET | `semantic_lowering.py` `_aware_iso` at all five emission sites; regression in `tests/unit/test_semantic_compiler.py` |
| Required-reality-facts lowering in the semantic lowerer | MET | `semantic_lowering.py` derives them from cited world_facts + citation-established initials; unit-tested |
| Cited factual resolution is recognized by every reviewer, not only the gates | MET | Gate 4b admits it; `world_review` settled-record block, coverage `exclusion_reviewer` inversion, trajectory-audit skip — each with regressions (`test_forecast_and_replay.py`, `test_grounding_gates.py`, `test_trajectory_audit.py`) |
| Reviews honor labeled symmetric ignorance (no deleting honest uncertainties) | MET | `world_review` legitimacy-rules block; regression in `test_forecast_and_replay.py` |
| Semantic-vs-direct promotion decision (PART-12 standard) | OPEN | Requires the final 8-case × 2-mode A/B on one unchanged commit; `direct` remains the default until then |

## Fidelity

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Cutoff admissibility mechanically enforced in the store/view | MET | `tests/unit/test_evidence.py` (post-cutoff access refused) |
| Pastcast retrieval is archived-capture-only | MET | Exercised by a real pastcast drill (first holdout seed attempt): live fetches refused, archive-only admission, and the empty admissible view diagnosed as `archive_coverage_failure` (`tests/unit/test_diagnosis_classification.py`) |
| Frozen replay preserves claim provenance (loader side) | MET | `scripts/_store_loader.py`; round-trip pinned in `tests/unit/test_harness_loader.py` |
| Exported stores carry full provenance (exporter side) | MET | `api._claim_record` writes every dataclass field; exporter→loader round-trip pinned in `tests/unit/test_harness_loader.py` |
| Coverage integrity (no verified evidence silently dropped) | MET | Gates in the compiler; exclusion challenges informed by the contract and settled record; exercised by real runs (geopolitical coverage refusal → informed pass) |

## Acceptance

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Acceptance questions frozen (not simplifiable) | MET | `artifacts/acceptance/questions.json` (frozen strings, five cases) |
| Holdout questions pre-registered | MET | `artifacts/ab_matrix.json` holdouts (questions/horizons fixed before any result; as_of pinned at store-generation) |
| Acceptance runs executed and scored on the current code | PENDING-VERIFICATION | slice92 wave green at/near candidate (individual COMPLETED 0.25, geopolitical COMPLETED 1.0, population rerunning); final wave on the unchanged launch candidate pending |
| A/B benchmark matrix + results | PENDING-VERIFICATION | `artifacts/ab_matrix.json` exists; holdout seed stores generating; final 8×2 run pending |

## Stability

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Deterministic replay of a written run | MET | `tests/invariants/test_forecast_and_replay.py` |
| Repeat-run variance on the same frozen store characterized | PENDING-VERIFICATION | Same-store reruns exist across fix commits (geopolitical ×3, individual ×3) and show plan-level variance at temperature 0; a same-commit repeat pair is still to be recorded |
| Live-route stability across repeated runs | OPEN | No repeated live runs recorded |

## Performance

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Per-run cost instrumentation (wall, calls, tokens) | MET | Emitted by both harnesses and the CLI (`run_audit.json`, run summaries) |
| Wall-clock / token budgets defined and demonstrated | PENDING-VERIFICATION | Per-run gateway budgets enforced (`set_budget`, per-run snapshot semantics, unit-tested); measured slice walls 7–11 min vs the 1–5 min consumer target — efficiency report pending |

## Product surface

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Question-only CLI (`sworldmodel forecast`) and trace inspection (`inspect`) | MET | Full live forecasts executed on the current code (holdout seed runs) |
| Frozen-store harnesses (full route + compile slice) | MET | Full frozen forecasts executed on the current code (slice92 wave) |
| Replay viewer (`viz/`) | PENDING-VERIFICATION | Present; not re-verified against traces produced by the current code |
| Documentation describes the real active path | PENDING-VERIFICATION | Updated 2026-07-26; awaiting final integration review at the launch candidate |
| `make check` green | MET | `ruff check`, `ruff format --check`, `mypy --strict` (52 files), full pytest suite all green at `1c6d328` |
