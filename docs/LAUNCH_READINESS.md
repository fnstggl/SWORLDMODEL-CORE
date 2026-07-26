# Launch readiness

Status of each launch gate, in three honest states:

- **MET** — enforced by code that is on the real path and by a test that passes in the
  current suite (256 tests green as of 2026-07-26, branch `fix/harness-c`), or by an
  artifact that exists in the repository. Nothing is marked MET on intention.
- **PENDING-VERIFICATION** — implemented, locally tested or smoke-tested, but not yet
  independently verified (integration review, or a real live/frozen run exercising it).
- **OPEN** — not done, or the evidence that would prove it does not exist yet.

A gate is met only when every row under it is MET. No gate is fully met today.

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
| Every ending owes a diagnosis (refusal and crash both write `diagnosis.json`) | PENDING-VERIFICATION | Implemented in `cli.py` / `scripts/frozen_forecast.py`; classification unit-tested (`tests/unit/test_diagnosis_classification.py`); no end-to-end failure drill on a real run yet |
| Compile-stage refusal keeps the research record (live route) | PENDING-VERIFICATION | `live_research._compile` attaches partial trace/store; `api._checkpoint_partial` writes it; not exercised on a real refused live run |
| Compile-stage refusal keeps the research record (frozen route, M-7) | PENDING-VERIFICATION | Mirrored in `FrozenResearchBackend.research`; awaits a real refused frozen run and integration review |
| Harness output dirs cannot inherit stale artifacts (M-6) | MET | `scripts/_store_loader.py` `prepare_run_dir` (stamp + clear); `tests/unit/test_harness_loader.py` |
| Live CLI `--trace` dirs cannot inherit stale artifacts | OPEN | `cli.py` only `mkdir`s the trace dir; the same stale-artifact risk the harnesses had still exists on the live route (src-side change, owned separately) |

## Compiler

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Direct compiler on the canonical path with all `compile_world` gates | MET | `tests/unit/test_compilation_shapes.py`, gate tests in the suite |
| Semantic mode: plan → independent review → static validation → deterministic lowering | PENDING-VERIFICATION | Implemented and unit-tested (`tests/unit/test_semantic_compiler.py`); full-route behavior on real questions not yet benchmarked |
| Re-perform-history validator rule (no occurrence dated at/before the cutoff) | MET | `semantic_plan.py` `validate_semantic_plan`; covered in `tests/unit/test_semantic_compiler.py` |
| Wake-rule derivation in the semantic lowerer | OPEN | Lowerer currently emits `wake_rules: []`; design contract in `docs/SEMANTIC_COMPILER.md`, implementation landing in parallel |
| Required-reality-facts lowering in the semantic lowerer | OPEN | Currently emits `required_reality_facts: []`; same parallel work |
| Semantic-vs-direct promotion decision (PART-12 standard) | OPEN | Requires the A/B benchmark below; `direct` remains the default |

## Fidelity

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Cutoff admissibility mechanically enforced in the store/view | MET | `tests/unit/test_evidence.py` (post-cutoff access refused) |
| Pastcast retrieval is archived-capture-only | PENDING-VERIFICATION | Implemented in retrieval; needs a live pastcast drill to verify end-to-end |
| Frozen replay preserves claim provenance (loader side, H-6) | MET | `scripts/_store_loader.py`; round-trip pinned in `tests/unit/test_harness_loader.py` |
| Exported stores carry full provenance (exporter side) | OPEN | `api.py` / `cli.py` still export the legacy 8 fields, so every store replayed today loads with defaulted provenance and the loud legacy warning; the extended export is landing in parallel (src-side) |
| Coverage integrity (no verified evidence silently dropped) | PENDING-VERIFICATION | Policy in `docs/COVERAGE_INTEGRITY.md`, gates in the compiler; not independently audited against a real run |

## Acceptance

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Acceptance questions frozen (not simplifiable) | MET | `artifacts/acceptance/questions.json` (frozen strings, five cases) |
| Holdout questions pre-registered | MET | `artifacts_holdouts.json` |
| Acceptance runs executed and scored on the current code | OPEN | No run artifacts exist in `artifacts/acceptance/` beyond the question set |
| A/B benchmark matrix + results (`artifacts/ab_matrix.json`, `artifacts/ab/RESULTS.md`) | OPEN | Neither artifact exists; the benchmark has not been run |

## Stability

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Deterministic replay of a written run | MET | `tests/invariants/test_forecast_and_replay.py` |
| Repeat-run variance on the same frozen store characterized | OPEN | Requires multiple frozen runs per question; no such runs recorded |
| Live-route stability across repeated runs | OPEN | No repeated live runs recorded |

## Performance

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Per-run cost instrumentation (wall, calls, tokens) | MET | Emitted by both harnesses (`metrics.json`, run summaries) |
| Wall-clock / token budgets defined and demonstrated | OPEN | No budget targets recorded, no benchmark runs to measure against |

## Product surface

| Item | Status | Evidence / what is missing |
| --- | --- | --- |
| Question-only CLI (`sworldmodel forecast`) and trace inspection (`inspect`) | PENDING-VERIFICATION | CLI exists and parses (smoke-tested); a full live forecast requires provider keys and has not been re-verified on the current code |
| Frozen-store harnesses (full route + compile slice) | PENDING-VERIFICATION | Both run to the gateway boundary without keys (smoke-tested); a full frozen forecast on the current code awaits the A/B runs |
| Replay viewer (`viz/`) | PENDING-VERIFICATION | Present; not verified against traces produced by the current code |
| Documentation describes the real active path | PENDING-VERIFICATION | `docs/SEMANTIC_COMPILER.md` status, `docs/GETTING_STARTED.md` and this checklist updated 2026-07-26; awaiting integration review |
| `make check` green | OPEN | `ruff check` and `mypy --strict` pass and the full pytest suite is green, but `make lint` fails: `ruff format --check` reports 4 `src/` files and `tests/unit/test_semantic_compiler.py` unformatted (pre-existing, owned by src-side work) |
