# Merging the universal simulator into the rebuilt live-research core

This records the merge of the **CORE-UNIVERSAL** line (one universal world simulator:
compiled WorldSpec, dynamic actions, safe universal effects, novel actions, declarative
terminals) into the **CORE-REBUILT** line (live question-only research, real DeepSeek,
source fetching + PDF, verification, lineage, cutoff, coverage integrity, repair,
tracing). CORE-REBUILT is the surviving branch.

The result is **one** production architecture, not two.

## Branches, SHAs, and rollback

| Role | Branch | SHA |
|---|---|---|
| Destination (surviving) | `claude/sworldmodel-core-rebuild-lcfpq1` | `e88a409` before merge |
| Source (merged in) | `claude/sworldmodel-core-universal-simulator-z8zsfo` | `621cef4` |
| Merge base | — | `b09c194` |
| Backup tag | `pre-universal-merge-20260724T204203Z` → `e88a409` | local |

The merge is a `--no-ff` merge commit, so `e88a409` remains an ancestor of the branch
and is always reachable for rollback (`git reset --hard e88a409`). The annotated backup
tag was created locally; the sandbox git proxy refuses non-branch refs, so it could not
be pushed — the `--no-ff` ancestry is the effective rollback guarantee.

## Capabilities preserved from CORE-REBUILT

Live research orchestration; Google News RSS + DuckDuckGo discovery; direct source
fetching; official-domain queries; **PDF parsing** (`pdf_text.py`); publication /
availability dating with UTC normalization; as-of cutoff enforcement; claim extraction
with supporting-excerpt verification (unsupported claims rejected); evidence ids and
lineage de-duplication; contradiction tracking; iterative research with saturation
stopping; DeepSeek HTTP transport with retries, backoff and bounded deterministic
repair; token/latency/stage accounting; the canonical `EvidenceStore`; the deterministic
**candidate inventory**; **evidence-to-world coverage integrity**; the **exclusion
challenge** (an independent LLM second opinion can veto a wrongful exclusion);
**targeted research repair** (`augment_for_coverage` + `_compile_with_repair`); trace
storage and replay; branch-mass accounting.

## Capabilities preserved from CORE-UNIVERSAL

`WorldSpec` (entities, actors, fields, resources, channels, documents); dynamically
compiled `ActionDefinition` records; the minimal **universal effect language**
(`effects.py`) as the only hardcoded action machinery; compiled **process graphs**;
**declarative terminal expressions** with universal operators (`expressions.py`);
**novel-action** interpretation and validation (`novel.py`); authority / feasibility /
timing / target / resource validation (`executor.py`); action execution independent of
action *names* (behavior is the compiled effects); **one universal event engine**
(`engine.py`); terminal evaluation with no scenario family.

## The canonical call path

```
api.forecast(question, as_of, horizon, config)
  └─ api.run_forecast
      ├─ config.research_backend.research(...)        live_research.LiveResearchBackend
      │     ├─ research_planner.plan_research          (free-text process summary)
      │     ├─ rss / search / source_fetch / pdf_text  (discovery + fetch)
      │     ├─ source_extract.extract_claims           (verified against fetched text)
      │     └─ world_compiler.compile_world_spec_live  (LLM compiles the WorldSpec,
      │            handed coverage.evidence_checklist so it cannot forget a verified item)
      ├─ api._compile_with_repair                      (up to 3 attempts)
      │     └─ world_compiler.compile_world
      │           ├─ build_base_world                  one authoritative WorldState
      │           ├─ reality.verify_reality            gate 1: structural truth
      │           ├─ world_compiler.world_spec_view    the EXACT compiled WorldSpec
      │           ├─ coverage.build_candidate_inventory
      │           ├─ coverage.assess_coverage          (+ exclusion_reviewer, live only)
      │           ├─ coverage.enforce_coverage         gate 2: nothing lost
      │           └─ uncertainty.enumerate_scenarios   weighted branches
      │     └─ on WorldIntegrityError → backend.augment_for_coverage → recompile
      ├─ engine.run                                    the one universal event runtime
      │     ├─ _release_scenario_data                  branch hypothesis as observable data
      │     ├─ _drive_activations                      EVENT-DRIVEN actor invocation
      │     │     ├─ actors.ActorRuntime.step          perceive → retrieve → reflect → intend
      │     │     ├─ executor.ActionExecutor.execute   validate → universal effects
      │     │     └─ novel.resolve_novel               novel-action route
      │     └─ engine.evaluate_terminal                declarative predicate over world state
      ├─ outcomes.aggregate                            weighted YES trajectories only
      └─ tracing.TraceContext                          ledger, coverage report, report.md
```

## Conflicts and their semantic resolutions

| File | Conflict | Resolution |
|---|---|---|
| `api.py` | rebuilt's coverage-repair loop vs universal's compile/engine call | **Fused**: kept `_compile_with_repair` and pointed it at the universal `compile_world` + `engine.run`. |
| `actors.py` | rebuilt's `_to_intent` vote-option coercion (`_closest_option`) vs universal `ActionChoice` | **Universal**. Coercion contradicts "invalid actions are rejected, never semantically coerced"; universal validates and rejects. `pending_questions` was re-added from rebuilt. |
| `prompts.py` | committee prompts vs universal prompts | **Universal**, plus rebuilt's evidence checklist injected into the world-compile prompt. |
| `deepseek_gateway.py` | rebuilt's large per-stage token budgets vs universal's new task kinds | **Fused**: rebuilt's anti-truncation rationale applied to `compile_world_spec` / `interpret_novel` / `exclusion_challenge`. |
| `source_fetch.py` | import adjacency; both sides fixed naive/aware datetimes | **Rebuilt** (owns fetching; its normalization is more thorough). |
| `README.md` | two different invariant lists | **Both** invariant sets kept. |
| `coverage.py` | keyed on `decision_body` / `decision_rule.kind` / `terminal_predicate` | **Generalized**: focal identities replace "decision body"; the dead `rule_kind` was dropped; resolution requirements read the declarative `TerminalExpression`. |
| `test_coverage.py` | committee-schema corpora | **Ported** to WorldSpec corpora (`_worlds.named_body_world`); all 10 coverage cases preserved. |
| `compiler.py` / `universal_compiler.py` / `runtime.py` (modify/delete) | rebuilt modified, universal deleted | **Deleted**, after extracting `_world_spec_view` → `world_compiler.world_spec_view`, `_exclusion_reviewer` → `world_compiler.exclusion_reviewer`, and the checklist injection into the live compile. |

## Duplicate production paths removed

`compiler.py`, `universal_compiler.py`, `runtime.py`, `mechanisms.py`, `protocols.py`,
`intents.py`, `tests/unit/test_universal_compiler.py`, `tests/unit/test_mechanisms.py`,
`tests/integration/test_generality.py`. No compatibility wrappers, re-export shims or
dormant runtimes remain. A runtime import audit (`tests/acceptance/test_merge.py`)
proves none is reachable from `api.forecast()`.

## What changed behaviorally

* **Event-driven scheduling.** `engine._drive_activations` replaced the fixed
  `for round: for actor:` loop. An actor is invoked only with a concrete trigger
  (`information`, `opportunity`, `pending_need`, `response_to_completed_action`), may be
  re-invoked when a completed action reaches it, and is otherwise left inert.
  `node.rounds` is now a per-actor *budget* bounding cascades, not a schedule. Every
  invocation records its trigger in the trace.
* **Coverage now inspects the compiled WorldSpec.** `world_spec_view` derives wiring
  from the compiled program itself: participants of process nodes, actions offered by
  nodes, fields read by preconditions/policies/uncertainties/terminal, and objects
  touched by compiled effects.
* **Persistent pending needs.** `ActorState.pending_questions` is carried across
  invocations, surfaced in the decision prompt, and is itself a scheduling trigger.

## Tests and smoke runs

* `137 passed` (unit, invariants, integration, acceptance); `ruff` clean;
  `mypy --strict` clean on 37 source files.
* Cross-domain, one entry point, no source changes: committee 0.80, geopolitical 0.40,
  individual response 0.60, negotiation 0.70, population behavior 0.50.
* Banxico pastcast: p = 0.72, coverage complete (27 material candidates), Brier 0.0784,
  directional call correct against the sealed post-cutoff result.
* Live DeepSeek, question only: research 5 queries / 9 sources fetched / 4 unsupported
  claims rejected / 4 verified claims → coverage complete → compiled a bespoke world
  (`publicly_release_model`, date-based declarative terminal) → resolved from
  trajectories. 12 calls, 28 139 tokens, 0 retries, 0 failures, ~6.2 s average latency.

## Remaining limitations

* The coverage gate's materiality rules are deterministic keyword lexicons; the LLM
  exclusion challenge only reviews the consequential kinds and is capped at 12 calls per
  run, so an unusual material item could still be classified as immaterial without
  challenge.
* A live compilation can legitimately produce a world with **no** actor participation
  (for example a pure state-check world whose terminal reads a date). The forecast still
  comes only from executed branches, but such a world exercises no actor cognition.
* `node.rounds` bounds cascades per node; a genuinely long interaction must be compiled
  as multiple process nodes or a larger budget.
* Branch weights for uncertain futures remain epistemic (symmetric-ignorance /
  explicit-model); the reported bounds expose that sensitivity.
* The offline `DeterministicGateway` is a calibrated stand-in used only by tests and the
  corpus fixtures; it is not a frontier model and is never used on the live path.
