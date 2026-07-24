# Evidence-to-world coverage integrity

> Research is worthless if the facts it finds are silently lost when the world is
> compiled. Every materially relevant verified item must be represented **and causally
> wired** into the simulated world, or explicitly excluded with a recorded,
> evidence-grounded reason.

Reality integrity (`docs/REALITY_INTEGRITY.md`) checks that the world the compiler
*produced* is internally faithful. Coverage integrity checks the step before that: it
compares the world the compiler produced against **everything the verified evidence
implied it should contain**, so nothing important can disappear between research and
simulation.

## The failure it prevents

A committee's evidence store held all five voting members, but the LLM compiler
represented only two — the other three were attested by fewer claims and were
overlooked. Worse, the compiler *also* under-counted the seat total to two, so the
reality-integrity seat check (`represented == expected`) passed vacuously. An
incomplete world was simulated as if it were complete. The bug is general: a person, an
organization, a binding rule, a scheduled event, a document, a channel, a resource, or
a relationship can all be dropped the same way.

## The pipeline (canonical, not a diagnostic)

```
verified evidence
  → deterministic candidate inventory        build_candidate_inventory()
  → the inventory is handed to the LLM        evidence_checklist() → compile prompts
  → LLM world compilation
  → deterministic coverage comparison         assess_coverage()
  → targeted research / compilation repair    _compile_with_repair()
  → world-integrity gate                      enforce_coverage() + verify_reality()
  → simulation
```

The gate runs inside `compiler.compile_world` on every process type. The
`CompilationCoverageReport` is stored on the `CompiledWorld` and written to the trace
(`coverage_report.json`) and the live audit.

## 1. Deterministic candidate inventory

`coverage.build_candidate_inventory(view, contract)` walks the **whole** verified
evidence store and extracts `EvidenceCandidate`s, each preserving its `claim_ids` and
`lineage_ids`:

- **Direct** candidates named in evidence: people, organizations, population groups,
  rules, documents, channels, resources, variables, scheduled events, prior actions.
- **Derived** structural candidates inferred from combinations of claims (memberships,
  resolution requirements). These are marked `is_inference=True` and cite the claims
  they were inferred from — never presented as directly observed facts.

Claims about the same entity **merge** into one canonical candidate (one entity → one
world object), carrying every contributing claim id. The inventory is deterministic:
the same evidence always yields the same candidates.

The inventory is not just used to *check* the compiler — it is **handed to** the
compiler. `evidence_checklist()` renders the material candidates into the
`compile_reality` / `compile_roster` prompts, so the model is given an explicit list of
what verified reality contains instead of being trusted to recall it.

## 2. Materiality

A candidate is **material** when omitting it could change the outcome, the causal
process, an actor's information or actions, authority/feasibility, a resource
constraint, an uncertainty branch, the timing of events, or the terminal condition.

Materiality is decided by deterministic code (conservative, so it blocks on genuine
loss without false refusals) and may be tightened by the LLM. Examples: a person is
material when spoken about with a role/vote verb; the decision-body organization is
material; a rule is material when it names a decision procedure (majority / unanimity /
quorum) or the frame keys on it; a scheduled event is material when it falls inside the
forecast window. Conflicting evidence is always material — it must be resolved, never
dropped.

## 3. Coverage comparison and the causal-use invariant

`assess_coverage(inventory, world_spec)` assigns exactly one disposition to every
candidate — `INCLUDED`, `EXCLUDED_IRRELEVANT`, `MERGED`, `UNCERTAIN`, or
`REQUIRED_BUT_UNRESOLVED`. **No candidate disappears without a disposition.**

Representation alone is not coverage. A candidate is `INCLUDED` only if it is
represented **and causally wired**: an actor that actually votes, a rule the terminal
tally uses, a document whose claims an actor can perceive. A person present in the
`WorldSpecification` but absent from the meeting — stored but never voting — is treated
as **missing**, exactly as if it were never represented. Each `INCLUDED` disposition
records which compiled objects represent the candidate and which causal uses it has.

## 4. Exclusion challenge

The compiler may not drop a candidate merely by labelling it irrelevant. For the
consequential kinds, an **independent** review pass (`exclusion_reviewer`, a live-LLM
call separate from the compiler) is asked whether omitting the item could plausibly
change any actor's knowledge, authority, feasible actions, a constraint, a branch,
timing, or the outcome. If it says yes, the exclusion is invalid: the candidate becomes
`UNCERTAIN` and blocks. Disagreement never resolves in favour of dropping the item.

## 5. Repair, then refuse

When the gate finds a material candidate missing, `run_forecast` runs **targeted
follow-up research** keyed on exactly the missing items and recompiles, up to a bounded
number of attempts (`LiveResearchBackend.augment_for_coverage`). If the missing items
still cannot be represented, `enforce_coverage` raises `WorldIntegrityError` and
simulation is refused. Simulation never continues with a world known to be incomplete.

## Acceptance tests

`tests/invariants/test_coverage.py` locks the ten cases: five members but two compiled
→ rejected; an organization with no named individual → still covered; an omitted
binding rule → blocked; an omitted in-window event → blocked; document accessibility;
an incidental person → excluded with a reason; same-entity claims merged; conflicting
evidence → unresolved (not a silent choice); every process type; and no candidate ever
silently dropped. Plus the repair loop (recovers via augmentation; still refuses when
augmentation cannot close the gap) and the exclusion challenge.
