# Failure-stage classification — every observed acceptance failure, by earliest failing stage

Scope: every distinct failure observed across the frozen acceptance runs at `ff5e9de`,
`590616d`, `a11fd55`, `438fc6a` (5 cases each), plus live probes. Stages:
research → semantic world design → review → semantic validation → lowering →
schedule viability → runtime → terminal evaluation → persistence. The current direct
compiler fuses "semantic world design" and "executable authoring" into one LLM call;
failures are attributed to the half that actually went wrong.

## Research (retrieval / claim verification)

| Failure | Cases | Status |
|---|---|---|
| IncompleteRead/reset destroyed the run | EU-Mercosur (pre-a11fd55) | FIXED (http.py containment) |
| News channel resolved 0 links | all (pre-a11fd55) | FIXED (decoder + provider chain) |
| False decisive contradiction recorded, unclearable by recompile | OPEC+ (a11fd55) | FIXED (adversarial reconciliation, 438fc6a) |
| Claim starvation → cannot ground the board / parties | Banxico 438fc6a (1–6 claims), EU-Mercosur (4–12 claims) | OPEN — research-stage, not compiler |

## Semantic world design (the "understand reality" half of the fused call)

| Failure | Cases | Status |
|---|---|---|
| Announcement modeled instead of production; total copied from an exogenous draw | Tesla (a11fd55 launder) | Gate added (`terminal_laundered_from_uncertainty`); compiler still cannot *build* the production world (438fc6a emptied it) |
| "Already achieved" asserted rather than cited | OPEC+ 438fc6a (`terminal_preresolved_without_evidence`), BoE rationale a11fd55 | Gate + unified repair added (103dce6); untested |
| Empty world / no producer at all | EU-Mercosur, Tesla 438fc6a, Banxico 438fc6a | OPEN — design half collapses under repair pressure |
| Wrong scale (7 countries → 1 participant) | OPEC+ (pre-a11fd55) | FIXED (`represents_count`) |
| Actor with no occasion (inert world) | BoE a11fd55 (`nothing_scheduled`), BoE 438fc6a (`nothing_can_act`) | Repairs improved; still oscillates |

## Executable authoring (the "write a consistent program" half of the fused call)

| Failure | Cases | Status |
|---|---|---|
| Effect keyed `field_id` instead of `field` → action wrote nothing | BoE (pre-a11fd55) | Patched via synonym coercion — symptom of model-authored syntax |
| Malformed compilation JSON / unknown expression operators | multiple probes | Patched via reparse repair — same class |
| Terminal reads a field no effect writes (name ≠ mechanism) | multiple | Gated; recurs as repair thrash |
| Oscillation across encodings of one intent (5-repair BoE thrash: preset → no-action → no-world-state → …) | BoE 438fc6a | Unified repair target added (103dce6); root cause is the fused call re-authoring the whole program each round |
| Reroll exhaustion misread (same code, different world) | Tesla 438fc6a | FIXED (spec-hash signatures, 103dce6) |

## Review / validation / lowering
Do not exist as separate stages in the direct path — their absence is why the failures
above surface only as end-of-pipeline gate refusals after full recompiles.

## Schedule viability
`nothing_scheduled` (BoE a11fd55) is discovered today by a full compile + gate pass;
no cheap pre-check exists.

## Runtime

| Failure | Cases | Status |
|---|---|---|
| Terminal never determined (schedule exhausted, unresolved 1.0) | Banxico a11fd55-r2 | OPEN — partly a design failure (terminal read state the vote never wrote) |
| Ungrounded 50/50 presented as an answer | Banxico, Tesla (a11fd55) | Integrity fields + verifier criterion added |

## Terminal evaluation
Result-equals-initialization (OPEC+ 438fc6a) — now refused at compile (103dce6).

## Persistence
SIGKILL mid-run loses nothing (research checkpoint verified); resume rejects foreign
commits. No open failures.

## Conclusion — what belongs to the compiler boundary

Counting distinct failure kinds: **11 of 16 open-or-recurring kinds sit in the fused
compile call** (5 semantic-design, 6 executable-authoring). Research contributes 1 open
kind (claim starvation), runtime 2 (both downstream of design defects). The compiler
boundary is where the failures live, and the two halves fail differently: the design
half picks wrong worlds under pressure; the authoring half emits inconsistent programs
even when the design is right. That is the hypothesis the semantic-compiler experiment
tests: separate them, let code own every symbol and every piece of syntax, and only the
causal meaning remains for the model to get right or wrong.

The smallest change that addresses this class: a semantic plan schema + one planning
call + one independent review call + mechanical validation + deterministic lowering
into the *existing* WorldSpec, behind a mode flag, A/B-tested against the direct path
on identical frozen evidence. No runtime change, no new repository, no domain
hardcoding.
