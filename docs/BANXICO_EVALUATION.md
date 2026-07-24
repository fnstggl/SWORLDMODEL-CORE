# Banxico evaluation (vertical slice)

A pastcast of whether the **25 June 2026** Banco de México Governing Board decision
would be a unanimous hold. This is a structural and causal test, not permission to
hardcode Banxico: the core contains no Banxico names, dates, rates, or special
branches. All Banxico facts live in `evaluation/banxico/corpus/corpus.json` and the
expected roster is asserted only in `tests/integration/test_banxico.py`.

- **Information cutoff (`as_of`):** `2026-05-14T23:59:59-06:00`
- **Horizon:** `2026-06-25T13:00:00-06:00`
- The simulator does not receive the June 25 result, post-decision reporting, or the
  July 9 minutes. The June 25 decision claim is in the store but excluded by cutoff.

## Verified reality (from cited, pre-cutoff evidence)

Five-seat Governing Board, decided by majority of five; the question predicate is
unanimity (5-0) for `hold`:

| seat | office | May-7-2026 vote |
|---|---|---|
| Victoria Rodríguez Ceja | Governor (chair) | cut |
| Omar Mejía Castelazo | Deputy Governor | cut |
| José Gabriel Cuadra García | Deputy Governor | cut |
| Galia Borja Gómez | Deputy Governor | hold |
| Jonathan Heath | Deputy Governor | hold |

The May-7 cut coalition (Rodríguez, Mejía, Cuadra) and the two holders (Borja, Heath)
are **researched and cited**, not assumed. The seven May-7 claims (decision, five
votes, guidance) share the lineage id `banxico_2026_05_07_decision` — one event, not
seven independent cases. The guidance ("the easing cycle is ending; maintaining the
rate would be appropriate going forward") makes `hold` the revealed intersection of
all five positions absent a major shock.

## Causal framing

The behavioral question is **not** "which persona does each member inhabit?" It is
"will any member move away from the newly-established common hold position before June
25?" The compiler builds, per member, a current inclination (`hold`, from prior vote +
guidance) and conditional reaction rules; the genuine uncertainty is over future data
(inflation surprise, growth weakness), each an epistemic branch with disclosed
provenance.

## Protocol

`distribute_briefing → introduce_proposal → request_statements → deliver_statements →
revise_proposal → open_decision → cast_votes → tally → publish`. Members state
positions, those statements are delivered, and the decision opens; each of five seats
casts a final vote; deterministic code tallies unanimity.

## Result (deterministic offline reasoner)

Simulation probability **0.736** (`weighted_simulated_trajectories`), fully resolved,
6 branches, 121 model calls. Because all mass resolves, the supported bounds collapse
to the point estimate `[0.736, 0.736]`.

| branch (conditions) | weight | five final votes | outcome |
|---|---|---|---|
| inflation in-line, growth in-line | 0.576 | all hold | **YES** |
| inflation in-line, growth moderate | 0.160 | all hold | **YES** |
| inflation upside, growth in-line | 0.144 | Borja/Heath hold; Rodríguez/Mejía/Cuadra hike | NO |
| inflation in-line, growth severe | 0.064 | Borja/Heath hold; cut coalition cut | NO |
| inflation upside, growth moderate | 0.040 | 2 hold / 3 hike | NO |
| inflation upside, growth severe | 0.016 | 2 hold / 3 hike | NO |

`P(YES) = 0.576 + 0.160 = 0.736`. Unanimity holds when no data surprise crosses a
reaction threshold; a *major* surprise breaks it, and the holders' focal-proposal
acceptance produces genuine 2-3 splits in the shock branches.

## Sealed artifacts

`make banxico` writes, before any outcome is loaded:

```
artifacts/banxico_2026_06_25/pre_outcome_forecast.json
artifacts/banxico_2026_06_25/pre_outcome_report.md      (contains the SHA-256)
artifacts/banxico_2026_06_25/event_ledger.jsonl
artifacts/banxico_2026_06_25/evidence_manifest.json
artifacts/banxico_2026_06_25/world_manifest.json
```

The pre-outcome forecast is hashed (SHA-256, printed in the report and the CLI
summary). `make banxico-eval` reads the sealed file, compares it to the known result
(revealed only post-cutoff), and writes `post_outcome_evaluation.md` **without
modifying the pre-outcome files**. For the run above: known result **YES** (unanimous
hold), **Brier 0.0697**, directional call **correct**.

## Causal audit

- **Did the actors' actual decisions produce the answer?** Yes — the probability is
  the weighted frequency of branches whose five simulated votes met the predicate.
- **Would deleting the actor calls change the answer?** Yes — with actor decisions
  disabled the run is `UNRESOLVED` with no point estimate.
- **Did actors perceive substantive colleague statements?** Yes — statements are
  produced, delivered, and reacted to (event ledger + actor-decision records).
- **Did any separate model override the trajectories?** No — there is one causal
  route; `probability_source == weighted_simulated_trajectories`.
- **Did any prior enter the forecast?** No.
- **Can the result be replayed from the event ledger?** Yes.
- **Is there any Banxico-specific code in the core?** No — enforced by the
  anti-hardcoding grep test.
