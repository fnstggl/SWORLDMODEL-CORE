# ACTOR-FIDELITY ACCEPTANCE SUITE — preregistered (EVAL-1, sealed per D8)

Status: DRAFT-FROZEN pending the user's veto window announced in-session; after that,
FROZEN. Wording, resolution rules, cutoffs, horizons, scoring, and outcome
commitments below may not change after results are seen. Frozen evidence stores are
built archive-only from pre-cutoff captures and content-hashed at freeze
(`store_manifest.json` per case). Known outcomes are NOT in this document:
implementation, compiler, runtime, and auditing agents must not read them. Outcomes
live in `artifacts/sealed/outcomes.json` (gitignored; sha256
`21ca6d6209a1df3b5a70042fe51220aee0cf01379c651c7d07e3fbe98d03cafb`), readable ONLY by
the forecast-integrity agent after a case's run directory is sealed. Each case's
outcome commitment is sha256(salt|case_id|outcome).

Honest caveat (recorded in MASTER_EXECUTION_PLAN D8): these are famous events; model
weights know them. The mechanical seal is cutoff-enforced stores, the universality
invariants (no case-specific code paths), and this preregistration — not model
amnesia. Scoring therefore weighs process evidence (actor invocation, communication
chains, temporal realism, responsibility classification) alongside outcome accuracy.

## Cases (8 core + 2 operational controls)

| ID | Question (verbatim) | Cutoff (as_of) | Horizon | Commitment |
| --- | --- | --- | --- | --- |
| IND-1 | Will Jerome Powell publicly signal in his August 2024 Jackson Hole speech that the Federal Reserve expects to begin cutting interest rates? | 2024-08-10T00:00:00Z | 2024-08-24T23:59:59Z | ff1fa858… |
| IND-2 | Will Joe Biden withdraw from the 2024 United States presidential race before August 1, 2024? | 2024-07-10T00:00:00Z | 2024-08-01T23:59:59Z | e30dc625… |
| INST-1 | Will the FOMC's September 2024 rate decision be a cut of exactly 25 basis points? | 2024-09-05T00:00:00Z | 2024-09-19T23:59:59Z | 300a754e… |
| INST-2 | Will the Bank of Japan raise its policy interest rate at its July 30–31, 2024 monetary policy meeting? | 2024-07-15T00:00:00Z | 2024-08-01T23:59:59Z | 7f417cf0… |
| NEG-1 | Will the United States and Russia complete a prisoner exchange that frees Evan Gershkovich before September 1, 2024? | 2024-07-15T00:00:00Z | 2024-09-01T23:59:59Z | 20a42f47… |
| ORG-1 | Will Sam Altman be reinstated as CEO of OpenAI within ten days of his November 17, 2023 removal? | 2023-11-18T12:00:00Z | 2023-11-27T23:59:59Z | 03fc309e… |
| COA-1 | Will Emmanuel Macron appoint a Prime Minister from the New Popular Front coalition before September 10, 2024? | 2024-07-20T00:00:00Z | 2024-09-10T23:59:59Z | 4bdd27fc… |
| MIX-1 | Will the Port of Baltimore's Fort McHenry federal channel be restored to its full original width and depth before June 15, 2024? | 2024-05-01T00:00:00Z | 2024-06-15T23:59:59Z | e4cb46fe… |
| OPC-1 | Will the U.S. TSA screen more than three million air passengers in a single day before July 10, 2024? | 2024-06-20T00:00:00Z | 2024-07-10T23:59:59Z | 1cb7c9b6… |
| OPC-2 | Will the fourth Bitcoin halving occur before April 25, 2024? | 2024-03-15T00:00:00Z | 2024-04-25T23:59:59Z | 9a9e3c97… |

Resolution rules: each question resolves YES iff authoritative contemporaneous
record establishes the named event within the window under the plain reading; the
resolution-contract stage must surface and adjudicate any wording ambiguity BEFORE
simulation (CWF-7), and an adjudication note is part of the frozen contract.

## Composition audit (against §6 of the directive)

Individual decision/communication: IND-1, IND-2 (≥2 ✓). Formal multi-member
institutional: INST-1, INST-2 (≥2 ✓). Negotiation with position-changing
communication: NEG-1 (✓). Organizational decision with several internal roles: ORG-1
(✓). Coalition/political/diplomatic: COA-1 (✓). Mixed actor+operational: MIX-1 (✓).
Genuine actor behavior required: IND-1, IND-2, INST-1, INST-2, NEG-1, ORG-1, COA-1 —
7 of 8 (≥6 ✓). Multi-actor interaction: IND-2, INST-1, INST-2, NEG-1, ORG-1, COA-1
(≥4 ✓). Respond to in-simulation information: IND-2 (party pressure events), ORG-1
(employee revolt), NEG-1 (in-window conviction) (≥2 ✓). Rejected/failed action →
reconsideration: ORG-1 (interim-CEO attempts fail; board reverses) (≥1 ✓).
Communication chain sender→delivery→notice→interpretation→response→consequence:
ORG-1, NEG-1 (≥1 ✓). Pure operational controls where zero actors may be legitimate:
OPC-1, OPC-2 (decorative-actor detection ✓).

## Preregistered scoring

Per case, semantic mode, frozen store, two same-commit repetitions:

1. **Publishability**: responsibility classification (only ACTOR_CAUSED /
   PROCESS_CAUSED / ACTOR_AND_PROCESS_CAUSED / FACTUALLY_RESOLVED publish).
2. **Outcome accuracy**: Brier score on published calibrated point estimates.
   Bounds-only (uncalibrated) results are scored separately as honest abstentions:
   report whether the sealed outcome lies inside the published bounds.
3. **Process fidelity** (actor-heavy cases): actors invoked ≥1 with cause; ≥1
   communication chain where composition requires it; temporal report has >1
   distinct simulation timestamp; no decorative-actor or one-step-operational
   reviewer flags.
4. **Control honesty** (OPC cases): zero decorative actors; operational progression
   with intermediate transitions (OPC-1) / deterministic process (OPC-2).
5. **Repeatability**: structural agreement between the two repetitions reported;
   disagreement surfaced, never averaged away.
6. **Leakage**: any post-cutoff claim in a store, or any access to
   `artifacts/sealed/` by a non-evaluation agent, invalidates the case (EVAL-3);
   invalidated cases are never replaced retroactively or counted.

Store freeze procedure: archive-only pastcast retrieval at the stated cutoff; store
content-hash + question/rule/cutoff/horizon recorded in the per-case
`store_manifest.json`; hashes appended to this document in ONE freeze commit before
any scoring run.
