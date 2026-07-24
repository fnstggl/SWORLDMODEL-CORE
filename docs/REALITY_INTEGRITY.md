# Reality integrity

> The simulation must conform to reality. Reality must never be altered to make the
> simulation easier to execute.

The reality-integrity gate (`reality.verify_reality`) runs **before** any behavioral
rollout and **raises** rather than repairs. A structurally false world is refused,
not rewritten into a different question.

## What is verified

`verify_reality(contract, evidence_view, actors, institution)` checks, in order:

1. **Seat count.** `contract.expected_voting_seats` (from the authoritative roster
   evidence) must equal the number of voting actors actually represented. A nine-seat
   board represented with five modeled units fails:

   ```
   WorldIntegrityError:
   expected voting seats: 9
   verified and represented seats: 5
   missing seats: 4
   ```

2. **Rule/roster agreement.** `decision_rule.total_seats` must equal the represented
   roster — a compressed roster cannot implicitly rescale the denominator.

3. **Threshold consistency.** The threshold must follow from the verified rule
   (`majority` ⇒ `floor(n/2)+1`, `unanimous` ⇒ `n`, …). A 26-of-50 threshold can never
   be rescaled to fit a smaller modeled roster.

4. **Vote powers.** Each seat's represented `vote_power` must match the institution's
   verified power. No silent reweighting.

5. **No duplicated seat.** The same underlying person may not occupy two seats.

6. **Required reality facts.** Every load-bearing fact (roster, prior votes, rule,
   current state, guidance, terminal date) must be satisfied by evidence available at
   the cutoff. A structurally unknown roster blocks rollout.

7. **Decisive contradictions.** Unresolved contradictory load-bearing claims block
   rollout (`EvidenceError`) — they are stored, not silently reconciled.

Only when every check passes is the verdict `VERIFIED`.

## Immutable contract

`ResolutionContract` is a frozen dataclass. Its load-bearing fields (question,
`as_of`, `horizon`, outcome space, target, decision body, units, terminal predicate,
decision rule, expected seats) are locked. The only permitted mutation is recording
which required facts were satisfied (`with_satisfied_facts`). Any downstream attempt
to change a locked field via `checked_replace` raises `ContractMutationError`.

## No actor-count budget removes seats

There is no roster-compression or actor-cap path in the core. A world that represents
fewer seats than the contract expects fails at check (1). Tests:
`tests/invariants/test_reality.py`.

## Tests

| invariant | test |
|---|---|
| nine-seat board compressed fails | `test_nine_seat_board_represented_as_fewer_fails` |
| missing member fails | `test_missing_member_fails` |
| duplicated seat fails | `test_duplicated_seat_fails` |
| incorrect vote weights fail | `test_incorrect_vote_weights_fail` |
| threshold inconsistent fails | `test_threshold_inconsistent_with_rule_fails` |
| contract mutation refused | `test_contract_load_bearing_fields_are_immutable` |
| actor budget can't drop seats | `test_actor_budget_cannot_remove_voting_seats` |
| unknown roster blocks rollout | `test_structurally_unknown_roster_blocks_rollout` |
