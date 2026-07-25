# Known limitations

Recorded because they were found and *not* fixed, and a reader should not have to
rediscover them. Each was reproduced; none is speculative.

## Research and verification

**A quotation with no sentence punctuation near it is one block.** A quantity must come
from the sentence that was quoted or an adjacent line (`source_extract._value_region`).
In a document with no `.`/`!`/`?` near the quote, that is the whole run-on block, so a
number wedged into it is reachable. The subject and proper-noun checks are the remaining
defence. This bounds ordinary sloppiness, which is the realistic failure; it is not a
defence against a document written to defeat it.

**`_ISO_DATE` matches inside hyphenated identifiers.** A string like `ref-2026-01-17-a`
yields a date. It can only ever make date verification *stricter* (the claim must then
find that date in its span), so it is a cosmetic imprecision rather than a hole.

**Mononym classification is claim-dependent.** Whether `Tesla` is recognised as an
organization depends on whether the claims about it use agency language near the name
(`coverage._acts_in`). The same name can therefore classify differently in two runs with
different evidence. Recognition can only ever *add* an organization candidate, never a
person, so it cannot manufacture a participant the reality gate then demands.

**Some domains block this network outright.** `consilium.europa.eu` and `opec.org`
returned HTTP 403 to every request, with a browser user-agent as well as ours. That is an
external constraint and it is recorded per-URL in `diagnosis.json → fetching`; it is not
worked around.

## Budgets

**There is no global research clock across repair rounds.** Each targeted follow-up runs
under a reduced budget (`live_research._followup_budget`), but the total across many
repairs is bounded only by the outer `timeout`. A pathological question can spend a long
time in repair before refusing.

## Runtime

**Correspondence at a single instant can still exhaust the no-progress guard.** When
compiled delivery and notice delays are zero, an exchange of messages happens at one
timestamp. Bounded exchanges drain and pass; a self-sustaining one is stopped and
reported unresolved, which is correct — but the underlying cause is that the compiler did
not give the messages realistic delays, and nothing currently pushes it to.
