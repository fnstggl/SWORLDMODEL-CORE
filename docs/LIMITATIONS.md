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

**A multi-token capitalized name that is a place is still classified as a person.**
Every token being capitalized separates "EU member states" from "Andrew Bailey", and a
head noun that is an instrument separates "Mercosur Agreement" from both. Neither
separates "Saudi Arabia". The remaining defences are that the role word must sit next to
the name before a seat is demanded (`coverage._near`), and that a name represented
through an organization it belongs to is covered rather than absent. A place named in
role terms right next to the name would still be owed a seat.

**The search channel degrades under repeated use, and the run cannot route around it.**
Four full acceptance passes inside three hours, each making roughly five hundred HTTP
requests, ended with most queries returning "empty result set, block, or challenge": one
case saw four URLs from five queries and compiled an empty world. It is recorded per
query in `diagnosis.discovery.search_failures` and now names itself as the root cause
rather than being reported as an actor-discovery failure, but nothing retries against a
different channel or backs off, so a throttled window is a window in which this system
cannot answer.

**Some domains block this network outright.** `consilium.europa.eu` and `opec.org`
returned HTTP 403 to every request, with a browser user-agent as well as ours. That is an
external constraint and it is recorded per-URL in `diagnosis.json → fetching`; it is not
worked around.

## Budgets

**The compile-and-repair ceiling is a wall-clock number, not a measure of progress.**
Repair stops when it stops getting somewhere, which is the right rule and has no time in
it; `ForecastConfig.max_compile_seconds` exists only because a question whose compiler
keeps producing genuinely different worlds could otherwise repair for as long as research
would feed it, and a live OPEC+ run was killed from outside at forty minutes. Reaching
the ceiling refuses with the last gate's own diagnosis, so the record says what was still
missing — but a question that would have converged on the next round is refused
identically to one that never would.

## Compilation

**Arithmetic on an undetermined operand makes the whole expression undetermined, and
the branch carries that as unresolved.** That is the honest outcome and it is also
indistinguishable, in the result, from a world whose process never fired. The
diagnosis separates them — `terminal_never_determined` names the terms actually
written — but the forecast line does not.

**Evidence as a producer is only as good as the citation the compiler attaches.** A
terminal term whose initial value cites claims that do not in fact establish it will
pass the producer gate. Claim ids are checked for existence against the evidence store,
not for whether the claim supports the value. The pre-rollout review is the check that
reads for support, and it is a model judgement, not a mechanical one.

## Runtime

**Correspondence at a single instant can still exhaust the no-progress guard.** When
compiled delivery and notice delays are zero, an exchange of messages happens at one
timestamp. Bounded exchanges drain and pass; a self-sustaining one is stopped and
reported unresolved, which is correct — but the underlying cause is that the compiler did
not give the messages realistic delays, and nothing currently pushes it to.
