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

**Both discovery channels can be externally unavailable at once, and then nothing can be
researched.** Probed directly at the end of this run:

* `lite.duckduckgo.com` — the web-search channel, used for both general and `site:`
  queries — returned `Connection reset by peer` on one query and "empty result set, block,
  or challenge" on the rest. Four acceptance passes inside three hours, each making around
  five hundred HTTP requests, is enough to earn that.
* Google News RSS answered normally: 54–100 items per query. Every one of them carried an
  opaque `CBMi…` article id, which encodes a server-side reference rather than the
  destination, so `rss.resolve_item_url` recovered **zero** publisher URLs. Following the
  redirect returns a 581 KB JavaScript interstitial with no publisher link in it. That is
  a permanent change on Google's side, not a throttle.

The system reports this correctly — `diagnosis.discovery.search_failures` per query,
`rss_requests.unresolvable_redirects` per feed, and a `discovery_failure` root cause when
half or more of the searches come back empty — but it cannot route around it. A window in
which both channels are unavailable is a window in which this system cannot answer, and
the runs in it refuse for the right reason with an empty evidence store.

A recovery path was built and then removed: the RSS items still name their publisher in
`<source url="...">`, and a publisher's own feed does list real article URLs. Measured, it
recovered 3–6 candidates per query where there had been none, but a publisher's feed is
what it published lately rather than an answer to the query, so requiring a headline match
returned nothing for three of four questions and false positives for the fourth — "2026 Q2
results earnings call" shares words with a Tesla deliveries headline. It was not worth the
fetch and extraction budget it spent, so it is not in the tree.

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

## Retrieval (this run)

**Jina Reader and Search require a funded key.** The key present returns HTTP 402
`InsufficientBalanceError` — it exists but has zero token balance — so the Reader fetch
fallback and the exact-title recovery rung are wired, tested against fixtures, and trip
cleanly under the health ledger, but did not fire against live sources in this run. When
the balance is funded they activate with no code change.

**The Google News decoder depends on an unofficial endpoint.** The `batchexecute` flow is
the one Google's own web app uses, but it is not a documented API. A protocol change
surfaces as a typed `protocol_change` failure rather than a wrong URL, and the run falls
back to title search and Serper — but a decoder that stops working is a discovery channel
lost until the flow is updated.

**Serper and DeepSeek are paid and external.** A funded account and a reachable endpoint
are preconditions this system cannot create. An outage of either is reported as an
external blocker with all work preserved, never as a simulation failure — but it still
means the case does not complete.
