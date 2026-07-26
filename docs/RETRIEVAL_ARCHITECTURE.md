# Retrieval architecture

Every source enters the one canonical evidence pipeline — fetch, extract, verify against
the fetched bytes, cutoff-check — regardless of which provider discovered it. A provider
finds URLs; it never supplies evidence. The chain is ordered by directness and cost, and
each rung exists because the one above it has a failure mode observed in a live run.

## Discovery, per logical query

1. **Official feeds and direct publisher fetch.** Official-domain and
   planned-authoritative-source queries run first and hold a reserved share of the query
   budget, so a broad question cannot spend its whole allowance on general search before
   reaching a primary source (`live_research._next_queries`, `rss.feed_urls_for`).

2. **Google News RSS + the decoder.** RSS gives headlines, publishers and publication
   times for exactly this query. Every item link is an opaque
   `news.google.com/rss/articles/CBMi…` reference — the id encodes a server-side
   reference, so the old base64 recovery returns nothing and following the link yields a
   JavaScript interstitial. `gnews_decode.GoogleNewsDecoder` runs the `batchexecute`
   flow the Google News web app uses: read `data-n-a-sg`/`data-n-a-ts` off the article
   page, POST a `garturlreq` envelope, read the publisher URL from the reply. Verified
   live: 12/12 real links resolved (Reuters, the Guardian, Electrek). Failures are typed
   — `challenge`, `throttled`, `protocol_change`, `network` — and never invent a URL.

3. **Jina Reader** (`r.jina.ai`) reads a *known* publisher URL that direct fetch could
   not — blocked, 401/403, empty, JavaScript-only, a hard PDF, truncated. Nowcast only:
   Reader reads the live page, which cannot stand in for the archived pre-cutoff content
   a pastcast requires (`live_research._reader_fallback`).

4. **Jina Search** (`s.jina.ai`) is used narrowly for exact-title recovery when a Google
   News link will not decode: `"EXACT HEADLINE" "PUBLISHER"`, both known from the RSS
   item. Never a subject query — Search costs far more than Reader.

5. **Serper** (`google.serper.dev`) is the last discovery fallback, used only when a
   round's free channels produced nothing at all. Every result still goes through the
   canonical pipeline.

6. **DuckDuckGo** remains an optional low-priority extra. Its blocks were the direct
   cause of an empty-evidence pass, so it is no longer load-bearing: a block causes
   provider fallback, not evidence starvation.

## Provider health and circuit breaking

`providers.ProviderHealth` scores every provider call — successes, empties, blocks,
timeouts, errors, consecutive failures. Three consecutive clear failures rest a provider
for the rest of the run, so a blocked engine is not re-asked dozens of times. The ledger
is in the trace (`provider_health`, `provider_requests`).

## Robust network handling

A protocol-level failure of one document — `IncompleteRead` on a truncated chunked body,
a malformed status line, a corrupt gzip stream — is caught in `http.py` and returned as a
document-level `HttpError` with the received byte count preserved. It rejects that page
and the research session continues. Before this, `IncompleteRead` (not an `OSError`)
escaped and destroyed a twenty-minute run.

## Secrets

`JINA_API_KEY`, `SERPER_API_KEY`, `DEEPSEEK_API_KEY` are read from the environment and
never logged, traced, persisted or committed. Authorization headers are redacted from
diagnostics; a test asserts no key value reaches a result or the call log.
