# The live product path (question → simulation)

The production command needs only a question:

```bash
sworldmodel forecast \
  --question "Will [event] happen by [date]?" \
  --as-of  "2026-05-14T23:59:59-04:00" \
  --horizon "2026-06-25T23:59:59-04:00"
```

No prepared corpus, roster, decision rule, protocol, reaction thresholds,
uncertainty variables, branch probabilities, memory seeds, source URLs, or terminal
configuration is supplied. The system builds all of it from live research + a live
LLM, then simulates and aggregates.

```
question
 → research_planner.plan_research        (LLM: backward research plan, queries, domains)
 → live_research.LiveResearchBackend      (iterative)
      DuckDuckGo + site: queries -> real URLs;  Google News RSS as a discovery signal
      source_fetch: fetch pages, extract text + publication date, reject blocked/binary
      source_extract: LLM extracts claims; a claim survives only if its excerpt is in
                      the fetched text (verification); lineage groups identical facts;
                      the cutoff excludes anything available after as_of
      loop with followup_queries until required facts are covered or the budget ends
 → universal_compiler.compile_reality      (LLM: roster, rule, terminal, actors, facts)
 → universal_compiler.compile_frame        (LLM: options, signals, reaction rules,
                                            guidance, genuine uncertainties + provenance)
 → reality.verify_reality                  (refuse a structurally false world)
 → compiler.compile_world                  (actors with conditional behavior; protocol)
 → runtime.run                             (live DeepSeek actors; environment executes)
 → mechanisms                              (deterministic tally / action terminal)
 → outcomes.aggregate                      (weighted YES trajectories)
 → tracing + live_audit.json               (auditable; real calls, tokens, latencies)
```

The corpus path (`--corpus`, the deterministic reasoner) and the fake-LLM tests
exercise the *same* compiler + runtime + aggregation; only the two front-ends
(research + gateway) differ.

## Live DeepSeek gateway (`deepseek_gateway.py`)

Implements `ModelGateway` over an injectable `HttpTransport`. Config from the
environment: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`),
`DEEPSEEK_MODEL` (default `deepseek-v4-flash`). Structured JSON output with schema
validation and bounded deterministic repair, request timeouts, exponential backoff on
transient failures (429/5xx/transport), real token/latency accounting from the API
`usage`, per-stage call counts, prompt hashes, raw-response preservation, and
per-task temperatures (low for extraction/compilation, higher for actor cognition).
A repeated provider failure raises `GatewayError` → the affected branch mass is left
**unresolved**, never a default vote or a prior.

The gateway is used for: research planning, evidence extraction, contradiction
analysis, resolution/world/uncertainty compilation, actor construction, actor
planning/reaction, communication, and reflection. It is **not** used to count votes,
apply thresholds, move the clock, mutate the world, fabricate sources, rewrite
verified reality, or write terminal outcomes — those are code.

## Live research (`live_research.py`, `search.py`, `rss.py`, `source_fetch.py`, `source_extract.py`)

- **Discovery.** DuckDuckGo (lite) yields real, fetchable publisher/official URLs via
  its `uddg` redirect parameter; `site:` queries reach official domains directly.
  Google News RSS is recorded as a discovery signal but its links are obfuscated
  redirects, so they are not fetched.
- **Verification.** Each page is fetched and its text extracted; blocked/binary/403
  pages are rejected. A claim is kept only if its supporting excerpt is actually
  present in the fetched text. A generated URL is never trusted on its own.
- **Cutoff.** A claim's `available_at` is its source's publication date (or, if
  undated, the retrieval time — which for a pastcast conservatively excludes it). The
  post-cutoff outcome is present in the store but excluded from every pre-outcome view.
- **Lineage.** Identical facts across articles share one lineage id — repeated
  reporting of one event is not counted as independent samples.
- **Iteration + budget.** The loop issues follow-up queries for still-missing required
  facts and stops on saturation or a configurable budget (rounds, queries, fetches,
  extract calls, seconds).

## Universal compiler (`universal_compiler.py`)

Two LLM calls compile the world from the cutoff-bounded evidence view: `compile_reality`
(decision body, rule, terminal, roster/actors, required facts) and `compile_frame`
(options, signals, evidence-grounded reaction rules, guidance, and only
outcome-sensitive uncertainties with provenance). Deterministic validation then
citation-checks every id against the store, normalizes per-variable weights, and
independently records `expected_voting_seats` from the body's evidence — so a roster
the LLM cannot fully name fails the integrity gate rather than being quietly shrunk.

## Generality (one runtime, many process types)

The terminal is dispatched by mechanism:

- `committee_vote` — unanimous / at-least-k / majority / **weighted-majority** for an
  option. Covers boards and, with strata as weighted voters, population questions.
- `actor_action` — did a target actor take a target action (respond, commit, act) by
  the horizon, read from the event history. Covers individual-response, negotiation,
  and organizational-action questions.

A generic `general_protocol` (context → actor turns → evaluate) runs the non-committee
cases through the same rollout engine. `evaluation/synthetic/` demonstrates four
distinct process types (small committee, seven-seat committee, weighted population,
individual response) through the one public `forecast()` entry point.

## Live gating

`ForecastConfig.is_live` is true only when the gateway is a live provider **and** the
research backend is live. The `banxico-live` / question-only `forecast` commands refuse
to run "live" otherwise. The deterministic gateway and corpus backend are confined to
tests and the historical-evaluation fixture; the live command never selects them.
