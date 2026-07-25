# Causal research policy

Research is driven by what the compiled world will need, not by a fixed number of
searches. It plans from the question, issues queries in small purposeful rounds, and
stops when it stops learning — or continues when a critical causal component is still
missing and a provider can plausibly supply it.

## Rounds, not batches

A round issues this round's queries across the channels (authoritative-first, with a
reserved share), fetches, extracts and verifies, detects contradictions, then generates
follow-up queries aimed at what is still missing. Saturation — two consecutive rounds
that add no usable claim — stops it. There is no fixed 12×12 structure and no
full-budget rerun for every repair.

## Cumulative budgets

Budgets are ceilings for the *whole question*, not per pass: `total_queries`,
`total_fetches`, `total_extract_calls`, `total_seconds` in `ResearchBudget`. Repair used
to receive a fresh reduced budget each time it ran, so a question that repaired six times
could spend six opening allowances and still be "inside the budget" — a live OPEC+ run
was killed from outside at forty minutes having issued 26 logical queries. Research now
stops when a question has cost what a question may cost, wherever in the run that happens.

## Targeted repair, not re-survey

When a gate refuses, `repair.plan_repair` turns the machine-readable failure code into
specific queries and a specific compiler instruction — a missing office-holder sends the
researcher after rosters, a missing production pathway after capacity and reporting
definitions. A compiler contradicting itself needs no research, only a reconciliation
instruction. The loop continues while each attempt changes the diagnosis or adds claims;
when an attempt does neither, the refusal is real rather than an artifact of the budget.

## Abstention over guessing

If a critical component cannot be grounded after official discovery, decoder recovery,
Jina Search and Serper, the run abstains and names the exact missing evidence. It does
not guess because a budget ended.

## What is recorded

Per run: logical queries, provider requests by provider, URLs discovered, publisher URLs
decoded, Reader/Search/Serper calls, pages accepted and rejected, verified/inferred/
hypothetical/rejected claim counts, contradictions, network requests, wall time. In
`research_trace.json` and summarized in `metrics.json`.
