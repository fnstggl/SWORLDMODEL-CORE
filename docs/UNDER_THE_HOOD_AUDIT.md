# Under-the-hood audit

For every acceptance case, the run writes a machine-readable account of exactly what it
did, so a reader can reconstruct and challenge it without re-running anything. This
document explains where each required item lives; the numbers themselves are in the
per-case artifacts and `metrics.json`, generated from those artifacts.

## Where each item is recorded

| item | artifact |
| --- | --- |
| question, cutoff, horizon, resolution contract | `diagnosis.json → research_planning`; `world_manifest.json` |
| retrieval mode, decided before research | `run.log` first line; `diagnosis.json → research_planning.retrieval_mode` |
| every logical query and its channel | `research_trace.json → queries` |
| provider requests by provider | `research_trace.json → provider_requests`, `provider_health` |
| Google News decoding results | `research_trace.json → rss_requests` (decoded / recovered_via_title_search) |
| direct fetch and Jina Reader outcomes | `research_trace.json → sources_fetched[].via`, `sources_rejected` |
| useful and rejected sources | `research_trace.json → sources_fetched`, `sources_rejected` |
| verified / inferred / hypothetical / unsupported claims | `diagnosis.json → epistemic_classification`; `evidence_store.json` |
| full compiled world (people, orgs, populations, processes, actions, channels, events, uncertainties, terminal) | `compiled_world.json`; `world_manifest.json` |
| inclusion and exclusion reasons | `compiled_world.json → structure_rationale`; `world_review.json` |
| every starting possible world and its weighting basis | `forecast.json → branches`; `structural_uncertainty.json` |
| every actor call (time, wake reason, visible info, memory, plan, decision, validation, consequence) | `actor_decisions.jsonl` |
| every process occurrence and world-state change | `event_ledger.jsonl` |
| complete time transitions | `event_ledger.jsonl` (chronological per branch) |
| terminal producer lineage | `diagnosis.json → runtime.terminal_producer_lineage` |
| resolved YES / NO / unresolved mass | `forecast.json`; `diagnosis.json → runtime` |
| aggregation arithmetic | `forecast.json`; `docs/FORECAST_INTEGRITY.md` |
| probability before vs after simulation | `diagnosis.json → forecast_integrity` |
| counterfactual if actor/process outputs removed | `diagnosis.json → forecast_integrity.counterfactual_note` |
| pre- and post-simulation auditor findings | `world_review.json`; `trajectory_audit.json` |
| final case classification | `trajectory_audit.json → classification` |

## Reading a case

`scripts/audit.py <case>` (in the scratchpad) prints the chronological walk; the trace
directory holds the byte-exact prompts (`actor_decisions.jsonl` carries each actor's
`exact_prompt` and `provider_response`) and every model call (`llm_calls.jsonl`). A
refusal writes the same stage-by-stage `diagnosis.json` as a completion, so a run that
stopped is as readable as one that finished.

<!-- SUMMARY: filled from the frozen run -->
