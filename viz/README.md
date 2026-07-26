# Replay viewer & run dossier

A read-only visualizer for simulation runs. It shows, for any run the pipeline has
produced, the world that was compiled, every actor and how it was modeled, the exact LLM
call each actor was given and exactly what it answered, the events its decision caused,
and the branch-by-branch trajectory — pausable and steppable one transformation at a time.
A second page, the **dossier**, reconstructs the complete run record: timeline, actors,
communications, processes, world state, model calls, branches & weights, terminal
lineage, cost, and audit (OBS-6).

It reads **only the artifacts a run already wrote**. It imports no `sworldmodel` code and
runs no simulation, so it keeps working across changes to the simulation itself: when the
trace format changes, only the adapters (`replay.py`, `dossier.py`) change and the pages
do not. No mock data, no separately reconstructed simulation, ever: every value on either
page comes from a recorded artifact, and anything a run did not record renders as an
explicit "not recorded by this run" notice — never a fabricated placeholder.

## Run it

```bash
cd /home/user/SWORLDMODEL-CORE
python3 viz/server.py                       # scans ./artifacts, serves 127.0.0.1:8765
python3 viz/server.py --root artifacts/slice92 --forensics artifacts/forensics
```

Open <http://127.0.0.1:8765> (replay) or <http://127.0.0.1:8765/dossier> (dossier). Pick
a run from the dropdown (every trace under `--root`, newest first). On the replay page,
Space plays/pauses; ← / → step one transformation; the right-hand **LOG** opens the full
record of the selected step; the **dossier ⧉** link opens the same run's dossier.

Any future run is viewable the moment it writes a trace directory; no frontend change is
needed.

## The dossier's ten views

1. **Timeline** — merged chronological events + actor invocations per branch; each row
   links to the exact prompt/response, its state diff, and its evidence lineage.
2. **Actors** — per invocation: wake reason, delivered/noticed/missed information,
   feasible actions, the EXACT prompt, the EXACT raw provider response, parsed intent,
   environment validation, and the events the decision applied. Long prompts collapse
   but always expand to the full text.
3. **Communications** — send → deliver → notice chains with times and content, including
   deliveries an actor never noticed.
4. **Processes** — non-actor transitions with inputs, outputs, and evidence.
5. **World state** — per-branch fields over time, scrubbing the recorded state diffs
   (initial state + ordered diffs = the branch, no model call); initial vs final vs the
   published forecast state.
6. **Model calls** — task kind, tokens, latency, timestamps, expandable prompt/response,
   totals. Fields the run did not log (e.g. pre-71393bb prompts) say so per call.
7. **Branches & weights** — weights, provenance labels, conditions, outcomes,
   pre-simulation outcomes, grounded flags, and the exact numeric probability
   substitution from the reconstruction.
8. **Terminal lineage** — claim → semantic object → runtime id → events → terminal,
   plus recomputed-vs-published terminal evaluations.
9. **Cost & timing** — per-stage aggregates from the model-call log.
10. **Audit** — world review, trajectory audit, forensic verdict and responsibility with
    severities; the banner at the top of every dossier shows the verdict, the
    responsibility classification, and the point-estimate calibration.

## What each file does

- `replay.py` — schema-aware adapter for the replay page. `build_replay(trace_dir)`
  normalizes one run's artifacts into `{meta, world, branches, llm_summary, refusal}`;
  `discover_traces(root)` lists every run under a root. Pure stdlib, no simulation imports.
- `dossier.py` — schema-aware adapter for the dossier page. `build_dossier(trace_dir,
  forensics_dir)` assembles the ten views and records, per artifact file, whether it came
  from the run directory, the forensics directory, or is absent.
- `server.py` — stdlib `http.server`. `GET /` replay page; `GET /dossier` dossier page;
  `/api/traces` lists runs; `/api/replay?trace=<dir>` one normalized run;
  `/api/dossier?trace=<dir>[&forensics=<dir>]` one run's dossier. Paths are confined to
  `--root` (forensics paths to `--root`/`--forensics`).
- `index.html` — the replay page (self-contained, no external assets).
- `dossier.html` — the dossier page (self-contained, dark/light aware; wide tables scroll
  in their own containers).

## Artifacts it reads

From the run directory: `forecast.json` (integrity block and per-branch
`pre_outcome`/`weight_grounded` when the run recorded them), `world_manifest.json`,
`structural_uncertainty.json`, `branch_schedule.json`, `event_ledger.jsonl`,
`actor_decisions.jsonl` (each actor's `exact_prompt` and `provider_response` verbatim),
`llm_calls.jsonl`, `evidence_store.json`, `evidence_manifest.json`,
`coverage_report.json`, `world_review.json`, `trajectory_audit.json`,
`compiled_world.json` (newer runs), `run_stamp.json`, `actor_grounding.json`,
`diagnosis.json` / `run_audit.json` (older runs). A refused run has no forecast; it still
shows the world that was built and the gate it stopped at.

Derived observability artifacts are read **from the run directory first** — the
production trace writer is adopting these filenames — and only then from the optional
`--forensics` directory (`scripts/forensics.py` output): `forensic_timeline.jsonl`,
`state_diffs.jsonl`, `communications.jsonl`, `process_transitions.jsonl`,
`branch_weight_history.jsonl`, `semantic_runtime_lineage.jsonl`,
`terminal_evaluations.jsonl`, `probability_reconstruction.json`,
`trajectory_responsibility.json`, `forensic_verdict.json`, `llm_calls_full.jsonl`.
`--forensics` may be one run's forensics directory or a root of per-run directories;
runs are matched by the `run_dir` recorded in `forensic_verdict.json`, then by basename
(default root: `<--root>/forensics` when it exists). The dossier header lists, file by
file, where every artifact actually came from.
