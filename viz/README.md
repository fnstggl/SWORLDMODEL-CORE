# Replay viewer

A read-only visualizer for simulation runs. It shows, for any run the pipeline has
produced, the world that was compiled, every actor and how it was modeled, the exact LLM
call each actor was given and exactly what it answered, the events its decision caused,
and the branch-by-branch trajectory — pausable and steppable one transformation at a time.

It reads **only the artifacts a run already wrote**. It imports no `sworldmodel` code and
runs no simulation, so it keeps working across changes to the simulation itself: when the
trace format changes, only `replay.py` changes and the page does not.

## Run it

```bash
cd /home/user/SWORLDMODEL-CORE
python3 viz/server.py                       # scans ./artifacts, serves 127.0.0.1:8765
python3 viz/server.py --root artifacts/acceptance --port 8765
```

Open <http://127.0.0.1:8765>. Pick a run from the dropdown (every trace under `--root`,
newest first). Space plays/pauses; ← / → step one transformation; the right-hand **LOG**
opens the full record of the selected step — the verbatim prompt and response, the event
payload, the resulting world state, plus tabs for the compiled world, every model call,
and the forecast integrity.

Any future run is viewable the moment it writes a trace directory (`--trace <dir>`); no
frontend change is needed.

## What each file does

- `replay.py` — the only schema-aware layer. `build_replay(trace_dir)` normalizes one
  run's artifacts into `{meta, world, branches, llm_summary, refusal}`;
  `discover_traces(root)` lists every run under a root. Pure stdlib, no simulation imports.
- `server.py` — stdlib `http.server`. `GET /` serves the page; `/api/traces` lists runs;
  `/api/replay?trace=<dir>` returns one normalized run. Trace paths are confined to
  `--root`.
- `index.html` — one self-contained page (no external assets): the world plane, the
  timeline transport, the branch selector, and the log panel.

## Artifacts it reads

`forecast.json`, `world_manifest.json`, `actor_grounding.json`, `actor_decisions.jsonl`
(carries each actor's `exact_prompt` and `provider_response` verbatim), `event_ledger.jsonl`,
`llm_calls.jsonl`, `diagnosis.json`, `trajectory_audit.json`, `run_audit.json`. A refused
run has no forecast; it still shows the world that was built and the gate it stopped at.
