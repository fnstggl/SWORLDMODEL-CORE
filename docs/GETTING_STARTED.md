# Getting started

Everything here works in a clean environment. The kernel is stdlib-only — the *only*
things a live forecast needs beyond Python are network access and provider keys.

## Requirements

- Python **3.11+** (`requires-python = ">=3.11"`; CI toolchain targets `py311`)
- No runtime dependencies. Dev tooling (`pytest`, `mypy`, `ruff`) comes from the
  `dev` extra.

## Install

```bash
make install            # = python3 -m pip install -e ".[dev]"
```

Or run without installing anything — every entry point works with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m sworldmodel forecast --help
PYTHONPATH=src python3 -m pytest -q          # full suite: invariants + unit, offline
```

## Environment keys

Set these as environment variables. **Names only are documented here — never commit a
value anywhere in this repository.**

| Variable | Used by | Needed for |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | model gateway (`deepseek_gateway.py`) | any live compile, actor call or forecast — including the frozen-store harnesses, which still run the real compiler and actors |
| `JINA_API_KEY` | page reader provider (`providers.py`) | live research (fetching and reading sources) |
| `SERPER_API_KEY` | search provider (`providers.py`) | live research (web search) |

The test suite needs **none** of them: it is fully offline and deterministic.

## Run a live forecast

```bash
export DEEPSEEK_API_KEY=...   # plus JINA_API_KEY / SERPER_API_KEY for live research
sworldmodel forecast \
  --question "Will [social/institutional event] happen by [date]?" \
  --as-of   "2026-07-26T00:00:00+00:00" \
  --horizon "2026-09-15T23:59:59+00:00" \
  --trace   artifacts/my_run
```

Pass the **current time** as `--as-of` for a nowcast; a past cutoff makes the run a
pastcast, where every source must be retrieved as its archived capture at that cutoff
(see `README.md`). Inspect the written trace with
`sworldmodel inspect artifacts/my_run --summary`.

The default compiler is **semantic** — the canonical route: question → research →
semantic causal-world plan → independent reality review → static semantic validation →
deterministic lowering → the existing runtime → trajectory-derived result → replay.
`--compiler direct` (one model call authors the WorldSpec) exists only as an explicit
diagnostic flag for controlled comparison and regression diagnosis; nothing ever falls
back to it, and a semantic refusal is the run's result. Every run records its compiler
mode in `run_stamp.json`, `research_trace.json` and `diagnosis.json`.

## Caching (optional, on by default)

Two on-disk record-and-replay caches speed up repeated runs of the same question
without changing what a run decides: accepted source text (replayed with its original
observation times, so cutoff admissibility is judged exactly as the live fetch judged
it) and extracted claims (keyed by content, question, prompt version, model and seed;
re-verified on replay). The live gateway also reuses byte-identical model requests
within a run, and DeepSeek's automatic context caching is exploited by keeping the
plan prompt an exact prefix of every revision prompt.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SWORLDMODEL_CACHE` | `off` disables all on-disk caching | on |
| `SWORLDMODEL_SOURCE_CACHE_DIR` | accepted-source store | `.cache/sources` |
| `SWORLDMODEL_EXTRACT_CACHE_DIR` | extraction store | `.cache/extractions` |
| `SWORLDMODEL_SOURCE_CACHE_TTL` | live-page TTL (seconds) | 21600 |
| `SWORLDMODEL_ARCHIVE_CACHE_TTL` | archive-capture TTL (seconds) | 2592000 |
| `SWORLDMODEL_EXTRACT_CACHE_TTL` | extraction TTL (seconds) | 604800 |

## Replay from a frozen evidence store

The frozen harnesses re-run everything after retrieval from a prior run's exported
`evidence_store.json` (they still call the model, so `DEEPSEEK_API_KEY` is required):

```bash
# Full canonical route (forecast, runtime, auditors) from a frozen store.
# The default mode is semantic; pass --mode direct ONLY for a controlled comparison.
PYTHONPATH=src python3 scripts/frozen_forecast.py \
  --store artifacts/<run>/evidence_store.json \
  --question "…" --as-of <iso> --horizon <iso> --out artifacts/replay1

# Compile-only slice (plan → review → validate → lower → gates), no simulation:
PYTHONPATH=src python3 scripts/semantic_slice.py \
  --store artifacts/<run>/evidence_store.json \
  --question "…" --as-of <iso> --horizon <iso> --out artifacts/slice1
```

Each run stamps its output directory (`run_stamp.json`) and clears any previous run's
pipeline artifacts first. A store exported without full per-claim provenance loads
with a loud `legacy store: provenance defaulted` warning — its authority ranking will
differ from the live run's.

## Checks

```bash
make test        # pytest (invariants + unit), offline
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict over src/sworldmodel
make check       # all three
```

Current honest status of these gates is tracked in `docs/LAUNCH_READINESS.md`.
