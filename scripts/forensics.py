#!/usr/bin/env python3
"""Forensic reconstruction of a completed run, from its own production artifacts.

A published probability is worth exactly as much as an independent reviewer's ability
to rebuild it. This tool rebuilds it: it reads only what a run already wrote, derives
the chronology, the per-branch weight accounting, the state transitions, the terminal
evaluations and the aggregation, and then re-derives the published number and compares.
A mismatch is CRITICAL and the run is marked FORENSICALLY_INVALID.

It also runs the trajectory-responsibility tests as *deterministic replays over the
recorded ledger* — no model call, no re-simulation:

  * remove every actor-produced event and re-evaluate each branch's terminal;
  * remove one actor's events at a time;
  * remove one non-actor process's events at a time;
  * keep the trajectory but flatten branch weights to uniform;
  * keep the weights but delete terminal-relevant actions.

The classification that comes out (TRAJECTORY_CAUSED … BRANCH_WEIGHTS_DOMINATED) is a
statement about *what produced the number*, and it is computed, never asserted.

Every replay, diff derivation, counterfactual and terminal re-evaluation happens in
:mod:`sworldmodel.replaycore` — the ONE ledger-replay implementation (decision D7),
shared verbatim with the production trace writer. This script is a consumer: it loads
the artifacts, runs the shared reconstruction, and emits the forensic artifact set.

Terminal re-evaluation uses the run's OWN executable world when the run persisted one
(`compiled_world.json`) — the same expression AST the engine evaluated. Runs written
before that artifact existed fall back to a declared-support subset of the terminal
grammar; anything outside it is reported as UNRECONSTRUCTABLE rather than guessed.

    PYTHONPATH=src python3 scripts/forensics.py --run artifacts/slice92/individual \
        --out artifacts/forensics/individual --label individual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sworldmodel.replaycore import (  # noqa: E402
    RATE_IN_PER_MTOK,
    RATE_OUT_PER_MTOK,
    RunRecord,
    branch_key,
    build_timeline,
    derive_state_diffs,
    extract_communications,
    extract_process_transitions,
    load_run_record,
    reconstruct_run,
    render_dossier,
    terminal_ast_from_world,
    terminal_field_reads,
)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def reconstruct(run: Path, label: str) -> dict[str, Any]:
    """Rebuild the run through the shared replay core (kept as the script's API)."""

    return reconstruct_run(load_run_record(run, label))


def emit(run: Path, out: Path, label: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    record: RunRecord = load_run_record(run, label)
    result = reconstruct_run(record)

    events = record.events
    decisions = record.decisions
    calls = record.calls

    def dump(name: str, rows: list[dict[str, Any]]) -> None:
        (out / name).write_text(
            "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows)
        )

    # 1. chronological timeline, grouped by branch then simulation time
    timeline = build_timeline(events, decisions)
    dump("forensic_timeline.jsonl", timeline)

    # 2. every LLM call, in order, complete as recorded
    full_calls = []
    for i, c in enumerate(calls, start=1):
        row = dict(c)
        row["call_number"] = c.get("call_number", i)
        row["estimated_cost_usd"] = round(
            int(c.get("tokens_in") or 0) / 1e6 * RATE_IN_PER_MTOK
            + int(c.get("tokens_out") or 0) / 1e6 * RATE_OUT_PER_MTOK,
            6,
        )
        for missing in ("prompt", "started_at", "ended_at", "latency_ms"):
            if missing not in row:
                row[f"__missing_{missing}"] = (
                    "not recorded by the run that produced this trace; "
                    "instrumented from commit 71393bb onward"
                )
        full_calls.append(row)
    dump("llm_calls_full.jsonl", full_calls)

    # 3. actor invocations — the complete decision record, as written
    dump("actor_invocations.jsonl", decisions)

    # 4. communications, with the honest delivery→notice join from actor decisions
    dump("communications.jsonl", extract_communications(events, decisions))

    # 5. non-actor process transitions
    dump("process_transitions.jsonl", extract_process_transitions(events))

    # 6. state diffs, replayed in ledger order from each branch's initial state
    initial_by_branch = {
        branch_key(str(b["branch_id"])): dict(b["initial_fields"]) for b in result["branches"]
    }
    terminal_fields = terminal_field_reads(
        terminal_ast_from_world(record.world), record.manifest.get("terminal") or {}
    )
    dump(
        "state_diffs.jsonl",
        derive_state_diffs(
            events, initial_by_branch=initial_by_branch, terminal_fields=terminal_fields
        ),
    )

    # 7. branch weight history
    weights = []
    for b in result["branches"]:
        weights.append(
            {
                "branch_id": b["branch_id"],
                "assumptions": b["conditions"],
                "starting_weight": b["weight"],
                "final_weight": b["weight"],
                "weight_changes": [],
                "provenance": result["weights"]["outcome_provenances"],
                "reason_for_starting_weight": (
                    "uniform split across the enumerated alternatives of each uncertainty; "
                    "every outcome carries "
                    + ", ".join(result["weights"]["outcome_provenances"] or ["(none recorded)"])
                ),
                "terminal_result": b["recomputed_outcome"],
                "contribution_to_probability": (
                    float(b["weight"]) if b["recomputed_outcome"] == "YES" else 0.0
                ),
            }
        )
    dump("branch_weight_history.jsonl", weights)

    # 8. semantic -> runtime lineage
    trace = _read_json(run / "research_trace.json", {})
    sem = trace.get("semantic_compilation") or {}
    rounds = trace.get("semantic_repair_rounds") or []
    if rounds and isinstance(rounds[-1], dict):
        sem = rounds[-1]
    mapping = sem.get("mapping") or []
    lineage = []
    for m in mapping if isinstance(mapping, list) else []:
        rid = str(m.get("runtime_id") or "")
        touching = [
            e.get("event_id")
            for e in events
            if rid and rid in json.dumps({"p": e.get("payload"), "k": e.get("kind")}, default=str)
        ]
        lineage.append(
            {
                "evidence_claim_ids": m.get("evidence_claim_ids") or [],
                "semantic_object": m.get("semantic"),
                "namespace": m.get("namespace"),
                "runtime_id": rid,
                "lowering_rule": m.get("lowering_rule"),
                "event_ids": touching,
                "executed": bool(touching),
            }
        )
    dump("semantic_runtime_lineage.jsonl", lineage)

    # 9. terminal evaluations
    dump(
        "terminal_evaluations.jsonl",
        [
            {
                "branch_id": b["branch_id"],
                "terminal_evaluated_via": b["terminal_evaluated_via"],
                "fields_read": b["final_fields"],
                "event_type_counts": b["final_event_type_counts"],
                "recomputed_resolved": b["recomputed_resolved"],
                "recomputed_outcome": b["recomputed_outcome"],
                "published_outcome": b["published_outcome"],
                "matches_published": b["matches_published"],
            }
            for b in result["branches"]
        ],
    )

    # 10-12. reconstruction, responsibility, verdict
    (out / "probability_reconstruction.json").write_text(
        json.dumps(
            {
                "published": result["published"],
                "recomputed": result["recomputed"],
                "aggregation_formula": (
                    "P(YES) = sum(branch_weight x yes_indicator) / sum(resolved branch_weight)"
                ),
                "numeric_substitution": result["recomputed"]["substitution"],
                "weights": result["weights"],
                "mismatches": result["mismatches"],
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    (out / "trajectory_responsibility.json").write_text(
        json.dumps(
            {
                "classification": result["responsibility"]["classification"],
                "signals": result["responsibility"],
                "per_branch_counterfactuals": {
                    b["branch_id"]: b["counterfactuals"] for b in result["branches"]
                },
                "method": (
                    "deterministic replay of the recorded event ledger with selected "
                    "events removed, then terminal re-evaluation; no simulation, no "
                    "model call"
                ),
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    # 13. the human-readable dossier
    (out / "run_dossier.html").write_text(render_dossier(result, timeline, decisions, full_calls))

    (out / "forensic_verdict.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=str) + "\n"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    result = emit(Path(args.run), Path(args.out), args.label)
    print(f"=== {args.label}: {result['verdict']}")
    print(f"  published  p = {result['published']['probability']}")
    print(f"  recomputed p = {result['recomputed']['probability']}")
    print(f"  substitution: {result['recomputed']['substitution']}")
    print(f"  weights: {result['weights']['flag']}")
    print(f"  provenance: {result['weights']['outcome_provenances']}")
    print(f"  classification: {result['responsibility']['classification']}")
    for m in result["mismatches"]:
        print(
            f"  {m['severity']} MISMATCH {m['field']}: published {m['published']} vs "
            f"recomputed {m['recomputed']}"
        )
    for n in result["reconstruction_notes"]:
        print(f"  note: {n}")
    return 0 if result["verdict"] == "RECONSTRUCTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
