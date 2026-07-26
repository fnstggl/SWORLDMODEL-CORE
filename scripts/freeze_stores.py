#!/usr/bin/env python3
"""Freeze evidence stores for the eight benchmark cases (5 acceptance + 3 holdouts).

One LIVE research pass per case — retrieval only, no compile — exported as the full
per-claim provenance record `evidence_store.json` beside its `research_trace.json`.
The A/B (scripts/run_ab.sh) then runs BOTH compiler modes from these identical frozen
stores, so the comparison measures the compilers, never retrieval variance.

Writes/updates `artifacts/ab_matrix.json` with each case's question, resolved cutoff
(nowcasts stamp the moment their research pass started), horizon and store path.

    PYTHONPATH=src python3 scripts/freeze_stores.py [--case NAME] [--out artifacts/stores]

Cases come from artifacts/acceptance/questions.json and artifacts_holdouts.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sworldmodel.api import _claim_record  # noqa: E402
from sworldmodel.deepseek_gateway import DeepSeekGateway  # noqa: E402
from sworldmodel.evidence import EvidenceStore  # noqa: E402
from sworldmodel.http import UrllibTransport  # noqa: E402
from sworldmodel.ids import canonical_json  # noqa: E402
from sworldmodel.live_research import (  # noqa: E402
    LiveResearchBackend,
    ResearchBudget,
    ResearchTrace,
)
from sworldmodel.research_planner import plan_research  # noqa: E402


def load_cases() -> list[dict[str, str]]:
    acceptance = json.loads((REPO / "artifacts/acceptance/questions.json").read_text())
    holdouts = json.loads((REPO / "artifacts_holdouts.json").read_text())
    out: list[dict[str, str]] = []
    for c in acceptance["cases"]:
        out.append(
            {
                "case": c["case"],
                "question": c["question"],
                "mode": c["mode"],
                "as_of": c["as_of"],
                "horizon": c["horizon"],
            }
        )
    for c in holdouts["holdouts"]:
        out.append(
            {
                "case": c["case"],
                "question": c["question"],
                "mode": c["mode"],
                "as_of": "generated at launch (process start time)",
                "horizon": c["horizon"],
            }
        )
    return out


def freeze(case: dict[str, str], out_dir: Path) -> dict[str, str]:
    if case["mode"] == "pastcast":
        as_of = datetime.fromisoformat(case["as_of"])
    else:
        as_of = datetime.now(UTC).replace(microsecond=0)
    horizon = datetime.fromisoformat(case["horizon"])

    transport = UrllibTransport()
    gateway = DeepSeekGateway(transport)
    backend = LiveResearchBackend(
        gateway,
        transport,
        budget=ResearchBudget(max_rounds=3, max_queries=20, max_seconds=420.0),
    )
    mode = backend.retrieval_mode(as_of)
    print(f"[{case['case']}] {mode.describe()}", file=sys.stderr, flush=True)
    plan = plan_research(gateway, case["question"], as_of, horizon)
    store = EvidenceStore()
    trace = ResearchTrace(retrieval_mode=mode.as_dict())
    t0 = time.monotonic()
    # Retrieval only: the freeze deliberately stops before any compile, because the
    # A/B's whole point is both compilers reading the identical store.
    backend._run_rounds(case["question"], as_of, plan, store, trace, ())  # noqa: SLF001
    wall = time.monotonic() - t0

    case_dir = out_dir / case["case"]
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "evidence_store.json").write_text(
        canonical_json([_claim_record(c) for c in store.all()]) + "\n"
    )
    (case_dir / "research_trace.json").write_text(canonical_json(trace.to_dict(plan, store)) + "\n")
    admissible = len(store.view(as_of).available())
    print(
        f"[{case['case']}] frozen: {len(store.all())} claims ({admissible} admissible) "
        f"in {wall:.0f}s, {gateway.call_count} model calls",
        file=sys.stderr,
        flush=True,
    )
    return {
        "case": case["case"],
        "store": str(case_dir / "evidence_store.json"),
        "question": case["question"],
        "as_of": as_of.isoformat(),
        "horizon": horizon.isoformat(),
        "mode": case["mode"],
        "claims": str(len(store.all())),
        "admissible": str(admissible),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="freeze only this case")
    ap.add_argument("--out", default="artifacts/stores")
    args = ap.parse_args()

    out_dir = REPO / args.out
    matrix_path = REPO / "artifacts/ab_matrix.json"
    matrix: dict = (
        json.loads(matrix_path.read_text())
        if matrix_path.exists()
        else {"note": "8-case A/B matrix; stores frozen by scripts/freeze_stores.py"}
    )
    rows = {r["case"]: r for r in matrix.get("cases", []) + matrix.get("holdouts", [])}

    failures = 0
    for case in load_cases():
        if args.case and case["case"] != args.case:
            continue
        try:
            rows[case["case"]] = freeze(case, out_dir)
        except Exception as exc:  # noqa: BLE001 — a failed freeze must not stop the rest
            failures += 1
            print(f"[{case['case']}] FREEZE FAILED: {exc}", file=sys.stderr, flush=True)

    holdout_names = {
        "holdout_science_institution",
        "holdout_sports_governance",
        "holdout_space_operations",
    }
    matrix["cases"] = [r for name, r in sorted(rows.items()) if name not in holdout_names]
    matrix["holdouts"] = [r for name, r in sorted(rows.items()) if name in holdout_names]
    matrix["frozen_at"] = datetime.now(UTC).isoformat()
    matrix_path.write_text(json.dumps(matrix, indent=1, sort_keys=True) + "\n")
    print(f"matrix: {matrix_path} ({len(rows)} case(s))", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
