#!/usr/bin/env python3
"""Summarize the five acceptance runs from their own artifacts.

Reads only what the runs wrote — diagnosis.json per case — so the report cannot say
anything the artifacts do not support. Prints a table plus per-case detail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CASES = ("geopolitical", "individual", "negotiation", "population", "committee")
ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "acceptance"


def load(case: str) -> dict | None:
    """The diagnosis this system wrote, not one a diagnostic pass left beside it.

    ``root_cause_vocabulary`` is the marker: it is emitted only by ``RunDiagnosis``, so
    a hand-written or older artifact in the same directory is skipped rather than
    silently reported as a run's own record.
    """

    for candidate in (ROOT / case / "run_trace" / "diagnosis.json", ROOT / case / "diagnosis.json"):
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text())
        if isinstance(data, dict) and "root_cause_vocabulary" in data:
            return data
    return None


def main() -> int:
    rows = []
    for case in CASES:
        d = load(case)
        if d is None:
            rows.append((case, "no artifact", "-", "-", "-", "-", "-"))
            continue
        ext, comp, rt = d["extraction"], d["world_compilation"], d["runtime"]
        producers = comp.get("causal_producers", {}) if comp.get("compiled") else {}
        rows.append(
            (
                case,
                d["outcome"],
                f"{ext['claims_stored']}/{ext['claim_candidates']}",
                str(len(producers.get("actors", []) or []))
                + "a/"
                + str(len(producers.get("external_processes", []) or []))
                + "p",
                str(rt.get("actor_invocations", "-")),
                str(d.get("model_calls", "-")),
                f"{d.get('wall_seconds', 0):.0f}s",
            )
        )

    head = ("case", "outcome", "claims", "producers", "actor calls", "model calls", "wall")
    widths = [max(len(str(r[i])) for r in (*rows, head)) for i in range(len(head))]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths, strict=True))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))

    for case in CASES:
        d = load(case)
        if d is None:
            continue
        print(f"\n=== {case} ===")
        print(f"question: {d['question']}")
        print(f"mode: {d['research_planning']['retrieval_mode'].get('mode')}")
        print(f"queries: {len(d['research_planning']['queries_issued'])}"
              f"  urls: {d['discovery']['urls_considered_count']}"
              f"  official: {len(d['discovery']['official_domain_urls_found'])}"
              f"  fetched: {d['fetching']['fetched_count']}")
        print(f"epistemic: {d['epistemic_classification'].get('counts')}")
        comp = d["world_compilation"]
        if comp.get("compiled"):
            print(f"producers: {comp['causal_producers']}")
            print(f"terminal producers: {comp['terminal_producers']}")
            orphan = comp.get("terminal_terms_without_a_producer")
            if orphan:
                print(f"TERMS WITH NO PRODUCER: {orphan}")
        ig = d["integrity_and_grounding"]
        print(f"repair attempts: {len(ig.get('repair_attempts', []))}"
              f"  stopped at: {ig.get('stopped_at_gate')}")
        rt = d["runtime"]
        if rt.get("ran"):
            print(f"runtime: {rt['event_count']} events, {rt['actor_invocations']} actor calls, "
                  f"{rt['resolved_branches']}/{rt['branches']} branches resolved, "
                  f"unresolved mass {rt['unresolved_mass']}")
            print(f"wake reasons: {rt['wake_reasons']}")
        print(f"root cause: {[c['cause'] for c in d['root_cause']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
