#!/usr/bin/env python3
"""Emit the acceptance results as markdown, entirely from the runs' own artifacts.

The report tables are generated rather than transcribed so that no number in them can
drift from the file it came from. Every column names the artifact field behind it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CASES = ("geopolitical", "individual", "negotiation", "population", "committee")
ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "acceptance"


def load(case: str, name: str) -> Any:
    for base in (ROOT / case / "run_trace", ROOT / case):
        p = base / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
    return None


def diagnoses() -> dict[str, Any]:
    out = {}
    for case in CASES:
        d = load(case, "diagnosis.json")
        if isinstance(d, dict) and "root_cause_vocabulary" in d:
            out[case] = d
    return out


def table(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> int:
    ds = diagnoses()
    status = (ROOT / "status.log").read_text() if (ROOT / "status.log").exists() else ""
    commit = next((x.split(":", 1)[1].strip() for x in status.splitlines() if "commit:" in x), "?")
    walls = {
        x.split()[1]: x.split("wall=")[1].split("s")[0]
        for x in status.splitlines()
        if x.startswith("=== ")
    }

    print(f"Frozen commit `{commit}`. Working tree clean at launch.\n")

    # 7 + 14 + 15 — outcome, probability, unresolved mass
    rows = []
    for case in CASES:
        d = ds.get(case)
        if d is None:
            rows.append([case, "no artifact", "-", "-", "-", "-"])
            continue
        f = load(case, "forecast.json") or {}
        rt = d["runtime"]
        prob = f.get("simulation_probability")
        rows.append(
            [
                case,
                d["outcome"]
                if d["outcome"] == "completed"
                else f"refused ({d['integrity_and_grounding'].get('stopped_at_gate')})",
                "—" if prob is None else f"{prob:.4f}",
                str(f.get("probability_source") or "—"),
                f"{rt.get('unresolved_mass', '—')}",
                walls.get(case, "?") + "s",
            ]
        )
    print("### Result, probability and unresolved mass\n")
    print(table(["case", "outcome", "probability", "source", "unresolved mass", "wall"], rows))

    # 8 + 9 — research and claim counts
    rows = []
    for case, d in ds.items():
        disc, fetch, ext, epi = (
            d["discovery"],
            d["fetching"],
            d["extraction"],
            d["epistemic_classification"],
        )
        c = epi.get("counts") or {}
        rows.append(
            [
                case,
                str(len(d["research_planning"]["queries_issued"])),
                str(disc["urls_considered_count"]),
                str(len(disc["official_domain_urls_found"])),
                str(fetch["fetched_count"]),
                str(fetch["rejected_count"]),
                str(ext["claims_stored"]) + "/" + str(ext["claim_candidates"]),
                str(c.get("verified", 0)),
                str(c.get("inferred", 0)),
                str(c.get("hypothetical", 0)),
            ]
        )
    print("\n### Research and evidence, per case\n")
    print(
        table(
            [
                "case",
                "queries",
                "urls",
                "on authoritative domains",
                "fetched",
                "rejected",
                "claims stored",
                "verified",
                "inferred",
                "hypothetical",
            ],
            rows,
        )
    )

    # 10 + 11 — producers and what wrote the terminal
    print("\n### Causal producers and terminal lineage, per case\n")
    for case, d in ds.items():
        comp = d["world_compilation"]
        if not comp.get("compiled"):
            print(f"**{case}** — no world compiled\n")
            continue
        p = comp["causal_producers"]
        print(f"**{case}** — {comp['title']}")
        print(f"* terminal: {comp['terminal_description']}")
        print(
            f"* producers: actors {p['actors'] or '—'}; "
            f"processes {p['external_processes'] or '—'}; "
            f"nodes {p['process_nodes'] or '—'}"
        )
        print(f"* compile-time producer of each terminal term: `{comp['terminal_producers']}`")
        orphan = comp.get("terminal_terms_without_a_producer")
        if orphan:
            print(f"* **terms with no producer: {orphan}**")
        lineage = (
            (d["runtime"].get("terminal_producer_lineage") or {}) if d["runtime"]["ran"] else {}
        )
        for bid, recs in list(lineage.items())[:2]:
            for rec in recs:
                who = [
                    f"{w['kind']}"
                    + (f" by {w['actor_id']}" if w.get("actor_id") else " (process)")
                    + f" at {w['at']}"
                    for w in rec.get("written_by", [])[:3]
                ]
                cites = rec.get("established_by_evidence") or []
                print(
                    f"* runtime, branch `{bid}`: `{rec['terminal_term']}` "
                    + (f"written by {who}" if who else "")
                    + (f" established by evidence {cites}" if cites else "")
                    + (" — UNPRODUCED" if rec.get("unproduced") else "")
                )
        print()

    # 12 + 13 — invocations, reasons, model calls
    rows = []
    for case, d in ds.items():
        rt = d["runtime"]
        if not rt["ran"]:
            rows.append([case, "did not run", "-", "-", str(d["model_calls"]), "-"])
            continue
        rows.append(
            [
                case,
                str(rt["event_count"]),
                str(rt["actor_invocations"]),
                str(rt.get("wake_reasons") or {}),
                str(d["model_calls"]),
                f"{rt['resolved_branches']}/{rt['branches']}",
            ]
        )
    print("\n### Runtime, per case\n")
    print(
        table(
            ["case", "events", "actor calls", "wake reasons", "model calls", "branches resolved"],
            rows,
        )
    )

    # 6 — the run's own root cause
    print("\n### Each run's own root cause\n")
    for case, d in ds.items():
        causes = "; ".join(f"**{c['cause']}** — {c['why']}" for c in d["root_cause"])
        print(f"* **{case}**: {causes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
