#!/usr/bin/env python3
"""Judge each acceptance case against its own artifacts, criterion by criterion.

Every check reads a file the run wrote. Nothing here re-runs anything, and nothing
asserts a property it cannot point at: a criterion with no artifact behind it reports
NO ARTIFACT rather than passing.

The bar is the one the run was set: a refusal, a crash, a timeout, a hollow answer and
an unproduced terminal are each a failure, and a completed process exit is not by itself
a pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CASES = ("geopolitical", "individual", "negotiation", "population", "committee")

# Read from engine.WAKE_* so a reason the runtime gains is not reported here as unknown,
# and a reason it never emits cannot be waved through.
WAKE_REASONS = frozenset(
    v
    for k, v in vars(__import__("sworldmodel.engine", fromlist=["engine"])).items()
    if k.startswith("WAKE_") and isinstance(v, str)
)
ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "acceptance"


def load(case: str, name: str) -> Any:
    # The trace the run wrote comes first. A file left beside it by earlier exploratory
    # work is not this run's record, and reading one as if it were is how a report ends
    # up describing a run that never happened.
    for base in (ROOT / case / "run_trace", ROOT / case):
        p = base / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
    return None


def lines(case: str, name: str) -> list[dict[str, Any]]:
    for base in (ROOT / case / "run_trace", ROOT / case):
        p = base / name
        if p.exists():
            return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return []


def checks(case: str) -> list[tuple[str, str, str]]:
    """(criterion, verdict, detail) for one case."""

    d = load(case, "diagnosis.json")
    if d is None or "root_cause_vocabulary" not in (d or {}):
        return [("ran at all", "NO ARTIFACT", "no diagnosis.json this run wrote")]

    out: list[tuple[str, str, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        out.append((name, "PASS" if ok else "FAIL", detail))

    # 1. It finished, and finishing is not the same as answering.
    outcome = d.get("outcome")
    check(
        "completed without refusing",
        outcome == "completed",
        f"outcome={outcome}"
        + (
            f", stopped at {d['integrity_and_grounding'].get('stopped_at_gate')}"
            if outcome != "completed"
            else ""
        ),
    )

    # 2. Research reached live sources and stored verified claims.
    disc, fetch, ext = d["discovery"], d["fetching"], d["extraction"]
    check(
        "reached sources and stored claims",
        fetch["fetched_count"] > 0 and ext["claims_stored"] > 0,
        f"{disc['urls_considered_count']} urls, {fetch['fetched_count']} fetched, "
        f"{ext['claims_stored']}/{ext['claim_candidates']} claims stored, "
        f"{len(disc['official_domain_urls_found'])} on requested authoritative domains",
    )

    # 3. The mode was decided and recorded before research, not inferred afterwards.
    mode = d["research_planning"]["retrieval_mode"]
    check(
        "retrieval mode pinned and recorded",
        mode.get("mode") in ("nowcast", "pastcast"),
        f"{mode.get('mode')} — cutoff {mode.get('as_of', '?')}",
    )

    # 4. Every claim carries an epistemic class.
    epi = d["epistemic_classification"]
    counts = epi.get("counts") or {}
    check(
        "every claim carries an epistemic class",
        bool(counts) and sum(counts.values()) == ext["claims_stored"],
        f"{counts} over {ext['claims_stored']} stored",
    )

    comp = d["world_compilation"]
    if not comp.get("compiled"):
        check("world compiled", False, "no world was compiled")
        return out

    # 5. Something in the world can produce the answer, and it is named.
    prod = comp.get("causal_producers") or {}
    check(
        "the world has a causal producer",
        bool(prod.get("actors") or prod.get("external_processes") or prod.get("process_nodes")),
        f"{len(prod.get('actors') or [])} actor(s), "
        f"{len(prod.get('external_processes') or [])} process(es), "
        f"{len(prod.get('process_nodes') or [])} node(s)",
    )

    # 6. No terminal term arrives from nowhere.
    orphans = comp.get("terminal_terms_without_a_producer") or []
    check(
        "no terminal term without a producer",
        not orphans,
        f"producers {comp.get('terminal_producers')}" if not orphans else f"orphans {orphans}",
    )

    rt = d["runtime"]
    if not rt.get("ran"):
        check("the simulation ran", False, str(rt.get("reason")))
        return out

    # 7. Mass actually resolved. An unresolved run is honest and is not an answer.
    unresolved = float(rt.get("unresolved_mass") or 0.0)
    check(
        "the terminal was determined",
        rt.get("resolved_branches", 0) > 0 and unresolved < 1.0,
        f"{rt['resolved_branches']}/{rt['branches']} branches resolved, "
        f"unresolved mass {unresolved}",
    )

    # 8. Every terminal term names what wrote it — where that is what the answer needed.
    #
    #    A branch that resolved NO because the producing action was never taken has an
    #    unwritten term, and that is the honest record of a person who declined to act:
    #    a live Bank of England branch resolved NO because Bailey waited, saying his July
    #    signal still stood. What may never happen is a branch resolving YES on a term
    #    nothing wrote — that is the answer arriving without being produced — or a term
    #    no branch anywhere could produce.
    lineage = rt.get("terminal_producer_lineage") or {}
    forecast = load(case, "forecast.json") or {}
    yes_branches = {
        str(b.get("branch_id"))
        for b in (forecast.get("branches") or [])
        if str(b.get("outcome") or "").upper() == "YES"
    }
    unproduced_in_yes = sorted(
        {
            f"{bid}:{rec.get('terminal_term')}"
            for bid, recs in lineage.items()
            for rec in recs
            if rec.get("unproduced") and bid in yes_branches
        }
    )
    produced_somewhere = {
        str(rec.get("terminal_term"))
        for recs in lineage.values()
        for rec in recs
        if not rec.get("unproduced")
    }
    all_terms = {str(rec.get("terminal_term")) for recs in lineage.values() for rec in recs}
    never = sorted(all_terms - produced_somewhere)
    check(
        "every YES was produced, and every term was producible",
        bool(lineage) and not unproduced_in_yes and not never,
        f"produced {sorted(produced_somewhere)}"
        + (f"; never produced in any branch {never}" if never else "")
        + (f"; YES with no writer {unproduced_in_yes}" if unproduced_in_yes else ""),
    )

    # 9. Actors and processes were actually invoked — the trace is a run, not a shape.
    ledger = lines(case, "event_ledger.jsonl")
    check(
        "actors or processes actually ran",
        bool(ledger) and rt.get("event_count", 0) > 0,
        f"{rt['event_count']} events, {rt['actor_invocations']} actor invocation(s), "
        f"wake reasons {rt.get('wake_reasons')}",
    )

    # 9b. Every actor call happened for a reason the runtime can name. A turn taken
    #     because it was somebody's turn is the thing this architecture does not have.
    wake = rt.get("wake_reasons") or {}
    unknown = sorted(set(wake) - set(WAKE_REASONS))
    check(
        "every actor call has a named wake reason",
        not unknown and (rt.get("actor_invocations", 0) == 0 or bool(wake)),
        f"{wake}" if not unknown else f"unrecognised wake reasons {unknown}",
    )

    # 9c. Branch uncertainty must not be the answer wearing a world's clothes.
    producers = comp.get("terminal_producers") or {}
    from_uncertainty = sorted(
        {
            f"{term} <- {who}"
            for term, whos in producers.items()
            for who in whos
            if str(who).startswith("uncertainty:")
        }
    )
    check(
        "no terminal term is written by an uncertainty",
        not from_uncertainty,
        "every producer is an action, a process or cited evidence"
        if not from_uncertainty
        else f"{from_uncertainty}",
    )

    # 10. The answer is replayable: the ledger is present and the branches are recorded
    #     with the world state each resolved from.
    forecast = load(case, "forecast.json")
    branches = (forecast or {}).get("branches") or []
    check(
        "replayable from the written trace",
        bool(ledger) and bool(branches) and all("world_state" in b for b in branches),
        f"{len(ledger)} ledger events, {len(branches)} branch record(s)",
    )

    # 11. The run's own diagnosis does not name a defect.
    causes = [c["cause"] for c in d["root_cause"]]
    check(
        "the run's own diagnosis is clean",
        causes == ["none"],
        f"root cause {causes}",
    )
    return out


def main() -> int:
    failed = 0
    for case in CASES:
        rows = checks(case)
        bad = [r for r in rows if r[1] != "PASS"]
        failed += bool(bad)
        head = f"{case}: {'PASS' if not bad else 'FAIL'}"
        print(f"\n{head}\n{'=' * len(head)}")
        width = max(len(r[0]) for r in rows)
        for name, verdict, detail in rows:
            print(f"  {verdict:11} {name.ljust(width)}  {detail}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} cases pass every criterion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
