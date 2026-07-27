#!/usr/bin/env python3
"""Summarize an A/B run: per-mode medians/p90s, per-case diffs, cache economics.

Reads artifacts/ab/ab_results.jsonl (written by scripts/run_ab.sh) and prints a
markdown report. Cost is reported in tokens; pass DEEPSEEK_PRICE_IN /
DEEPSEEK_PRICE_OUT / DEEPSEEK_PRICE_CACHED (USD per 1M tokens) to also price it —
prices are never invented here.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "artifacts/ab/ab_results.jsonl"


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[int(idx)]


def _fmt_min(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def main() -> int:
    rows = [json.loads(x) for x in RESULTS.read_text().splitlines() if x.strip()]
    by_mode: dict[str, list[dict]] = {"semantic": [], "direct": []}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    price_in = os.environ.get("DEEPSEEK_PRICE_IN")
    price_out = os.environ.get("DEEPSEEK_PRICE_OUT")
    price_cached = os.environ.get("DEEPSEEK_PRICE_CACHED")

    print(f"commit: {rows[0]['commit'] if rows else '?'}   runs: {len(rows)}\n")
    print(
        "| mode | n | wall median | wall p90 | calls med | tokens in med | out med "
        "| cached med | cache hit rate | memo |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    def med(values: list[float], fmt: str = "{:,.0f}") -> str:
        return fmt.format(statistics.median(values)) if values else "—"

    for mode in ("semantic", "direct"):
        runs = by_mode.get(mode) or []
        walls = [float(r.get("wall") or 0) for r in runs if r.get("wall")]
        calls = [float(r.get("calls") or 0) for r in runs if r.get("calls") is not None]
        tin = [float(r.get("tokens_in") or 0) for r in runs]
        tout = [float(r.get("tokens_out") or 0) for r in runs]
        tcached = [float(r.get("tokens_cached") or 0) for r in runs]
        rates = [float(r.get("cache_hit_rate") or 0) for r in runs]
        memo = sum(int(r.get("memo_reuses") or 0) for r in runs)
        wall_med = _fmt_min(statistics.median(walls)) if walls else "—"
        wall_p90 = _fmt_min(_pct(walls, 0.9)) if walls else "—"
        print(
            f"| {mode} | {len(runs)} | {wall_med} | {wall_p90} | {med(calls)} "
            f"| {med(tin)} | {med(tout)} | {med(tcached)} "
            f"| {med(rates, '{:.1%}')} | {memo} |"
        )

    if price_in and price_out:
        print("\n### Priced cost (from operator-supplied rates, USD per 1M tokens)\n")
        for mode in ("semantic", "direct"):
            runs = by_mode.get(mode) or []
            cin = sum(
                float(r.get("tokens_in") or 0) - float(r.get("tokens_cached") or 0) for r in runs
            )
            cc = sum(float(r.get("tokens_cached") or 0) for r in runs)
            cout = sum(float(r.get("tokens_out") or 0) for r in runs)
            total = (
                cin / 1e6 * float(price_in)
                + cout / 1e6 * float(price_out)
                + cc / 1e6 * float(price_cached or price_in)
            )
            print(f"- {mode}: ${total:.4f} across {len(runs)} runs")
    else:
        print(
            "\ncost basis: token totals above (no provider list price supplied; "
            "set DEEPSEEK_PRICE_IN/OUT/CACHED to price them)"
        )

    print("\n### Per-case outcomes (semantic vs direct)\n")
    print("| case | semantic | p | wall | calls | direct | p | wall | calls |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        cells: list[str] = [case]
        for mode in ("semantic", "direct"):
            r = next((x for x in rows if x["case"] == case and x["mode"] == mode), None)
            if r is None:
                cells += ["(missing)", "—", "—", "—"]
                continue
            status = str(r.get("status") or f"exit {r.get('exit')}")
            if status == "refused":
                status = f"refused ({r.get('failure_stage')})"
            p = r.get("probability")
            cells += [
                status,
                "—" if p is None else f"{float(p):.2f}",
                _fmt_min(float(r.get("wall") or 0)) if r.get("wall") else "—",
                str(r.get("calls") if r.get("calls") is not None else "—"),
            ]
        print("| " + " | ".join(cells) + " |")

    print("\n### Semantic per-stage call medians\n")
    stage_totals: dict[str, list[int]] = {}
    for r in by_mode.get("semantic") or []:
        for stage, n in (r.get("calls_by_stage") or {}).items():
            stage_totals.setdefault(stage, []).append(int(n))
    for stage, ns in sorted(stage_totals.items()):
        print(f"- {stage}: median {statistics.median(ns):.0f} (over {len(ns)} runs)")

    walls_sem = [float(r.get("wall") or 0) for r in by_mode.get("semantic") or [] if r.get("wall")]
    if walls_sem:
        med = statistics.median(walls_sem)
        verdict = "REACHED" if med <= 300 else "NOT reached"
        print(
            f"\n1–5 minute consumer target: median semantic wall {_fmt_min(med)} — {verdict} "
            f"(p90 {_fmt_min(_pct(walls_sem, 0.9))})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
