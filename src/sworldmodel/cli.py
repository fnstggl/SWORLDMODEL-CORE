"""Command-line entry point.

There is one command that produces a forecast, and it takes only the substantive
question and its window::

    sworldmodel forecast --question "..." --as-of <iso> --horizon <iso>

No corpus, roster, institution, actor list, protocol, action list, threshold, branch
weight, source list or terminal mechanism may be supplied — because none may be
*needed*. Everything else the world requires is researched and compiled from the
question. There is no ``--corpus`` escape hatch and no offline mode: if the live path
cannot do it, that is a fact about the system and it is reported, not routed around.

``inspect`` reads a written trace directory and shows exactly what each actor was sent
and what it returned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .api import run_forecast
from .config import ForecastConfig
from .diagnosis import ForecastRefused, RunDiagnosis
from .engine import RunBudget
from .ids import canonical_json
from .live_research import ResearchBudget
from .models import ForecastResult
from .source_fetch import requires_archived_copy
from .tracing import TraceContext


def _print_summary(result: ForecastResult, forecast_hash: str, out_dir: Path | None) -> None:
    print(f"Question: {result.question}")
    print(f"Status: {result.status.value}")
    p = result.simulation_probability
    print(
        f"Simulation probability: {'—' if p is None else f'{p:.4f}'}  "
        f"(source: {result.probability_source})"
    )
    lb = result.lower_bound if result.lower_bound is not None else 0.0
    ub = result.upper_bound if result.upper_bound is not None else 1.0
    print(f"Unconditional bounds: [{lb:.4f}, {ub:.4f}]")
    print(
        f"Mass — resolved YES {result.resolved_yes_mass:.4f}, NO {result.resolved_no_mass:.4f}, "
        f"unresolved {result.unresolved_mass:.4f}"
    )
    print(
        f"Integrity: {result.integrity_manifest.integrity_verdict.value} "
        f"(participants expected {result.integrity_manifest.expected_participants}, "
        f"represented {result.integrity_manifest.represented_participants})"
    )
    print(f"Branches: {len(result.branch_outcomes)}  |  model calls: {result.model_call_count}")
    for b in result.branch_outcomes:
        recs = ", ".join(f"{k}={v}" for k, v in b.records[:6])
        state = b.outcome if b.resolved else f"UNRESOLVED({b.unresolved_reason})"
        print(f"  - {b.branch_id} w={b.weight:.4f} [{recs}] -> {state}")
    if out_dir is not None:
        print(f"Artifacts: {out_dir}")
        if forecast_hash:
            print(f"Forecast SHA-256: {forecast_hash}")


def _audit(config: ForecastConfig, ctx: TraceContext, wall_seconds: float) -> dict[str, Any]:
    """What actually happened on the wire and in the runtime.

    Every field here is *derived* from instrumentation. Nothing asserts a property as a
    literal — a hardcoded ``"prepared_corpus_read": false`` proves nothing, which is
    why the corpus path was removed rather than reported on.
    """

    gw = config.gateway
    bundle = ctx.bundle
    transport = getattr(config.research_backend, "transport", None)
    http_calls = getattr(transport, "calls", [])
    lat = gw.latencies_ms
    per_branch = {
        bid: {
            "actor_calls": dict(d.actor_call_counts),
            "batches": d.batches,
            "events": d.events,
            "stop_reason": d.stop_reason,
            "unfired_in_horizon": d.unfired_in_horizon,
            "pending_beyond_horizon": len(d.pending_beyond_horizon),
        }
        for bid, d in ctx.run_result.diagnostics.items()
    }
    return {
        "live": config.is_live,
        "gateway_class": type(gw).__name__,
        "research_backend_class": type(config.research_backend).__name__,
        "model": gw.model_id,
        "wall_seconds": round(wall_seconds, 1),
        "actual_model_calls": gw.call_count,
        "model_calls_by_stage": gw.stage_call_counts(),
        "tokens_in": gw.total_tokens_in,
        "tokens_out": gw.total_tokens_out,
        "retries": gw.retries,
        "failed_calls": gw.failed_calls,
        "avg_latency_ms": int(sum(lat) / len(lat)) if lat else 0,
        "max_latency_ms": max(lat) if lat else 0,
        "http_requests": len(http_calls),
        "research": bundle.live_trace,
        "coverage": ctx.compiled.coverage_report.to_dict(),
        "branches": per_branch,
        "actor_invocations_total": len(ctx.run_result.actor_decisions),
    }


def _print_audit(audit: dict[str, Any]) -> None:
    print("\n--- Run audit ---")
    print(f"live: {audit['live']}  gateway: {audit['gateway_class']}  model: {audit['model']}")
    print(f"research backend: {audit['research_backend_class']}")
    print(f"wall clock: {audit['wall_seconds']}s")
    print(f"model calls: {audit['actual_model_calls']} {audit['model_calls_by_stage']}")
    print(
        f"tokens in/out: {audit['tokens_in']}/{audit['tokens_out']}  "
        f"retries: {audit['retries']}  failed: {audit['failed_calls']}"
    )
    print(f"latency avg/max ms: {audit['avg_latency_ms']}/{audit['max_latency_ms']}")
    print(f"live HTTP requests: {audit['http_requests']}")
    r = audit.get("research") or {}
    print(
        f"research: {len(r.get('queries', []))} queries, "
        f"{len(r.get('sources_fetched', []))} sources fetched, "
        f"{len(r.get('sources_rejected', []))} rejected, "
        f"{r.get('claim_count', 0)} claims; stop: {r.get('stop_reason', '')}"
    )
    print(f"actor invocations: {audit['actor_invocations_total']}")
    for bid, b in audit.get("branches", {}).items():
        calls = ", ".join(f"{a}×{n}" for a, n in sorted(b["actor_calls"].items()))
        print(
            f"  {bid}: {b['batches']} batches, {b['events']} events, "
            f"actor calls [{calls or 'none'}]; stop: {b['stop_reason']}"
        )
    _print_coverage(audit.get("coverage") or {})


def _print_coverage(cov: dict[str, Any]) -> None:
    if not cov:
        return
    t = cov.get("totals", {})
    print(
        f"coverage: {cov.get('coverage_verdict', '?')} — "
        f"{t.get('material', 0)}/{t.get('total', 0)} material; "
        f"included {t.get('included', 0)}, merged {t.get('merged', 0)}, "
        f"excluded {t.get('excluded', 0)}, uncertain {t.get('uncertain', 0)}, "
        f"unresolved {t.get('unresolved', 0)}"
    )
    for miss in cov.get("missing_material_candidates", []):
        print(f"  MISSING: {miss}")


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


def cmd_forecast(args: argparse.Namespace) -> int:
    as_of = datetime.fromisoformat(args.as_of)
    horizon = datetime.fromisoformat(args.horizon)
    process_started = datetime.now(as_of.tzinfo)
    archive_only = requires_archived_copy(as_of, process_started)
    mode = "PASTCAST (archive-only)" if archive_only else "NOWCAST (current pages admissible)"
    print(f"Retrieval mode: {mode}")
    print(f"Process started: {process_started.isoformat()}  requested as_of: {as_of.isoformat()}")
    out = Path(args.trace) if args.trace else None

    config = ForecastConfig.live(
        seed=args.seed,
        max_branches=args.max_branches,
        max_structures=args.max_structures,
        research_budget=ResearchBudget(
            max_rounds=args.research_rounds,
            max_queries=args.max_queries,
            max_seconds=args.research_seconds,
        ),
        model=args.model,
        budget=RunBudget(
            max_events=args.max_events,
            max_actor_calls=args.max_actor_calls,
        ),
    )
    if not config.is_live:  # the only gate: a "forecast" that is not live is not one
        print("forecast requires a live gateway and live research backend", file=sys.stderr)
        return 2

    start = time.monotonic()
    try:
        result, ctx = run_forecast(args.question, as_of, horizon, config)
    except ForecastRefused as refusal:
        # A refusal is a result about the world-supply pipeline, and it is the result
        # most worth reading. Writing only a traceback made the four questions that
        # refused the least diagnosable part of the system.
        wall = time.monotonic() - start
        diagnosis = RunDiagnosis(
            question=args.question,
            as_of=as_of,
            horizon=horizon,
            bundle=refusal.bundle,
            repair_log=refusal.repair_log,
            failure=refusal.__cause__ or refusal,
            failure_stage=refusal.stage,
            wall_seconds=wall,
            model_calls=config.gateway.call_count,
        )
        _write_diagnosis(out, diagnosis, refusal)
        print(f"REFUSED at {refusal.stage}: {refusal.__cause__ or refusal}", file=sys.stderr)
        for cause in diagnosis.root_cause():
            print(f"  root cause: {cause['cause']} — {cause['why']}", file=sys.stderr)
        if out is not None:
            print(f"  diagnosis: {out / 'diagnosis.json'}", file=sys.stderr)
        return 1
    wall = time.monotonic() - start

    forecast_hash = ""
    audit = _audit(config, ctx, wall)
    if out is not None:
        forecast_hash = ctx.write(out, sealed_names=args.seal, gateway_calls=config.gateway.calls)
        (out / "run_audit.json").write_text(canonical_json(audit) + "\n")
        (out / "diagnosis.json").write_text(
            canonical_json(
                RunDiagnosis(
                    question=args.question,
                    as_of=as_of,
                    horizon=horizon,
                    bundle=ctx.bundle,
                    compiled=ctx.compiled,
                    run_result=ctx.run_result,
                    repair_log=ctx.repair_log,
                    world_review=ctx.world_review,
                    wall_seconds=wall,
                    model_calls=config.gateway.call_count,
                ).as_dict()
            )
            + "\n"
        )
    _print_summary(result, forecast_hash, out)
    _print_audit(audit)
    return 0


def _write_diagnosis(out: Path | None, diagnosis: RunDiagnosis, refusal: ForecastRefused) -> None:
    """Write everything the refused run learned before it stopped."""

    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.json").write_text(canonical_json(diagnosis.as_dict()) + "\n")
    if refusal.bundle is not None:
        (out / "research_trace.json").write_text(
            canonical_json(refusal.bundle.live_trace or {}) + "\n"
        )
        (out / "evidence_store.json").write_text(
            canonical_json(
                [
                    {
                        "id": c.id,
                        "proposition": c.proposition,
                        "normalized_value": c.normalized_value,
                        "entities": list(c.entities),
                        "epistemic_type": c.epistemic_type.value,
                        "source_url": c.source_url,
                        "supporting_excerpt": c.supporting_excerpt,
                        "available_at": c.available_at.isoformat(),
                    }
                    for c in refusal.bundle.evidence_store.all()
                ]
            )
            + "\n"
        )
        (out / "compiled_world.json").write_text(
            canonical_json(diagnosis.world_compilation()) + "\n"
        )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show what each actor was actually sent and what it actually returned.

    Nothing is reconstructed: the prompts printed are the byte-exact strings recorded
    when they were sent.
    """

    run_dir = Path(args.run_directory)
    decisions_path = run_dir / "actor_decisions.jsonl"
    if not decisions_path.exists():
        print(f"no actor_decisions.jsonl in {run_dir}", file=sys.stderr)
        return 2
    records = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
    if args.actor:
        records = [r for r in records if r.get("actor_id") == args.actor]
    if args.branch:
        records = [r for r in records if r.get("branch_id") == args.branch]
    if not records:
        print("no actor decisions match that filter", file=sys.stderr)
        return 2

    if args.summary:
        return _print_invocation_summary(records)

    for rec in records[: args.limit]:
        print("=" * 78)
        print(
            f"{rec.get('actor_id')} @ {rec.get('branch_time')}  branch={rec.get('branch_id')}  "
            f"stage={rec.get('stage')}"
        )
        print(f"WOKEN BECAUSE : {rec.get('wake_reason')} — {rec.get('wake_detail')}")
        print(f"trigger events: {rec.get('trigger_event_ids')}")
        print(f"noticed        : {rec.get('noticed_observation_ids')}")
        print(f"memories       : {rec.get('retrieved_memory_ids')}")
        print(f"plan before    : {json.dumps(rec.get('plan_before'))}")
        print(f"disposition    : {rec.get('plan_disposition')}")
        print(f"plan after     : {json.dumps(rec.get('plan_after'))}")
        print(f"intent         : {json.dumps(rec.get('intent'))}")
        print(f"world said     : {rec.get('validation_status')} — {rec.get('validation_reason')}")
        if args.prompts:
            print("\n--- EXACT PROMPT SENT ---")
            print((rec.get("decision_context") or {}).get("rendered_prompt") or "(not recorded)")
            print("\n--- EXACT RESPONSE ---")
            print(json.dumps((rec.get("decision_context") or {}).get("provider_response")))
        print()
    return 0


def _print_invocation_summary(records: list[dict[str, Any]]) -> int:
    """How often each actor was invoked, and why. The point of this view is that the
    counts differ — between actors and between branches — because different things
    happened to them."""

    per: dict[tuple[str, str], list[str]] = {}
    for r in records:
        per.setdefault((str(r.get("branch_id")), str(r.get("actor_id"))), []).append(
            str(r.get("wake_reason"))
        )
    print(f"{'branch':28s} {'actor':24s} {'calls':6s} reasons")
    for (branch, actor), reasons in sorted(per.items()):
        counts: dict[str, int] = {}
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
        print(f"{branch[:27]:28s} {actor[:23]:24s} {len(reasons):<6d} {detail}")
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sworldmodel")
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser("forecast", help="question-only live forecast")
    fc.add_argument("--question", required=True)
    fc.add_argument(
        "--as-of",
        required=True,
        help="ISO datetime cutoff. If it is in the past, every source must be "
        "retrieved as its archived capture at that time and un-archived sources are "
        "refused; pass the current time for a nowcast",
    )
    fc.add_argument("--horizon", required=True, help="ISO datetime resolution horizon")
    fc.add_argument("--model", default=None, help="override model id")
    fc.add_argument("--max-branches", type=int, default=6)
    fc.add_argument(
        "--max-structures",
        type=int,
        default=3,
        help="how many competing causal structures to simulate when the evidence "
        "leaves the structure open (1 takes the compiled structure as given)",
    )
    fc.add_argument("--max-queries", type=int, default=14)
    fc.add_argument("--research-rounds", type=int, default=3)
    fc.add_argument("--research-seconds", type=float, default=300.0)
    fc.add_argument("--max-events", type=int, default=600)
    fc.add_argument("--max-actor-calls", type=int, default=80)
    fc.add_argument("--seed", type=int, default=0)
    fc.add_argument("--trace", default=None, help="output directory for the replayable trace")
    fc.add_argument(
        "--seal",
        action="store_true",
        help="write pre-outcome artifacts under sealed names (for a blind pastcast)",
    )
    fc.set_defaults(func=cmd_forecast)

    ia = sub.add_parser("inspect", help="inspect actor invocations in a written trace")
    ia.add_argument("run_directory")
    ia.add_argument("--actor", default=None)
    ia.add_argument("--branch", default=None)
    ia.add_argument("--limit", type=int, default=20)
    ia.add_argument("--prompts", action="store_true", help="print exact prompts and responses")
    ia.add_argument("--summary", action="store_true", help="per-actor invocation counts and causes")
    ia.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
