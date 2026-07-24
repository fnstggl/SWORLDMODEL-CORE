"""Command-line entry point.

Primary (production) command — question only, live research + live DeepSeek:

    sworldmodel forecast --question "..." --as-of <iso> --horizon <iso>

Evaluation / test utilities (never the live path):

    forecast --corpus DIR ...   run a prepared corpus with the deterministic reasoner
    banxico run | evaluate      sealed corpus pastcast fixture + post-outcome eval
    banxico-live                question-only LIVE Banxico acceptance run
    synthetic run               corpus generalization proofs

Scenario constants (the Banxico cutoff/horizon, corpus path) live here in the
evaluation harness, never in the core package.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .api import run_forecast
from .config import ForecastConfig
from .ids import canonical_json, sha256_hex
from .live_research import ResearchBudget
from .models import ForecastResult
from .research import CorpusResearchBackend
from .tracing import TraceContext

# --- Banxico evaluation harness constants (NOT core) -----------------------------
BANXICO_AS_OF = datetime.fromisoformat("2026-05-14T23:59:59-06:00")
BANXICO_HORIZON = datetime.fromisoformat("2026-06-25T23:59:59-06:00")
BANXICO_QUESTION = (
    "Will the Bank of Mexico (Banxico) Governing Board unanimously hold its policy "
    "interest rate at its June 25, 2026 monetary policy decision (a 5-0 vote to keep "
    "the rate unchanged)?"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
BANXICO_CORPUS = _REPO_ROOT / "evaluation" / "banxico" / "corpus"
BANXICO_ARTIFACTS = _REPO_ROOT / "artifacts" / "banxico_2026_06_25"
BANXICO_LIVE_ARTIFACTS = _REPO_ROOT / "artifacts" / "banxico_live"
SYNTHETIC_ROOT = _REPO_ROOT / "evaluation" / "synthetic"


# ---------------------------------------------------------------------------
# Shared printing
# ---------------------------------------------------------------------------


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
    print(f"Bounds: [{lb:.4f}, {ub:.4f}]")
    print(
        f"Mass — resolved YES {result.resolved_yes_mass:.4f}, NO {result.resolved_no_mass:.4f}, "
        f"unresolved {result.unresolved_mass:.4f}"
    )
    print(
        f"Integrity: {result.integrity_manifest.integrity_verdict.value} "
        f"(seats expected {result.integrity_manifest.expected_voting_seats}, "
        f"represented {result.integrity_manifest.represented_voting_seats})"
    )
    print(f"Branches: {len(result.branch_outcomes)}  |  model calls: {result.model_call_count}")
    for b in result.branch_outcomes:
        votes = ", ".join(f"{a}={o}" for a, o in b.votes)
        state = b.outcome if b.resolved else f"UNRESOLVED({b.unresolved_reason})"
        print(f"  - {b.branch_id} w={b.weight:.4f} [{votes}] -> {state}")
    if out_dir is not None:
        print(f"Artifacts: {out_dir}")
        if forecast_hash:
            print(f"Forecast SHA-256: {forecast_hash}")


def _live_audit(config: ForecastConfig, ctx: TraceContext) -> dict[str, Any]:
    gw = config.gateway
    bundle = ctx.bundle
    transport = getattr(config.research_backend, "transport", None)
    http_calls = getattr(transport, "calls", [])
    lat = gw.latencies_ms
    return {
        "prepared_corpus_read": False,
        "deterministic_gateway_used": not gw.is_live,
        "live": config.is_live,
        "model": gw.model_id,
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
    }


def _print_live_audit(audit: dict[str, Any]) -> None:
    print("\n--- Live audit ---")
    print(f"prepared corpus read: {audit['prepared_corpus_read']}")
    print(f"deterministic gateway used: {audit['deterministic_gateway_used']}")
    print(f"actual DeepSeek calls: {audit['actual_model_calls']} {audit['model_calls_by_stage']}")
    print(
        f"tokens in/out: {audit['tokens_in']}/{audit['tokens_out']}  retries: {audit['retries']}  failed: {audit['failed_calls']}"
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
    # Show what verified reality entered the world, was merged, or excluded and why.
    for c in cov.get("candidates", []):
        disp = c.get("disposition")
        if disp in ("included", "merged"):
            print(f"  [{disp}] {c['kind']}: {c['identity']} -> {','.join(c['causal_uses']) or '—'}")
        elif disp in ("excluded_irrelevant", "uncertain", "required_but_unresolved"):
            print(f"  [{disp}] {c['kind']}: {c['identity']} — {c['reason']}")


# ---------------------------------------------------------------------------
# Corpus (offline) utility path
# ---------------------------------------------------------------------------


def _run_corpus(
    corpus_dir: Path, question: str, as_of: datetime, out_dir: Path | None, *, sealed: bool
) -> ForecastResult:
    backend = CorpusResearchBackend(corpus_dir)
    bundle = backend.research(question, as_of, as_of)
    effective_as_of = bundle.as_of or as_of
    config = ForecastConfig.offline(backend, trace_dir=None)
    result, ctx = run_forecast(question, effective_as_of, bundle.horizon, config)
    forecast_hash = ""
    if out_dir is not None:
        forecast_hash = ctx.write(out_dir, sealed_names=sealed, gateway_calls=config.gateway.calls)
    _print_summary(result, forecast_hash, out_dir)
    return result


# ---------------------------------------------------------------------------
# Live (production) path
# ---------------------------------------------------------------------------


def _budget_from_args(args: argparse.Namespace) -> ResearchBudget:
    return ResearchBudget(
        max_rounds=getattr(args, "research_rounds", 3),
        max_queries=getattr(args, "max_queries", 12),
        max_seconds=getattr(args, "research_seconds", 240.0),
    )


def _run_live(
    question: str,
    as_of: datetime,
    horizon: datetime,
    out_dir: Path | None,
    *,
    args: argparse.Namespace,
    sealed: bool,
    now: datetime | None = None,
) -> tuple[ForecastResult, dict[str, Any]]:
    config = ForecastConfig.live(
        seed=getattr(args, "seed", 0),
        max_branches=getattr(args, "max_branches", 6),
        research_budget=_budget_from_args(args),
        now=now,
        model=getattr(args, "model", None),
    )
    if not config.is_live:  # gating: refuse to run a "live" command without a live config
        raise RuntimeError("live command requires a live gateway + live research backend")
    result, ctx = run_forecast(question, as_of, horizon, config)
    forecast_hash = ""
    audit = _live_audit(config, ctx)
    if out_dir is not None:
        forecast_hash = ctx.write(out_dir, sealed_names=sealed, gateway_calls=config.gateway.calls)
        (out_dir / "live_audit.json").write_text(canonical_json(audit) + "\n")
    _print_summary(result, forecast_hash, out_dir)
    _print_live_audit(audit)
    return result, audit


def cmd_inspect_actors(args: argparse.Namespace) -> int:
    """Print the exact prompts each actor received, plus a comparison report.

    Reads a written run directory; nothing is reconstructed — the prompts printed are
    the byte-exact strings recorded when they were sent to the provider.
    """

    run_dir = Path(args.run_directory)
    decisions_path = run_dir / "actor_decisions.jsonl"
    if not decisions_path.exists():
        print(f"no actor_decisions.jsonl in {run_dir}", file=sys.stderr)
        return 2
    records = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
    stage = args.stage
    chosen: dict[str, dict[str, Any]] = {}
    for r in records:
        if stage and r.get("stage") != stage:
            continue
        chosen.setdefault(str(r.get("actor_id")), r)
    if not chosen:
        print(f"no actor decisions recorded for stage {stage!r}", file=sys.stderr)
        return 2

    grounding_path = run_dir / "actor_grounding.json"
    grounding = json.loads(grounding_path.read_text()) if grounding_path.exists() else {}
    profiles = {p["actor_id"]: p for p in grounding.get("profiles", [])}

    for aid, rec in chosen.items():
        print("=" * 78)
        print(f"ACTOR {aid}  ({rec.get('canonical_identity')})  stage={rec.get('stage')}")
        print("=" * 78)
        prof = profiles.get(aid, {})
        if prof:
            print(f"previous_observed_action : {prof.get('previous_observed_action')}")
            incl = prof.get("current_evidence_grounded_inclination") or {}
            print(f"current_inclination      : {incl.get('content')} [{incl.get('provenance')}]")
            print(f"direct_statements        : {len(prof.get('direct_statements', []))}")
            print(f"evidence_claim_ids       : {prof.get('claim_ids')}")
            print(f"missing_information      : {prof.get('missing_information')}")
        print(f"retrieved_memory_ids     : {rec.get('retrieved_memory_ids')}")
        print(f"prompt_hash              : {rec.get('prompt_hash')}")
        print(f"intent                   : {json.dumps(rec.get('intent'))}")
        print("\n--- EXACT PROMPT SENT TO THE PROVIDER ---")
        print(rec.get("exact_prompt") or "(not recorded)")
        print("\n--- EXACT PROVIDER RESPONSE ---")
        print(json.dumps(rec.get("provider_response")))
        print()

    _print_actor_comparison(chosen, profiles)
    return 0


def _print_actor_comparison(
    chosen: dict[str, dict[str, Any]], profiles: dict[str, dict[str, Any]]
) -> None:
    print("=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(
        f"{'actor':28s} {'previous action (verified)':34s} "
        f"{'inclination (inferred)':34s} {'stmts':6s} {'claims':6s}"
    )
    for aid in chosen:
        p = profiles.get(aid, {})
        incl = (p.get("current_evidence_grounded_inclination") or {}).get("content", "")
        prev = str(p.get("previous_observed_action") or "—")
        print(
            f"{aid:28s} {prev[:33]:34s} {str(incl)[:33]:34s} "
            f"{len(p.get('direct_statements', [])):<6d} {len(p.get('claim_ids', [])):<6d}"
        )
    # How much of each prompt is actor-specific rather than shared boilerplate.
    prompts = {a: r.get("exact_prompt") or "" for a, r in chosen.items()}
    ids = list(prompts)
    if len(ids) > 1:
        import difflib

        base = prompts[ids[0]]
        print(f"\nprompt similarity vs {ids[0]}:")
        for aid in ids[1:]:
            ratio = difflib.SequenceMatcher(None, base, prompts[aid]).ratio()
            print(f"  {aid:28s} {ratio:.4f}")


def cmd_forecast(args: argparse.Namespace) -> int:
    as_of = datetime.fromisoformat(args.as_of)
    out = Path(args.trace) if args.trace else None
    if args.corpus:  # offline test utility
        _run_corpus(Path(args.corpus), args.question, as_of, out, sealed=False)
        return 0
    if not args.horizon:
        print("--horizon is required for a live forecast", file=sys.stderr)
        return 2
    horizon = datetime.fromisoformat(args.horizon)
    _run_live(args.question, as_of, horizon, out, args=args, sealed=False)
    return 0


def cmd_banxico_live(args: argparse.Namespace) -> int:
    result, audit = _run_live(
        BANXICO_QUESTION,
        BANXICO_AS_OF,
        BANXICO_HORIZON,
        BANXICO_LIVE_ARTIFACTS,
        args=args,
        sealed=True,
    )
    # Acceptance proof lines (do not modify the sealed pre-outcome files).
    proof = {
        "prepared_corpus_read": audit["prepared_corpus_read"],
        "deterministic_gateway_used": audit["deterministic_gateway_used"],
        "actual_deepseek_calls>0": audit["actual_model_calls"] > 0,
        "live_http_retrievals>0": audit["http_requests"] > 0,
        "official_sources_fetched>0": len((audit.get("research") or {}).get("sources_fetched", []))
        > 0,
        "world_compiler_llm_calls>0": audit["model_calls_by_stage"].get("compile_reality", 0) > 0,
        "actor_llm_calls>0": audit["model_calls_by_stage"].get("actor_decision", 0) > 0,
        "forecast_source": result.probability_source,
    }
    (BANXICO_LIVE_ARTIFACTS / "acceptance.json").write_text(canonical_json(proof) + "\n")
    print("\n--- Acceptance ---")
    for k, v in proof.items():
        print(f"{k}: {v}")
    return 0


# ---------------------------------------------------------------------------
# Corpus fixtures (kept as test utilities)
# ---------------------------------------------------------------------------


def cmd_banxico_run(_: argparse.Namespace) -> int:
    _run_corpus(BANXICO_CORPUS, BANXICO_QUESTION, BANXICO_AS_OF, BANXICO_ARTIFACTS, sealed=True)
    return 0


def cmd_banxico_evaluate(_: argparse.Namespace) -> int:
    forecast_path = BANXICO_ARTIFACTS / "pre_outcome_forecast.json"
    if not forecast_path.exists():
        print("No sealed pre-outcome forecast found. Run `banxico run` first.", file=sys.stderr)
        return 2
    forecast_text = forecast_path.read_text().rstrip("\n")
    sealed = json.loads(forecast_text)
    seal_hash = sha256_hex(forecast_text)
    backend = CorpusResearchBackend(BANXICO_CORPUS)
    bundle = backend.research("banxico", BANXICO_AS_OF, BANXICO_AS_OF)
    outcome = bundle.outcome
    if outcome is None:
        print("Corpus has no post-cutoff outcome to evaluate against.", file=sys.stderr)
        return 2
    actual_yes = 1.0 if str(outcome["resolved"]).upper() == "YES" else 0.0
    p = sealed.get("simulation_probability")
    lines = [
        "# Banxico post-outcome evaluation\n",
        f"Sealed pre-outcome forecast SHA-256: `{seal_hash}`\n",
        f"Known result (revealed only post-cutoff): **{outcome['resolved']}** — {outcome.get('description', '')}\n",
        f"Simulation probability: {'—' if p is None else f'{p:.4f}'}\n",
        f"Supported bounds: [{sealed['lower_bound']:.4f}, {sealed['upper_bound']:.4f}]\n",
    ]
    if p is not None:
        brier = (p - actual_yes) ** 2
        lines.append(f"Brier score (simulation vs. actual): **{brier:.4f}**\n")
        direction = "correct" if (p >= 0.5) == (actual_yes >= 0.5) else "incorrect"
        lines.append(f"Directional call: **{direction}**\n")
    lines.append(
        "\n_This evaluation reads the sealed forecast; it does not modify the pre-outcome files._\n"
    )
    (BANXICO_ARTIFACTS / "post_outcome_evaluation.md").write_text("".join(lines))
    print("".join(lines))
    return 0


def cmd_synthetic_run(_: argparse.Namespace) -> int:
    if not SYNTHETIC_ROOT.exists():
        print("No synthetic scenarios found.", file=sys.stderr)
        return 2
    rc = 0
    for corpus_dir in sorted(SYNTHETIC_ROOT.iterdir()):
        if not (corpus_dir / "corpus.json").exists():
            continue
        print(f"\n=== synthetic: {corpus_dir.name} ===")
        try:
            _run_corpus(
                corpus_dir, f"synthetic:{corpus_dir.name}", BANXICO_AS_OF, None, sealed=False
            )
        except Exception as exc:  # noqa: BLE001 - surface refusals honestly
            print(f"  refused/failed: {type(exc).__name__}: {exc}")
            rc = 1
    return rc


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="deepseek", help="live model provider (deepseek)")
    p.add_argument("--research", default="live", help="research mode (live)")
    p.add_argument("--model", default=None, help="override model id")
    p.add_argument("--max-branches", type=int, default=6)
    p.add_argument("--max-queries", type=int, default=12)
    p.add_argument("--research-rounds", type=int, default=3)
    p.add_argument("--research-seconds", type=float, default=240.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace", default=None, help="output dir for artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sworldmodel")
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser("forecast", help="question-only live forecast (or --corpus test utility)")
    fc.add_argument("--question", required=True)
    fc.add_argument("--as-of", required=True, help="ISO datetime")
    fc.add_argument("--horizon", default=None, help="ISO datetime (required for live)")
    fc.add_argument("--corpus", default=None, help="offline test utility: run a prepared corpus")
    _add_live_args(fc)
    fc.set_defaults(func=cmd_forecast)

    bl = sub.add_parser("banxico-live", help="question-only LIVE Banxico acceptance run")
    _add_live_args(bl)
    bl.set_defaults(func=cmd_banxico_live)

    ia = sub.add_parser("inspect-actors", help="show the exact prompt each actor received")
    ia.add_argument("run_directory", help="a written run/trace directory")
    ia.add_argument("--stage", default="decision", help="protocol stage (default: decision)")
    ia.set_defaults(func=cmd_inspect_actors)

    banxico = sub.add_parser("banxico", help="Banxico corpus fixture (test utility)")
    bsub = banxico.add_subparsers(dest="sub", required=True)
    bsub.add_parser("run", help="run pastcast + seal artifacts").set_defaults(func=cmd_banxico_run)
    bsub.add_parser("evaluate", help="evaluate against known result").set_defaults(
        func=cmd_banxico_evaluate
    )

    synthetic = sub.add_parser("synthetic", help="corpus generalization proofs")
    ssub = synthetic.add_subparsers(dest="sub", required=True)
    ssub.add_parser("run", help="run synthetic committees").set_defaults(func=cmd_synthetic_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
