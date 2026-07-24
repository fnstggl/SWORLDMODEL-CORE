"""Command-line entry point.

Subcommands:
  banxico run       run the Banxico pastcast and write the SEALED pre-outcome artifacts
  banxico evaluate  compare the sealed forecast to the known result (post-cutoff)
  synthetic run     run the synthetic-committee generalization proofs
  forecast          run an arbitrary corpus

Scenario-specific constants (the Banxico cutoff, corpus path) live here in the
evaluation harness, never in the core package.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .api import run_forecast
from .config import ForecastConfig
from .ids import sha256_hex
from .models import ForecastResult
from .research import CorpusResearchBackend

# --- Banxico evaluation harness constants (NOT core) -----------------------------
BANXICO_AS_OF = datetime.fromisoformat("2026-05-14T23:59:59-06:00")
_REPO_ROOT = Path(__file__).resolve().parents[2]
BANXICO_CORPUS = _REPO_ROOT / "evaluation" / "banxico" / "corpus"
BANXICO_ARTIFACTS = _REPO_ROOT / "artifacts" / "banxico_2026_06_25"
SYNTHETIC_ROOT = _REPO_ROOT / "evaluation" / "synthetic"


def _run_corpus(
    corpus_dir: Path, question: str, as_of: datetime, out_dir: Path | None, *, sealed: bool
) -> None:
    backend = CorpusResearchBackend(corpus_dir)
    bundle = backend.research(question, as_of, as_of)  # horizon/as_of come from the bundle
    effective_as_of = bundle.as_of or as_of
    config = ForecastConfig.offline(backend, trace_dir=None)
    result, ctx = run_forecast(question, effective_as_of, bundle.horizon, config)

    forecast_hash = ""
    if out_dir is not None:
        forecast_hash = ctx.write(out_dir, sealed_names=sealed, gateway_calls=config.gateway.calls)

    _print_summary(result, forecast_hash, out_dir)


def _print_summary(result: ForecastResult, forecast_hash: str, out_dir: Path | None) -> None:
    print(f"Question: {result.question}")
    print(f"Status: {result.status.value}")
    p = result.simulation_probability
    print(
        f"Simulation probability: {'—' if p is None else f'{p:.4f}'}  (source: {result.probability_source})"
    )
    print(f"Bounds: [{result.lower_bound:.4f}, {result.upper_bound:.4f}]")
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
        print(f"Pre-outcome forecast SHA-256: {forecast_hash}")


def cmd_banxico_run(_: argparse.Namespace) -> int:
    _run_corpus(
        BANXICO_CORPUS,
        "Will the Banxico Governing Board's June 25, 2026 interest-rate decision be a unanimous hold?",
        BANXICO_AS_OF,
        BANXICO_ARTIFACTS,
        sealed=True,
    )
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


def cmd_forecast(args: argparse.Namespace) -> int:
    as_of = datetime.fromisoformat(args.as_of)
    out = Path(args.trace) if args.trace else None
    _run_corpus(Path(args.corpus), args.question, as_of, out, sealed=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sworldmodel")
    sub = parser.add_subparsers(dest="command", required=True)

    banxico = sub.add_parser("banxico", help="Banxico vertical slice")
    bsub = banxico.add_subparsers(dest="sub", required=True)
    bsub.add_parser("run", help="run pastcast + seal artifacts").set_defaults(func=cmd_banxico_run)
    bsub.add_parser("evaluate", help="evaluate against known result").set_defaults(
        func=cmd_banxico_evaluate
    )

    synthetic = sub.add_parser("synthetic", help="generalization proofs")
    ssub = synthetic.add_subparsers(dest="sub", required=True)
    ssub.add_parser("run", help="run synthetic committees").set_defaults(func=cmd_synthetic_run)

    fc = sub.add_parser("forecast", help="run an arbitrary corpus")
    fc.add_argument("--corpus", required=True)
    fc.add_argument("--question", required=True)
    fc.add_argument("--as-of", required=True, help="ISO datetime")
    fc.add_argument("--trace", default=None, help="output dir for artifacts")
    fc.set_defaults(func=cmd_forecast)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
