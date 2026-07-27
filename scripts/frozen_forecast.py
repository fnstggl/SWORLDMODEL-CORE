#!/usr/bin/env python3
"""Run the COMPLETE canonical forecast route from a frozen evidence store.

This is the end-to-end inner-loop harness: everything after retrieval is the real
production path — ``api.forecast()`` itself, with its repair loop, pre-rollout world
review, structural assessment, the event-driven runtime, outcome aggregation, both
auditors, and the full trace write that the replay viewer reads. The ONLY substitution
is at the external boundary: claims come from a prior run's exported store instead of a
live retrieval pass. Nothing mocks the compiler, lowerer, scheduler, runtime, actor
interface, terminal evaluator or replay.

    PYTHONPATH=src python3 scripts/frozen_forecast.py \
        --store artifacts/.../evidence_store.json --mode semantic \
        --question "…" --as-of <iso> --horizon <iso> --out <dir>

Exit codes: 0 completed forecast; 1 refused (diagnosis written); 4 unexpected error.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _store_loader import load_store, prepare_run_dir  # noqa: E402

from sworldmodel.api import compile_for_mode, forecast  # noqa: E402
from sworldmodel.config import ForecastConfig  # noqa: E402
from sworldmodel.deepseek_gateway import DeepSeekGateway  # noqa: E402
from sworldmodel.diagnosis import ForecastRefused, RunDiagnosis  # noqa: E402
from sworldmodel.evidence import EvidenceStore  # noqa: E402
from sworldmodel.http import UrllibTransport  # noqa: E402
from sworldmodel.ids import canonical_json  # noqa: E402
from sworldmodel.research import assemble_bundle  # noqa: E402


class FrozenResearchBackend:
    """The retrieval boundary, replayed: real compile, frozen claims.

    ``research()`` compiles the world through the mode's REAL compiler (the same
    ``compile_for_mode`` production entry the repair loop and structural alternatives
    use) from a store loaded off disk. ``augment_targeted`` is absent on purpose: a
    frozen store cannot grow, so repair rounds recompile from the same evidence —
    exactly the fairness contract of the A/B.
    """

    is_live = True

    def __init__(self, store: EvidenceStore, config_ref: dict) -> None:
        self.store = store
        self._config_ref = config_ref

    def research(self, question: str, as_of: datetime, horizon: datetime):
        config = self._config_ref["config"]
        try:
            data = compile_for_mode(
                config,
                question,
                as_of,
                horizon,
                self.store.view(as_of),
            )
            # Assembly is INSIDE the guard. It refuses in its own right — a compilation
            # that declares no horizon, a world_spec the parser cannot read — and while
            # it sat outside, those refusals reached ``run_forecast`` carrying no
            # partials at all: ``_checkpoint_partial`` returned False, so the run was
            # filed as stage="research", the whole evidence record went unwritten, and
            # ``_replan_initial_compile`` — which exists precisely to give a
            # recompilable compile-stage refusal its registered repair — was never
            # reached. The compile stage does not end until the bundle exists.
            bundle = assemble_bundle(self.store, data)
        except Exception as exc:
            # Mirror live_research._compile: the initial compile runs inside
            # ``research()``, so without this a compile-stage refusal on the frozen
            # route is misfiled as stage="research" with no research artifacts. The
            # frozen route's research IS the loaded store — attach it so the caller
            # checkpoints it and names the true stage. Every exception type: a raw
            # parser ValueError discarded the record exactly like a gate refusal did.
            exc.partial_live_trace = {  # type: ignore[attr-defined]
                "frozen_store": True,
                "claim_count": len(self.store.all()),
            }
            exc.partial_evidence_store = self.store  # type: ignore[attr-defined]
            raise
        live_trace: dict = {"frozen_store": True, "claim_count": len(self.store.all())}
        if "_semantic" in data:
            live_trace["semantic_compilation"] = data["_semantic"]
        from dataclasses import replace

        return replace(bundle, live_trace=live_trace)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--horizon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("semantic", "direct"), default="semantic")
    ap.add_argument("--max-branches", type=int, default=4)
    ap.add_argument("--max-structures", type=int, default=2)
    args = ap.parse_args()

    as_of = datetime.fromisoformat(args.as_of)
    horizon = datetime.fromisoformat(args.horizon)
    out = Path(args.out)
    # Stamp the run and clear any prior run's pipeline artifacts, so a refusal here
    # can never leave a stale forecast.json for downstream readers to score as fresh.
    prepare_run_dir(out, question=args.question, as_of=as_of, horizon=horizon, mode=args.mode)

    store = load_store(Path(args.store))
    gateway = DeepSeekGateway(UrllibTransport())
    ref: dict = {}
    backend = FrozenResearchBackend(store, ref)
    config = ForecastConfig(
        gateway=gateway,
        research_backend=backend,  # type: ignore[arg-type]
        trace_dir=out,
        max_branches=args.max_branches,
        max_structures=args.max_structures,
        compiler_mode=args.mode,
    )
    ref["config"] = config

    t0 = time.monotonic()
    try:
        result = forecast(args.question, as_of, horizon, config)
    except ForecastRefused as refusal:
        wall = time.monotonic() - t0
        diagnosis = RunDiagnosis(
            question=args.question,
            as_of=as_of,
            horizon=horizon,
            bundle=refusal.bundle,
            repair_log=refusal.repair_log,
            failure=refusal.__cause__ or refusal,
            failure_stage=refusal.stage,
            wall_seconds=wall,
            model_calls=gateway.call_count,
            compiler_mode=args.mode,
        )
        (out / "diagnosis.json").write_text(canonical_json(diagnosis.as_dict()) + "\n")
        details = getattr(refusal.__cause__, "details", {}) or {}
        print(f"REFUSED [{args.mode}] at {refusal.stage}: {refusal.__cause__ or refusal}")
        print(f"failure: {details.get('failure')}")
        # The classification, and — as loudly — what this run could NOT observe. A run
        # of this harness named `archive_coverage_failure` and `discovery_failure` about
        # a store whose claims were all admissible and a backend that issues no queries;
        # both lines were only ever in the JSON, so nobody reading the console saw the
        # diagnosis at all, let alone that two of its three causes rested on counters no
        # stage of the run had written.
        for cause in diagnosis.root_cause():
            print(f"  root cause: {cause['cause']} — {cause['why']}")
        for miss in diagnosis.unobserved():
            print(
                f"  NOT OBSERVED: {miss['measurement']} — {miss['why']}; "
                f"{miss['cause_neither_inferred_nor_ruled_out']} could be neither "
                "inferred nor ruled out"
            )
        print(f"repair attempts: {len(diagnosis.integrity_and_grounding()['repair_attempts'])}")
        print(f"wall: {wall:.0f}s  calls: {gateway.call_count}")
        return 1
    except Exception as stopped:  # noqa: BLE001 — every ending owes a diagnosis
        wall = time.monotonic() - t0
        diagnosis = RunDiagnosis(
            question=args.question,
            as_of=as_of,
            horizon=horizon,
            failure=stopped,
            failure_stage="simulation",
            wall_seconds=wall,
            model_calls=gateway.call_count,
            compiler_mode=args.mode,
        )
        (out / "diagnosis.json").write_text(canonical_json(diagnosis.as_dict()) + "\n")
        print(f"FAILED [{args.mode}] in simulation after {wall:.0f}s: {stopped}")
        return 4
    wall = time.monotonic() - t0
    p = result.simulation_probability
    print(f"COMPLETED [{args.mode}]  status={result.status.value}")
    print(f"probability: {'—' if p is None else f'{p:.4f}'} ({result.probability_source})")
    print(
        f"mass: YES {result.resolved_yes_mass:.3f} / NO {result.resolved_no_mass:.3f} / "
        f"unresolved {result.unresolved_mass:.3f}"
    )
    print(
        f"branches: {len(result.branch_outcomes)}  wall: {wall:.0f}s  calls: {gateway.call_count}"
    )
    print(f"tokens: {gateway.total_tokens_in}/{gateway.total_tokens_out}")
    for b in result.branch_outcomes:
        state = b.outcome if b.resolved else f"UNRESOLVED({b.unresolved_reason})"
        print(f"  - {b.branch_id} w={b.weight:.3f} -> {state}")
    print(f"trace: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
