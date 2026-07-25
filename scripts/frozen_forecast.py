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
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sworldmodel.api import compile_for_mode, forecast  # noqa: E402
from sworldmodel.config import ForecastConfig  # noqa: E402
from sworldmodel.deepseek_gateway import DeepSeekGateway  # noqa: E402
from sworldmodel.diagnosis import ForecastRefused, RunDiagnosis  # noqa: E402
from sworldmodel.evidence import EvidenceClaim, EvidenceStore  # noqa: E402
from sworldmodel.http import UrllibTransport  # noqa: E402
from sworldmodel.ids import canonical_json  # noqa: E402
from sworldmodel.models import (  # noqa: E402
    AuthorityLevel,
    EpistemicType,
    SourceType,
)
from sworldmodel.research import assemble_bundle  # noqa: E402


def load_store(path: Path) -> EvidenceStore:
    raw = json.loads(path.read_text())
    store = EvidenceStore()
    for c in raw:
        available = datetime.fromisoformat(c["available_at"])
        url = str(c.get("source_url") or "")
        host = urlparse(url).hostname or "unknown"
        store.add(
            EvidenceClaim(
                id=str(c["id"]),
                proposition=str(c["proposition"]),
                normalized_value=str(c.get("normalized_value") or ""),
                entities=tuple(c.get("entities") or ()),
                valid_from=available,
                valid_until=None,
                published_at=available,
                available_at=available,
                source_id=host,
                source_url=url,
                source_title=host,
                source_type=SourceType("contemporaneous_reporting"),
                authority_level=AuthorityLevel["MEDIUM"],
                supporting_excerpt=str(c.get("supporting_excerpt") or ""),
                lineage_event_id=f"ev_{c['id']}",
                epistemic_type=EpistemicType(str(c.get("epistemic_type") or "observation")),
                confidence=0.8,
                retrieved_at=available,
            )
        )
    return store


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
        data = compile_for_mode(
            config,
            question,
            as_of,
            horizon,
            self.store.view(as_of),
        )
        bundle = assemble_bundle(self.store, data)
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
    out.mkdir(parents=True, exist_ok=True)

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
        print(f"wall: {wall:.0f}s  calls: {gateway.call_count}")
        return 1
    wall = time.monotonic() - t0
    p = result.simulation_probability
    print(f"COMPLETED [{args.mode}]  status={result.status.value}")
    print(f"probability: {'—' if p is None else f'{p:.4f}'} ({result.probability_source})")
    print(
        f"mass: YES {result.resolved_yes_mass:.3f} / NO {result.resolved_no_mass:.3f} / "
        f"unresolved {result.unresolved_mass:.3f}"
    )
    print(f"branches: {len(result.branch_outcomes)}  wall: {wall:.0f}s  calls: {gateway.call_count}")
    print(f"tokens: {gateway.total_tokens_in}/{gateway.total_tokens_out}")
    for b in result.branch_outcomes:
        state = b.outcome if b.resolved else f"UNRESOLVED({b.unresolved_reason})"
        print(f"  - {b.branch_id} w={b.weight:.3f} -> {state}")
    print(f"trace: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
