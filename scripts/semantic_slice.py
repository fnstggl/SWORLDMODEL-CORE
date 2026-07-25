#!/usr/bin/env python3
"""Slice harness: run the semantic compiler path on one frozen evidence store.

This is the fast inner loop for the compiler boundary. It exercises the REAL production
functions — the planner and reviewer prompts, the parser, the validator, the lowerer,
`assemble_bundle` and every `compile_world` gate — on a frozen evidence store, without
re-running research or the simulation. No mocks stand in for any of those stages; the
only substitution is the input: claims come from a prior run's exported store instead
of a live retrieval pass.

The exported store is a reduced projection (id, proposition, value, entities,
availability, url, excerpt). Reconstruction keeps every one of those real and defaults
only the provenance decoration (source_id from the URL host, published_at from
available_at) — stated here so nobody mistakes the harness for a full-fidelity replay
of research.

    PYTHONPATH=src python3 scripts/semantic_slice.py \
        --store artifacts/acceptance/individual/run_trace/evidence_store.json \
        --question "…" --as-of 2026-07-25T19:36:02+00:00 --horizon 2026-09-15T23:59:59+00:00 \
        --out /tmp/slice_individual

Stages are reported one by one: plan → review → validate → lower → gates. The exit
code is 0 only when every stage stands.
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

from sworldmodel.deepseek_gateway import DeepSeekGateway  # noqa: E402
from sworldmodel.errors import SWorldModelError, WorldIntegrityError  # noqa: E402
from sworldmodel.evidence import EvidenceClaim, EvidenceStore  # noqa: E402
from sworldmodel.http import UrllibTransport  # noqa: E402
from sworldmodel.models import (  # noqa: E402
    AuthorityLevel,
    EpistemicType,
    ResolutionContract,
    SourceType,
)
from sworldmodel.research import assemble_bundle  # noqa: E402
from sworldmodel.semantic_compile import semantic_compile_live  # noqa: E402
from sworldmodel.world_compiler import compile_world  # noqa: E402


def load_store(path: Path, as_of: datetime) -> EvidenceStore:
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


def _metrics(gateway, *, mode: str, outcome: str, failure: str, seconds: float) -> dict:
    by_kind: dict[str, int] = {}
    for c in gateway.calls:
        by_kind[c.task_kind] = by_kind.get(c.task_kind, 0) + 1
    return {
        "mode": mode,
        "outcome": outcome,
        "failure": failure,
        "compile_seconds": round(seconds, 1),
        "llm_calls": len(gateway.calls),
        "calls_by_kind": by_kind,
        "tokens_in": gateway.total_tokens_in,
        "tokens_out": gateway.total_tokens_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--horizon", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mode",
        choices=("semantic", "direct"),
        default="semantic",
        help="which compiler to exercise on the same frozen store — the A/B switch",
    )
    args = ap.parse_args()

    as_of = datetime.fromisoformat(args.as_of)
    horizon = datetime.fromisoformat(args.horizon)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    store = load_store(Path(args.store), as_of)
    view = store.view(as_of)
    print(f"[store]   {len(store.all())} claims, {len(view.available())} admissible at cutoff")

    gateway = DeepSeekGateway(UrllibTransport())
    t0 = time.monotonic()
    try:
        if args.mode == "direct":
            from sworldmodel.world_compiler import compile_world_spec_live

            compilation, _resp = compile_world_spec_live(
                gateway, args.question, as_of, horizon, view
            )
        else:
            compilation, _resp = semantic_compile_live(
                gateway, args.question, as_of, horizon, view
            )
    except (WorldIntegrityError, SWorldModelError) as exc:
        stage = str(getattr(exc, "details", {}).get("failure") or "unknown")
        seconds = time.monotonic() - t0
        print(f"[REFUSED] {args.mode} path stopped: {stage}")
        print(f"          {exc}")
        (out / "refusal.json").write_text(
            json.dumps({"failure": stage, "message": str(exc), "details": {
                k: v for k, v in getattr(exc, "details", {}).items()
                if isinstance(v, (str, int, bool, list))
            }}, indent=1, default=str)
        )
        (out / "metrics.json").write_text(
            json.dumps(
                _metrics(gateway, mode=args.mode, outcome="refused_compile",
                         failure=stage, seconds=seconds),
                indent=1,
            )
        )
        return 2
    plan_seconds = time.monotonic() - t0

    semantic = compilation.get("_semantic") or {}
    review = semantic.get("review") or {}
    print(f"[plan]    ok in {plan_seconds:.1f}s — review verdict: {review.get('verdict')}")
    for r in (review.get("reasons") or [])[:6]:
        print(f"          · {r}")
    (out / "semantic_plan.json").write_text(
        json.dumps(semantic.get("plan"), indent=1, default=str)
    )
    (out / "semantic_map.json").write_text(
        json.dumps(semantic.get("mapping"), indent=1, default=str)
    )
    (out / "lowered_compilation.json").write_text(
        json.dumps({k: v for k, v in compilation.items() if k != "_semantic"}, indent=1, default=str)
    )

    spec_dict = compilation["world_spec"]
    print(
        f"[lower]   entities={len(spec_dict['entities'])} actions={len(spec_dict['actions'])} "
        f"nodes={len(spec_dict['process']['nodes'])} externals={len(spec_dict['external_processes'])} "
        f"fields={len(spec_dict['fields'])} uncertainties={len(compilation['uncertainties'])}"
    )

    data = {
        "world_spec": compilation["world_spec"],
        "uncertainties": compilation["uncertainties"],
        "world_facts": compilation["world_facts"],
        "required_reality_facts": compilation["required_reality_facts"],
        "reality": {
            "subject_entity": compilation.get("subject_entity"),
            "resolution_units": compilation.get("resolution_units"),
            "target_outcome": compilation.get("target_outcome"),
            "expected_participants": compilation.get("expected_participants"),
            "as_of": as_of.isoformat(),
            "horizon": horizon.isoformat(),
        },
    }
    try:
        bundle = assemble_bundle(store, data)
        contract = ResolutionContract(
            question=args.question,
            as_of=as_of,
            horizon=horizon,
            subject_entity=bundle.subject_entity,
            resolution_units=bundle.resolution_units,
            terminal=bundle.spec.terminal,
            target_outcome=bundle.target_outcome,
            expected_participants=bundle.expected_participants,
        )
        compiled = compile_world(
            contract,
            view,
            bundle.spec,
            bundle.uncertainties,
            bundle.world_facts,
            gateway=gateway,
            seed=0,
            max_branches=4,
        )
    except SWorldModelError as exc:
        details = getattr(exc, "details", {}) or {}
        print(f"[GATES]   refused: {details.get('failure') or type(exc).__name__}")
        print(f"          {exc}")
        (out / "gate_refusal.json").write_text(
            json.dumps({"failure": details.get("failure"), "message": str(exc)}, indent=1, default=str)
        )
        (out / "metrics.json").write_text(
            json.dumps(
                _metrics(gateway, mode=args.mode, outcome="refused_gates",
                         failure=str(details.get("failure") or type(exc).__name__),
                         seconds=time.monotonic() - t0),
                indent=1,
            )
        )
        return 3
    print(
        f"[gates]   ALL PASS — compiled world stands "
        f"(branches possible: {len(compiled.scenario_set.scenarios)})"
    )
    (out / "metrics.json").write_text(
        json.dumps(
            _metrics(gateway, mode=args.mode, outcome="compiled",
                     failure="", seconds=time.monotonic() - t0),
            indent=1,
        )
    )
    print(f"[done]    artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
