"""Regression tests for the shared frozen-store harness loader (scripts/_store_loader.py).

The loader is the fidelity boundary of every frozen replay: ``render_evidence`` sorts
claims by descending authority and truncates, so a loader that flattens provenance
changes which claims the compiler ever sees. These tests pin the contract: a
full-fidelity export round-trips every provenance field intact; a legacy 8-field
export falls back to the historical defaults and says so loudly.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "harness_store_loader", _SCRIPTS / "_store_loader.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loader = _load_module()

T = datetime.fromisoformat


FULL_CLAIM = {
    "id": "c_official",
    "proposition": "The MPC voted 5-4 to hold Bank Rate at 4.0%",
    "normalized_value": "hold",
    "entities": ["Bank of England", "Andrew Bailey"],
    "epistemic_type": "observation",
    "source_url": "https://www.bankofengland.co.uk/monetary-policy-summary",
    "supporting_excerpt": "The Committee voted by a majority of 5-4 to maintain Bank Rate.",
    "available_at": "2026-06-19T12:05:00+00:00",
    # Full-fidelity provenance the extended export writes:
    "published_at": "2026-06-19T12:00:00+00:00",
    "valid_from": "2026-06-19T11:00:00+00:00",
    "valid_until": "2026-08-06T11:00:00+00:00",
    "source_id": "bankofengland.co.uk/mps",
    "source_type": "official_institutional",
    "authority_level": 4,
    "confidence": 0.97,
    "retrieved_at": "2026-07-01T09:00:00+00:00",
    "lineage_event_id": "ev_mpc_june_decision",
}

LEGACY_CLAIM = {
    "id": "c_blog",
    "proposition": "A commentator speculates the next move is a cut",
    "normalized_value": "speculation",
    "entities": [],
    "epistemic_type": "observation",
    "source_url": "https://example-blog.net/rates-hot-take",
    "supporting_excerpt": "I reckon they cut next time.",
    "available_at": "2026-06-20T00:00:00+00:00",
}


def _write_store(tmp_path: Path, claims: list[dict]) -> Path:
    p = tmp_path / "evidence_store.json"
    p.write_text(json.dumps(claims))
    return p


def test_full_fidelity_round_trip(tmp_path, capsys):
    store = loader.load_store(_write_store(tmp_path, [FULL_CLAIM]))
    (claim,) = store.all()

    # The provenance that drives authority ranking survives exactly.
    assert claim.authority_level == AuthorityLevel.AUTHORITATIVE
    assert claim.source_type == SourceType.OFFICIAL_INSTITUTIONAL
    # Real dates survive — none collapsed onto available_at.
    assert claim.published_at == T("2026-06-19T12:00:00+00:00")
    assert claim.valid_from == T("2026-06-19T11:00:00+00:00")
    assert claim.valid_until == T("2026-08-06T11:00:00+00:00")
    assert claim.available_at == T("2026-06-19T12:05:00+00:00")
    assert claim.retrieved_at == T("2026-07-01T09:00:00+00:00")
    assert claim.published_at != claim.available_at
    # Identity and confidence survive.
    assert claim.source_id == "bankofengland.co.uk/mps"
    assert claim.confidence == 0.97
    assert claim.lineage_event_id == "ev_mpc_june_decision"
    assert claim.epistemic_type == EpistemicType.OBSERVATION
    # A full-fidelity store is loaded silently: no legacy warning.
    assert "legacy store" not in capsys.readouterr().err


def test_explicit_null_valid_until_is_a_real_value_not_a_default(tmp_path, capsys):
    claim_dict = dict(FULL_CLAIM, valid_until=None, valid_from=None)
    store = loader.load_store(_write_store(tmp_path, [claim_dict]))
    (claim,) = store.all()
    assert claim.valid_until is None
    assert claim.valid_from is None
    assert "legacy store" not in capsys.readouterr().err


def test_defensive_enum_encodings(tmp_path):
    # The export format is owned by another change; accept names as well as values.
    claim_dict = dict(FULL_CLAIM, authority_level="HIGH", source_type="PRIMARY_RECORD")
    store = loader.load_store(_write_store(tmp_path, [claim_dict]))
    (claim,) = store.all()
    assert claim.authority_level == AuthorityLevel.HIGH
    assert claim.source_type == SourceType.PRIMARY_RECORD


def test_legacy_store_falls_back_with_loud_warning(tmp_path, capsys):
    store = loader.load_store(_write_store(tmp_path, [LEGACY_CLAIM]))
    (claim,) = store.all()

    available = T("2026-06-20T00:00:00+00:00")
    assert claim.authority_level == AuthorityLevel.MEDIUM
    assert claim.source_type == SourceType.CONTEMPORANEOUS_REPORTING
    assert claim.published_at == available
    assert claim.valid_from == available
    assert claim.valid_until is None
    assert claim.retrieved_at == available
    assert claim.source_id == "example-blog.net"
    assert claim.confidence == 0.8
    assert claim.lineage_event_id == "ev_c_blog"

    err = capsys.readouterr().err
    assert (
        "legacy store: provenance defaulted (authority ranking will differ from the live run)"
        in err
    )
    # One line, not one per claim.
    assert err.count("legacy store") == 1


def test_mixed_store_keeps_full_claims_and_warns_once(tmp_path, capsys):
    store = loader.load_store(_write_store(tmp_path, [FULL_CLAIM, LEGACY_CLAIM]))
    by_id = {c.id: c for c in store.all()}

    official = by_id["c_official"]
    blog = by_id["c_blog"]
    assert official.authority_level == AuthorityLevel.AUTHORITATIVE
    assert blog.authority_level == AuthorityLevel.MEDIUM
    # The H-6 failure mode: both used to load as MEDIUM contemporaneous_reporting, so
    # the official release and the blog post ranked identically in render_evidence.
    ranked = sorted(store.all(), key=lambda c: (-int(c.authority_level), c.id))
    assert [c.id for c in ranked] == ["c_official", "c_blog"]
    assert capsys.readouterr().err.count("legacy store") == 1


def test_full_claim_with_lower_authority_than_legacy_default(tmp_path):
    # LOW must survive as LOW — fidelity means never rounding up to the old default.
    claim_dict = copy.deepcopy(FULL_CLAIM)
    claim_dict.update(authority_level=1, source_type="low_quality", confidence=0.3)
    store = loader.load_store(_write_store(tmp_path, [claim_dict]))
    (claim,) = store.all()
    assert claim.authority_level == AuthorityLevel.LOW
    assert claim.source_type == SourceType.LOW_QUALITY
    assert claim.confidence == 0.3


def test_prepare_run_dir_clears_stale_artifacts_and_stamps(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    stale = [
        "forecast.json",
        "compiled_world.json",
        "diagnosis.json",
        "metrics.json",
        "semantic_plan.json",
        "semantic_map.json",
        "lowered_compilation.json",
        "report.md",
        "actor_decisions.jsonl",
    ]
    for name in stale:
        (out / name).write_text("{}")
    (out / "notes.txt").write_text("operator notes, not a pipeline artifact")

    loader.prepare_run_dir(
        out,
        question="Will it happen?",
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
        horizon=datetime(2026, 9, 1, tzinfo=UTC),
        mode="semantic",
    )

    for name in stale:
        assert not (out / name).exists(), f"stale {name} survived"
    # Only pipeline artifacts are cleared; the directory and other files stay.
    assert out.is_dir()
    assert (out / "notes.txt").exists()

    stamp = json.loads((out / "run_stamp.json").read_text())
    import hashlib

    assert stamp["question_sha256"] == hashlib.sha256(b"Will it happen?").hexdigest()
    assert stamp["as_of"] == "2026-07-01T00:00:00+00:00"
    assert stamp["horizon"] == "2026-09-01T00:00:00+00:00"
    assert stamp["mode"] == "semantic"
    assert stamp["commit"]
    assert stamp["started_at"]


def test_exporter_to_loader_round_trip_preserves_recorded_contradictions(tmp_path):
    """A recorded decisive contradiction must survive export → load, or the coverage
    gate's conflict check flips silently on replay."""

    import json
    import sys
    from datetime import datetime
    from pathlib import Path as _P

    scripts = str(_P(__file__).resolve().parent.parent.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from _store_loader import load_store

    from sworldmodel.api import _claim_record
    from sworldmodel.evidence import EvidenceClaim, EvidenceStore
    from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType

    def claim(cid: str) -> EvidenceClaim:
        t = datetime.fromisoformat("2026-01-05T00:00:00+00:00")
        return EvidenceClaim(
            id=cid,
            proposition=f"p {cid}",
            normalized_value=cid,
            entities=("X",),
            valid_from=t,
            valid_until=None,
            published_at=t,
            available_at=t,
            source_id="s",
            source_url="https://example.test/a",
            source_title="s",
            source_type=SourceType("contemporaneous_reporting"),
            authority_level=AuthorityLevel.MEDIUM,
            supporting_excerpt="e",
            lineage_event_id=f"ev_{cid}",
            epistemic_type=EpistemicType("observation"),
            confidence=0.9,
            retrieved_at=t,
        )

    store = EvidenceStore()
    store.add(claim("c-a"))
    store.add(claim("c-b"))
    store.record_contradiction("c-a", "c-b")
    out = tmp_path / "evidence_store.json"
    out.write_text(json.dumps([_claim_record(c) for c in store.all()], default=str))
    replayed = load_store(out)
    assert {tuple(sorted(pair)) for pair in replayed.contradictions()} == {("c-a", "c-b")}


def test_gateway_budget_is_per_run_not_lifetime():
    from sworldmodel.gateway import GatewayRequest, GatewayResponse, ModelGateway

    class _G(ModelGateway):
        is_live = False
        model_id = "test"

        def _generate(self, request):
            return GatewayResponse(
                task_kind=request.task_kind,
                data={"ok": True},
                raw_text="{}",
                model="test",
                params={},
                seed=0,
                prompt_hash="x",
                tokens_in=1,
                tokens_out=1,
            )

    g = _G()
    g.set_budget(max_calls=2)
    for _ in range(2):
        g.generate(GatewayRequest(task_kind="t", prompt="p", context={}, seed=0))
    g.set_budget(max_calls=2)  # a NEW run: ceiling resets against a fresh snapshot
    g.generate(GatewayRequest(task_kind="t", prompt="p", context={}, seed=0))


def test_the_canonical_artifact_list_covers_every_writer() -> None:
    """The clearing list drifted when it lived in the harness: tracing.py had grown six
    artifacts (pre_outcome_forecast.json, coverage_report.json, actor_grounding.json,
    branch_schedule.json, structural_uncertainty.json, evidence_manifest.json) that a
    stale run could leave beside a fresh refusal with nothing to delete them. The list
    now lives beside the writers; this pins every known writer's filenames into it and
    scans the sources for simple `dir / "name.json"` drift."""

    import re
    from pathlib import Path

    from sworldmodel import rundir
    from sworldmodel.rundir import PIPELINE_ARTIFACTS

    for name in (
        "forecast.json",
        "pre_outcome_forecast.json",
        "report.md",
        "pre_outcome_report.md",
        "coverage_report.json",
        "actor_grounding.json",
        "branch_schedule.json",
        "structural_uncertainty.json",
        "evidence_manifest.json",
        "world_review.json",
        "trajectory_audit.json",
    ):
        assert name in PIPELINE_ARTIFACTS, name

    src = Path(rundir.__file__).resolve().parent
    pattern = re.compile(r'/\s*"([a-z_0-9]+\.(?:json|jsonl|md))"')
    written: set[str] = set()
    for py in src.glob("*.py"):
        written |= {m.group(1) for m in pattern.finditer(py.read_text())}
    missing = written - set(PIPELINE_ARTIFACTS) - {"run_stamp.json"}
    assert not missing, f"artifacts written but never cleared: {sorted(missing)}"
