"""Run-directory preparation: one canonical artifact list, one clearing rule.

Downstream readers resolve a run's artifacts by filename, so a refusal that writes
only ``diagnosis.json`` beside a PREVIOUS run's ``forecast.json`` gets the stale
forecast scored as this run's result. Every entry point that owns an output
directory — the live CLI's ``--trace`` and both frozen-store harnesses — clears the
prior run's pipeline artifacts before starting and stamps the directory with the new
run's identity.

The list lives HERE, beside the writers, because it drifted when it lived in the
harness: the harness cleared the files it knew about while ``tracing.py`` had grown
``pre_outcome_forecast.json``, ``coverage_report.json``, ``actor_grounding.json``,
``branch_schedule.json``, ``structural_uncertainty.json`` and
``evidence_manifest.json`` — six artifacts a stale run could leave beside a fresh
refusal with nothing to delete them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Everything any run (live CLI or frozen harness, completed or refused) may write into
# an output directory. Deleted before a run starts so nothing stale can be read as
# fresh. ``run_stamp.json`` is deliberately absent: it is the marker this module
# rewrites, not a pipeline artifact.
PIPELINE_ARTIFACTS = (
    # the forecast and its reports
    "forecast.json",
    "pre_outcome_forecast.json",
    "report.md",
    "pre_outcome_report.md",
    # the compiled world and its assessments
    "compiled_world.json",
    "world_manifest.json",
    "world_review.json",
    "coverage_report.json",
    "actor_grounding.json",
    "structural_uncertainty.json",
    "evidence_manifest.json",
    # the simulation record
    "actor_decisions.jsonl",
    "event_ledger.jsonl",
    "llm_calls.jsonl",
    "branch_schedule.json",
    "trajectory_audit.json",
    "run_audit.json",
    # the observability set (OBS-1..4, OBS-7, OBS-8), derived at trace-write time
    # from the run's own record through the shared replay core (decision D7)
    "branch_initial_state.json",
    "state_diffs.jsonl",
    "communications.jsonl",
    "process_transitions.jsonl",
    "run_dossier.html",
    # research and diagnosis
    "research_trace.json",
    "evidence_store.json",
    "diagnosis.json",
    # harness-only outputs (harmless to clear from a CLI trace directory)
    "lowered_compilation.json",
    "refusal.json",
    "gate_refusal.json",
    "metrics.json",
)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def prepare_run_dir(
    out: Path, *, question: str, as_of: datetime, horizon: datetime, mode: str
) -> None:
    """Make ``out`` safe for a fresh run: clear stale artifacts, then stamp it.

    Every pipeline artifact a prior run could have left is deleted up front (the
    directory itself is kept), and ``run_stamp.json`` records which
    question/cutoff/mode/commit the surviving artifacts belong to.
    """

    out.mkdir(parents=True, exist_ok=True)
    for name in PIPELINE_ARTIFACTS:
        (out / name).unlink(missing_ok=True)
    for stale in out.glob("semantic_*.json"):
        stale.unlink(missing_ok=True)
    stamp = {
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "as_of": as_of.isoformat(),
        "horizon": horizon.isoformat(),
        "mode": mode,
        "commit": _git_commit(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    (out / "run_stamp.json").write_text(json.dumps(stamp, indent=1, sort_keys=True) + "\n")
