"""Read-only adapter: one run's complete observability dossier from real artifacts.

This module builds the data behind the dossier page (``viz/dossier.html``): timeline,
actors, communications, processes, world state, model calls, branches & weights,
terminal lineage, cost, and audit. Like ``replay.py`` it is schema-aware and nothing
else: it imports no ``sworldmodel`` code, runs no simulation, and never invents data.

Every value shown comes from an artifact the run (or its offline forensic
reconstruction) actually wrote. Each derived artifact is read from the run directory
first — the production trace writer is adopting the forensic filenames — and only then
from an optional forensics directory (the output of ``scripts/forensics.py``). When a
file exists in neither place, the section carries an explicit "not recorded by this
run" notice instead of a fabricated placeholder.

Artifacts read (run directory first, then the forensics directory):

    always written by a run today:
        forecast.json world_manifest.json structural_uncertainty.json
        branch_schedule.json event_ledger.jsonl actor_decisions.jsonl llm_calls.jsonl
        evidence_manifest.json evidence_store.json coverage_report.json
        world_review.json trajectory_audit.json compiled_world.json (newer runs)
        run_stamp.json diagnosis.json (older runs)
    forensic derivations (moving into the production trace under the same names):
        forensic_timeline.jsonl state_diffs.jsonl communications.jsonl
        process_transitions.jsonl branch_weight_history.jsonl
        semantic_runtime_lineage.jsonl terminal_evaluations.jsonl
        probability_reconstruction.json trajectory_responsibility.json
        forensic_verdict.json llm_calls_full.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The derived files scripts/forensics.py writes today and the trace writer is adopting.
FORENSIC_MARKERS = (
    "forensic_verdict.json",
    "forensic_timeline.jsonl",
    "state_diffs.jsonl",
    "probability_reconstruction.json",
)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_lines(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def _resolve_trace(trace_dir: str | Path) -> Path:
    """A case directory or its run_trace subdirectory both work as input."""

    p = Path(trace_dir)
    for marker in ("forecast.json", "diagnosis.json"):
        if (p / "run_trace" / marker).exists():
            return p / "run_trace"
    return p


def resolve_forensics(trace_dir: str | Path, forensics: str | Path | None) -> Path | None:
    """The forensics directory for one run, or None.

    ``forensics`` may be the run's own forensics directory, or a root holding one
    subdirectory per run label (``artifacts/forensics/{individual,...}``). A root is
    matched to the run first by the ``run_dir`` its forensic_verdict.json records, then
    by directory basename. No match → None (the dossier then says what is absent).
    """

    if forensics is None:
        return None
    f = Path(forensics)
    if not f.is_dir():
        return None
    if any((f / m).exists() for m in FORENSIC_MARKERS):
        return f
    trace = _resolve_trace(trace_dir)
    label = trace.parent.name if trace.name == "run_trace" else trace.name
    trace_posix = trace.resolve().as_posix()
    base_match: Path | None = None
    try:
        subdirs = sorted(p for p in f.iterdir() if p.is_dir())
    except OSError:
        return None
    for sub in subdirs:
        verdict = _load(sub / "forensic_verdict.json") or {}
        run_dir = str(verdict.get("run_dir") or "").strip("/")
        if run_dir and trace_posix.endswith(run_dir):
            return sub
        if sub.name == label:
            base_match = sub
    return base_match


class _Reader:
    """Reads one artifact by name — run directory first, forensics fallback.

    Records, for every file asked for, where it actually came from, so the page can
    show honest provenance ("run", "forensics", or absent) per section.
    """

    def __init__(self, run: Path, forensics: Path | None) -> None:
        self.run = run
        self.forensics = forensics
        self.sources: dict[str, dict[str, Any]] = {}

    def _locate(self, name: str) -> tuple[Path | None, str | None]:
        p = self.run / name
        if p.exists():
            return p, "run"
        if self.forensics is not None:
            q = self.forensics / name
            if q.exists():
                return q, "forensics"
        return None, None

    def _record(self, name: str, path: Path | None, origin: str | None) -> None:
        self.sources[name] = {"from": origin, "path": str(path) if path else None}

    def json(self, name: str) -> Any:
        path, origin = self._locate(name)
        self._record(name, path, origin)
        return _load(path) if path else None

    def jsonl(self, name: str) -> list[dict[str, Any]] | None:
        """None when the file exists nowhere; [] when it exists and is empty."""

        path, origin = self._locate(name)
        self._record(name, path, origin)
        return _load_lines(path) if path else None

    def notice(self, name: str, what: str) -> str | None:
        src = self.sources.get(name) or {}
        if src.get("from") == "run":
            return None
        if src.get("from") == "forensics":
            return (
                f"{what}: read from the offline forensic reconstruction "
                f"({src.get('path')}), not written by this production run."
            )
        return (
            f"not recorded by this run: {name} is absent from the run directory"
            + (
                " and from the forensics directory"
                if self.forensics is not None
                else " and no matching forensics directory is available"
            )
            + f". {what} cannot be shown."
        )


# ---------------------------------------------------------------------------
# branch-id normalization (FD-16): forecast/decisions carry structure-qualified ids
# ("primary/sc_x:v"); the ledger and the per-branch forensic files carry the bare
# local id ("sc_x:v"). Join on the local id, display the qualified id.
# ---------------------------------------------------------------------------


def _structure_prefixes(*id_lists: list[str | None]) -> set[str]:
    prefixes: set[str] = set()
    for ids in id_lists:
        for bid in ids:
            if bid and "/" in bid:
                prefixes.add(bid.split("/", 1)[0])
    return prefixes


def _localizer(prefixes: set[str]):  # noqa: ANN202 — tiny closure
    def local_of(bid: str | None) -> str:
        if bid and "/" in bid and bid.split("/", 1)[0] in prefixes:
            return bid.split("/", 1)[1]
        return bid or ""

    return local_of


def _event_rank(kind: str, payload: dict[str, Any]) -> int:
    """Tie-break within one instant: seed(0) → invocation(1) → effect(2) → result(3)."""

    if kind in ("release_data", "branch_opened", "reveal", "seed"):
        return 0
    data = payload.get("data") if isinstance(payload, dict) else None
    if kind in ("create_event", "result") and (
        "outcome" in payload or (isinstance(data, dict) and "outcome" in data)
    ):
        return 3
    return 2


def build_dossier(trace_dir: str | Path, forensics_dir: str | Path | None = None) -> dict[str, Any]:
    """The whole dossier, normalized for the page. Safe on partial and zero-actor runs."""

    run = _resolve_trace(trace_dir)
    forensics = resolve_forensics(run, forensics_dir) if forensics_dir else None
    r = _Reader(run, forensics)

    forecast = r.json("forecast.json") or {}
    manifest = r.json("world_manifest.json") or {}
    stamp = r.json("run_stamp.json") or {}
    diagnosis = r.json("diagnosis.json") or {}
    structural = r.json("structural_uncertainty.json")
    schedule = r.json("branch_schedule.json") or {}
    events = r.jsonl("event_ledger.jsonl") or []
    decisions = r.jsonl("actor_decisions.jsonl")
    llm = r.jsonl("llm_calls.jsonl") or []
    review = r.json("world_review.json")
    trajectory = r.json("trajectory_audit.json")
    grounding = r.json("actor_grounding.json") or {}

    # Forensic-derived set: run dir first, forensics fallback, honest absence.
    state_diffs = r.jsonl("state_diffs.jsonl")
    comms = r.jsonl("communications.jsonl")
    procs = r.jsonl("process_transitions.jsonl")
    weight_history = r.jsonl("branch_weight_history.jsonl")
    lineage = r.jsonl("semantic_runtime_lineage.jsonl")
    terminal_evals = r.jsonl("terminal_evaluations.jsonl")
    prob_recon = r.json("probability_reconstruction.json")
    responsibility = r.json("trajectory_responsibility.json")
    verdict = r.json("forensic_verdict.json")
    llm_full = r.jsonl("llm_calls_full.jsonl")
    if decisions is None:
        # Older forensic sets copied the decisions under this name.
        decisions = r.jsonl("actor_invocations.jsonl")
    decisions = decisions if decisions is not None else []

    evidence = _evidence_map(r)

    prefixes = _structure_prefixes(
        [b.get("branch_id") for b in forecast.get("branches") or []],
        [d.get("branch_id") for d in decisions],
        [w.get("branch_id") for w in weight_history or []],
        list(schedule.keys()) if isinstance(schedule, dict) else [],
    )
    local_of = _localizer(prefixes)

    events_by_id = {e.get("event_id"): e for e in events if e.get("event_id")}
    diffs_by_event = {d.get("event_id"): d for d in (state_diffs or []) if d.get("event_id")}

    invocations = _invocations(decisions, events_by_id, local_of)

    return {
        "meta": _meta(run, forensics, forecast, stamp, diagnosis),
        "sources": r.sources,
        "banner": _banner(forecast, verdict, responsibility, prob_recon, r),
        "timeline": _timeline(forecast, events, invocations, diffs_by_event, local_of, r),
        "actors": {
            "recorded": r.sources.get("actor_decisions.jsonl", {}).get("from") is not None,
            "count": len(invocations),
            "invocations": invocations,
            "grounding": grounding,
            "notice": r.notice("actor_decisions.jsonl", "per-invocation actor records"),
            "zero_actor_note": (
                "this run recorded zero actor invocations — the compiled world has no "
                "acting entities (see the Audit view for whether the reviewer accepted that)"
                if r.sources.get("actor_decisions.jsonl", {}).get("from") and not invocations
                else None
            ),
        },
        "communications": _communications(comms, events_by_id, invocations, local_of, r),
        "processes": {
            "recorded": procs is not None,
            "rows": [
                {
                    "time": p.get("time"),
                    "branch": local_of(p.get("branch_id")),
                    "operation": p.get("operation"),
                    "event_id": p.get("event_id"),
                    "inputs": p.get("inputs"),
                    "outputs": p.get("outputs"),
                    "evidence_claim_ids": p.get("evidence_claim_ids") or [],
                }
                for p in procs or []
            ],
            "notice": r.notice("process_transitions.jsonl", "non-actor process transitions"),
        },
        "world_state": _world_state(forecast, state_diffs, local_of, r),
        "llm": _llm(llm, llm_full, r),
        "branches": _branches(
            forecast,
            manifest,
            structural,
            schedule,
            weight_history,
            terminal_evals,
            verdict,
            prob_recon,
            local_of,
            r,
        ),
        "lineage": _lineage(lineage, terminal_evals, local_of, r),
        "cost": _cost(llm, llm_full, prob_recon, stamp, diagnosis, r),
        "audit": _audit(review, trajectory, verdict, responsibility, r),
        "evidence": evidence,
    }


def _meta(
    run: Path,
    forensics: Path | None,
    forecast: dict[str, Any],
    stamp: dict[str, Any],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_dir": str(run),
        "forensics_dir": str(forensics) if forensics else None,
        "case": run.parent.name if run.name == "run_trace" else run.name,
        "question": forecast.get("question") or diagnosis.get("question") or "(unknown)",
        "as_of": forecast.get("as_of") or stamp.get("as_of") or diagnosis.get("as_of"),
        "horizon": forecast.get("horizon") or stamp.get("horizon") or diagnosis.get("horizon"),
        "status": forecast.get("status") or diagnosis.get("outcome") or "unknown",
        "probability": forecast.get("simulation_probability"),
        "probability_source": forecast.get("probability_source"),
        "lower_bound": forecast.get("lower_bound"),
        "upper_bound": forecast.get("upper_bound"),
        "resolved_yes_mass": forecast.get("resolved_yes_mass"),
        "resolved_no_mass": forecast.get("resolved_no_mass"),
        "unresolved_mass": forecast.get("unresolved_mass"),
        "model": forecast.get("model"),
        "model_call_count": forecast.get("model_call_count"),
        "token_usage": forecast.get("token_usage"),
        "integrity_verdict": forecast.get("integrity_verdict"),
        "contract_id": forecast.get("contract_id"),
        "commit": stamp.get("commit"),
        "mode": stamp.get("mode"),
        "started_at": stamp.get("started_at"),
        "wall_seconds": diagnosis.get("wall_seconds"),
    }


def _banner(
    forecast: dict[str, Any],
    verdict: dict[str, Any] | None,
    responsibility: dict[str, Any] | None,
    prob_recon: dict[str, Any] | None,
    r: _Reader,
) -> dict[str, Any]:
    """Verdict / responsibility / calibration for the always-visible banner.

    Each element names where it came from; anything unrecorded says so explicitly.
    """

    v = verdict or {}
    resp_class = (responsibility or {}).get("classification") or (
        (v.get("responsibility") or {}).get("classification")
    )
    # The forecast integrity block (validity triple) is being added to forecast.json by
    # the trace writer; read it when present, otherwise fall back to the recorded
    # weight-grounding facts from the probability reconstruction.
    integ = forecast.get("integrity") or forecast.get("forecast_integrity") or {}
    calibrated = integ.get("point_estimate_calibrated")
    if calibrated is None:
        calibrated = integ.get("point_estimate_is_calibrated")
    weights = (prob_recon or {}).get("weights") or (v.get("weights") if v else None) or {}
    return {
        "verdict": v.get("verdict"),
        "audit_classification": v.get("audit_classification"),
        "verdict_source": (r.sources.get("forensic_verdict.json") or {}).get("from"),
        "verdict_notice": r.notice("forensic_verdict.json", "the forensic verdict"),
        "responsibility": resp_class,
        "responsibility_signals": (responsibility or {}).get("signals"),
        "responsibility_source": (r.sources.get("trajectory_responsibility.json") or {}).get(
            "from"
        ),
        "responsibility_notice": r.notice(
            "trajectory_responsibility.json", "the responsibility classification"
        ),
        "calibration": {
            "point_estimate_calibrated": calibrated,
            "recorded_in_forecast": bool(integ),
            "validity_triple": {
                k: integ.get(k)
                for k in (
                    "trace_reproducible",
                    "causal_simulation_valid",
                    "point_estimate_calibrated",
                )
                if k in integ
            }
            or None,
            "probability_source": forecast.get("probability_source"),
            "all_weights_ungrounded": weights.get("all_weights_ungrounded"),
            "weights_flag": weights.get("flag"),
            "note": (
                None
                if integ
                else "forecast.json carries no integrity block in this run; "
                "weight-grounding facts shown from probability_reconstruction.json"
                if weights
                else "forecast.json carries no integrity block and no probability "
                "reconstruction is available — calibration not recorded by this run"
            ),
        },
    }


def _invocations(
    decisions: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
    local_of,  # noqa: ANN001 — closure
) -> list[dict[str, Any]]:
    out = []
    for i, d in enumerate(decisions):
        intent = d.get("intent") or {}
        delivered = d.get("delivered_observation_ids") or []
        noticed = d.get("noticed_observation_ids") or []
        missed = [x for x in delivered if x not in noticed]
        out.append(
            {
                "index": i,
                "branch_id": d.get("branch_id"),
                "branch": local_of(d.get("branch_id")),
                "time": d.get("branch_time"),
                "actor_id": d.get("actor_id"),
                "actor_name": d.get("canonical_identity") or d.get("actor_id"),
                "stage": d.get("stage"),
                "wake_reason": d.get("wake_reason"),
                "wake_detail": d.get("wake_detail"),
                "delivered_observation_ids": delivered,
                "noticed_observation_ids": noticed,
                "missed_observation_ids": missed,
                "trigger_event_ids": d.get("trigger_event_ids") or [],
                "retrieved_memory_ids": d.get("retrieved_memory_ids") or [],
                "feasible_actions": d.get("feasible_actions") or [],
                "exact_prompt": d.get("exact_prompt"),
                "provider_response": d.get("provider_response"),
                "intent": intent,
                "action": intent.get("action_id")
                or intent.get("compiled_action_id")
                or intent.get("mode")
                or intent.get("action_mode"),
                "validation_status": d.get("validation_status"),
                "validation_reason": d.get("validation_reason"),
                "applied_event_ids": d.get("applied_event_ids") or [],
                "resulting_events": [
                    {
                        "event_id": eid,
                        "kind": (events_by_id.get(eid) or {}).get("kind"),
                        "time": (events_by_id.get(eid) or {}).get("time"),
                        "payload": (events_by_id.get(eid) or {}).get("payload"),
                    }
                    for eid in d.get("applied_event_ids") or []
                ],
                "plan_before": d.get("plan_before"),
                "plan_after": d.get("plan_after"),
                "plan_disposition": d.get("plan_disposition"),
                "state_before": d.get("state_before"),
                "state_after": d.get("state_after"),
                "local_view": d.get("local_view"),
                "actor_evidence_claim_ids": d.get("actor_evidence_claim_ids") or [],
                "model": d.get("model"),
                "prompt_hash": d.get("prompt_hash"),
                "world_version_at_decision": d.get("world_version_at_decision"),
            }
        )
    return out


def _timeline(
    forecast: dict[str, Any],
    events: list[dict[str, Any]],
    invocations: list[dict[str, Any]],
    diffs_by_event: dict[str, dict[str, Any]],
    local_of,  # noqa: ANN001 — closure
    r: _Reader,
) -> dict[str, Any]:
    """Chronological merge, per branch, of the two primary records the run wrote:
    the event ledger and the actor invocations. Rows carry their joins: the matching
    state diff (by event id), the evidence claim ids, and the invocation index that
    opens the exact prompt/response in the Actors view. Nothing is synthesized."""

    order: list[str] = []
    display: dict[str, str] = {}
    for bid in (
        [b.get("branch_id") for b in forecast.get("branches") or []]
        + [i["branch_id"] for i in invocations]
        + [e.get("branch_id") for e in events]
    ):
        if not bid:
            continue
        loc = local_of(bid)
        if loc not in display:
            display[loc] = bid
            order.append(loc)

    branches = []
    for loc in order:
        rows: list[dict[str, Any]] = []
        for i, e in enumerate(events):
            if local_of(e.get("branch_id")) != loc:
                continue
            payload = e.get("payload") or {}
            rows.append(
                {
                    "type": "event",
                    "time": e.get("time"),
                    "_order": (e.get("time") or "", _event_rank(e.get("kind") or "", payload), i),
                    "kind": e.get("kind"),
                    "event_id": e.get("event_id"),
                    "actor_id": e.get("actor_id"),
                    "visibility": e.get("visibility"),
                    "payload": payload,
                    "evidence_claim_ids": e.get("evidence_claim_ids") or [],
                    "state_diff": diffs_by_event.get(e.get("event_id")),
                }
            )
        for inv in invocations:
            if inv["branch"] != loc:
                continue
            rows.append(
                {
                    "type": "invocation",
                    "time": inv["time"],
                    "_order": (inv["time"] or "", 1, inv["index"]),
                    "actor_id": inv["actor_id"],
                    "actor_name": inv["actor_name"],
                    "invocation_index": inv["index"],
                    "wake_reason": inv["wake_reason"],
                    "action": inv["action"],
                    "validation_status": inv["validation_status"],
                    "trigger_event_ids": inv["trigger_event_ids"],
                    "applied_event_ids": inv["applied_event_ids"],
                    "evidence_claim_ids": inv["actor_evidence_claim_ids"],
                }
            )
        rows.sort(key=lambda x: x["_order"])
        for n, row in enumerate(rows):
            row["seq"] = n
            row.pop("_order", None)
        branches.append({"id": display[loc], "local_id": loc, "rows": rows})

    return {
        "branches": branches,
        "state_diff_notice": r.notice("state_diffs.jsonl", "per-event state diffs"),
    }


def _communications(
    comms: list[dict[str, Any]] | None,
    events_by_id: dict[str, dict[str, Any]],
    invocations: list[dict[str, Any]],
    local_of,  # noqa: ANN001 — closure
    r: _Reader,
) -> dict[str, Any]:
    """Send → deliver → notice chains. Delivery/notice come from the invocation records
    (delivered_observation_ids / noticed_observation_ids), joined by event id."""

    rows = []
    for c in comms or []:
        eid = c.get("event_id")
        ledger = events_by_id.get(eid) or {}
        payload = ledger.get("payload") or {}
        delivered_to = [
            {"actor_id": i["actor_id"], "invocation_index": i["index"], "time": i["time"]}
            for i in invocations
            if eid in i["delivered_observation_ids"]
        ]
        noticed_by = [
            {"actor_id": i["actor_id"], "invocation_index": i["index"], "time": i["time"]}
            for i in invocations
            if eid in i["noticed_observation_ids"]
        ]
        noticed_ids = {n["invocation_index"] for n in noticed_by}
        missed_by = [d for d in delivered_to if d["invocation_index"] not in noticed_ids]
        rows.append(
            {
                "time": c.get("time"),
                "branch": local_of(c.get("branch_id")),
                "kind": c.get("kind"),
                "event_id": eid,
                "sender": c.get("sender"),
                "recipients": c.get("recipients") or [],
                "visibility": c.get("visibility"),
                "content": c.get("content") or payload.get("text"),
                "payload": payload or None,
                "delivered_to": delivered_to,
                "noticed_by": noticed_by,
                "missed_by": missed_by,
            }
        )
    return {
        "recorded": comms is not None,
        "rows": rows,
        "notice": r.notice("communications.jsonl", "communication events"),
    }


def _world_state(
    forecast: dict[str, Any],
    state_diffs: list[dict[str, Any]] | None,
    local_of,  # noqa: ANN001 — closure
    r: _Reader,
) -> dict[str, Any]:
    """Per-branch field evolution, scrubbing the ordered state diffs; initial vs final.

    The diffs are the recorded reconstruction (initial state + ordered diffs replay the
    branch with no model call — D7); the forecast's per-branch world_state is shown
    beside the diff-derived final state so a reader can compare the two records."""

    by_branch: dict[str, list[dict[str, Any]]] = {}
    for d in state_diffs or []:
        by_branch.setdefault(local_of(d.get("branch_id")), []).append(d)

    branches = []
    forecast_branches = forecast.get("branches") or []
    seen = set()
    for fb in forecast_branches:
        loc = local_of(fb.get("branch_id"))
        seen.add(loc)
        steps = sorted(
            by_branch.get(loc, []),
            key=lambda x: (x.get("simulation_time") or "", x.get("event_id") or ""),
        )
        branches.append(
            {
                "id": fb.get("branch_id"),
                "local_id": loc,
                "initial": (steps[0].get("state_before") if steps else None),
                "final": (steps[-1].get("state_after") if steps else None),
                "forecast_final": fb.get("world_state"),
                "steps": [
                    {
                        "time": s.get("simulation_time"),
                        "event_id": s.get("event_id"),
                        "operation": s.get("runtime_operation"),
                        "triggered_by": s.get("triggered_by"),
                        "diff": s.get("diff"),
                        "state_after": s.get("state_after"),
                        "evidence_claim_ids": s.get("evidence_claim_ids") or [],
                    }
                    for s in steps
                ],
            }
        )
    for loc, steps in by_branch.items():
        if loc in seen:
            continue
        steps = sorted(
            steps, key=lambda x: (x.get("simulation_time") or "", x.get("event_id") or "")
        )
        branches.append(
            {
                "id": loc,
                "local_id": loc,
                "initial": steps[0].get("state_before"),
                "final": steps[-1].get("state_after"),
                "forecast_final": None,
                "steps": [
                    {
                        "time": s.get("simulation_time"),
                        "event_id": s.get("event_id"),
                        "operation": s.get("runtime_operation"),
                        "triggered_by": s.get("triggered_by"),
                        "diff": s.get("diff"),
                        "state_after": s.get("state_after"),
                        "evidence_claim_ids": s.get("evidence_claim_ids") or [],
                    }
                    for s in steps
                ],
            }
        )
    return {
        "recorded": state_diffs is not None,
        "branches": branches,
        "notice": r.notice("state_diffs.jsonl", "the ordered per-branch state diffs"),
    }


_MISSING_PREFIX = "__missing_"


def _llm(
    llm: list[dict[str, Any]],
    llm_full: list[dict[str, Any]] | None,
    r: _Reader,
) -> dict[str, Any]:
    rows = llm_full if llm_full and len(llm_full) >= len(llm) else llm
    calls = []
    missing_notes: dict[str, str] = {}
    for i, c in enumerate(rows):
        for k, v in c.items():
            if k.startswith(_MISSING_PREFIX) and isinstance(v, str):
                missing_notes.setdefault(k[len(_MISSING_PREFIX) :], v)
        calls.append(
            {
                "seq": i,
                "call_number": c.get("call_number") or i + 1,
                "task_kind": c.get("task_kind"),
                "model": c.get("model"),
                "tokens_in": c.get("tokens_in"),
                "tokens_out": c.get("tokens_out"),
                "retries": c.get("retries"),
                "seed": c.get("seed"),
                "prompt_hash": c.get("prompt_hash"),
                "prompt": c.get("prompt") or c.get("exact_prompt"),
                "response": c.get("response"),
                "started_at": c.get("started_at"),
                "ended_at": c.get("ended_at"),
                "latency_ms": c.get("latency_ms"),
                "estimated_cost_usd": c.get("estimated_cost_usd"),
            }
        )
    by_kind: dict[str, int] = {}
    tokens_in = tokens_out = retries = 0
    cost_vals = [c["estimated_cost_usd"] for c in calls if c["estimated_cost_usd"] is not None]
    for c in calls:
        by_kind[c["task_kind"] or "?"] = by_kind.get(c["task_kind"] or "?", 0) + 1
        tokens_in += int(c["tokens_in"] or 0)
        tokens_out += int(c["tokens_out"] or 0)
        retries += int(c["retries"] or 0)

    def presence(field: str, label: str) -> str:
        if any(c.get(field) is not None for c in calls):
            return "recorded"
        if field in missing_notes:
            return f"not recorded by this run — {missing_notes[field]}"
        return f"{label} not recorded by this run"

    return {
        "calls": calls,
        "totals": {
            "count": len(calls),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "retries": retries,
            "estimated_cost_usd": round(sum(cost_vals), 6) if cost_vals else None,
            "by_kind": by_kind,
        },
        "field_presence": {
            "prompt": presence("prompt", "verbatim prompts"),
            "latency_ms": presence("latency_ms", "per-call latency"),
            "started_at": presence("started_at", "per-call timestamps"),
            "estimated_cost_usd": presence("estimated_cost_usd", "per-call cost estimates"),
        },
        "notice": r.notice("llm_calls.jsonl", "the model-call log"),
        "full_source": (r.sources.get("llm_calls_full.jsonl") or {}).get("from"),
    }


def _branches(
    forecast: dict[str, Any],
    manifest: dict[str, Any],
    structural: dict[str, Any] | None,
    schedule: dict[str, Any],
    weight_history: list[dict[str, Any]] | None,
    terminal_evals: list[dict[str, Any]] | None,
    verdict: dict[str, Any] | None,
    prob_recon: dict[str, Any] | None,
    local_of,  # noqa: ANN001 — closure
    r: _Reader,
) -> dict[str, Any]:
    hist_by_loc = {local_of(w.get("branch_id")): w for w in weight_history or []}
    evals_by_loc = {local_of(t.get("branch_id")): t for t in terminal_evals or []}
    verdict_by_loc = {
        local_of(b.get("branch_id")): b for b in (verdict or {}).get("branches") or []
    }
    sched_by_loc = (
        {local_of(k): v for k, v in schedule.items()} if isinstance(schedule, dict) else {}
    )

    rows = []
    for b in forecast.get("branches") or []:
        loc = local_of(b.get("branch_id"))
        hist = hist_by_loc.get(loc)
        vb = verdict_by_loc.get(loc)
        rows.append(
            {
                "id": b.get("branch_id"),
                "local_id": loc,
                "weight": b.get("weight"),
                "conditions": b.get("conditions") or {},
                "outcome": b.get("outcome"),
                "resolved": b.get("resolved"),
                "unresolved_reason": b.get("unresolved_reason"),
                # Being added to forecast.json by the trace writer; absent → shown as
                # "not recorded by this run" on the page, never invented.
                "pre_outcome": b.get("pre_outcome"),
                "weight_grounded": b.get("weight_grounded"),
                "world_state": b.get("world_state") or {},
                "history": {
                    "starting_weight": hist.get("starting_weight"),
                    "final_weight": hist.get("final_weight"),
                    "provenance": hist.get("provenance"),
                    "reason_for_starting_weight": hist.get("reason_for_starting_weight"),
                    "weight_changes": hist.get("weight_changes"),
                    "contribution_to_probability": hist.get("contribution_to_probability"),
                    "assumptions": hist.get("assumptions"),
                    "terminal_result": hist.get("terminal_result"),
                }
                if hist
                else None,
                "schedule": sched_by_loc.get(loc),
                "terminal_evaluation": evals_by_loc.get(loc),
                "counterfactuals": (vb or {}).get("counterfactuals"),
                "initial_fields": (vb or {}).get("initial_fields"),
                "final_fields": (vb or {}).get("final_fields"),
            }
        )

    pr = prob_recon or {}
    return {
        "rows": rows,
        "uncertainties": manifest.get("uncertainty") or [],
        "structural_uncertainty": structural,
        "probability": {
            "recorded": prob_recon is not None,
            "aggregation_formula": pr.get("aggregation_formula"),
            "numeric_substitution": pr.get("numeric_substitution")
            or (pr.get("recomputed") or {}).get("substitution"),
            "published": pr.get("published"),
            "recomputed": pr.get("recomputed"),
            "weights": pr.get("weights"),
            "mismatches": pr.get("mismatches") or [],
            "notice": r.notice(
                "probability_reconstruction.json", "the exact probability substitution"
            ),
        },
        "pre_outcome_recorded": any("pre_outcome" in b for b in forecast.get("branches") or []),
        "weight_history_notice": r.notice("branch_weight_history.jsonl", "branch weight history"),
    }


def _lineage(
    lineage: list[dict[str, Any]] | None,
    terminal_evals: list[dict[str, Any]] | None,
    local_of,  # noqa: ANN001 — closure
    r: _Reader,
) -> dict[str, Any]:
    records = [
        {
            "namespace": rec.get("namespace"),
            "semantic_object": rec.get("semantic_object"),
            "runtime_id": rec.get("runtime_id"),
            "lowering_rule": rec.get("lowering_rule"),
            "executed": rec.get("executed"),
            "event_ids": rec.get("event_ids") or [],
            "evidence_claim_ids": rec.get("evidence_claim_ids") or [],
        }
        for rec in lineage or []
    ]
    return {
        "recorded": lineage is not None,
        "records": records,
        "terminals": [dict(t, branch=local_of(t.get("branch_id"))) for t in terminal_evals or []],
        "notice": r.notice("semantic_runtime_lineage.jsonl", "claim → semantic → runtime lineage"),
        "terminals_notice": r.notice("terminal_evaluations.jsonl", "terminal evaluations"),
    }


def _cost(
    llm: list[dict[str, Any]],
    llm_full: list[dict[str, Any]] | None,
    prob_recon: dict[str, Any] | None,
    stamp: dict[str, Any],
    diagnosis: dict[str, Any],
    r: _Reader,
) -> dict[str, Any]:
    rows = llm_full if llm_full and len(llm_full) >= len(llm) else llm
    stages: dict[str, dict[str, Any]] = {}
    for c in rows:
        kind = c.get("task_kind") or "?"
        s = stages.setdefault(
            kind,
            {
                "task_kind": kind,
                "calls": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "retries": 0,
                "estimated_cost_usd": 0.0,
                "cost_recorded": False,
                "latency_ms": 0,
                "latency_recorded": False,
            },
        )
        s["calls"] += 1
        s["tokens_in"] += int(c.get("tokens_in") or 0)
        s["tokens_out"] += int(c.get("tokens_out") or 0)
        s["retries"] += int(c.get("retries") or 0)
        if c.get("estimated_cost_usd") is not None:
            s["estimated_cost_usd"] += float(c["estimated_cost_usd"])
            s["cost_recorded"] = True
        if c.get("latency_ms") is not None:
            s["latency_ms"] += int(c["latency_ms"])
            s["latency_recorded"] = True
    by_stage = sorted(stages.values(), key=lambda s: -s["tokens_out"])
    for s in by_stage:
        if not s["cost_recorded"]:
            s["estimated_cost_usd"] = None
        else:
            s["estimated_cost_usd"] = round(s["estimated_cost_usd"], 6)
        if not s["latency_recorded"]:
            s["latency_ms"] = None
    total_cost = [s["estimated_cost_usd"] for s in by_stage if s["estimated_cost_usd"] is not None]
    recomputed = (prob_recon or {}).get("recomputed") or {}
    return {
        "by_stage": by_stage,
        "totals": {
            "calls": len(rows),
            "tokens_in": sum(int(c.get("tokens_in") or 0) for c in rows),
            "tokens_out": sum(int(c.get("tokens_out") or 0) for c in rows),
            "estimated_cost_usd": round(sum(total_cost), 6) if total_cost else None,
        },
        "cost_note": recomputed.get("estimated_cost_note"),
        "wall_seconds": diagnosis.get("wall_seconds"),
        "started_at": stamp.get("started_at"),
        "latency_notice": (
            None
            if any(s["latency_ms"] is not None for s in by_stage)
            else "per-call latency not recorded by this run — stage timing cannot be shown"
        ),
    }


def _audit(
    review: dict[str, Any] | None,
    trajectory: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
    responsibility: dict[str, Any] | None,
    r: _Reader,
) -> dict[str, Any]:
    v = dict(verdict or {})
    v.pop("branches", None)  # merged into the Branches view
    return {
        "world_review": review,
        "world_review_notice": r.notice("world_review.json", "the pre-simulation world review"),
        "trajectory_audit": trajectory,
        "trajectory_audit_notice": r.notice(
            "trajectory_audit.json", "the post-simulation trajectory audit"
        ),
        "verdict": v or None,
        "verdict_notice": r.notice("forensic_verdict.json", "the forensic verdict"),
        "responsibility": responsibility,
        "responsibility_notice": r.notice(
            "trajectory_responsibility.json", "the responsibility classification"
        ),
    }


def _evidence_map(r: _Reader) -> dict[str, dict[str, Any]]:
    """Claim id → claim record, for evidence-lineage links. Manifest first, store fallback."""

    manifest = r.json("evidence_manifest.json") or {}
    claims = manifest.get("claims")
    if not claims:
        store = r.json("evidence_store.json")
        claims = store if isinstance(store, list) else []
    keep = (
        "proposition",
        "normalized_value",
        "entities",
        "epistemic_type",
        "authority_level",
        "available_at",
        "available_by_cutoff",
        "published_at",
        "confidence",
        "source_url",
        "url",
        "source",
    )
    out: dict[str, dict[str, Any]] = {}
    for c in claims or []:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        out[c["id"]] = {k: c[k] for k in keep if k in c}
    return out
