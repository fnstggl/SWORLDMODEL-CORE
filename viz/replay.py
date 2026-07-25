"""Read-only adapter: turn one run's trace artifacts into a replayable timeline.

This is the *only* file that knows the artifact schema. The frontend renders whatever
this returns, so when the simulation's trace format changes, this adapter changes and
the frontend does not. Nothing here imports ``sworldmodel`` or runs any simulation — it
reads the JSON and JSONL the run already wrote:

    forecast.json          the branches, their weights, conditions, outcomes, final state
    world_manifest.json    the compiled world — actors, actions, terminal, uncertainties
    actor_grounding.json   per-actor grounding text (what each agent was told it is)
    actor_decisions.jsonl  every actor call: exact prompt, provider response, intent, ...
    event_ledger.jsonl     every world event, chronological within a branch
    llm_calls.jsonl        every model call with its task kind and token counts
    diagnosis.json         the stage-by-stage record (used for refused runs)
    trajectory_audit.json  the post-simulation classification

Every read is defensive: a refused run has a compiled world and a diagnosis but no
forecast and no decisions, and it must still be viewable — it shows the world that was
built and the stage it stopped at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    if (p / "run_trace" / "forecast.json").exists() or (p / "run_trace" / "diagnosis.json").exists():
        return p / "run_trace"
    return p


def build_replay(trace_dir: str | Path) -> dict[str, Any]:
    """The whole run, normalized for replay. Safe on partial or refused runs."""

    d = _resolve_trace(trace_dir)
    forecast = _load(d / "forecast.json") or {}
    manifest = _load(d / "world_manifest.json") or {}
    diagnosis = _load(d / "diagnosis.json") or {}
    grounding = _load(d / "actor_grounding.json") or {}
    decisions = _load_lines(d / "actor_decisions.jsonl")
    events = _load_lines(d / "event_ledger.jsonl")
    llm = _load_lines(d / "llm_calls.jsonl")
    audit = _load(d / "run_audit.json") or {}
    trajectory = _load(d / "trajectory_audit.json") or {}

    return {
        "meta": _meta(d, forecast, diagnosis, audit, trajectory),
        "world": _world(manifest, diagnosis, grounding),
        "branches": _branches(forecast, decisions, events),
        "llm_summary": _llm_summary(llm, audit),
        "refusal": _refusal(diagnosis) if not forecast else None,
    }


def _meta(
    d: Path,
    forecast: dict[str, Any],
    diagnosis: dict[str, Any],
    audit: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    integ = diagnosis.get("forecast_integrity") or {}
    mode = ((diagnosis.get("research_planning") or {}).get("retrieval_mode") or {}).get("mode")
    return {
        "trace_dir": str(d),
        "case": d.parent.name if d.name == "run_trace" else d.name,
        "question": forecast.get("question") or diagnosis.get("question") or "(unknown)",
        "as_of": forecast.get("as_of") or diagnosis.get("as_of"),
        "horizon": forecast.get("horizon") or diagnosis.get("horizon"),
        "mode": mode,
        "model": forecast.get("model") or audit.get("model"),
        "status": forecast.get("status") or diagnosis.get("outcome") or "unknown",
        "probability": forecast.get("simulation_probability"),
        "probability_source": forecast.get("probability_source"),
        "resolved_yes_mass": forecast.get("resolved_yes_mass"),
        "resolved_no_mass": forecast.get("resolved_no_mass"),
        "unresolved_mass": forecast.get("unresolved_mass"),
        "lower_bound": forecast.get("lower_bound"),
        "upper_bound": forecast.get("upper_bound"),
        "model_calls": forecast.get("model_call_count") or diagnosis.get("model_calls"),
        "wall_seconds": diagnosis.get("wall_seconds") or audit.get("wall_seconds"),
        "trajectory_classification": trajectory.get("classification"),
        "integrity": {
            "probability_before": integ.get("probability_before_simulation"),
            "probability_after": integ.get("probability_after_simulation"),
            "calibrated": integ.get("point_estimate_is_calibrated"),
            "counterfactual": integ.get("counterfactual_note"),
            "ungrounded_variables": integ.get("ungrounded_variables") or [],
        },
    }


def _world(
    manifest: dict[str, Any], diagnosis: dict[str, Any], grounding: dict[str, Any]
) -> dict[str, Any]:
    comp = diagnosis.get("world_compilation") or {}
    entities = comp.get("entities") or []
    actor_ids = set((comp.get("causal_producers") or {}).get("actors") or [])
    ground_map = grounding if isinstance(grounding, dict) else {}

    actors = []
    others = []
    for e in entities:
        node = {
            "id": e.get("entity_id"),
            "name": e.get("name"),
            "kind": e.get("kind"),
            "role": e.get("role"),
            "authority": e.get("authority") or [],
            "scale": e.get("representation_scale"),
            "represents_count": e.get("represents_count"),
            "is_actor": bool(e.get("is_actor")) or (e.get("entity_id") in actor_ids),
            "cited_claims": e.get("cited_claims") or [],
        }
        g = ground_map.get(e.get("entity_id"))
        if isinstance(g, dict):
            node["grounding"] = g.get("rendered") or g.get("grounding") or _grounding_text(g)
        (actors if node["is_actor"] else others).append(node)

    return {
        "title": comp.get("title") or manifest.get("title"),
        "rationale": comp.get("structure_rationale"),
        "actors": actors,
        "entities": others,
        "actions": [
            {
                "id": a.get("action_id"),
                "meaning": a.get("meaning"),
                "authority": a.get("required_authority") or [],
                "writes": a.get("writes") or [],
                "effects": a.get("effect_ops") or a.get("effects") or [],
            }
            for a in (comp.get("actions") or manifest.get("compiled_actions") or [])
        ],
        "external_processes": (comp.get("causal_producers") or {}).get("external_processes") or [],
        "process_nodes": (comp.get("causal_producers") or {}).get("process_nodes") or [],
        "terminal": manifest.get("terminal") or {"description": comp.get("terminal_description")},
        "terminal_producers": comp.get("terminal_producers") or {},
        "uncertainties": _uncertainties(manifest),
        "fields": comp.get("fields") or [],
    }


def _grounding_text(g: dict[str, Any]) -> str:
    items = g.get("items") or g.get("records") or []
    lines = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):
            lines.append(f"{it.get('content', '')} [{it.get('provenance', '')}]")
    return "\n".join(lines)


def _uncertainties(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for u in manifest.get("uncertainty") or []:
        if not isinstance(u, dict):
            continue
        out.append(
            {
                "variable": u.get("variable"),
                "why_unknown": u.get("why_unknown"),
                "outcomes": [
                    {
                        "value": o.get("value"),
                        "weight": o.get("weight"),
                        "provenance": o.get("provenance"),
                    }
                    for o in (u.get("outcomes") or [])
                    if isinstance(o, dict)
                ],
            }
        )
    return out


def _structure_prefixes(
    forecast: dict[str, Any], decisions: list[dict[str, Any]]
) -> set[str]:
    """The namespace each structure prefixes its branch ids with (``primary``, ...).

    The forecast and the actor decisions carry structure-qualified branch ids
    (``primary/sc_x:v``); the event ledger carries the bare local id (``sc_x:v``). Only
    the first segment of an id that is a known structure namespace may be stripped, so an
    ordinary local id that happens to contain ``/`` is left intact.
    """

    prefixes: set[str] = set()
    for bid in [b.get("branch_id") for b in forecast.get("branches") or []] + [
        r.get("branch_id") for r in decisions
    ]:
        if bid and "/" in bid:
            prefixes.add(bid.split("/", 1)[0])
    return prefixes


def _branches(
    forecast: dict[str, Any],
    decisions: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # The forecast and decisions carry structure-qualified ids (``primary/sc_x``); the
    # ledger carries the bare local id (``sc_x``). Match on the local id so one branch
    # shows its actor decision and the world events that decision caused together, instead
    # of splitting into two half-branches — the whole point of the replay.
    prefixes = _structure_prefixes(forecast, decisions)

    def local_of(bid: str | None) -> str:
        if bid and "/" in bid and bid.split("/", 1)[0] in prefixes:
            return bid.split("/", 1)[1]
        return bid or ""

    # Canonical order and display id: prefer the structure-qualified id the forecast uses,
    # falling back to a decision's id, then to a bare ledger id for an engine-only run.
    order: list[str] = []
    display_by_local: dict[str, str] = {}
    for bid in (
        [b.get("branch_id") for b in forecast.get("branches") or []]
        + [r.get("branch_id") for r in decisions]
        + [e.get("branch_id") for e in events]
    ):
        if not bid:
            continue
        loc = local_of(bid)
        if loc not in display_by_local:
            display_by_local[loc] = bid
            order.append(loc)

    by_full = {b.get("branch_id"): b for b in forecast.get("branches") or []}
    out = []
    for loc in order:
        display_id = display_by_local[loc]
        b = by_full.get(display_id, {})
        b_events = [e for e in events if local_of(e.get("branch_id")) == loc]
        b_decisions = [r for r in decisions if local_of(r.get("branch_id")) == loc]
        out.append(
            {
                "id": display_id,
                "local_id": loc,
                "structure": display_id.split("/", 1)[0]
                if "/" in display_id and display_id.split("/", 1)[0] in prefixes
                else None,
                "weight": b.get("weight"),
                "conditions": b.get("conditions") or {},
                "outcome": b.get("outcome"),
                "resolved": b.get("resolved"),
                "unresolved_reason": b.get("unresolved_reason"),
                "final_state": b.get("world_state") or {},
                "steps": _merge_steps(b_events, b_decisions),
            }
        )
    return out


def _event_rank(kind: str, payload: dict[str, Any]) -> int:
    """Where an event sits relative to the decision at the same instant.

    A branch is seeded (its uncertain condition revealed) before anyone decides; the
    actor then decides; the effects that decision applies land after it; a terminal
    result event is the last word. Time is the primary order — this only breaks ties
    within one instant so the sequence reads the way it happened.
    """

    if kind in ("release_data", "branch_opened", "reveal", "seed"):
        return 0  # branch setup / condition revealed — before the decision
    data = payload.get("data") if isinstance(payload, dict) else None
    is_result = kind in ("create_event", "result") and (
        "outcome" in payload or (isinstance(data, dict) and "outcome" in data)
    )
    if is_result:
        return 3  # terminal result — after the effects
    return 2  # an effect the decision applied — after the decision (rank 1)


def _merge_steps(
    events: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One chronological sequence per branch: seed, decision, the effects it applied.

    Events and decisions each arrive in file order (chronological within the branch).
    They are merged by time; ties within one instant are ordered branch-seed (0) →
    decision (1) → applied effect (2) → terminal result (3), so a reader sees the branch
    open, the actor decide, the world change it caused, and finally the outcome.
    """

    steps: list[dict[str, Any]] = []
    for i, e in enumerate(events):
        steps.append(
            {
                "type": "event",
                "time": e.get("time"),
                "order": (e.get("time") or "", _event_rank(e.get("kind") or "", e.get("payload") or {}), i),
                "event_id": e.get("event_id"),
                "kind": e.get("kind"),
                "actor_id": e.get("actor_id"),
                "payload": e.get("payload") or {},
                "visibility": e.get("visibility"),
                "evidence_claim_ids": e.get("evidence_claim_ids") or [],
            }
        )
    for i, r in enumerate(decisions):
        intent = r.get("intent") or {}
        lv = r.get("local_view") or {}
        steps.append(
            {
                "type": "decision",
                "time": r.get("branch_time"),
                "order": (r.get("branch_time") or "", 1, i),
                "actor_id": r.get("actor_id"),
                "actor_name": r.get("canonical_identity") or lv.get("name") or r.get("actor_id"),
                "stage": r.get("stage"),
                "wake_reason": r.get("wake_reason"),
                "wake_detail": r.get("wake_detail"),
                "trigger_event_ids": r.get("trigger_event_ids") or [],
                "applied_event_ids": r.get("applied_event_ids") or [],
                "feasible_actions": r.get("feasible_actions") or [],
                "intent": intent,
                "action_mode": intent.get("mode") or intent.get("action_mode"),
                "action_id": intent.get("action_id") or intent.get("compiled_action_id"),
                "rationale": intent.get("rationale") or intent.get("reasoning"),
                "validation_status": r.get("validation_status"),
                "validation_reason": r.get("validation_reason"),
                "plan_before": r.get("plan_before"),
                "plan_after": r.get("plan_after"),
                "local_view": _trim_view(lv),
                "exact_prompt": r.get("exact_prompt") or "",
                "provider_response": r.get("provider_response"),
                "model": r.get("model"),
                "prompt_hash": r.get("prompt_hash"),
            }
        )
    steps.sort(key=lambda s: s["order"])
    for n, s in enumerate(steps):
        s["seq"] = n
        s.pop("order", None)
    return steps


def _trim_view(lv: dict[str, Any]) -> dict[str, Any]:
    """The parts of an actor's local view that show what it could see when it decided."""

    keep = (
        "why_you_are_deciding_now",
        "observations",
        "observed_fields",
        "beliefs",
        "goals",
        "public_facts",
        "retrieved_memories",
        "your_commitments",
        "pending_information_needs",
        "active_plan",
        "actor_grounding",
        "role",
        "authority",
        "stage",
    )
    return {k: lv[k] for k in keep if k in lv}


def _llm_summary(llm: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    tokens_in = tokens_out = retries = 0
    calls: list[dict[str, Any]] = []
    for i, c in enumerate(llm):
        kind = c.get("task_kind", "?")
        ti = int(c.get("tokens_in") or 0)
        to = int(c.get("tokens_out") or 0)
        rt = int(c.get("retries") or 0)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        tokens_in += ti
        tokens_out += to
        retries += rt
        calls.append(
            {
                "seq": i,
                "task_kind": kind,
                "model": c.get("model"),
                "tokens_in": ti,
                "tokens_out": to,
                "retries": rt,
                "prompt_hash": c.get("prompt_hash"),
                "response": c.get("response"),
            }
        )
    return {
        "total": len(llm),
        "by_kind": by_kind,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retries": retries,
        "sequence": [c.get("task_kind") for c in llm],
        "calls": calls,
    }


def _refusal(diagnosis: dict[str, Any]) -> dict[str, Any] | None:
    if diagnosis.get("outcome") != "refused":
        return None
    ig = diagnosis.get("integrity_and_grounding") or {}
    return {
        "stage": diagnosis.get("failure_stage"),
        "gate": ig.get("stopped_at_gate"),
        "message": ig.get("gate_message") or diagnosis.get("failure"),
        "root_cause": diagnosis.get("root_cause") or [],
        "repair_attempts": [
            {"failure": r.get("failure"), "outcome": r.get("outcome")}
            for r in ig.get("repair_attempts") or []
        ],
    }


def discover_traces(root: str | Path) -> list[dict[str, Any]]:
    """Every run trace under ``root``, newest first — for the trace picker.

    A directory is a trace when it holds a diagnosis or a forecast, so both completed and
    refused runs are listed. Reusable across questions: it finds whatever is there.
    """

    root = Path(root)
    by_dir: dict[str, dict[str, Any]] = {}
    for marker in root.rglob("diagnosis.json"):
        # A case dir often holds a copy of run_trace/diagnosis.json beside the real
        # run_trace/ one. Resolve both to the same canonical trace dir and keep one entry,
        # preferring whichever carries the actor decisions (the richer, real trace).
        d = _resolve_trace(marker.parent)
        key = str(d)
        forecast = _load(d / "forecast.json") or {}
        diag = _load(d / "diagnosis.json") or {}
        case = d.parent.name if d.name == "run_trace" else d.name
        entry = {
            "path": key,
            "case": case,
            "question": (forecast.get("question") or diag.get("question") or "")[:120],
            "status": forecast.get("status") or diag.get("outcome") or "unknown",
            "mtime": (d / "diagnosis.json").stat().st_mtime
            if (d / "diagnosis.json").exists()
            else marker.stat().st_mtime,
            "actor_calls": _count_lines(d / "actor_decisions.jsonl"),
            "branches": len(forecast.get("branches") or []),
        }
        prior = by_dir.get(key)
        if prior is None or entry["actor_calls"] >= prior["actor_calls"]:
            by_dir[key] = entry
    found = list(by_dir.values())
    found.sort(key=lambda x: x["mtime"], reverse=True)
    return found


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for x in path.read_text().splitlines() if x.strip())
    except OSError:
        return 0
