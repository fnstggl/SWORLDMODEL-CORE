#!/usr/bin/env python3
"""Forensic reconstruction of a completed run, from its own production artifacts.

A published probability is worth exactly as much as an independent reviewer's ability
to rebuild it. This tool rebuilds it: it reads only what a run already wrote, derives
the chronology, the per-branch weight accounting, the state transitions, the terminal
evaluations and the aggregation, and then re-derives the published number and compares.
A mismatch is CRITICAL and the run is marked FORENSICALLY_INVALID.

It also runs the trajectory-responsibility tests as *deterministic replays over the
recorded ledger* — no model call, no re-simulation:

  * remove every actor-produced event and re-evaluate each branch's terminal;
  * remove one actor's events at a time;
  * remove one non-actor process's events at a time;
  * keep the trajectory but flatten branch weights to uniform;
  * keep the weights but delete terminal-relevant actions.

The classification that comes out (TRAJECTORY_CAUSED … BRANCH_WEIGHTS_DOMINATED) is a
statement about *what produced the number*, and it is computed, never asserted.

Terminal re-evaluation uses the run's OWN executable world when the run persisted one
(`compiled_world.json`) — the same expression AST the engine evaluated. Runs written
before that artifact existed fall back to a declared-support subset of the terminal
grammar; anything outside it is reported as UNRECONSTRUCTABLE rather than guessed.

    PYTHONPATH=src python3 scripts/forensics.py --run artifacts/slice92/individual \
        --out artifacts/forensics/individual --label individual
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Provider list price for the model these runs used, in dollars per million tokens.
# Recorded here as an explicit, labeled ESTIMATE: the trace records tokens, which are a
# fact, and cost is a rate applied to them. It is never written as if it were measured.
RATE_IN_PER_MTOK = 0.28
RATE_OUT_PER_MTOK = 0.42


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Terminal evaluation
# --------------------------------------------------------------------------- #


class Unreconstructable(Exception):
    """The terminal cannot be re-evaluated from what this run recorded."""


def _terminal_ast(run: Path) -> dict[str, Any] | None:
    """The executable terminal, when the run persisted its executable world."""

    world = _read_json(run / "compiled_world.json")
    if isinstance(world, dict):
        spec = world.get("world_spec")
        if isinstance(spec, dict) and isinstance(spec.get("terminal"), dict):
            return dict(spec["terminal"])
    return None


_CALL = re.compile(r"^(\w+)\((.*)\)$", re.DOTALL)


def _split_args(text: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _eval_rendered(expr: str, fields: dict[str, Any], counts: dict[str, int]) -> Any:
    """Evaluate the RENDERED terminal string a pre-compiled-world run recorded.

    Deliberately tiny and total: it supports exactly the forms these runs used, and
    raises :class:`Unreconstructable` for anything else. Guessing at an unsupported
    operator would manufacture the very confidence this tool exists to check.
    """

    expr = expr.strip()
    if expr in ("True", "False"):
        return expr == "True"
    if expr == "None":
        return None
    m = _CALL.match(expr)
    if not m:
        if expr.startswith("'") and expr.endswith("'"):
            return expr[1:-1]
        try:
            return float(expr) if "." in expr else int(expr)
        except ValueError:
            raise Unreconstructable(f"unparsable terminal atom: {expr!r}") from None
    op, arg_text = m.group(1), m.group(2)
    args = _split_args(arg_text)
    if op == "field":
        return fields.get(_eval_rendered(args[0], fields, counts))
    if op == "event_count":
        return counts.get(str(_eval_rendered(args[0], fields, counts)), 0)
    if op == "count":
        return counts.get(str(_eval_rendered(args[0], fields, counts)), 0)
    vals = [_eval_rendered(a, fields, counts) for a in args]
    if op == "equals":
        return vals[0] == vals[1]
    if op == "not":
        return not vals[0]
    if op == "and":
        return all(bool(v) for v in vals)
    if op == "or":
        return any(bool(v) for v in vals)
    if op in ("greater_than", "greater_or_equal", "less_than", "less_or_equal"):
        a, b = vals[0], vals[1]
        if a is None or b is None:
            return False
        a, b = float(a), float(b)
        return {
            "greater_than": a > b,
            "greater_or_equal": a >= b,
            "less_than": a < b,
            "less_or_equal": a <= b,
        }[op]
    if op in ("multiply", "add", "subtract"):
        nums = [0.0 if v is None else float(v) for v in vals]
        if op == "multiply":
            out = 1.0
            for n in nums:
                out *= n
            return out
        if op == "add":
            return sum(nums)
        return nums[0] - nums[1]
    raise Unreconstructable(f"unsupported terminal operator {op!r}")


def _eval_ast(node: Any, fields: dict[str, Any], counts: dict[str, int]) -> Any:
    """Evaluate the executable terminal AST with the engine's own evaluator."""

    from sworldmodel.expressions import evaluate
    from sworldmodel.worldspec import parse_expr

    from _forensic_world import ForensicWorld  # local shim, see below

    return evaluate(parse_expr(node), ForensicWorld(fields, counts))


def evaluate_terminal(
    terminal: dict[str, Any] | None,
    rendered: dict[str, Any],
    fields: dict[str, Any],
    counts: dict[str, int],
) -> tuple[bool, str | None, str]:
    """(resolved, outcome, how) for one branch state. Never guesses."""

    if terminal is not None:
        yes = bool(_eval_ast(terminal.get("yes_when"), fields, counts))
        unres_node = terminal.get("unresolved_when")
        unres = bool(_eval_ast(unres_node, fields, counts)) if unres_node is not None else False
        how = "executable AST (compiled_world.json)"
    else:
        yes_src = str(rendered.get("yes_when") or "")
        unres_src = str(rendered.get("unresolved_when") or "False")
        yes = bool(_eval_rendered(yes_src, fields, counts))
        unres = bool(_eval_rendered(unres_src, fields, counts))
        how = "rendered terminal string (world_manifest.json) — no executable world persisted"
    if unres:
        return False, None, how
    return True, ("YES" if yes else "NO"), how


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #


def _branch_key(branch_id: str) -> str:
    """Branch ids appear with and without the structure prefix across artifacts."""

    return branch_id.split("/", 1)[1] if "/" in branch_id else branch_id


def _fields_from_events(
    events: list[dict[str, Any]], keep: Any = lambda e: True
) -> tuple[dict[str, Any], dict[str, int]]:
    """Replay the recorded ledger into (fields, event-type counts).

    This is the whole counterfactual machine: ``keep`` selects which recorded events
    are allowed to have happened, and the terminal is then re-evaluated over the
    result. Nothing is simulated and no model is called.
    """

    fields: dict[str, Any] = {}
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        if not keep(e):
            continue
        payload = e.get("payload") or {}
        kind = e.get("kind")
        if kind == "release_data":
            for k, v in (payload.get("fields") or {}).items():
                fields[k] = v
        elif kind == "set_field":
            if "field" in payload:
                fields[str(payload["field"])] = payload.get("value")
        elif kind == "adjust_field":
            name = str(payload.get("field", ""))
            delta = payload.get("delta")
            if name and delta is not None:
                fields[name] = float(fields.get(name) or 0) + float(delta)
        elif kind == "create_event":
            et = payload.get("event_type")
            if et:
                counts[str(et)] += 1
        elif kind == "append_record":
            coll = payload.get("collection")
            if coll:
                counts[str(coll)] += 1
    return fields, dict(counts)


def _coerce(value: Any) -> Any:
    """Recorded state values are stringified in the branch table; restore their type."""

    if not isinstance(value, str):
        return value
    if value in ("True", "False"):
        return value == "True"
    if value == "None":
        return None
    try:
        return float(value) if ("." in value or "e" in value.lower()) else int(value)
    except ValueError:
        return value


def _recorded_final_fields(branch: dict[str, Any]) -> dict[str, Any]:
    """The branch's final field values as the run itself recorded them."""

    out: dict[str, Any] = {}
    for key, value in (branch.get("world_state") or {}).items():
        if str(key).startswith("field:"):
            out[str(key)[len("field:") :]] = _coerce(value)
    return out


def _initial_fields(run: Path, branch: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """The branch's state before anything ran.

    Preferred source is the executable world the run persisted. Runs written before
    that artifact existed are reconstructed the only sound way available: a field that
    appears in the recorded FINAL state while no recorded event ever wrote it cannot
    have been produced by the trajectory, so its final value is also its initial value.
    That inference is itself the forensic finding for such a field — it was carried in,
    not produced — and it is reported as such rather than hidden.
    """

    world = _read_json(run / "compiled_world.json")
    if isinstance(world, dict):
        spec = world.get("world_spec") or {}
        fields = spec.get("fields") or []
        if fields:
            return {
                str(f["field_id"]): f.get("initial")
                for f in fields
                if isinstance(f, dict) and f.get("field_id") is not None
            }
    written = _fields_written_by_events(events)
    return {k: v for k, v in _recorded_final_fields(branch).items() if k not in written}


def _fields_written_by_events(events: list[dict[str, Any]]) -> set[str]:
    """Every field name some recorded event actually wrote."""

    written: set[str] = set()
    for e in events:
        payload = e.get("payload") or {}
        if e.get("kind") == "release_data":
            written |= {str(k) for k in (payload.get("fields") or {})}
        elif e.get("kind") in ("set_field", "adjust_field") and "field" in payload:
            written.add(str(payload["field"]))
    return written


def reconstruct(run: Path, label: str) -> dict[str, Any]:
    forecast = _read_json(run / "forecast.json", {})
    manifest = _read_json(run / "world_manifest.json", {})
    schedule = _read_json(run / "branch_schedule.json", {})
    stamp = _read_json(run / "run_stamp.json", {})
    events = _read_jsonl(run / "event_ledger.jsonl")
    decisions = _read_jsonl(run / "actor_decisions.jsonl")
    calls = _read_jsonl(run / "llm_calls.jsonl")
    audit = _read_json(run / "trajectory_audit.json", {})
    review = _read_json(run / "world_review.json", {})
    terminal_ast = _terminal_ast(run)
    rendered = manifest.get("terminal") or {}
    initial = _initial_fields(run)

    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        by_branch[_branch_key(str(e.get("branch_id") or ""))].append(e)

    branches = forecast.get("branches") or []
    per_branch: list[dict[str, Any]] = []
    notes: list[str] = []

    for b in branches:
        bid = str(b["branch_id"])
        key = _branch_key(bid)
        evs = by_branch.get(key, [])
        actor_events = [e for e in evs if e.get("actor_id")]
        process_events = [e for e in evs if not e.get("actor_id")]

        def state(keep: Any) -> tuple[dict[str, Any], dict[str, int]]:
            f, c = _fields_from_events(evs, keep)
            return {**initial, **f}, c

        full_fields, full_counts = state(lambda e: True)
        try:
            resolved, outcome, how = evaluate_terminal(
                terminal_ast, rendered, full_fields, full_counts
            )
        except Unreconstructable as exc:
            notes.append(f"{bid}: {exc}")
            resolved, outcome, how = False, None, f"UNRECONSTRUCTABLE: {exc}"

        entry: dict[str, Any] = {
            "branch_id": bid,
            "weight": b.get("weight"),
            "conditions": b.get("conditions"),
            "published_outcome": b.get("outcome"),
            "published_resolved": b.get("resolved"),
            "recomputed_outcome": outcome,
            "recomputed_resolved": resolved,
            "terminal_evaluated_via": how,
            "event_count": len(evs),
            "actor_event_count": len(actor_events),
            "process_event_count": len(process_events),
            "actor_invocations": (schedule.get(bid, {}) or {}).get("actor_invocations", {}),
            "final_fields": full_fields,
            "final_event_type_counts": full_counts,
            "matches_published": (outcome == b.get("outcome")) if resolved else None,
        }

        # -- counterfactual replays ------------------------------------------
        cf: dict[str, Any] = {}

        def outcome_of(keep: Any) -> str | None:
            f, c = state(keep)
            try:
                r, o, _ = evaluate_terminal(terminal_ast, rendered, f, c)
            except Unreconstructable:
                return "UNRECONSTRUCTABLE"
            return o if r else "UNRESOLVED"

        cf["all_actor_output_removed"] = outcome_of(lambda e: not e.get("actor_id"))
        cf["all_process_output_removed"] = outcome_of(lambda e: bool(e.get("actor_id")))
        cf["everything_removed"] = outcome_of(lambda e: False)
        actors = sorted({str(e["actor_id"]) for e in actor_events if e.get("actor_id")})
        cf["per_actor_removed"] = {
            a: outcome_of(lambda e, a=a: e.get("actor_id") != a) for a in actors
        }
        kinds = sorted({str(e.get("kind")) for e in process_events})
        cf["per_process_kind_removed"] = {
            k: outcome_of(lambda e, k=k: bool(e.get("actor_id")) or e.get("kind") != k)
            for k in kinds
        }
        cf["terminal_relevant_actions_removed"] = outcome_of(
            lambda e: e.get("kind") not in ("create_event", "append_record", "set_field")
        )
        entry["counterfactuals"] = cf
        per_branch.append(entry)

    # -- probability reconstruction -----------------------------------------
    yes = sum(float(b["weight"]) for b in per_branch if b["recomputed_outcome"] == "YES")
    no = sum(float(b["weight"]) for b in per_branch if b["recomputed_outcome"] == "NO")
    unres = sum(
        float(b["weight"])
        for b in per_branch
        if not b["recomputed_resolved"] or b["recomputed_outcome"] is None
    )
    resolved_mass = yes + no
    recomputed_p = (yes / resolved_mass) if resolved_mass > 1e-12 else None
    published_p = forecast.get("simulation_probability")

    terms = [
        f"{float(b['weight']):.4f}x{1 if b['recomputed_outcome'] == 'YES' else 0}"
        for b in per_branch
    ]
    substitution = (
        f"({' + '.join(terms)}) / {resolved_mass:.4f} = {yes:.4f} / {resolved_mass:.4f} = "
        + (f"{recomputed_p:.6f}" if recomputed_p is not None else "undefined")
    )

    weights = [float(b["weight"]) for b in per_branch]
    uniform = len(set(round(w, 9) for w in weights)) == 1 and len(weights) > 1
    perfect_fraction = uniform and recomputed_p is not None

    # weight provenance, from the run's own uncertainty record
    uncertainties = manifest.get("uncertainty") or []
    provenances = sorted(
        {
            str(o.get("provenance"))
            for u in uncertainties
            for o in (u.get("outcomes") or [])
            if o.get("provenance")
        }
    )
    all_ungrounded = bool(provenances) and all(
        p in ("symmetric_ignorance_assumption", "sensitivity_only") for p in provenances
    )

    # -- responsibility classification --------------------------------------
    published_outcomes = {b["recomputed_outcome"] for b in per_branch}
    actor_calls_total = sum(len(d) for d in [decisions])
    any_actor_changed = any(
        b["counterfactuals"]["all_actor_output_removed"] != b["recomputed_outcome"]
        for b in per_branch
    )
    any_process_changed = any(
        b["counterfactuals"]["all_process_output_removed"] != b["recomputed_outcome"]
        for b in per_branch
    )
    pre_resolved_all_yes = all(
        b["counterfactuals"]["everything_removed"] == "YES" for b in per_branch
    )

    # Is each branch's answer already fixed by its own condition tuple? If no two
    # branches share a tuple and none disagrees, the answer is decided the moment the
    # branch is chosen, and the probability is just the weight of the winning cells —
    # whatever happened in between.
    tuples: dict[str, set[str]] = defaultdict(set)
    for b in per_branch:
        tuples[json.dumps(b["conditions"], sort_keys=True)].add(str(b["recomputed_outcome"]))
    answer_is_a_function_of_the_conditions = (
        len(tuples) == len(per_branch)
        and all(len(v) == 1 for v in tuples.values())
        and len({next(iter(v)) for v in tuples.values()}) > 1
    )

    if any(b["matches_published"] is False for b in per_branch) or (
        published_p is not None
        and recomputed_p is not None
        and abs(published_p - recomputed_p) > 1e-9
    ):
        classification = "INVALID_TRACE"
    elif recomputed_p is None:
        classification = "UNRESOLVED"
    elif pre_resolved_all_yes:
        classification = "FACTUALLY_RESOLVED"
    elif not any_actor_changed and not any_process_changed:
        classification = "INITIAL_ASSUMPTIONS_DOMINATED"
    elif uniform and all_ungrounded and answer_is_a_function_of_the_conditions:
        # The trajectory may well have been necessary to produce any YES at all — that
        # is recorded separately — but it did not fix the NUMBER. Equal, ungrounded
        # weights over cells whose answers are decided by their own conditions make the
        # magnitude an enumeration artifact: re-weight the same cells to any
        # evidence-plausible asymmetry and the probability moves with nothing else
        # changing. That is what dominated means here.
        classification = "BRANCH_WEIGHTS_DOMINATED"
    elif any_actor_changed and uniform and all_ungrounded:
        classification = "PARTLY_TRAJECTORY_CAUSED"
    elif any_actor_changed or any_process_changed:
        classification = "TRAJECTORY_CAUSED"
    else:
        classification = "INITIAL_ASSUMPTIONS_DOMINATED"

    tokens_in = sum(int(c.get("tokens_in") or 0) for c in calls)
    tokens_out = sum(int(c.get("tokens_out") or 0) for c in calls)
    est_cost = tokens_in / 1e6 * RATE_IN_PER_MTOK + tokens_out / 1e6 * RATE_OUT_PER_MTOK

    mismatches: list[dict[str, Any]] = []

    def check(name: str, published: Any, recomputed: Any) -> None:
        if published is None:
            return
        ok = (
            abs(float(published) - float(recomputed)) < 1e-6
            if isinstance(published, (int, float)) and isinstance(recomputed, (int, float))
            else published == recomputed
        )
        if not ok:
            mismatches.append(
                {
                    "field": name,
                    "published": published,
                    "recomputed": recomputed,
                    "severity": "CRITICAL",
                }
            )

    check("simulation_probability", published_p, recomputed_p)
    check("resolved_yes_mass", forecast.get("resolved_yes_mass"), yes)
    check("resolved_no_mass", forecast.get("resolved_no_mass"), no)
    check("unresolved_mass", forecast.get("unresolved_mass"), unres)
    check("model_call_count", forecast.get("model_call_count"), len(calls))
    check("token_usage", forecast.get("token_usage"), tokens_in + tokens_out)

    return {
        "label": label,
        "run_dir": str(run),
        "commit": stamp.get("commit"),
        "question": forecast.get("question"),
        "published": {
            "probability": published_p,
            "probability_source": forecast.get("probability_source"),
            "status": forecast.get("status"),
            "yes_mass": forecast.get("resolved_yes_mass"),
            "no_mass": forecast.get("resolved_no_mass"),
            "unresolved_mass": forecast.get("unresolved_mass"),
            "lower_bound": forecast.get("lower_bound"),
            "upper_bound": forecast.get("upper_bound"),
            "model_calls": forecast.get("model_call_count"),
            "tokens": forecast.get("token_usage"),
        },
        "recomputed": {
            "probability": recomputed_p,
            "yes_mass": yes,
            "no_mass": no,
            "unresolved_mass": unres,
            "substitution": substitution,
            "model_calls": len(calls),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "estimated_cost_usd": round(est_cost, 4),
            "estimated_cost_note": (
                f"ESTIMATE: recorded tokens x list rate (${RATE_IN_PER_MTOK}/Mtok in, "
                f"${RATE_OUT_PER_MTOK}/Mtok out); the trace records tokens, not price"
            ),
            "events": len(events),
            "actor_decisions": len(decisions),
        },
        "weights": {
            "branch_weights": weights,
            "all_equal": uniform,
            "perfect_fraction_from_branch_weights": perfect_fraction,
            "outcome_provenances": provenances,
            "all_weights_ungrounded": all_ungrounded,
            "flag": (
                "PERFECT_FRACTION_FROM_BRANCH_WEIGHTS"
                if perfect_fraction
                else "no equal-weight artifact"
            ),
        },
        "branches": per_branch,
        "responsibility": {
            "classification": classification,
            "any_actor_output_changed_an_outcome": any_actor_changed,
            "any_process_output_changed_an_outcome": any_process_changed,
            "terminal_already_satisfied_with_everything_removed": pre_resolved_all_yes,
            "actor_invocations_total": actor_calls_total,
        },
        "audit_classification": audit.get("classification"),
        "world_review_disposition": review.get("disposition"),
        "world_review_blocking": review.get("blocking_failures"),
        "mismatches": mismatches,
        "reconstruction_notes": notes,
        "verdict": (
            "FORENSICALLY_INVALID"
            if mismatches or classification == "INVALID_TRACE"
            else "RECONSTRUCTED"
        ),
    }


# --------------------------------------------------------------------------- #
# Artifact emission
# --------------------------------------------------------------------------- #


def emit(run: Path, out: Path, label: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    result = reconstruct(run, label)

    events = _read_jsonl(run / "event_ledger.jsonl")
    decisions = _read_jsonl(run / "actor_decisions.jsonl")
    calls = _read_jsonl(run / "llm_calls.jsonl")

    def dump(name: str, rows: list[dict[str, Any]]) -> None:
        (out / name).write_text(
            "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows)
        )

    # 1. chronological timeline, grouped by branch then simulation time
    timeline: list[dict[str, Any]] = []
    seq = 0
    for e in sorted(events, key=lambda e: (str(e.get("branch_id")), str(e.get("time")))):
        seq += 1
        payload = e.get("payload") or {}
        timeline.append(
            {
                "seq": seq,
                "branch_id": e.get("branch_id"),
                "simulation_time": e.get("time"),
                "stage": "simulation",
                "occurrence": e.get("kind"),
                "actor_id": e.get("actor_id"),
                "event_id": e.get("event_id"),
                "detail": {
                    k: payload.get(k)
                    for k in ("field", "value", "event_type", "collection", "fields", "to")
                    if k in payload
                },
                "evidence_claim_ids": e.get("evidence_claim_ids") or [],
            }
        )
    for d in decisions:
        seq += 1
        timeline.append(
            {
                "seq": seq,
                "branch_id": d.get("branch_id"),
                "simulation_time": d.get("branch_time"),
                "stage": "actor_invocation",
                "occurrence": f"actor_awakened:{d.get('wake_reason')}",
                "actor_id": d.get("actor_id"),
                "detail": {
                    "intent_mode": (d.get("intent") or {}).get("mode"),
                    "action_id": (d.get("intent") or {}).get("action_id"),
                    "validation_status": d.get("validation_status"),
                },
            }
        )
    timeline.sort(key=lambda r: (str(r["branch_id"]), str(r["simulation_time"]), r["seq"]))
    dump("forensic_timeline.jsonl", timeline)

    # 2. every LLM call, in order, complete as recorded
    full_calls = []
    for i, c in enumerate(calls, start=1):
        row = dict(c)
        row["call_number"] = c.get("call_number", i)
        row["estimated_cost_usd"] = round(
            int(c.get("tokens_in") or 0) / 1e6 * RATE_IN_PER_MTOK
            + int(c.get("tokens_out") or 0) / 1e6 * RATE_OUT_PER_MTOK,
            6,
        )
        for missing in ("prompt", "started_at", "ended_at", "latency_ms"):
            if missing not in row:
                row[f"__missing_{missing}"] = (
                    "not recorded by the run that produced this trace; "
                    "instrumented from commit 71393bb onward"
                )
        full_calls.append(row)
    dump("llm_calls_full.jsonl", full_calls)

    # 3. actor invocations — the complete decision record, as written
    dump("actor_invocations.jsonl", decisions)

    # 4. communications
    comms = []
    for e in events:
        if e.get("kind") in ("send_information", "deliver_information", "create_event"):
            payload = e.get("payload") or {}
            comms.append(
                {
                    "branch_id": e.get("branch_id"),
                    "kind": e.get("kind"),
                    "time": e.get("time"),
                    "sender": e.get("actor_id"),
                    "recipients": payload.get("to") or payload.get("participants") or [],
                    "visibility": e.get("visibility"),
                    "content": payload.get("information_created")
                    or payload.get("event_type")
                    or payload.get("description"),
                    "event_id": e.get("event_id"),
                }
            )
    dump("communications.jsonl", comms)

    # 5. non-actor process transitions
    procs = []
    for e in events:
        if e.get("actor_id"):
            continue
        payload = e.get("payload") or {}
        procs.append(
            {
                "branch_id": e.get("branch_id"),
                "time": e.get("time"),
                "operation": e.get("kind"),
                "inputs": payload.get("fields") or payload.get("branch_conditions") or {},
                "outputs": {
                    k: payload.get(k) for k in ("field", "value", "event_type") if k in payload
                },
                "event_id": e.get("event_id"),
                "evidence_claim_ids": e.get("evidence_claim_ids") or [],
            }
        )
    dump("process_transitions.jsonl", procs)

    # 6. state diffs, replayed in ledger order
    diffs = []
    for bid in sorted({str(e.get("branch_id")) for e in events}):
        fields: dict[str, Any] = {}
        for e in [e for e in events if str(e.get("branch_id")) == bid]:
            before = dict(fields)
            payload = e.get("payload") or {}
            if e.get("kind") == "release_data":
                fields.update(payload.get("fields") or {})
            elif e.get("kind") == "set_field" and "field" in payload:
                fields[str(payload["field"])] = payload.get("value")
            after = dict(fields)
            if before != after:
                diffs.append(
                    {
                        "branch_id": bid,
                        "event_id": e.get("event_id"),
                        "simulation_time": e.get("time"),
                        "triggered_by": e.get("actor_id") or f"process:{e.get('kind')}",
                        "state_before": before,
                        "state_after": after,
                        "diff": {
                            k: {"from": before.get(k), "to": after.get(k)}
                            for k in after
                            if before.get(k) != after.get(k)
                        },
                        "runtime_operation": e.get("kind"),
                        "evidence_claim_ids": e.get("evidence_claim_ids") or [],
                    }
                )
    dump("state_diffs.jsonl", diffs)

    # 7. branch weight history
    weights = []
    for b in result["branches"]:
        weights.append(
            {
                "branch_id": b["branch_id"],
                "assumptions": b["conditions"],
                "starting_weight": b["weight"],
                "final_weight": b["weight"],
                "weight_changes": [],
                "provenance": result["weights"]["outcome_provenances"],
                "reason_for_starting_weight": (
                    "uniform split across the enumerated alternatives of each uncertainty; "
                    "every outcome carries "
                    + ", ".join(result["weights"]["outcome_provenances"] or ["(none recorded)"])
                ),
                "terminal_result": b["recomputed_outcome"],
                "contribution_to_probability": (
                    float(b["weight"]) if b["recomputed_outcome"] == "YES" else 0.0
                ),
            }
        )
    dump("branch_weight_history.jsonl", weights)

    # 8. semantic -> runtime lineage
    trace = _read_json(run / "research_trace.json", {})
    sem = trace.get("semantic_compilation") or {}
    rounds = trace.get("semantic_repair_rounds") or []
    if rounds and isinstance(rounds[-1], dict):
        sem = rounds[-1]
    mapping = sem.get("mapping") or []
    lineage = []
    for m in mapping if isinstance(mapping, list) else []:
        rid = str(m.get("runtime_id") or "")
        touching = [
            e.get("event_id")
            for e in events
            if rid
            and rid
            in json.dumps({"p": e.get("payload"), "k": e.get("kind")}, default=str)
        ]
        lineage.append(
            {
                "evidence_claim_ids": m.get("evidence_claim_ids") or [],
                "semantic_object": m.get("semantic"),
                "namespace": m.get("namespace"),
                "runtime_id": rid,
                "lowering_rule": m.get("lowering_rule"),
                "event_ids": touching,
                "executed": bool(touching),
            }
        )
    dump("semantic_runtime_lineage.jsonl", lineage)

    # 9. terminal evaluations
    dump(
        "terminal_evaluations.jsonl",
        [
            {
                "branch_id": b["branch_id"],
                "terminal_evaluated_via": b["terminal_evaluated_via"],
                "fields_read": b["final_fields"],
                "event_type_counts": b["final_event_type_counts"],
                "recomputed_resolved": b["recomputed_resolved"],
                "recomputed_outcome": b["recomputed_outcome"],
                "published_outcome": b["published_outcome"],
                "matches_published": b["matches_published"],
            }
            for b in result["branches"]
        ],
    )

    # 10-12. reconstruction, responsibility, verdict
    (out / "probability_reconstruction.json").write_text(
        json.dumps(
            {
                "published": result["published"],
                "recomputed": result["recomputed"],
                "aggregation_formula": (
                    "P(YES) = sum(branch_weight x yes_indicator) / sum(resolved branch_weight)"
                ),
                "numeric_substitution": result["recomputed"]["substitution"],
                "weights": result["weights"],
                "mismatches": result["mismatches"],
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    (out / "trajectory_responsibility.json").write_text(
        json.dumps(
            {
                "classification": result["responsibility"]["classification"],
                "signals": result["responsibility"],
                "per_branch_counterfactuals": {
                    b["branch_id"]: b["counterfactuals"] for b in result["branches"]
                },
                "method": (
                    "deterministic replay of the recorded event ledger with selected "
                    "events removed, then terminal re-evaluation; no simulation, no "
                    "model call"
                ),
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    (out / "forensic_verdict.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=str) + "\n"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    result = emit(Path(args.run), Path(args.out), args.label)
    print(f"=== {args.label}: {result['verdict']}")
    print(f"  published  p = {result['published']['probability']}")
    print(f"  recomputed p = {result['recomputed']['probability']}")
    print(f"  substitution: {result['recomputed']['substitution']}")
    print(f"  weights: {result['weights']['flag']}")
    print(f"  provenance: {result['weights']['outcome_provenances']}")
    print(f"  classification: {result['responsibility']['classification']}")
    if result["mismatches"]:
        for m in result["mismatches"]:
            print(f"  CRITICAL MISMATCH {m['field']}: {m['published']} vs {m['recomputed']}")
    for n in result["reconstruction_notes"]:
        print(f"  note: {n}")
    return 0 if result["verdict"] == "RECONSTRUCTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
