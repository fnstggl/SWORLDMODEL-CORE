"""Tracing and the under-the-hood report.

Writes the replayable event ledger, the LLM-call log, actor-decision records, the
evidence and world manifests, and the human-readable report. The report shows enough
to reconstruct the probability by hand: every branch, the decisive world state, the
deterministic terminal evaluation, and the aggregate arithmetic. Serialization is
deterministic so artifacts are hashable and comparable across runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import replaycore
from .compiled import CompiledWorld
from .engine import RunResult
from .evidence import EvidenceStore
from .ids import canonical_json, sha256_hex
from .models import ForecastResult, ResolutionContract
from .research import ResearchBundle
from .temporal_report import write_temporal_report


@dataclass
class TraceContext:
    contract: ResolutionContract
    evidence_store: EvidenceStore
    as_of: datetime
    bundle: ResearchBundle
    compiled: CompiledWorld
    run_result: RunResult
    forecast: ForecastResult
    model_id: str
    # What the compiler concluded about whether it compiled the *right* world, and the
    # provider response that said so. Carried in the trace because a reader must be able
    # to see which structures were simulated and which could not be represented.
    structure_assessment: Any = None
    structure_response: Any = None
    # The targeted-repair attempts this run made, for the diagnosis artifact.
    repair_log: Any = None
    # The pre-rollout world audit: what it asked, and what it answered.
    world_review: Any = None
    # The post-simulation trajectory audit: how the run actually unfolded, and the
    # mechanical classification of what kind of result it was.
    trajectory_audit: Any = None
    _calls_override: list[Any] = field(default_factory=list)

    # -- serializable payloads --------------------------------------------------

    def forecast_payload(self) -> dict[str, Any]:
        f = self.forecast
        return {
            "question": f.question,
            "contract_id": f.contract.contract_id,
            "as_of": self.as_of.isoformat(),
            "horizon": f.contract.horizon.isoformat(),
            "probability_source": f.probability_source,
            "status": f.status.value,
            "simulation_probability": f.simulation_probability,
            "lower_bound": f.lower_bound,
            "upper_bound": f.upper_bound,
            "resolved_mass": f.resolved_mass,
            "resolved_yes_mass": f.resolved_yes_mass,
            "resolved_no_mass": f.resolved_no_mass,
            "unresolved_mass": f.unresolved_mass,
            "integrity_verdict": f.integrity_manifest.integrity_verdict.value,
            "expected_participants": f.integrity_manifest.expected_participants,
            "represented_participants": f.integrity_manifest.represented_participants,
            "model": self.model_id,
            "model_call_count": f.model_call_count,
            "token_usage": f.token_usage,
            # The integrity block was computed on every run and written to no artifact:
            # p_before, the shift the simulation produced, and the counterfactual note
            # were visible only on the console. They are the numbers a reviewer needs
            # FIRST — whether the trajectories moved the answer at all — and a forensic
            # audit had to rebuild them from the raw ledger to find out.
            "integrity": f.integrity.as_dict() if f.integrity is not None else None,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "weight": b.weight,
                    "resolved": b.resolved,
                    "outcome": b.outcome,
                    "unresolved_reason": b.unresolved_reason,
                    "conditions": dict(b.key_conditions),
                    "world_state": dict(b.records),
                    # Per branch, what the terminal already said at initialization.
                    # Comparing it with `outcome` is how a reader sees which branches
                    # the simulation actually decided.
                    "pre_resolved": b.pre_resolved,
                    "pre_outcome": b.pre_outcome,
                    "weight_grounded": b.weight_grounded,
                    "event_count": b.event_count,
                }
                for b in f.branch_outcomes
            ],
        }

    def evidence_manifest(self) -> dict[str, Any]:
        """Every claim with enough provenance to be re-verified against its source.

        A manifest that records only that a claim exists is not an audit trail. Each
        entry carries the URL actually fetched (the archived capture, for a pastcast),
        the excerpt it was verified against, and the content hash of the document it
        came from, so a reader can go back to the page and check.
        """

        claims = self.evidence_store.all()
        provenance = {r["claim_id"]: r for r in self.evidence_store.provenance_records()}
        return {
            "claim_count": len(claims),
            "independent_event_count": len(self.evidence_store.independent_event_ids()),
            "as_of": self.as_of.isoformat(),
            "lineage_groups": {
                k: [c.id for c in v] for k, v in self.evidence_store.lineage_groups().items()
            },
            "contradictions": self.evidence_store.contradictions(),
            "claims": [
                {
                    "id": c.id,
                    "proposition": c.proposition,
                    "normalized_value": c.normalized_value,
                    "entities": list(c.entities),
                    "published_at": c.published_at.isoformat(),
                    "available_at": c.available_at.isoformat(),
                    "available_by_cutoff": c.is_available_at(self.as_of),
                    "source_id": c.source_id,
                    "source_type": c.source_type.value,
                    "authority_level": int(c.authority_level),
                    "epistemic_type": c.epistemic_type.value,
                    "lineage_event_id": c.lineage_event_id,
                    "source_url": c.source_url,
                    "supporting_excerpt": c.supporting_excerpt,
                    "provenance": provenance.get(c.id, {}),
                }
                for c in claims
            ],
        }

    def world_manifest(self) -> dict[str, Any]:
        m = self.compiled.manifest
        spec = self.compiled.spec
        return {
            "integrity_verdict": m.integrity_verdict.value,
            "expected_participants": m.expected_participants,
            "represented_participants": m.represented_participants,
            "verified_roles": [list(x) for x in m.verified_roles],
            "verified_authorities": [[a, list(p)] for a, p in m.verified_authorities],
            "verified_rules": list(m.verified_rules),
            "missing_required_facts": list(m.missing_required_facts),
            "unresolved_conflicts": list(m.unresolved_conflicts),
            "evidence_coverage": m.evidence_coverage,
            "compiled_actions": [
                {
                    "action_id": a.action_id,
                    "meaning": a.meaning,
                    "required_authority": list(a.required_authority),
                    "effects": [e.op for e in a.effects],
                }
                for a in spec.actions
            ],
            "process_graph": [
                {
                    "node_id": n.node_id,
                    "stage": n.stage,
                    "participants": list(n.participants),
                    "action_ids": list(n.action_ids),
                    "allow_novel": n.allow_novel,
                }
                for n in spec.process.nodes
            ],
            "terminal": {
                "yes_when": _expr_repr(spec.terminal.yes_when),
                "unresolved_when": _expr_repr(spec.terminal.unresolved_when),
                "description": spec.terminal.description,
            },
            "uncertainty": [
                {
                    "variable": u.variable_id,
                    "why_unknown": u.why_unknown,
                    "reversal_capable": u.reversal_capable,
                    "outcomes": [
                        {
                            "value": o.value,
                            "weight": o.weight.value,
                            "provenance": o.weight.provenance.value,
                        }
                        for o in u.outcomes
                    ],
                }
                for u in self.compiled.uncertainty_variables
            ],
        }

    def coverage_manifest(self) -> dict[str, Any]:
        """The evidence-to-world coverage report: what verified reality contained and
        what became of every candidate during compilation."""

        return self.compiled.coverage_report.to_dict()

    def event_ledger_lines(self) -> list[str]:
        return [
            canonical_json(
                {
                    "event_id": e.event_id,
                    "branch_id": e.branch_id,
                    "time": e.time.isoformat(),
                    "kind": e.kind,
                    "actor_id": e.actor_id,
                    "payload": dict(e.payload),
                    "visibility": e.visibility.value,
                    "evidence_claim_ids": list(e.evidence_claim_ids),
                }
            )
            for e in self.run_result.event_ledger
        ]

    def llm_call_lines(self) -> list[str]:
        """Every provider call, in order, complete enough to reconstruct the run.

        ``call_number`` gives the chronology an independent reviewer needs; ``prompt``
        and ``raw_text`` are the exact request and the exact unparsed response beside
        the parsed one, so a reviewer can check the parse rather than trust it. A log
        carrying only a prompt hash proves that a call happened and nothing about what
        it asked.
        """

        lines = []
        for i, r in enumerate(self._gateway_calls(), start=1):
            lines.append(
                canonical_json(
                    {
                        "call_number": i,
                        "task_kind": r.task_kind,
                        "model": r.model,
                        "seed": r.seed,
                        "prompt_hash": r.prompt_hash,
                        "started_at": r.started_at,
                        "ended_at": r.ended_at,
                        "latency_ms": r.latency_ms,
                        "tokens_in": r.tokens_in,
                        "tokens_out": r.tokens_out,
                        "retries": r.retries,
                        "validation_failures": list(r.validation_failures),
                        "prompt": r.prompt,
                        "raw_text": r.raw_text,
                        "response": r.data,
                    }
                )
            )
        return lines

    def actor_decision_lines(self) -> list[str]:
        """One record per actor invocation — the stable contract a replay or a frontend
        reads.

        Each record answers, without reconstruction: what woke this actor and when, what
        had been delivered to it, what it actually noticed, which memories it retrieved,
        what its plan was before, what it decided to do with that plan, what it intended,
        what the world allowed, and what its state was afterwards. The prompt is the
        byte-exact string that was sent.
        """

        out = []
        for d in self.run_result.actor_decisions:
            ctx = dict(d.decision_context)
            prompt = ctx.pop("rendered_prompt", "")
            response = ctx.pop("provider_response", None)
            out.append(
                canonical_json(
                    {
                        "branch_id": d.branch_id,
                        "actor_id": d.actor_id,
                        "canonical_identity": ctx.get("canonical_identity"),
                        "branch_time": d.branch_time,
                        "stage": d.stage,
                        "wake_reason": d.wake_reason,
                        "wake_detail": d.wake_detail,
                        "trigger_event_ids": d.trigger_event_ids,
                        "delivered_observation_ids": d.delivered_observation_ids,
                        "noticed_observation_ids": d.noticed_observation_ids,
                        "retrieved_memory_ids": d.retrieved_memory_ids,
                        "plan_before": d.plan_before,
                        "plan_disposition": d.plan_disposition,
                        "plan_after": d.plan_after,
                        "state_before": d.state_before,
                        "state_after": d.state_after,
                        "actor_evidence_claim_ids": ctx.get("actor_evidence_claim_ids", []),
                        "feasible_actions": ctx.get("feasible_actions", []),
                        "local_view": ctx,
                        "exact_prompt": prompt,
                        "provider_response": response,
                        "intent": d.intent,
                        "validation_status": d.validation_status,
                        "validation_reason": d.validation_reason,
                        "applied_event_ids": d.event_ids,
                        "world_version_at_decision": d.world_version_at_decision,
                        "prompt_hash": d.prompt_hash,
                        "model": d.model,
                    }
                )
            )
        return out

    def actor_grounding_manifest(self) -> dict[str, Any]:
        """Each actor's grounding profile, with provenance for every element."""

        out: dict[str, Any] = {}
        for aid, state in self.compiled.base_world.actors.items():
            profile = state.grounding
            as_dict = getattr(profile, "as_dict", None)
            if callable(as_dict):
                out[aid] = as_dict()
        return out

    def _gateway_calls(self) -> list[Any]:
        return list(self._calls_override)

    # -- writing ----------------------------------------------------------------

    def write(
        self, out_dir: Path, *, sealed_names: bool = False, gateway_calls: list[Any] | None = None
    ) -> str:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._calls_override = gateway_calls or []
        forecast_name = "pre_outcome_forecast.json" if sealed_names else "forecast.json"
        report_name = "pre_outcome_report.md" if sealed_names else "report.md"

        forecast_text = canonical_json(self.forecast_payload())
        (out_dir / forecast_name).write_text(forecast_text + "\n")
        forecast_hash = sha256_hex(forecast_text)

        (out_dir / "evidence_manifest.json").write_text(
            canonical_json(self.evidence_manifest()) + "\n"
        )
        (out_dir / "world_manifest.json").write_text(canonical_json(self.world_manifest()) + "\n")
        # The executable world itself, not a summary of it. world_manifest.json renders
        # the terminal as a human-readable string; replaying a run needs the exact dict
        # parse_world_spec consumed. Without it a completed forecast cannot be
        # re-executed or independently re-evaluated without recompiling through an LLM.
        executed = getattr(self.bundle, "executed_compilation", None)
        if executed is not None:
            (out_dir / "compiled_world.json").write_text(canonical_json(executed) + "\n")
        (out_dir / "coverage_report.json").write_text(
            canonical_json(self.coverage_manifest()) + "\n"
        )
        (out_dir / "actor_grounding.json").write_text(
            canonical_json(self.actor_grounding_manifest()) + "\n"
        )
        (out_dir / "structural_uncertainty.json").write_text(
            canonical_json(self.structure_manifest()) + "\n"
        )
        (out_dir / "branch_schedule.json").write_text(
            canonical_json(self.schedule_manifest()) + "\n"
        )
        (out_dir / "event_ledger.jsonl").write_text("\n".join(self.event_ledger_lines()) + "\n")
        (out_dir / "llm_calls.jsonl").write_text("\n".join(self.llm_call_lines()) + "\n")
        (out_dir / "actor_decisions.jsonl").write_text(
            "\n".join(self.actor_decision_lines()) + "\n"
        )
        write_temporal_report(out_dir, self.run_result)  # §9 counters (TMP-2, FD-13)
        for name, audit in (
            ("world_review.json", self.world_review),
            ("trajectory_audit.json", self.trajectory_audit),
        ):
            audit_dict = getattr(audit, "as_dict", None)
            if callable(audit_dict):
                (out_dir / name).write_text(canonical_json(audit_dict()) + "\n")
        self._write_observability(out_dir)
        (out_dir / report_name).write_text(self.render_report(forecast_hash))
        return forecast_hash

    # -- observability (OBS-1..4, OBS-7, OBS-8; derived through the D7 replay core) --

    def _executable_world(self) -> Any:
        """The run's own executable world, for replay: the exact compilation dict the
        engine executed when the bundle carries it, else the parsed spec itself."""

        executed = getattr(self.bundle, "executed_compilation", None)
        return executed if executed is not None else self.compiled.spec

    def _observability_record(self, out_dir: Path) -> replaycore.RunRecord:
        """This run, presented to the replay core exactly as a later reader sees it."""

        def as_dict(obj: Any) -> dict[str, Any]:
            method = getattr(obj, "as_dict", None)
            return dict(method()) if callable(method) else {}

        stamp_path = out_dir / "run_stamp.json"
        try:
            stamp = json.loads(stamp_path.read_text()) if stamp_path.exists() else {}
        except (OSError, ValueError):
            stamp = {}
        return replaycore.RunRecord(
            label=out_dir.name or "run",
            run_dir=str(out_dir),
            forecast=self.forecast_payload(),
            manifest=self.world_manifest(),
            schedule=self.schedule_manifest(),
            stamp=stamp if isinstance(stamp, dict) else {},
            events=list(self.run_result.event_ledger),
            decisions=list(self.run_result.actor_decisions),
            calls=self._gateway_calls(),
            audit=as_dict(self.trajectory_audit),
            review=as_dict(self.world_review),
            world=self._executable_world(),
        )

    def _write_observability(self, out_dir: Path) -> None:
        """Persist the complete observability set, at trace-write time, derived from
        the run's own record through the ONE shared replay core (decision D7).

        Everything here is a deterministic replay of what this run already recorded —
        no model call, no re-simulation, no second replay implementation. A value the
        run did not record is written as absent/unknown, never guessed.
        """

        world = self._executable_world()
        initials = replaycore.initial_fields_from_world(world) or {}
        events = list(self.run_result.event_ledger)
        decisions = list(self.run_result.actor_decisions)
        by_branch = replaycore.group_events_by_branch(events)

        # OBS-1: per namespaced branch, the complete field state after this branch's
        # scenario conditions and before any simulated effect.
        initial_states: dict[str, Any] = {}
        initial_by_key: dict[str, dict[str, Any]] = {}
        for b in self.run_result.branch_outcomes:
            key = replaycore.branch_key(b.branch_id)
            branch_events = by_branch.get(key, [])
            state = replaycore.branch_initial_state(initials, branch_events)
            initial_by_key[key] = state
            initial_states[b.branch_id] = {
                "branch_key": key,
                "conditions": dict(b.key_conditions),
                "fields": state,
                "source": (
                    "executed compilation field initials + this branch's scenario "
                    "release_data conditions; a declared field with no initial and no "
                    "scenario condition reads null (never written, never invented)"
                ),
            }
        # A branch that appears in the ledger without a branch outcome still replays
        # from the compiled initials rather than silently from nothing.
        for key, branch_events in by_branch.items():
            if key not in initial_by_key:
                initial_by_key[key] = replaycore.branch_initial_state(initials, branch_events)
        (out_dir / "branch_initial_state.json").write_text(canonical_json(initial_states) + "\n")

        # OBS-2 (§12): every state diff, with trigger, authority lineage and terminal
        # relevance; initial state + these diffs reconstruct every branch exactly.
        terminal_fields = replaycore.terminal_field_reads(
            replaycore.terminal_ast_from_world(world),
            dict(self.world_manifest().get("terminal") or {}),
        )
        diffs = replaycore.derive_state_diffs(
            events, initial_by_branch=initial_by_key, terminal_fields=terminal_fields
        )
        _write_jsonl(out_dir / "state_diffs.jsonl", diffs)

        # OBS-3 (§11): every communication, with the honest delivery→notice join
        # against the recorded actor decisions.
        _write_jsonl(
            out_dir / "communications.jsonl",
            replaycore.extract_communications(events, decisions),
        )

        # OBS-4: every non-actor process transition.
        _write_jsonl(
            out_dir / "process_transitions.jsonl",
            replaycore.extract_process_transitions(events),
        )

        # OBS-8 (§13): the chronological dossier, rendered from the same
        # reconstruction the forensic tool performs — at trace-write time.
        record = self._observability_record(out_dir)
        result = replaycore.reconstruct_run(record)
        timeline = replaycore.build_timeline(events, decisions)
        (out_dir / "run_dossier.html").write_text(
            replaycore.render_dossier(
                result,
                timeline,
                decisions,
                record.calls,
                title_prefix="Run dossier",
                artifacts_note=replaycore.RUN_ARTIFACTS_NOTE,
            )
        )

    def structure_manifest(self) -> dict[str, Any]:
        """Whether the compiled causal structure was treated as settled, which
        alternatives were simulated, and which could not be represented."""

        a = self.structure_assessment
        as_dict = getattr(a, "as_dict", None)
        return (
            as_dict()
            if callable(as_dict)
            else {"is_material": False, "reason": "not assessed", "alternatives": []}
        )

    def schedule_manifest(self) -> dict[str, Any]:
        """The branch calendars: what ran, what stopped them, and what was still
        scheduled when the question's window closed. This is what a replay or a
        visualization reads to reconstruct time without re-running anything."""

        return {
            branch: {
                "batches": d.batches,
                "events": d.events,
                "stop_reason": d.stop_reason,
                "actor_invocations": dict(d.actor_call_counts),
                "unfired_in_horizon": d.unfired_in_horizon,
                "pending_beyond_horizon": d.pending_beyond_horizon,
            }
            for branch, d in sorted(self.run_result.diagnostics.items())
        }

    # -- report -----------------------------------------------------------------

    def render_report(self, forecast_hash: str) -> str:
        f = self.forecast
        m = self.compiled.manifest
        spec = self.compiled.spec
        lines: list[str] = []
        add = lines.append

        add("# Under-the-hood forecast report\n")
        add(f"**Question:** {f.question}\n")
        add(
            f"**Information cutoff (as_of):** {self.as_of.isoformat()}  \n"
            f"**Horizon:** {f.contract.horizon.isoformat()}\n"
        )
        add(f"**Pre-outcome forecast SHA-256:** `{forecast_hash}`\n")

        add("## 1. Resolution contract")
        add(
            f"- subject entity: {f.contract.subject_entity}\n"
            f"- resolution units: {f.contract.resolution_units}\n"
            f"- target (YES condition): {f.contract.target_outcome}\n"
            f"- declarative terminal: {spec.terminal.description or _expr_repr(spec.terminal.yes_when)}\n"
            f"- expected participants: {f.contract.expected_participants}\n"
        )

        add("## 2-4. Evidence, contradictions, lineage")
        em = self.evidence_manifest()
        add(
            f"- claims: {em['claim_count']} across {em['independent_event_count']} "
            f"independent events (lineage-deduplicated)\n"
            f"- contradictions: {em['contradictions'] or 'none'}\n"
            f"- research plan (backward): {' '.join(self.bundle.research_plan)}\n"
        )

        add("## 5. Verified entities, roles and capabilities")
        for aid, role in m.verified_roles:
            auth = dict(m.verified_authorities).get(aid, ())
            add(f"- {aid} — {role} [authority: {', '.join(auth) or 'none'}]")
        add(
            f"\n- expected participants: {m.expected_participants}, represented: "
            f"{m.represented_participants}, verdict: {m.integrity_verdict.value}\n"
        )

        add("## 6. Compiled actions (scenario-specific, mapped to universal effects)")
        for a in spec.actions:
            add(
                f"- **{a.action_id}**: {a.meaning} -> effects [{', '.join(e.op for e in a.effects)}]"
            )
        add("")

        add("## 7. Compiled process graph, external processes and wake rules")
        for n in spec.process.nodes:
            when = n.at or (f"after {n.after_node}+{n.delay_seconds}s" if n.after_node else "start")
            add(
                f"- **{n.node_id}** ({n.stage}) at {when}; participants "
                f"{list(n.participants) or 'none'} -> {list(n.next_nodes) or 'end'}"
            )
        for proc in spec.external_processes:
            add(
                f"- external **{proc.process_id}**: {proc.description} "
                f"({len(proc.occurrences)} occurrences)"
            )
        for rule in spec.wake_rules:
            add(f"- wake **{rule.rule_id}**: wakes {list(rule.wakes)} because {rule.reason}")
        add("")

        add("## 8. External uncertain events and branch weights")
        for u in self.compiled.uncertainty_variables:
            add(f"- **{u.variable_id}** ({u.why_unknown}); reversal-capable={u.reversal_capable}")
            for o in u.outcomes:
                add(f"    - {o.value}: weight {o.weight.value} [{o.weight.provenance.value}]")
        add("")

        add("## 9. Actor invocations (what woke them -> what they intended -> what happened)")
        add(
            "Invocation counts differ by actor and by branch because different things "
            "happened to them. Nothing here is scheduled.\n"
        )
        for d in self.run_result.actor_decisions:
            add(_invocation_line(d))
        add("")

        add("### 9b. Invocations per actor per branch")
        per: dict[tuple[str, str], list[str]] = {}
        for d in self.run_result.actor_decisions:
            per.setdefault((d.branch_id, d.actor_id), []).append(d.wake_reason)
        for (branch, actor), reasons in sorted(per.items()):
            counts: dict[str, int] = {}
            for r in reasons:
                counts[r] = counts.get(r, 0) + 1
            add(
                f"- {branch} / {actor}: {len(reasons)} — "
                + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
            )
        add("")

        add("### 9c. Branch execution diagnostics")
        for bid, diag in sorted(self.run_result.diagnostics.items()):
            add(
                f"- {bid}: {diag.batches} time-batches, {diag.events} events, "
                f"stopped because *{diag.stop_reason}*; "
                f"{diag.unfired_in_horizon} entries never fired in-horizon, "
                f"{len(diag.pending_beyond_horizon)} scheduled beyond the horizon"
            )
        add("")

        add("## 10. Branches, decisive world state, terminal outcomes")
        for b in f.branch_outcomes:
            state = ", ".join(f"{k}={v}" for k, v in b.records[:8])
            add(
                f"- **{b.branch_id}** (weight {b.weight:.4f}) conditions={dict(b.key_conditions)}: "
                f"[{state}] -> "
                f"{b.outcome if b.resolved else 'UNRESOLVED: ' + (b.unresolved_reason or '')}"
            )
        add("")

        add("## 11. Aggregate calculation")
        add(
            f"- resolved YES mass: {f.resolved_yes_mass:.4f}\n"
            f"- resolved NO mass: {f.resolved_no_mass:.4f}\n"
            f"- unresolved mass: {f.unresolved_mass:.4f}\n"
            f"- **simulation probability (conditional on resolved):** "
            f"{_fmt(f.simulation_probability)}\n"
            f"- supported lower bound: {_fmt(f.lower_bound)}\n"
            f"- supported upper bound: {_fmt(f.upper_bound)}\n"
            f"- probability source: {f.probability_source}\n"
        )

        add("## 12. Model-call count and token use")
        add(f"- model: {self.model_id}\n- calls: {f.model_call_count}\n- tokens: {f.token_usage}\n")

        if f.diagnostics:
            add("\n## Diagnostic comparator (NOT part of the forecast)")
            for k, v in f.diagnostics:
                add(f"- {k}: {v}")
            add(
                "\n_This reference-class figure is a labeled comparator only. It does not "
                "contribute to the simulation probability above._"
            )

        add("\n## Limitations")
        for lim in f.limitations:
            add(f"- {lim}")

        return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """One canonical-JSON line per row; an empty derivation writes an empty file, so a
    reader can tell 'nothing happened' from 'nothing was persisted'."""

    path.write_text("".join(canonical_json(r) + "\n" for r in rows))


def _invocation_line(d: Any) -> str:
    """One actor invocation, rendered for the report — total over every record shape.

    A no-feasible-action wake records ``intent={}``, and a hard ``c['mode']`` here
    crashed the whole trace write of a COMPLETED direct-mode run 559 seconds in,
    leaving an exit-4 diagnosis beside a finished forecast. A report line must not be
    able to destroy the artifacts it reports on.
    """

    c = d.intent or {}
    what = c.get("action_id") or c.get("novel_description") or ""
    return (
        f"- [{d.branch_id}] {d.branch_time} {d.actor_id} woken by "
        f"*{d.wake_reason}* ({d.wake_detail}); plan: {d.plan_disposition}; "
        f"intent {c.get('mode', 'none')} {what} "
        f"-> {d.validation_status} ({d.validation_reason})"
    )


def _expr_repr(expr: Any) -> str:
    from .worldspec import Expr

    if not isinstance(expr, Expr):
        return repr(expr)
    if expr.op == "const":
        return repr(expr.args[0] if expr.args else None)
    return f"{expr.op}(" + ", ".join(_expr_repr(a) for a in expr.args) + ")"


def _fmt(x: float | None) -> str:
    return "no point estimate (zero resolved mass)" if x is None else f"{x:.4f}"


def _observed_ids(observations: object) -> list[str]:
    """Event ids the actor actually perceived, from its recorded local view."""

    if not isinstance(observations, list):
        return []
    return [str(o.get("obs_id")) for o in observations if isinstance(o, dict)]
