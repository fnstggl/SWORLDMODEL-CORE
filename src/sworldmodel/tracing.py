"""Tracing and the under-the-hood report.

Writes the replayable event ledger, the LLM-call log, actor-decision records, the
evidence and world manifests, and the human-readable report. The report shows enough
to reconstruct the probability by hand: every branch, every vote, the deterministic
tally, and the aggregate arithmetic. Serialization is deterministic so artifacts are
hashable and comparable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .compiler import CompiledWorld
from .evidence import EvidenceStore
from .ids import canonical_json, sha256_hex
from .models import ForecastResult, ResolutionContract
from .research import ResearchBundle
from .runtime import RunResult


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
            "expected_voting_seats": f.integrity_manifest.expected_voting_seats,
            "represented_voting_seats": f.integrity_manifest.represented_voting_seats,
            "model": self.model_id,
            "model_call_count": f.model_call_count,
            "token_usage": f.token_usage,
            "branches": [
                {
                    "branch_id": b.branch_id,
                    "weight": b.weight,
                    "resolved": b.resolved,
                    "outcome": b.outcome,
                    "unresolved_reason": b.unresolved_reason,
                    "conditions": dict(b.key_conditions),
                    "votes": dict(b.votes),
                    "tally": dict(b.final_tally),
                }
                for b in f.branch_outcomes
            ],
        }

    def evidence_manifest(self) -> dict[str, Any]:
        claims = self.evidence_store.all()
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
                }
                for c in claims
            ],
        }

    def world_manifest(self) -> dict[str, Any]:
        m = self.compiled.manifest
        return {
            "integrity_verdict": m.integrity_verdict.value,
            "expected_voting_seats": m.expected_voting_seats,
            "represented_voting_seats": m.represented_voting_seats,
            "verified_roles": [list(x) for x in m.verified_roles],
            "verified_authorities": [[a, list(p)] for a, p in m.verified_authorities],
            "verified_rules": list(m.verified_rules),
            "verified_previous_actions": list(m.verified_previous_actions),
            "verified_memberships": [[i, list(ms)] for i, ms in m.verified_memberships],
            "missing_required_facts": list(m.missing_required_facts),
            "unresolved_conflicts": list(m.unresolved_conflicts),
            "evidence_coverage": m.evidence_coverage,
            "causal_graph": {
                "nodes": list(self.compiled.causal_graph.nodes),
                "edges": [
                    {"cause": e.cause, "effect": e.effect, "mechanism": e.mechanism}
                    for e in self.compiled.causal_graph.edges
                ],
            },
            "protocol": list(self.compiled.protocol.kinds()),
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
        # We record real prompts/outputs only; no fabricated chain-of-thought.
        lines = []
        for r in self._gateway_calls():
            lines.append(
                canonical_json(
                    {
                        "task_kind": r.task_kind,
                        "model": r.model,
                        "seed": r.seed,
                        "prompt_hash": r.prompt_hash,
                        "tokens_in": r.tokens_in,
                        "tokens_out": r.tokens_out,
                        "retries": r.retries,
                        "response": r.data,
                    }
                )
            )
        return lines

    def actor_decision_lines(self) -> list[str]:
        return [
            canonical_json(
                {
                    "branch_id": d.branch_id,
                    "actor_id": d.actor_id,
                    "stage": d.stage,
                    "retrieved_memory_ids": d.retrieved_memory_ids,
                    "local_view": d.decision_context,
                    "intent": d.intent,
                    "validation": d.validation,
                    "applied_event_ids": d.event_ids,
                    "prompt_hash": d.prompt_hash,
                    "model": d.model,
                }
            )
            for d in self.run_result.actor_decisions
        ]

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
        (out_dir / "event_ledger.jsonl").write_text("\n".join(self.event_ledger_lines()) + "\n")
        (out_dir / "llm_calls.jsonl").write_text("\n".join(self.llm_call_lines()) + "\n")
        (out_dir / "actor_decisions.jsonl").write_text(
            "\n".join(self.actor_decision_lines()) + "\n"
        )
        (out_dir / report_name).write_text(self.render_report(forecast_hash))
        return forecast_hash

    # -- report -----------------------------------------------------------------

    def render_report(self, forecast_hash: str) -> str:
        f = self.forecast
        m = self.compiled.manifest
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
            f"- decision body: {f.contract.decision_body}\n"
            f"- subject entity: {f.contract.subject_entity}\n"
            f"- resolution units: {f.contract.resolution_units}\n"
            f"- outcome space: {list(f.contract.outcome_space)}\n"
            f"- target: {f.contract.target_outcome}\n"
            f"- terminal predicate: {f.contract.terminal_predicate.yes_condition} "
            f"for option {f.contract.terminal_predicate.target_option!r}\n"
            f"- decision rule: {f.contract.decision_rule.kind} "
            f"{f.contract.decision_rule.threshold}/{f.contract.decision_rule.total_seats}\n"
            f"- expected voting seats: {f.contract.expected_voting_seats}\n"
        )

        add("## 2-4. Evidence, contradictions, lineage")
        em = self.evidence_manifest()
        add(
            f"- claims: {em['claim_count']} across {em['independent_event_count']} "
            f"independent events (lineage-deduplicated)\n"
            f"- contradictions: {em['contradictions'] or 'none'}\n"
            f"- research plan (backward): {' '.join(self.bundle.research_plan)}\n"
        )

        add("## 5. Verified roster and voting rule")
        for aid, role in m.verified_roles:
            voting = aid in self.compiled.base_world.voting_actor_ids()
            add(f"- {aid} — {role}{' (voting seat)' if voting else ''}")
        add(
            f"\n- verified rule: {', '.join(m.verified_rules)}  \n"
            f"- expected seats: {m.expected_voting_seats}, represented: "
            f"{m.represented_voting_seats}, verdict: {m.integrity_verdict.value}\n"
        )

        add("## 6. Causal graph")
        for e in self.compiled.causal_graph.edges:
            add(f"- {e.cause} --{e.mechanism}--> {e.effect}")
        add("")

        add("## 7. External uncertain events and branch weights")
        for u in self.compiled.uncertainty_variables:
            add(f"- **{u.variable_id}** ({u.why_unknown}); reversal-capable={u.reversal_capable}")
            for o in u.outcomes:
                add(f"    - {o.value}: weight {o.weight.value} [{o.weight.provenance.value}]")
        add("")

        add("## 10. Protocol graph")
        add(f"- {' -> '.join(self.compiled.protocol.kinds())}\n")

        add("## 11-20. Actor decisions (local view -> intent -> events)")
        for d in self.run_result.actor_decisions:
            intent = d.intent
            add(
                f"- [{d.branch_id}] {d.actor_id} @ {d.stage}: intent={intent['kind']} "
                f"{intent['payload']} — {intent['rationale']}"
            )
            add(f"    retrieved memories: {d.retrieved_memory_ids}")
            add(f"    referenced observations: {intent['referenced_observation_ids']}")
        add("")

        add("## 19-23. Branches, votes, deterministic tally, outcomes")
        for b in f.branch_outcomes:
            add(
                f"- **{b.branch_id}** (weight {b.weight:.4f}) conditions={dict(b.key_conditions)}: "
                f"votes={dict(b.votes)} tally={dict(b.final_tally)} -> "
                f"{b.outcome if b.resolved else 'UNRESOLVED: ' + (b.unresolved_reason or '')}"
            )
        add("")

        add("## 24. Aggregate calculation")
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

        add("## 25. Model-call count and token use")
        add(f"- model: {self.model_id}\n- calls: {f.model_call_count}\n- tokens: {f.token_usage}\n")

        add("## 26. Why the forecast sits where it does")
        add(_explanation(self))

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


def _fmt(x: float | None) -> str:
    return "no point estimate (zero resolved mass)" if x is None else f"{x:.4f}"


def _explanation(ctx: TraceContext) -> str:
    f = ctx.forecast
    guidance = ctx.compiled.frame.guidance_option
    n_yes = sum(1 for b in f.branch_outcomes if b.resolved and b.outcome == "YES")
    n_total = sum(1 for b in f.branch_outcomes if b.resolved)
    return (
        f"The initial common position was {guidance!r}. In {n_yes} of {n_total} resolved "
        f"branches every seat's independently-reasoned final vote coincided with the "
        f"target predicate; in the rest, an uncertain future signal crossed a member's "
        f"reaction threshold and moved a vote, breaking the target. The probability is the "
        f"weighted share of branches whose actual simulated votes met the predicate — not a "
        f"prior and not a separate institution model."
    )
