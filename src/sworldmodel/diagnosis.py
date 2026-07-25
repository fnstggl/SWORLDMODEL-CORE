"""A machine-readable account of what one run did, and where it stopped.

Written on success *and* on refusal. That second half is the point: the previous
acceptance run wrote its trace only after a forecast completed, so the four questions
that refused produced no artifact at all — the only record of the most important failures
in the system was a stack trace on stderr, and diagnosing them meant re-running the
pipeline under ad-hoc instrumentation to see what it had already known and thrown away.

The record follows the causal route, so a reader can see the stage at which reality
stopped reaching the simulation:

    planning -> discovery -> fetching -> extraction -> epistemic classification
             -> world compilation -> integrity and grounding -> runtime -> root cause

Nothing here interprets. It reports counts, URLs, reasons and lineage that already exist
in the trace, arranged so the failing stage is visible rather than inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .compiled import CompiledWorld
from .engine import RunResult
from .errors import SWorldModelError
from .repair import RepairLog
from .research import ResearchBundle
from .world_compiler import terminal_producers

# The classification vocabulary. A run's root cause must be expressible in these terms;
# "research recall" is deliberately absent, because it names a symptom and every one of
# these names a mechanism that can be found and fixed.
ROOT_CAUSES = (
    "query_planning_failure",
    "discovery_failure",
    "authoritative_source_starvation",
    "incorrect_nowcast_pastcast_mode",
    "archive_coverage_failure",
    "fetch_failure",
    "document_parsing_failure",
    "claim_extraction_failure",
    "claim_verification_too_weak",
    "claim_verification_too_strict",
    "entity_resolution_failure",
    "actor_discovery_failure",
    "organization_process_discovery_failure",
    "compiler_omission",
    "incorrect_representation_scale",
    "over_strict_grounding_gate",
    "under_strict_integrity_gate",
    "terminal_supplied_rather_than_produced",
    "repeated_wake_up_loop",
    "no_progress_detection_failure",
    "provider_failure",
    "budget_configuration_failure",
    "external_information_unavailable",
)


class ForecastRefused(SWorldModelError):
    """A refusal that carries everything the run had learned when it stopped.

    Raised in place of the bare integrity error so the caller can write a diagnosis
    rather than only a traceback. ``__cause__`` keeps the original.
    """

    def __init__(
        self,
        cause: BaseException,
        *,
        stage: str,
        bundle: ResearchBundle | None = None,
        repair_log: RepairLog | None = None,
    ) -> None:
        self.stage = stage
        self.bundle = bundle
        self.repair_log = repair_log
        super().__init__(f"refused at {stage}: {cause}")


@dataclass
class RunDiagnosis:
    """Assembled from whatever the run produced, however far it got."""

    question: str
    as_of: datetime
    horizon: datetime
    bundle: ResearchBundle | None = None
    compiled: CompiledWorld | None = None
    run_result: RunResult | None = None
    repair_log: RepairLog | None = None
    failure: BaseException | None = None
    failure_stage: str = ""
    wall_seconds: float = 0.0
    model_calls: int = 0
    notes: list[str] = field(default_factory=list)

    # -- sections ------------------------------------------------------------

    def _trace(self) -> dict[str, Any]:
        return dict((self.bundle.live_trace if self.bundle else None) or {})

    def research_planning(self) -> dict[str, Any]:
        t = self._trace()
        plan = dict(t.get("plan") or {})
        return {
            "question": self.question,
            "as_of": self.as_of.isoformat(),
            "horizon": self.horizon.isoformat(),
            "retrieval_mode": t.get("retrieval_mode") or {"mode": "not recorded"},
            "queries_issued": [q.get("query") for q in t.get("queries", [])],
            "query_channels": _counts(q.get("channel", "?") for q in t.get("queries", [])),
            "authoritative_domains_requested": list(t.get("official_domains") or []),
            "missing_facts_the_queries_target": list(t.get("required_facts") or []),
            "resolution_event": t.get("resolution_event"),
            "process_summary": t.get("process_summary"),
            "plan_keys": sorted(plan),
        }

    def discovery(self) -> dict[str, Any]:
        t = self._trace()
        official = list(t.get("official_domains") or [])
        attempted = list(t.get("attempted_urls") or [])
        return {
            "urls_considered": attempted,
            "urls_considered_count": len(attempted),
            "official_domain_urls_found": [
                u for u in attempted if any(d and d in u for d in official)
            ],
            "rss_requests": t.get("rss_requests", []),
            "search_failures": t.get("search_failures", []),
            "queries_used": len(t.get("queries", [])),
            "stop_reason": t.get("stop_reason"),
        }

    def fetching(self) -> dict[str, Any]:
        t = self._trace()
        fetched = list(t.get("sources_fetched") or [])
        rejected = list(t.get("sources_rejected") or [])
        return {
            "fetched": fetched,
            "fetched_count": len(fetched),
            "rejected": rejected,
            "rejected_count": len(rejected),
            "rejection_reasons": _counts(_reason(r) for r in rejected),
            "archived_captures_used": sum(1 for f in fetched if f.get("archived_at")),
        }

    def extraction(self) -> dict[str, Any]:
        t = self._trace()
        calls = list(t.get("extraction_calls") or [])
        rejected = [
            r for r in (t.get("sources_rejected") or []) if "claim not verified" in _reason(r)
        ]
        store_claims = len(self.bundle.evidence_store.claims) if self.bundle else 0
        return {
            "extraction_calls": len(calls),
            "calls_returning_nothing": sum(1 for c in calls if not c.get("claims_returned")),
            "documents_truncated": sum(1 for c in calls if c.get("document_truncated")),
            "claim_candidates": sum(int(c.get("claims_returned") or 0) for c in calls),
            "claims_stored": store_claims,
            "claims_admissible_at_cutoff": int(t.get("admissible_claim_count") or 0),
            "verification_rejections": [_reason(r) for r in rejected],
            "verification_rejection_reasons": _counts(
                _verification_kind(_reason(r)) for r in rejected
            ),
            "contradictions": list(t.get("contradictions") or []),
        }

    def epistemic_classification(self) -> dict[str, Any]:
        """Every stored claim with the class it carries and what supports it."""

        if self.bundle is None:
            return {"claims": [], "note": "research did not complete"}
        claims = []
        for claim in self.bundle.evidence_store.all():
            claims.append(
                {
                    "id": claim.id,
                    "proposition": claim.proposition,
                    "normalized_value": claim.normalized_value,
                    "class": _claim_class(claim.epistemic_type.value, claim.supporting_excerpt),
                    "epistemic_type": claim.epistemic_type.value,
                    "why": (
                        "excerpt verified against the fetched document"
                        if claim.supporting_excerpt
                        else "stored without a supporting excerpt"
                    ),
                    "source_url": claim.source_url,
                    "authority": int(claim.authority_level),
                    "available_at": claim.available_at.isoformat(),
                }
            )
        return {
            "claims": claims,
            "counts": _counts(c["class"] for c in claims),
        }

    def world_compilation(self) -> dict[str, Any]:
        spec = self.compiled.spec if self.compiled else (self.bundle.spec if self.bundle else None)
        if spec is None:
            return {"compiled": False, "reason": "no world was compiled"}
        actor_ids = {a.entity_id for a in spec.actors}
        producers = terminal_producers(spec)
        return {
            "compiled": True,
            "title": spec.title,
            "structure_rationale": spec.structure_rationale,
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "kind": e.kind,
                    "representation_scale": e.representation_scale,
                    "represents_count": e.represents_count,
                    "is_actor": e.entity_id in actor_ids or e.is_actor,
                    "role": e.role,
                    "authority": list(e.authority),
                    "cited_claims": list(e.evidence_claim_ids),
                }
                for e in spec.entities
            ],
            "causal_producers": {
                "actors": sorted(actor_ids),
                "external_processes": [p.process_id for p in spec.external_processes],
                "process_nodes": [n.node_id for n in spec.process.nodes],
            },
            "actions": [
                {
                    "action_id": a.action_id,
                    "meaning": a.meaning,
                    "eligible": list(a.eligible_actors),
                    "required_authority": list(a.required_authority),
                }
                for a in spec.actions
            ],
            "fields": [f.field_id for f in spec.fields],
            "terminal_description": spec.terminal.description,
            "terminal_producers": {k: list(v) for k, v in sorted(producers.items())},
            "terminal_terms_without_a_producer": sorted(k for k, v in producers.items() if not v),
        }

    def integrity_and_grounding(self) -> dict[str, Any]:
        out: dict[str, Any] = {"stopped_at_gate": None, "repair_attempts": []}
        if self.repair_log is not None:
            out["repair_attempts"] = self.repair_log.as_list()
        if isinstance(self.failure, SWorldModelError):
            details = getattr(self.failure, "details", {}) or {}
            out["stopped_at_gate"] = details.get("failure") or type(self.failure).__name__
            out["gate_message"] = str(self.failure).splitlines()[0]
            out["gate_details"] = {k: v for k, v in details.items() if k != "failure"}
        if self.compiled is not None:
            m = self.compiled.manifest
            out["manifest"] = {
                "verdict": m.integrity_verdict.value,
                "expected_participants": m.expected_participants,
                "represented_participants": m.represented_participants,
                "verified_entities": list(m.verified_entities),
                "missing_required_facts": list(m.missing_required_facts),
                "unresolved_conflicts": list(m.unresolved_conflicts),
                "evidence_coverage": m.evidence_coverage,
                "notes": list(m.notes),
            }
            out["actor_grounding"] = self.compiled.actor_grounding.as_dict()
            out["coverage_complete"] = self.compiled.coverage_report.is_complete
        return out

    def runtime(self) -> dict[str, Any]:
        if self.run_result is None:
            return {"ran": False, "reason": self.failure_stage or "the world never compiled"}
        r = self.run_result
        return {
            "ran": True,
            "event_count": len(r.event_ledger),
            "actor_invocations": len(r.actor_decisions),
            "wake_reasons": _counts(d.wake_reason for d in r.actor_decisions),
            "validation_outcomes": _counts(d.validation_status for d in r.actor_decisions),
            "branches": len(r.branch_outcomes),
            "resolved_branches": sum(1 for b in r.branch_outcomes if b.resolved),
            "unresolved_mass": round(sum(b.weight for b in r.branch_outcomes if not b.resolved), 6),
            "truncated_mass": r.truncated_mass,
            "truncated_reason": r.truncated_reason,
            "per_branch": {
                bid: {
                    "stop_reason": d.stop_reason,
                    "batches": getattr(d, "batches", None),
                    "unfired_in_horizon": getattr(d, "unfired_in_horizon", None),
                    "pending_beyond_horizon": len(d.pending_beyond_horizon),
                }
                for bid, d in r.diagnostics.items()
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "as_of": self.as_of.isoformat(),
            "horizon": self.horizon.isoformat(),
            "outcome": "refused" if self.failure is not None else "completed",
            "failure_stage": self.failure_stage,
            "failure": str(self.failure).splitlines()[0] if self.failure else "",
            "wall_seconds": round(self.wall_seconds, 1),
            "model_calls": self.model_calls,
            "research_planning": self.research_planning(),
            "discovery": self.discovery(),
            "fetching": self.fetching(),
            "extraction": self.extraction(),
            "epistemic_classification": self.epistemic_classification(),
            "world_compilation": self.world_compilation(),
            "integrity_and_grounding": self.integrity_and_grounding(),
            "runtime": self.runtime(),
            "root_cause": self.root_cause(),
            "root_cause_vocabulary": list(ROOT_CAUSES),
            "notes": self.notes,
        }

    # -- classification ------------------------------------------------------

    def root_cause(self) -> list[dict[str, str]]:
        """Name the mechanism, from the evidence in this run's own record.

        Deliberately mechanical. It reads counts the run produced and points at the
        stage where reality stopped propagating; a human still judges, but never has to
        start from "it refused".
        """

        out: list[dict[str, str]] = []
        ext = self.extraction()
        fetch = self.fetching()
        disc = self.discovery()
        gate = self.integrity_and_grounding().get("stopped_at_gate")

        if ext["claims_stored"] == 0 and ext["claim_candidates"] > 0:
            out.append(
                {
                    "cause": "claim_verification_too_strict",
                    "why": f"{ext['claim_candidates']} candidate claim(s) extracted, none stored: "
                    f"{ext['verification_rejection_reasons']}",
                }
            )
        if ext["extraction_calls"] and ext["calls_returning_nothing"] == ext["extraction_calls"]:
            out.append(
                {
                    "cause": "document_parsing_failure",
                    "why": f"all {ext['extraction_calls']} extraction call(s) returned nothing — "
                    "the window the model was shown did not contain the document",
                }
            )
        if fetch["fetched_count"] == 0 and fetch["rejected_count"] > 0:
            out.append(
                {
                    "cause": "fetch_failure",
                    "why": f"{fetch['rejected_count']} source(s) rejected, none fetched: "
                    f"{fetch['rejection_reasons']}",
                }
            )
        if disc["urls_considered_count"] == 0:
            out.append(
                {"cause": "discovery_failure", "why": "no candidate URL was discovered at all"}
            )
        elif not disc["official_domain_urls_found"] and disc["urls_considered_count"] > 0:
            out.append(
                {
                    "cause": "authoritative_source_starvation",
                    "why": "no URL on any requested authoritative domain was reached",
                }
            )
        if gate in ("no_causal_producer", "actors_ungrounded"):
            out.append(
                {
                    "cause": "actor_discovery_failure"
                    if gate == "no_causal_producer"
                    else "over_strict_grounding_gate",
                    "why": f"the run stopped at the {gate} gate",
                }
            )
        if gate in (
            "terminal_has_no_producer",
            "actors_cannot_reach_terminal",
            "environment_presets_terminal",
        ):
            out.append(
                {
                    "cause": "terminal_supplied_rather_than_produced",
                    "why": f"the run stopped at the {gate} gate",
                }
            )
        if gate == "coverage_incomplete":
            out.append(
                {"cause": "compiler_omission", "why": "the run stopped at the coverage gate"}
            )
        rt = self.runtime()
        if rt.get("ran") and self.failure_stage == "simulation":
            out.append({"cause": "repeated_wake_up_loop", "why": "the event loop did not settle"})
        if not out:
            out.append(
                {
                    "cause": "unclassified" if self.failure is not None else "none",
                    "why": (
                        f"the run stopped at {gate or self.failure_stage!r} and no rule "
                        "here recognised the mechanism — classify it by hand rather than "
                        "reading this as a clean run"
                        if self.failure is not None
                        else "the run completed without stopping at a stage"
                    ),
                }
            )
        return out


# ---------------------------------------------------------------------------


def _counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _reason(record: dict[str, Any]) -> str:
    return str(record.get("reason") or "")


def _verification_kind(reason: str) -> str:
    """Group a rejection by the rule that produced it, so a systematic rule that is
    discarding real evidence stands out from one-off bad claims."""

    low = reason.lower()
    for marker, label in (
        ("does not appear verbatim", "excerpt not in document"),
        ("date(s) the claim asserts", "asserted date unsupported"),
        ("value(s) the claim asserts", "asserted value unsupported"),
        ("mention the claim's subject", "subject not in span"),
        ("never mentions", "proper noun not in document"),
        ("no supporting excerpt", "no excerpt supplied"),
        ("names no subject", "no subject"),
        ("states no proposition", "no proposition"),
    ):
        if marker in low:
            return label
    return "other"


def _claim_class(epistemic_type: str, excerpt: str) -> str:
    """The four-class label for a stored claim.

    A stored claim always carries a verified excerpt, so an observation is VERIFIED; a
    model-labeled inference stays INFERRED however well cited; a hypothesis is
    HYPOTHETICAL. Nothing UNSUPPORTED reaches the store — that is what the store is for.
    """

    if epistemic_type == "observation":
        return "verified" if excerpt else "inferred"
    if epistemic_type == "inference":
        return "inferred"
    return "hypothetical"
