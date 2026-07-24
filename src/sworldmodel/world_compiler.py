"""The world compiler: turn a compiled :class:`WorldSpec` into a verified, runnable
:class:`CompiledWorld`, and (live) drive the LLM that authors that WorldSpec.

Two entry points share the same output:

* :func:`compile_world` — build the base world from an already-parsed WorldSpec (from an
  authored corpus offline, or from the live LLM), run the reality-integrity gate, and
  enumerate genuine-uncertainty branches.
* :func:`compile_world_spec_live` — ask the model to compile the entire world (entities,
  actors, actions, process graph, declarative terminal, uncertainties) from verified
  evidence for an arbitrary question, then normalize/citation-check it.

Nothing here is question-specific. A committee, a negotiation, a population process and
a geopolitical process differ only in the *data* the compiler emits — never in code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .actors import ActorState
from .compiled import CompiledWorld
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .models import (
    BranchWeight,
    EpistemicType,
    RequiredRealityFact,
    ResolutionContract,
    UncertaintyOutcome,
    UncertaintySpec,
    UncertaintyVariable,
    WeightProvenance,
)
from .prompts import render_world_compile_prompt
from .reality import verify_reality
from .uncertainty import enumerate_scenarios
from .world import WorldFact, WorldState
from .worldspec import ActorSpec, EntitySpec, WorldSpec

_VALID_PROVENANCE = {p.value for p in WeightProvenance}


# ---------------------------------------------------------------------------
# Offline / shared compilation from a parsed WorldSpec
# ---------------------------------------------------------------------------


def build_base_world(
    spec: WorldSpec,
    contract: ResolutionContract,
    evidence: EvidenceView,
    world_facts: tuple[WorldFact, ...],
) -> WorldState:
    entities_by_id = {e.entity_id: e for e in spec.entities}
    actor_states: dict[str, ActorState] = {}
    for aspec in spec.actors:
        entity = entities_by_id.get(aspec.entity_id) or _default_actor_entity(aspec)
        actor_states[aspec.entity_id] = ActorState.from_spec(
            entity, aspec, default_time=contract.as_of
        )

    fields = {f.field_id: f.initial for f in spec.fields if f.initial is not None}
    resources = {f"{r.resource_id}@{r.holder_entity_id}": r.quantity for r in spec.resources}
    documents = {d.document_id: dict(d.fields) for d in spec.documents}

    return WorldState(
        branch_id="root",
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.DIRECT_EMPIRICAL, "verified initial world"),
        time=contract.as_of,
        contract=contract,
        evidence=evidence,
        entities=spec.entities,
        actors=actor_states,
        verified_facts=world_facts,
        fields=tuple(sorted(fields.items())),
        resources=tuple(sorted(resources.items())),
        documents=tuple((k, tuple(sorted(v.items()))) for k, v in sorted(documents.items())),
        stage="initial",
    )


def compile_world(
    contract: ResolutionContract,
    evidence: EvidenceView,
    spec: WorldSpec,
    uncertainties: tuple[UncertaintySpec, ...],
    world_facts: tuple[WorldFact, ...],
    *,
    seed: int = 0,
    max_branches: int = 24,
    compile_responses: tuple[Any, ...] = (),
) -> CompiledWorld:
    base_world = build_base_world(spec, contract, evidence, world_facts)

    # Reality-integrity gate — raises if the world is not faithful.
    manifest = verify_reality(contract, evidence, base_world.actors)

    scenario_set = enumerate_scenarios(
        uncertainties, base_world.fields_dict(), max_branches=max_branches
    )
    uvars = _uncertainty_variables(uncertainties)
    return CompiledWorld(
        base_world=base_world,
        spec=spec,
        scenario_set=scenario_set,
        manifest=manifest,
        uncertainty_variables=uvars,
        compile_responses=tuple(compile_responses),
    )


def _default_actor_entity(aspec: ActorSpec) -> EntitySpec:
    return EntitySpec(entity_id=aspec.entity_id, name=aspec.entity_id, kind="person", is_actor=True)


def _uncertainty_variables(
    uncertainties: tuple[UncertaintySpec, ...],
) -> tuple[UncertaintyVariable, ...]:
    out: list[UncertaintyVariable] = []
    for spec in uncertainties:
        out.append(
            UncertaintyVariable(
                variable_id=spec.variable,
                description=f"Future value of {spec.variable}",
                why_unknown=spec.why_unknown,
                how_it_affects_outcome=(
                    "Can cross an actor reaction threshold or a terminal predicate and change the outcome."
                ),
                constraining_evidence_ids=spec.constraining_evidence_ids,
                reversal_capable=spec.reversal_capable,
                outcomes=spec.outcomes,
                behavioral_equivalence_note=(
                    "Outcomes are behaviorally equivalent when none crosses any reaction threshold."
                ),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Parsers for the compilation-input pieces carried alongside the WorldSpec
# ---------------------------------------------------------------------------


def parse_uncertainties(
    items: Any, available_ids: set[str] | None = None
) -> tuple[UncertaintySpec, ...]:
    out: list[UncertaintySpec] = []
    for u in items or []:
        outcomes = u.get("outcomes") or []
        total = sum(float(o.get("weight", 0)) for o in outcomes)
        if total <= 0 or not outcomes:
            continue
        parsed_outcomes: list[UncertaintyOutcome] = []
        for o in outcomes:
            prov = o.get("provenance")
            provenance = (
                WeightProvenance(prov)
                if prov in _VALID_PROVENANCE
                else WeightProvenance.EXPLICIT_MODEL
            )
            parsed_outcomes.append(
                UncertaintyOutcome(
                    value=str(o.get("value", "")),
                    weight=BranchWeight(
                        value=float(o.get("weight", 0)) / total,
                        provenance=provenance,
                        source_detail=str(o.get("source_detail", o.get("description", ""))),
                    ),
                    field_effects=tuple((str(k), v) for k, v in (o.get("field_effects") or [])),
                    description=str(o.get("description", "")),
                )
            )
        constraining = tuple(u.get("constraining_evidence_ids", []) or [])
        if available_ids is not None:
            constraining = tuple(c for c in constraining if c in available_ids)
        out.append(
            UncertaintySpec(
                variable=str(u.get("variable", "")),
                why_unknown=str(u.get("why_unknown", "")),
                reversal_capable=bool(u.get("reversal_capable", True)),
                outcomes=tuple(parsed_outcomes),
                constraining_evidence_ids=constraining,
            )
        )
    return tuple(out)


def parse_world_facts(items: Any, default_time: datetime) -> tuple[WorldFact, ...]:
    facts: list[WorldFact] = []
    for i, wf in enumerate(items or []):
        at = wf.get("available_at")
        facts.append(
            WorldFact(
                fact_id=f"fact_{i}",
                text=str(wf.get("text", "")),
                evidence_claim_ids=tuple(wf.get("evidence_claim_ids", []) or []),
                available_at=datetime.fromisoformat(at) if isinstance(at, str) else default_time,
                epistemic_type=EpistemicType(wf.get("epistemic_type", "observation")),
            )
        )
    return tuple(facts)


def parse_required_facts(items: Any) -> tuple[RequiredRealityFact, ...]:
    return tuple(
        RequiredRealityFact(
            key=str(rf.get("key", "")),
            description=str(rf.get("description", "")),
            evidence_claim_ids=tuple(rf.get("evidence_claim_ids", []) or []),
        )
        for rf in (items or [])
    )


# ---------------------------------------------------------------------------
# Live LLM compilation of the WorldSpec from evidence
# ---------------------------------------------------------------------------


def _render_evidence(view: EvidenceView, *, limit: int = 160) -> str:
    claims = sorted(view.available(), key=lambda c: (-int(c.authority_level), c.id))[:limit]
    return "\n".join(
        f"{c.id} | {c.proposition} = {c.normalized_value} "
        f"[auth {int(c.authority_level)}, {c.source_type.value}, {c.published_at.date()}]"
        for c in claims
    )


def compile_world_spec_live(
    gateway: ModelGateway, question: str, as_of: datetime, horizon: datetime, view: EvidenceView
) -> tuple[dict[str, Any], Any]:
    """Ask the model to compile the whole causal world for an arbitrary question, then
    normalize and citation-check it. Returns ``(compilation_dict, gateway_response)``
    where the dict has keys ``world_spec``, ``uncertainties``, ``world_facts``,
    ``required_reality_facts`` and reality metadata."""

    ctx = {
        "question": question,
        "as_of": as_of.isoformat(),
        "horizon": horizon.isoformat(),
        "evidence": _render_evidence(view),
    }
    resp = gateway.generate(
        GatewayRequest(
            task_kind="compile_world_spec",
            prompt=render_world_compile_prompt(ctx),
            context={"question": question},
            seed=int(prompt_hash("world" + question)[:8], 16),
            expected_keys=("world_spec",),
        )
    )
    return _normalize_compilation(resp.data, view, as_of, horizon), resp


def _normalize_compilation(
    data: dict[str, Any], view: EvidenceView, as_of: datetime, horizon: datetime
) -> dict[str, Any]:
    available = {c.id for c in view.available()}
    ws = data.get("world_spec") or {}

    def _filter(node: Any) -> None:
        if isinstance(node, dict):
            if "evidence_claim_ids" in node and isinstance(node["evidence_claim_ids"], list):
                node["evidence_claim_ids"] = [
                    str(i) for i in node["evidence_claim_ids"] if str(i) in available
                ]
            for v in node.values():
                _filter(v)
        elif isinstance(node, list):
            for v in node:
                _filter(v)

    _filter(ws)
    data["world_spec"] = ws
    data["subject_entity"] = _s(data.get("subject_entity")) or _s(ws.get("title")) or "the subject"
    data["resolution_units"] = _s(data.get("resolution_units")) or "the outcome"
    data["target_outcome"] = _s(data.get("target_outcome")) or "the YES condition"
    data["as_of"] = as_of.isoformat()
    data["horizon"] = horizon.isoformat()
    return data


def _s(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
