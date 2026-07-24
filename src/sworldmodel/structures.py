"""Structural uncertainty: which causal world is even the right one.

Every other uncertainty in this system is *inside* a world — what a value turns out to
be, what someone decides. This module handles the uncertainty one level up: whether the
causal structure we compiled is the structure that actually produces the outcome.

That is a real and frequently decisive unknown. The evidence may not settle whether a
body decides as a unit or through factions, whether the binding path is a formal
procedure or an informal one, whether the outcome is driven by the people we found or by
an external process nobody named. A run that compiles one confident world and reports a
point estimate has quietly asserted an answer to that question.

So: after the primary world is compiled, the model is asked whether a *materially
different* structure could produce a different answer on the same evidence. If it says
yes, each alternative is compiled into a full :class:`~sworldmodel.worldspec.WorldSpec`
and simulated exactly like the primary one — same runtime, same gates, same terminal
evaluator — and the branch masses are combined.

Two honesty rules govern this:

* An alternative that cannot be compiled, or that fails the integrity gates, is **not**
  dropped. Its mass becomes explicitly unresolved and widens the reported bounds,
  because "we could not represent this possibility" is not the same as "this
  possibility does not exist".
* Structural weights are epistemic and are labeled as such. When the model cannot
  defend a split, the weakest provenance applies and the bounds carry the doubt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .evidence import EvidenceView
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .models import ResolutionContract, WeightProvenance
from .worldspec import WorldSpec


@dataclass(frozen=True)
class StructuralAlternative:
    """One materially different causal structure the evidence leaves open."""

    structure_id: str
    what_differs: str
    rationale: str
    weight: float
    provenance: WeightProvenance
    could_reverse_outcome: bool
    evidence_claim_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "what_differs": self.what_differs,
            "rationale": self.rationale,
            "weight": self.weight,
            "provenance": self.provenance.value,
            "could_reverse_outcome": self.could_reverse_outcome,
            "evidence_claim_ids": list(self.evidence_claim_ids),
        }


@dataclass(frozen=True)
class StructuralAssessment:
    """What the compiler concluded about its own structural choice."""

    is_material: bool
    reason: str
    primary_weight: float
    alternatives: tuple[StructuralAlternative, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_material": self.is_material,
            "reason": self.reason,
            "primary_weight": self.primary_weight,
            "alternatives": [a.as_dict() for a in self.alternatives],
        }

    @classmethod
    def certain(cls, reason: str) -> StructuralAssessment:
        return cls(is_material=False, reason=reason, primary_weight=1.0, alternatives=())


def render_structure_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "You compiled a causal world to resolve a question. Before it is simulated,",
            "state honestly whether it is the RIGHT world.",
            f"QUESTION: {context.get('question')}",
            f"as_of: {context.get('as_of')}   horizon: {context.get('horizon')}",
            "## THE STRUCTURE YOU COMPILED\n" + str(context.get("primary_summary", "")),
            "## EVIDENCE\n" + str(context.get("evidence", "")),
            "Is there a MATERIALLY DIFFERENT causal structure, consistent with this same",
            "evidence, that could produce a different answer? Materially different means a",
            "different answer to one of: who actually decides; whether a body acts as one",
            "unit or through internal factions; whether the binding pathway is this formal",
            "process or a different (possibly informal) one; whether an external process",
            "rather than anyone's decision determines the outcome; or what level of thing",
            "the deciding entity is.",
            "A different *wording* of the same structure is not material. Neither is a",
            "different guess about a value inside this structure — that is ordinary",
            "uncertainty and belongs elsewhere.",
            "If the evidence clearly settles the structure, say so and set is_material",
            "false. Do not manufacture doubt; overstating structural uncertainty is as",
            "dishonest as hiding it.",
            'Reply with JSON: {"is_material": true|false, "reason": "",'
            ' "primary_weight": 0.0-1.0,'
            ' "alternatives": [{"structure_id": "snake_case", "what_differs": "the exact'
            ' causal difference, concretely enough to compile a world from",'
            ' "rationale": "why the evidence leaves this open",'
            ' "weight": 0.0-1.0, "could_reverse_outcome": true|false,'
            ' "provenance": "symmetric_ignorance_assumption|explicit_model_distribution|'
            "direct_empirical_distribution|market_or_survey_distribution|"
            'calibrated_behavior_model|sensitivity_only_branch",'
            ' "evidence_claim_ids": []}]}.'
            " Weights must sum with primary_weight to 1.0.",
        ]
    )


def assess_structure(
    contract: ResolutionContract,
    evidence: EvidenceView,
    spec: WorldSpec,
    *,
    gateway: ModelGateway,
    seed: int,
    max_alternatives: int = 2,
    evidence_render: str = "",
) -> tuple[StructuralAssessment, GatewayResponse | None]:
    """Ask whether the compiled structure is the one that decides this question."""

    context = {
        "question": contract.question,
        "as_of": contract.as_of.isoformat(),
        "horizon": contract.horizon.isoformat(),
        "primary_summary": summarize_structure(spec),
        "evidence": evidence_render,
    }
    resp = gateway.generate(
        GatewayRequest(
            task_kind="assess_structure",
            prompt=render_structure_prompt(context),
            context=context,
            seed=seed,
            expected_keys=("is_material",),
        )
    )
    data = resp.data
    if not bool(data.get("is_material")):
        return (
            StructuralAssessment.certain(
                str(data.get("reason", "the evidence determines the causal structure"))
            ),
            resp,
        )

    alternatives: list[StructuralAlternative] = []
    for raw in list(data.get("alternatives") or [])[:max_alternatives]:
        if not isinstance(raw, dict):
            continue
        differs = str(raw.get("what_differs", "")).strip()
        if not differs:
            continue  # an alternative we cannot compile from is not an alternative
        prov = str(raw.get("provenance", ""))
        alternatives.append(
            StructuralAlternative(
                structure_id=str(raw.get("structure_id", f"alt_{len(alternatives)}")),
                what_differs=differs,
                rationale=str(raw.get("rationale", "")),
                weight=max(0.0, float(raw.get("weight", 0.0) or 0.0)),
                provenance=(
                    WeightProvenance(prov)
                    if prov in {p.value for p in WeightProvenance}
                    # An unlabeled structural weight is the least defensible kind there
                    # is: it is a guess about which world we are in.
                    else WeightProvenance.SYMMETRIC_IGNORANCE
                ),
                could_reverse_outcome=bool(raw.get("could_reverse_outcome", True)),
                evidence_claim_ids=tuple(str(c) for c in (raw.get("evidence_claim_ids") or [])),
            )
        )
    alternatives = [a for a in alternatives if a.weight > 0]
    if not alternatives:
        return (
            StructuralAssessment.certain(
                "structural uncertainty was claimed but no compilable alternative was given"
            ),
            resp,
        )

    primary = float(data.get("primary_weight", 0.0) or 0.0)
    total = primary + sum(a.weight for a in alternatives)
    if total <= 0:
        return StructuralAssessment.certain("structural weights were not usable"), resp
    return (
        StructuralAssessment(
            is_material=True,
            reason=str(data.get("reason", "")),
            primary_weight=primary / total,
            alternatives=tuple(
                StructuralAlternative(
                    structure_id=a.structure_id,
                    what_differs=a.what_differs,
                    rationale=a.rationale,
                    weight=a.weight / total,
                    provenance=a.provenance,
                    could_reverse_outcome=a.could_reverse_outcome,
                    evidence_claim_ids=a.evidence_claim_ids,
                )
                for a in alternatives
            ),
        ),
        resp,
    )


def summarize_structure(spec: WorldSpec) -> str:
    """A compact description of the compiled causal structure — enough to judge it
    against, without reproducing the whole program."""

    actors = [
        f"{e.entity_id} ({e.role or e.kind}, scale={e.representation_scale}"
        + (f", represents {e.represents_count}" if e.represents_count else "")
        + ")"
        for e in spec.entities
        if e.is_actor
    ]
    return json.dumps(
        {
            "title": spec.title,
            "deciding_actors": actors,
            "other_entities": [e.entity_id for e in spec.entities if not e.is_actor],
            "actions": [f"{a.action_id}: {a.meaning}" for a in spec.actions],
            "process": [
                f"{n.node_id}@{n.at or ('after ' + n.after_node if n.after_node else 'start')}"
                f" [{n.stage}] participants={list(n.participants)}"
                for n in spec.process.nodes
            ],
            "external_processes": [p.process_id for p in spec.external_processes],
            "terminal": spec.terminal.description,
        },
        indent=1,
        sort_keys=True,
        default=str,
    )


def alternative_compile_instruction(alt: StructuralAlternative) -> str:
    """The extra instruction that makes the compiler build *this* structure instead."""

    return (
        "COMPILE AN ALTERNATIVE CAUSAL STRUCTURE.\n"
        "A previous compilation of this same question produced a different structure. "
        "You are now compiling the alternative described below, which the evidence also "
        "leaves open. Build it as a complete, self-consistent world in its own right — "
        "do not hedge between the two, and do not reproduce the other structure.\n"
        f"THE DIFFERENCE: {alt.what_differs}\n"
        f"WHY THE EVIDENCE LEAVES IT OPEN: {alt.rationale}\n"
        "If this structure cannot in fact be grounded in the evidence, say so by "
        "producing no actors rather than inventing support for it."
    )
