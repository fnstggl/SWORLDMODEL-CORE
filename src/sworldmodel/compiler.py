"""The general world compiler.

The compiler builds a scenario from verified evidence, but it may not invent the
world freely: it is constrained structured generation followed by deterministic
validation and the reality-integrity gate. It produces actors with *conditional*
behavior (an inclination that follows from evidence plus reaction rules), an
institution, a protocol graph, a causal graph, and the genuine uncertainty set —
then refuses to proceed unless the represented world matches verified reality.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actors import ActorState
from .evidence import EvidenceView
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .models import (
    ActorDefinition,
    BranchWeight,
    CausalEdge,
    CausalGraph,
    ConditionalBehavior,
    Entity,
    InstitutionSpec,
    ReactionRule,
    RealityManifest,
    ResolutionContract,
    ScenarioFrame,
    UncertaintyVariable,
    WeightProvenance,
)
from .prompts import render_compile_prompt
from .protocols import ProtocolGraph, committee_protocol
from .reality import verify_reality
from .research import ResearchBundle
from .uncertainty import ScenarioSet, enumerate_scenarios
from .world import WorldState


@dataclass(frozen=True)
class CompiledWorld:
    base_world: WorldState
    protocol: ProtocolGraph
    scenario_set: ScenarioSet
    causal_graph: CausalGraph
    manifest: RealityManifest
    uncertainty_variables: tuple[UncertaintyVariable, ...]
    frame: ScenarioFrame
    chair_actor_id: str
    proposal_option: str
    proposal_text: str
    briefing_text: str
    compile_responses: tuple[GatewayResponse, ...]


def _rule_to_dict(r: ReactionRule) -> dict[str, object]:
    return {
        "trigger_signal": r.trigger_signal,
        "direction": r.direction,
        "threshold": r.threshold,
        "moves_to_option": r.moves_to_option,
        "rationale": r.rationale,
        "evidence_claim_ids": list(r.evidence_claim_ids),
    }


def _rule_from_dict(d: dict[str, object]) -> ReactionRule:
    return ReactionRule(
        trigger_signal=str(d["trigger_signal"]),
        direction=str(d["direction"]),
        threshold=float(d["threshold"]),  # type: ignore[arg-type]
        moves_to_option=str(d["moves_to_option"]),
        rationale=str(d.get("rationale", "")),
        evidence_claim_ids=tuple(d.get("evidence_claim_ids", [])),  # type: ignore[arg-type]
    )


def _choose_chair(bundle: ResearchBundle) -> str:
    for m in bundle.members:
        if "introduce_proposal" in m.authority or "chair" in m.authority:
            return m.actor_id
    for m in bundle.members:
        if (
            "governor" in m.role.lower()
            or "chair" in m.role.lower()
            or "president" in m.role.lower()
        ):
            return m.actor_id
    voting = [m for m in bundle.members if m.is_voting_seat]
    return (voting or bundle.members)[0].actor_id


def compile_world(
    contract: ResolutionContract,
    evidence: EvidenceView,
    bundle: ResearchBundle,
    gateway: ModelGateway,
    *,
    seed: int,
    max_branches: int = 24,
) -> CompiledWorld:
    frame = bundle.frame

    # 1. Constrained structured generation: derive conditional behavior per member.
    members_ctx = [
        {
            "actor_id": m.actor_id,
            "name": m.name,
            "role": m.role,
            "is_voting_seat": m.is_voting_seat,
            "vote_power": m.vote_power,
            "prior_action": m.prior_action,
            "authority": list(m.authority),
            "evidence_claim_ids": list(m.evidence_claim_ids),
        }
        for m in bundle.members
    ]
    frame_ctx = {
        "guidance_option": frame.guidance_option,
        "guidance_text": frame.guidance_text,
        "acceptance_tolerance": frame.acceptance_tolerance,
        "reaction_rules": [_rule_to_dict(r) for r in frame.reaction_rules],
        "signals": [
            {"name": s.name, "baseline": s.baseline, "description": s.description}
            for s in frame.signals
        ],
    }
    ev_text, ev_ids = evidence.render_view(
        contract.decision_body + " " + " ".join(frame.options), limit=40
    )
    compile_ctx = {
        "options": list(frame.options),
        "members": members_ctx,
        "frame": frame_ctx,
        "evidence_view": ev_text,
        "evidence_claim_ids": list(ev_ids),
    }
    resp = gateway.generate(
        GatewayRequest(
            task_kind="compile_world",
            prompt=render_compile_prompt(compile_ctx),
            context=compile_ctx,
            seed=seed,
            expected_keys=("actors",),
        )
    )
    compiled_actors = {a["actor_id"]: a for a in resp.data.get("actors", [])}

    # 2. Deterministic validation into typed ActorDefinitions.
    actor_states: dict[str, ActorState] = {}
    entities: list[Entity] = []
    for m in bundle.members:
        gen = compiled_actors.get(m.actor_id)
        if gen is None:
            # The compiler must produce every verified member; absence is a defect.
            inclination = frame.guidance_option or m.prior_action or frame.options[0]
            gen = {
                "current_inclination": inclination,
                "reaction_rules": [_rule_to_dict(r) for r in frame.reaction_rules],
                "dissent_threshold": 0.5,
                "reasoning": "fallback: compiler omitted this member",
            }
        cb = ConditionalBehavior(
            current_inclination=str(gen["current_inclination"]),
            reaction_rules=tuple(_rule_from_dict(d) for d in gen.get("reaction_rules", [])),
            dissent_threshold=float(gen.get("dissent_threshold", 0.5)),
            reasoning=str(gen.get("reasoning", "")),
            acceptance_tolerance=frame.acceptance_tolerance,
            evidence_claim_ids=tuple(gen.get("evidence_claim_ids", m.evidence_claim_ids)),
        )
        definition = ActorDefinition(
            actor_id=m.actor_id,
            name=m.name,
            role=m.role,
            authority=m.authority,
            is_voting_seat=m.is_voting_seat,
            vote_power=m.vote_power,
            conditional_behavior=cb,
            stable_identity=m.stable_identity,
            memory_seeds=m.memory_seeds,
            evidence_claim_ids=m.evidence_claim_ids,
        )
        actor_states[m.actor_id] = ActorState.from_definition(
            definition, default_time=contract.as_of
        )
        entities.append(
            Entity(
                entity_id=m.actor_id,
                name=m.name,
                kind="person",
                evidence_claim_ids=m.evidence_claim_ids,
            )
        )

    institution = InstitutionSpec(
        institution_id=bundle.institution_id,
        name=bundle.institution_name,
        member_actor_ids=tuple(m.actor_id for m in bundle.members),
        decision_rule=bundle.decision_rule,
        seat_vote_powers=tuple(
            (m.actor_id, m.vote_power) for m in bundle.members if m.is_voting_seat
        ),
        evidence_claim_ids=bundle.decision_rule.evidence_claim_ids,
    )
    entities.append(
        Entity(
            entity_id=bundle.institution_id,
            name=bundle.institution_name,
            kind="institution",
        )
    )

    # 3. Reality-integrity gate — raises if the world is not faithful.
    manifest = verify_reality(contract, evidence, actor_states, institution)

    # 4. Assemble protocol, scenarios, uncertainty and causal graph.
    chair = _choose_chair(bundle)
    proposal_option = (
        frame.guidance_option or contract.terminal_predicate.target_option or frame.options[0]
    )
    proposal_text = frame.guidance_text or f"Maintain the current stance ({proposal_option})."
    briefing_text = f"Staff briefing for the {bundle.decision_body} decision."
    voting_ids = tuple(m.actor_id for m in bundle.members if m.is_voting_seat)
    protocol = committee_protocol(
        chair_actor_id=chair,
        proposal_option=proposal_option,
        proposal_text=proposal_text,
        briefing_text=briefing_text,
        voting_actor_ids=voting_ids,
        allow_revision=True,
    )
    scenario_set = enumerate_scenarios(frame, max_branches=max_branches)
    uncertainty_variables = _uncertainty_variables(frame)
    causal_graph = _causal_graph(frame, voting_ids)

    base_world = WorldState(
        branch_id="root",
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.DIRECT_EMPIRICAL, "verified initial world"),
        time=contract.as_of,
        contract=contract,
        evidence=evidence,
        entities=tuple(entities),
        actors=actor_states,
        institution=institution,
        verified_facts=bundle.world_facts,
        uncertain_facts=(),
        proposals=(),
        signals=tuple(sorted((s.name, s.baseline) for s in frame.signals)),
        protocol_stage="initial",
    )
    return CompiledWorld(
        base_world=base_world,
        protocol=protocol,
        scenario_set=scenario_set,
        causal_graph=causal_graph,
        manifest=manifest,
        uncertainty_variables=uncertainty_variables,
        frame=frame,
        chair_actor_id=chair,
        proposal_option=proposal_option,
        proposal_text=proposal_text,
        briefing_text=briefing_text,
        compile_responses=(resp,),
    )


def _uncertainty_variables(frame: ScenarioFrame) -> tuple[UncertaintyVariable, ...]:
    out: list[UncertaintyVariable] = []
    for spec in frame.uncertainty:
        out.append(
            UncertaintyVariable(
                variable_id=spec.signal,
                description=f"Future value of {spec.signal}",
                why_unknown=spec.why_unknown,
                how_it_affects_outcome=(
                    "Can cross a member reaction threshold and change a preferred option."
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


def _causal_graph(frame: ScenarioFrame, voting_ids: tuple[str, ...]) -> CausalGraph:
    nodes = [s.name for s in frame.signals] + list(voting_ids) + ["proposal", "votes", "terminal"]
    edges: list[CausalEdge] = []
    for s in frame.signals:
        for aid in voting_ids:
            edges.append(CausalEdge(cause=s.name, effect=aid, mechanism="reaction_rule"))
    for aid in voting_ids:
        edges.append(CausalEdge(cause="proposal", effect=aid, mechanism="focal_anchoring"))
        edges.append(CausalEdge(cause=aid, effect="votes", mechanism="cast_vote"))
    edges.append(CausalEdge(cause="votes", effect="terminal", mechanism="deterministic_tally"))
    return CausalGraph(nodes=tuple(nodes), edges=tuple(edges))
