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
from .coverage import (
    CompilationCoverageReport,
    EvidenceCandidate,
    ExclusionReviewer,
    WorldObject,
    WorldSpecView,
    assess_coverage,
    build_candidate_inventory,
    enforce_coverage,
)
from .evidence import EvidenceView
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .grounding import (
    ActorGroundingProfile,
    ActorGroundingReport,
    assess_actor_grounding,
    enforce_actor_grounding,
    profile_from_member,
)
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
from .protocols import ProtocolGraph, committee_protocol, general_protocol
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
    coverage_report: CompilationCoverageReport
    actor_grounding: ActorGroundingReport


def _rule_to_dict(r: ReactionRule) -> dict[str, object]:
    return {
        "trigger_signal": r.trigger_signal,
        "direction": r.direction,
        "threshold": r.threshold,
        "moves_to_option": r.moves_to_option,
        "rationale": r.rationale,
        "evidence_claim_ids": list(r.evidence_claim_ids),
    }


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _rule_from_dict(d: dict[str, object]) -> ReactionRule:
    return ReactionRule(
        trigger_signal=str(d.get("trigger_signal", "")),
        direction=str(d.get("direction", "above")),
        threshold=_as_float(d.get("threshold"), 0.5),
        moves_to_option=str(d.get("moves_to_option", "")),
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
        (contract.decision_body or "") + " " + " ".join(frame.options), limit=40
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
    profiles: list[ActorGroundingProfile] = []
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
            current_inclination=str(
                gen.get("current_inclination")
                or frame.guidance_option
                or (m.prior_action or frame.options[0])
            ),
            # Reaction structure is authoritative from the evidence-grounded frame, not
            # re-parsed from per-actor LLM output (which varies in shape).
            reaction_rules=frame.reaction_rules,
            dissent_threshold=_as_float(gen.get("dissent_threshold"), 0.5),
            reasoning=str(gen.get("reasoning", "")),
            acceptance_tolerance=frame.acceptance_tolerance,
            evidence_claim_ids=tuple(gen.get("evidence_claim_ids", m.evidence_claim_ids)),
        )
        # Ground the actor as a specific person: the VERIFIED previous action is kept as
        # its own field and is never overwritten by the inferred current inclination.
        profile = profile_from_member(
            actor_id=m.actor_id,
            name=m.name,
            role=m.role,
            authority=m.authority,
            previous_action=m.prior_action,
            previous_action_claim_ids=m.evidence_claim_ids,
            memory_seeds=tuple((s.content, s.evidence_claim_ids) for s in m.memory_seeds),
            inclination=cb.current_inclination,
            inclination_claim_ids=cb.evidence_claim_ids,
            reaction_rules=tuple((r.trigger_signal, r.moves_to_option) for r in cb.reaction_rules),
            valid_time=contract.as_of.date().isoformat(),
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
            grounding=profile,
        )
        profiles.append(profile)
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

    # 3b. Actor-grounding gate — refuse to simulate named people as generic role
    #     templates, or to let one actor carry another's personal history.
    grounding_report = assess_actor_grounding(tuple(profiles))
    enforce_actor_grounding(grounding_report)

    # 4. Assemble protocol, scenarios, uncertainty and causal graph.
    chair = _choose_chair(bundle)
    proposal_option = (
        frame.guidance_option or contract.terminal_predicate.target_option or frame.options[0]
    )
    proposal_text = frame.guidance_text or f"Maintain the current stance ({proposal_option})."
    briefing_text = f"Context briefing for the {bundle.decision_body} decision."
    voting_ids = tuple(m.actor_id for m in bundle.members if m.is_voting_seat)
    if contract.terminal_predicate.mechanism == "actor_action":
        # A non-committee question: the outcome is an actor taking (or not taking) an
        # action. Actors act in sequence; the terminal is read from the event history.
        turn_order = tuple(m.actor_id for m in bundle.members)
        protocol = general_protocol(turn_order=turn_order, briefing_text=briefing_text)
    else:
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

    # 5. Evidence-to-world coverage gate — refuse to simulate a world that silently
    #    dropped a materially relevant, verified evidence candidate. This is enforced
    #    deterministically and generalizes beyond the roster to every world element.
    wired_actor_ids = (
        set(voting_ids)
        if contract.terminal_predicate.mechanism != "actor_action"
        else {m.actor_id for m in bundle.members}
    )
    spec_view, signal_claim_ids = _world_spec_view(
        contract, bundle, actor_states, institution, frame, wired_actor_ids
    )
    inventory = build_candidate_inventory(evidence, contract, signal_claim_ids=signal_claim_ids)
    coverage_report = assess_coverage(
        inventory, spec_view, exclusion_reviewer=_exclusion_reviewer(gateway)
    )
    enforce_coverage(coverage_report)

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
        coverage_report=coverage_report,
        actor_grounding=grounding_report,
    )


# Kinds whose incidental exclusion is most dangerous to get wrong — a lost rule,
# event, document, channel, org, population, or external process can silently change
# the outcome. Persons excluded for lack of any role signal, and generic
# variables/prior-actions, are far less likely to be miscategorized, so they are not
# challenged (keeping the live LLM cost bounded).
_CHALLENGE_KINDS = frozenset(
    {
        "rule",
        "scheduled_event",
        "document",
        "channel",
        "organization",
        "population_group",
        "external_process",
        "causal_relation",
        "resource",
    }
)
_MAX_CHALLENGES = 12


def _exclusion_reviewer(gateway: ModelGateway) -> ExclusionReviewer | None:
    """An independent LLM review of borderline exclusions (live gateways only).

    The compiler may not drop a candidate merely by deeming it irrelevant: for the
    consequential kinds, a separate model pass is asked whether omitting it could
    plausibly change an actor's knowledge, authority, feasible actions, a constraint,
    a branch, timing, or the outcome. A "yes" invalidates the exclusion and blocks.
    Offline/deterministic gateways get no reviewer, so the gate stays deterministic.
    """

    if not getattr(gateway, "is_live", False):
        return None
    budget = {"n": 0}

    def review(cand: EvidenceCandidate) -> bool:
        if cand.kind.value not in _CHALLENGE_KINDS or budget["n"] >= _MAX_CHALLENGES:
            return False
        budget["n"] += 1
        prompt = (
            "An automated compiler is about to EXCLUDE the following verified evidence "
            "item from a simulated world as irrelevant. Challenge that decision.\n\n"
            f"ITEM ({cand.kind.value}): {cand.canonical_identity}\n"
            f"DESCRIPTION: {cand.description}\n\n"
            "Would including or removing this item plausibly change an actor's "
            "knowledge, authority, feasible actions, a resource constraint, the causal "
            "pathway, an uncertainty branch, the timing of events, or the terminal "
            'outcome? Reply JSON {"could_matter": true|false, "why": "..."}. '
            "Answer true only if it plausibly could."
        )
        try:
            resp = gateway.generate(
                GatewayRequest(
                    task_kind="exclusion_challenge",
                    prompt=prompt,
                    context={"candidate_id": cand.candidate_id},
                    seed=0,
                    expected_keys=("could_matter",),
                )
            )
        except Exception:
            # A provider failure must not silently drop the item: treat as "could matter".
            return True
        val = resp.data.get("could_matter")
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "yes", "1")

    return review


def _world_spec_view(
    contract: ResolutionContract,
    bundle: ResearchBundle,
    actor_states: dict[str, ActorState],
    institution: InstitutionSpec,
    frame: ScenarioFrame,
    wired_actor_ids: set[str],
) -> tuple[WorldSpecView, frozenset[str]]:
    """Describe the compiled world for the coverage comparison: every world object,
    whether it is causally wired, and the claims actors can actually perceive."""

    objects: list[WorldObject] = []
    accessible: set[str] = set()

    for aid, st in actor_states.items():
        d = st.definition
        seed_ids = {cid for s in d.memory_seeds for cid in s.evidence_claim_ids}
        claim_ids = tuple(sorted(set(d.evidence_claim_ids) | seed_ids))
        accessible.update(claim_ids)
        wired = aid in wired_actor_ids
        uses: tuple[str, ...] = ("actor_view",)
        if wired and d.is_voting_seat:
            uses = ("actor_view", "vote", "terminal")
        elif wired:
            uses = ("actor_view", "action")
        objects.append(
            WorldObject(
                object_id=aid,
                kind="actor",
                name=d.name,
                claim_ids=claim_ids,
                wired=wired,
                uses=uses,
            )
        )

    objects.append(
        WorldObject(
            object_id=institution.institution_id,
            kind="institution",
            name=institution.name,
            claim_ids=institution.evidence_claim_ids,
            wired=True,
            uses=("membership", "terminal"),
        )
    )
    objects.append(
        WorldObject(
            object_id="decision_rule",
            kind="rule",
            name=f"{contract.decision_rule.kind} rule",
            claim_ids=contract.decision_rule.evidence_claim_ids,
            wired=True,
            uses=("terminal",),
        )
    )
    objects.append(
        WorldObject(
            object_id="terminal",
            kind="terminal",
            name=(
                f"{contract.terminal_predicate.mechanism}:"
                f"{contract.terminal_predicate.yes_condition}"
            ),
            claim_ids=contract.terminal_predicate.evidence_claim_ids,
            wired=True,
            uses=("terminal",),
        )
    )

    # Guidance is a first-class part of the world: it anchors every actor's initial
    # inclination, so the evidence behind it is causally wired (focal anchoring).
    if frame.guidance_evidence_ids:
        accessible.update(frame.guidance_evidence_ids)
        objects.append(
            WorldObject(
                object_id="guidance",
                kind="guidance",
                name=f"guidance:{frame.guidance_option or ''}",
                claim_ids=frame.guidance_evidence_ids,
                wired=True,
                uses=("focal_anchoring", "actor_view"),
            )
        )

    reaction_signals = {r.trigger_signal for r in frame.reaction_rules}
    uncertainty_signals = {u.signal for u in frame.uncertainty}
    signal_claim_ids: set[str] = set(frame.guidance_evidence_ids)
    for s in frame.signals:
        signal_claim_ids.update(s.evidence_claim_ids)
        accessible.update(s.evidence_claim_ids)
        wired = s.name in reaction_signals or s.name in uncertainty_signals
        objects.append(
            WorldObject(
                object_id=f"signal:{s.name}",
                kind="signal",
                name=s.name,
                claim_ids=s.evidence_claim_ids,
                wired=wired,
                uses=("reaction",) if wired else (),
            )
        )
    for r in frame.reaction_rules:
        signal_claim_ids.update(r.evidence_claim_ids)
    for u in frame.uncertainty:
        signal_claim_ids.update(u.constraining_evidence_ids)
        objects.append(
            WorldObject(
                object_id=f"uncertainty:{u.signal}",
                kind="scheduled_event",
                name=u.signal,
                claim_ids=u.constraining_evidence_ids,
                wired=True,
                uses=("reaction", "branch"),
            )
        )

    for wf in bundle.world_facts:
        accessible.update(wf.evidence_claim_ids)
        objects.append(
            WorldObject(
                object_id=wf.fact_id,
                kind="world_fact",
                name=wf.text[:48],
                claim_ids=wf.evidence_claim_ids,
                wired=True,
                uses=("actor_view",),
            )
        )

    view = WorldSpecView(
        objects=tuple(objects),
        accessible_claim_ids=frozenset(accessible),
        decision_body=contract.decision_body,
        subject_entity=contract.subject_entity,
    )
    return view, frozenset(signal_claim_ids)


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
