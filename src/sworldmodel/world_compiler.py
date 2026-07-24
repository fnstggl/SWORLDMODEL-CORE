"""The one world compiler: turn a compiled :class:`WorldSpec` into a verified, runnable
:class:`CompiledWorld`, and (live) drive the LLM that authors that WorldSpec.

Two entry points share the same output:

* :func:`compile_world` — build the base world from an already-parsed WorldSpec (from an
  authored corpus offline, or from the live LLM), run the reality-integrity gate **and
  the evidence-to-world coverage gate against that exact WorldSpec**, and enumerate
  genuine-uncertainty branches.
* :func:`compile_world_spec_live` — ask the model to compile the entire world (entities,
  actors, actions, process graph, declarative terminal, uncertainties) from verified
  evidence for an arbitrary question, handing it the deterministic evidence checklist so
  it cannot silently forget a verified item, then normalize/citation-check the result.

Nothing here is question-specific. A committee, a negotiation, a population process and
a geopolitical process differ only in the *data* the compiler emits — never in code.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from .actors import ActorState
from .compiled import CompiledWorld
from .coverage import (
    EvidenceCandidate,
    ExclusionReviewer,
    WorldObject,
    WorldSpecView,
    assess_coverage,
    build_candidate_inventory,
    enforce_coverage,
    evidence_checklist,
)
from .errors import GatewayError
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .grounding import (
    ActorGroundingProfile,
    assess_actor_grounding,
    enforce_actor_grounding,
    profile_from_member,
)
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
        state = ActorState.from_spec(entity, aspec, default_time=contract.as_of)
        # Ground each actor as the specific real entity it is, with provenance on every
        # element, so its prompt carries its own verified history rather than a template.
        state.grounding = actor_grounding_profile(entity, aspec, contract)
        actor_states[aspec.entity_id] = state

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
    gateway: ModelGateway | None = None,
    seed: int = 0,
    max_branches: int = 24,
    compile_responses: tuple[Any, ...] = (),
) -> CompiledWorld:
    """Build the runnable world and run both integrity gates before any simulation.

    The coverage gate compares the verified-evidence candidate inventory against the
    *exact* :class:`WorldSpec` returned here — the same object the engine executes — so
    a fact that research verified cannot be silently absent from the simulated world.
    """

    base_world = build_base_world(spec, contract, evidence, world_facts)

    # Gate 1 — reality integrity: refuse a structurally false world.
    manifest = verify_reality(contract, evidence, base_world.actors)

    # Gate 2 — actors must be specific grounded entities, not generic role templates.
    profiles = tuple(
        st.grounding
        for st in base_world.actors.values()
        if isinstance(st.grounding, ActorGroundingProfile)
    )
    grounding_report = assess_actor_grounding(profiles)
    enforce_actor_grounding(grounding_report)

    # Gate 3 — evidence-to-world coverage against the exact compiled WorldSpec.
    view, signal_claim_ids = world_spec_view(spec, base_world, uncertainties, world_facts)
    inventory = build_candidate_inventory(
        evidence,
        contract,
        signal_claim_ids=signal_claim_ids,
        focal_identities=tuple(e.name for e in spec.entities) + (spec.title,),
    )
    coverage_report = assess_coverage(
        inventory, view, exclusion_reviewer=exclusion_reviewer(gateway)
    )
    enforce_coverage(coverage_report)

    scenario_set = enumerate_scenarios(
        uncertainties, base_world.fields_dict(), max_branches=max_branches
    )
    uvars = _uncertainty_variables(uncertainties)
    return CompiledWorld(
        base_world=base_world,
        spec=spec,
        scenario_set=scenario_set,
        manifest=manifest,
        coverage_report=coverage_report,
        actor_grounding=grounding_report,
        uncertainty_variables=uvars,
        compile_responses=tuple(compile_responses),
    )


def _default_actor_entity(aspec: ActorSpec) -> EntitySpec:
    return EntitySpec(entity_id=aspec.entity_id, name=aspec.entity_id, kind="person", is_actor=True)


# Entity kinds that are deliberately synthetic stand-ins rather than named real people.
_CONSTRUCTED_KINDS = frozenset({"population_group", "stratum", "segment", "cohort"})


def actor_grounding_profile(
    entity: EntitySpec, aspec: ActorSpec, contract: ResolutionContract
) -> ActorGroundingProfile:
    """Build this actor's grounded profile from its compiled evidence.

    Works for any kind of actor: a named person, a head of state, an organization
    acting as a unit, or a constructed population stratum. The actor's verified history
    (its memory seeds) is preserved as history with its own citations; the compiled
    disposition is recorded separately and explicitly marked as an inference.
    """

    seeds: list[tuple[str, tuple[str, ...]]] = []
    for raw in aspec.memory_seeds:
        d = dict(raw)
        content = str(d.get("content", "")).strip()
        if content:
            cids = tuple(str(c) for c in (d.get("evidence_claim_ids") or []))
            seeds.append((content, cids))

    policy = aspec.policy
    inclination = ""
    if policy.default_action_id:
        params = policy.default_params_dict
        detail = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        inclination = (
            f"{policy.default_action_id}({detail})" if detail else policy.default_action_id
        )
    reactions = tuple((r.when_field, r.action_id or "wait") for r in policy.rules if r.when_field)

    profile = profile_from_member(
        actor_id=entity.entity_id,
        name=entity.name,
        role=entity.role,
        authority=entity.authority,
        previous_action=None,  # history lives in the seeds and is sorted by its wording
        memory_seeds=tuple(seeds),
        inclination=inclination or None,
        reaction_rules=reactions,
        valid_time=contract.as_of.isoformat(),
    )
    constructed = entity.kind in _CONSTRUCTED_KINDS
    weight = entity.attributes_dict.get("weight") if constructed else None
    return replace(
        profile,
        claim_ids=tuple(entity.evidence_claim_ids),
        is_constructed_representative=constructed,
        population_weight=float(weight) if isinstance(weight, (int, float)) else None,
    )


# ---------------------------------------------------------------------------
# Coverage: describe the exact compiled WorldSpec for the integrity comparison
# ---------------------------------------------------------------------------


def world_spec_view(
    spec: WorldSpec,
    base_world: WorldState,
    uncertainties: tuple[UncertaintySpec, ...],
    world_facts: tuple[WorldFact, ...],
) -> tuple[WorldSpecView, frozenset[str]]:
    """Describe the compiled world for the coverage comparison: every world object,
    whether it is *causally wired* into the simulation, and the claims actors can
    actually perceive.

    "Wired" is read off the compiled program itself, with no domain assumptions:
    an entity is wired when it is an actor that some process node invites to act (or a
    target/participant of a compiled action); an action is wired when a process node
    offers it; a field is wired when an action, a precondition, a policy rule, an
    uncertainty or the terminal reads it; a document/resource/channel is wired when a
    compiled effect touches it.
    """

    objects: list[WorldObject] = []
    accessible: set[str] = set()

    participants = _wired_participants(spec)
    offered_actions = _offered_actions(spec)
    referenced_fields = _referenced_fields(spec, uncertainties)
    touched_objects = _touched_objects(spec)
    signal_claim_ids: set[str] = set()

    # -- entities (actors and everything else the world names) ------------------
    actor_ids = {a.entity_id for a in spec.actors}
    for ent in spec.entities:
        seed_claims = {
            cid
            for a in spec.actors
            if a.entity_id == ent.entity_id
            for s in a.memory_seeds
            for cid in (dict(s).get("evidence_claim_ids") or [])
        }
        claim_ids = tuple(sorted(set(ent.evidence_claim_ids) | {str(c) for c in seed_claims}))
        accessible.update(claim_ids)
        is_actor = ent.entity_id in actor_ids or ent.is_actor
        uses: tuple[str, ...]
        if is_actor:
            wired = ent.entity_id in participants
            uses = ("actor_view", "action") if wired else ("actor_view",)
            kind = "actor"
        else:
            wired = ent.entity_id in touched_objects
            uses = ("effect_target",) if wired else ()
            kind = ent.kind or "object"
        objects.append(
            WorldObject(
                object_id=ent.entity_id,
                kind=kind,
                name=ent.name,
                claim_ids=claim_ids,
                wired=wired,
                uses=uses,
            )
        )

    # -- compiled actions: the scenario-specific things actors may do -----------
    for action in spec.actions:
        signal_claim_ids.update(action.evidence_claim_ids)
        accessible.update(action.evidence_claim_ids)
        wired = action.action_id in offered_actions
        objects.append(
            WorldObject(
                object_id=f"action:{action.action_id}",
                kind="rule",
                name=f"{action.action_id}: {action.meaning}",
                claim_ids=action.evidence_claim_ids,
                wired=wired,
                uses=("action", "authority") if wired else (),
            )
        )

    # -- typed world fields -----------------------------------------------------
    for f in spec.fields:
        signal_claim_ids.update(f.evidence_claim_ids)
        accessible.update(f.evidence_claim_ids)
        wired = f.field_id in referenced_fields
        objects.append(
            WorldObject(
                object_id=f"field:{f.field_id}",
                kind="signal",
                name=f.field_id,
                claim_ids=f.evidence_claim_ids,
                wired=wired,
                uses=("reaction",) if wired else (),
            )
        )

    # -- documents / resources / channels the compiled effects touch ------------
    for doc in spec.documents:
        objects.append(
            WorldObject(
                object_id=f"document:{doc.document_id}",
                kind="document",
                name=doc.document_id,
                wired=doc.document_id in touched_objects,
                uses=("effect_target",) if doc.document_id in touched_objects else (),
            )
        )
    for res in spec.resources:
        objects.append(
            WorldObject(
                object_id=f"resource:{res.resource_id}",
                kind="resource",
                name=res.resource_id,
                wired=res.resource_id in touched_objects,
                uses=("resource",) if res.resource_id in touched_objects else (),
            )
        )
    for ch in spec.channels:
        objects.append(
            WorldObject(
                object_id=f"channel:{ch.channel_id}",
                kind="channel",
                name=ch.channel_id,
                wired=True,
                uses=("actor_view",),
            )
        )

    # -- the process graph and the declarative terminal -------------------------
    for node in spec.process.nodes:
        objects.append(
            WorldObject(
                object_id=f"node:{node.node_id}",
                kind="scheduled_event",
                name=f"{node.node_id} {node.description}".strip(),
                wired=True,
                uses=("process",),
            )
        )
    objects.append(
        WorldObject(
            object_id="terminal",
            kind="terminal",
            name=spec.terminal.description or "declarative terminal condition",
            wired=True,
            uses=("terminal",),
        )
    )

    # -- genuine uncertainties (branch-forming) ---------------------------------
    for u in uncertainties:
        signal_claim_ids.update(u.constraining_evidence_ids)
        accessible.update(u.constraining_evidence_ids)
        objects.append(
            WorldObject(
                object_id=f"uncertainty:{u.variable}",
                kind="scheduled_event",
                name=u.variable,
                claim_ids=u.constraining_evidence_ids,
                wired=True,
                uses=("reaction", "branch"),
            )
        )

    # -- verified world facts every actor can read ------------------------------
    for wf in world_facts:
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
        subject_entity=base_world.contract.subject_entity,
    )
    return view, frozenset(signal_claim_ids)


def _wired_participants(spec: WorldSpec) -> set[str]:
    """Actors some process node actually invites to act (directly, by role, or via *)."""

    all_actor_ids = {a.entity_id for a in spec.actors}
    roles = {e.entity_id: e.role for e in spec.entities}
    wired: set[str] = set()
    for node in spec.process.nodes:
        for sel in node.participants:
            if sel == "*":
                wired |= all_actor_ids
            elif sel.startswith("role:"):
                role = sel[5:]
                wired |= {aid for aid in all_actor_ids if roles.get(aid) == role}
            elif sel in all_actor_ids:
                wired.add(sel)
    return wired


def _offered_actions(spec: WorldSpec) -> set[str]:
    all_ids = {a.action_id for a in spec.actions}
    offered: set[str] = set()
    for node in spec.process.nodes:
        if node.action_ids == ("*",) or not node.action_ids:
            offered |= all_ids
        else:
            offered |= {a for a in node.action_ids if a in all_ids}
    return offered


def _referenced_fields(spec: WorldSpec, uncertainties: tuple[UncertaintySpec, ...]) -> set[str]:
    """Every field the compiled program reads or writes anywhere."""

    used: set[str] = set()
    for u in uncertainties:
        used.add(u.variable)
        for o in u.outcomes:
            used.update(name for name, _ in o.field_effects)
    for action in spec.actions:
        used |= _expr_fields(action.preconditions)
        for eff in action.effects:
            used |= _effect_fields(eff)
    for node in spec.process.nodes:
        used |= _expr_fields(node.condition)
        for eff in node.effects:
            used |= _effect_fields(eff)
    for a in spec.actors:
        used.update(r.when_field for r in a.policy.rules if r.when_field)
    used |= _expr_fields(spec.terminal.yes_when) | _expr_fields(spec.terminal.unresolved_when)
    return used


def _expr_fields(expr: Any) -> set[str]:
    from .worldspec import Expr

    if not isinstance(expr, Expr):
        return set()
    out: set[str] = set()
    if expr.op == "field" and expr.args:
        first = expr.args[0]
        if isinstance(first, str):
            out.add(first)
        elif isinstance(first, Expr) and first.op == "const" and first.args:
            out.add(str(first.args[0]))
    for a in expr.args:
        out |= _expr_fields(a)
    return out


def _effect_fields(eff: Any) -> set[str]:
    params = eff.params_dict
    out: set[str] = set()
    if eff.op in ("set_field", "adjust_field"):
        name = params.get("field")
        if isinstance(name, str):
            out.add(name)
    for key in ("fields", "info_fields", "data", "levels"):
        sub = params.get(key)
        if isinstance(sub, dict):
            out.update(str(k) for k in sub)
    return out


def _touched_objects(spec: WorldSpec) -> set[str]:
    """Documents, resources and entities named by any compiled effect or action target."""

    touched: set[str] = set()

    def scan(effects: Any) -> None:
        for eff in effects:
            p = eff.params_dict
            for key in ("document", "resource", "holder", "from", "to", "target", "audience"):
                val = p.get(key)
                if isinstance(val, str) and not val.startswith("$"):
                    touched.add(val)
                elif isinstance(val, list):
                    touched.update(
                        str(v) for v in val if isinstance(v, str) and not v.startswith("$")
                    )

    for action in spec.actions:
        scan(action.effects)
        touched.update(
            t for t in action.valid_targets if t not in ("*",) and not t.startswith("role:")
        )
        touched.update(r for r, _ in action.resource_costs)
    for node in spec.process.nodes:
        scan(node.effects)
    return touched


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


def exclusion_reviewer(gateway: ModelGateway | None) -> ExclusionReviewer | None:
    """An independent LLM review of borderline exclusions (live gateways only).

    The compiler may not drop a candidate merely by deeming it irrelevant: for the
    consequential kinds, a separate model pass is asked whether omitting it could
    plausibly change an actor's knowledge, authority, feasible actions, a constraint,
    a branch, timing, or the outcome. A "yes" invalidates the exclusion and blocks.
    Offline/deterministic gateways get no reviewer, so the gate stays deterministic.
    """

    if gateway is None or not getattr(gateway, "is_live", False):
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
        except GatewayError:
            # A provider failure must not silently drop the item: treat as "could matter".
            return True
        val = resp.data.get("could_matter")
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("true", "yes", "1")

    return review


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
        # The deterministic inventory of what verified reality actually contains. The
        # model is handed this explicitly so it cannot silently forget a verified item
        # while compiling; the coverage gate then checks the compiled world against it.
        "checklist": evidence_checklist(view, as_of=as_of, horizon=horizon),
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

    # Keep only required-reality facts the compiler could actually ground in available
    # evidence. A fact the model *names* but cannot cite (e.g. "the body has authority",
    # or a terminal date already given by the contract horizon) is not a verified
    # load-bearing fact and must not block the run — the participant-count, grounding and
    # coverage checks are independent and evidence-grounded, so real gaps are still
    # caught. A fact that DOES cite evidence is left untouched, so a citation that is
    # unavailable by the cutoff still refuses the run.
    grounded_facts = []
    for rf in data.get("required_reality_facts") or []:
        cited = [str(i) for i in (rf.get("evidence_claim_ids") or []) if str(i) in available]
        rf["evidence_claim_ids"] = cited
        if cited:
            grounded_facts.append(rf)
    data["required_reality_facts"] = grounded_facts

    data["subject_entity"] = _s(data.get("subject_entity")) or _s(ws.get("title")) or "the subject"
    data["resolution_units"] = _s(data.get("resolution_units")) or "the outcome"
    data["target_outcome"] = _s(data.get("target_outcome")) or "the YES condition"
    data["as_of"] = as_of.isoformat()
    data["horizon"] = horizon.isoformat()
    return data


def _s(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
