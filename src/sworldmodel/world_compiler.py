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
from .errors import GatewayError, WorldIntegrityError
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
    orphans = sorted(a.entity_id for a in spec.actors if a.entity_id not in entities_by_id)
    if orphans:
        # Inventing an entity here would put a person in the simulation that verified
        # reality never described, and the coverage gate — which enumerates entities —
        # would never see them.
        raise WorldIntegrityError(
            f"actors {orphans} were compiled without a matching entity. Every actor must be "
            "a declared entity carrying its own evidence citations.",
            details={
                "failure": "orphan_actors",
                "recompilable": True,
                "orphan_actors": orphans,
            },
        )
    actor_states: dict[str, ActorState] = {}
    for aspec in spec.actors:
        entity = entities_by_id[aspec.entity_id]
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
    manifest = verify_reality(contract, evidence, base_world.actors, spec)

    # Gate 2 — actors must be specific grounded entities, not generic role templates.
    profiles = tuple(
        st.grounding
        for st in base_world.actors.values()
        if isinstance(st.grounding, ActorGroundingProfile)
    )
    grounding_report = assess_actor_grounding(profiles)
    enforce_actor_grounding(grounding_report)

    # Gate 3 — the outcome must be produced by what actors do, not supplied to them.
    enforce_outcome_is_produced(spec, uncertainties)

    # Gate 4 — evidence-to-world coverage against the exact compiled WorldSpec.
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


# Representations that are deliberately synthetic stand-ins rather than named real
# people. ``representation_scale`` is the authority because it is the field the compiler
# is actually asked for; ``kind`` is checked too, but the schema only ever offers
# person/organization/object/document/channel there, so keying the exemption off `kind`
# alone — as this did — meant a compiled population stratum was never once recognized as
# one, and was then judged by the standards of a named individual.
_CONSTRUCTED_SCALES = frozenset({"population_stratum", "network"})
_CONSTRUCTED_KINDS = frozenset({"population_group", "stratum", "segment", "cohort"})


def _is_constructed(entity: EntitySpec) -> bool:
    return entity.representation_scale in _CONSTRUCTED_SCALES or entity.kind in _CONSTRUCTED_KINDS


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

    # An actor's inclination is its compiled *reasoning* about why it leans as it does,
    # marked as an inference. It is never a pre-selected action: nothing here may tell
    # the runtime what this actor is going to do.
    inclination = aspec.reasoning.strip()
    reactions: tuple[tuple[str, str], ...] = ()

    profile = profile_from_member(
        actor_id=entity.entity_id,
        name=entity.name,
        role=entity.role,
        authority=entity.authority,
        previous_action=None,  # history lives in the seeds and is sorted by its wording
        memory_seeds=tuple(seeds),
        inclination=inclination or None,
        # The compiled reasoning is an inference *from this entity's cited evidence*, so
        # it carries those citations and is marked SUPPORTED_INFERENCE. An entity with
        # no citations has no grounded inclination either, and the actor is told nothing
        # rather than told a guess.
        inclination_claim_ids=tuple(entity.evidence_claim_ids),
        reaction_rules=reactions,
        valid_time=contract.as_of.isoformat(),
    )
    constructed = _is_constructed(entity)
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

    # -- external (non-agent) processes -----------------------------------------
    for pid, description, claim_ids in _external_process_objects(spec):
        accessible.update(claim_ids)
        objects.append(
            WorldObject(
                object_id=f"external:{pid}",
                kind="scheduled_event",
                name=description[:64],
                claim_ids=claim_ids,
                wired=True,
                uses=("external_process",),
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
        used |= _expr_fields(node.entry_condition)
        for eff in node.effects:
            used |= _effect_fields(eff)
    for proc in spec.external_processes:
        for occ in proc.occurrences:
            used |= _expr_fields(occ.condition)
            for eff in occ.effects:
                used |= _effect_fields(eff)
    for rule in spec.wake_rules:
        if rule.on_field_change:
            used.add(rule.on_field_change)
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


def _expr_collections(expr: Any) -> set[str]:
    """Record collections a declarative expression reads (``count``/``sum``/``values``)."""

    from .worldspec import Expr

    if not isinstance(expr, Expr):
        return set()
    out: set[str] = set()
    if expr.op in ("count", "sum", "values", "exists") and expr.args:
        first = expr.args[0]
        if isinstance(first, str):
            out.add(first)
        elif isinstance(first, Expr) and first.op == "const" and first.args:
            out.add(str(first.args[0]))
    for a in expr.args:
        out |= _expr_collections(a)
    return out


def _effect_writes(effects: Any) -> tuple[set[str], set[str]]:
    fields: set[str] = set()
    collections: set[str] = set()
    for eff in effects:
        fields |= _effect_fields(eff)
        if eff.op == "append_record":
            coll = eff.params_dict.get("collection")
            if isinstance(coll, str):
                collections.add(coll)
    return fields, collections


def _action_writes(spec: WorldSpec) -> tuple[set[str], set[str]]:
    """What the compiled actions can actually change: (fields, record collections)."""

    fields: set[str] = set()
    collections: set[str] = set()
    for action in spec.actions:
        f, c = _effect_writes(action.effects)
        fields |= f
        collections |= c
    return fields, collections


def terminal_producers(spec: WorldSpec) -> dict[str, tuple[str, ...]]:
    """For every term the terminal reads, what in this world can write it.

    This is the compile-time half of terminal producer lineage: before anything runs,
    each terminal term must name at least one thing whose *operation* could set it — an
    action somebody takes, a process node that fires, or an external/operational process
    that advances. The runtime half records which of those actually did it.

    An uncertainty is deliberately not a producer. A branch condition may influence a
    producer; it may not stand in for one. A terminal term whose only writer is an
    uncertainty's ``field_effects`` is the answer wearing the costume of a world state.
    """

    terms = _expr_fields(spec.terminal.yes_when) | _expr_collections(spec.terminal.yes_when)
    producers: dict[str, list[str]] = {t: [] for t in terms}

    def record(label: str, effects: Any) -> None:
        fields, colls = _effect_writes(effects)
        for term in fields | colls:
            if term in producers:
                producers[term].append(label)

    for action in spec.actions:
        record(f"action:{action.action_id}", action.effects)
    for node in spec.process.nodes:
        record(f"process_node:{node.node_id}", node.effects)
    for proc in spec.external_processes:
        for i, occ in enumerate(proc.occurrences):
            record(f"external_process:{proc.process_id}#{i}", occ.effects)
    return {k: tuple(v) for k, v in producers.items()}


def _environment_preset_terminal_terms(spec: WorldSpec, terminal_fields: set[str]) -> set[str]:
    """Terminal terms a scheduled non-agent effect writes to a literal, unconditionally.

    "Unconditionally" is the load-bearing part. A process node whose entry condition
    reads a field an action can write is a *consequence* of what actors did — a session
    that tallies the votes cast into it is exactly right. A node with no such condition,
    writing a constant, is the compiler putting the answer on the calendar.
    """

    action_writes, _ = _action_writes(spec)
    preset: set[str] = set()

    def literal_targets(effects: Any) -> set[str]:
        out: set[str] = set()
        for eff in effects:
            # Only `set_field` counts. `adjust_field` accumulates, and accumulation with
            # a fixed step is exactly how an honest operational process models
            # throughput — a line that builds so many units per shift is production, not
            # an announcement. Declaring the term to *be* a value is the defect.
            if eff.op != "set_field":
                continue
            params = eff.params_dict
            name = params.get("field")
            value = params.get("value")
            # A value that references a parameter or another field is computed from the
            # world; only a bare literal is the environment asserting an outcome.
            if isinstance(name, str) and not (isinstance(value, str) and value.startswith("$")):
                out.add(name)
        return out

    for node in spec.process.nodes:
        if _expr_fields(node.entry_condition) & action_writes:
            continue  # gated on something actors do
        preset |= literal_targets(node.effects) & terminal_fields
    for proc in spec.external_processes:
        for occ in proc.occurrences:
            if _expr_fields(occ.condition) & action_writes:
                continue
            preset |= literal_targets(occ.effects) & terminal_fields
    return preset


def enforce_outcome_is_produced(
    spec: WorldSpec, uncertainties: tuple[UncertaintySpec, ...]
) -> None:
    """Refuse a world whose answer nobody has to do anything to produce.

    This is the gate that catches a forecast dressed as a simulation. If the terminal
    reads only things that compiled *actions* never write — typically because the
    compiler encoded the decision itself as an uncertain input field — then the branch
    weights are the forecast and the actors are scenery. Every trajectory "resolves"
    without anyone acting, and the reported probability is the model's prior on that
    field wearing the label ``weighted_simulated_trajectories``.

    The world must be one in which the outcome is *produced*: at least one compiled
    action must be able to move at least one term the terminal reads.
    """

    terminal_fields = _expr_fields(spec.terminal.yes_when)
    terminal_colls = _expr_collections(spec.terminal.yes_when)
    if not spec.actions and not spec.external_processes:
        raise WorldIntegrityError(
            "the compiled world has no actions and no external processes — nobody can "
            "do anything and nothing runs, so the outcome cannot be produced",
            details={"failure": "nothing_can_act", "recompilable": True},
        )
    if not spec.process.nodes and not spec.external_processes:
        # A world may legitimately be driven entirely by external processes and wake
        # rules rather than a procedural graph. What it may not be is a world in which
        # nothing is scheduled to happen at all.
        raise WorldIntegrityError(
            "the compiled world has no process and no external processes — nothing is "
            "scheduled to happen, so no actor will ever be in a position to act",
            details={"failure": "nothing_scheduled", "recompilable": True},
        )

    # Every terminal term must have at least one producer. A world where actors act
    # busily on fields the terminal never reads, while the terminal's own terms arrive
    # from branch weights, is the most dangerous shape this gate exists to catch: it
    # looks alive and its answer was fixed before anyone opened their mouth.
    producers = terminal_producers(spec)
    orphans = sorted(term for term, who in producers.items() if not who)
    uncertain = {u.variable for u in uncertainties} | {
        name for u in uncertainties for o in u.outcomes for name, _ in o.field_effects
    }
    written_fields, written_colls = _action_writes(spec)

    if not orphans:
        # An environment that simply announces the answer is the third form of the same
        # defect. A live run compiled a world in which a scheduled process node carried
        # `set_field(<the terminal term>, True)` with a literal value and no entry
        # condition: it fired before the actor was ever invoked, the terminal was already
        # decided, and the actor — woken afterwards — observed that the thing had
        # happened and waited. The forecast was 1.0000 with zero producing actions, and
        # this gate passed it because *some* action could in principle have written the
        # term. In a world with actors, a term the terminal reads may not be set to a
        # constant by the scenery.
        preset = _environment_preset_terminal_terms(spec, terminal_fields)
        if spec.actors and preset:
            raise WorldIntegrityError(
                f"the environment writes the answer: {sorted(preset)} is set to a fixed "
                "value by a scheduled process that no actor influences, so the terminal "
                "is decided before anyone acts and the actors are observers of their own "
                "outcome",
                details={
                    "failure": "environment_presets_terminal",
                    "recompilable": True,
                    "terms preset by the environment": sorted(preset),
                    "actors": [a.entity_id for a in spec.actors],
                    "producers by terminal term": {
                        k: list(v) for k, v in sorted(producers.items())
                    },
                },
            )
        # A world that compiled actors owes those actors a causal role. If every
        # terminal term is written only by processes while people deliberate over
        # fields the terminal never reads, the deliberation is decoration — the same
        # defect as an uncertainty writing the answer, one layer further out.
        if spec.actors and not (
            (terminal_fields & written_fields) or (terminal_colls & written_colls)
        ):
            raise WorldIntegrityError(
                "this world compiled actors who cannot affect the outcome: every "
                "terminal term is written by a process, and no action any actor can "
                "take moves any of them. Either the actors belong in the causal path "
                "or they do not belong in the world",
                details={
                    "failure": "actors_cannot_reach_terminal",
                    "recompilable": True,
                    "actors": [a.entity_id for a in spec.actors],
                    "terminal reads fields": sorted(terminal_fields),
                    "fields any action can write": sorted(written_fields),
                    "producers by terminal term": {
                        k: list(v) for k, v in sorted(producers.items())
                    },
                },
            )
        return

    raise WorldIntegrityError(
        "the outcome is an input, not a result: nothing in this world can produce "
        f"{orphans} — no action, process node or external process writes it — so every "
        "branch resolves without anything happening and the answer would be the branch "
        "weights rather than the simulation",
        details={
            "failure": "terminal_has_no_producer",
            "recompilable": True,
            "terminal terms with no producer": orphans,
            "terminal reads fields": sorted(terminal_fields),
            "terminal reads collections": sorted(terminal_colls),
            "fields any action can write": sorted(written_fields),
            "collections any action can write": sorted(written_colls),
            "terminal terms supplied by uncertainty instead": sorted(set(orphans) & uncertain),
            "producers by terminal term": {k: list(v) for k, v in sorted(producers.items())},
        },
    )


def _external_process_objects(spec: WorldSpec) -> list[tuple[str, str, tuple[str, ...]]]:
    """External (non-agent) processes are part of the compiled world and must be
    visible to the coverage gate, or a verified scheduled release could be represented
    and still be reported as missing."""

    return [
        (p.process_id, p.description or p.process_id, p.evidence_claim_ids)
        for p in spec.external_processes
    ]


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
            # An unlabeled weight is an unjustified weight. Defaulting to a strong
            # label (explicit_model) would let an invented number outrank an honest
            # one in the weakest-provenance ordering that stamps the branch.
            provenance = (
                WeightProvenance(prov)
                if prov in _VALID_PROVENANCE
                else WeightProvenance.SYMMETRIC_IGNORANCE
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
                depends_on=tuple(str(d) for d in (u.get("depends_on") or [])),
                release_at=_parse_iso(u.get("release_at")),
            )
        )
    return tuple(out)


def _parse_iso(value: object) -> datetime | None:
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _epistemic(raw: object, has_citations: bool) -> EpistemicType:
    """Read an epistemic label the model produced.

    Unrecognized labels are common — a compiler will happily write "rule" or "fact" —
    and they must not crash a run. They also must not be promoted to OBSERVATION, which
    is what a permissive default would do: an unrecognized label means we do not know
    the epistemic status, and the one thing we may not do is call it established fact.
    An uncited statement is a hypothesis regardless of what it was labeled.
    """

    if isinstance(raw, str):
        try:
            return EpistemicType(raw.strip().lower())
        except ValueError:
            pass
    return EpistemicType.INFERENCE if has_citations else EpistemicType.HYPOTHESIS


def parse_world_facts(items: Any, default_time: datetime) -> tuple[WorldFact, ...]:
    facts: list[WorldFact] = []
    for i, wf in enumerate(items or []):
        at = wf.get("available_at")
        cites = tuple(str(c) for c in (wf.get("evidence_claim_ids") or []))
        facts.append(
            WorldFact(
                fact_id=f"fact_{i}",
                text=str(wf.get("text", "")),
                evidence_claim_ids=cites,
                available_at=datetime.fromisoformat(at) if isinstance(at, str) else default_time,
                epistemic_type=_epistemic(wf.get("epistemic_type"), bool(cites)),
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


def render_evidence(view: EvidenceView, *, limit: int = 160) -> str:
    claims = sorted(view.available(), key=lambda c: (-int(c.authority_level), c.id))[:limit]
    return "\n".join(
        f"{c.id} | {c.proposition} = {c.normalized_value} "
        f"[auth {int(c.authority_level)}, {c.source_type.value}, {c.published_at.date()}]"
        for c in claims
    )


def compile_world_spec_live(
    gateway: ModelGateway,
    question: str,
    as_of: datetime,
    horizon: datetime,
    view: EvidenceView,
    *,
    extra_instruction: str = "",
    structure_id: str = "primary",
) -> tuple[dict[str, Any], Any]:
    """Ask the model to compile the whole causal world for an arbitrary question, then
    normalize and citation-check it. Returns ``(compilation_dict, gateway_response)``
    where the dict has keys ``world_spec``, ``uncertainties``, ``world_facts``,
    ``required_reality_facts`` and reality metadata."""

    ctx = {
        "question": question,
        "as_of": as_of.isoformat(),
        "horizon": horizon.isoformat(),
        "evidence": render_evidence(view),
        # The deterministic inventory of what verified reality actually contains. The
        # model is handed this explicitly so it cannot silently forget a verified item
        # while compiling; the coverage gate then checks the compiled world against it.
        "checklist": evidence_checklist(view, as_of=as_of, horizon=horizon),
        # Set only when compiling a structural alternative: the same evidence, a
        # different causal structure the evidence also leaves open.
        "extra_instruction": extra_instruction,
    }
    resp = gateway.generate(
        GatewayRequest(
            task_kind="compile_world_spec",
            prompt=render_world_compile_prompt(ctx),
            context={"question": question, "structure_id": structure_id},
            seed=int(prompt_hash("world" + question + structure_id)[:8], 16),
            expected_keys=("world_spec",),
        )
    )
    data = _normalize_compilation(resp.data, view, as_of, horizon), resp
    compiled_dict = data[0]
    spec_dict = compiled_dict.get("world_spec")
    if isinstance(spec_dict, dict):
        spec_dict.setdefault("structure_id", structure_id)
    return compiled_dict, resp


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
    # A normalized compilation is complete on its own: every consumer reads the same
    # `reality` block, so an alternative structure compiled through this function is
    # assembled by exactly the same code as the primary one.
    reality = dict(data.get("reality") or {})
    reality.setdefault("as_of", data["as_of"])
    reality.setdefault("horizon", data["horizon"])
    reality.setdefault("subject_entity", data["subject_entity"])
    reality.setdefault("resolution_units", data["resolution_units"])
    reality.setdefault("target_outcome", data["target_outcome"])
    if data.get("expected_participants") is not None:
        reality.setdefault("expected_participants", data["expected_participants"])
    data["reality"] = reality
    return data


def _s(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
