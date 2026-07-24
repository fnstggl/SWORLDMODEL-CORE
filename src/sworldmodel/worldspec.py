"""The compiled world specification — the *program* the LLM writes per question.

Nothing in this module knows what a committee, a vote, a negotiation, an election,
or a population is. It defines only the *shape* of an arbitrary compiled world:

* typed entities (some of which are actors) with capabilities and authority;
* typed world fields, resources, documents, and communication channels;
* :class:`ActionDefinition` records — the scenario-specific actions the compiler
  discovered, each mapping to a list of safe universal :class:`Effect` operations;
* a :class:`ProcessGraph` of nodes (what can happen, in what order, gated by
  declarative conditions, who acts, who observes);
* declarative :class:`Expr` predicates (preconditions, terminal condition).

The *runtime* (``engine``/``effects``/``expressions``) executes any world expressed
here. The *compiler* (``world_compiler`` live, or an authored corpus offline) fills
it in. Adding a new kind of question adds new **data** here — never new Python.

``Expr`` and ``Effect`` are deliberately data-only. An ``Expr`` is a tree of the
universal operators evaluated by :mod:`expressions`; an ``Effect`` names one of the
universal world operations executed by :mod:`effects`. The LLM can therefore only
compose the fixed primitives — it can never write arbitrary code or assert a raw
consequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Declarative expression tree (evaluated by expressions.evaluate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """A node in a declarative expression. ``op`` is one of the universal operators;
    ``args`` are literals (str/number/bool/None) or nested :class:`Expr`.

    The set of operators is fixed and universal (comparison, existence, count, sum,
    all/any, before/after/duration, boolean, and a few world accessors). It contains
    no question-specific term. See :mod:`expressions`.
    """

    op: str
    args: tuple[Any, ...] = ()


def parse_expr(obj: Any) -> Expr:
    """Parse a JSON-ish object into an :class:`Expr`.

    Accepted forms::

        3.14 | "text" | true | null           -> const literal
        {"op": "field", "args": ["inflation"]}
        {"field": "inflation"}                 -> shorthand for the above
        {"const": <value>}

    Any nested arg that is itself a dict is parsed recursively; bare literals are
    wrapped as ``const``.
    """

    if isinstance(obj, Expr):
        return obj
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return Expr("const", (obj,))
    if isinstance(obj, dict):
        if "const" in obj:
            return Expr("const", (obj["const"],))
        if "op" in obj:
            raw_args = obj.get("args", [])
            args = tuple(_parse_arg(a) for a in raw_args)
            return Expr(str(obj["op"]), args)
        # Shorthands: a single-key dict {op_name: arg_or_args}.
        if len(obj) == 1:
            (k, v) = next(iter(obj.items()))
            args = tuple(_parse_arg(a) for a in v) if isinstance(v, list) else (_parse_arg(v),)
            return Expr(str(k), args)
    raise ValueError(f"cannot parse expression from {obj!r}")


def _parse_arg(a: Any) -> Any:
    if isinstance(a, dict):
        return parse_expr(a)
    return a


def true_expr() -> Expr:
    return Expr("const", (True,))


# ---------------------------------------------------------------------------
# Effect — one safe universal world operation with (possibly bound) parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    """A single universal world operation. ``op`` is one of the fixed effect ops in
    :mod:`effects`; ``params`` are literals or binding strings resolved at execution
    time (``$actor``, ``$target``, ``$param.<name>``, ``$self.<attr>``).

    An action's *behavior* is fully determined by its effects, not by its name.
    """

    op: str
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)


def parse_effect(obj: dict[str, Any]) -> Effect:
    op = str(obj["op"])
    params = {k: v for k, v in obj.items() if k != "op"}
    return Effect(op=op, params=tuple(sorted(params.items())))


def parse_effects(items: Any) -> tuple[Effect, ...]:
    return tuple(parse_effect(o) for o in (items or []))


# ---------------------------------------------------------------------------
# Entities and actors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySpec:
    """Any entity in the world: a person, organization, object, document, resource,
    or channel. ``is_actor`` marks the ones that perceive and act. ``authority`` is a
    free set of capability tokens (e.g. ``"vote"``, ``"sign_treaty"``, ``"reply"``)
    that actions may require — the tokens are compiled from evidence, not enumerated
    in code."""

    entity_id: str
    name: str
    kind: str  # "person" | "organization" | "object" | "document" | "channel" | ...
    is_actor: bool = False
    role: str = ""
    authority: tuple[str, ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()  # generic typed facts (e.g. ("weight", 40))
    evidence_claim_ids: tuple[str, ...] = ()

    @property
    def attributes_dict(self) -> dict[str, Any]:
        return dict(self.attributes)


@dataclass(frozen=True)
class ActorPolicyRule:
    """A compiled, deterministic behavioral rule the offline reasoner executes: when
    an observed field crosses a threshold (or equals a value), the actor takes a
    specific compiled/novel action. This generalizes reaction rules with no notion of
    "vote" or "option". The live LLM actor decides freely; this only makes the
    deterministic path reproducible."""

    when_field: str
    op: str  # "above" | "below" | "equals" | "present"
    value: Any
    action_id: str = ""  # "" -> wait
    params: tuple[tuple[str, Any], ...] = ()
    novel: tuple[tuple[str, Any], ...] = ()  # optional novel-action proposal instead of action_id

    @property
    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def novel_dict(self) -> dict[str, Any]:
        return dict(self.novel)


@dataclass(frozen=True)
class ActorPolicy:
    """The compiled disposition of an actor for the deterministic path: an ordered
    rule list (first match wins) and a default action taken when it is this actor's
    turn and no rule fires."""

    default_action_id: str = ""  # "" -> wait
    default_params: tuple[tuple[str, Any], ...] = ()
    default_novel: tuple[tuple[str, Any], ...] = ()
    rules: tuple[ActorPolicyRule, ...] = ()

    @property
    def default_params_dict(self) -> dict[str, Any]:
        return dict(self.default_params)

    @property
    def default_novel_dict(self) -> dict[str, Any]:
        return dict(self.default_novel)


@dataclass(frozen=True)
class ActorSpec:
    """An entity that acts, plus its compiled disposition and seed memories."""

    entity_id: str
    policy: ActorPolicy
    memory_seeds: tuple[tuple[tuple[str, Any], ...], ...] = ()  # each seed: dict-as-sorted-items
    reasoning: str = ""


# ---------------------------------------------------------------------------
# World scaffolding: fields, resources, channels, documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    value_type: str  # "number" | "string" | "bool"
    initial: Any = None
    description: str = ""
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceSpec:
    resource_id: str
    holder_entity_id: str
    quantity: float
    description: str = ""


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: str
    description: str = ""
    participants: tuple[str, ...] = ()  # entity ids or roles


@dataclass(frozen=True)
class DocumentSpec:
    document_id: str
    fields: tuple[tuple[str, Any], ...] = ()
    description: str = ""


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    name: str
    value_type: str = "string"  # "string" | "number" | "option" | "entity"
    required: bool = False
    choices: tuple[Any, ...] = ()  # for "option": the valid values


@dataclass(frozen=True)
class ActionDefinition:
    """A scenario-specific action the compiler discovered, mapped to safe universal
    effects. Its *meaning is its effects* — renaming ``action_id`` while keeping
    ``effects`` unchanged does not change what it does.

    ``eligible_actors`` may list entity ids, ``role:<role>`` selectors, or ``"*"``.
    ``required_authority`` are capability tokens the actor must hold. ``preconditions``
    is a declarative :class:`Expr` over world state that must be true. ``resource_costs``
    are (resource_id, amount) that must be available and are consumed. Timing gates the
    action to stages / a time window. ``valid_targets`` optionally restricts targets.
    """

    action_id: str
    meaning: str
    eligible_actors: tuple[str, ...] = ("*",)
    required_authority: tuple[str, ...] = ()
    parameters: tuple[ParamSpec, ...] = ()
    preconditions: Expr = field(default_factory=true_expr)
    resource_costs: tuple[tuple[str, float], ...] = ()
    stages: tuple[str, ...] = ()  # if set, action available only in these node stages
    not_before: Any = None  # ISO datetime string or None
    not_after: Any = None
    valid_targets: tuple[str, ...] = ()  # entity ids/roles/"*" (empty -> no target)
    visibility: str = "public"  # public | private | role
    effects: tuple[Effect, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()

    def eligible(self, actor_role: str, actor_id: str) -> bool:
        for sel in self.eligible_actors:
            if sel == "*" or sel == actor_id:
                return True
            if sel.startswith("role:") and sel[5:] == actor_role:
                return True
        return False


# ---------------------------------------------------------------------------
# Process graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessNode:
    """One moment in the compiled process. The runtime, on reaching a node:

    1. skips it if ``condition`` is false;
    2. advances time by ``advance_seconds`` (and to ``at`` if given);
    3. applies environment ``effects`` (briefings, data releases, scheduled fires);
    4. for each of ``rounds``, lets every participant act, offering the compiled
       actions in ``action_ids`` (``"*"`` = all this actor is eligible for) plus, if
       ``allow_novel``, a novel-action proposal.

    ``participants`` are entity ids, ``role:<role>`` selectors, or ``"*"`` (all
    actors). ``stage`` is a free label the compiler chooses; it gates action timing
    and is shown to actors. Nothing here says "committee" or "vote"."""

    node_id: str
    description: str = ""
    stage: str = ""
    condition: Expr = field(default_factory=true_expr)
    advance_seconds: int = 0
    at: Any = None  # ISO datetime string or None
    effects: tuple[Effect, ...] = ()
    participants: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ("*",)
    allow_novel: bool = True
    rounds: int = 1


@dataclass(frozen=True)
class ProcessGraph:
    nodes: tuple[ProcessNode, ...] = ()

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self.nodes)


# ---------------------------------------------------------------------------
# Terminal condition (declarative — no fixed families)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalExpression:
    """The declarative resolution condition compiled per question.

    ``yes_when`` is a boolean :class:`Expr` over the final world state; ``unresolved_when``
    (default: never) marks states where the process did not actually determine the
    answer (e.g. required records missing), which are reported as *unresolved* rather
    than forced to NO. The evaluator hardcodes only universal operators."""

    yes_when: Expr
    unresolved_when: Expr = field(default_factory=lambda: Expr("const", (False,)))
    description: str = ""


# ---------------------------------------------------------------------------
# The full compiled world
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionChoice:
    """What an actor decides to attempt on its turn. ``mode`` is ``compiled_action``
    (pick one of the feasible compiled actions), ``novel_action`` (propose something
    the compiler did not anticipate — routed through validation, never auto-executed),
    or ``wait``. The actor states an *intention*; the environment produces the
    consequence."""

    mode: str  # "compiled_action" | "novel_action" | "wait"
    action_id: str = ""
    params: tuple[tuple[str, Any], ...] = ()
    target: str = ""
    novel_description: str = ""
    novel_intended_effect: str = ""
    novel_target: str = ""
    novel_params: tuple[tuple[str, Any], ...] = ()
    rationale: str = ""
    referenced_memory_ids: tuple[str, ...] = ()
    referenced_observation_ids: tuple[str, ...] = ()

    @property
    def params_dict(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def novel_params_dict(self) -> dict[str, Any]:
        return dict(self.novel_params)


@dataclass(frozen=True)
class WorldSpec:
    """Everything the LLM compiles for one arbitrary question."""

    title: str
    entities: tuple[EntitySpec, ...]
    actors: tuple[ActorSpec, ...]
    fields: tuple[FieldSpec, ...]
    resources: tuple[ResourceSpec, ...]
    channels: tuple[ChannelSpec, ...]
    documents: tuple[DocumentSpec, ...]
    actions: tuple[ActionDefinition, ...]
    process: ProcessGraph
    terminal: TerminalExpression
    subject_entity: str = ""
    resolution_units: str = ""

    def action(self, action_id: str) -> ActionDefinition | None:
        for a in self.actions:
            if a.action_id == action_id:
                return a
        return None

    def actor_entities(self) -> tuple[EntitySpec, ...]:
        actor_ids = {a.entity_id for a in self.actors}
        return tuple(e for e in self.entities if e.entity_id in actor_ids or e.is_actor)


# ---------------------------------------------------------------------------
# JSON parsing (corpus-authored or LLM-compiled). Tolerant of missing keys.
# ---------------------------------------------------------------------------


def _items(d: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(d.items())) if isinstance(d, dict) else ()


def _strs(v: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in v) if isinstance(v, list) else ()


def parse_entity(d: dict[str, Any]) -> EntitySpec:
    return EntitySpec(
        entity_id=str(d["entity_id"]),
        name=str(d.get("name", d["entity_id"])),
        kind=str(d.get("kind", "person")),
        is_actor=bool(d.get("is_actor", False)),
        role=str(d.get("role", "")),
        authority=_strs(d.get("authority")),
        attributes=_items(d.get("attributes")),
        evidence_claim_ids=_strs(d.get("evidence_claim_ids")),
    )


def parse_actor(d: dict[str, Any]) -> ActorSpec:
    p = d.get("policy") or {}
    rules = tuple(
        ActorPolicyRule(
            when_field=str(r.get("when_field", "")),
            op=str(r.get("op", "present")),
            value=r.get("value"),
            action_id=str(r.get("action_id", "")),
            params=_items(r.get("params")),
            novel=_items(r.get("novel")),
        )
        for r in (p.get("rules") or [])
    )
    return ActorSpec(
        entity_id=str(d["entity_id"]),
        policy=ActorPolicy(
            default_action_id=str(p.get("default_action_id", "")),
            default_params=_items(p.get("default_params")),
            default_novel=_items(p.get("default_novel")),
            rules=rules,
        ),
        memory_seeds=tuple(_items(s) for s in (d.get("memory_seeds") or [])),
        reasoning=str(d.get("reasoning", "")),
    )


def parse_action(d: dict[str, Any]) -> ActionDefinition:
    return ActionDefinition(
        action_id=str(d["action_id"]),
        meaning=str(d.get("meaning", "")),
        eligible_actors=_strs(d.get("eligible_actors")) or ("*",),
        required_authority=_strs(d.get("required_authority")),
        parameters=tuple(
            ParamSpec(
                name=str(p["name"]),
                value_type=str(p.get("type", p.get("value_type", "string"))),
                required=bool(p.get("required", False)),
                choices=tuple(p.get("choices", []) or []),
            )
            for p in (d.get("parameters") or [])
        ),
        preconditions=parse_expr(d["preconditions"]) if d.get("preconditions") else true_expr(),
        resource_costs=tuple((str(rc[0]), float(rc[1])) for rc in (d.get("resource_costs") or [])),
        stages=_strs(d.get("stages")),
        not_before=d.get("not_before"),
        not_after=d.get("not_after"),
        valid_targets=_strs(d.get("valid_targets")),
        visibility=str(d.get("visibility", "public")),
        effects=parse_effects(d.get("effects")),
        evidence_claim_ids=_strs(d.get("evidence_claim_ids")),
    )


def parse_node(d: dict[str, Any]) -> ProcessNode:
    return ProcessNode(
        node_id=str(d["node_id"]),
        description=str(d.get("description", "")),
        stage=str(d.get("stage", "")),
        condition=parse_expr(d["condition"]) if d.get("condition") else true_expr(),
        advance_seconds=int(d.get("advance_seconds", 0)),
        at=d.get("at"),
        effects=parse_effects(d.get("effects")),
        participants=_strs(d.get("participants")),
        action_ids=_strs(d.get("action_ids")) or ("*",),
        allow_novel=bool(d.get("allow_novel", True)),
        rounds=int(d.get("rounds", 1)),
    )


def parse_terminal(d: dict[str, Any]) -> TerminalExpression:
    return TerminalExpression(
        yes_when=parse_expr(d["yes_when"]),
        unresolved_when=(
            parse_expr(d["unresolved_when"])
            if d.get("unresolved_when")
            else Expr("const", (False,))
        ),
        description=str(d.get("description", "")),
    )


def parse_world_spec(d: dict[str, Any]) -> WorldSpec:
    return WorldSpec(
        title=str(d.get("title", "")),
        entities=tuple(parse_entity(e) for e in d.get("entities", [])),
        actors=tuple(parse_actor(a) for a in d.get("actors", [])),
        fields=tuple(
            FieldSpec(
                field_id=str(f["field_id"]),
                value_type=str(f.get("value_type", "number")),
                initial=f.get("initial"),
                description=str(f.get("description", "")),
                evidence_claim_ids=_strs(f.get("evidence_claim_ids")),
            )
            for f in d.get("fields", [])
        ),
        resources=tuple(
            ResourceSpec(
                resource_id=str(r["resource_id"]),
                holder_entity_id=str(r["holder_entity_id"]),
                quantity=float(r.get("quantity", 0.0)),
                description=str(r.get("description", "")),
            )
            for r in d.get("resources", [])
        ),
        channels=tuple(
            ChannelSpec(
                channel_id=str(c["channel_id"]),
                description=str(c.get("description", "")),
                participants=_strs(c.get("participants")),
            )
            for c in d.get("channels", [])
        ),
        documents=tuple(
            DocumentSpec(
                document_id=str(doc["document_id"]),
                fields=_items(doc.get("fields")),
                description=str(doc.get("description", "")),
            )
            for doc in d.get("documents", [])
        ),
        actions=tuple(parse_action(a) for a in d.get("actions", [])),
        process=ProcessGraph(
            nodes=tuple(parse_node(n) for n in d.get("process", {}).get("nodes", []))
        ),
        terminal=parse_terminal(d["terminal"]),
        subject_entity=str(d.get("subject_entity", "")),
        resolution_units=str(d.get("resolution_units", "")),
    )
