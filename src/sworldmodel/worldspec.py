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
        # `{"op": "true"}` / `{"op": "false"}` are the natural way to write a constant
        # and are not operators. Reading them as one is shape coercion, not invention;
        # leaving them raised "unknown expression operator 'false'" from inside the
        # evaluator, while finalizing a branch, after the whole run had been paid for.
        if str(obj.get("op", "")).lower() in ("true", "false"):
            return Expr("const", (str(obj["op"]).lower() == "true",))
        if "op" in obj:
            # `args` is a list by schema, and a model will nonetheless sometimes write
            # the single argument bare: {"op": "const", "args": false}. Iterating that
            # raises TypeError from inside a parser, which killed a live run before it
            # could write any diagnosis at all. A lone argument is a one-argument list;
            # reading it as one changes no meaning and costs nothing.
            raw_args = obj.get("args", [])
            if raw_args is None:
                raw_args = []
            elif not isinstance(raw_args, (list, tuple)):
                raw_args = [raw_args]
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
# Structural monotonicity of an expression
# ---------------------------------------------------------------------------
#
# WHY THIS IS COMPUTED AND NEVER DECLARED. The runtime has one guard that refuses to
# report a resolved outcome for a branch that stopped before its schedule was empty:
# reporting one would claim we watched the process finish when we stopped watching.
# That reasoning is correct for `field('deliveries') > 400000` — more simulation
# genuinely could move the number — and wrong for `event_count(X) > 0`, where the
# events already happened and no further simulation can un-happen them. The whole
# distinction is *monotonicity*, and the only safe place to get it from is the shape of
# the expression itself. A `monotone: true` flag on a compiled world would be the same
# shape as `terminal_sensitivity` (FD-36): self-declared by the planner, checked in one
# direction, and evaded by declaring the value that nothing tests.
#
# The direction of a sub-expression is how its VALUE may move as the world evolves:
#
#   MONOTONE_UP      never decreases (for booleans: False may become True, never back)
#   MONOTONE_DOWN    never increases
#   MONOTONE_FIXED   never changes
#   MONOTONE_UNKNOWN anything
#
# Only two things in this world model are structurally append-only: the record
# collections (``world.apply`` only ever ``.append``s to them) and the event history
# (an applied event is never removed and its status never leaves APPLIED). Everything
# else — fields, resources, document fields, the stage, the clock — is rewritable, so
# every expression that reads one is UNKNOWN and keeps the conservative behaviour.
#
# Deliberate carve-outs, each of which costs only conservatism:
#   * ``sum`` is UNKNOWN even over an append-only collection: a negative summand makes
#     an append-only total fall, and nothing here can know the sign.
#   * ``multiply``/``divide``/``abs`` are UNKNOWN: a sign flip inverts the direction.
#   * ``now``/``stage`` are UNKNOWN, so the temporal operators are too. The clock does
#     advance monotonically, but nothing this analysis exists for needs it, and a
#     direction that has to be right for banking to be sound is not worth widening for
#     a case nobody has.
#   * ``equals``/``not_equals``/``contains`` are UNKNOWN: a count passing through 3
#     satisfies ``equals(count, 3)`` on the way and falsifies it immediately after.

MONOTONE_UP = "non_decreasing"
MONOTONE_DOWN = "non_increasing"
MONOTONE_FIXED = "fixed"
MONOTONE_UNKNOWN = "unknown"

# Operators that read state the world may rewrite. A ``where`` predicate containing one
# is not a property of the record in front of it — it can disqualify records that
# already matched — so the aggregate over it is not append-only either.
_MUTABLE_READ_OPS = frozenset(
    {
        "field",
        "stage",
        "now",
        "resource",
        "document_field",
        "count",
        "sum",
        "values",
        "exists",
        "event_count",
    }
)

# The operators a ``where`` predicate may use and still be a pure function of one
# record. A whitelist, not a blacklist: an operator this analysis has never heard of
# must not be assumed harmless.
_RECORD_LOCAL_OPS = frozenset(
    {
        "const",
        "item",
        "horizon",
        "as_of",
        "equals",
        "not_equals",
        "greater_than",
        "less_than",
        "greater_or_equal",
        "less_or_equal",
        "contains",
        "and",
        "or",
        "not",
        "all",
        "any",
        "before",
        "after",
        "duration",
        "add",
        "subtract",
        "multiply",
        "divide",
        "min",
        "max",
        "abs",
        "round",
    }
)

# The append-only aggregates, and the term namespace each one grows out of. The
# namespace matches ``world_compiler._expr_terms`` so a banked cause and the
# compile-time producer gate name the same thing the same way.
_APPEND_ONLY_AGGREGATES = {"count": "collection", "exists": "collection", "event_count": "event"}


def expression_monotonicity(expr: Any) -> str:
    """Which way this expression's value can move as the world evolves.

    Total and conservative: anything not proven to move in one direction is
    ``MONOTONE_UNKNOWN``. A bare literal (an ``args`` entry that is not an
    :class:`Expr`) is ``MONOTONE_FIXED``.
    """

    if not isinstance(expr, Expr):
        return MONOTONE_FIXED
    op, args = expr.op, expr.args

    if op in ("const", "horizon", "as_of"):
        return MONOTONE_FIXED

    if op in _APPEND_ONLY_AGGREGATES:
        # The collection/event-type being aggregated must be a compile-time literal.
        # ``count(field('which_list'))`` counts a DIFFERENT list once the field moves,
        # so its growth is not the collection's growth.
        if not args or _literal_name(args[0]) is None:
            return MONOTONE_UNKNOWN
        where = args[1] if len(args) > 1 else None
        return MONOTONE_UP if _record_local(where) else MONOTONE_UNKNOWN

    # ``and``/``or`` are monotone INCREASING in every argument, so they carry a common
    # direction through and destroy a mixed one.
    if op in ("and", "or"):
        return _combine([expression_monotonicity(a) for a in args])

    if op == "not":
        return _invert(expression_monotonicity(args[0])) if args else MONOTONE_UNKNOWN

    # Comparison: increasing in the side that must be larger, decreasing in the other.
    # This is where direction is won or lost — ``count(x) > 0`` is monotone toward YES
    # and ``count(x) < 3`` is monotone toward NO, and they differ only here.
    if op in ("greater_than", "greater_or_equal"):
        if len(args) < 2:
            return MONOTONE_UNKNOWN
        return _combine(
            [expression_monotonicity(args[0]), _invert(expression_monotonicity(args[1]))]
        )
    if op in ("less_than", "less_or_equal"):
        if len(args) < 2:
            return MONOTONE_UNKNOWN
        return _combine(
            [_invert(expression_monotonicity(args[0])), expression_monotonicity(args[1])]
        )

    # Arithmetic that cannot flip a direction whatever the operands' signs.
    if op in ("add", "min", "max"):
        return _combine([expression_monotonicity(a) for a in args]) if args else MONOTONE_UNKNOWN
    if op == "subtract":
        if not args:
            return MONOTONE_UNKNOWN
        return _combine(
            [expression_monotonicity(args[0])]
            + [_invert(expression_monotonicity(a)) for a in args[1:]]
        )

    return MONOTONE_UNKNOWN


def monotone_sources(expr: Any) -> tuple[str, ...]:
    """The append-only terms an expression's growth actually rests on.

    Namespaced ``collection:<name>`` / ``event:<type>``, matching the compiler's term
    vocabulary. Only aggregates this analysis proved append-only are listed, so the
    result is exactly the set of things a caller can go looking for a cause in.
    """

    out: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, Expr):
            return
        space = _APPEND_ONLY_AGGREGATES.get(node.op)
        if space is not None and expression_monotonicity(node) == MONOTONE_UP:
            name = _literal_name(node.args[0])
            if name:
                out.add(f"{space}:{name}")
        for arg in node.args:
            walk(arg)

    walk(expr)
    return tuple(sorted(out))


def _literal_name(arg: Any) -> str | None:
    """The compile-time string an argument names, or None if the world decides it."""

    if isinstance(arg, str):
        return arg or None
    if isinstance(arg, Expr) and arg.op == "const" and arg.args:
        value = arg.args[0]
        return str(value) if isinstance(value, str) and value else None
    return None


def _record_local(where: Any) -> bool:
    """Whether a ``where`` predicate reads only the record in front of it."""

    if where is None or not isinstance(where, Expr):
        return True  # absent, or a bare literal
    if where.op in _MUTABLE_READ_OPS or where.op not in _RECORD_LOCAL_OPS:
        return False
    return all(_record_local(a) for a in where.args)


def _combine(directions: list[str]) -> str:
    """The direction of a function that is monotone increasing in every argument."""

    if any(d == MONOTONE_UNKNOWN for d in directions):
        return MONOTONE_UNKNOWN
    if all(d == MONOTONE_FIXED for d in directions):
        return MONOTONE_FIXED
    if all(d in (MONOTONE_UP, MONOTONE_FIXED) for d in directions):
        return MONOTONE_UP
    if all(d in (MONOTONE_DOWN, MONOTONE_FIXED) for d in directions):
        return MONOTONE_DOWN
    return MONOTONE_UNKNOWN


def _invert(direction: str) -> str:
    if direction == MONOTONE_UP:
        return MONOTONE_DOWN
    if direction == MONOTONE_DOWN:
        return MONOTONE_UP
    return direction


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


# The canonical parameter name each effect op reads, and the synonyms a model reaches
# for. The effect schema names the ops but cannot spell out every op's params, so the
# compiler guesses — a live Bank of England run emitted `set_field` keyed on `field_id`
# (the key it had just seen on field *definitions*) rather than `field`, so the effect
# wrote nothing the gate could see and the world was refused for an action that did
# nothing. Renaming a synonym to the canonical key is shape coercion, not content
# invention: the field the effect names does not change, only the key it is under.
_EFFECT_KEY_SYNONYMS: dict[str, dict[str, str]] = {
    "set_field": {"field_id": "field", "target": "field", "name": "field", "key": "field"},
    "adjust_field": {"field_id": "field", "target": "field", "name": "field", "key": "field"},
    "append_record": {"record": "collection", "list": "collection"},
    "create_event": {"type": "event_type", "kind": "event_type"},
    "schedule_event": {"type": "event_type", "kind": "event_type"},
}


def parse_effect(obj: dict[str, Any]) -> Effect:
    op = str(obj.get("op", ""))
    params = {k: v for k, v in obj.items() if k != "op"}
    synonyms = _EFFECT_KEY_SYNONYMS.get(op, {})
    for alias, target in synonyms.items():
        # Only fill in the canonical key from a synonym when it is genuinely absent, so
        # an effect that already names it correctly is never disturbed.
        if alias in params and target not in params:
            params[target] = params.pop(alias)
    return Effect(op=op, params=tuple(sorted(params.items())))


def parse_effects(items: Any) -> tuple[Effect, ...]:
    """Every effect the compiler wrote that names an operation.

    An entry with no ``op`` names no world operation and cannot be executed, so it is
    dropped rather than crashing the parse. That is safe precisely because it is
    visible downstream: if the dropped effect was the only producer of a terminal term,
    the producer-lineage gate refuses the world and says so.
    """

    return tuple(parse_effect(o) for o in as_objects(items) if str(o.get("op", "")).strip())


# ---------------------------------------------------------------------------
# Entities and actors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySpec:
    """Any entity in the world: a person, organization, object, document, resource,
    or channel. ``is_actor`` marks the ones that perceive and act. ``authority`` is a
    free set of capability tokens (e.g. ``"vote"``, ``"sign_treaty"``, ``"reply"``)
    that actions may require — the tokens are compiled from evidence, not enumerated
    in code.

    ``representation_scale`` is the compiler's explicit choice of *what level of thing
    this is*: one person, an organization acting as a unit, a subunit, a stratum of a
    population, a network, or a non-agent process. Choosing it wrongly is a modeling
    error the trace should be able to show, so it is recorded rather than implied.
    ``represents_count`` says how many real units this entity stands for, so an
    aggregate can never silently masquerade as a single decision-maker.
    """

    entity_id: str
    name: str
    kind: str  # "person" | "organization" | "object" | "document" | "channel" | ...
    is_actor: bool = False
    role: str = ""
    authority: tuple[str, ...] = ()
    attributes: tuple[tuple[str, Any], ...] = ()  # generic typed facts (e.g. ("weight", 40))
    evidence_claim_ids: tuple[str, ...] = ()
    representation_scale: str = "individual"
    represents_count: int | None = None

    @property
    def attributes_dict(self) -> dict[str, Any]:
        return dict(self.attributes)


@dataclass(frozen=True)
class ActorSpec:
    """An entity that acts: its evidence-grounded starting state.

    There is no compiled behavioral policy and no default action. An actor's behavior
    comes from the model that plays it; if that model cannot be reached, the branch's
    mass stays unresolved. A compiled ``default_action_id`` would be exactly the
    "deterministic stand-in wearing the actor's name" this runtime exists to prevent.

    The initial plan is *grounded*, not invented: ``initial_plan_basis`` must say which
    verified schedule, role obligation or existing commitment makes it admissible, and
    a sparse grounded plan is preferred over a detailed fictional one.
    """

    entity_id: str
    memory_seeds: tuple[tuple[tuple[str, Any], ...], ...] = ()  # each seed: dict-as-sorted-items
    reasoning: str = ""
    goals: tuple[str, ...] = ()
    initial_plan_goal: str = ""
    initial_plan_steps: tuple[tuple[tuple[str, Any], ...], ...] = ()
    initial_plan_basis: str = ""
    initial_plan_evidence_ids: tuple[str, ...] = ()
    initial_commitments: tuple[tuple[tuple[str, Any], ...], ...] = ()

    def initial_plan_steps_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(s) for s in self.initial_plan_steps)

    def initial_commitment_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(c) for c in self.initial_commitments)


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
    """A document or object the world contains.

    ``evidence_claim_ids`` cite the verified claims that establish its initial contents.
    A document whose fields are already true when the window opens — an agreement the
    record shows was signed in January, asked about in July — is only admissible with
    them: the citation is what separates a fact the world starts from and a fact the
    compiler asserted.
    """

    document_id: str
    fields: tuple[tuple[str, Any], ...] = ()
    description: str = ""
    evidence_claim_ids: tuple[str, ...] = ()


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

    Taking an action is not the same as the action *happening*. ``duration_seconds``
    is how long it takes; its effects land at completion, not at the moment of
    intention. ``completion_conditions`` are re-checked at completion — an action begun
    in a world that has since changed can fail, and it fails visibly rather than being
    applied to a world its actor never saw. ``delivery_delay_seconds`` and
    ``notice_delay_seconds`` separate a consequence occurring from it reaching someone
    and from that person actually noticing it.
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
    duration_seconds: int = 0
    completion_conditions: Expr = field(default_factory=true_expr)
    delivery_delay_seconds: int = 0
    notice_delay_seconds: int = 0
    observers: tuple[str, ...] = ()  # who may observe the result; empty -> from visibility

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
    """One scheduled moment of the compiled process, placed on the real calendar.

    A node is *not* a turn. When its scheduled time arrives the runtime checks
    ``entry_condition``; if true it applies the environment ``effects`` and sets the
    ``stage``, and the participants named here gain a decision **opportunity** — a
    reason to be woken, which the engine turns into an actor invocation only if the
    actor is actually free and affected. Nothing here calls anybody a fixed number of
    times, and there is deliberately no ``rounds`` field: how often an actor acts is
    decided by what happens to it.

    Timing is either absolute (``at``) or relative to another node's completion
    (``after_node`` + ``delay_seconds``). ``next_nodes`` are entered when this node
    completes, so a compiled process is a graph on the calendar rather than a list the
    runtime walks. ``deadline`` marks a real cutoff that itself wakes participants.

    ``participants`` are entity ids, ``role:<role>`` selectors, or ``"*"``. ``stage``
    is a free label the compiler chooses; it gates action feasibility and is shown to
    actors. Nothing here says "committee", "vote", or names any kind of question.
    """

    node_id: str
    description: str = ""
    stage: str = ""
    entry_condition: Expr = field(default_factory=true_expr)
    at: Any = None  # ISO datetime string
    after_node: str = ""  # schedule relative to another node's completion
    delay_seconds: int = 0
    effects: tuple[Effect, ...] = ()
    participants: tuple[str, ...] = ()
    action_ids: tuple[str, ...] = ("*",)
    allow_novel: bool = True
    deadline: Any = None  # ISO datetime string; wakes participants when reached
    next_nodes: tuple[str, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessGraph:
    nodes: tuple[ProcessNode, ...] = ()

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self.nodes)

    def node(self, node_id: str) -> ProcessNode | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def roots(self) -> tuple[ProcessNode, ...]:
        """Nodes that are not entered by another node, i.e. those the runtime seeds the
        schedule with. Everything else is reached causally."""

        entered = {nid for n in self.nodes for nid in n.next_nodes}
        return tuple(n for n in self.nodes if n.node_id not in entered and not n.after_node)


@dataclass(frozen=True)
class ExternalOccurrence:
    """One happening of a non-agent process, at an exact time."""

    at: Any  # ISO datetime string
    description: str = ""
    effects: tuple[Effect, ...] = ()
    condition: Expr = field(default_factory=true_expr)


@dataclass(frozen=True)
class ExternalProcess:
    """A causally relevant part of the world that is not an actor: a scheduled data
    release, a market or administrative clock, a delivery system, a legal deadline, a
    publication cycle.

    These evolve through typed world events on their own schedule. Compiling them is
    what makes it unnecessary to invent an LLM "actor" whose only job is to make the
    weather happen.
    """

    process_id: str
    description: str = ""
    occurrences: tuple[ExternalOccurrence, ...] = ()
    evidence_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class WakeRule:
    """A compiled, scenario-specific reason for an actor to be brought back.

    The compiler discovers *what matters to whom* in this particular world; the runtime
    owns when it fires and enforces it. Only mechanically checkable forms exist, so no
    relevance score is ever invented.
    """

    rule_id: str
    wakes: tuple[str, ...] = ()  # entity ids / role: selectors / "*"
    reason: str = ""
    on_record_in: str = ""  # a record appended to this collection
    on_field_change: str = ""  # this world field changed
    on_event_type: str = ""  # a create_event with this event_type
    on_information_from: str = ""  # this actor communicated
    evidence_claim_ids: tuple[str, ...] = ()

    def is_checkable(self) -> bool:
        return bool(
            self.on_record_in
            or self.on_field_change
            or self.on_event_type
            or self.on_information_from
        )


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
    external_processes: tuple[ExternalProcess, ...] = ()
    wake_rules: tuple[WakeRule, ...] = ()
    structure_id: str = "primary"
    structure_rationale: str = ""

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
    if isinstance(v, (list, tuple)):
        return tuple(str(x) for x in v)
    # A single value written bare where a list belongs is that list with one entry.
    if isinstance(v, str) and v.strip():
        return (v,)
    return ()


def _number(value: Any, default: float = 0.0) -> float:
    """A quantity the compiler wrote, or the default when it wrote nothing usable.

    ``d.get("quantity", 0.0)`` returns ``None`` when the key is present and null, which
    is a shape a model produces routinely. A live OPEC+ run died on it — ``float(None)``
    inside the parser, during a repair recompile, past every gate that would have turned
    it into a diagnosis, leaving a traceback and no artifacts at all.
    """

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pairs(value: Any) -> tuple[tuple[str, float], ...]:
    """``[[resource, amount], ...]`` — keeping only the entries that are actually pairs.

    A half-written cost names no amount, and unpacking it raised IndexError from inside
    the parser. Dropping it is safe and visible: an action whose cost was dropped is
    cheaper than intended, and the executor still refuses it if the resource is absent.
    """

    out: list[tuple[str, float]] = []
    for item in value if isinstance(value, (list, tuple)) else ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            out.append((str(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def as_objects(value: Any) -> list[dict[str, Any]]:
    """Read a list-of-objects field however the compiler happened to write it.

    A model asked for a list of objects will sometimes emit one bare object, ``null``,
    or a string. Each of those crashed a parser with ``TypeError`` or ``AttributeError``
    — a live run ending in a stack trace, before any diagnosis could be written, because
    of punctuation.

    This coerces *shape* and never invents *content*: a lone object becomes a
    one-element list, an absent field becomes an empty one, and anything that is not an
    object is dropped rather than guessed at. What was not said stays unsaid, and the
    integrity gates still see exactly what the compiler actually produced.
    """

    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [x for x in value if isinstance(x, dict)]
    return []


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
        representation_scale=str(d.get("representation_scale", "individual")),
        represents_count=(
            int(d["represents_count"]) if d.get("represents_count") is not None else None
        ),
    )


def parse_actor(d: dict[str, Any]) -> ActorSpec:
    plan = d.get("initial_plan") or {}
    return ActorSpec(
        entity_id=str(d["entity_id"]),
        memory_seeds=tuple(_items(s) for s in (d.get("memory_seeds") or [])),
        reasoning=str(d.get("reasoning", "")),
        goals=_strs(d.get("goals")),
        initial_plan_goal=str(plan.get("goal", "")),
        initial_plan_steps=tuple(_items(s) for s in (plan.get("steps") or [])),
        initial_plan_basis=str(plan.get("basis", "")),
        initial_plan_evidence_ids=_strs(plan.get("evidence_claim_ids")),
        initial_commitments=tuple(_items(c) for c in (d.get("commitments") or [])),
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
        resource_costs=_pairs(d.get("resource_costs")),
        stages=_strs(d.get("stages")),
        not_before=d.get("not_before"),
        not_after=d.get("not_after"),
        valid_targets=_strs(d.get("valid_targets")),
        visibility=str(d.get("visibility", "public")),
        effects=parse_effects(d.get("effects")),
        evidence_claim_ids=_strs(d.get("evidence_claim_ids")),
        duration_seconds=int(d.get("duration_seconds", 0) or 0),
        completion_conditions=(
            parse_expr(d["completion_conditions"])
            if d.get("completion_conditions")
            else true_expr()
        ),
        delivery_delay_seconds=int(d.get("delivery_delay_seconds", 0) or 0),
        notice_delay_seconds=int(d.get("notice_delay_seconds", 0) or 0),
        observers=_strs(d.get("observers")),
    )


def parse_node(d: dict[str, Any]) -> ProcessNode:
    cond = d.get("entry_condition") or d.get("condition")
    return ProcessNode(
        node_id=str(d["node_id"]),
        description=str(d.get("description", "")),
        stage=str(d.get("stage", "")),
        entry_condition=parse_expr(cond) if cond else true_expr(),
        at=d.get("at"),
        after_node=str(d.get("after_node", "")),
        delay_seconds=int(d.get("delay_seconds", 0) or 0),
        effects=parse_effects(d.get("effects")),
        participants=_strs(d.get("participants")),
        action_ids=_strs(d.get("action_ids")) or ("*",),
        allow_novel=bool(d.get("allow_novel", True)),
        deadline=d.get("deadline"),
        next_nodes=_strs(d.get("next_nodes")),
        evidence_claim_ids=_strs(d.get("evidence_claim_ids")),
    )


def parse_external_process(d: dict[str, Any]) -> ExternalProcess:
    return ExternalProcess(
        process_id=str(d["process_id"]),
        description=str(d.get("description", "")),
        occurrences=tuple(
            ExternalOccurrence(
                at=o.get("at"),
                description=str(o.get("description", "")),
                effects=parse_effects(o.get("effects")),
                condition=parse_expr(o["condition"]) if o.get("condition") else true_expr(),
            )
            for o in (d.get("occurrences") or [])
        ),
        evidence_claim_ids=_strs(d.get("evidence_claim_ids")),
    )


def parse_wake_rule(d: dict[str, Any]) -> WakeRule:
    return WakeRule(
        rule_id=str(d.get("rule_id", "")),
        wakes=_strs(d.get("wakes")),
        reason=str(d.get("reason", "")),
        on_record_in=str(d.get("on_record_in", "")),
        on_field_change=str(d.get("on_field_change", "")),
        on_event_type=str(d.get("on_event_type", "")),
        on_information_from=str(d.get("on_information_from", "")),
        evidence_claim_ids=_strs(d.get("evidence_claim_ids")),
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


def _process_block(value: Any) -> dict[str, Any]:
    """The process graph, whether written as {"nodes": [...]} or as the bare node list."""

    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {"nodes": list(value)}
    return {}


def parse_world_spec(d: dict[str, Any]) -> WorldSpec:
    return WorldSpec(
        title=str(d.get("title", "")),
        entities=tuple(parse_entity(e) for e in as_objects(d.get("entities"))),
        actors=tuple(parse_actor(a) for a in as_objects(d.get("actors"))),
        fields=tuple(
            FieldSpec(
                field_id=str(f["field_id"]),
                value_type=str(f.get("value_type", "number")),
                initial=f.get("initial"),
                description=str(f.get("description", "")),
                evidence_claim_ids=_strs(f.get("evidence_claim_ids")),
            )
            for f in as_objects(d.get("fields"))
        ),
        resources=tuple(
            ResourceSpec(
                resource_id=str(r["resource_id"]),
                holder_entity_id=str(r["holder_entity_id"]),
                quantity=_number(r.get("quantity")),
                description=str(r.get("description", "")),
            )
            for r in as_objects(d.get("resources"))
        ),
        channels=tuple(
            ChannelSpec(
                channel_id=str(c["channel_id"]),
                description=str(c.get("description", "")),
                participants=_strs(c.get("participants")),
            )
            for c in as_objects(d.get("channels"))
        ),
        documents=tuple(
            DocumentSpec(
                document_id=str(doc["document_id"]),
                fields=_items(doc.get("fields")),
                description=str(doc.get("description", "")),
                evidence_claim_ids=_strs(doc.get("evidence_claim_ids")),
            )
            for doc in as_objects(d.get("documents"))
        ),
        actions=tuple(parse_action(a) for a in as_objects(d.get("actions"))),
        process=ProcessGraph(
            nodes=tuple(
                parse_node(n) for n in as_objects(_process_block(d.get("process")).get("nodes"))
            )
        ),
        terminal=parse_terminal(d["terminal"]),
        subject_entity=str(d.get("subject_entity", "")),
        resolution_units=str(d.get("resolution_units", "")),
        external_processes=tuple(
            parse_external_process(p) for p in as_objects(d.get("external_processes"))
        ),
        wake_rules=tuple(
            r
            for r in (parse_wake_rule(w) for w in as_objects(d.get("wake_rules")))
            if r.is_checkable()
        ),
        structure_id=str(d.get("structure_id", "primary")),
        structure_rationale=str(d.get("structure_rationale", "")),
    )
