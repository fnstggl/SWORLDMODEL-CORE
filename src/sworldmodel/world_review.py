"""Is this the right world? — asked once, before the rollout budget is spent.

The integrity gates are mechanical and they are the ones that can refuse a run: they
check that participants the evidence names are present, that actors are grounded, that
every terminal term has a producer, that nothing verified was dropped. What they cannot
check is whether the world is *plausible as a description of reality* — whether the real
process that produces the outcome is actually in the world, whether an actor is present
only as decoration, whether an uncertainty variable is the final answer wearing a
costume, whether the world models the production of the outcome or merely the event
that reports it.

This is one model call against the compiled world, made before any actor is invoked,
and it is adversarial by instruction: the model is told to attack the world, not to
praise it. Each of the fourteen questions comes back as an :class:`AuditFinding` with a
severity; CRITICAL and HIGH findings are blocking and route the world into targeted
repair. A finding that cannot state an evidence basis from the provided material is
LOW by rule — an unsupported attack is worth no more than an unsupported world.

It is not a gate and it cannot pass a world the gates refuse; it is a way to notice
that we are about to spend several minutes and a hundred provider calls simulating
something obviously wrong, and to send that finding into targeted repair instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from .compiled import CompiledWorld
from .errors import GatewayError
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .uncertainty import UNGROUNDED_PROVENANCES
from .world_compiler import terminal_producers

__all__ = ["AuditFinding", "WorldReview", "mechanical_world_checks", "review_world"]

SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "PASS")

# Severities that warrant stopping for targeted repair before any rollout is paid for.
_BLOCKING = frozenset({"CRITICAL", "HIGH"})

# The fourteen attacks. Completeness against the evidence store, grounding, producer
# lineage and coverage are absent on purpose: those are decided mechanically against
# the evidence store, not by asking a model's opinion of them.
_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "what_process_produces_outcome",
        "What real process produces this outcome, in one sentence?",
    ),
    (
        "process_is_represented",
        "Is that process actually represented in the compiled world?",
    ),
    (
        "material_components_missing",
        "Which material people, organizations, populations or external systems are missing?",
    ),
    (
        "decorative_actors",
        "Is any actor included only decoratively (its decisions cannot move the outcome)?",
    ),
    (
        "material_actor_compressed_away",
        "Is any material actor compressed away?",
    ),
    (
        "representation_scale_is_right",
        "Is each causal producer modeled at the level that actually produces the "
        "outcome — an individual where a person decides, an organization where a body "
        "acts as one, an operational or population process where the outcome is "
        "throughput or aggregate behavior rather than anyone's choice?",
    ),
    (
        "counts_and_thresholds_preserved",
        "Are real participant counts, authority and decision thresholds preserved?",
    ),
    (
        "production_not_reporting",
        "Is the world modeling the production of the outcome, or merely the event that reports it?",
    ),
    (
        "numbers_have_evidence",
        "Does every numerical starting value, rate or capacity carry evidence citations?",
    ),
    (
        "uncertainty_is_answer_in_disguise",
        "Is any uncertainty variable simply the final answer wearing a costume?",
    ),
    (
        "branch_weights_arbitrary",
        "Is any branch weight arbitrary (not grounded in frequencies, reference cases, "
        "market/survey evidence or documented base rates)?",
    ),
    (
        "actors_get_realistic_information",
        "Does each actor receive realistic local information rather than omniscient state?",
    ),
    (
        "terminal_preresolved",
        "Could the terminal already resolve before simulation begins?",
    ),
    (
        "expert_would_call_incomplete",
        "Would a reasonable domain expert call this world materially incomplete for this question?",
    ),
)


@dataclass(frozen=True)
class AuditFinding:
    """One audit question's verdict: how badly the world fails it, and on what basis.

    ``severity`` is one of :data:`SEVERITIES`. ``finding`` is one sentence saying what
    is wrong (or, for PASS, why the world survives the attack); ``evidence_basis`` is
    one sentence citing what in the provided material decided it.
    """

    key: str
    severity: str
    finding: str
    evidence_basis: str

    @property
    def is_blocking(self) -> bool:
        return self.severity in _BLOCKING

    def as_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "severity": self.severity,
            "finding": self.finding,
            "evidence_basis": self.evidence_basis,
        }


@dataclass(frozen=True)
class WorldReview:
    """The review's findings, and whether they warrant repair before rollout.

    ``answers`` is kept for existing consumers and is synthesized from ``findings``:
    a question is "ok" only when its finding is PASS.
    """

    answers: tuple[tuple[str, bool, str], ...] = ()  # (key, ok, why)
    failed_blocking: tuple[str, ...] = ()
    concerns: tuple[str, ...] = ()
    error: str = ""
    # What the caller did about it. A review recorded with blocking failures beside a
    # world that was then recompiled describes a world nobody simulated, and reads as a
    # run that ignored its own review.
    disposition: str = "not acted on"
    findings: tuple[AuditFinding, ...] = ()

    @property
    def should_repair(self) -> bool:
        return bool(self.failed_blocking)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answers": [{"question": k, "ok": ok, "why": why} for k, ok, why in self.answers],
            "findings": [f.as_dict() for f in self.findings],
            "blocking_failures": list(self.failed_blocking),
            "concerns": list(self.concerns),
            "error": self.error,
            "disposition": self.disposition,
        }

    def with_error(self, error: str) -> WorldReview:
        return replace(self, error=error)

    def repair_instruction(self) -> str:
        lines = [
            "An adversarial review of your compiled world before simulation found these "
            "problems. Fix exactly these and change nothing else:"
        ]
        by_key = {f.key: f for f in self.findings}
        fallback = {k: why for k, ok, why in self.answers if not ok}
        for key in self.failed_blocking:
            f = by_key.get(key)
            if f is not None:
                lines.append(
                    f"- {key} [{f.severity}]: {f.finding} (evidence basis: {f.evidence_basis})"
                )
            else:
                lines.append(f"- {key}: {fallback.get(key, '')}")
        lines.append(
            "Do not invent support for anything. If the evidence does not establish "
            "something, represent it as an uncertainty or leave it out."
        )
        return "\n".join(lines)


def _when(value: Any) -> str | None:
    """A compiled time, rendered for the summary.

    ``ProcessNode.at`` and ``ExternalOccurrence.at`` are typed ``Any`` and hold the ISO
    *string* the compiler emitted — the runtime parses them where it needs a datetime.
    Assuming a datetime here crashed a live Bank of England run at the review call,
    after the world had compiled and passed every gate.
    """

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def summarize_world(compiled: CompiledWorld) -> dict[str, Any]:
    """The compiled world, small enough to review and complete enough to review from."""

    spec = compiled.spec
    actor_ids = {a.entity_id for a in spec.actors}
    return {
        "title": spec.title,
        "structure_rationale": spec.structure_rationale,
        "resolution_condition": spec.terminal.description,
        "entities": [
            {
                "name": e.name,
                "kind": e.kind,
                "scale": e.representation_scale,
                "represents_count": e.represents_count,
                "acts": e.entity_id in actor_ids,
                "role": e.role,
                "authority": list(e.authority),
                "cited": len(e.evidence_claim_ids),
            }
            for e in spec.entities
        ],
        "actions": [
            {"id": a.action_id, "meaning": a.meaning, "cited": len(a.evidence_claim_ids)}
            for a in spec.actions
        ],
        "fields": [
            {
                "id": f.field_id,
                "type": f.value_type,
                "initial": f.initial,
                "cited": len(f.evidence_claim_ids),
            }
            for f in spec.fields
        ],
        "calendar": [
            {"node": n.node_id, "at": _when(n.at), "what": n.description}
            for n in spec.process.nodes
        ],
        "external_processes": [
            {
                "id": p.process_id,
                "what": p.description,
                "occurrences": [_when(o.at) for o in p.occurrences],
            }
            for p in spec.external_processes
        ],
        "uncertainties": [
            {
                "variable": u.variable_id,
                "why_unknown": u.why_unknown,
                # The citation count was invisible here while every other section
                # showed one — so for a world whose numbers live in its uncertainty,
                # "do the numbers have evidence?" could only ever be answered no,
                # against alternatives that in fact cited the constraint claims.
                "cited": len(u.constraining_evidence_ids),
                "outcomes": [
                    {
                        "value": o.value,
                        "weight": o.weight.value,
                        "weight_provenance": o.weight.provenance.value,
                        "weight_source": o.weight.source_detail,
                    }
                    for o in u.outcomes
                ],
            }
            for u in compiled.uncertainty_variables
        ],
        "terminal_producers": {k: list(v) for k, v in sorted(terminal_producers(spec).items())},
    }


# ---------------------------------------------------------------------------
# What the review decides for itself (CWF-6)
#
# Five attacks that need no opinion, because they are facts about the compiled world:
# a terminal set in one step, one uncertain multiplier deciding the result, that
# multiplier carrying no evidence, no intermediate state between the start and the
# answer, and a world that never spans the period in which the outcome is really made.
# They are computed identically for both compiler modes, from the executable alone.
# ---------------------------------------------------------------------------

MECHANICAL_KEYS = (
    "terminal_set_in_one_step",
    "single_uncertain_multiplier_decides",
    "multiplier_lacks_evidence",
    "no_intermediate_production_state",
    "world_skips_the_causal_period",
    # FD-31. Whether the cadence a repeated process runs to could be checked against
    # anything at all. Separate from the moment count because it reports the state of the
    # CHECK, not the state of the world: with no recurrence declaration compiled, how
    # often a process really happens is asserted by the number of occurrences somebody
    # typed, and this says so instead of passing quietly.
    "recurrence_is_declared",
    # FD-41. A world claiming the record already answered the question used to short
    # circuit this whole function to a single PASS, so the entire settled-record class
    # of run received no mechanical scrutiny at all — its only remaining attack was one
    # LLM opinion with no mechanical backstop. These three examine the citation the
    # claim rests on. They do not adjudicate resolution SCOPE (CWF-7, separate work);
    # they make sure the preresolved path cannot publish on an unexamined citation.
    "cited_resolution_claims_exist",
    "cited_resolution_rests_on_the_record",
    "cited_resolution_subject_matches",
)


def _mech(key: str, severity: str, finding: str, basis: str) -> AuditFinding:
    return AuditFinding(key=key, severity=severity, finding=finding, evidence_basis=basis)


def _dated_effects(spec: Any) -> list[tuple[str, str, Any]]:
    """(label, timestamp, effect) for every non-agent effect, in the order it fires."""

    items: list[tuple[tuple[int, str, int], tuple[str, str, Any]]] = []
    index = 0
    for node in spec.process.nodes:
        at = str(node.at or "")
        for eff in node.effects:
            items.append(((0 if at else 1, at, index), (f"process_node:{node.node_id}", at, eff)))
            index += 1
    for proc in spec.external_processes:
        for occ in proc.occurrences:
            at = str(occ.at or "")
            for eff in occ.effects:
                items.append(
                    ((0 if at else 1, at, index), (f"external_process:{proc.process_id}", at, eff))
                )
                index += 1
    items.sort(key=lambda kv: kv[0])
    return [value for _key, value in items]


def _probe_value(value: Any, fields: dict[str, Any]) -> Any:
    """An effect's value expression evaluated against a field assignment, or None."""

    from .expressions import evaluate
    from .worldspec import Expr, parse_expr

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    expr: Any = value
    if isinstance(value, dict) and "op" in value:
        try:
            expr = parse_expr(value)
        except (ValueError, KeyError, TypeError):
            return None
    if not isinstance(expr, Expr):
        return None
    try:
        return evaluate(expr, _ProbeWorld(fields))
    except Exception:  # noqa: BLE001 — a probe that cannot evaluate simply declines
        return None


class _ProbeWorld:
    """A world made of nothing but a field assignment, for evaluating one expression."""

    def __init__(self, fields: dict[str, Any]) -> None:
        self._fields = fields

    def get_field(self, name: str) -> Any:
        return self._fields.get(name)

    def get_records(self, collection: str) -> list[dict[str, Any]]:
        return []

    def get_events(self, event_type: str) -> list[dict[str, Any]]:
        return []

    def get_resource(self, resource_id: str, holder: str) -> float:
        return 0.0

    def get_document_field(self, document_id: str, field_name: str) -> Any:
        return None

    def get_stage(self) -> str:
        return ""

    def get_now(self) -> datetime:
        return datetime.now(UTC)

    def get_horizon(self) -> datetime:
        return datetime.now(UTC)

    def get_as_of(self) -> datetime:
        return datetime.now(UTC)


def _apply_probe_effect(eff: Any, fields: dict[str, Any]) -> None:
    """One field write applied to a probe assignment, in place."""

    params = eff.params_dict
    name = params.get("field")
    if not isinstance(name, str):
        return
    if eff.op == "set_field":
        value = _probe_value(params.get("value"), fields)
        if value is not None:
            fields[name] = value
    elif eff.op == "adjust_field":
        delta = _probe_value(params.get("delta", params.get("amount")), fields)
        current = fields.get(name)
        if isinstance(delta, (int, float)) and isinstance(current, (int, float)):
            fields[name] = current + delta


def _forced_agent_effects(spec: Any) -> list[Any]:
    """Writes the world cannot avoid, though an actor makes them (FD-33).

    A field that exactly one action writes and that no mechanism writes at all is not a
    decision: the actor's only alternative is to leave the field unset, so if the world
    ever gets a value there, that is the value. Placing such an action on the terminal
    quantity was a universal skeleton key — it hid the terminal behind an effect the
    forward probe never ran, so ``_flipping_uncertainties`` reported nothing and the
    review concluded no draw decided the answer.

    A field two actions can write differently IS a decision and is deliberately excluded:
    the probe must never assume which way an actor chose.
    """

    mechanism, agent = _field_writers(spec)
    return [
        writers[0][1]
        for name, writers in sorted(agent.items())
        if len(writers) == 1 and name not in mechanism
    ]


def _forward_fields(spec: Any, seed: dict[str, Any]) -> dict[str, Any]:
    """The world's own mechanisms run forward from a seeded branch draw.

    Followed by the writes only one actor can make and nothing else can (see
    :func:`_forced_agent_effects`), applied twice so a chain of them settles.
    """

    fields: dict[str, Any] = {f.field_id: f.initial for f in spec.fields if f.initial is not None}
    fields.update(seed)
    for _label, _at, eff in _dated_effects(spec):
        _apply_probe_effect(eff, fields)
    forced = _forced_agent_effects(spec)
    for _pass in range(2):
        for eff in forced:
            _apply_probe_effect(eff, fields)
    return fields


def _terminal_under(spec: Any, seed: dict[str, Any]) -> bool | None:
    from .expressions import evaluate

    try:
        return bool(evaluate(spec.terminal.yes_when, _ProbeWorld(_forward_fields(spec, seed))))
    except Exception:  # noqa: BLE001 — an undetermined probe decides nothing
        return None


def _flipping_uncertainties(compiled: CompiledWorld) -> list[Any]:
    """Uncertainty variables whose alternatives alone flip the terminal.

    Computed by running the world's own arithmetic forward under each alternative — so
    it is right for a product, an accumulation or any chain the compiler emitted, and it
    needs no knowledge of what the question is about.
    """

    spec = compiled.spec
    flipping = []
    for u in compiled.uncertainty_variables:
        sides = set()
        for outcome in u.outcomes:
            seed = dict(outcome.field_effects)
            result = _terminal_under(spec, seed)
            if result is not None:
                sides.add(result)
        if len(sides) > 1:
            flipping.append(u)
    return flipping


def _uncertainty_field_ids(compiled: CompiledWorld) -> set[str]:
    out: set[str] = set()
    for u in compiled.uncertainty_variables:
        out.add(u.variable_id)
        out.update(name for o in u.outcomes for name, _v in o.field_effects)
    return out


# ---------------------------------------------------------------------------
# The terminal's lineage closure (FD-28)
#
# Every check below used to key on the terminal's DIRECT writers, which meant one alias
# hop disarmed all of them: split `total := base x factor` into `projection := base x
# factor` followed by `total := projection` and the deciding draw is no longer read by
# anything that writes a terminal term, so the review's own `_flipping_uncertainties`
# would correctly report that the draw flips the answer and the check beside it would
# report PASS. The static side already had the right idea — `semantic_plan._lineage_states`
# — and D3 used it, which is why D3 caught what the review could not.
#
# This is the compiled twin of that closure, computed from the executable so both
# compiler modes are judged identically. It deliberately traverses ACTION effects too: a
# lineage that runs through an actor's write is still a lineage, and pretending
# otherwise is the other half of the same evasion (FD-33).
# ---------------------------------------------------------------------------


def _effect_write(eff: Any) -> tuple[str | None, set[str]]:
    """The field one compiled effect writes, and the fields its amount reads.

    ``delta`` and ``amount`` are both accepted for ``adjust_field`` because the two
    compiler modes emit different keys for it, and a check that reads only one of them
    sees an accumulation as reading nothing at all.
    """

    if eff.op not in ("set_field", "adjust_field"):
        return None, set()
    params = eff.params_dict
    name = params.get("field")
    if not isinstance(name, str):
        return None, set()
    value = (
        params.get("value") if eff.op == "set_field" else params.get("delta", params.get("amount"))
    )
    return name, _reads_fields(value)


def _field_writers(spec: Any) -> tuple[dict[str, list[tuple[str, Any]]], dict[str, list[tuple[str, Any]]]]:
    """Every field write in the world, as (what mechanisms do, what actors do).

    Each maps field id -> [(label, effect)]. Kept apart rather than merged because the
    checks mean different things by them: an actor writing the terminal is a reason to
    ask HOW it decides, not the same thing as the environment announcing a figure.
    """

    mechanism: dict[str, list[tuple[str, Any]]] = {}
    agent: dict[str, list[tuple[str, Any]]] = {}
    for label, _at, eff in _dated_effects(spec):
        name, _reads = _effect_write(eff)
        if name is not None:
            mechanism.setdefault(name, []).append((label, eff))
    for action in spec.actions:
        for eff in action.effects:
            name, _reads = _effect_write(eff)
            if name is not None:
                agent.setdefault(name, []).append((f"action:{action.action_id}", eff))
    return mechanism, agent


def _lineage_fields(spec: Any, seeds: set[str]) -> set[str]:
    """Every field whose value can reach ``seeds`` through this world's own writes.

    The compiled counterpart of :func:`sworldmodel.semantic_plan._lineage_states`, and
    deliberately the same closure: "terminal-relevant" means a field that something the
    terminal reads is computed from, however many hops away.
    """

    mechanism, agent = _field_writers(spec)
    relevant = set(seeds)
    while True:
        grown = set(relevant)
        for name in sorted(relevant):
            for _label, eff in mechanism.get(name, []) + agent.get(name, []):
                _n, reads = _effect_write(eff)
                grown |= reads
        if grown == relevant:
            return relevant
        relevant = grown


def _is_relabelling(eff: Any) -> bool:
    """True when a write copies one field and adds nothing: ``x := y``.

    A rename is not a production stage. This is what tells the intermediate the recharge
    world really produces — a diverted volume that is then ADDED to a cited log — from
    the intermediate an evasion invents, which exists only to be copied into the terminal
    so that the terminal's direct writer no longer reads the draw.
    """

    from .worldspec import Expr, parse_expr

    if eff.op != "set_field":
        return False
    value: Any = eff.params_dict.get("value")
    if isinstance(value, dict) and "op" in value:
        try:
            value = parse_expr(value)
        except (ValueError, KeyError, TypeError):
            return False
    return isinstance(value, Expr) and value.op == "field" and len(_reads_fields(value)) == 1


def _announced_chain(
    name: str,
    mechanism: dict[str, list[tuple[str, Any]]],
    agent: dict[str, list[tuple[str, Any]]],
    seen: frozenset[str] = frozenset(),
) -> list[str] | None:
    """The chain of set-once writes by which ``name`` is announced, or None if built.

    The lineage generalisation of D4's one-step test, and it reduces to exactly the old
    test at depth one: a field written by a single ``set_field`` whose inputs nothing in
    the world produces is announced. When its inputs ARE produced, the same question is
    asked of each of them, so a rename in the middle buys nothing.

    An actor's write ends the chain unless it is a pure relabelling. A world in which
    people genuinely produce the outcome owes no operational depth; a world in which one
    actor's only affordance copies a figure the environment computed is the environment
    announcing the answer with a signature on it.
    """

    if name in seen:
        return None
    writers = mechanism.get(name, []) + agent.get(name, [])
    if len(writers) != 1:
        return None
    label, eff = writers[0]
    if eff.op != "set_field":
        return None
    if agent.get(name) and not _is_relabelling(eff):
        return None
    _n, reads = _effect_write(eff)
    chain = [f"{name} (set once by {label})"]
    for source in sorted((reads & (set(mechanism) | set(agent))) - {name}):
        rest = _announced_chain(source, mechanism, agent, seen | {name})
        if rest is None:
            return None
        chain.extend(rest)
    return chain


def _production_stages(
    lineage: set[str],
    field_terms: set[str],
    mechanism: dict[str, list[tuple[str, Any]]],
    agent: dict[str, list[tuple[str, Any]]],
) -> list[str]:
    """Lineage fields the world genuinely builds, rather than merely relabels.

    A field counts when the world does something to it — writes it more than once, or
    accumulates into it — or when something does more than copy it: adds it to a stock,
    or combines it with another field. An intermediate whose only consumer assigns it
    straight into the terminal is a rename, and a rename is not a production stage.
    """

    stages: list[str] = []
    writes = [
        (target, eff)
        for table in (mechanism, agent)
        for target, items in table.items()
        for _label, eff in items
    ]
    for name in sorted((lineage & set(mechanism)) - field_terms):
        writers = mechanism[name]
        if len(writers) > 1 or any(eff.op == "adjust_field" for _label, eff in writers):
            stages.append(name)
            continue
        for target, eff in writes:
            if target == name or target not in lineage:
                continue
            _t, reads = _effect_write(eff)
            if name in reads and (eff.op == "adjust_field" or len(reads) > 1):
                stages.append(name)
                break
    return stages


# ---------------------------------------------------------------------------
# Cadence (FD-25, FD-31)
# ---------------------------------------------------------------------------

_PERIOD_UNIT_SECONDS = {"W": 604800.0, "D": 86400.0, "H": 3600.0, "M": 60.0, "S": 1.0}


def _declared_period_seconds(proc: Any) -> float | None:
    """The recurrence period a compiled process declares, in seconds, or None.

    The semantic layer is growing a first-class recurrence declaration (``every`` with
    ``from``/``until``) and the ruling is that it must survive lowering onto the compiled
    process rather than being expanded away into a flat occurrence list — an expansion is
    what runs, the declaration is what can be audited. Until that attribute exists there
    is nothing here to read, and the caller must SAY it could not verify the cadence
    rather than passing quietly: a gate that cannot run must never read as a gate that
    ran and approved.

    Tolerant of the shapes the declaration might arrive in — a number of seconds, a
    ``timedelta``, or an ISO-8601 duration of fixed-length units ("P1W", "PT12H").
    Months and years are deliberately unsupported: they are not fixed lengths, so a
    period expressed in them cannot be checked against timestamps without a calendar.
    """

    from datetime import timedelta

    raw = next(
        (
            value
            for attr in ("every", "period", "recurrence", "recurs_every")
            if (value := getattr(proc, attr, None)) is not None
        ),
        None,
    )
    if raw is None:
        return None
    if isinstance(raw, timedelta):
        return raw.total_seconds()
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw) or None
    period = getattr(raw, "every", None) or getattr(raw, "period", None) or raw
    text = str(period).strip().upper()
    if not text.startswith("P"):
        return None
    total, number, in_time = 0.0, "", False
    for char in text[1:]:
        if char == "T":
            in_time = True
        elif char.isdigit() or char == ".":
            number += char
        elif char in _PERIOD_UNIT_SECONDS and number:
            unit = "M" if (char == "M" and in_time) else char
            if char == "M" and not in_time:
                return None  # months are not a fixed length
            total += float(number) * _PERIOD_UNIT_SECONDS[unit]
            number = ""
        else:
            return None
    return total or None


def _repetition_groups(proc: Any, lineage: set[str]) -> list[tuple[str, list[Any]]]:
    """Occurrences of one process that repeat the same write, grouped by that write.

    "The same thing happening again" is what a cadence declares, and it is the only thing
    a count can be wrong about. Two occurrences that write different fields are two
    stages of a process, not two turns of a cycle, so they are never grouped — which is
    why the recharge world's divert-then-infiltrate pair is not treated as a repetition
    with an undeclared period.
    """

    groups: dict[tuple[tuple[str, str], ...], list[Any]] = {}
    for occ in proc.occurrences:
        signature = tuple(
            sorted(
                (eff.op, name)
                for eff in occ.effects
                if (name := _effect_write(eff)[0]) is not None and name in lineage
            )
        )
        if signature:
            groups.setdefault(signature, []).append(occ)
    return [
        (", ".join(f"{op} {name}" for op, name in signature), occurrences)
        for signature, occurrences in sorted(groups.items())
        if len(occurrences) > 1
    ]


def _cited_terminal_claim_ids(spec: Any) -> dict[str, tuple[str, ...]]:
    """Per terminal term, the claim ids its ``evidence:`` producers cite.

    Read from :func:`terminal_producers`, which is the same source
    ``_cited_factual_resolution`` uses to decide the world is settled — so these are
    exactly the citations the settled-record claim rests on, and nothing else.
    """

    out: dict[str, tuple[str, ...]] = {}
    for term, producers in terminal_producers(spec).items():
        ids: list[str] = []
        for who in producers:
            text = str(who)
            if text.startswith("evidence:"):
                ids.extend(i for i in text[len("evidence:") :].split(",") if i)
        out[term] = tuple(dict.fromkeys(ids))
    return out


def _significant_tokens(text: str) -> set[str]:
    """Words long enough to identify a subject, lowercased.

    Deliberately crude and deliberately generous: this decides whether a citation is
    plausibly ABOUT the world's subject at all, not whether it resolves the question.
    Short words carry no subject.
    """

    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {w for w in cleaned.split() if len(w) >= 4}


def _preresolved_checks(compiled: CompiledWorld, evidence: Any) -> tuple[AuditFinding, ...]:
    """Examine the citation a settled-record world rests on (FD-41).

    The operational-depth checks below are meaningless here — a question the record has
    already answered is not re-produced inside the window, and demanding a production
    process for it is what manufactured an absolute NO on a live Bank of England run.
    But "meaningless" was implemented as "return one PASS and check nothing", which left
    the entire OPEC+/EU-Mercosur class of run with no mechanical gate whatsoever: any
    claim id, attached to any initial value, established any outcome.

    So the attack moves to where it belongs — the citation itself:

    * the cited ids must name claims that actually exist and are admissible at the
      cutoff (an id that names nothing grounds nothing);
    * at least one of them must be an OBSERVATION carrying a verified excerpt, not an
      inference or a hypothesis (a record that "already answered" the question is a
      record, not a conclusion the compiler drew);
    * and they must be about the world's own subject.

    Scope adjudication — whether the cited act matches the question's instrument, degree
    and specificity — is CWF-7 and is not attempted here; the LLM review's
    ``terminal_preresolved`` attack still carries that, now with a mechanical floor
    underneath it.
    """

    cited = _cited_terminal_claim_ids(compiled.spec)
    all_ids = tuple(dict.fromkeys(i for ids in cited.values() for i in ids))
    if evidence is None:
        # Only reachable from a caller that supplied no evidence view. Not a defect in
        # the world, so it cannot block — but it must never read as an examined citation.
        return (
            _mech(
                "cited_resolution_claims_exist",
                "MEDIUM",
                "this world resolves on cited record and the citation could NOT be "
                f"examined here: no evidence view was supplied to check {sorted(all_ids)}",
                "computed from the compiled world: terminal producers are all evidence: "
                "citations, and no store was available to resolve them",
            ),
        )

    findings: list[AuditFinding] = []
    resolved: list[Any] = []
    missing: list[str] = []
    for claim_id in all_ids:
        try:
            resolved.append(evidence.get(claim_id))
        except Exception:  # noqa: BLE001 — unknown id, or one past the cutoff
            missing.append(claim_id)

    if missing:
        findings.append(
            _mech(
                "cited_resolution_claims_exist",
                "CRITICAL",
                f"this world answers the question from the record, and the record it "
                f"cites is not there: {sorted(missing)} name no claim admissible at the "
                "cutoff — the resolution rests on a citation to nothing",
                "computed from the compiled world: terminal-producing evidence: ids "
                "resolved against the evidence store at the cutoff",
            )
        )
    else:
        findings.append(
            _mech(
                "cited_resolution_claims_exist",
                "PASS",
                f"every claim the settled record cites exists and is admissible: {sorted(all_ids)}",
                "computed from the compiled world: terminal-producing evidence: ids "
                "resolved against the evidence store at the cutoff",
            )
        )

    observed = [
        c
        for c in resolved
        if getattr(getattr(c, "epistemic_type", None), "value", "") == "observation"
        and str(getattr(c, "supporting_excerpt", "")).strip()
    ]
    if resolved and not observed:
        findings.append(
            _mech(
                "cited_resolution_rests_on_the_record",
                "CRITICAL",
                "the outcome is declared already established, but not one cited claim is "
                "an observation carrying a verified excerpt — every one is an inference "
                "or a hypothesis, so what answers the question is a conclusion somebody "
                "drew rather than a record of what happened",
                "computed from the compiled world: epistemic_type and supporting_excerpt "
                "of every claim the terminal's producers cite",
            )
        )
    else:
        findings.append(
            _mech(
                "cited_resolution_rests_on_the_record",
                "PASS" if observed else "MEDIUM",
                f"{len(observed)} cited claim(s) are observations with verified excerpts"
                if observed
                else "the terminal cites no claims at all, so there is no record to weigh",
                "computed from the compiled world: epistemic_type and supporting_excerpt "
                "of every claim the terminal's producers cite",
            )
        )

    # Subject match, in the ONE direction that cannot false-positive. Asking "does the
    # claim mention the subject_entity" fails on ordinary paraphrase — a world about "the
    # EU-Mercosur agreement" is legitimately resolved by a claim about the "European
    # Commission" — and a gate that refuses correct worlds is worse than the hole it
    # closes. What admits no paraphrase defense is a cited record that shares NOTHING
    # with the world it supposedly settles: not the subject, not the title, not any
    # entity's name, not the resolution units, not a word of the terminal's own
    # description. That is evidence about something else, and it is checked against every
    # string the world offers so the bar is as generous as it can be while still meaning
    # something. Whether the cited act matches the question's instrument, degree and
    # specificity is CWF-7 and is not decided here.
    world_text = " ".join(
        [
            str(getattr(compiled.spec, "subject_entity", "") or ""),
            str(getattr(compiled.spec, "resolution_units", "") or ""),
            str(getattr(compiled.spec, "title", "") or ""),
            str(getattr(compiled.spec.terminal, "description", "") or ""),
            " ".join(str(getattr(e, "name", "")) for e in compiled.spec.entities),
        ]
    )
    wanted = _significant_tokens(world_text)
    if wanted and resolved:
        haystack: set[str] = set()
        for c in resolved:
            haystack |= _significant_tokens(str(getattr(c, "proposition", "")))
            haystack |= _significant_tokens(str(getattr(c, "supporting_excerpt", "")))
            haystack |= _significant_tokens(" ".join(getattr(c, "entities", ()) or ()))
        overlap = sorted(wanted & haystack)
        findings.append(
            _mech(
                "cited_resolution_subject_matches",
                "PASS" if overlap else "HIGH",
                f"the cited record and this world are about the same thing ({overlap[:8]})"
                if overlap
                else "the cited record has nothing in common with the world it settles — "
                f"not its subject ({str(getattr(compiled.spec, 'subject_entity', ''))!r}), "
                "its title, any entity's name, its resolution units or its terminal's own "
                "description: the claim that the question is already answered rests on "
                "evidence about something else",
                "computed from the compiled world: every naming string the spec offers "
                "against the propositions, excerpts and entities of every cited claim",
            )
        )
    else:
        # Nothing to compare. Absence of a subject is not evidence of a mismatch, so this
        # never fails on it.
        findings.append(
            _mech(
                "cited_resolution_subject_matches",
                "MEDIUM",
                "the subject match could not be computed: this world offers no naming "
                "string, or its terminal cites no claims",
                "computed from the compiled world: the spec's naming strings and the cited claims",
            )
        )
    return tuple(findings)


def mechanical_world_checks(
    compiled: CompiledWorld, evidence: Any = None
) -> tuple[AuditFinding, ...]:
    """The CWF-6 attacks that are decided by the compiled world, not by an opinion.

    Each check contributes at least one finding — its violation, or a PASS saying what
    was checked — so a review that found nothing still shows what it looked at. They run
    on the executable, so the direct compiler and the semantic compiler are judged by
    exactly the same standard.

    ``evidence`` is the cutoff-filtered view the world was compiled from. It is used only
    on the settled-record path, where the thing to attack is the citation rather than the
    machinery; the operational checks need nothing but the executable.
    """

    from .world_compiler import _cited_factual_resolution, _effect_produces, _expr_terms

    spec = compiled.spec
    if _cited_factual_resolution(spec, compiled.base_world):
        return (
            _mech(
                "terminal_set_in_one_step",
                "PASS",
                "the record already answers this question: every terminal term is "
                "established by cited pre-cutoff evidence, so there is no production "
                "process to demand",
                "computed from the compiled world: a cited factual resolution",
            ),
        ) + _preresolved_checks(compiled, evidence)
    terms = _expr_terms(spec.terminal.yes_when)
    field_terms = {t.split(":", 1)[1] for t in terms if t.startswith("field:")}
    action_written: set[str] = set()
    for action in spec.actions:
        for eff in action.effects:
            action_written |= _effect_produces(eff)
    if not field_terms:
        return (
            _mech(
                "terminal_set_in_one_step",
                "PASS",
                "the terminal reads no quantity, so the operational-depth checks have no "
                "quantity to follow",
                f"computed from the compiled world: terminal terms {sorted(terms)}",
            ),
        )
    # FD-33. This used to return that same single PASS whenever ANY action wrote ANY
    # terminal term, which made one actor affordance on the terminal quantity a skeleton
    # key: it collapsed the review from five checks to one, and the four it surrendered
    # were the ones that catch the shape underneath. An actor writing the terminal is a
    # reason to scrutinise HOW it decides, not a reason to stop asking.
    actor_terms = sorted(t.split(":", 1)[1] for t in (action_written & terms) if t.startswith("field:"))

    effects = _dated_effects(spec)
    findings: list[AuditFinding] = []
    uncertain_fields = _uncertainty_field_ids(compiled)
    mechanism, agent = _field_writers(spec)
    lineage = _lineage_fields(spec, field_terms)

    # 1 & 2 — the terminal quantity announced rather than built, and one uncertain
    # multiplier deciding it. Both now read the whole lineage: a draw scaled into any
    # field the terminal is computed from is a draw scaled into the terminal.
    one_step: list[str] = []
    for name in sorted(field_terms):
        chain = _announced_chain(name, mechanism, agent)
        if chain is not None:
            one_step.extend(chain)
    multiplier_writers: list[tuple[str, str]] = []
    for name in sorted(lineage):
        for _label, eff in mechanism.get(name, []) + agent.get(name, []):
            _n, reads = _effect_write(eff)
            multiplier_writers.extend((name, v) for v in sorted(reads & uncertain_fields))
    multiplier_writers = list(dict.fromkeys(multiplier_writers))
    if one_step:
        findings.append(
            _mech(
                "terminal_set_in_one_step",
                "CRITICAL",
                f"the terminal quantity is written once and never built: {one_step} — "
                "nothing accumulates, nothing intermediate is produced, so the world "
                "reports a figure rather than operating the process that makes it",
                "computed from the compiled world: following each terminal term back "
                "through its writers, every step is a single set_field and the chain "
                "bottoms out in inputs no effect produces",
            )
        )
    else:
        findings.append(
            _mech(
                "terminal_set_in_one_step",
                "PASS",
                "the terminal quantity is produced by more than one step, or from "
                "inputs the world itself produces"
                + (f", or by what actors do ({actor_terms})" if actor_terms else ""),
                "computed from the compiled world: the writer chain of each terminal "
                f"term, over the lineage {sorted(lineage)}",
            )
        )

    # 3 & 4 — does one uncertain factor decide it, and does that factor cite anything?
    flipping = {u.variable_id: u for u in _flipping_uncertainties(compiled)}
    decisive = [(term, variable) for term, variable in multiplier_writers if variable in flipping]
    if decisive:
        uncited = sorted(
            {
                variable
                for _t, variable in decisive
                if not flipping[variable].constraining_evidence_ids
            }
        )
        findings.append(
            _mech(
                "single_uncertain_multiplier_decides",
                "HIGH" if uncited else "MEDIUM",
                f"the result is decided by an uncertain factor read into the terminal "
                f"quantity's lineage: {decisive} — running the world's own arithmetic "
                "under each alternative flips the answer, so the branch draw is the "
                "forecast",
                "computed from the compiled world: the terminal resolves both ways "
                "across that variable's alternatives with nothing else changed, and the "
                "variable is read by a writer of the terminal's lineage "
                f"{sorted(lineage)}",
            )
        )
        if uncited:
            findings.append(
                _mech(
                    "multiplier_lacks_evidence",
                    "CRITICAL",
                    f"the factor that decides the result cites no evidence: {uncited} — "
                    "an invented number is doing the work the world is supposed to do",
                    "computed from the compiled world: the deciding uncertainty has no "
                    "constraining_evidence_ids",
                )
            )
        else:
            findings.append(
                _mech(
                    "multiplier_lacks_evidence",
                    "PASS",
                    "the deciding factor carries constraining evidence",
                    "computed from the compiled world: constraining_evidence_ids present",
                )
            )
    else:
        findings.append(
            _mech(
                "single_uncertain_multiplier_decides",
                "PASS",
                "no single uncertain factor read into the terminal quantity's lineage "
                "flips the answer on its own",
                "computed from the compiled world: terminal probed under every "
                "alternative of every uncertainty, against the lineage "
                f"{sorted(lineage)}",
            )
        )

    # 5 — is there anything between the start of the world and its answer?
    # Only lineage fields count, and only where the world does more with them than copy
    # them: an intermediate nothing reads, or one that exists solely to be assigned into
    # the terminal, is a rename dressed as a production stage.
    intermediate = _production_stages(lineage, field_terms, mechanism, agent)
    findings.append(
        _mech(
            "no_intermediate_production_state",
            "PASS" if intermediate else "HIGH",
            f"the mechanisms produce intermediate state: {intermediate}"
            if intermediate
            else "the world's mechanisms build nothing between the initial world and the "
            "answer — every field the terminal is computed from is either the terminal "
            "quantity itself or a relabelling of it, so there is no production, demand "
            "or capacity state in between",
            "computed from the compiled world: fields in the terminal's lineage written "
            "by process nodes and external occurrences, and what reads them",
        )
    )

    # 6 — does the world exist across the period in which the outcome is really made?
    findings.extend(_causal_period_findings(spec, effects, lineage))
    return tuple(findings)


def _causal_period_findings(
    spec: Any, effects: list[tuple[str, str, Any]], lineage: set[str]
) -> list[AuditFinding]:
    """Does the world span the period the outcome is made in, and per process (FD-25/31).

    Two defects, one mechanism. The moment count used to be taken WORLD-WIDE, so a
    correctly-scheduled process laundered an under-scheduled one: a live review reported
    "the world acts at 10 distinct moments" beside a compiled world whose occurrences sat
    at two dates per process. And ``len(moments) > 1`` was the whole test, so ten
    occurrences of a "weekly" cycle all dated the same instant passed as soon as one setup
    step supplied a second timestamp.

    So the moments counted are the moments at which the terminal's LINEAGE is written —
    the same closure the depth checks use — and repetition is judged per process: a group
    of occurrences repeating the same write must carry as many distinct timestamps as it
    has members, because that is what "it happened again" means.
    """

    findings: list[AuditFinding] = []
    moments = sorted({at for _label, at, eff in effects if at and _effect_write(eff)[0] in lineage})
    crowded: list[str] = []
    repeating: list[tuple[str, str, list[Any], float | None]] = []
    for proc in spec.external_processes:
        for what, occurrences in _repetition_groups(proc, lineage):
            stamps = {str(occ.at) for occ in occurrences if occ.at}
            repeating.append((proc.process_id, what, occurrences, _declared_period_seconds(proc)))
            if len(stamps) < len(occurrences):
                crowded.append(
                    f"{proc.process_id} repeats '{what}' {len(occurrences)} times at "
                    f"{len(stamps)} distinct moment(s) {sorted(stamps)}"
                )
    if len(moments) <= 1:
        findings.append(
            _mech(
                "world_skips_the_causal_period",
                "HIGH",
                "everything that makes this world's answer happens at a single moment "
                f"({moments or 'no dated occurrence at all'}) — it jumps from the cutoff "
                "to the answer and simulates none of the period in which the outcome is "
                "made",
                "computed from the compiled world: the timestamps of every non-agent "
                f"effect that writes the terminal's lineage {sorted(lineage)}",
            )
        )
    elif crowded:
        findings.append(
            _mech(
                "world_skips_the_causal_period",
                "HIGH",
                f"a process claims repetition it does not schedule: {crowded} — the same "
                "change is typed again and again at one instant, so the world asserts a "
                "cadence it never spans and the count of repetitions is doing the work "
                "the calendar should",
                "computed from the compiled world: per process, occurrences repeating "
                "one write against the distinct timestamps they carry",
            )
        )
    else:
        findings.append(
            _mech(
                "world_skips_the_causal_period",
                "PASS",
                f"the terminal's lineage is written at {len(moments)} distinct moments: "
                f"{moments}, and no process repeats a change without spreading it",
                "computed from the compiled world: the timestamps of every non-agent "
                f"effect that writes the terminal's lineage {sorted(lineage)}",
            )
        )
    findings.append(_cadence_finding(repeating))
    return findings


def _cadence_finding(
    repeating: list[tuple[str, str, list[Any], float | None]],
) -> AuditFinding:
    """Whether a repeated process's cadence could be checked against a declaration.

    The count of repetitions can decide the answer on its own — eight typed cycles and
    ten typed cycles land on opposite sides of a threshold with no uncertainty anywhere
    in the world — and an enumerated occurrence list contains no claim about how often
    the thing really happens, so there is nothing to check the count against.

    When no declaration exists this says so rather than passing quietly. An inert hook
    that reports PASS is worse than no hook: a gate that cannot run must never read as a
    gate that ran and approved.
    """

    if not repeating:
        return _mech(
            "recurrence_is_declared",
            "PASS",
            "no non-agent process repeats the same change, so there is no cadence for a "
            "declaration to pin down",
            "computed from the compiled world: occurrences of each external process "
            "grouped by the write they repeat",
        )
    undeclared = sorted({f"{pid} ('{what}' x{len(occ)})" for pid, what, occ, p in repeating if p is None})
    if undeclared:
        return _mech(
            "recurrence_is_declared",
            "MEDIUM",
            f"this check could NOT run: {undeclared} repeat by enumeration and declare no "
            "recurrence period, so how often the process really happens is asserted by "
            "the number of occurrences that were typed and nothing verifies it",
            "computed from the compiled world: no compiled process carries a recurrence "
            "declaration to check the enumerated occurrences against",
        )
    wrong = sorted(
        f"{pid} ('{what}')"
        for pid, what, occurrences, period in repeating
        if period is not None and not _spacing_matches(occurrences, period)
    )
    if wrong:
        return _mech(
            "recurrence_is_declared",
            "HIGH",
            f"the occurrences contradict the declared recurrence: {wrong} — the world "
            "runs to a different cadence than the one it claims, so the enumerated count "
            "is not the declaration's consequence",
            "computed from the compiled world: gaps between consecutive occurrences "
            "against each process's declared period",
        )
    return _mech(
        "recurrence_is_declared",
        "PASS",
        "every repeated process declares a recurrence period and its occurrences keep to it",
        "computed from the compiled world: gaps between consecutive occurrences against "
        "each process's declared period",
    )


def _spacing_matches(occurrences: list[Any], period: float, *, tolerance: float = 0.25) -> bool:
    """Do consecutive occurrences sit roughly one declared period apart?"""

    stamps: list[datetime] = []
    for occ in occurrences:
        try:
            stamps.append(datetime.fromisoformat(str(occ.at)))
        except (TypeError, ValueError):
            return False
    stamps.sort()
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:], strict=False)]
    return bool(gaps) and all(abs(gap - period) <= tolerance * period for gap in gaps)


def _reads_fields(value: Any) -> set[str]:
    from .world_compiler import _value_field_reads

    return _value_field_reads(value)


def _parse_findings(data: Any, known: Iterable[str] | None = None) -> tuple[AuditFinding, ...]:
    """Model output -> findings. Pure: no gateway, no world, no side effects.

    Tolerant of the shapes a model actually produces (``question`` for ``key``, ``why``
    for ``finding``, lower-case severities), and it enforces the rule the prompt
    states: a CRITICAL or HIGH finding that states no evidence basis is demoted to LOW,
    because an unsupported attack must not block a world the gates passed. An unknown
    severity is read as MEDIUM — visible, never blocking. One finding per known key;
    unknown keys and repeats are dropped.
    """

    keys = frozenset(known) if known is not None else frozenset(k for k, _ in _QUESTIONS)
    items = data.get("findings", []) if isinstance(data, dict) else []
    out: dict[str, AuditFinding] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or item.get("question") or "").strip()
        if key not in keys or key in out:
            continue
        severity = str(item.get("severity", "")).strip().upper()
        if severity not in SEVERITIES:
            severity = "MEDIUM"
        finding = str(item.get("finding") or item.get("why") or "").strip()
        basis = str(item.get("evidence_basis", "")).strip()
        if severity in _BLOCKING and not basis:
            severity = "LOW"
        out[key] = AuditFinding(key=key, severity=severity, finding=finding, evidence_basis=basis)
    return tuple(out.values())


def _from_findings(findings: tuple[AuditFinding, ...]) -> WorldReview:
    """Assemble the review from parsed findings. Pure, so it can be tested as such."""

    return WorldReview(
        answers=tuple((f.key, f.severity == "PASS", f.finding) for f in findings),
        failed_blocking=tuple(f.key for f in findings if f.is_blocking),
        concerns=tuple(
            f"{f.key}: {f.finding}" for f in findings if f.severity in ("MEDIUM", "LOW")
        ),
        findings=findings,
    )


def review_world(
    compiled: CompiledWorld,
    evidence: EvidenceView,
    gateway: ModelGateway,
    *,
    question: str,
    evidence_render: str,
) -> WorldReview:
    """Ask whether this is the right world.

    Never raises. This is an advisory step that runs *after* every mechanical gate has
    already passed the world, so a fault here can only ever destroy a run that was
    otherwise sound — which is exactly what happened: a live Bank of England run
    compiled a real world, cleared every gate, and then died in this function's own
    summary helper because a compiled ``at`` is an ISO string rather than a datetime.
    An opinion about a world must never be able to stop it.
    """

    try:
        mechanical = mechanical_world_checks(compiled, evidence)
    except Exception:  # noqa: BLE001 — a check that cannot run blocks nothing
        mechanical = ()
    try:
        return _review(
            compiled,
            gateway,
            question=question,
            evidence_render=evidence_render,
            mechanical=mechanical,
        )
    except Exception as exc:  # noqa: BLE001 — see the docstring: advisory, never fatal
        # The mechanical findings survive a failed model call: they were decided by the
        # compiled world, not by the opinion that could not be obtained.
        return _from_findings(mechanical).with_error(
            f"the review could not run: {type(exc).__name__}: {exc}"
        )


def _settled_record_block(compiled: CompiledWorld) -> str:
    """The resolution-basis paragraph for a world that already answers from citations.

    A live Bank of England run compiled the one legitimate preresolved state — the
    outcome state initial-true on cited pre-cutoff record, endorsed by the plan's own
    independent review — and this review then attacked it for lacking a production
    process, forcing a recompile whose world demanded the already-made statement be
    made AGAIN inside the window. That recompile changed the question's meaning and
    manufactured an absolute NO. The audit's classifier already knows this state
    (``factual_resolution``); the reviewer has to know it too, and aim its attack at
    the only thing still attackable: whether the cited record establishes the outcome
    as the question means it.
    """

    from .world_compiler import _cited_factual_resolution

    if not _cited_factual_resolution(compiled.spec, compiled.base_world):
        return ""
    return (
        "## THE WORLD CLAIMS A CITED FACTUAL RESOLUTION\n"
        "The terminal already resolves YES at the cutoff, and every term it reads is "
        "established by cited verified record of pre-cutoff events. That is the one "
        "legitimate preresolved state: a question the record has already answered is "
        "not re-produced inside the window, and a repair that demands the outcome be "
        "performed again after the cutoff changes the question's meaning. Do NOT fail "
        "this world for a preresolved terminal, an unrepresented production process, "
        "or decorative actors on that basis alone. Attack the citation instead: does "
        "the cited record establish the outcome exactly as the question means it — "
        "same subject, same act, same specificity, same instrument and degree? If it "
        "does not, fail terminal_preresolved and name the precise gap between what "
        "the record says and what the question asks."
    )


def _review(
    compiled: CompiledWorld,
    gateway: ModelGateway,
    *,
    question: str,
    evidence_render: str,
    mechanical: tuple[AuditFinding, ...] = (),
) -> WorldReview:
    body = "\n".join(f"{key}: {text}" for key, text in _QUESTIONS)
    prompt = "\n\n".join(
        segment
        for segment in [
            "You are the adversarial reality auditor of a compiled simulation world, "
            "run BEFORE any rollout budget is spent. Your job is to ATTACK this world, "
            "not to praise it: find where it fails as a description of the real process "
            "that produces this outcome. Assume the compiler flattered itself; make it "
            "prove otherwise. Be concrete, and ground every attack in the material "
            "below.",
            f"QUESTION THE WORLD MUST RESOLVE: {question}",
            "## THE COMPILED WORLD\n"
            + json.dumps(summarize_world(compiled), indent=2, sort_keys=True, default=str),
            # The review fights the system's own legitimacy rules unless told them: a
            # population run declared an input uncertainty with honestly-labeled
            # symmetric weights over cited anchors, this review called the weights
            # arbitrary and the uncertainty an answer in disguise, and the forced
            # recompile deleted it — leaving a single-branch world that could only end
            # unresolved. What the mechanical gates already permit and police is not
            # for this review to re-litigate.
            "## WHAT IS ALREADY LEGAL HERE\n"
            "Branch weights labeled symmetric_ignorance_assumption are not arbitrary: "
            "the label is the honest state of knowledge, and the runtime reports "
            "bounds instead of a calibrated point wherever such weights matter. Fail "
            "branch_weights_arbitrary only for a weight wearing a GROUNDED provenance "
            "its citations do not support. Likewise an uncertainty is the answer in "
            "disguise only when the terminal reads its drawn value through no real "
            "computation — a mechanical gate upstream already refuses that. An "
            "uncertainty over an input the terminal computes from via cited anchors "
            "is the honest shape of not knowing; demanding its removal produces a "
            "world that can only end unresolved.",
            _settled_record_block(compiled),
            "## THE VERIFIED EVIDENCE IT WAS BUILT FROM\n" + evidence_render,
            "## ANSWER EACH\n" + body,
            "For every question return exactly one finding object: "
            '{"key": "<key>", "severity": "CRITICAL"|"HIGH"|"MEDIUM"|"LOW"|"PASS", '
            '"finding": "<one sentence: what is wrong, or why the world survives>", '
            '"evidence_basis": "<one sentence citing what in the compiled world or the '
            'evidence above decided it>"}. '
            "CRITICAL means the simulation would be meaningless; HIGH means materially "
            "wrong and must be repaired before simulating; MEDIUM is a real concern; "
            "LOW is minor; PASS means the world survives your attack on that question. "
            "A finding without a stated evidence basis from the provided material must "
            "be severity LOW. "
            'Reply with JSON {"findings": [<one object per key, all 14 keys>]}.',
        ]
        if segment
    )
    try:
        resp = gateway.generate(
            GatewayRequest(
                task_kind="world_review",
                prompt=prompt,
                context={"question": question},
                seed=int(prompt_hash("review" + question)[:8], 16),
                expected_keys=("findings",),
            )
        )
    except GatewayError as exc:
        return _from_findings(mechanical).with_error(f"the review could not run: {exc}")

    findings = _parse_findings(resp.data)
    if _every_weight_wears_an_ignorance_label(compiled):
        # A deterministic backstop for the legitimacy rule the prompt states: when
        # every branch weight in the world already carries an ungrounded-provenance
        # label, "the weights are arbitrary" cannot block — the label IS the honest
        # state, and the runtime prices it as bounds rather than a calibrated point.
        # A weight CLAIMING a grounded provenance stays fully attackable.
        findings = tuple(
            replace(
                f,
                severity="MEDIUM",
                finding=f.finding
                + " [not blocking: every weight already wears an ungrounded-provenance "
                "label, which the runtime prices as bounds]",
            )
            if f.key == "branch_weights_arbitrary" and f.is_blocking
            else f
            for f in findings
        )
    # The mechanical findings are the world's own facts and are not up for negotiation:
    # they are appended after the model's opinions and win any key collision.
    keys = {f.key for f in mechanical}
    return _from_findings(tuple(f for f in findings if f.key not in keys) + mechanical)


def _every_weight_wears_an_ignorance_label(compiled: CompiledWorld) -> bool:
    """True when every uncertainty-outcome weight is honestly labeled ungrounded."""

    outcomes = [o for u in compiled.uncertainty_variables for o in u.outcomes]
    return bool(outcomes) and all(o.weight.provenance in UNGROUNDED_PROVENANCES for o in outcomes)
