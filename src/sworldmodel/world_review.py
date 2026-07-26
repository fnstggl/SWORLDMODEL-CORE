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


def _forward_fields(spec: Any, seed: dict[str, Any]) -> dict[str, Any]:
    """The world's own mechanisms run forward from a seeded branch draw."""

    fields: dict[str, Any] = {f.field_id: f.initial for f in spec.fields if f.initial is not None}
    fields.update(seed)
    for _label, _at, eff in _dated_effects(spec):
        params = eff.params_dict
        name = params.get("field")
        if not isinstance(name, str):
            continue
        if eff.op == "set_field":
            value = _probe_value(params.get("value"), fields)
            if value is not None:
                fields[name] = value
        elif eff.op == "adjust_field":
            delta = _probe_value(params.get("delta", params.get("amount")), fields)
            current = fields.get(name)
            if isinstance(delta, (int, float)) and isinstance(current, (int, float)):
                fields[name] = current + delta
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


def mechanical_world_checks(compiled: CompiledWorld) -> tuple[AuditFinding, ...]:
    """The CWF-6 attacks that are decided by the compiled world, not by an opinion.

    Each check contributes at least one finding — its violation, or a PASS saying what
    was checked — so a review that found nothing still shows what it looked at. They run
    on the executable, so the direct compiler and the semantic compiler are judged by
    exactly the same standard.
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
        )
    terms = _expr_terms(spec.terminal.yes_when)
    field_terms = {t.split(":", 1)[1] for t in terms if t.startswith("field:")}
    action_written: set[str] = set()
    for action in spec.actions:
        for eff in action.effects:
            action_written |= _effect_produces(eff)
    if not field_terms or (action_written & terms):
        return (
            _mech(
                "terminal_set_in_one_step",
                "PASS",
                "the terminal is reached through what actors do, or reads no quantity, "
                "so the operational-depth checks do not apply to this world",
                "computed from the compiled world: terminal terms "
                f"{sorted(terms)}, action-written terms {sorted(action_written & terms)}",
            ),
        )

    effects = _dated_effects(spec)
    findings: list[AuditFinding] = []
    uncertain_fields = _uncertainty_field_ids(compiled)
    written_by_mechanism = {
        str(eff.params_dict.get("field"))
        for _label, _at, eff in effects
        if eff.op in ("set_field", "adjust_field") and isinstance(eff.params_dict.get("field"), str)
    }

    # 1 & 2 — the terminal quantity set in one step, and by one uncertain multiplier.
    one_step: list[str] = []
    multiplier_writers: list[tuple[str, str]] = []
    for name in sorted(field_terms):
        writers = [
            (label, eff)
            for label, _at, eff in effects
            if eff.op in ("set_field", "adjust_field") and eff.params_dict.get("field") == name
        ]
        for label, eff in writers:
            value = eff.params_dict.get("value", eff.params_dict.get("delta"))
            reads = _reads_fields(value)
            # A branch draw scaled straight into the terminal quantity, however many
            # stages the world has: the count of writers decides whether it is also
            # one-step, not whether the draw is doing the deciding.
            multiplier_writers.extend((name, v) for v in sorted(reads & uncertain_fields))
            if (
                len(writers) == 1
                and eff.op == "set_field"
                and not (reads & (written_by_mechanism - {name}))
            ):
                one_step.append(f"{name} (set once by {label})")
    if one_step:
        findings.append(
            _mech(
                "terminal_set_in_one_step",
                "CRITICAL",
                f"the terminal quantity is written once and never built: {one_step} — "
                "nothing accumulates, nothing intermediate is produced, so the world "
                "reports a figure rather than operating the process that makes it",
                "computed from the compiled world: the only writer of each term is a "
                "single set_field whose inputs no other effect produces",
            )
        )
    else:
        findings.append(
            _mech(
                "terminal_set_in_one_step",
                "PASS",
                "the terminal quantity is produced by more than one step, or from "
                "inputs the world itself produces",
                "computed from the compiled world: terminal-term writers and their inputs",
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
                f"the result is decided by an uncertain factor read straight into the "
                f"terminal quantity: {decisive} — running the world's own arithmetic "
                "under each alternative flips the answer, so the branch draw is the "
                "forecast",
                "computed from the compiled world: the terminal resolves both ways "
                "across that variable's alternatives with nothing else changed",
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
                "no single uncertain factor read into the terminal quantity flips the "
                "answer on its own",
                "computed from the compiled world: terminal probed under every "
                "alternative of every uncertainty",
            )
        )

    # 5 — is there anything between the start of the world and its answer?
    intermediate = sorted(written_by_mechanism - field_terms)
    findings.append(
        _mech(
            "no_intermediate_production_state",
            "PASS" if intermediate else "HIGH",
            f"the mechanisms produce intermediate state: {intermediate}"
            if intermediate
            else "the world's mechanisms write nothing except the terminal quantity "
            "itself — there is no production, demand or capacity state between the "
            "initial world and the answer",
            "computed from the compiled world: fields written by process nodes and "
            "external occurrences",
        )
    )

    # 6 — does the world exist across the period in which the outcome is really made?
    moments = sorted({at for _label, at, _eff in effects if at})
    findings.append(
        _mech(
            "world_skips_the_causal_period",
            "PASS" if len(moments) > 1 else "HIGH",
            f"the world acts at {len(moments)} distinct moments: {moments}"
            if len(moments) > 1
            else "everything this world does happens at a single moment "
            f"({moments or 'no dated occurrence at all'}) — it jumps from the cutoff to "
            "the answer and simulates none of the period in which the outcome is made",
            "computed from the compiled world: the timestamps of every non-agent effect",
        )
    )
    return tuple(findings)


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
        mechanical = mechanical_world_checks(compiled)
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
