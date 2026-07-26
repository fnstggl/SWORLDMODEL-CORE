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
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .compiled import CompiledWorld
from .errors import GatewayError
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .world_compiler import terminal_producers

__all__ = ["AuditFinding", "WorldReview", "review_world"]

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
        return _review(compiled, gateway, question=question, evidence_render=evidence_render)
    except Exception as exc:  # noqa: BLE001 — see the docstring: advisory, never fatal
        return WorldReview(error=f"the review could not run: {type(exc).__name__}: {exc}")


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
        return WorldReview(error=f"the review could not run: {exc}")

    return _from_findings(_parse_findings(resp.data))
