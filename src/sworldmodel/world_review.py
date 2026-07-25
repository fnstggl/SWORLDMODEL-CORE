"""Is this the right world? — asked once, before the rollout budget is spent.

The integrity gates are mechanical and they are the ones that can refuse a run: they
check that participants the evidence names are present, that actors are grounded, that
every terminal term has a producer, that nothing verified was dropped. What they cannot
check is whether the world is *plausible as a description of reality* — whether the
resolution contract matches the question asked, whether a detail arrived from evidence or
from the compiler's imagination, whether an event was placed on a date because a source
gave that date or because it seemed about right.

This is one model call against the compiled world, made before any actor is invoked. It
is not a gate and it cannot pass a world the gates refuse; it is a way to notice that we
are about to spend several minutes and a hundred provider calls simulating something
obviously wrong, and to send that finding into targeted repair instead.

The questions it asks are the ones the mechanical checks do not cover. The ones they do
cover are left to them, because a model's opinion about whether a roster is complete is
worth less than a comparison against the evidence store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .compiled import CompiledWorld
from .errors import GatewayError
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .world_compiler import terminal_producers

__all__ = ["WorldReview", "review_world"]

# The questions worth a model's judgement, with whether a "no" should stop the rollout
# and go to repair. Completeness, grounding, producer lineage and coverage are absent on
# purpose: those are decided against the evidence store, not by asking.
_QUESTIONS: tuple[tuple[str, str, bool], ...] = (
    (
        "contract_is_exact",
        "Does the compiled resolution condition state exactly what the question asks — "
        "same subject, same threshold, same units, same window? A condition that is "
        "close but not the same resolves a different question.",
        True,
    ),
    (
        "representation_scale_is_right",
        "Is each causal producer modeled at the level that actually produces the "
        "outcome — an individual where a person decides, an organization where a body "
        "acts as one, an operational or population process where the outcome is "
        "throughput or aggregate behavior rather than anyone's choice?",
        True,
    ),
    (
        "nothing_unsupported_was_added",
        "Does the world contain any specific detail — a meeting, a position, a "
        "relationship, a quantity, a date — that the evidence does not support and that "
        "is not declared as an uncertainty? Answer no if any such detail is present.",
        True,
    ),
    (
        "private_and_future_states_are_labeled",
        "Are unobservable things — private preferences, intentions, future data, "
        "responses — represented as uncertainty or marked inference, rather than "
        "asserted as established fact?",
        False,
    ),
    (
        "event_timing_is_evidence_grounded",
        "Is each dated event on the calendar there because a source gives that date, "
        "rather than because the date seems plausible?",
        False,
    ),
    (
        "important_parts_are_present",
        "Is any organization, process or population that materially affects this "
        "outcome missing from the world entirely?",
        False,
    ),
)


@dataclass(frozen=True)
class WorldReview:
    """The review's answers, and whether they warrant repair before rollout."""

    answers: tuple[tuple[str, bool, str], ...] = ()  # (key, ok, why)
    failed_blocking: tuple[str, ...] = ()
    concerns: tuple[str, ...] = ()
    error: str = ""

    @property
    def should_repair(self) -> bool:
        return bool(self.failed_blocking)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answers": [{"question": k, "ok": ok, "why": why} for k, ok, why in self.answers],
            "blocking_failures": list(self.failed_blocking),
            "concerns": list(self.concerns),
            "error": self.error,
        }

    def repair_instruction(self) -> str:
        lines = [
            "A review of your compiled world before simulation found these problems. "
            "Fix exactly these and change nothing else:"
        ]
        by_key = {k: why for k, ok, why in self.answers if not ok}
        lines.extend(f"- {k}: {by_key.get(k, '')}" for k in self.failed_blocking)
        lines.append(
            "Do not invent support for anything. If the evidence does not establish "
            "something, represent it as an uncertainty or leave it out."
        )
        return "\n".join(lines)


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
        "calendar": [
            {"node": n.node_id, "at": n.at.isoformat() if n.at else None, "what": n.description}
            for n in spec.process.nodes
        ],
        "external_processes": [
            {
                "id": p.process_id,
                "what": p.description,
                "occurrences": [o.at.isoformat() if o.at else None for o in p.occurrences],
            }
            for p in spec.external_processes
        ],
        "uncertainties": [
            {"variable": u.variable_id, "why_unknown": u.why_unknown}
            for u in compiled.uncertainty_variables
        ],
        "terminal_producers": {k: list(v) for k, v in sorted(terminal_producers(spec).items())},
    }


def review_world(
    compiled: CompiledWorld,
    evidence: EvidenceView,
    gateway: ModelGateway,
    *,
    question: str,
    evidence_render: str,
) -> WorldReview:
    """Ask whether this is the right world. Never raises: a review that cannot run is
    recorded as not having run, and the rollout proceeds under the mechanical gates."""

    body = "\n".join(f"{key}: {text}" for key, text, _ in _QUESTIONS)
    prompt = "\n\n".join(
        [
            "You are auditing a compiled simulation world BEFORE it is run, to catch a "
            "world that is obviously not the one the question is about. Be concrete and "
            "be willing to say no.",
            f"QUESTION THE WORLD MUST RESOLVE: {question}",
            "## THE COMPILED WORLD\n"
            + json.dumps(summarize_world(compiled), indent=2, sort_keys=True, default=str),
            "## THE VERIFIED EVIDENCE IT WAS BUILT FROM\n" + evidence_render,
            "## ANSWER EACH\n" + body,
            'Reply with JSON {"answers": [{"question": "<key>", "ok": true|false, '
            '"why": "<one sentence, citing what in the world or the evidence decided it>"}]}. '
            "ok=true means the world is satisfactory on that question.",
        ]
    )
    try:
        resp = gateway.generate(
            GatewayRequest(
                task_kind="world_review",
                prompt=prompt,
                context={"question": question},
                seed=int(prompt_hash("review" + question)[:8], 16),
                expected_keys=("answers",),
            )
        )
    except GatewayError as exc:
        return WorldReview(error=f"the review could not run: {exc}")

    blocking = {key for key, _, is_blocking in _QUESTIONS if is_blocking}
    known = {key for key, _, _ in _QUESTIONS}
    answers: list[tuple[str, bool, str]] = []
    failed: list[str] = []
    concerns: list[str] = []
    for item in resp.data.get("answers", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("question", "")).strip()
        if key not in known:
            continue
        ok = item.get("ok")
        ok_bool = ok if isinstance(ok, bool) else str(ok).strip().lower() in ("true", "yes", "1")
        why = str(item.get("why", "")).strip()
        answers.append((key, ok_bool, why))
        if not ok_bool:
            (failed if key in blocking else concerns).append(
                key if key in blocking else f"{key}: {why}"
            )
    return WorldReview(
        answers=tuple(answers),
        failed_blocking=tuple(failed),
        concerns=tuple(concerns),
    )
