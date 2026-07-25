"""Did the simulation actually simulate? — asked once, after a completed run.

The pre-rollout review (:mod:`world_review`) attacks the compiled world before any
budget is spent. This is the other bookend: an audit of what the run *did*, from the
run's own record, after every branch has finished.

Two layers. The mechanical layer reads the event ledger, the actor decision records
and the terminal lineage directly and needs no model at all: an actor invoked twice
with byte-identical intent did not decide twice; a branch that resolved YES on a term
nothing produced staged its outcome; a per-branch outcome map that is just the branch
conditions read back, over symmetric-ignorance weights, is the forecast repeating its
initialization. The model layer is one optional call that judges what only judgement
can: whether time advanced realistically, whether anyone knew the impossible, whether
the trajectory resembles how the real event could unfold.

The audit also *classifies* the run — mechanically, never by asking — as a genuine
actor simulation, an operational process simulation, a factual resolution (the record
had already answered the question), unresolved, or invalid. Like the pre-rollout
review it is advisory and can never destroy the run it describes: ``audit_trajectory``
never raises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .compiled import CompiledWorld
from .engine import ActorDecisionRecord, RunResult, terminal_lineage
from .errors import GatewayError
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .models import BranchOutcome, WeightProvenance
from .world_review import AuditFinding, _parse_findings

__all__ = ["TrajectoryAudit", "audit_trajectory", "mechanical_trajectory_checks"]

CLASSIFICATIONS = (
    "genuine_actor_simulation",
    "operational_process_simulation",
    "factual_resolution",
    "unresolved",
    "invalid",
)

# The judgements only a model can make about a finished trajectory. Everything that can
# be computed from the run's own data is in the mechanical layer instead.
_MODEL_QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "time_advanced_realistically",
        "Did simulated time advance realistically for this kind of process, or did the "
        "trajectory collapse into a single instant or an implausible rhythm?",
    ),
    (
        "actor_calls_causally_motivated",
        "For each actor invocation, was there a real causal reason it was called at "
        "that moment — something that reached it — rather than a turn being taken?",
    ),
    (
        "impossible_knowledge",
        "Did any actor know something it could not yet know — a future value, another "
        "actor's private state, or its own branch's uncertain outcome?",
    ),
    (
        "delivery_order_respected",
        "Did communications obey delivery order — no one reacting to information "
        "before the trajectory shows it reaching them?",
    ),
    (
        "consequences_externally_validated",
        "Did actions cause their consequences through environment validation, rather "
        "than actors asserting outcomes directly?",
    ),
    (
        "resembles_real_event",
        "Does the trajectory resemble how the real event could plausibly unfold?",
    ),
)


@dataclass(frozen=True)
class TrajectoryAudit:
    """The audit's findings (mechanical + model) and the run's mechanical class.

    ``classification`` is one of :data:`CLASSIFICATIONS`, decided from the run's own
    data and never by a model. ``error`` records a model layer (or whole-audit) fault;
    an audit with an error makes no claim beyond what its findings state.
    """

    findings: tuple[AuditFinding, ...] = ()
    classification: str = "unresolved"
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "findings": [f.as_dict() for f in self.findings],
            "error": self.error,
        }


def audit_trajectory(
    compiled: CompiledWorld,
    run_result: RunResult,
    gateway: ModelGateway | None,
    *,
    question: str,
) -> TrajectoryAudit:
    """Audit a completed run. Never raises: this describes a run that already finished,
    and a fault in the description must not be able to destroy the thing described.
    With ``gateway=None`` only the mechanical layer runs."""

    try:
        return _audit(compiled, run_result, gateway, question=question)
    except Exception as exc:  # noqa: BLE001 — advisory, never fatal (see docstring)
        return TrajectoryAudit(error=f"the audit could not run: {type(exc).__name__}: {exc}")


def _audit(
    compiled: CompiledWorld,
    run_result: RunResult,
    gateway: ModelGateway | None,
    *,
    question: str,
) -> TrajectoryAudit:
    mechanical = mechanical_trajectory_checks(compiled, run_result)
    classification = _classify(compiled, run_result, mechanical)
    if gateway is None:
        return TrajectoryAudit(findings=mechanical, classification=classification)
    model_findings, error = _model_review(run_result, gateway, question=question)
    return TrajectoryAudit(
        findings=mechanical + model_findings,
        classification=classification,
        error=error,
    )


# ---------------------------------------------------------------------------
# Mechanical checks — computed from the run's own data, no model involved
# ---------------------------------------------------------------------------


def mechanical_trajectory_checks(
    compiled: CompiledWorld, run_result: RunResult
) -> tuple[AuditFinding, ...]:
    """Every check that can be decided from the run record alone.

    Each check contributes at least one finding: its violations, or a single PASS
    stating what was checked — so an audit that found nothing shows what it looked at
    rather than showing nothing.
    """

    findings: list[AuditFinding] = []
    findings.extend(_repeated_equivalent_calls(run_result))
    findings.extend(_branch_label_leakage(compiled, run_result))
    findings.extend(_result_equals_initialization(compiled, run_result))
    findings.extend(_unproduced_yes(compiled, run_result))
    findings.extend(_time_advanced(run_result))
    return tuple(findings)


def _pass(key: str, checked: str) -> AuditFinding:
    return AuditFinding(
        key=key,
        severity="PASS",
        finding=checked,
        evidence_basis="computed from the run's own record",
    )


def _repeated_equivalent_calls(run_result: RunResult) -> list[AuditFinding]:
    """The same actor making the byte-identical decision twice in one branch is the
    runtime spinning, not a person deciding twice."""

    groups: dict[tuple[str, str, str], int] = {}
    for d in run_result.actor_decisions:
        key = (d.branch_id, d.actor_id, json.dumps(d.intent, sort_keys=True, default=str))
        groups[key] = groups.get(key, 0) + 1
    out = [
        AuditFinding(
            key="repeated_equivalent_calls",
            severity="HIGH",
            finding=(
                f"actor {actor!r} in branch {branch!r} was invoked {n} times with "
                "byte-identical intent — the same decision re-made, not a new one"
            ),
            evidence_basis=f"{n} actor decision records share intent JSON {intent[:120]!r}",
        )
        for (branch, actor, intent), n in sorted(groups.items())
        if n >= 2
    ]
    return out or [
        _pass(
            "repeated_equivalent_calls",
            "no actor repeated a byte-identical intent within any branch",
        )
    ]


def _decision_prompt(d: ActorDecisionRecord) -> str:
    """The byte-exact prompt the actor was sent, wherever this build records it."""

    prompt: Any = getattr(d, "exact_prompt", "")
    if isinstance(prompt, str) and prompt:
        return prompt
    rendered = dict(d.decision_context).get("rendered_prompt", "")
    return rendered if isinstance(rendered, str) else ""


def _is_before(when: datetime | None, release: datetime | None) -> bool:
    """Did this decision happen before anything was released into its branch?"""

    if release is None:
        return True  # nothing was ever released, so everything precedes it
    if when is None:
        return True  # an unreadable time is checked, not excused
    try:
        return when < release
    except TypeError:  # naive vs aware timestamps — check rather than crash
        return True


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _branch_label_leakage(compiled: CompiledWorld, run_result: RunResult) -> list[AuditFinding]:
    """A prompt that names the branch's condition value as settled fact before any
    ``release_data`` event delivered it is the runtime whispering the answer.

    Information released into the world legitimately reaches actors, so a decision made
    at or after the branch's first ``release_data`` event is not flagged. This is the
    conservative form: it requires the literal word "branch" beside the condition
    value, and reports MEDIUM rather than claiming certainty.
    """

    conditions = {s.scenario_id: s.conditions for s in compiled.scenario_set.scenarios}
    first_release: dict[str, datetime] = {}
    for ev in run_result.event_ledger:
        if ev.kind != "release_data":
            continue
        cur = first_release.get(ev.branch_id)
        if cur is None or ev.time < cur:
            first_release[ev.branch_id] = ev.time

    out: list[AuditFinding] = []
    for d in run_result.actor_decisions:
        prompt = _decision_prompt(d)
        if not prompt or "branch" not in prompt.lower():
            continue
        if not _is_before(_parse_time(d.branch_time), first_release.get(d.branch_id)):
            continue  # the value had legitimately entered the world by then
        low = prompt.lower()
        for variable, value in conditions.get(d.branch_id, ()):
            if value and str(value).lower() in low:
                release = first_release.get(d.branch_id)
                out.append(
                    AuditFinding(
                        key="branch_label_leakage",
                        severity="MEDIUM",
                        finding=(
                            f"the prompt for {d.actor_id!r} in branch {d.branch_id!r} names "
                            f"the branch condition value {value!r} (variable {variable!r}) "
                            "before any release_data event delivered it"
                        ),
                        evidence_basis=(
                            f"prompt contains 'branch' and {value!r}; first release_data in "
                            "this branch is "
                            + (f"at {release.isoformat()}" if release else "absent")
                        ),
                    )
                )
                break
    return out or [
        _pass(
            "branch_label_leakage",
            "no actor prompt names its branch's condition value before a release_data "
            "event delivered it",
        )
    ]


def _result_equals_initialization(
    compiled: CompiledWorld, run_result: RunResult
) -> list[AuditFinding]:
    """If every branch's YES/NO is just its own condition read back, and the condition
    weights are symmetric ignorance, the output probability is the input prior."""

    key = "result_equals_initialization"
    outcomes = run_result.branch_outcomes
    checked = "branch outcomes are not a pure function of their symmetric-ignorance conditions"
    if len(outcomes) < 2 or not all(b.resolved and b.outcome for b in outcomes):
        return [_pass(key, checked)]

    scenarios = {s.scenario_id: s for s in compiled.scenario_set.scenarios}
    if not all(
        (s := scenarios.get(b.branch_id)) is not None
        and s.provenance is WeightProvenance.SYMMETRIC_IGNORANCE
        for b in outcomes
    ):
        return [_pass(key, checked)]

    separating = ""
    variables = sorted({var for b in outcomes for var, _ in b.key_conditions})
    for variable in variables:
        value_outcomes: dict[str, set[str]] = {}
        covered = True
        for b in outcomes:
            value = dict(b.key_conditions).get(variable)
            if value is None:
                covered = False
                break
            value_outcomes.setdefault(str(value), set()).add(str(b.outcome))
        if (
            covered
            and all(len(v) == 1 for v in value_outcomes.values())
            and len({next(iter(v)) for v in value_outcomes.values()}) > 1
        ):
            separating = variable
            break
    if not separating:
        return [_pass(key, checked)]
    return [
        AuditFinding(
            key=key,
            severity="HIGH",
            finding="the forecast repeats its initialization",
            evidence_basis=(
                f"every branch's YES/NO is a function of its condition on {separating!r} "
                "alone, and every branch weight carries symmetric-ignorance provenance"
            ),
        )
    ]


def _unproduced_yes(compiled: CompiledWorld, run_result: RunResult) -> list[AuditFinding]:
    """A YES whose terminal lineage contains an unproduced term was asserted, not
    produced: nothing that happened in the branch wrote it and no citation established
    it before the window opened."""

    out: list[AuditFinding] = []
    for b in run_result.branch_outcomes:
        if not (b.resolved and b.outcome == "YES"):
            continue
        world = run_result.final_worlds.get(b.branch_id)
        if world is None:
            out.append(
                AuditFinding(
                    key="unproduced_yes",
                    severity="CRITICAL",
                    finding=(
                        f"branch {b.branch_id!r} resolved YES but recorded no final world, "
                        "so nothing shows what produced the outcome"
                    ),
                    evidence_basis="run_result.final_worlds has no entry for this branch",
                )
            )
            continue
        for term in terminal_lineage(world, compiled.spec.terminal, compiled.spec):
            if not term.get("unproduced"):
                continue
            out.append(
                AuditFinding(
                    key="unproduced_yes",
                    severity="CRITICAL",
                    finding=(
                        f"branch {b.branch_id!r} resolved YES while terminal term "
                        f"{term.get('terminal_term')!r} was never produced by anything "
                        "that happened"
                    ),
                    evidence_basis=(
                        f"terminal lineage shows writer_count={term.get('writer_count')} "
                        f"and established_by_evidence={term.get('established_by_evidence')}"
                    ),
                )
            )
    return out or [
        _pass("unproduced_yes", "every YES branch's terminal terms were produced or established")
    ]


def _time_advanced(run_result: RunResult) -> list[AuditFinding]:
    """A resolved branch whose every event carries one timestamp reached its outcome
    without time passing — a tableau, not a trajectory."""

    resolved = {b.branch_id for b in run_result.branch_outcomes if b.resolved}
    stamps: dict[str, set[datetime]] = {}
    counts: dict[str, int] = {}
    for ev in run_result.event_ledger:
        stamps.setdefault(ev.branch_id, set()).add(ev.time)
        counts[ev.branch_id] = counts.get(ev.branch_id, 0) + 1
    out = [
        AuditFinding(
            key="time_advanced",
            severity="MEDIUM",
            finding=f"simulated time never advanced in branch {bid!r}, yet the branch resolved",
            evidence_basis=(
                f"all {counts[bid]} event(s) in the branch carry the same timestamp "
                f"{next(iter(stamps[bid])).isoformat()}"
            ),
        )
        for bid in sorted(resolved)
        if len(stamps.get(bid, set())) == 1
    ]
    return out or [
        _pass("time_advanced", "every resolved branch's events span more than one timestamp")
    ]


# ---------------------------------------------------------------------------
# Classification — mechanical, from the run record
# ---------------------------------------------------------------------------


def _classify(
    compiled: CompiledWorld,
    run_result: RunResult,
    mechanical: tuple[AuditFinding, ...],
) -> str:
    if any(f.severity == "CRITICAL" for f in mechanical):
        return "invalid"
    resolved = [b for b in run_result.branch_outcomes if b.resolved]
    if not resolved:
        return "unresolved"
    if _is_factual_resolution(compiled, run_result, resolved):
        return "factual_resolution"
    if not run_result.actor_decisions:
        return "operational_process_simulation"
    return "genuine_actor_simulation"


def _is_factual_resolution(
    compiled: CompiledWorld, run_result: RunResult, resolved: list[BranchOutcome]
) -> bool:
    """Every terminal term, in every resolved branch, established by pre-window
    evidence with no runtime writer: the record answered the question before the
    simulation began."""

    saw_term = False
    for b in resolved:
        world = run_result.final_worlds.get(b.branch_id)
        if world is None:
            return False
        for term in terminal_lineage(world, compiled.spec.terminal, compiled.spec):
            saw_term = True
            if term.get("writer_count") or not term.get("established_by_evidence"):
                return False
    return saw_term


# ---------------------------------------------------------------------------
# The model layer — one call, following the world_review pattern, never raises
# ---------------------------------------------------------------------------


def _trajectory_digest(run_result: RunResult) -> dict[str, Any]:
    """Per branch: the ordered event ledger and every actor call, compact enough to
    put in front of a model and complete enough to judge from."""

    branches: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def branch(bid: str) -> dict[str, list[dict[str, Any]]]:
        return branches.setdefault(bid, {"events": [], "actor_calls": []})

    for ev in run_result.event_ledger:  # already in per-branch order
        branch(ev.branch_id)["events"].append(
            {
                "time": ev.time.isoformat(),
                "kind": ev.kind,
                "actor": ev.actor_id,
                "payload": json.dumps(ev.payload_dict, sort_keys=True, default=str)[:200],
            }
        )
    for d in run_result.actor_decisions:
        branch(d.branch_id)["actor_calls"].append(
            {
                "time": d.branch_time,
                "wake_reason": d.wake_reason,
                "intent_mode": str(d.intent.get("mode", "")),
                "intent_action": str(d.intent.get("action_id", "")),
                "validation": d.validation_status,
            }
        )
    for b in branches.values():
        b["actor_calls"].sort(key=lambda c: str(c["time"]))
    return {
        "branches": branches,
        "outcomes": {
            b.branch_id: {
                "resolved": b.resolved,
                "outcome": b.outcome,
                "conditions": dict(b.key_conditions),
            }
            for b in run_result.branch_outcomes
        },
    }


def _model_review(
    run_result: RunResult, gateway: ModelGateway, *, question: str
) -> tuple[tuple[AuditFinding, ...], str]:
    body = "\n".join(f"{key}: {text}" for key, text in _MODEL_QUESTIONS)
    prompt = "\n\n".join(
        [
            "You are auditing the trajectory of a COMPLETED simulation, to determine "
            "whether it actually simulated the production of its outcome or merely "
            "staged it. ATTACK the trajectory; do not praise it. Ground every attack "
            "in the record below.",
            f"QUESTION THE SIMULATION WAS RESOLVING: {question}",
            "## THE TRAJECTORY, PER BRANCH\n"
            + json.dumps(_trajectory_digest(run_result), indent=2, sort_keys=True, default=str),
            "## ANSWER EACH\n" + body,
            "For every question return exactly one finding object: "
            '{"key": "<key>", "severity": "CRITICAL"|"HIGH"|"MEDIUM"|"LOW"|"PASS", '
            '"finding": "<one sentence: what is wrong, or why the trajectory survives>", '
            '"evidence_basis": "<one sentence citing the events or actor calls above>"}. '
            "CRITICAL means the trajectory is meaningless; HIGH means materially wrong; "
            "MEDIUM is a real concern; LOW is minor; PASS means the trajectory survives "
            "your attack on that question. A finding without a stated evidence basis "
            "from the provided material must be severity LOW. "
            'Reply with JSON {"findings": [<one object per key, all 6 keys>]}.',
        ]
    )
    try:
        resp = gateway.generate(
            GatewayRequest(
                task_kind="trajectory_audit",
                prompt=prompt,
                context={"question": question},
                seed=int(prompt_hash("trajectory_audit" + question)[:8], 16),
                expected_keys=("findings",),
            )
        )
    except GatewayError as exc:
        return (), f"the trajectory review could not run: {exc}"
    return _parse_findings(resp.data, known=tuple(k for k, _ in _MODEL_QUESTIONS)), ""
