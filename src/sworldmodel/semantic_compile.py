"""The semantic compiler path: plan → independent review → validate → lower.

The direct compiler asks one model call to understand the causal world AND author an
internally consistent executable program. This path splits the two jobs. The planning
call produces only causal meaning in the semantic schema; an independent review call
judges that meaning against the evidence with no access to the planner's reasoning;
mechanical validation and deterministic lowering then own every symbol and every piece
of executable syntax. The output is the exact compilation dict the direct path
produces, so `assemble_bundle`, every compile gate, the runtime, the auditors and the
replay viewer are the same executor for both modes.

Bounded, not open-ended: at most one targeted revision (plus one reparse of unreadable
JSON), then the refusal is real and carries the exact unresolved reasons.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .coverage import evidence_checklist
from .errors import GatewayError, WorldIntegrityError
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .semantic_lowering import LoweringGap, lower_plan
from .semantic_plan import (
    COMPARISONS,
    PROCESS_KINDS,
    REPRESENTATION_SCALES,
    STATE_TYPES,
    STRUCTURAL_TYPES,
    TERMINAL_FORMS,
    WEIGHT_PROVENANCES,
    SemanticPlan,
    SemanticPlanError,
    parse_semantic_plan,
    validate_semantic_plan,
)
from .world_compiler import _normalize_compilation, render_evidence

# The exact output format, stated rather than guessed. Object types are universal world
# structure; every real-world meaning inside them is open-ended natural language.
SEMANTIC_SCHEMA = f"""Return a SINGLE JSON object with exactly these keys:
{{
 "resolution": {{
   "question": "<the exact question>",
   "yes_condition": "<ordinary-language YES rule>",
   "subject_entity": "<who or what the question is about>",
   "resolution_units": "<what is counted or decided>",
   "target_outcome": "<one line: what YES means>",
   "expected_participants": <int or null — decision-relevant actors the EVIDENCE names>,
   "evidence_claim_ids": ["<claim ids supporting the resolution rule>"]
 }},
 "entities": [{{
   "name": "<canonical real-world name>",
   "structural_type": "<one of {list(STRUCTURAL_TYPES)}>",
   "role": "<their real role>",
   "representation_scale": "<one of {list(REPRESENTATION_SCALES)}>",
   "represents_count": <int or null — how many real members this object stands for>,
   "decides": <true if its own decisions can move the outcome>,
   "authority": "<what it may do, in ordinary language>",
   "why_material": "<why it could change the answer>",
   "evidence_claim_ids": ["..."]
 }}],
 "states": [{{
   "name": "<ordinary-language name, unique>",
   "owner": "<entity name or 'world'>",
   "state_type": "<one of {list(STATE_TYPES)}>",
   "unit": "<unit or ''>",
   "initial": <verified value, or "UNKNOWN" when the evidence does not establish one>,
   "why_material": "...",
   "evidence_claim_ids": ["<required when initial is a precise number>"]
 }}],
 "events": [{{
   "name": "<unique name>",
   "meaning": "<unrestricted ordinary-language meaning of the occurrence>",
   "participants": {{"<semantic role>": "<entity name>"}},
   "visibility": "public|private",
   "information_created": "<what becomes known, or ''>",
   "evidence_claim_ids": ["..."]
 }}],
 "affordances": [{{
   "name": "<unique name>",
   "meaning": "<what the actor does, in ordinary language>",
   "actor": "<entity name with decides=true>",
   "target": "<entity name or ''>",
   "authority_required": "<ordinary language>",
   "preconditions": "<ordinary language or ''>",
   "visibility": "public|private",
   "duration_seconds": <int>,
   "changes": [<see CHANGES>],
   "evidence_claim_ids": ["..."]
 }}],
 "processes": [{{
   "name": "<unique name>",
   "meaning": "<what this moment or mechanism is>",
   "kind": "<one of {list(PROCESS_KINDS)}>",
   "participants": ["<entity names — actor_moment: who gains the opportunity to act>"],
   "allowed_affordances": ["<affordance names available at the moment, or omit for all>"],
   "inputs": ["<state names read>"],
   "at": "<ISO datetime for an actor_moment's dated occasion, else null>",
   "deadline": "<ISO or null>",
   "occurrences": [{{
     "description": "...",
     "at": "<ISO datetime or null>",
     "after_process": "<process name it follows, or null>",
     "delay_seconds": <int>,
     "changes": [<see CHANGES>]
   }}],
   "information_produced": "<or ''>",
   "evidence_claim_ids": ["..."]
 }}],
 "uncertainties": [{{
   "name": "<unique name>",
   "what_unknown": "...",
   "why_unknown": "...",
   "affects_state": "<state name the draw sets — NEVER the state the terminal reads>",
   "release_at": "<ISO or null>",
   "alternatives": [{{
     "value": <the value>,
     "weight": <float or null — null when nothing supports a split>,
     "provenance": "<one of {list(WEIGHT_PROVENANCES)}>",
     "grounding": "<what supports this alternative>",
     "evidence_claim_ids": ["..."]
   }}]
 }}],
 "terminal": {{
   "form": "<one of {list(TERMINAL_FORMS)}>",
   "event": "<event name — for event_exists>",
   "state": "<state name — for state_equals / quantity_comparison>",
   "value": <value — for state_equals>,
   "comparison": "<one of {list(COMPARISONS)} — for quantity_comparison / record_count>",
   "threshold": <number or value object — for quantity_comparison / record_count>,
   "record_event": "<event name — for record_count>",
   "parts": [<nested terminal objects — for all_of / any_of / not>]
 }},
 "terminal_producer_note": "<what produces the resolving state, through which causal
   path, and why initialization or an uncertainty draw does not already write the
   answer>",
 "world_facts": [{{"text": "...", "evidence_claim_ids": ["..."]}}]
}}

CHANGES — the only universal change operations (they apply to dynamically named
real-world objects; there is no domain event list):
  {{"op": "set", "target": "<state name>", "value": <literal |
      {{"kind":"state","state":"<state name>"}} |
      {{"kind":"product"|"sum","parts":[<values>]}}>}}
  {{"op": "increase"|"decrease", "target": "<state name>", "amount": <same value forms>}}
  {{"op": "record_event", "target": "<event name>", "detail": "<what is recorded>"}}
  {{"op": "send", "target": "<information description>", "recipients": ["<entity names>"],
      "detail": "<the information>"}}

Never invent runtime identifiers, field names, effect operations or expression trees —
name things in ordinary language and code will mint every symbol.

CONSISTENCY REQUIREMENTS (checked mechanically; a violation costs a revision round):
- every name a change, process, affordance, uncertainty or terminal references must
  appear in its own declaration list (states in "states", events in "events", …);
- every actor_moment process needs a dated occasion: "at" or "deadline";
- every operational / scheduled_release process needs at least one occurrence with
  "at" or "after_process";
- every entity with decides=true needs at least one affordance and cited evidence;
- a precise initial number needs evidence_claim_ids, otherwise write "UNKNOWN";
- the state the terminal reads must be written by an affordance or process occurrence,
  or carry a cited initial value — and must never be set by an uncertainty, nor set
  from uncertainty draws alone (bare copy OR arithmetic over only-uncertain states):
  production needs at least one evidence-grounded input;
- only actor_moment processes have participants, and every actor with an affordance
  must be a participant of at least one actor_moment — that dated occasion is the only
  thing that ever invokes them;
- every occurrence and every actor_moment is dated STRICTLY AFTER the cutoff:
  anything already done by the cutoff is a cited initial state value or a world_facts
  entry, never a simulated occurrence — the simulation cannot re-perform history;
- every causally material item the EVIDENCE contains must appear somewhere in the plan:
  as an entity, a state, a process, an uncertainty — or, when it is verified context
  that shapes the world without being part of the mechanism, as a world_facts entry
  citing its claim ids. A verified claim the plan silently omits is a coverage refusal;
  this applies just as much when the record already settles the question."""

_PLAN_RULES = """RULES FOR THE CAUSAL WORLD (they are about meaning, not syntax):

WHO COULD CHANGE THE ANSWER, NOT ONLY WHO PERFORMS THE FINAL ACT. Ask of every material
party: absent from the world, could the answer differ? Include exactly those, at the
representation scale that is causally faithful, with represents_count when one object
stands for many real members. Never turn independent decision-makers into one actor,
and never expand an aggregate that genuinely decides as one unit.

PRODUCTION, NOT REPORTING. A reported figure is produced by operations — throughput,
demand, constraints — not by the report. Model the producing mechanism as operational
processes whose occurrences increase the quantity across the window, with uncertainty
on the drivers (rate, demand, disruption), never on the total itself.

ALREADY SETTLED vs STILL OPEN. If verified claims available at the cutoff establish
that the outcome has already happened, say so: give the resolving state its established
initial value WITH those claim ids and no producer. Otherwise start the resolving state
neutral or UNKNOWN and let declared affordances and processes produce it inside the
window. Never mix the two.

UNCERTAINTY IS THE INPUT, NEVER THE ANSWER. An uncertainty sets what the world does TO
the actors — incoming data, demand, a release, an interpretation. It never sets the
state the terminal reads, and no affordance or process may set the terminal state to a
bare copy of an uncertain state. Weights need grounding; when nothing supports a split,
set weight null with provenance symmetric_ignorance_assumption — never invent 50/50.

EVERY NUMBER NEEDS A SOURCE. A precise initial quantity cites claim ids or is UNKNOWN.
UNKNOWN stays UNKNOWN — the runtime treats reading it as honestly unresolved.

ACTORS ARE REAL OCCUPANTS OF REAL ROLES. A verified office and authority is sufficient
grounding (cite the claims); a name with nothing behind it is not an actor. Give each
deciding entity the genuine alternative affordances its role affords — including the
ones that would resolve the question NO — and place its real dated occasions to act as
actor_moment processes with at/deadline inside the window."""


def _plan_prompt(
    question: str,
    as_of: datetime,
    horizon: datetime,
    evidence: str,
    *,
    checklist: str = "",
    extra_instruction: str = "",
    prior_plan: dict[str, Any] | None = None,
    corrections: list[str] | None = None,
) -> str:
    parts = [
        "Design the causal world for this question as a SEMANTIC PLAN. Describe meaning "
        "only — code will mint every identifier and every piece of executable syntax. "
        "Do NOT assume the question is a vote, committee, negotiation, market or any "
        "other template; discover the real process from the evidence and cite ONLY "
        "claim ids present in it.",
        f"QUESTION: {question}",
        f"as_of: {as_of.isoformat()}   horizon: {horizon.isoformat()}",
        f"EVIDENCE (id | proposition = value [meta]):\n{evidence}",
        (
            "WHAT VERIFIED EVIDENCE CONTAINS. Every causally material item below must "
            "appear in the plan — as an entity, state, process, uncertainty, or a "
            "world_facts entry citing its claim ids. The coverage gate checks the "
            "compiled world against exactly this inventory:\n" + checklist
        )
        if checklist
        else "",
        _PLAN_RULES,
        SEMANTIC_SCHEMA,
    ]
    if extra_instruction:
        parts.insert(1, extra_instruction)
    if prior_plan is not None and corrections:
        parts.insert(
            1,
            "REVISE the previous semantic plan. Apply exactly these corrections and "
            "change nothing else:\n- "
            + "\n- ".join(corrections)
            + "\nPREVIOUS PLAN:\n"
            + json.dumps(prior_plan, indent=1, default=str),
        )
    return "\n\n".join(p for p in parts if p)


_REVIEW_CHECKLIST = """Check, against the evidence only:
1. Are people, organizations, populations or processes that could change the answer
   absent from the plan?
2. Is anything present that is decorative — unable to move any terminal-relevant state?
3. Is the representation scale right — no independent decision-makers compressed into
   one actor, no aggregate needlessly expanded, represents_count faithful?
4. Is production modeled rather than reporting — does anything report a result no
   mechanism produces?
5. Is information flow realistic — do actors learn things through declared events and
   sends, or by magic?
6. Are the schedules supported by evidence — do the dated occasions really exist?
7. Is every precise number grounded in cited claims?
8. Are private assumptions presented as fact anywhere?
9. Does every terminal term have a real producer (affordance, process occurrence, or a
   cited established initial value)?
10. Does an uncertainty or an initial value already determine the answer?
11. Is the evidence sufficient for this world at all?
12. Is the world causally complete without unnecessary breadth?"""


def _review_prompt(
    question: str,
    as_of: datetime,
    horizon: datetime,
    evidence: str,
    plan_json: dict[str, Any],
) -> str:
    return "\n\n".join(
        [
            "You are an independent reality reviewer. Judge whether this SEMANTIC PLAN "
            "is the faithful causal world for the question, given ONLY the verified "
            "evidence. You did not write the plan and owe it nothing.",
            f"QUESTION: {question}",
            f"as_of: {as_of.isoformat()}   horizon: {horizon.isoformat()}",
            f"EVIDENCE (id | proposition = value [meta]):\n{evidence}",
            f"SEMANTIC PLAN:\n{json.dumps(plan_json, indent=1, default=str)}",
            _REVIEW_CHECKLIST,
            # Without these standing rules the reviewers fight each other: this one
            # abstained over a labeled-ignorance uncertainty on a cited anchor — the
            # exact shape the pre-rollout review is required to honor — so the same
            # frozen store oscillated between approve-then-destroy and abstain across
            # runs at temperature zero.
            """What is already legal in this system — do not abstain over it, and do
not demand its removal:
- A future quantity the evidence constrains only qualitatively may be modeled as an
  uncertainty over a cited present anchor, with alternatives carrying labeled
  symmetric-ignorance weights. That is the honest shape of not knowing: the runtime
  reports bounds, not a calibrated point, wherever such weights matter. Where the
  record quantifies something, the alternative's value must cite it; where it does
  not, declared ignorance is legal. Prefer REVISE naming what an alternative should
  be anchored to.
- An initial state value established by cited pre-cutoff record may already satisfy
  the terminal (a factual resolution). Attack the citation's sufficiency — does the
  record establish the outcome as the question means it — never the absence of
  in-window re-production.

ABSTAIN is only for a question whose outcome has NO cited anchor and NO representable
producer at all — where every faithful world would have to invent its facts. If a
faithful-but-bounded world exists, REVISE toward it instead.""",
            """Return JSON:
{"verdict": "APPROVE" | "REVISE" | "ABSTAIN",
 "reasons": ["<one line per finding>"],
 "corrections": ["<for REVISE: exact semantic corrections, one per line — name the
   object and the change, e.g. 'add entity X because …', 'state Y initial must be
   UNKNOWN because no claim establishes it', 'terminal producer must be Z'>"]}
ABSTAIN means no faithful world exists even after every legal revision above.""",
        ]
    )


def _call_planner(
    gateway: ModelGateway,
    question: str,
    as_of: datetime,
    horizon: datetime,
    evidence: str,
    *,
    checklist: str = "",
    extra_instruction: str,
    prior: dict[str, Any] | None,
    corrections: list[str] | None,
    attempt: int,
) -> tuple[dict[str, Any], Any]:
    resp = gateway.generate(
        GatewayRequest(
            task_kind="semantic_plan",
            prompt=_plan_prompt(
                question,
                as_of,
                horizon,
                evidence,
                checklist=checklist,
                extra_instruction=extra_instruction,
                prior_plan=prior,
                corrections=corrections,
            ),
            context={"question": question, "attempt": attempt},
            seed=int(prompt_hash(f"semantic{question}{attempt}")[:8], 16),
            expected_keys=("resolution", "terminal"),
        )
    )
    data = resp.data
    return (data if isinstance(data, dict) else {}), resp


def _call_reviewer(
    gateway: ModelGateway,
    question: str,
    as_of: datetime,
    horizon: datetime,
    evidence: str,
    plan_json: dict[str, Any],
) -> tuple[str, list[str], list[str], Any]:
    try:
        resp = gateway.generate(
            GatewayRequest(
                task_kind="semantic_review",
                prompt=_review_prompt(question, as_of, horizon, evidence, plan_json),
                context={"question": question},
                seed=int(prompt_hash(f"semreview{question}")[:8], 16),
                expected_keys=("verdict",),
            )
        )
    except GatewayError:
        # An unreachable reviewer neither approves nor blocks: the mechanical validator
        # and every downstream gate still stand. Record the abstention honestly.
        return "REVIEWER_UNAVAILABLE", [], [], None
    data = resp.data if isinstance(resp.data, dict) else {}
    verdict = str(data.get("verdict") or "").strip().upper()
    reasons = [str(r) for r in data.get("reasons") or [] if str(r).strip()]
    corrections = [str(c) for c in data.get("corrections") or [] if str(c).strip()]
    if verdict not in ("APPROVE", "REVISE", "ABSTAIN"):
        verdict = "APPROVE" if not corrections else "REVISE"
    return verdict, reasons, corrections, resp


def semantic_compile_live(
    gateway: ModelGateway,
    question: str,
    as_of: datetime,
    horizon: datetime,
    view: EvidenceView,
    *,
    extra_instruction: str = "",
    structure_id: str = "primary",
    prior_plan: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Compile via the semantic path. Same signature and return contract as
    ``compile_world_spec_live`` — ``(compilation_dict, last_gateway_response)`` — so
    the two modes are interchangeable at every call site.

    ``prior_plan`` makes a repair a REVISION rather than a re-roll: a review-forced
    recompile that re-planned from scratch replaced a plan whose downside alternative
    cited the constraint claims with four uncited growth-only scenarios — the cited
    NO branch vanished and the run reported 1.0. With the prior plan supplied, the
    planner is held to the same discipline the validator rounds already use: apply
    exactly the named corrections and change nothing else.
    """

    evidence = render_evidence(view)
    checklist = evidence_checklist(view, as_of=as_of, horizon=horizon)
    known = frozenset(c.id for c in view.available())
    responses: list[Any] = []

    def build(
        prior: dict[str, Any] | None, corrections: list[str] | None, attempt: int
    ) -> tuple[SemanticPlan, dict[str, Any]]:
        raw, resp = _call_planner(
            gateway,
            question,
            as_of,
            horizon,
            evidence,
            checklist=checklist,
            extra_instruction=extra_instruction,
            prior=prior,
            corrections=corrections,
            attempt=attempt,
        )
        responses.append(resp)
        try:
            plan = parse_semantic_plan(raw)
        except SemanticPlanError as exc:
            # One reparse round: unreadable shape is a defect of one response, not of
            # the world. Name every field error exactly.
            raw2, resp2 = _call_planner(
                gateway,
                question,
                as_of,
                horizon,
                evidence,
                checklist=checklist,
                extra_instruction=extra_instruction,
                prior=raw,
                corrections=[f"fix the plan's shape: {e}" for e in exc.errors[:12]],
                attempt=attempt + 100,
            )
            responses.append(resp2)
            plan = parse_semantic_plan(raw2)  # a second failure propagates
            raw = raw2
        return plan, raw

    if prior_plan is not None and extra_instruction:
        plan, raw = build(
            prior_plan,
            [
                "apply the repair instruction above to the previous plan; keep every "
                "cited value, alternative, and structure the instruction does not name"
            ],
            0,
        )
    else:
        plan, raw = build(None, None, 0)
    errors = validate_semantic_plan(plan, as_of=as_of, horizon=horizon, known_claim_ids=known)
    validator_rounds = 0
    while errors and validator_rounds < 2:
        # Mechanical inconsistencies first, so the reviewer always judges a coherent
        # plan. Two bounded rounds, every finding named each time: the error list
        # shrinks monotonically when the fixes land, and a plan still broken after two
        # precise rounds has a real coherence problem.
        validator_rounds += 1
        plan, raw = build(raw, [f"validator: {e}" for e in errors[:16]], validator_rounds)
        errors = validate_semantic_plan(plan, as_of=as_of, horizon=horizon, known_claim_ids=known)
    if errors:
        raise WorldIntegrityError(
            "the semantic plan is invalid after validator rounds: " + "; ".join(errors[:6]),
            details={
                "failure": "semantic_plan_invalid",
                "recompilable": True,
                "semantic_errors": errors,
            },
        )

    # The independent review judges every plan that will be lowered — including one
    # the validator round produced. No plan reaches lowering unreviewed.
    verdict, reasons, corrections, rresp = _call_reviewer(
        gateway, question, as_of, horizon, evidence, raw
    )
    if rresp is not None:
        responses.append(rresp)
    if verdict == "ABSTAIN":
        raise WorldIntegrityError(
            "the independent reality review abstained: the evidence cannot support "
            "a faithful causal world for this question",
            details={
                "failure": "semantic_review_abstained",
                "recompilable": False,
                "reasons": reasons,
            },
        )
    if verdict == "REVISE" and corrections:
        # One targeted revision on the reviewer's exact corrections, then one
        # validator-only round if the revision broke a mechanical rule.
        plan, raw = build(raw, [f"reviewer: {c}" for c in corrections[:16]], 2)
        errors = validate_semantic_plan(plan, as_of=as_of, horizon=horizon, known_claim_ids=known)
        if errors:
            plan, raw = build(raw, [f"validator: {e}" for e in errors[:16]], 3)
            errors = validate_semantic_plan(
                plan, as_of=as_of, horizon=horizon, known_claim_ids=known
            )
        if errors:
            raise WorldIntegrityError(
                "the revised semantic plan is still invalid: " + "; ".join(errors[:6]),
                details={
                    "failure": "semantic_plan_invalid",
                    "recompilable": True,
                    "semantic_errors": errors,
                    "review_reasons": reasons,
                },
            )

    def lower_guarded(p: SemanticPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        # A lowering defect must surface as a named, recompilable refusal — never a
        # bare KeyError/StopIteration escaping as an uncaught traceback past every
        # diagnosis the run could have written.
        try:
            return lower_plan(p, structure_id=structure_id)
        except (LoweringGap, WorldIntegrityError):
            raise
        except Exception as exc:  # noqa: BLE001 — see above
            raise WorldIntegrityError(
                f"the lowerer failed on a validated plan: {type(exc).__name__}: {exc}",
                details={
                    "failure": "semantic_lowering_error",
                    "recompilable": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            ) from exc

    try:
        compilation, mapping = lower_guarded(plan)
    except LoweringGap as gap:
        if not gap.must_refuse:
            # must_refuse=False marks a gap no plan rephrasing can close — an
            # unresolved reference is a defect of the validator, not of the plan's
            # wording — so a revision round would burn a model call reproducing the
            # same failure. The gap itself is the refusal.
            raise
        # One revision naming the gap, then the gap is real and refuses.
        plan, raw = build(
            raw,
            [
                "the plan uses a construct the universal change mapping cannot "
                f"represent — express the same meaning differently: {gap}"
            ],
            3,
        )
        errors = validate_semantic_plan(plan, as_of=as_of, horizon=horizon, known_claim_ids=known)
        if errors:
            raise WorldIntegrityError(
                "the revised semantic plan is invalid: " + "; ".join(errors[:6]),
                details={
                    "failure": "semantic_plan_invalid",
                    "recompilable": True,
                    "semantic_errors": errors,
                },
            ) from gap
        compilation, mapping = lower_guarded(plan)

    # One normalization boundary for both compiler modes: it synthesizes the `reality`
    # block every consumer of a compilation reads (without it, `assemble_bundle` refuses
    # and every repair round dies), and it strips any claim id not actually in this
    # run's evidence — a fabricated citation must not survive to be read as grounding.
    compilation = _normalize_compilation(compilation, view, as_of, horizon)
    compilation["_semantic"] = {
        "plan": raw,
        "mapping": mapping,
        "review": {
            "verdict": verdict or "NOT_REVIEWED",
            "reasons": reasons,
            "revision_applied": bool(verdict == "REVISE" and corrections),
        },
        "compiler_mode": "semantic",
    }
    return compilation, (responses[-1] if responses else None)
