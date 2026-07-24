"""LLM-driven universal world + uncertainty compilation from live evidence.

Given a verified evidence store, the LLM compiles the structured reality (decision
body, roster, rule, terminal predicate, actors, required facts) and the causal /
uncertainty frame (options, signals, conditional reaction rules, guidance, genuine
outcome-sensitive uncertainties with provenance). Deterministic validation then
normalizes and citation-checks the output before the reality-integrity gate runs.

Nothing here is scenario-specific: it produces a general :class:`ResearchBundle` that
the existing compiler + runtime consume. The manually authored corpus frame is not
used on this path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .coverage import evidence_checklist
from .errors import GatewayError
from .evidence import EvidenceStore, EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .models import WeightProvenance
from .reality import expected_majority_threshold
from .research import ResearchBundle, assemble_bundle
from .research_planner import ResearchPlan

_VALID_PROVENANCE = {p.value for p in WeightProvenance}


def _candidate_persons(view: EvidenceView) -> list[str]:
    """Distinct multi-word (person-like) entities named across the available claims.

    An evidence-derived checklist so the roster call enumerates every named actor,
    including those attested only once. Names come only from verified claims.
    """

    seen: dict[str, None] = {}
    for c in view.available():
        for ent in c.entities:
            e = ent.strip()
            if e and len(e.split()) >= 2 and e not in seen:
                seen[e] = None
    return list(seen)


def _render_evidence(view: EvidenceView, *, limit: int = 140) -> str:
    claims = sorted(view.available(), key=lambda c: (-int(c.authority_level), c.id))[:limit]
    return "\n".join(
        f"{c.id} | {c.proposition} = {c.normalized_value} "
        f"[auth {int(c.authority_level)}, {c.source_type.value}, {c.published_at.date()}]"
        for c in claims
    )


def compile_reality(
    gateway: ModelGateway, question: str, as_of: datetime, horizon: datetime, view: EvidenceView
) -> dict[str, Any]:
    evidence = _render_evidence(view)
    checklist = evidence_checklist(view, as_of=as_of, horizon=horizon)
    prompt = f"""Compile the VERIFIED reality for this forecasting question from the
evidence below. Cite ONLY claim ids that appear in the evidence list. Do not invent
members, offices, rules, or citations.

QUESTION: {question}
as_of: {as_of.isoformat()}   horizon: {horizon.isoformat()}

MATERIAL EVIDENCE CANDIDATES (a deterministic inventory of what verified reality
contains — every material person, organization, rule, and event below MUST be
represented in your output or it will be rejected; do not drop any):
{checklist}

EVIDENCE (id | proposition = value [meta]):
{evidence}

Return JSON with keys:
- decision_body, subject_entity, resolution_units (strings)
- institution_id (slug), institution_name
- decision_rule: {{"kind":"majority"|"unanimous"|"supermajority"|"plurality","total_seats":<int>,"evidence_claim_ids":[...]}}
- expected_voting_seats: the TRUE size of the deciding body per the evidence (an int);
  report the real size even if you cannot name every seat.
- target_option: the option that constitutes YES.
- terminal: {{"mechanism":"committee_vote","yes_condition":"unanimous_for_option"|"at_least_k_for_option"|"majority_for_option","target_option":"...","k":<int or null>,"evidence_claim_ids":[...]}}
- members: [ {{"actor_id":"snake_case","name":"...","role":"...","is_voting_seat":true,"vote_power":1,"prior_action":"<their most recent relevant choice or null>","authority":["vote"...],"evidence_claim_ids":[...],"memory_seeds":[{{"content":"evidence-grounded prior fact in first person","kind":"episodic","importance":0.8,"evidence_claim_ids":[...]}}]}} ]
- world_facts: [ {{"text":"...","evidence_claim_ids":[...],"epistemic_type":"observation"}} ]
- required_reality_facts: [ {{"key":"roster"|"decision_rule"|"prior_votes"|"current_state"|"guidance"|"terminal_date","description":"...","evidence_claim_ids":[...]}} ]

Give the chair/leader authority ["vote","introduce_proposal","chair"]."""
    resp = gateway.generate(
        GatewayRequest(
            task_kind="compile_reality",
            prompt=prompt,
            context={"question": question},
            seed=int(prompt_hash("reality" + question)[:8], 16),
            expected_keys=("members", "decision_rule", "terminal", "expected_voting_seats"),
        )
    )
    return _normalize_reality(resp.data, view, as_of, horizon)


def compile_roster(
    gateway: ModelGateway,
    question: str,
    view: EvidenceView,
    reality: dict[str, Any],
    horizon: datetime,
) -> list[dict[str, Any]]:
    """A focused call that enumerates every decision-maker named in the evidence.

    Kept separate from the structural compile because a single mega-prompt tends to
    under-enumerate the roster; a dedicated prompt reliably lists all named actors.
    """

    evidence = _render_evidence(view)
    candidates = _candidate_persons(view)
    checklist = evidence_checklist(
        view,
        decision_body=str(reality.get("decision_body") or ""),
        subject_entity=str(reality.get("subject_entity") or ""),
        as_of=view.as_of,
        horizon=horizon,
    )
    prompt = f"""From the evidence below, ENUMERATE EVERY individual decision-maker
(board/committee member, voting seat, or the focal actor) NAMED in the evidence and
relevant to the question. List ALL of them — do not summarize or omit any. A person
named in ANY claim (including a vote/attribution claim like "X voted to hold") is a
member and MUST be listed even if mentioned only once. Cite only claim ids that appear
in the evidence.

QUESTION: {question}
DECISION BODY: {reality.get("decision_body")}
The body has approximately {reality.get("expected_voting_seats")} decision-makers.
INDIVIDUALS NAMED IN THE EVIDENCE (include every one who is a decision-maker): {candidates}
MATERIAL EVIDENCE CANDIDATES (deterministic inventory — represent every person here):
{checklist}

EVIDENCE (id | proposition = value):
{evidence}

Return JSON {{"members": [ {{"actor_id":"snake_case","name":"Full Name","role":"...",
"is_voting_seat":true,"vote_power":1,"prior_action":"<most recent relevant choice or
null>","authority":["vote"],"evidence_claim_ids":[...],"memory_seeds":[{{"content":
"evidence-grounded prior fact in the first person","kind":"episodic","importance":0.8,
"evidence_claim_ids":[...]}}]}} ]}}. Give the chair/leader/governor authority
["vote","introduce_proposal","chair"]. If the evidence names no individuals, return
{{"members": []}}."""
    resp = gateway.generate(
        GatewayRequest(
            task_kind="compile_roster",
            prompt=prompt,
            context={"question": question},
            seed=int(prompt_hash("roster" + question)[:8], 16),
            expected_keys=("members",),
        )
    )
    members = resp.data.get("members") or []
    available = _available_ids(view)
    for m in members:
        m["evidence_claim_ids"] = _filter_ids(m.get("evidence_claim_ids"), available)
        m.setdefault("is_voting_seat", True)
        m.setdefault("vote_power", 1)
        m.setdefault("authority", ["vote"])
        for s in m.get("memory_seeds", []):
            s["evidence_claim_ids"] = _filter_ids(s.get("evidence_claim_ids"), available)
    return members


def compile_frame(
    gateway: ModelGateway, question: str, view: EvidenceView, reality: dict[str, Any]
) -> dict[str, Any]:
    evidence = _render_evidence(view)
    options = reality.get("_options_hint") or ["cut", "hold", "hike"]
    prompt = f"""Compile the causal/uncertainty frame for this question. Use only
conditional structure — never encode the final answer. Weights are evidence-conditioned
estimates or clearly labeled ignorance; do NOT fabricate precision from one example.

QUESTION: {question}
DECISION BODY: {reality.get("decision_body")}
TARGET OPTION (YES): {reality.get("target_option")}
EVIDENCE (id | proposition = value):
{evidence}

Return JSON with keys:
- options: the mutually exclusive choices an INDIVIDUAL decision-maker picks between at
  the decision (e.g. the concrete policy actions each member can vote for). These are
  per-actor ballot choices, NOT the yes/no resolution of the question and NOT aggregate
  outcomes — never encode "unanimous_X" or "majority_X" as an option; unanimity/majority
  is decided by the terminal from the individual votes.
- signals: [ {{"name":"snake_case","baseline":0.0,"description":"...","evidence_claim_ids":[...]}} ]
- reaction_rules: [ {{"trigger_signal":"<a signal name>","direction":"above"|"below","threshold":<float>,"moves_to_option":"<an option>","rationale":"...","evidence_claim_ids":[...]}} ]
- guidance_option: the newly-established common position (an option) or null.
- guidance_text: the guidance in words. guidance_evidence_ids: [...]
- acceptance_tolerance: a float 0..1.
- uncertainty: [ {{"signal":"<a signal>","why_unknown":"...","reversal_capable":true,"constraining_evidence_ids":[...],"outcomes":[ {{"value":"...","weight":<float>,"provenance":"symmetric_ignorance_assumption"|"explicit_model_distribution"|"calibrated_behavior_model"|"market_or_survey_distribution","source_detail":"...","signal_effects":[["<signal>",<float>]],"description":"..."}} ]}} ]

Only include an uncertainty if changing it could flip an actor's option. The base case
(no threshold-crossing surprise) should carry most weight over a short horizon."""
    try:
        resp = gateway.generate(
            GatewayRequest(
                task_kind="compile_uncertainty",
                prompt=prompt,
                context={"question": question, "options": options},
                seed=int(prompt_hash("frame" + question)[:8], 16),
                # `uncertainty` may legitimately be empty (no threshold-crossing driver),
                # so only `options` is required; the rest is normalized defensively.
                expected_keys=("options",),
            )
        )
        data = resp.data
    except GatewayError:
        # A frame-stage provider failure must not abort the whole product. Degrade to a
        # base-case frame (the options and guidance we already verified, no modeled
        # surprise) so the simulation still runs a real, if less nuanced, trajectory.
        # The absence of uncertainty is visible in the trace, never hidden.
        data = _base_case_frame(reality, options)
    return _normalize_frame(data, view)


def _base_case_frame(reality: dict[str, Any], options: list[str]) -> dict[str, Any]:
    """A minimal valid frame: the verified options, no modeled surprise, and no guidance.

    Guidance is left null on purpose — with no compiled common position, each actor
    falls back to its own verified prior action rather than being nudged toward the
    question's YES option, so the degraded base case stays neutral.
    """

    return {
        "options": list(options),
        "signals": [],
        "reaction_rules": [],
        "guidance_option": None,
        "guidance_text": "",
        "guidance_evidence_ids": [],
        "acceptance_tolerance": 0.5,
        "uncertainty": [],
    }


def build_live_bundle(
    gateway: ModelGateway,
    question: str,
    as_of: datetime,
    horizon: datetime,
    store: EvidenceStore,
    plan: ResearchPlan,
) -> ResearchBundle:
    view = store.view(as_of)
    reality = compile_reality(gateway, question, as_of, horizon, view)
    # A dedicated roster call enumerates the named decision-makers reliably; use it
    # when it names at least as many actors as the structural compile did. If the
    # dedicated call fails, fall back to the roster already compiled with reality.
    try:
        roster = compile_roster(gateway, question, view, reality, horizon)
    except GatewayError:
        roster = []
    if len(roster) >= len(reality.get("members") or []):
        reality["members"] = roster
    frame = compile_frame(gateway, question, view, reality)
    reality["_options_hint"] = frame.get("options")
    data = {
        "reality": {**reality, "as_of": as_of.isoformat(), "horizon": horizon.isoformat()},
        "frame": frame,
        "world_facts": reality.pop("world_facts", []),
        "required_reality_facts": reality.pop("required_reality_facts", []),
        "outcome": None,
    }
    return assemble_bundle(store, data)


# ---------------------------------------------------------------------------
# Deterministic validation / normalization
# ---------------------------------------------------------------------------


def _available_ids(view: EvidenceView) -> set[str]:
    return {c.id for c in view.available()}


def _filter_ids(ids: object, available: set[str]) -> list[str]:
    if not isinstance(ids, list):
        return []
    return [str(i) for i in ids if str(i) in available]


def _normalize_reality(
    data: dict[str, Any], view: EvidenceView, as_of: datetime, horizon: datetime
) -> dict[str, Any]:
    available = _available_ids(view)
    expected = int(data.get("expected_voting_seats") or 0)
    members = data.get("members") or []
    for m in members:
        m["evidence_claim_ids"] = _filter_ids(m.get("evidence_claim_ids"), available)
        m.setdefault("is_voting_seat", True)
        m.setdefault("vote_power", 1)
        m.setdefault("authority", ["vote"])
        for s in m.get("memory_seeds", []):
            s["evidence_claim_ids"] = _filter_ids(s.get("evidence_claim_ids"), available)
            s.setdefault("valid_time", as_of.isoformat())
    if expected <= 0:
        expected = len(members)

    rule = data.get("decision_rule") or {}
    kind = rule.get("kind") or "majority"
    total = expected
    if kind == "unanimous":
        threshold = total
    elif kind == "majority":
        threshold = expected_majority_threshold(total)
    else:
        threshold = int(rule.get("threshold") or expected_majority_threshold(total))
    data["decision_rule"] = {
        "kind": kind,
        "total_seats": total,
        "threshold": threshold,
        "evidence_claim_ids": _filter_ids(rule.get("evidence_claim_ids"), available),
    }
    data["expected_voting_seats"] = expected

    terminal = data.get("terminal") or {}
    terminal.setdefault("mechanism", "committee_vote")
    terminal.setdefault("yes_condition", "unanimous_for_option")
    terminal["target_option"] = terminal.get("target_option") or data.get("target_option")
    terminal["evidence_claim_ids"] = _filter_ids(terminal.get("evidence_claim_ids"), available)
    data["terminal"] = terminal
    data["target_option"] = data.get("target_option") or terminal.get("target_option")

    for wf in data.get("world_facts", []):
        wf["evidence_claim_ids"] = _filter_ids(wf.get("evidence_claim_ids"), available)
        wf.setdefault("available_at", as_of.isoformat())
        wf.setdefault("epistemic_type", "observation")
    # Keep only required-reality facts the compiler could actually ground in available
    # evidence. A fact the model *names* but cannot cite (e.g. a terminal date that is
    # already given by the contract horizon) is not a verified load-bearing fact and
    # must not block the run — the seat-count, rule-consistency, and coverage checks are
    # independent and evidence-grounded, so real gaps are still caught.
    grounded_facts = []
    for rf in data.get("required_reality_facts", []):
        rf["evidence_claim_ids"] = _filter_ids(rf.get("evidence_claim_ids"), available)
        if rf["evidence_claim_ids"]:
            grounded_facts.append(rf)
    data["required_reality_facts"] = grounded_facts

    # Coerce every string field so a null/missing LLM value can never crash the
    # compiler (it either compiles or the reality gate refuses — never a TypeError).
    body = _s(data.get("decision_body")) or "the deciding body"
    data["decision_body"] = body
    data["institution_name"] = _s(data.get("institution_name")) or body
    data["institution_id"] = _s(data.get("institution_id")) or "deciding_body"
    data["subject_entity"] = _s(data.get("subject_entity")) or body
    data["resolution_units"] = _s(data.get("resolution_units")) or "the decision"
    data["target_option"] = _s(data.get("target_option")) or "yes"
    terminal["target_option"] = _s(terminal.get("target_option")) or data["target_option"]
    if not isinstance(data.get("authoritative_sources"), list):
        data["authoritative_sources"] = []
    if not isinstance(data.get("members"), list):
        data["members"] = []
    return data


def _s(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_frame(data: dict[str, Any], view: EvidenceView) -> dict[str, Any]:
    available = _available_ids(view)
    # Coerce any null/non-list collection to a list so a sparse LLM frame cannot crash.
    for key in ("options", "signals", "reaction_rules", "uncertainty"):
        if not isinstance(data.get(key), list):
            data[key] = []
    for s in data["signals"]:
        s["evidence_claim_ids"] = _filter_ids(s.get("evidence_claim_ids"), available)
        s.setdefault("baseline", 0.0)
    for r in data["reaction_rules"]:
        r["evidence_claim_ids"] = _filter_ids(r.get("evidence_claim_ids"), available)
    data["guidance_evidence_ids"] = _filter_ids(data.get("guidance_evidence_ids"), available)
    if not isinstance(data.get("acceptance_tolerance"), int | float):
        data["acceptance_tolerance"] = 0.5

    normalized_unc = []
    for u in data["uncertainty"]:
        outcomes = u.get("outcomes") or []
        total = sum(float(o.get("weight", 0)) for o in outcomes)
        if total <= 0 or len(outcomes) < 1:
            continue
        for o in outcomes:
            o["weight"] = float(o.get("weight", 0)) / total  # per-variable conservation
            prov = o.get("provenance")
            if prov not in _VALID_PROVENANCE:
                o["provenance"] = WeightProvenance.EXPLICIT_MODEL.value
            o.setdefault("signal_effects", [[u.get("signal"), 0.0]])
        u["constraining_evidence_ids"] = _filter_ids(u.get("constraining_evidence_ids"), available)
        normalized_unc.append(u)
    data["uncertainty"] = normalized_unc
    if not data["options"]:
        data["options"] = ["yes", "no"]
    return data
