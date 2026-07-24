"""Prompt rendering.

Prompts are rendered from plain context dictionaries — the *same* dictionaries the
deterministic gateway reasons over. This guarantees the invariant that the view
recorded in the trace is the view given to the model. A live LLM gateway sends
these strings verbatim; the deterministic gateway reads the structured context but
still hashes the string, so both share one auditable prompt record.
"""

from __future__ import annotations

import json
from typing import Any


def _block(title: str, body: Any) -> str:
    if isinstance(body, str):
        rendered = body
    else:
        rendered = json.dumps(body, indent=2, sort_keys=True, default=str)
    return f"## {title}\n{rendered}"


def render_compile_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "You are compiling ONE faithful causal world from verified evidence.",
            "Build actors ONLY for the verified members below. Do not invent members,",
            "offices, relationships, or historical cases. For each actor, express a",
            "current inclination that follows from evidence (prior vote + guidance) and",
            "conditional reaction rules that say what observable signal would move them.",
            _block("OPTIONS", context.get("options")),
            _block("VERIFIED MEMBERS", context.get("members")),
            _block("EVIDENCE-GROUNDED FRAME", context.get("frame")),
            _block("EVIDENCE VIEW", context.get("evidence_view", "")),
            'Reply with JSON: {"actors": [{actor_id, current_inclination,'
            " reaction_rules, dissent_threshold, reasoning, evidence_claim_ids}]}",
        ]
    )


def render_decision_prompt(context: dict[str, Any]) -> str:
    """Render this actor's decision prompt.

    The prompt is split into two clearly separated sections so the model can never
    confuse what everyone knows with what *this specific person* knows:

    * ACTOR-SPECIFIC GROUNDING — this actor's identity, own previous actions, own
      statements, own commitments and inferred inclination, each carrying an explicit
      epistemic mark (verified / inferred / hypothesis / unknown).
    * SHARED WORLD CONTEXT — the stage, options, proposal and public facts that every
      participant may validly hold.
    """

    identity = context.get("canonical_identity") or context.get("name") or context["actor_id"]
    grounding = str(context.get("actor_grounding") or "").strip()
    if not grounding:  # no compiled profile: fall back to the raw identity fields
        grounding = (
            f"You are {identity}.\nRole: {context.get('role', '')}\n"
            f"Authority: {', '.join(context.get('authority') or []) or '(none recorded)'}"
        )
    lines = [
        f"You are {identity}. Act as this specific person, not as a generic role.",
        "You act inside a persistent world. Emit ONE typed intention. You may only",
        "state intentions (send a message, make a statement, request information,",
        "introduce/revise/support/oppose a proposal, make a commitment, act, cast a",
        "vote, or wait). You may NOT assert any consequence (that someone was",
        "persuaded, that a proposal passed, that a vote succeeded).",
        "=" * 70 + "\n## ACTOR-SPECIFIC GROUNDING (yours alone)\n" + "=" * 70,
        grounding,
        "Items marked VERIFIED_OBSERVATION are established fact. Items marked",
        "SUPPORTED_INFERENCE are reasoned expectations, not facts — you may act against",
        "them if your own record and the current situation warrant. Items marked UNKNOWN",
        "are genuinely not known: do not invent them.",
        "=" * 70 + "\n## SHARED WORLD CONTEXT (available to participants)\n" + "=" * 70,
        _block("STAGE", context.get("stage")),
        _block("OPTIONS", context.get("options")),
        _block("ACTIVE PROPOSAL", context.get("active_proposal")),
        _block("RETRIEVED MEMORIES", context.get("retrieved_memories")),
        _block("NEW OBSERVATIONS (only what you actually received)", context.get("observations")),
        _block("PUBLIC FACTS AVAILABLE TO YOU", context.get("public_facts")),
        _block("FEASIBLE ACTIONS", context.get("feasible_actions")),
        "Reply with JSON: {kind, vote_option, rationale, expected_effect,"
        " referenced_memory_ids, referenced_observation_ids}. `kind` MUST be exactly"
        " one of: send_message, make_statement, request_information,"
        " introduce_proposal, revise_proposal, support_proposal, oppose_proposal,"
        " make_commitment, operational_action, cast_vote, wait. To vote use"
        ' kind="cast_vote" and set vote_option to EXACTLY one of the OPTIONS strings'
        " above, copied verbatim — do not paraphrase it. Qualitative reasoning only —"
        " never numbers.",
    ]
    return "\n\n".join(lines)


def render_reflect_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"You are {context.get('actor_id')}. Reflect on new observations and update",
            "your durable memories and plan. Reflection changes only your own internal",
            "state; it never changes external reality.",
            _block("CURRENT BELIEFS", context.get("beliefs")),
            _block("NEW OBSERVATIONS", context.get("observations")),
            "Reply with JSON: {beliefs_update, new_memories, plan_note}.",
        ]
    )
