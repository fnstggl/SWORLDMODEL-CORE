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
    lines = [
        f"You are {context.get('name', context['actor_id'])}, role {context.get('role', '')}.",
        "You act inside a persistent world. Emit ONE typed intention. You may only",
        "state intentions (send a message, make a statement, request information,",
        "introduce/revise/support/oppose a proposal, make a commitment, act, cast a",
        "vote, or wait). You may NOT assert any consequence (that someone was",
        "persuaded, that a proposal passed, that a vote succeeded).",
        _block("YOUR AUTHORITY", context.get("authority")),
        _block("STAGE", context.get("stage")),
        _block("YOUR CURRENT INCLINATION", context.get("current_inclination")),
        _block("YOUR REACTION RULES", context.get("reaction_rules")),
        _block("ACTIVE PROPOSAL", context.get("active_proposal")),
        _block("RETRIEVED MEMORIES", context.get("retrieved_memories")),
        _block("NEW OBSERVATIONS (only what you actually received)", context.get("observations")),
        _block("PUBLIC FACTS AVAILABLE TO YOU", context.get("public_facts")),
        _block("FEASIBLE ACTIONS", context.get("feasible_actions")),
        "Reply with JSON: {kind, vote_option, rationale, expected_effect,"
        " referenced_memory_ids, referenced_observation_ids}. If you vote, vote_option"
        " must be exactly one option. Qualitative reasoning only — never numbers.",
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
