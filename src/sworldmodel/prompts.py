"""Prompt rendering.

Prompts are rendered from plain context dictionaries — the *same* dictionaries the
deterministic gateway reasons over. This guarantees the invariant that the view
recorded in the trace is the view given to the model. A live LLM gateway sends these
strings verbatim; the deterministic gateway reads the structured context but still
hashes the string, so both share one auditable prompt record.

None of these prompts names a vote, committee, proposal, or any question family. The
actor is offered whatever compiled actions its world currently makes feasible.
"""

from __future__ import annotations

import json
from typing import Any


def _block(title: str, body: Any) -> str:
    rendered = (
        body if isinstance(body, str) else json.dumps(body, indent=2, sort_keys=True, default=str)
    )
    return f"## {title}\n{rendered}"


def render_decision_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"You are {context.get('name', context['actor_id'])}, role "
            f"{context.get('role', '')}. You act inside a persistent, simulated world.",
            f"The world exists to resolve this question: {context.get('question', '')}",
            "Choose ONE action for this turn. You may pick one of the FEASIBLE ACTIONS",
            "compiled for your world, or — if none fits your situation — propose a NOVEL",
            "action describing what you attempt, its target, parameters and intended",
            "effect. A novel action does NOT automatically happen: the world decides.",
            "You state an intention only; you may never assert a consequence.",
            _block("YOUR AUTHORITY (capabilities you hold)", context.get("authority")),
            _block("YOUR ATTRIBUTES", context.get("attributes")),
            _block("CURRENT STAGE", context.get("stage")),
            _block("WHAT YOU HAVE OBSERVED (only what reached you)", context.get("observations")),
            _block("FIELD LEVELS YOU CAN READ", context.get("observed_fields")),
            _block("RETRIEVED MEMORIES", context.get("retrieved_memories")),
            _block("PUBLIC FACTS", context.get("public_facts")),
            _block("FEASIBLE COMPILED ACTIONS", context.get("feasible_actions")),
            f"Novel actions allowed: {context.get('allow_novel', True)}",
            'Reply with JSON: {"action_mode":"compiled_action|novel_action|wait",'
            ' "compiled_action_id":"", "params":{}, "target":"",'
            ' "novel_action":{"description":"","target":"","parameters":{},"intended_effect":""},'
            ' "reasoning":"", "referenced_memory_ids":[], "referenced_observation_ids":[]}.'
            " Qualitative reasoning only.",
        ]
    )


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


def render_novel_interpret_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "An actor proposed a NOVEL action the compiler did not anticipate. Decide",
            "whether it can be represented SAFELY using ONLY the universal world",
            "operations below, and if so, translate it into a concrete list of those",
            "operations. Do not invent new operations. If the action cannot be safely",
            "represented, set representable=false — never approximate it into something",
            "else.",
            _block(
                "UNIVERSAL WORLD OPERATIONS (the only allowed ops)", context.get("universal_ops")
            ),
            _block("ACTOR", {"id": context.get("actor_id"), "authority": context.get("authority")}),
            _block(
                "PROPOSED NOVEL ACTION",
                {
                    "description": context.get("description"),
                    "target": context.get("target"),
                    "parameters": context.get("parameters"),
                    "intended_effect": context.get("intended_effect"),
                },
            ),
            _block("WORLD FIELDS", context.get("world_fields")),
            _block("ENTITIES", context.get("entities")),
            _block("RESOURCES", context.get("resources")),
            'Reply with JSON: {"representable":true|false, "reason":"",'
            ' "required_authority":[capability tokens the actor must hold],'
            ' "effects":[{"op":"<universal op>", ...params...}]}.',
        ]
    )


def render_world_compile_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Compile the ACTUAL causal world required to resolve this question, from the",
            "verified evidence below. Do NOT assume it is a vote, committee, negotiation,",
            "election, market, or any template. Discover what the real process is: who",
            "the entities and actors are, their roles/authority/capabilities, the objects,",
            "documents, resources and channels, the real events that can occur and in what",
            "order, what each actor can do, and the exact world-state condition that makes",
            "the answer YES. Cite ONLY claim ids present in the evidence.",
            f"QUESTION: {context.get('question')}",
            f"as_of: {context.get('as_of')}   horizon: {context.get('horizon')}",
            _block("EVIDENCE (id | proposition = value [meta])", context.get("evidence")),
            _block("SCHEMA (fill this shape)", _WORLD_SCHEMA),
        ]
    )


_WORLD_SCHEMA = """Return a SINGLE JSON object:
{
 "subject_entity": "...", "resolution_units": "...",
 "target_outcome": "one-line description of what YES means",
 "expected_participants": <int number of decision-relevant actors, or null>,
 "world_spec": {
   "title": "...",
   "entities": [{"entity_id":"snake","name":"...","kind":"person|organization|object|document|channel",
                 "is_actor":true,"role":"...","authority":["capability_token",...],
                 "attributes":{"key":value},"evidence_claim_ids":["..."]}],
   "actors": [{"entity_id":"snake","reasoning":"why they lean as they do",
               "memory_seeds":[{"content":"first-person prior fact","kind":"episodic","importance":0.8,
                                "evidence_claim_ids":["..."]}],
               "policy":{"default_action_id":"<an action id or ''>","default_params":{},
                         "rules":[{"when_field":"<a field>","op":"above|below|equals|present",
                                   "value":<x>,"action_id":"<id>","params":{}}]}}],
   "fields":[{"field_id":"snake","value_type":"number|string|bool","initial":<v>,"description":"..."}],
   "resources":[{"resource_id":"snake","holder_entity_id":"<entity>","quantity":<n>}],
   "channels":[{"channel_id":"snake","participants":["..."]}],
   "documents":[{"document_id":"snake","fields":{"k":v}}],
   "actions":[{"action_id":"snake_verb","meaning":"...","eligible_actors":["*"|"role:X"|"<entity>"],
               "required_authority":["capability_token"],"parameters":[{"name":"","type":"option|string|number",
               "choices":[...]}],"preconditions":{"op":"...","args":[...]},"resource_costs":[["res",amt]],
               "stages":["<node stage>"],"valid_targets":["*"|"role:X"|"<entity>"],"visibility":"public|private",
               "effects":[{"op":"append_record|set_field|deliver_information|create_event|update_commitment|"
               "transfer_resource|consume_resource|create_or_update_document|release_data","...":"..."}],
               "evidence_claim_ids":["..."]}],
   "process":{"nodes":[{"node_id":"snake","stage":"<label>","description":"...",
               "condition":{"op":"...","args":[...]},"advance_seconds":0,
               "effects":[{"op":"...", ...}],"participants":["*"|"role:X"|"<entity>"],
               "action_ids":["*"|"<action id>"],"allow_novel":true,"rounds":1}]},
   "terminal":{"yes_when":{"op":"...","args":[...]},"unresolved_when":{"op":"...","args":[...]},
               "description":"plain-language resolution condition"}
 },
 "uncertainties":[{"variable":"<a field>","why_unknown":"...","reversal_capable":true,
   "constraining_evidence_ids":["..."],
   "outcomes":[{"value":"...","weight":0.7,"provenance":"symmetric_ignorance_assumption|explicit_model_distribution|"
   "calibrated_behavior_model|market_or_survey_distribution","field_effects":[["<field>",<level>]],"description":"..."}]}],
 "world_facts":[{"text":"...","evidence_claim_ids":["..."],"epistemic_type":"observation"}],
 "required_reality_facts":[{"key":"...","description":"...","evidence_claim_ids":["..."]}]
}
Terminal/precondition operators (universal, the only ones): const, field, stage, now, horizon, as_of,
count, sum, values, exists, event_count, resource, document_field, item, equals, not_equals,
greater_than, less_than, greater_or_equal, less_or_equal, contains, all, any, before, after, duration,
and, or, not. A count/sum over a record collection may take a trailing where-expr using item("value")."""
