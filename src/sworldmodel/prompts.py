"""Prompt rendering.

Prompts are built from plain context dictionaries and both **sent and recorded**, so
the view in the trace is byte-identical to the view the provider received.

None of these prompts names a vote, a committee, a proposal, a negotiation, an
election, a market, or any other kind of question. The compiler is asked what the world
*is*; the actor is offered whatever its world currently makes feasible. If a prompt here
ever starts describing a mechanism, the architecture has regressed.
"""

from __future__ import annotations

import json
from typing import Any


def _block(title: str, body: Any) -> str:
    rendered = (
        body if isinstance(body, str) else json.dumps(body, indent=2, sort_keys=True, default=str)
    )
    return f"## {title}\n{rendered}"


# ---------------------------------------------------------------------------
# Actor decision
# ---------------------------------------------------------------------------


def render_decision_prompt(context: dict[str, Any]) -> str:
    """Render this actor's decision prompt.

    Two things make this prompt different from "an LLM playing a character":

    * The actor is told **why it is being asked right now**. It is not taking a turn;
      something specific reached it.
    * The actor is told **what it was already doing**. Its first decision is what to do
      with the plan it already has, not what to think about the world from scratch.

    The prompt separates what this specific person knows from what any participant
    knows, and marks the epistemic status of every grounded item, so inference is never
    presented to the actor as fact.
    """

    identity = context.get("canonical_identity") or context.get("name") or context["actor_id"]
    grounding = str(context.get("actor_grounding") or "").strip()
    if not grounding:
        grounding = (
            f"You are {identity}.\nRole: {context.get('role', '')}\n"
            f"Authority: {', '.join(context.get('authority') or []) or '(none recorded)'}"
        )
    return "\n\n".join(
        [
            f"You are {identity}. Act as this specific person, not as a generic role.",
            f"The world exists to resolve this question: {context.get('question', '')}",
            f"It is now {context.get('branch_time', '')} in this world.",
            "=" * 70 + "\n## WHY YOU ARE BEING ASKED NOW\n" + "=" * 70,
            _block("TRIGGER", context.get("why_you_are_deciding_now")),
            "You have not been called to take a scheduled turn. Something specific",
            "reached you. Decide in that light.",
            "=" * 70 + "\n## WHAT YOU WERE ALREADY DOING\n" + "=" * 70,
            _block("YOUR ACTIVE PLAN", context.get("active_plan")),
            _block("ACTION YOU ARE CURRENTLY CARRYING OUT", context.get("current_action")),
            _block("YOUR COMMITMENTS", context.get("your_commitments")),
            _block(
                "INFORMATION YOU ARE STILL WAITING FOR", context.get("pending_information_needs")
            ),
            _block(
                "CONDITIONS YOU SAID YOU WOULD REVISIT ON", context.get("your_revisit_conditions")
            ),
            "FIRST decide what happens to your plan. Continuing is a real and usually",
            "correct answer: most events do not overturn what a person is already doing.",
            "Change your plan only if what reached you actually warrants it.",
            "=" * 70 + "\n## ACTOR-SPECIFIC GROUNDING (yours alone)\n" + "=" * 70,
            grounding,
            "Items marked VERIFIED_OBSERVATION are established fact. Items marked",
            "SUPPORTED_INFERENCE are reasoned expectations, not facts — you may act against",
            "them if your own record and the current situation warrant. Items marked UNKNOWN",
            "are genuinely not known: do not invent them. Do not invent personal",
            "circumstances, meetings, relationships or events that are not recorded here.",
            "=" * 70 + "\n## SHARED WORLD CONTEXT\n" + "=" * 70,
            _block("YOUR AUTHORITY (capabilities you hold)", context.get("authority")),
            _block("YOUR ATTRIBUTES", context.get("attributes")),
            _block("YOUR GOALS", context.get("goals")),
            _block("YOUR CURRENT BELIEFS", context.get("beliefs")),
            _block("CURRENT STAGE", context.get("stage")),
            _block(
                "WHAT YOU HAVE NOTICED (only what actually reached you)",
                context.get("observations"),
            ),
            _block("FIELD LEVELS YOU CAN READ", context.get("observed_fields")),
            _block("RETRIEVED MEMORIES", context.get("retrieved_memories")),
            _block("PUBLIC FACTS", context.get("public_facts")),
            _block("FEASIBLE ACTIONS AVAILABLE TO YOU RIGHT NOW", context.get("feasible_actions")),
            f"Novel actions allowed: {context.get('allow_novel', True)}",
            "Choose ONE action. You may pick one of the feasible compiled actions, or — if",
            "none fits your situation — propose a NOVEL action describing what you attempt,",
            "its target, parameters and intended effect. A novel action does NOT",
            "automatically happen: the world decides. You state an intention only; you may",
            "never assert a consequence. If an action needs a parameter (an option, an",
            "amount, a recipient) you must state it explicitly — nothing will be chosen on",
            "your behalf, and an action missing a parameter is refused.",
            "If the right thing to do is nothing yet, choose wait and say what you are",
            "waiting for.",
            _DECISION_SCHEMA,
        ]
    )


_DECISION_SCHEMA = """Reply with a SINGLE JSON object:
{
 "plan_disposition": "continue|revise|interrupt|replace|complete|abandon|none",
   // continue  - keep doing what you were doing, unchanged
   // revise    - same plan, updated steps
   // interrupt - pause this plan to handle something urgent (it can resume later)
   // replace   - this plan no longer makes sense; state a new one
   // complete  - the plan is finished
   // abandon   - the plan is off; nothing replaces it
   // none      - you had no plan and are not forming one
 "plan_update": {"goal":"", "steps":[{"description":"","intended_action_id":"","at":"<ISO time or null>"}],
                 "basis":"which verified schedule, role obligation, existing commitment or step you
                          already began makes this plan admissible — required if you set one",
                 "interrupt_when":["what would make you drop this"], "evidence_claim_ids":[]},
 "action_mode": "compiled_action|novel_action|wait",
 "compiled_action_id": "", "params": {}, "target": "",
 "novel_action": {"description":"","target":"","parameters":{},"intended_effect":""},
 "reasoning": "",
 "information_needs": [{"question":"","asked_of":"<actor id or ''>","deadline":"<ISO or null>"}],
 "resolved_information_needs": ["<question text you now consider answered>"],
 "new_commitments": [{"text":"","due":"<ISO or null>","to_actor":""}],
 "discharged_commitments": ["<commitment text>"],
 "revisit_when": [{"description":"", "at":"<ISO or null>", "on_field_change":"<field or ''>",
                   "on_information_from":"<actor id or ''>", "on_record_in":"<collection or ''>"}],
 "unresolved_questions": [],
 "reflection_needed": false,
 "referenced_memory_ids": [], "referenced_observation_ids": []
}
Qualitative reasoning only. Do not output probabilities or scores."""


def render_reflect_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"You are {context.get('actor_id')}. Reflect on what you have just learned and",
            "update your durable understanding. Reflection changes only your own internal",
            "state; it never changes external reality and never makes anything happen.",
            _block("CURRENT BELIEFS", context.get("beliefs")),
            _block("NEW OBSERVATIONS", context.get("observations")),
            'Reply with JSON: {"beliefs_update": [], "new_memories": [{"content":"",'
            ' "kind":"thought|semantic", "importance":0.0, "evidence_claim_ids":[]}]}.',
        ]
    )


# ---------------------------------------------------------------------------
# Novel action interpretation
# ---------------------------------------------------------------------------


def render_novel_interpret_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "An actor proposed a NOVEL action the compiler did not anticipate. Decide",
            "whether it can be represented SAFELY using ONLY the universal world",
            "operations below, and if so, translate it into a concrete list of those",
            "operations. Do not invent new operations. If the action cannot be safely",
            "represented, set representable=false — never approximate it into something",
            "else, and never map it onto whichever existing action is closest.",
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
            "State every capability this act would genuinely require. Under-stating the",
            "required authority would let an actor do something it has no standing to do.",
            'Reply with JSON: {"representable":true|false, "reason":"",'
            ' "required_authority":[capability tokens the actor must hold],'
            ' "effects":[{"op":"<universal op>", ...params...}]}.',
        ]
    )


# ---------------------------------------------------------------------------
# World compilation
# ---------------------------------------------------------------------------


def render_world_compile_prompt(context: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "Compile the ACTUAL causal world required to resolve this question, from the",
            "verified evidence below. Do NOT assume it is a vote, committee, negotiation,",
            "election, market, campaign, or any other template — discover what the real",
            "process is. Who and what exists; who can do what and on whose authority; what",
            "the objects, documents, resources and channels are; which parts of the world",
            "move on their own without anyone deciding; what really happens, in what order,",
            "on what dates; and the exact world-state condition that makes the answer YES.",
            "Cite ONLY claim ids present in the evidence.",
            f"QUESTION: {context.get('question')}",
            f"as_of: {context.get('as_of')}   horizon: {context.get('horizon')}",
            _block("EVIDENCE (id | proposition = value [meta])", context.get("evidence")),
            _block(
                "MATERIAL ITEMS FOUND IN VERIFIED EVIDENCE — represent EVERY one of these "
                "in the compiled world (as an entity, action, field, document, resource, "
                "process node, external process or terminal term), or the run will be refused",
                context.get("checklist", ""),
            ),
            _COMPILE_RULES,
            _block("SCHEMA (fill this shape)", _WORLD_SCHEMA),
        ]
    )


_COMPILE_RULES = """RULES THE COMPILED WORLD MUST SATISFY:

REPRESENTATION SCALE. For every entity, choose the level that is causally faithful and say which:
individual, organization (acting as one unit), subunit, population_stratum, network, or
external_process. Do not turn a body of independent decision-makers into one actor, and do not
invent a million agents where an aggregate process is the honest representation. If an entity stands
for many real units, give represents_count.

ACTORS ARE SPECIFIC PEOPLE OR BODIES. The world MUST contain at least one actor whose decisions
actually produce the outcome, and every entity that decides must appear in both `entities`
(is_actor true) and `actors`. Each actor MUST carry at least one memory_seed stating, in the first
person, something that entity itself verifiably did or said, citing the claim ids that support it.
An actor with only a name and a role is rejected and the run is refused. If the evidence records
nothing about an actor, say exactly that in the seed rather than inventing a fact, and never give
one actor another actor's history.

PLANS MUST BE GROUNDED, NOT INVENTED. An actor's initial_plan is optional and should be sparse. Give
one only where evidence, a published schedule, a role obligation or an existing commitment supports
it, and state that support in `basis`. Do not fabricate daily routines, personal activities,
absences, distractions or unrelated obligations to make anyone look alive. When you do not know what
someone is doing, say nothing: the runtime preserves their existing state rather than inventing
activity.

ACTIONS ARE DISCOVERED, NOT ASSUMED. Compile the actions this specific world actually affords, with
the authority each requires and the preconditions that gate it. An action's meaning is its effects.
Give duration_seconds where an act genuinely takes time, and delivery/notice delays where a
consequence genuinely takes time to reach or be seen by others. Use completion_conditions for what
must still be true when the act completes for it to succeed — this is how an attempt can fail.

THE PROCESS IS A CALENDAR, NOT A SCRIPT. Process nodes are moments placed on real dates (`at`), or
entered when another node completes (`after_node` + `delay_seconds`, `next_nodes`). `participants`
are those who GAIN AN OPPORTUNITY to act at that node — not a list of people who each take one turn.
Nobody is invoked on a schedule. Do not write a node whose only purpose is to give everyone a turn.

WHAT WAKES PEOPLE. Use `wake_rules` to say what, in THIS world, is important enough to bring a
specific actor back: a record appearing, a field changing, an event type occurring, a particular
person communicating. Without a rule, an actor that notices something merely remembers it and
carries on. This is deliberate; state the rules that matter.

PARTS OF THE WORLD THAT ARE NOT PEOPLE. Put scheduled data releases, market or administrative
clocks, publication cycles, delivery systems and legal deadlines in `external_processes`. Do not
create an actor whose only job is to make the weather happen.

UNCERTAINTY. Declare only genuine unknowns that can change the answer. Give each outcome's weight a
provenance, and say `release_at` if the evidence establishes when the unknown value becomes public.
If two unknowns are not independent, do NOT list them separately — list one uncertainty whose
outcomes are joint states, or the run is refused. When no defensible point weight exists, use
symmetric_ignorance_assumption rather than inventing precision.

THE TERMINAL IS AN EXPRESSION OVER WORLD STATE. Write the condition that makes the answer YES in
terms of what will actually be true in the world, using only the universal operators. Use
`unresolved_when` for states where the process genuinely did not determine the answer — an
undetermined world must report as unresolved, never be rounded to NO."""


_WORLD_SCHEMA = """Return a SINGLE JSON object:
{
 "subject_entity": "...", "resolution_units": "...",
 "target_outcome": "one-line description of what YES means",
 "expected_participants": <int number of decision-relevant actors the EVIDENCE names, or null>,
 "world_spec": {
   "title": "...",
   "structure_rationale": "why this is the causal structure the evidence supports",
   "entities": [{"entity_id":"snake","name":"...","kind":"person|organization|object|document|channel",
                 "is_actor":true,"role":"...","authority":["capability_token",...],
                 "representation_scale":"individual|organization|subunit|population_stratum|network|external_process",
                 "represents_count":<int or null>,
                 "attributes":{"key":value},"evidence_claim_ids":["..."]}],
   "actors": [{"entity_id":"snake","reasoning":"why they lean as they do",
               "goals":["..."],
               "memory_seeds":[{"content":"first-person prior fact","kind":"episodic","importance":0.8,
                                "evidence_claim_ids":["..."]}],
               "initial_plan":{"goal":"","basis":"the verified schedule/obligation/commitment behind it",
                               "steps":[{"description":"","intended_action_id":"","at":"<ISO or null>"}],
                               "evidence_claim_ids":["..."]},
               "commitments":[{"text":"","due":"<ISO or null>","to_actor":""}]}],
   "fields":[{"field_id":"snake","value_type":"number|string|bool","initial":<v>,"description":"..."}],
   "resources":[{"resource_id":"snake","holder_entity_id":"<entity>","quantity":<n>}],
   "channels":[{"channel_id":"snake","participants":["..."]}],
   "documents":[{"document_id":"snake","fields":{"k":v}}],
   "actions":[{"action_id":"snake_verb","meaning":"...","eligible_actors":["*"|"role:X"|"<entity>"],
               "required_authority":["capability_token"],"parameters":[{"name":"","type":"option|string|number",
               "required":true,"choices":[...]}],"preconditions":{"op":"...","args":[...]},
               "resource_costs":[["res",amt]],"stages":["<node stage>"],
               "valid_targets":["*"|"role:X"|"<entity>"],"visibility":"public|private",
               "duration_seconds":0,"delivery_delay_seconds":0,"notice_delay_seconds":0,
               "completion_conditions":{"op":"...","args":[...]},
               "effects":[{"op":"append_record|set_field|adjust_field|deliver_information|create_event|"
               "schedule_event|update_commitment|transfer_resource|consume_resource|"
               "create_or_update_document|release_data|advance_time","...":"..."}],
               "evidence_claim_ids":["..."]}],
   "process":{"nodes":[{"node_id":"snake","stage":"<label>","description":"...",
               "at":"<ISO datetime>","after_node":"<node id or ''>","delay_seconds":0,
               "entry_condition":{"op":"...","args":[...]},
               "effects":[{"op":"...", ...}],"participants":["*"|"role:X"|"<entity>"],
               "action_ids":["*"|"<action id>"],"allow_novel":true,
               "deadline":"<ISO or null>","next_nodes":["<node id>"],"evidence_claim_ids":["..."]}]},
   "external_processes":[{"process_id":"snake","description":"a non-agent process",
               "occurrences":[{"at":"<ISO>","description":"...","effects":[{"op":"...", ...}]}],
               "evidence_claim_ids":["..."]}],
   "wake_rules":[{"rule_id":"snake","wakes":["*"|"role:X"|"<entity>"],"reason":"why this matters to them",
               "on_record_in":"<collection or ''>","on_field_change":"<field or ''>",
               "on_event_type":"<event_type or ''>","on_information_from":"<entity or ''>",
               "evidence_claim_ids":["..."]}],
   "terminal":{"yes_when":{"op":"...","args":[...]},"unresolved_when":{"op":"...","args":[...]},
               "description":"plain-language resolution condition"}
 },
 "uncertainties":[{"variable":"<a field>","why_unknown":"...","reversal_capable":true,
   "constraining_evidence_ids":["..."],"depends_on":["<other variable>"],
   "release_at":"<ISO or null>",
   "outcomes":[{"value":"...","weight":0.7,"provenance":"symmetric_ignorance_assumption|explicit_model_distribution|"
   "calibrated_behavior_model|market_or_survey_distribution|direct_empirical_distribution|sensitivity_only_branch",
   "field_effects":[["<field>",<level>]],"description":"..."}]}],
 "world_facts":[{"text":"...","evidence_claim_ids":["..."],"epistemic_type":"observation"}],
 "required_reality_facts":[{"key":"...","description":"...","evidence_claim_ids":["..."]}]
}
Expression operators (universal, the only ones): const, field, stage, now, horizon, as_of,
count, sum, values, exists, event_count, resource, document_field, item, equals, not_equals,
greater_than, less_than, greater_or_equal, less_or_equal, contains, all, any, before, after, duration,
and, or, not. A count/sum over a record collection may take a trailing where-expr using item("value")."""
