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
            "Every item above carries its epistemic class, and they mean different things:",
            "  VERIFIED   — established fact, directly supported by a source. Rely on it.",
            "  INFERRED / SUPPORTED_INFERENCE — a defensible conclusion from verified facts,",
            "               not itself a fact. You may act against it when your own record and",
            "               the situation warrant, and you should say so when you do.",
            "  HYPOTHETICAL — an open alternative that nothing has settled. It is not",
            "               something you know; it is something this world exists to resolve.",
            "  UNKNOWN    — looked for and not found. Do not fill it in.",
            "Your own position may be marked INFERRED or HYPOTHETICAL. That is not a gap to",
            "paper over: reason from your office, your obligations and your record, and reach",
            "the position those actually support. Do not invent personal circumstances,",
            "meetings, relationships or events that are not recorded here.",
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
            # The flag and the invitation used to disagree: the line below said "you may
            # propose a NOVEL action" unconditionally, so a moment compiled with
            # allow_novel false announced the restriction and then invited the actor past
            # it in the next sentence. The engine only enforces the flag when there are
            # NO feasible compiled actions, so on every other wake the invitation was the
            # whole of the rule — which is how the novel path kept firing at moments the
            # compiler had closed.
            *(
                [
                    "Choose ONE action. You may pick one of the feasible compiled "
                    "actions, or — if none fits your situation — propose a NOVEL action "
                    "describing what you attempt, its target, parameters and intended "
                    "effect. A novel action does NOT automatically happen: the world "
                    "decides. You state an intention only; you may never assert a "
                    "consequence.",
                ]
                if context.get("allow_novel", True)
                else [
                    "Choose ONE action from the feasible compiled actions above. This "
                    "moment does not admit novel actions: if none of them fits your "
                    "situation, choose wait and say what you are waiting for rather "
                    "than proposing something else.",
                ]
            ),
            "If an action needs a parameter (an option, an",
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
                "WHAT VERIFIED EVIDENCE CONTAINS. Every item here that is causally "
                "relevant to the outcome must appear in the compiled world — as an "
                "entity, action, field, document, resource, process node, external "
                "process or terminal term — or the run is refused. Items that are "
                "genuinely incidental to this question may be left out; the gate asks "
                "for a stated reason, not for everything",
                context.get("checklist", ""),
            ),
            str(context.get("extra_instruction") or ""),
            _COMPILE_RULES,
            _block("SCHEMA (fill this shape)", _WORLD_SCHEMA),
        ]
    )


_COMPILE_RULES = """RULES THE COMPILED WORLD MUST SATISFY:

WHAT PRODUCES THE OUTCOME. First decide what actually causes this outcome in the real world, then
represent that. A causal producer may be a person, an organization, an institutional body, a subunit,
a coalition, a population stratum, a network, a market, an administrative process, a production
system, a logistical system, or an external physical or economic process. Some questions are settled
by a handful of named people; some are settled by throughput, demand and a reporting calendar with no
individual deciding anything. Model whichever is true here.

Never invent a person who "decides" an aggregate. A quarterly delivery total is produced by
production, inventory, logistics and demand — not by an executive choosing a number. If the honest
answer is that no individual controls the outcome, compile no actors and put the causal machinery in
`external_processes`, operational `process` nodes and fields. That world is valid.

DO NOT MODEL THE ANNOUNCEMENT INSTEAD OF THE PRODUCTION. A scheduled release that publishes a figure
does not produce that figure; it reports whatever the operating world had already produced by then.
When the outcome is a quantity accumulated over a period — output, deliveries, volume, cases, votes —
compile the things that ADD to it: the producing units, their rate, the periods they operate in, and
what constrains them. Give those occurrences `adjust_field` effects that accumulate the quantity
across the window, and let the reporting event merely observe the total. Uncertainty then belongs on
the rate, the demand or the disruption — never on the total itself.

WHO COULD CHANGE THE ANSWER, NOT ONLY WHO PERFORMS THE FINAL ACT. The terminal act has one
performer; the outcome usually has many causes. Before you finish, ask of every material party: if
this were absent from the world, could the answer differ? If yes, it belongs in the world — as an
actor, an entity a process carries, an information channel, or an external process. If no, leave it
out and say why in structure_rationale.

Concretely, for a question about one person's public statement, the incoming data they have said
they depend on, the institution's own analysis, the scheduled occasions on which they speak, the
positions their colleagues have taken publicly, and what has actually reached them by each simulated
moment are all candidates — include the ones the evidence supports and that could move the answer.
For a question about a company's quarterly output, the demand segments that respond differently, the
company's own pricing and allocation decisions, the plants and their constraints, inventory entering
the period, logistics and the reporting definition are all candidates on the same test.

Do not add a party for realism decoration. An actor whose decisions cannot move any term the
terminal reads is scenery, and the run is refused for it.

REPRESENTATION SCALE. For every entity, choose the level that is causally faithful and say which:
individual, organization (acting as one unit), subunit, population_stratum, network, or
external_process. Do not turn a body of independent decision-makers into one actor, and do not
invent a million agents where an aggregate process is the honest representation. If an entity stands
for many real units, give represents_count — an aggregate is a compression, and a coalition of seven
countries deciding as one unit is one actor with represents_count 7, never one participant. Where
the decision rule matters, the real threshold governs: a body needing five of nine votes needs five
of nine however many objects the world uses to represent it. Never rewrite a real threshold to fit
the number of simulation objects, and never hide material internal disagreement inside an
aggregate — if members can genuinely differ in a way that changes the outcome, compile them
separately.

EVERY NUMBER NEEDS A SOURCE. A starting level, a capacity, a rate, a constraint, a threshold: each
must trace to evidence, and the field or entity carrying it must cite the claim ids. Do not invent a
plausible rate to make a process executable — that is the forecast smuggled in as a parameter. When
the evidence gives no usable number, say so honestly instead: declare the range as an uncertainty
with outcomes the evidence supports, model the alternative structures, or leave the term unproduced
so the branch reports unresolved. An ungrounded precise operating model is worse than an honest
unresolved one.

ACTORS ARE REAL OCCUPANTS OF REAL ROLES. Compile an actor for each person or body whose own
decisions genuinely move this outcome. Every entity that decides must appear in both `entities`
(is_actor true) and `actors`.

You do NOT need a first-person quotation to model someone. An actor is admissible when the evidence
establishes that they hold the relevant office, that the office carries relevant authority, and that
they sit inside the causal boundary. Grounding is ranked, strongest first:
  1. their own action or first-person statement
  2. verified office, membership and authority
  3. documented prior behavior
  4. official institutional or organizational policy
  5. contemporaneous reporting naming them
  6. role-level or institution-level behavioral evidence
Level 2 is enough. A minister, governor, commissioner, negotiator or executive does not stop existing
because no retrieved source quotes their private preference — that preference is exactly what the
simulation is for. Give it as an uncertainty, not as a fact and not as a reason to omit them.

What is refused is an actor with NO surviving citation of any kind: a name with nothing behind it.
So cite the claims that establish each actor's role and authority on the entity, put whatever the
evidence genuinely records in `memory_seeds` with its claim ids, and where the record is silent say
so plainly instead of inventing a statement. Never give one actor another actor's history.

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

UNCERTAINTY IS THE INPUT, NEVER THE ANSWER. Declare only genuine unknowns that can change the answer:
incoming data, private inclinations, interpretations, demand, throughput, implementation success,
attention, timing, delay. Give each outcome's weight a provenance, and say `release_at` if the
evidence establishes when the unknown value becomes public. If two unknowns are not independent, do
NOT list them separately — list one uncertainty whose outcomes are joint states, or the run is
refused.

WEIGHTS ARE EVIDENCE OR THEY ARE NOTHING. A relative weight may be used only when something
supports it: an empirical frequency, a documented base rate, a verified reference case, market or
survey evidence, or observed current-state evidence — name which in the provenance and cite the
claims. When nothing supports a split, use symmetric_ignorance_assumption and understand what that
means: it is a declaration that the weights are arbitrary, the run will report scenario bounds
rather than a calibrated point estimate, and inventing precision instead would be worse. Never
choose 0.5/0.5 because there are two possibilities. Two possibilities is not evidence about their
relative likelihood.

An uncertainty may set exogenous conditions. It may NEVER set a term the terminal reads — not the
decision, not the vote result, not whether it was unanimous, not the final count, not whether the
agreement was signed, not any other encoding of YES or NO. A world whose terminal terms come from
branch weights is refused, because then the branch weights are the forecast and the simulation is
scenery.

THE TERMINAL IS AN EXPRESSION OVER WORLD STATE, AND SOMETHING MUST PRODUCE IT. Write the condition
that makes the answer YES in terms of what will actually be true in the world, using only the
universal operators. Every term it reads must be written by something that runs: an action an actor
takes, a process node, or an external/operational process. Check each term before you finish — if
nothing writes it, the world is incomplete. Check it by name: the effect has to name the same
string the terminal reads. An action called vote_cut whose effects do not set the field the
terminal reads is a name, not a mechanism, and the namespaces are separate — append_record to a
collection does not set a field of the same name, and a document field is not a world field. Use `unresolved_when` for states where the process
genuinely did not determine the answer — an undetermined world must report as unresolved, never be
rounded to NO.

There is a fourth producer, and it is the verified evidence itself. If the record already
establishes what the question asks — the agreement was signed four months before the cutoff, the
figure has been reported, the meeting has happened — do not invent a future event to produce it
again. Put the established value in the field or document the terminal reads and cite the claim ids
that establish it, on that field or document. The citation is what makes it admissible: an initial
value with claim ids is a fact the record establishes, and an initial value without them is you
asserting an outcome, which will be refused. Do not use this for something you merely expect."""


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
   "documents":[{"document_id":"snake","fields":{"k":v},"evidence_claim_ids":["..."]}],
   "actions":[{"action_id":"snake_verb","meaning":"...","eligible_actors":["*"|"role:X"|"<entity>"],
               "required_authority":["capability_token"],"parameters":[{"name":"","type":"option|string|number",
               "required":true,"choices":[...]}],"preconditions":{"op":"...","args":[...]},
               "resource_costs":[["res",amt]],"stages":["<node stage>"],
               "valid_targets":["*"|"role:X"|"<entity>"],"visibility":"public|private",
               "duration_seconds":0,"delivery_delay_seconds":0,"notice_delay_seconds":0,
               "completion_conditions":{"op":"...","args":[...]},
               "effects":[{"op":"<one op below>", "<its params>":"..."}],
               "evidence_claim_ids":["..."]}],
   // EFFECT OPS AND THEIR EXACT PARAMETER KEYS (use these keys verbatim):
   //   set_field      {"field":"<field_id>","value":<v or an expression>}
   //   adjust_field   {"field":"<field_id>","delta":<number>}
   //   append_record  {"collection":"<name>","key":"...","value":<v>}
   //   create_event   {"event_type":"<label>","text":"...","data":{...}}
   //   schedule_event {"event_type":"<label>","text":"...","at":"<ISO>"}
   //   release_data   {"fields":{"<field_id>":<v>}}
   //   deliver_information {"text":"...","info_fields":{...},"to":["<actor>"]}
   //   set the field the TERMINAL reads, spelled identically. A set_field whose "field"
   //   is not a field the terminal reads writes nothing that matters.
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
add, subtract, multiply, divide, min, max, abs, round, and, or, not. A count/sum over a record
collection may take a trailing where-expr using item("value").
An effect parameter may itself be an expression, and that is how a quantity gets produced: an
effect that sets a quarter's output to multiply(field("last_quarter"), field("demand_multiplier"))
computes it from the world at the moment it fires. Prefer that to writing a number you worked out
yourself — a figure you compute here is your estimate, not something the world produced."""
