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
import re
from datetime import datetime
from typing import Any

from .coverage import evidence_checklist
from .errors import GatewayError, WorldIntegrityError
from .evidence import EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .semantic_lowering import LoweringGap, lower_plan
from .semantic_plan import (
    ALTERNATIVE_CHANGE_KINDS,
    COMPARISONS,
    PROCESS_KINDS,
    REPRESENTATION_SCALES,
    STATE_KINDS,
    STATE_TYPES,
    STRUCTURAL_TYPES,
    TERMINAL_FORMS,
    TERMINAL_SENSITIVITIES,
    WEIGHT_PROVENANCES,
    SemanticPlan,
    SemanticPlanError,
    defects_in,
    parse_semantic_plan,
    validate_semantic_plan,
)
from .world_compiler import _normalize_compilation, render_evidence

# Names in a claim's entity list that are not parties at all: dates, years, quantities,
# numbered instances of a meeting. Filtered out of the per-name brief because a heading
# for "2.2 million barrels per day" is noise, not a participant. Deliberately narrow —
# anything it is unsure of stays in, because the brief is evidence for the planner to
# read, never an instruction about who must be in the world.
_NOT_A_NAME = re.compile(
    r"^\s*(?:\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)


def participant_brief(
    view: EvidenceView, *, max_names: int = 24, max_claims: int = 8, max_chars: int = 24_000
) -> str:
    """The same verified claims, re-projected under the names they attest.

    ``render_evidence`` lists claims in authority order, which is the right ordering for
    judging what is established and the wrong one for seeing what any single party does:
    a party's acts are scattered across the list, and one line naming eight countries
    reads as a fact about the group rather than as eight facts about its members. A live
    OPEC+ compile put nine parties in the world and gave eight of them nothing to do,
    while the record it was reading attributes a standing position to Iraq and a
    declared flexibility to increase, pause or reverse to all seven meeting participants.

    So this adds no information and asserts nothing: every line is a claim already in the
    view, printed under each name that claim attests, with the excerpt where the excerpt
    says more than the proposition. What it changes is that a party with nothing under
    its own heading is *visible* as a party with nothing to do — which is exactly the
    judgement the planner has to make about whether it belongs in the world at all.
    """

    claims = view.available()
    by_name: dict[str, list[Any]] = {}
    for c in claims:
        for name in c.entities:
            n = name.strip()
            if not n or _NOT_A_NAME.match(n):
                continue
            by_name.setdefault(n, []).append(c)
    if not by_name:
        return ""
    ordered = sorted(by_name, key=lambda n: (-len(by_name[n]), n))[:max_names]
    lines = [
        "WHAT THE RECORD SAYS ABOUT EACH NAME IT ATTESTS. These are the same claims as "
        "above, grouped by the names each one names — no new evidence, and no claim "
        "about who belongs in the world. Read it to see what the record actually "
        "attributes to each name: which acts it takes, which authority it is said to "
        "hold, and where it appears only as a name in someone else's sentence. An "
        "affordance you give a party must be traceable to a line under that party's own "
        "heading.",
    ]
    # Bounded overall as well as per name: a live store with a hundred claims across
    # thirty names would otherwise put more than a hundred kilobytes of duplicated
    # evidence in front of the planner, and a brief that crowds out the plan is not a
    # help. Names are already ordered by how much the record says about them, so the
    # truncation drops the thinnest headings first.
    used = len(lines[0])
    for name in ordered:
        block = [f"- {name}"]
        for c in sorted(by_name[name], key=lambda c: (-int(c.authority_level), c.id))[:max_claims]:
            block.append(f"    {c.id} | {c.proposition}")
            excerpt = (c.supporting_excerpt or "").strip()
            if excerpt and excerpt[:60] != c.proposition[:60]:
                block.append(f'      "{excerpt[:400]}"')
        size = sum(len(x) + 1 for x in block)
        if used + size > max_chars:
            break
        lines.extend(block)
        used += size
    return "\n".join(lines)


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
   "terminal_state_it_can_change": "<which state the terminal reads, or is computed
     from, this entity can move — and through what act or mechanism>",
   "information_received": "<what this entity learns during the window, through which
     declared events, releases or messages>",
   "if_removed": "<what the world loses, and how the answer could differ, if this
     entity were deleted>",
   "evidence_claim_ids": ["..."]
 }}],
 "excluded_candidates": [{{
   "name": "<a person, organization, population or process the evidence names that you
     deliberately did NOT put in the world>",
   "why_immaterial": "<why its removal cannot materially change the answer>",
   "evidence_claim_ids": ["..."]
 }}],
 "states": [{{
   "name": "<ordinary-language name, unique>",
   "owner": "<entity name or 'world' — for a stock, who HOLDS the quantity>",
   "state_type": "<one of {list(STATE_TYPES)}>",
   "kind": "<one of {list(STATE_KINDS)} — see WHAT KIND OF THING THIS IS; default level>",
   "capacity": <stock only: the physical ceiling it can be filled to, or null>,
   "conserved_floor": <stock only: the level it cannot be drawn through; default 0>,
   "period": "<flow only: ISO-8601 duration the rate is quoted over, e.g. P1W>",
   "not_a_stock_because": "<level only, and ONLY when something draws this quantity down
     or the terminal reads it as an accumulating total: why this quantity is a reading or
     a record rather than something held somewhere>",
   "unit": "<REQUIRED for a quantity — what it is measured in. A unit written 'per'
     something (vehicles per week, acre-feet/day) IS a rate and must be kind=flow>",
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
   "changes": [<see CHANGES — an act that MOVES something uses set/increase/decrease, an
     act that OCCURS uses record_event, and an act that TELLS somebody uses send. Most
     acts in a many-sided situation do two of these at once: taking a position both
     moves the actor's own declared state and tells the other parties about it, e.g.
     [{{"op":"set","target":"<this party's stated position>","value":"<the position>"}},
      {{"op":"send","target":"<what is communicated>","recipients":["<the other parties>"],
        "detail":"<what they now know>"}}]>],
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
   "recurrence": {{
     "period": "<ISO-8601 duration this mechanism repeats on, e.g. P1D, P1W, P1M, PT6H>",
     "start": "<ISO datetime of the first firing>",
     "end": "<ISO datetime after which it stops>",
     "description": "<what happens at each firing>",
     "changes": [<see CHANGES — applied at EVERY firing>]
   }},
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
     "meaning": "<what is true about the world if this alternative holds>",
     "why_unresolved": "<why the record does not settle whether it holds>",
     "changes": [<one or more of {list(ALTERNATIVE_CHANGE_KINDS)} — what differs under
       it: the world's structure, an actor's state, or a process's state>],
     "terminal_sensitivity": "<one of {list(TERMINAL_SENSITIVITIES)}>",
     "evidence_claim_ids": ["<the claims that establish this as a real possibility>"]
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
 "zero_actor_justification": {{
   "no_material_decision": "<REQUIRED ONLY when no entity has decides=true: which human
     or population decisions could bear on this outcome, and why the record shows none
     of them can move it inside the window>",
   "process_sufficiency": "<why the non-agent process alone is causally sufficient>",
   "evidence_claim_ids": ["<the claims that establish both>"]
 }},
 "single_multiplier_exemption": {{
   "empirical_model": "<REQUIRED ONLY when one multiplier genuinely stands for a whole
     operating system: the documented empirical model that licenses it>",
   "parameter_uncertainty": "<how the parameter's uncertainty is grounded>",
   "evidence_claim_ids": ["<the claims documenting the model>"]
 }},
 "world_facts": [{{"text": "...", "evidence_claim_ids": ["..."]}}]
}}

RATES AND QUANTITIES ARE DIFFERENT THINGS. A rate is a quantity per unit of time, and
it produces nothing until you say how long it ran: writing {{"op":"increase","target":
"<total>","amount":{{"kind":"state","state":"<a weekly rate>"}}}} adds a rate to a total
and is refused. Multiply it by the time that firing covers —
{{"kind":"product","parts":[{{"kind":"state","state":"<the rate>"}},
{{"kind":"duration","value":"P1W"}}]}} — so the total is the rate times the time it ran
rather than the rate times however many dates appear in the plan.

WHAT KIND OF THING A STATE IS. Every state declares its kind, because conservation
follows from it and nothing else can supply it:
  stock — a real quantity HELD somewhere: water behind a dam, grain in an elevator,
    vehicles on a lot, beds on a ward, berths at a quay, ballots in a box. Give it the
    holder in "owner", a known starting amount, and — if anything adds to it — the
    "capacity" it fills up to. Change it only with increase/decrease: a stock MOVES, it
    is never "set", and the runtime refuses any move that would overdraw or overfill it.
  flow  — a RATE, and therefore only meaningful with the "period" it is quoted over
    (P1D, P1W, P1M). A rate produces nothing until a process applies it, so the process
    that applies it must fire at that rate across its whole window.
  level — a reading, an indicator, a category, a boolean, or a running record of what
    has already happened. Unconstrained. If something DECREASES a quantity-level, or the
    terminal reads one the world accumulates, say in "not_a_stock_because" why it is not
    a quantity held somewhere — otherwise declare it a stock.

CADENCE. A mechanism that repeats declares "recurrence" (period + start + end + the
changes each firing makes) and code enumerates every firing. Do NOT hand-write the
dates: a weekly process typed as two dates becomes a twice-a-quarter process, and the
answer becomes a fact about your typing. Use "occurrences" only for genuinely irregular
or dependent firings, and never alongside a recurrence.

CHANGES — the only universal change operations (they apply to dynamically named
real-world objects; there is no domain event list):
  {{"op": "set", "target": "<state name>", "value": <literal |
      {{"kind":"state","state":"<state name>"}} |
      {{"kind":"product"|"sum","parts":[<values>]}} |
      {{"kind":"duration","value":"<ISO-8601 duration, e.g. P1W>"}}>}}
  {{"op": "increase"|"decrease", "target": "<state name>", "amount": <same value forms>,
      "drawn_from": "<for an increase that carries quantity OUT of a stock: that stock's
        name. The same firing must decrease it by the same amount — one movement>"}}
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
- every person / organization / coalition / institution / population entity holds at
  least one affordance — a party that can do nothing belongs in excluded_candidates or
  in world_facts, not in the world;
- a world with two or more deciding entities carries at least one channel between
  participants: an affordance with a "send" to another party, or one recording an event
  that reaches another party;
- a quantity anything decreases is a stock, or says in not_a_stock_because why it may go
  below zero; a stock starts at a known amount, is never "set", and declares a capacity
  if anything adds to it; a flow declares a positive ISO-8601 period;
- a process that applies a flow either declares a recurrence at that flow's own period,
  or enumerates every occurrence its window requires at that period — a rate applied
  fewer times than the window holds makes the total an artifact of the schedule;
- a recurrence needs a positive period, a start before its end, at least two firings,
  its own changes, and no hand-written occurrences beside it;
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
- every entity answers all five representation questions (why_material,
  terminal_state_it_can_change, information_received, authority, if_removed), and every
  candidate you deliberately left out is listed in excluded_candidates with why its
  removal cannot matter;
- every uncertainty alternative states meaning, why_unresolved, changes and
  terminal_sensitivity — an alternative that cannot say what it means is filler, and an
  alternative that decides the terminal while citing nothing is refused;
- a terminal quantity may not be produced by a single non-agent change reading only
  inputs nothing else in the world produces (one set / one multiplier is a reported
  figure, not a process) unless single_multiplier_exemption is declared and cited;
- alternatives of one uncertainty may not land on opposite sides of the terminal
  threshold unless their VALUES are grounded in cited evidence: two invented numbers
  either side of the break-even ARE the answer, and the plan is refused;
- a world with no deciding entity needs zero_actor_justification with citations, and
  every deciding entity needs a path by which its decisions reach the terminal;
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
and never expand an aggregate that genuinely decides as one unit. For each one you
include, answer the five representation questions in its fields; for each one you leave
out, say in excluded_candidates why removing it cannot change the answer. A world with
NO deciding entity is legitimate only when the evidence shows no material human or
population decision can move this outcome and the non-agent process is sufficient on its
own — say that, with citations, in zero_actor_justification. An entity whose decisions
cannot reach anything the terminal reads is decoration: give it its real causal path or
leave it out.

PRODUCTION, NOT REPORTING. A reported figure is produced by operations — throughput,
demand, constraints — not by the report. Model the producing mechanism as operational
processes whose occurrences increase the quantity across the window, with uncertainty
on the drivers (rate, demand, disruption), never on the total itself. One final
set-the-total, or one cited base times one invented factor, is a forecast wearing a
world's clothes: it has no intermediate state, spans no time, and its answer is decided
by the factor you chose. Give the quantity grounded inputs, at least one intermediate
state the mechanism updates, and occurrences across the real causal period.

PHYSICAL QUANTITIES ARE CONSERVED, AND RATES NEED CADENCES. When the mechanism moves
real things — units built, delivered, released, admitted, loaded, counted — declare each
of those quantities a stock with its holder, its starting amount and, where anything
adds to it, its capacity; then the world cannot deliver what it does not have or fill
what has no room. Declare the rates that drive it as flows with their periods, and give
the process that applies them its recurrence. A world that models throughput as bare
arithmetic on plain levels can and does run its inventory tens of thousands of units
below zero for a whole quarter and resolve on the result.

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
Every alternative must say what it MEANS, why the record leaves it open, what it changes
and how the terminal responds under it: an alternative you cannot describe is filler
carrying mass it has not earned. Above all, never place invented numbers either side of
the threshold the question turns on — if the values are not in the record, the answer
would be your choice of numbers rather than anything the world does.

EVERY NUMBER NEEDS A SOURCE. A precise initial quantity cites claim ids or is UNKNOWN.
UNKNOWN stays UNKNOWN — the runtime treats reading it as honestly unresolved.

ACTORS ARE REAL OCCUPANTS OF REAL ROLES. A verified office and authority is sufficient
grounding (cite the claims); a name with nothing behind it is not an actor. Give each
deciding entity the genuine alternative affordances its role affords — including the
ones that would resolve the question NO — and place its real dated occasions to act as
actor_moment processes with at/deadline inside the window.

A PARTY IN THE WORLD IS A PARTY THAT CAN ACT. Start from what the record attributes to
each party under its own name, and give that party the act it describes: urging a
reassessment, pressing for a level, reaffirming caution, retaining the flexibility to
increase or pause or reverse, agreeing to an adjustment, acting on the output it alone
controls. Those are affordances, and the record naming a party doing one is exactly the
grounding an affordance needs (cite the claim). Every person, organization, coalition,
institution or population you place in "entities" must end up holding at least one.

There are two ways to fail this and they are not symmetric. Inventing an act the record
does not attribute to a party is a FABRICATED ACTOR and is the worse failure: it puts a
capability into the world that nothing supports, and the answer then turns on a power
you granted. But emptying the world is not the safe alternative. Deleting the parties
rather than equipping them leaves one actor deciding alone, which is the same defect
with fewer names in it, and a world whose answer nobody can influence is not a
simulation of anything. So: equip the parties the record shows acting; exclude only a
party the record shows taking no act that bears on this outcome, saying so in
excluded_candidates; and if the outcome genuinely IS one party's decision to take —
which happens, and is a legitimate world — say that plainly with expected_participants
matching, rather than compressing several real deciders into one aggregate.

A BODY THAT DECIDES BY ITS MEMBERS AGREEING IS NOT ONE DECIDER. Collapsing a group into
a single object with represents_count is right only when its members have no separate
position to hold — a delegation voting as instructed, a bloc with one mandate. It is
WRONG wherever the record shows the members arriving at the decision: if it says they
met and agreed, that is several parties and the rule by which they settle, and the thing
your simulation has to play out is whether they agree THIS time. Two tests, both read
off the record and not off your judgement of how important the members are:
  · does the record name the members individually as the parties who take the decision?
  · does it attribute a distinct position, demand, reservation or act to any one of them
    by name — urging, pressing, objecting, reaffirming, holding out?
If either is yes, that member holds its own position and its own affordances, and
absorbing it into the aggregate deletes evidence you were given. Note what the second
test costs you if you ignore it: a member with a named standing position, dropped into
excluded_candidates as immaterial, is a verified claim the world silently lost — the
coverage gate reads that as evidence destroyed and refuses the run.

POSITIONS BEFORE OUTCOMES. Where several parties bear on one decision, the world needs
the state that sits BETWEEN them: what each party has said, conceded, committed to or
refused so far. Declare that intermediate state, give each participant the act by which
it moves its own part of it — advocating a level, resisting one, signalling a position,
committing, withholding agreement, acting unilaterally on what it alone controls — and
let the party that performs the final act read it. An affordance whose only change is to
set the state the terminal reads IS the answer rather than a route to it: the world then
contains one decision, however many names are standing around it.

PARTICIPANTS MUST BE ABLE TO TELL EACH OTHER. A world whose occupants cannot communicate
is not a model of a social situation. The "send" change is how one party's act reaches
another, and it is almost always under-used: an act that in reality everybody hears
about — a statement, a position taken at a meeting, a notified decision, a public
reaffirmation — is written as a change to a state and nothing else, so in the simulation
nobody learns it happened and nobody can respond. Where two or more entities decide, at
least one of them must hold an affordance that carries information TO another:

  {"op": "send", "target": "<what is communicated>",
   "recipients": ["<the other parties, by entity name>"],
   "detail": "<what they now know>"}

or a declared event whose participants include them. That is what lets a position
propagate: one party signals, another sees it, and can decide differently because of it.
Model the channels the record shows are really there — the meetings they hold, the
statements they issue, the decisions they notify each other of — never invented
back-channels, and never a message to yourself: you already know what you just did."""


def _plan_prompt(
    question: str,
    as_of: datetime,
    horizon: datetime,
    evidence: str,
    *,
    checklist: str = "",
    brief: str = "",
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
        brief,
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
12. Is the world causally complete without unnecessary breadth?
13. Does each included entity's representation record hold up — can it really alter the
    terminal-relevant state it claims, does it really receive that information, and
    would removing it really change what it says it would?
14. Is any excluded candidate's "why_immaterial" actually false on this evidence?
15. Does any uncertainty alternative carry mass without carrying meaning — a filler
    value, a residual label, a value the record never names?
16. Does every party in the world hold an act the record actually attributes to it — and
    conversely, is any affordance an act the record never says that party takes? Both are
    failures: a party that can do nothing is scenery, and an invented capability is a
    fabricated actor.
17. Can a position propagate? If two or more parties decide, is there a real channel by
    which one learns what another has done — or does each decide in isolation while the
    outcome rests on one party's single act?"""


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
    brief: str = "",
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
                brief=brief,
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
    brief = participant_brief(view)
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
            brief=brief,
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
                brief=brief,
                extra_instruction=extra_instruction,
                prior=raw,
                corrections=[f"fix the plan's shape: {e}" for e in exc.errors[:12]],
                attempt=attempt + 100,
            )
            responses.append(resp2)
            try:
                plan = parse_semantic_plan(raw2)
            except SemanticPlanError as exc2:
                # A second unreadable shape is a real refusal — but it must refuse AS
                # the pipeline's own refusal type. Raw SemanticPlanError is a
                # ValueError: it bypassed the repair registry, was misfiled as a
                # research-stage failure, and threw away a completed live research
                # record (a FIFA holdout run lost 10 queries' evidence to
                # "terminal.parts[0]: all_of needs at least 2 part(s)").
                raise WorldIntegrityError(
                    "the semantic plan is unreadable after a shape-correction round: "
                    + "; ".join(exc2.errors[:6]),
                    details={
                        "failure": "semantic_plan_invalid",
                        "recompilable": True,
                        "semantic_errors": list(exc2.errors),
                    },
                ) from exc2
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
                # Which named causal-world defects fired, so a refused run is readable
                # without parsing prose and a reviewer can see the rule by name.
                "semantic_defects": defects_in(errors),
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
                    "semantic_defects": defects_in(errors),
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
                    "semantic_defects": defects_in(errors),
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
