"""Turn a refusal into the next thing to look for.

A gate that refuses a world has, in the act of refusing, said something quite specific:
*this element is missing*. The previous behaviour threw that away. It re-sent the same
compile prompt with the error text appended, against the same evidence store, up to
three times, and then gave up. Three identical readings of the same facts is a reroll,
not a repair — and "three" was an arbitrary number that decided the fate of a question.

Repair here is driven by the *element* that is missing. A missing office-holder sends
the researcher after rosters and appointment records; a missing procedure sends it after
procedural rules and prior instances; a missing production pathway sends it after
capacity, facilities and reporting definitions. Where the failure is the compiler
contradicting itself or omitting something already in the evidence store, no research is
warranted at all and the repair is a reconciliation instruction.

Termination is not a counter. Repair continues while it is *getting somewhere* — while
each attempt either changes the diagnosed failure or adds new evidence — and stops when
an attempt does neither, which is the honest signal that there is no further defensible
source or representation path to try. The research backend's own budget bounds the total
cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import WorldIntegrityError

__all__ = ["RepairPlan", "plan_repair", "RepairLog"]


@dataclass(frozen=True)
class RepairPlan:
    """What to do about one specific refusal."""

    failure: str
    missing_element: str
    queries: tuple[str, ...]
    instruction: str

    @property
    def needs_research(self) -> bool:
        return bool(self.queries)

    def as_dict(self) -> dict[str, object]:
        return {
            "failure": self.failure,
            "missing_element": self.missing_element,
            "targeted_queries": list(self.queries),
            "needs_research": self.needs_research,
            "compiler_instruction": self.instruction,
        }


@dataclass
class RepairLog:
    """The record of what was tried, for the diagnosis artifact."""

    attempts: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        plan: RepairPlan | None,
        *,
        failure: str,
        message: str,
        claims_before: int,
        claims_after: int,
        outcome: str,
    ) -> None:
        self.attempts.append(
            {
                "failure": failure,
                "message": message.splitlines()[0] if message else "",
                "plan": plan.as_dict() if plan else None,
                "evidence_claims_before": claims_before,
                "evidence_claims_after": claims_after,
                "new_claims": max(0, claims_after - claims_before),
                "outcome": outcome,
            }
        )

    def as_list(self) -> list[dict[str, object]]:
        return list(self.attempts)


# ---------------------------------------------------------------------------


# The interrogative frame a forecasting question is wrapped in. Searching for it is
# searching for the wrong thing: a repair query built from the raw question reads "how is
# Will Tesla report more than 400,000 vehicle deliveries for the third quarter of 2026
# decided procedure steps", which is not a phrase any source contains. What a search
# engine needs is the subject.
_INTERROGATIVE = re.compile(
    r"^\s*(will|does|did|is|are|was|were|can|could|should|would|has|have|by\s+when|how"
    r"|what|when|who|whether)\b[\s,]*",
    re.IGNORECASE,
)
_TRAILING_WINDOW = re.compile(
    r"\s+(before|by|on or before|at or before|prior to|no later than)\s+[^,]*$", re.IGNORECASE
)


def _subject(question: str, subject_entity: str = "") -> str:
    """A search phrase for this question.

    Prefers the subject entity the research planner already derived, because that is the
    thing sources are written about. Falls back to the question with its interrogative
    frame and resolution window stripped.
    """

    if subject_entity.strip():
        return subject_entity.strip()[:140]
    text = question.strip().rstrip("?")
    text = _INTERROGATIVE.sub("", text)
    text = _TRAILING_WINDOW.sub("", text)
    return (text or question).strip()[:140]


def _strings(value: object) -> list[str]:
    """Detail values arrive as ``object``; read a list of strings out of one safely."""

    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def plan_repair(
    exc: WorldIntegrityError, question: str, *, subject_entity: str = ""
) -> RepairPlan | None:
    """Map a refusal to targeted research and a specific compiler instruction.

    Returns ``None`` when the failure is not one repair can address — an unrecognized
    failure, or one where retrying would only mask a real defect.
    """

    failure = str(exc.details.get("failure") or "")
    subject = _subject(question, subject_entity)
    builder = _PLANS.get(failure)
    if builder is None:
        return None
    return builder(exc, subject)


def _no_causal_producer(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="no_causal_producer",
        missing_element="who or what produces this outcome",
        queries=(
            f"who decides {subject}",
            f"{subject} decision-making body members current",
            f"{subject} official responsible authority",
            f"how is {subject} determined process",
        ),
        instruction=(
            "The previous compilation contained nothing that could produce the outcome: "
            "no actor and no process. Identify the real causal producer from the "
            "evidence. It does not have to be a person — an organization, a production "
            "or administrative process, a market or a population can produce an "
            "outcome, and for an aggregate quantity that is usually the truthful "
            "answer. If people decide it, compile them as actors: a verified office and "
            "authority is sufficient grounding and you do not need a first-person "
            "quotation. If throughput or administration produces it, compile "
            "external_processes and operational process nodes instead. What you may not "
            "do is compile an empty world, and what you may never do is invent a person "
            "who 'decides' a quantity that is really produced by operations."
        ),
    )


def _actors_ungrounded(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    names = [a.split("(")[0].strip() for a in _strings(exc.details.get("ungrounded_actors"))]
    queries: list[str] = []
    for name in names[:4]:
        clean = name.split(":")[0].strip()
        if not clean or "constructed representative" in name:
            # A synthetic stand-in has no name to look up. Searching for the identifier
            # a compiler invented for it — "other_board_members role authority appointed"
            # — is searching for a string no source contains, and a live Banxico run spent
            # a repair round doing exactly that. What is missing is the size of the group,
            # not a record of a person.
            continue
        # The identifier is what the gate reports; a search engine needs the words in it.
        queries.append(f"{clean.replace('_', ' ')} {subject} role authority appointed")
        queries.append(f"{clean.replace('_', ' ')} statement position record")
    if any("constructed representative" in n for n in names):
        queries.append(f"{subject} how many members full membership size")
    return RepairPlan(
        failure="actors_ungrounded",
        missing_element=f"evidence attaching {names or ['the compiled actors']} to the world",
        queries=tuple(queries),
        instruction=(
            "These actors carried no surviving citation of any kind: "
            f"{names}. An actor needs evidence that attaches it to this world — its "
            "office and authority is enough, and so is documented prior conduct, "
            "institutional policy or contemporaneous reporting that names it. Cite the "
            "claims that establish each actor's role on its entity. If the evidence "
            "genuinely establishes nothing about someone, remove them rather than "
            "inventing a record for them.\n"
            "A stand-in for people you cannot name is a different case, and deleting it "
            "is the wrong repair: the group is real even when its members are not in the "
            "record. Give it represents_count and cite the claim establishing how many "
            "it stands for. If nothing establishes the size, do not make it an actor at "
            "all — model the unnamed remainder as an uncertainty over the outcome, so "
            "the branches carry what is unknown instead of an actor pretending to know "
            "it. Removing it and leaving the world with nobody in it is not a repair."
        ),
    )


def _participants_omitted(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    absent = _strings(exc.details.get("absent from the world"))
    return RepairPlan(
        failure="participants_omitted",
        missing_element=f"entities for {absent}",
        queries=(),  # the evidence already has them; this is a compiler omission
        instruction=(
            f"The evidence names {absent} in role terms and your compiled world does "
            "not contain them at all. This is an omission, not a judgement call: add "
            "each of them as an entity with the claim ids that name them. Then decide "
            "the honest representation — an actor if their own decisions move this "
            "outcome, otherwise an entity whose effect the process carries — and say "
            "which via representation_scale."
        ),
    )


def _declared_participants_not_represented(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    declared = exc.details.get("declared by the compiled world")
    represented = exc.details.get("represented in the world")
    return RepairPlan(
        failure="declared_participants_not_represented",
        missing_element=f"a roster of {declared} (the world contains {represented})",
        queries=(f"{subject} full membership list all members names",),
        instruction=(
            f"You declared expected_participants={declared} and the world you compiled "
            f"contains {represented} entities in total. One of the two is wrong. Either "
            f"add the missing participants as entities, or correct expected_participants "
            "to the number the evidence actually establishes — and set it to null if the "
            "evidence does not establish a count. Do not invent a participant to reach a "
            "number. Note that this is a count of entities, not of deliberating actors: "
            "a participant the evidence names may be an entity whose effect a process "
            "carries rather than an actor that decides, and that choice is not the "
            "problem here."
        ),
    )


def _terminal_has_no_producer(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    orphans = _strings(exc.details.get("terminal terms with no producer"))
    writable_fields = _strings(exc.details.get("fields any action can write"))
    writable_colls = _strings(exc.details.get("collections any action can write"))
    connect = ""
    if writable_fields or writable_colls:
        connect = (
            f"Your actions already write {writable_fields or 'no fields'} and "
            f"{writable_colls or 'no collections'}, and the terminal reads {orphans}. "
            "Those are different names, so this is a wiring mistake rather than a missing "
            "mechanism: either give one of your actions an effect that writes exactly the "
            "term the terminal reads, or write the terminal over the term your actions "
            "actually produce. Namespaces are not interchangeable — a record appended to "
            "a collection does not set a field of the same name, and a document field is "
            "not a world field.\n"
        )
    elif exc.details.get("terminal reads"):
        connect = (
            "No action in your world writes any field or collection at all. An action "
            "whose effects change nothing the terminal can read is a name, not a "
            "mechanism: give each action the effect that records what doing it actually "
            "changes in the world.\n"
        )
    return RepairPlan(
        failure="terminal_has_no_producer",
        missing_element=f"a mechanism that produces {orphans}",
        queries=(
            f"how is {subject} decided procedure",
            f"{subject} production capacity output per quarter",
            f"{subject} how the figure is compiled and reported definition",
            f"{subject} schedule timeline announcement date",
        ),
        instruction=(
            f"Nothing in your world writes {orphans}, so the terminal read values that "
            "only a branch weight supplied.\n"
            + connect
            + "A common form of this mistake is modelling the ANNOUNCEMENT and omitting "
            "the PRODUCTION. A scheduled release that publishes a number does not "
            "produce that number; it reports whatever the operating world had already "
            "produced by then.\n"
            "So compile the pathway that actually generates each term. If it is a "
            "quantity accumulated over a period — output, deliveries, volume, cases, "
            "votes — model the things that add to it: the production or activity units, "
            "their rate, the periods in which they operate, and the constraints on them. "
            "Give the operating process `adjust_field` occurrences that accumulate the "
            "quantity across the window, and let the reporting event merely observe the "
            "total. If it is a discrete decision or act, model the actor doing it and "
            "the action whose effect writes the field.\n"
            "Uncertainty belongs on the INPUTS to that pathway — the rate, the demand, "
            "the disruption, the shortfall. It may never write the term itself."
        ),
    )


def _terminal_never_resolvable(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="terminal_never_resolvable",
        missing_element="an unresolved condition that some world can fail",
        queries=(),
        instruction=(
            "Your unresolved condition is true under every assignment of the fields it "
            f"reads, so no trajectory could ever resolve: {exc.details.get('unresolved_when')}. "
            "Check the connective. 'Unresolved if it is neither A nor B' is "
            "and(not_equals(x, A), not_equals(x, B)) — written with or(...) it is true "
            "for every value, because nothing can equal both. Write unresolved_when for "
            "the states in which the process genuinely did not determine the answer, and "
            "leave it out entirely if there are none."
        ),
    )


def _environment_presets_terminal(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    terms = _strings(exc.details.get("terms preset by the environment"))
    return RepairPlan(
        failure="environment_presets_terminal",
        missing_element=f"a producer for {terms} other than the calendar",
        queries=(f"{subject} how the decision is actually taken and recorded",),
        instruction=(
            f"A scheduled process in your world sets {terms} to a fixed value that no "
            "actor influences, so the answer is decided before anyone acts. Remove that "
            "effect. Model instead what the actors do that produces the outcome, and let "
            "the process node record or tally the result of their actions — gate it on a "
            "field their actions write, rather than asserting the value itself."
        ),
    )


def _actors_cannot_reach_terminal(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="actors_cannot_reach_terminal",
        missing_element="an action connecting the compiled actors to the outcome",
        queries=(f"{subject} what the decision-makers actually do procedure",),
        instruction=(
            "Your actors deliberate over fields the terminal never reads, so their "
            "decisions cannot change the answer. Either give them an action whose "
            "effects write a term the terminal reads, or — if this outcome genuinely is "
            "produced by processes rather than by anyone's decision — remove the actors "
            "and model it as an operational world."
        ),
    )


def _nothing_can_act(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="nothing_can_act",
        missing_element="actions or processes",
        queries=(
            f"{subject} procedure steps what happens",
            f"{subject} schedule of events dates",
        ),
        instruction=(
            "Your world has neither actions nor external processes, so nothing can "
            "happen in it. Compile what the participants in this world can actually do, "
            "and the non-agent processes that run on their own.\n"
            "If you emptied the world to satisfy an earlier refusal, that was the wrong "
            "move: an actor the evidence cannot ground should become an uncertainty or "
            "an entity a process carries, not a deletion. Keep whoever the evidence does "
            "establish and give them the actions their office lets them take."
        ),
    )


def _nothing_scheduled(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="nothing_scheduled",
        missing_element="a calendar of events",
        queries=(
            f"{subject} scheduled dates calendar next meeting",
            f"{subject} publication release schedule",
        ),
        instruction=(
            "Nothing in your world is scheduled to occur, so no actor will ever be in a "
            "position to act. Place the real dated moments of this process on the "
            "calendar as process nodes, and put scheduled releases and administrative "
            "clocks in external_processes."
        ),
    )


def _malformed_compilation(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="malformed_compilation",
        missing_element="a compilation the schema can read",
        queries=(),
        instruction=(
            "Your previous compilation could not be parsed: "
            f"{exc.details.get('parser_error')}. Emit the same world again with the "
            'schema\'s exact shapes — every expression is {"op": "...", "args": [...]} '
            "with args ALWAYS a list even when there is one argument, every *_ids field "
            'is a list of strings, and every effect is an object with an "op". Change '
            "nothing about what the world contains."
        ),
    )


def _unknown_expression_operator(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    unknown = exc.details.get("unknown operators")
    provides = exc.details.get("operators this runtime provides")
    return RepairPlan(
        failure="unknown_expression_operator",
        missing_element=f"expressions written with the runtime's own operators ({unknown})",
        queries=(),
        instruction=(
            f"These expression operators do not exist in this runtime: {unknown}. Rewrite "
            "those expressions using only the universal operators, which are exactly: "
            f'{provides}. A constant is {{"op": "const", "args": [true]}}. Change '
            "nothing about what the world contains — only how the conditions are written."
        ),
    )


def _decisive_evidence_contradiction(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    pairs = _strings(exc.details.get("contradictions"))
    return RepairPlan(
        failure="decisive_evidence_contradiction",
        missing_element=f"an authoritative source settling {pairs[:3]}",
        queries=(
            f"{subject} official statement confirmed",
            f"{subject} press release announcement official",
            f"{subject} latest confirmed decision",
        ),
        instruction=(
            f"Two verified claims contradict each other about a matter of fact: {pairs}. "
            "If newly retrieved evidence settles it, compile the world the authoritative "
            "source supports and cite it. If the disagreement is really about what will "
            "happen rather than what is the case, it is not a contradiction at all — "
            "represent it as an uncertainty with both outcomes and let the simulation "
            "resolve it. Do not pick a side without a source."
        ),
    )


def _terminal_reads_no_world_state(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    return RepairPlan(
        failure="terminal_reads_no_world_state",
        missing_element="a terminal condition that depends on the world",
        queries=(f"{subject} how the outcome is decided and recorded",),
        instruction=(
            "Your terminal condition does not read any world state, so its value is "
            "fixed before the simulation starts. Write it over the things this world "
            "actually contains: a field an action sets, a collection actions append to, "
            "a document an action creates, a resource a transfer moves, an event type "
            "actions emit, or the process stage. Then make sure something in the world "
            "actually writes whatever you chose."
        ),
    )


def _uncertainty_writes_terminal(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    encoded = _strings(exc.details.get("terminal terms written by an uncertainty"))
    return RepairPlan(
        failure="uncertainty_writes_terminal",
        missing_element=f"an exogenous uncertainty instead of {encoded}",
        queries=(),
        instruction=(
            f"Your uncertainty writes {encoded}, which the terminal reads. That makes the "
            "branch weights the answer whatever anyone does — and if the same term is "
            "also written by an action, the branch simply overrules the actor.\n"
            "An uncertainty is for what the world does TO the actors: incoming data, a "
            "release, demand, a delay, an interpretation, someone else's move. It is "
            "never for what an actor decides, and never for the outcome itself.\n"
            "So branch over the INPUT and let the decision follow from it. If the "
            "uncertainty is really about whether someone will act, delete it: that is "
            "precisely what the simulation is for."
        ),
    )


def _orphan_actors(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    orphans = _strings(exc.details.get("orphan_actors"))
    return RepairPlan(
        failure="orphan_actors",
        missing_element=f"entities for actors {orphans}",
        queries=(),
        instruction=(
            f"Actors {orphans} appear in `actors` with no matching record in `entities`. "
            "Add the entity for each, carrying its own evidence_claim_ids, or drop the "
            "actor."
        ),
    )


def _required_facts_unverified(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    missing = _strings(exc.details.get("missing"))
    return RepairPlan(
        failure="required_facts_unverified",
        missing_element=f"evidence for {missing}",
        queries=tuple(f"{subject} {m.split(':')[0]}" for m in missing[:4]),
        instruction=(
            f"You declared these facts load-bearing and cited nothing available for "
            f"them: {missing}. Either cite claims that are in the evidence and available "
            "by the cutoff, or stop declaring them required — a required_reality_fact is "
            "a promise that the world rests on it."
        ),
    )


def _coverage_incomplete(exc: WorldIntegrityError, subject: str) -> RepairPlan:
    missing = _strings(exc.details.get("missing_material_candidates"))
    return RepairPlan(
        failure="coverage_incomplete",
        missing_element=f"representation of {missing[:6]}",
        queries=tuple(f"{subject} {m}"[:140] for m in missing[:4]),
        instruction=(
            f"Verified evidence contains {missing[:6]}, and your compiled world "
            "represents none of it. Each item must appear as an entity, action, field, "
            "document, resource, process node, external process or terminal term — or "
            "be genuinely incidental to this question, in which case leave it out and "
            "the gate will ask an independent reviewer whether that was defensible."
        ),
    )


_PLANS = {
    "no_causal_producer": _no_causal_producer,
    "actors_ungrounded": _actors_ungrounded,
    "participants_omitted": _participants_omitted,
    "declared_participants_not_represented": _declared_participants_not_represented,
    "terminal_never_resolvable": _terminal_never_resolvable,
    "terminal_has_no_producer": _terminal_has_no_producer,
    "actors_cannot_reach_terminal": _actors_cannot_reach_terminal,
    "environment_presets_terminal": _environment_presets_terminal,
    "nothing_can_act": _nothing_can_act,
    "nothing_scheduled": _nothing_scheduled,
    "orphan_actors": _orphan_actors,
    "uncertainty_writes_terminal": _uncertainty_writes_terminal,
    "terminal_reads_no_world_state": _terminal_reads_no_world_state,
    "decisive_evidence_contradiction": _decisive_evidence_contradiction,
    "unknown_expression_operator": _unknown_expression_operator,
    "malformed_compilation": _malformed_compilation,
    "required_facts_unverified": _required_facts_unverified,
    "coverage_incomplete": _coverage_incomplete,
}
