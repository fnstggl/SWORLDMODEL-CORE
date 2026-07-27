"""W4 / W5 — a world that says it has several parties in it must be a society.

The defect these gates end was live and it passed everything. `phase2/geopolitical2`
compiled nine entities — OPEC+, Saudi Arabia, Russia, Iraq, Kuwait, Algeria, Kazakhstan,
Oman, UAE — and the entire world contained one action, whose one effect set the one
boolean the terminal read. Eight of the nine parties held no affordance at all: they
could not advocate, resist, defect, stall or bargain, and none of them could tell another
anything. The world had the *form* of a many-sided situation and the *arithmetic* of a
single switch, and the run's integrity verdict was "verified".

It passed because every existing gate asks about the entities that DECIDE, and the plan
said only one of them did — while separately declaring eight decision-relevant
participants and putting represents_count 8 on the aggregate that absorbed them. Claiming
the participants twice and delivering them once is the contradiction, and it is what the
rules here are keyed to.

The domain below is invented and is nobody's acceptance question — three townships
sharing a hill — because the property is about shape, not subject. What is proven:

* the many-party-one-actor world is refused, naming the parties that cannot act;
* clearing ``decides`` does not silence it — the escape the old error text handed out;
* equipping those parties with grounded acts ADMITS the same world, which is the whole
  ordering rule: the gate is worthless with nothing to admit;
* a world genuinely settled by one decider still compiles, with nothing manufactured —
  the failure these gates must never cause;
* a mechanism is not a party and is never asked to act;
* a many-party world whose positions cannot reach anybody is refused;
* a position lowers to a real delivery to the OTHER parties, and never to its author.
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from sworldmodel.effects import _audience
from sworldmodel.semantic_lowering import lower_plan
from sworldmodel.semantic_plan import parse_semantic_plan, validate_semantic_plan

AS_OF = datetime.fromisoformat("2026-04-01T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-07-31T23:59:59+00:00")

# The record this world is compiled from. Every affordance below is traceable to one of
# these lines, because an affordance the evidence does not support is a fabricated actor
# — the failure that matters more than a missing one.
KNOWN = frozenset(
    {
        "c-g1",  # the committee announces the stocking limit
        "c-g2",  # the three townships hold and state positions at the Michaelmas meeting
        "c-g3",  # Corran has a standing position that the limit is too low
        "c-g4",  # each township controls the stock it puts on its own apportionment
        "c-g5",  # the estate factor's sole authority, in the single-decider variant
    }
)

MEETING = "2026-06-24T10:00:00+00:00"


def _validate(plan: dict[str, Any]) -> list[str]:
    return validate_semantic_plan(
        parse_semantic_plan(plan), as_of=AS_OF, horizon=HORIZON, known_claim_ids=KNOWN
    )


def _defects(errors: list[str]) -> set[str]:
    return {e.split(":", 1)[0] for e in errors}


def _township(name: str, *, decides: bool, claims: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "structural_type": "organization",
        "role": "township holding a share of the Ard Fell common grazing",
        "representation_scale": "organization",
        "represents_count": None,
        "decides": decides,
        "authority": "states its position on the stocking limit and controls the stock "
        "it puts on its own apportionment",
        "why_material": f"{name} is one of the three shareholders whose agreement the "
        "committee needs before it raises the limit",
        "terminal_state_it_can_change": "moves its own stated position, which the "
        "committee reads before it announces a raised limit",
        "information_received": "the positions the other townships state at the Michaelmas meeting",
        "if_removed": "one of the three positions the committee weighs disappears and "
        "the announcement could go the other way",
        "evidence_claim_ids": claims,
    }


def grazing_plan(*, equipped: bool) -> dict[str, Any]:
    """Three townships and the committee that announces their stocking limit.

    ``equipped=False`` is the shipped defect's exact shape: the committee is the only
    party that can act, it carries represents_count 3 so the declared participant count
    is satisfied, and the three townships stand in the world holding nothing. With
    ``equipped=True`` each township states its position and tells the others, the
    committee reads those positions before announcing, and the same world is admitted.
    """

    townships = ["Ardgour", "Beinn Dubh", "Corran"]
    committee: dict[str, Any] = {
        "name": "Ard Fell Grazings Committee",
        "structural_type": "coalition",
        "role": "announces the common grazing's stocking limit for the season",
        "representation_scale": "organization",
        # The heart of it: in the defect the aggregate stands in for all three, which is
        # what let the plan claim three participants while modelling one.
        "represents_count": 3 if not equipped else None,
        "decides": True,
        "authority": "announces the stocking limit once the townships have been heard",
        "why_material": "its announcement is the act the question asks about",
        "terminal_state_it_can_change": "sets whether a raised stocking limit has been announced",
        "information_received": "the positions the townships state at the meeting",
        "if_removed": "nothing announces a limit and the question cannot resolve YES",
        "evidence_claim_ids": ["c-g1"],
    }
    entities: list[dict[str, Any]] = [committee] + [
        _township(
            n,
            decides=equipped,
            claims=["c-g2", "c-g3", "c-g4"] if n == "Corran" else ["c-g2", "c-g4"],
        )
        for n in townships
    ]

    states: list[dict[str, Any]] = [
        {
            "name": "raised stocking limit announced",
            "owner": "Ard Fell Grazings Committee",
            "state_type": "boolean",
            "initial": "UNKNOWN",
            "why_material": "the terminal reads it",
            "evidence_claim_ids": ["c-g1"],
        }
    ]
    affordances: list[dict[str, Any]] = [
        {
            "name": "announce the raised limit",
            "meaning": "the committee announces that the stocking limit is raised",
            "actor": "Ard Fell Grazings Committee",
            "target": "",
            "authority_required": "the committee's authority to set the limit",
            "preconditions": "the townships have stated their positions at the "
            "Michaelmas meeting and none has withheld agreement",
            "visibility": "public",
            "duration_seconds": 0,
            "changes": [{"op": "set", "target": "raised stocking limit announced", "value": True}],
            "evidence_claim_ids": ["c-g1"],
        }
    ]
    if equipped:
        for n in townships:
            states.append(
                {
                    "name": f"{n} stated position on raising the limit",
                    "owner": n,
                    "state_type": "category",
                    "initial": "UNKNOWN",
                    "why_material": f"what {n} has said is one of the three positions "
                    "the committee weighs",
                    "evidence_claim_ids": ["c-g2"],
                }
            )
            for stance, verb, claim in (
                ("supports raising", "presses for a higher limit", "c-g3"),
                ("withholds agreement", "objects that the hill cannot carry it", "c-g2"),
            ):
                affordances.append(
                    {
                        "name": f"{n} {verb}",
                        "meaning": f"{n} {verb} at the Michaelmas meeting, and the other "
                        "townships hear it",
                        "actor": n,
                        "target": "",
                        "authority_required": "its share of the common grazing",
                        "preconditions": "",
                        "visibility": "public",
                        "duration_seconds": 0,
                        "changes": [
                            {
                                "op": "set",
                                "target": f"{n} stated position on raising the limit",
                                "value": stance,
                            },
                            {
                                "op": "send",
                                "target": f"{n}'s position on the stocking limit",
                                "recipients": [x for x in townships if x != n]
                                + ["Ard Fell Grazings Committee"],
                                "detail": f"{n} {stance}",
                            },
                        ],
                        "evidence_claim_ids": [claim],
                    }
                )

    return {
        "resolution": {
            "question": "Will the Ard Fell Grazings Committee announce a raised stocking "
            "limit before the end of July?",
            "yes_condition": "The committee announces a raised stocking limit.",
            "subject_entity": "Ard Fell Grazings Committee",
            "resolution_units": "an announced stocking limit",
            "target_outcome": "a raised stocking limit is announced",
            "expected_participants": 4 if equipped else 3,
            "evidence_claim_ids": ["c-g1"],
        },
        "entities": entities,
        "excluded_candidates": [
            {
                "name": "Ard Fell estate factor",
                "why_immaterial": "the factor collects the grazing rent but the record "
                "gives the limit to the committee, so removing the factor cannot change "
                "whether a raised limit is announced",
                "evidence_claim_ids": ["c-g1"],
            }
        ],
        "states": states,
        "events": [],
        "affordances": affordances,
        "processes": [
            {
                "name": "the Michaelmas grazings meeting",
                "meaning": "the dated meeting at which the townships are heard and the "
                "committee settles the limit",
                "kind": "actor_moment",
                "participants": (
                    ["Ard Fell Grazings Committee"] + townships
                    if equipped
                    else ["Ard Fell Grazings Committee"]
                ),
                "inputs": [],
                "at": MEETING,
                "deadline": "2026-07-31T23:59:59+00:00",
                "occurrences": [],
                "evidence_claim_ids": ["c-g2"],
            }
        ],
        "uncertainties": [],
        "terminal": {
            "form": "state_equals",
            "state": "raised stocking limit announced",
            "value": True,
        },
        "terminal_producer_note": "Nothing establishes the announcement at the cutoff; "
        "only the committee's own affordance sets it, at the Michaelmas meeting, after "
        "the townships have stated their positions.",
        "world_facts": [],
    }


def factor_plan() -> dict[str, Any]:
    """The same hill under a different record: one party decides, and that is the world.

    This is the case these gates must never break. The record here gives the limit to the
    estate factor alone; the townships are consulted by courtesy and the record shows them
    taking no act that bears on it. The honest world therefore has one decider and says
    so, and nothing in it is manufactured to reach a count.
    """

    return {
        "resolution": {
            "question": "Will the Ard Fell estate factor raise the stocking limit before "
            "the end of July?",
            "yes_condition": "The factor raises the stocking limit.",
            "subject_entity": "Ard Fell estate factor",
            "resolution_units": "a raised stocking limit",
            "target_outcome": "the factor raises the limit",
            "expected_participants": 1,
            "evidence_claim_ids": ["c-g5"],
        },
        "entities": [
            {
                "name": "Ard Fell estate factor",
                "structural_type": "person",
                "role": "sets the stocking limit under the estate's own regulations",
                "representation_scale": "individual",
                "represents_count": None,
                "decides": True,
                "authority": "raises or holds the stocking limit at will",
                "why_material": "the outcome is the factor's own act",
                "terminal_state_it_can_change": "sets whether the limit has been raised",
                "information_received": "the grazings clerk's stock returns",
                "if_removed": "nobody can raise the limit and the question cannot "
                "resolve YES at all",
                "evidence_claim_ids": ["c-g5"],
            }
        ],
        "excluded_candidates": [
            {
                "name": "Ardgour, Beinn Dubh and Corran townships",
                "why_immaterial": "the record shows the factor consults them as a "
                "courtesy and sets the limit regardless of what they say, so removing "
                "them cannot change whether the limit is raised",
                "evidence_claim_ids": ["c-g5"],
            }
        ],
        "states": [
            {
                "name": "stocking limit raised",
                "owner": "Ard Fell estate factor",
                "state_type": "boolean",
                "initial": "UNKNOWN",
                "why_material": "the terminal reads it",
                "evidence_claim_ids": ["c-g5"],
            }
        ],
        "events": [],
        "affordances": [
            {
                "name": "raise the limit",
                "meaning": "the factor raises the stocking limit",
                "actor": "Ard Fell estate factor",
                "target": "",
                "authority_required": "the factor's authority over the limit",
                "preconditions": "",
                "visibility": "public",
                "duration_seconds": 0,
                "changes": [{"op": "set", "target": "stocking limit raised", "value": True}],
                "evidence_claim_ids": ["c-g5"],
            },
            {
                "name": "hold the limit",
                "meaning": "the factor leaves the stocking limit where it is",
                "actor": "Ard Fell estate factor",
                "target": "",
                "authority_required": "the factor's authority over the limit",
                "preconditions": "",
                "visibility": "public",
                "duration_seconds": 0,
                "changes": [{"op": "set", "target": "stocking limit raised", "value": False}],
                "evidence_claim_ids": ["c-g5"],
            },
        ],
        "processes": [
            {
                "name": "the summer stock review",
                "meaning": "the dated review at which the factor settles the limit",
                "kind": "actor_moment",
                "participants": ["Ard Fell estate factor"],
                "inputs": [],
                "at": MEETING,
                "deadline": "2026-07-31T23:59:59+00:00",
                "occurrences": [],
                "evidence_claim_ids": ["c-g5"],
            }
        ],
        "uncertainties": [],
        "terminal": {"form": "state_equals", "state": "stocking limit raised", "value": True},
        "terminal_producer_note": "Only the factor's own affordance sets it; nothing "
        "establishes it at the cutoff and no uncertainty writes it.",
        "world_facts": [],
    }


# ---------------------------------------------------------------------------
# W4 — the many-party world that contains one decision
# ---------------------------------------------------------------------------


def test_a_world_that_declares_many_parties_and_lets_one_act_is_refused() -> None:
    """The shipped defect, in a domain that is not the shipped one.

    Three shareholders whose agreement the committee needs, and the committee holds the
    world's only act. Every reference resolves, the terminal has a real producer, the
    declared participant count is satisfied by represents_count — which is exactly how
    nine entities and one action reached a "verified" verdict.
    """

    errors = _validate(grazing_plan(equipped=False))
    assert "INERT_PARTICIPANT" in _defects(errors), errors
    inert = next(e for e in errors if e.startswith("INERT_PARTICIPANT"))
    for township in ("Ardgour", "Beinn Dubh", "Corran"):
        assert township in inert, "the refusal names the parties that cannot act"
    # And it says what to do about it in both directions, because "add an actor" alone
    # is the instruction that manufactures one.
    assert "excluded_candidates" in inert
    assert "fabricated actor" in inert


def test_clearing_decides_does_not_silence_the_society_gate() -> None:
    """The escape the old error text handed out, closed.

    ``entity X decides but has no affordance — give it the actions its role affords, or
    mark it decides=false`` was written by the same model that writes the affordance
    list, so a party could be stripped of agency for free and the only check that existed
    went quiet. The society rule reads structural type and affordances, never the flag,
    so demoting a party changes which rule fires and not whether the world is refused.
    """

    claimed = grazing_plan(equipped=False)
    for entity in claimed["entities"][1:]:
        entity["decides"] = True  # claims agency, still holds nothing
    with_flag = _validate(claimed)
    assert any("decides but has no affordance" in e for e in with_flag)

    demoted = grazing_plan(equipped=False)  # the same world with the flag cleared
    assert not any("decides but has no affordance" in e for e in _validate(demoted))
    assert "INERT_PARTICIPANT" in _defects(_validate(demoted)), "the demotion must not pay"

    # The old text's own advice must be gone from the message that offered it.
    text = next(e for e in with_flag if "decides but has no affordance" in e)
    assert "mark it decides=false" not in text
    assert "excluded_candidates" in text


def test_equipping_every_party_admits_the_same_world() -> None:
    """The half that makes the gate worth having: a real society compiles.

    Each township moves its own stated position and tells the others; the committee reads
    those positions before it announces. Nothing here is invented — every affordance
    cites the claim that attributes that act to that party.
    """

    assert _validate(grazing_plan(equipped=True)) == []


def test_a_genuinely_single_decider_world_needs_no_manufactured_participants() -> None:
    """The failure these gates must never cause.

    If the evidence supports one decider, one decider is the correct world. The three
    townships are still named — in excluded_candidates, where the omission is argued for
    — and nothing demands they be given powers the record does not give them.
    """

    errors = _validate(factor_plan())
    assert errors == []
    assert _defects(errors) & {"INERT_PARTICIPANT", "NO_PARTICIPANT_CHANNEL"} == set()


def test_a_mechanism_in_a_many_party_world_is_never_asked_to_act() -> None:
    """A market is not a party. The society rules ask "can this act?" of a township and
    not of the store-lamb price, because the price is something the world moves."""

    plan = grazing_plan(equipped=True)
    plan["entities"].append(
        {
            "name": "the store-lamb market",
            "structural_type": "market",
            "role": "sets what a lamb off the hill is worth",
            "representation_scale": "external_process",
            "represents_count": None,
            "decides": False,
            "authority": "none — it is a price, not a party",
            "why_material": "the price is what makes a higher limit worth asking for",
            "terminal_state_it_can_change": "none directly; it shapes the positions the "
            "townships state",
            "information_received": "nothing — it is not told things",
            "if_removed": "the townships' reasons for their positions lose their anchor",
            "evidence_claim_ids": ["c-g4"],
        }
    )
    assert _validate(plan) == []


# ---------------------------------------------------------------------------
# W5 — a position that cannot reach anybody is not a position
# ---------------------------------------------------------------------------


def test_a_many_party_world_with_no_channel_between_participants_is_refused() -> None:
    """Four parties deciding in four separate rooms is not a negotiation.

    Strip the sends and every township can still move its own position — but nobody can
    observe anybody, nothing propagates, and the trajectory is four monologues sharing a
    clock.
    """

    plan = grazing_plan(equipped=True)
    for affordance in plan["affordances"]:
        affordance["changes"] = [c for c in affordance["changes"] if c["op"] != "send"]
    errors = _validate(plan)
    assert "NO_PARTICIPANT_CHANNEL" in _defects(errors), errors


def test_telling_only_the_party_who_decides_is_still_a_real_channel() -> None:
    """The rule asks whether the world contains a channel between participants at all,
    not how wide it is.

    Here each township states its position only to the committee — no township hears
    another. That is a thinner world than the meeting really is, and it is still one in
    which a position propagates to somebody who can act on it, so it is admitted. What
    is refused is the world where nothing anybody does can be observed by anybody.
    """

    plan = grazing_plan(equipped=True)
    for affordance in plan["affordances"]:
        for change in affordance["changes"]:
            if change["op"] == "send":
                change["recipients"] = ["Ard Fell Grazings Committee"]
    assert _validate(plan) == []


def test_a_stated_position_lowers_to_a_real_delivery_to_the_other_parties() -> None:
    """W5's machinery, end to end: a semantic ``send`` becomes the runtime's proven
    ``deliver_information``, addressed to the parties who did not already know."""

    plan = grazing_plan(equipped=True)
    compilation, _mapping = lower_plan(parse_semantic_plan(plan))
    spec = compilation["world_spec"]
    id_by_name = {e["name"]: e["entity_id"] for e in spec["entities"]}

    deliveries = [
        (action, eff)
        for action in spec["actions"]
        for eff in action["effects"]
        if eff["op"] == "deliver_information"
    ]
    assert deliveries, "a world whose parties state positions must deliver them"
    for action, eff in deliveries:
        author = action["eligible_actors"][0]
        recipients = set(_audience("deliver_information", eff))
        assert recipients, "a delivery with no audience is a broadcast, not a message"
        assert author not in recipients, "nobody is told what they themselves just did"
        assert eff["text"], "an empty message carries nothing"
    # Every township hears from both of the others, which is what makes it a channel
    # rather than an announcement.
    for name in ("Ardgour", "Beinn Dubh", "Corran"):
        heard_from = {
            action["eligible_actors"][0]
            for action, eff in deliveries
            if id_by_name[name] in set(_audience("deliver_information", eff))
        }
        assert len(heard_from) == 2, f"{name} must be able to hear both other townships"


# ---------------------------------------------------------------------------
# The audience an act is addressed to (diagnosis: runtime-convergence)
# ---------------------------------------------------------------------------


def test_an_author_is_never_in_the_audience_of_the_event_it_records() -> None:
    """An audience is who should learn of a thing, and the author already knows.

    Lowering addressed an action's own ``create_event`` to its own participants,
    including the actor performing it, so the act was delivered back to its author,
    noticed, and counted as directed information to wake them with. Reproduced live by
    ``runtime-convergence``: a governor spent 79 of its 80 actor calls on
    ``directed_information`` wakes that were all echoes of its own act, and the branch
    died unresolved on budget. ``world.observers_of`` already declines to show an actor
    its own act unless the audience says otherwise — so lowering was defeating a guard
    the runtime already had.
    """

    plan = grazing_plan(equipped=True)
    plan["events"] = [
        {
            "name": "the limit announcement",
            "meaning": "the committee announces the raised limit to the townships",
            "participants": {
                "announcer": "Ard Fell Grazings Committee",
                "heard_by": "Corran",
            },
            "visibility": "private",
            "information_created": "the limit is raised",
            "evidence_claim_ids": ["c-g1"],
        }
    ]
    plan["affordances"][0]["changes"].append(
        {"op": "record_event", "target": "the limit announcement", "detail": "raised"}
    )
    assert _validate(plan) == []

    compilation, _mapping = lower_plan(parse_semantic_plan(plan))
    spec = compilation["world_spec"]
    id_by_name = {e["name"]: e["entity_id"] for e in spec["entities"]}
    creates = [
        (action, eff)
        for action in spec["actions"]
        for eff in action["effects"]
        if eff["op"] == "create_event"
    ]
    assert creates
    for action, eff in creates:
        audience = set(_audience("create_event", eff))
        assert id_by_name["Corran"] in audience, "the party it is announced TO still hears it"
        assert action["eligible_actors"][0] not in audience, "the announcer already knows"


def test_a_message_addressed_only_to_its_author_is_not_turned_into_a_broadcast() -> None:
    """The one place the rule stops.

    ``deliver_information`` with an empty audience is PUBLIC to the executor, so dropping
    the author from a message addressed to nobody else would turn a private note into a
    broadcast — a louder failure than the echo it was fixing. A send with no other
    recipient is left exactly as declared.
    """

    plan = grazing_plan(equipped=True)
    corran = next(a for a in plan["affordances"] if a["name"].startswith("Corran"))
    send = next(c for c in corran["changes"] if c["op"] == "send")
    send["recipients"] = ["Corran"]
    compilation, _mapping = lower_plan(parse_semantic_plan(plan))
    spec = compilation["world_spec"]
    id_by_name = {e["name"]: e["entity_id"] for e in spec["entities"]}
    action = next(a for a in spec["actions"] if a["eligible_actors"] == [id_by_name["Corran"]])
    eff = next(e for e in action["effects"] if e["op"] == "deliver_information")
    assert set(_audience("deliver_information", eff)) == {id_by_name["Corran"]}


def test_the_defect_and_its_fix_differ_only_in_what_the_parties_can_do() -> None:
    """The two worlds are the same world.

    Same question, same committee, same three townships, same terminal, same dated
    meeting. What the gate separates is not breadth or detail — it is whether the parties
    the plan itself says the outcome turns on are able to do anything about it.
    """

    scenery = grazing_plan(equipped=False)
    society = grazing_plan(equipped=True)
    assert scenery["resolution"]["question"] == society["resolution"]["question"]
    assert scenery["terminal"] == society["terminal"]
    assert [e["name"] for e in scenery["entities"]] == [e["name"] for e in society["entities"]]
    assert len(scenery["affordances"]) == 1
    assert len(society["affordances"]) == 7  # one per township position, plus the announcement
    assert _defects(_validate(scenery)) == {"INERT_PARTICIPANT"}
    assert _validate(society) == []

    # And the admitted world really is executable: every party reaches the floor with
    # something to do there.
    compilation, _mapping = lower_plan(parse_semantic_plan(copy.deepcopy(society)))
    spec = compilation["world_spec"]
    actors_with_actions = {a for action in spec["actions"] for a in action["eligible_actors"]}
    assert len(actors_with_actions) == 4
    node = spec["process"]["nodes"][0]
    assert set(node["participants"]) == actors_with_actions
