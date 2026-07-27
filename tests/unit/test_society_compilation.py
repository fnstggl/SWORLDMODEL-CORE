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

from _fakes import ProgrammableGateway, build_bundle, wait_decision
from sworldmodel.effects import _audience
from sworldmodel.engine import run
from sworldmodel.models import ResolutionContract
from sworldmodel.semantic_lowering import lower_plan
from sworldmodel.semantic_plan import parse_semantic_plan, validate_semantic_plan
from sworldmodel.world_compiler import compile_world

AS_OF = datetime.fromisoformat("2026-04-01T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-07-31T23:59:59+00:00")

# The record this world is compiled from. Every affordance below is traceable to one of
# these lines, because an affordance the evidence does not support is a fabricated actor
# — the failure that matters more than a missing one.
CLAIMS: dict[str, dict[str, Any]] = {
    "c-g1": {
        "proposition": "The Ard Fell Grazings Committee announces the common grazing's "
        "stocking limit for each season",
        "value": "the committee announces the limit",
        "entities": ["Ard Fell Grazings Committee"],
    },
    "c-g2": {
        "proposition": "Ardgour, Beinn Dubh and Corran state their positions on the "
        "stocking limit at the Michaelmas grazings meeting",
        "value": "positions stated at the meeting",
        "entities": ["Ardgour", "Beinn Dubh", "Corran", "Ard Fell Grazings Committee"],
    },
    "c-g3": {
        "proposition": "Corran has pressed for a higher stocking limit at successive "
        "grazings meetings",
        "value": "Corran presses for a higher limit",
        "entities": ["Corran"],
    },
    "c-g4": {
        "proposition": "Each township controls the stock it puts on its own apportionment "
        "of the Ard Fell common grazing",
        "value": "each township controls its own apportionment",
        "entities": ["Ardgour", "Beinn Dubh", "Corran"],
    },
    "c-g5": {
        "proposition": "The Ard Fell estate factor sets the stocking limit under the "
        "estate's regulations and consults the townships as a courtesy",
        "value": "the factor sets the limit",
        "entities": ["Ard Fell estate factor"],
    },
}
KNOWN = frozenset(CLAIMS)

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
        # One entry per township, and each says what the record puts under that name
        # before saying why it cannot matter here. This fixture was written the other
        # way — one entry named "Ardgour, Beinn Dubh and Corran townships" carrying one
        # argument — and EXCLUSION_UNARGUED caught it in the file whose subject is that
        # exact defect. Corran's own entry has to survive contact with c-g3, which says
        # Corran has pressed for a higher limit at successive meetings: it is immaterial
        # to THIS question because the factor sets the limit whatever it hears, not
        # because nothing is recorded under Corran's name.
        "excluded_candidates": [
            {
                "name": "Ardgour",
                "record_attributes": "states its position on the stocking limit at the "
                "Michaelmas meeting and controls the stock on its own apportionment",
                "why_immaterial": "the record shows the factor sets the limit under the "
                "estate's regulations and consults the townships as a courtesy, so a "
                "position Ardgour states cannot change whether the limit is raised",
                "evidence_claim_ids": ["c-g2", "c-g4", "c-g5"],
            },
            {
                "name": "Beinn Dubh",
                "record_attributes": "states its position on the stocking limit at the "
                "Michaelmas meeting and controls the stock on its own apportionment",
                "why_immaterial": "same record, same courtesy consultation: nothing "
                "Beinn Dubh states reaches the factor's own decision on the limit",
                "evidence_claim_ids": ["c-g2", "c-g4", "c-g5"],
            },
            {
                "name": "Corran",
                "record_attributes": "has pressed for a higher stocking limit at "
                "successive grazings meetings, and controls the stock on its own "
                "apportionment",
                "why_immaterial": "Corran's pressing is recorded and the record equally "
                "shows the factor sets the limit regardless of what the townships say, "
                "so the pressing cannot change whether the limit is raised",
                "evidence_claim_ids": ["c-g3", "c-g4", "c-g5"],
            },
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


def test_two_aggregates_of_the_same_people_are_not_two_participants() -> None:
    """`phase2/geopolitical3` is why "one actor" is not the whole rule.

    That world declared twenty-three participants across eleven entities and had TWO
    acting objects — OPEC+ standing for 23 and a seven-country group standing for 7 —
    with the eight named countries listed in the world between them, holding nothing.
    Two actors looks like a society and is not one when both of them are aggregates of
    the very parties also standing there: the plan compresses the members to satisfy the
    count and displays them to look populated, which claims them twice.
    """

    plan = grazing_plan(equipped=False)
    # A second acting aggregate, so "at most one party acts" no longer holds.
    plan["entities"].insert(
        1,
        {
            "name": "the three townships in common",
            "structural_type": "coalition",
            "role": "the shareholders acting as one body when they agree",
            "representation_scale": "organization",
            "represents_count": 3,
            "decides": True,
            "authority": "settles a common position when the townships agree one",
            "why_material": "a common position is what the committee acts on",
            "terminal_state_it_can_change": "moves the position the committee reads",
            "information_received": "what each township says at the meeting",
            "if_removed": "the committee has no common position to act on",
            "evidence_claim_ids": ["c-g2"],
        },
    )
    plan["affordances"].append(
        {
            "name": "settle a common position",
            "meaning": "the townships in common settle their position",
            "actor": "the three townships in common",
            "target": "",
            "authority_required": "the townships' common agreement",
            "preconditions": "",
            "visibility": "public",
            "duration_seconds": 0,
            "changes": [{"op": "set", "target": "raised stocking limit announced", "value": True}],
            "evidence_claim_ids": ["c-g2"],
        }
    )
    plan["processes"][0]["participants"].append("the three townships in common")

    errors = _validate(plan)
    assert "INERT_PARTICIPANT" in _defects(errors), errors
    inert = next(e for e in errors if e.startswith("INERT_PARTICIPANT"))
    assert "stand in for their members" in inert, "the refusal names the contradiction"
    for township in ("Ardgour", "Beinn Dubh", "Corran"):
        assert township in inert


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
    # Two defects, one world: the committee cannot act for three townships that are
    # standing beside it holding nothing (INERT_PARTICIPANT), and it has not said what
    # licenses it to act for them at all (AGGREGATE_UNJUSTIFIED). The second is the one
    # the recorded OPEC+ bench trips instead of the first, because a plan that deletes
    # the members rather than displaying them leaves INERT_PARTICIPANT nobody to name.
    assert _defects(_validate(scenery)) == {"INERT_PARTICIPANT", "AGGREGATE_UNJUSTIFIED"}
    assert _validate(society) == []

    # And the admitted world really is executable: every party reaches the floor with
    # something to do there.
    compilation, _mapping = lower_plan(parse_semantic_plan(copy.deepcopy(society)))
    spec = compilation["world_spec"]
    actors_with_actions = {a for action in spec["actions"] for a in action["eligible_actors"]}
    assert len(actors_with_actions) == 4
    node = spec["process"]["nodes"][0]
    assert set(node["participants"]) == actors_with_actions


# ---------------------------------------------------------------------------
# The society, run through the real engine
# ---------------------------------------------------------------------------


def _cited(plan: Any, out: set[str] | None = None) -> set[str]:
    out = set() if out is None else out
    if isinstance(plan, dict):
        for key, value in plan.items():
            if key == "evidence_claim_ids" and isinstance(value, list):
                out |= {str(v) for v in value}
            else:
                _cited(value, out)
    elif isinstance(plan, list):
        for item in plan:
            _cited(item, out)
    return out


def _compiled(plan: dict[str, Any], gateway: Any) -> Any:
    compilation, _mapping = lower_plan(parse_semantic_plan(plan))
    bundle = build_bundle(
        {
            "world_spec": compilation["world_spec"],
            "uncertainties": compilation["uncertainties"],
            "world_facts": compilation["world_facts"],
            "required_reality_facts": compilation["required_reality_facts"],
            "reality": {
                "subject_entity": compilation["subject_entity"],
                "resolution_units": compilation["resolution_units"],
                "target_outcome": compilation["target_outcome"],
                "expected_participants": compilation["expected_participants"],
                "as_of": AS_OF.isoformat(),
                "horizon": HORIZON.isoformat(),
            },
            "claims": [
                dict(CLAIMS[c], id=c, published_at="2026-03-01T00:00:00+00:00")
                for c in sorted(_cited(plan) & KNOWN)
            ],
        }
    )
    return compile_world(
        ResolutionContract(
            question=str(plan["resolution"]["question"]),
            as_of=AS_OF,
            horizon=HORIZON,
            subject_entity=bundle.subject_entity,
            resolution_units=bundle.resolution_units,
            terminal=bundle.spec.terminal,
            target_outcome=bundle.target_outcome,
            expected_participants=bundle.expected_participants,
        ),
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=2,
    )


def test_a_position_stated_by_one_party_reaches_the_others_in_a_real_run() -> None:
    """W5 end to end, through the engine rather than through the compiled artifact.

    Corran presses for a higher limit — the act the record attributes to it — and nobody
    else acts, so anything the others learn, they learned from Corran. The assertion is
    on the runtime's own three transitions, which it records separately: the message
    passed visibility, was DELIVERED, and was NOTICED. Across the whole artifact tree
    that had never once happened between two actors — the six `deliver_information`
    events in it are four environment drops with no author and two with empty text.
    """

    plan = grazing_plan(equipped=True)
    probe = ProgrammableGateway(
        {
            "actor_decision": wait_decision("probe"),
            "reflect": {"beliefs_update": [], "new_memories": []},
        }
    )
    spec = _compiled(plan, probe).spec
    by_name = {e.name: e.entity_id for e in spec.entities}
    corran = by_name["Corran"]
    corran_action = next(
        a.action_id for a in spec.actions if corran in a.eligible_actors and "presses" in a.meaning
    )

    def decide(ctx: dict[str, Any]) -> dict[str, Any]:
        if ctx.get("actor_id") == corran:
            return {
                "plan_disposition": "continue",
                "action_mode": "compiled_action",
                "compiled_action_id": corran_action,
                "params": {},
                "reasoning": "the hill can carry more",
            }
        return wait_decision("waiting to hear what the others say")

    gw = ProgrammableGateway(
        {"actor_decision": decide, "reflect": {"beliefs_update": [], "new_memories": []}}
    )
    result = run(_compiled(plan, gw), gw, seed=0)

    # Corran really acted, through the compiled affordance and not by invention.
    assert any(d.intent.get("action_id") == corran_action for d in result.actor_decisions), (
        "Corran never took the act the record attributes to it"
    )

    # The message Corran's act carried, as the runtime built it.
    messages = {
        e.event_id
        for e in result.event_ledger
        if e.kind == "deliver_information" and e.actor_id == corran
    }
    assert messages, "stating a position produced no message at all"

    noticed_by = {
        d.actor_id for d in result.actor_decisions if messages & set(d.noticed_observation_ids)
    }
    assert noticed_by, "no party in the world could tell that Corran had said anything"
    assert corran not in noticed_by, "nobody is told what they themselves just did"
    assert noticed_by & {by_name["Ardgour"], by_name["Beinn Dubh"]}, (
        "the other townships must be able to hear a position, not only the committee"
    )


# ---------------------------------------------------------------------------
# The compression the earlier gates could not see: parties that are never listed
#
# W4's rules read the parties that are IN the world, and they work — the world above
# is refused by name. What they cannot see is the plan that never lists the parties at
# all, and that is the plan the measured compiler actually writes. The recorded N=10
# OPEC+ bench (`docs/SOCIETY_BENCH.md`) produced seven worlds; every one of them put
# the member countries in `excluded_candidates` or nowhere, so `INERT_PARTICIPANT` had
# nobody to name and every society gate passed a world with one actor. One plan
# declared twenty-three decision-relevant participants, built one entity carrying
# represents_count 23, argued a single exclusion, left six named countries out of the
# plan entirely — and compiled.
#
# The asymmetry those plans were answering is priced, not stylistic. In the measured
# plans an included party costs ~1,430 characters (an entity record answering five
# representation questions, plus an affordance with its changes and citations); an
# excluded one cost 139, could be shared across six names in a single entry, and could
# be argued with the string "Same as Saudi Arabia." The rules below make an omission
# cost what an inclusion costs, and make an aggregate say what it absorbed. None of
# them can be satisfied by inventing a participant, and each is satisfiable by any
# world that is right.
# ---------------------------------------------------------------------------


def _validate_with_record(plan: dict[str, Any]) -> list[str]:
    """Validate with the store's own entity lists in front of the validator."""

    return validate_semantic_plan(
        parse_semantic_plan(plan),
        as_of=AS_OF,
        horizon=HORIZON,
        known_claim_ids=KNOWN,
        claim_entities={cid: tuple(c["entities"]) for cid, c in CLAIMS.items()},
    )


def test_an_aggregate_deciding_for_its_members_must_say_what_it_absorbed() -> None:
    """``represents_count: 3`` is the cheapest sentence in the schema.

    It satisfies the participant count, it satisfies the reviewer's "represents_count
    faithful?" question, and it turns three parties into one integer without the plan
    ever saying who they were. The compression stays legal — a delegation voting as
    instructed is a real thing — but it is now claimed, in the same shape a world with
    no actors at all has to claim itself.
    """

    plan = grazing_plan(equipped=False)
    # Strip the townships so nothing else fires: one aggregate, standing for three
    # parties the plan never mentions again. This is the measured shape.
    plan["entities"] = [e for e in plan["entities"] if e["name"] == "Ard Fell Grazings Committee"]
    plan["processes"][0]["participants"] = ["Ard Fell Grazings Committee"]
    assert _defects(_validate(plan)) == {"AGGREGATE_UNJUSTIFIED"}, _validate(plan)

    plan["aggregate_justifications"] = [
        {
            "aggregate": "Ard Fell Grazings Committee",
            "members": "Ardgour, Beinn Dubh and Corran",
            "members_hold_no_separate_position": "the townships state positions at the "
            "meeting and the committee announces the limit; nothing in the record shows "
            "a township able to withhold or act on the limit itself",
            "evidence_claim_ids": ["c-g2"],
        }
    ]
    assert _validate(plan) == [], "a stated, cited aggregate claim is admitted"


def test_an_aggregate_claim_without_evidence_is_not_a_claim() -> None:
    """The three fields are all load-bearing: who, what shows it, and the citation.

    Without the citation this is the planner asserting its own compression, which is
    the failure `zero_actor_justification` and `single_multiplier_exemption` are both
    shaped to prevent.
    """

    plan = grazing_plan(equipped=False)
    plan["entities"] = [e for e in plan["entities"] if e["name"] == "Ard Fell Grazings Committee"]
    plan["processes"][0]["participants"] = ["Ard Fell Grazings Committee"]
    for broken in (
        {"aggregate": "Ard Fell Grazings Committee", "members": "", "evidence_claim_ids": ["c-g2"]},
        {
            "aggregate": "Ard Fell Grazings Committee",
            "members": "Ardgour, Beinn Dubh and Corran",
            "members_hold_no_separate_position": "they act as one",
            "evidence_claim_ids": [],
        },
        {
            "aggregate": "some other body",
            "members": "Ardgour, Beinn Dubh and Corran",
            "members_hold_no_separate_position": "they act as one",
            "evidence_claim_ids": ["c-g2"],
        },
    ):
        plan["aggregate_justifications"] = [broken]
        assert "AGGREGATE_UNJUSTIFIED" in _defects(_validate(plan)), broken


def test_one_exclusion_entry_removes_one_party() -> None:
    """Six parties left the measured world inside one entry's ``name``.

    ``"Saudi Arabia, Russia, Kuwait, Algeria, Kazakhstan, Oman"`` is one judgement about
    a class, and whether each of those countries holds a position of its own is six
    questions. The rule reads only the unambiguous separators, so a real single name
    like "Trinidad and Tobago" is untouched.
    """

    plan = factor_plan()
    plan["excluded_candidates"] = [
        {
            "name": "Ardgour, Beinn Dubh and Corran",
            "record_attributes": "state their positions at the Michaelmas meeting",
            "why_immaterial": "the factor sets the limit whatever they say",
            "evidence_claim_ids": ["c-g2", "c-g5"],
        }
    ]
    errors = _validate(plan)
    assert _defects(errors) == {"EXCLUSION_UNARGUED"}, errors
    assert "one entry per party" in errors[0]

    plan["excluded_candidates"][0]["name"] = "Trinidad and Tobago"
    assert "EXCLUSION_UNARGUED" not in _defects(_validate(plan)), (
        "a country whose real name contains 'and' is one party, not a list"
    )


def test_an_exclusion_may_not_borrow_another_exclusions_reasoning() -> None:
    """One measured plan excluded six countries with "Same as Saudi Arabia."

    The record attributes different things to different names. An exclusion that points
    at another exclusion has not read what it attributes to this one — and an argument
    that opens with the same word while actually arguing about this party is untouched,
    which is the boundary the factor world's own Beinn Dubh entry sits on.
    """

    plan = factor_plan()
    plan["excluded_candidates"][1]["why_immaterial"] = "Same as Ardgour."
    errors = _validate(plan)
    assert _defects(errors) == {"EXCLUSION_UNARGUED"}, errors
    assert "Beinn Dubh" in errors[0]

    assert _validate(factor_plan()) == [], (
        "the unmodified world argues each exclusion for its own party and is admitted"
    )


def test_a_party_the_grounding_claim_names_cannot_simply_be_absent() -> None:
    """The hole `INERT_PARTICIPANT` cannot reach: the party is never listed.

    The committee's authority to act here is grounded in the claim that says the three
    townships state their positions at the meeting. Reading that sentence as authority
    for one party and as silence about the other three is the compression, and until
    now nothing looked at it — `ExcludedCandidate`'s own docstring says an omission
    nobody had to justify is indistinguishable from an omission nobody noticed.
    """

    plan = factor_plan()
    plan["excluded_candidates"] = []
    # The factor's act now rests on the meeting claim, which names all three townships.
    plan["affordances"][0]["evidence_claim_ids"] = ["c-g2", "c-g5"]

    # Without the record in front of it the validator guesses at nothing.
    assert _validate(plan) == []

    errors = _validate_with_record(plan)
    assert _defects(errors) == {"PARTY_UNACCOUNTED"}, errors
    for township in ("Ardgour", "Beinn Dubh", "Corran"):
        assert any(township in e for e in errors), township
    assert "world_facts" in errors[0], "the message says how a non-party is answered for"


def test_accounting_for_a_named_party_needs_what_the_record_puts_under_its_name() -> None:
    """An exclusion is an argument about something, so it has to name the something.

    ``why_immaterial`` alone can be written without ever reading the claim: the plan
    below excludes all three townships on the very claim it cites as the factor's
    authority, and says nothing about what that claim attributes to them. The
    correction is one sentence, and it is the sentence the exclusion is an argument
    against.
    """

    plan = factor_plan()
    plan["affordances"][0]["evidence_claim_ids"] = ["c-g2", "c-g5"]
    # c-g2 names the committee as well, so it too is accounted for — argued, and with
    # what the record puts under its name — leaving only the three blanks under test.
    plan["excluded_candidates"].append(
        {
            "name": "Ard Fell Grazings Committee",
            "record_attributes": "hears the townships' positions at the Michaelmas "
            "meeting; the record gives it no part in the estate limit",
            "why_immaterial": "the limit under this question is the estate factor's, "
            "set under the estate's own regulations, so the grazings committee has "
            "nothing to move",
            "evidence_claim_ids": ["c-g2", "c-g5"],
        }
    )
    for entry in plan["excluded_candidates"][:3]:
        entry["record_attributes"] = ""
    errors = _validate_with_record(plan)
    assert _defects(errors) == {"EXCLUSION_UNARGUED"}, errors
    assert len(errors) == 3, "one finding per party, never one for the class"

    assert _validate_with_record(factor_plan()) == [], (
        "the single-decider world argues each of its three exclusions and is admitted "
        "with the whole record in front of the validator"
    )


def test_the_equipped_world_is_admitted_with_the_record_in_front_of_it() -> None:
    """The ordering rule: no new gate ships without the mechanism that passes it.

    Every party the grounding claims name is in the world holding the act the record
    attributes to it, so nothing here has anything to report — which is what makes the
    rules above a check on compression rather than a tax on breadth.
    """

    assert _validate_with_record(grazing_plan(equipped=True)) == []
