"""The same defects, in different syntax — the class the CWF regressions did not have.

``test_causal_world_fidelity`` proves the Phase-3 gates fire on the worlds they were
written from. That is not the same as proving they fire on the *shape* of those worlds,
and an adversarial review showed it was not true: renaming one intermediate and citing
one of two straddling legs reproduced the exact failure the gates exist to end — a cited
base times an invented deciding factor — and it passed the static validator and all four
mechanical review checks. Every attack in this file is that: a world whose defect is
identical to one the suite already covers, wearing different arithmetic.

Each test names the defect it reproduces and the register entry it closes:

* FD-28 (CW-A) — one alias hop between the draw and the terminal disarmed D4 and every
  CWF-6 check, because they read the terminal's DIRECT writers. They now read the
  terminal's lineage closure, so a rename in the middle buys nothing.
* FD-33 (CW-G) — one actor affordance on the terminal quantity collapsed the review from
  five checks to one. An actor writing the terminal is a reason to scrutinise how it
  decides, not a reason to stop asking.
* FD-25 / FD-31 (CW-E) — the moment count was taken world-wide, so a correctly-scheduled
  process laundered an under-scheduled one, and ``len(moments) > 1`` was the whole test,
  so every occurrence of a "weekly" process at one instant passed as soon as a setup step
  supplied a second timestamp.
* FD-45 — a fault in the mechanical pass was swallowed into zero findings, which reads
  exactly like a world that was examined and approved.

The fixtures are the invented domains from the CWF suite — an aquifer recharge district
and a bay ferry terminal — because the attack is on the shape and the shape has to be the
one already believed safe. Nothing here names a real question, quantity or threshold.
"""

from __future__ import annotations

import copy
from typing import Any

import test_causal_world_fidelity as T

from _fakes import ProgrammableGateway
from sworldmodel.world_review import mechanical_world_checks, review_world

# ---------------------------------------------------------------------------
# Plan surgery. Every helper takes a world the suite already believes in and
# changes only how its arithmetic is spelled.
# ---------------------------------------------------------------------------


def _findings(plan: dict[str, Any]) -> dict[str, Any]:
    return {f.key: f for f in mechanical_world_checks(T._compile(plan))}


def _state(name: str, unit: str, why: str) -> dict[str, Any]:
    return {
        "name": name,
        "owner": "world",
        "state_type": "quantity",
        "unit": unit,
        "initial": "UNKNOWN",
        "why_material": why,
        "evidence_claim_ids": [],
    }


def _read(name: str) -> dict[str, Any]:
    return {"kind": "state", "state": name}


def _relabelled(base: dict[str, Any], hops: tuple[str, ...]) -> dict[str, Any]:
    """Push the tally's arithmetic back through ``hops`` renames.

    The world computes exactly what it computed before. The only difference is that the
    effect writing the terminal no longer mentions the draw — which was the whole of the
    evasion, because every check keyed on what the terminal's own writer reads.
    """

    plan = copy.deepcopy(base)
    process = plan["processes"][0]
    value = copy.deepcopy(process["occurrences"][0]["changes"][0]["value"])
    for index, hop in enumerate(hops):
        plan["states"].append(_state(hop, "crossings", "a stage of the published tally"))
        if index == 0:
            process["occurrences"][0]["changes"] = [{"op": "set", "target": hop, "value": value}]
        else:
            process["occurrences"].append(
                {
                    "description": f"the tally is carried into {hop}",
                    "at": f"2026-02-2{5 + index}T06:00:00+00:00",
                    "changes": [
                        {"op": "set", "target": hop, "value": _read(hops[index - 1])},
                    ],
                }
            )
    process["occurrences"].append(
        {
            "description": "the tally is published from the last stage",
            "at": "2026-02-28T06:00:00+00:00",
            "changes": [
                {
                    "op": "set",
                    "target": "recorded crossings this quarter",
                    "value": _read(hops[-1]),
                }
            ],
        }
    )
    return plan


def _cite_one_leg(plan: dict[str, Any]) -> dict[str, Any]:
    """Cite one of the two straddling alternatives and invent the other."""

    plan["uncertainties"][0]["alternatives"][0]["evidence_claim_ids"] = ["c-f2"]
    return plan


def _cadence(cycles: int, *, at: str | None = None) -> dict[str, Any]:
    """The recharge world's infiltration typed ``cycles`` times, with no uncertainty.

    The straddle here is the recurrence COUNT: eight typed cycles and ten typed cycles
    land on opposite sides of the threshold with no branch weight anywhere in the world.
    ``at`` dates every cycle at one instant, which is the FD-31 shape.
    """

    plan = T.recharge_plan(cited_driver=True)
    plan["uncertainties"] = []
    plan["states"][2] = {
        "name": "winter runoff fraction",
        "owner": "world",
        "state_type": "quantity",
        "unit": "fraction",
        "initial": 0.75,
        "why_material": "the district's standing storm-rule diversion share",
        "evidence_claim_ids": ["c-a2"],
    }
    plan["processes"][0]["occurrences"] = [
        {
            "description": "the season-opening storm sets the weekly diversion rate",
            "at": "2026-01-12T06:00:00+00:00",
            "changes": [
                {
                    "op": "set",
                    "target": "diverted storm runoff",
                    "value": {
                        "kind": "product",
                        "parts": [
                            {"kind": "literal", "value": 100},
                            _read("winter runoff fraction"),
                        ],
                    },
                }
            ],
        }
    ] + [
        {
            "description": f"weekly infiltration cycle {i + 1}",
            "at": at or f"2026-01-{19 + i:02d}T06:00:00+00:00",
            "changes": [
                {
                    "op": "increase",
                    "target": "recorded recharge volume",
                    "amount": _read("diverted storm runoff"),
                }
            ],
        }
        for i in range(cycles)
    ]
    return plan


def _actor_publishes_the_tally() -> dict[str, Any]:
    """The Tesla shape with one signature on it: an actor is the terminal's only writer.

    The mechanism computes the projection and stops; a manager whose single affordance
    copies that projection into the recorded tally is the only thing that ever writes the
    answer. Placing one actor on the terminal quantity used to be a universal skeleton
    key, because the review surrendered every check but one the moment it saw it.
    """

    plan = T.ferry_plan(factors=(1.1, 1.4))
    plan["processes"][0]["occurrences"][0]["changes"] = [
        {
            "op": "set",
            "target": "projected crossings",
            "value": {
                "kind": "product",
                "parts": [_read("last season crossings"), _read("seasonal crossing factor")],
            },
        }
    ]
    plan["states"].append(
        _state("projected crossings", "crossings", "the projection the manager publishes")
    )
    plan["entities"].append(
        {
            "name": "Terminal operations manager",
            "structural_type": "person",
            "role": "publishes the quarterly crossing tally",
            "representation_scale": "individual",
            "decides": True,
            "authority": "publishes the quarter's recorded crossings",
            "why_material": "no figure is recorded until the manager publishes it",
            "terminal_state_it_can_change": "recorded crossings this quarter",
            "information_received": "the projection from the tally system",
            "if_removed": "no tally is ever recorded",
            # c-f3 names the manager; c-f1 records the crossings the manager reports on.
            # Citing only c-f1 made the actor attest itself from the figure it decides.
            "evidence_claim_ids": ["c-f3", "c-f1"],
        }
    )
    plan["affordances"] = [
        {
            "name": "publish the quarterly tally",
            "meaning": "publish the projection as the quarter's recorded crossings",
            "actor": "Terminal operations manager",
            "authority_required": "tally authority",
            "visibility": "public",
            "changes": [
                {
                    "op": "set",
                    "target": "recorded crossings this quarter",
                    "value": _read("projected crossings"),
                }
            ],
            "evidence_claim_ids": ["c-f3", "c-f1"],
        }
    ]
    plan["processes"].append(
        {
            "name": "quarter-end publication",
            "meaning": "the dated moment the manager publishes the tally",
            "kind": "actor_moment",
            "participants": ["Terminal operations manager"],
            "allowed_affordances": ["publish the quarterly tally"],
            "at": "2026-02-27T09:00:00+00:00",
            "occurrences": [],
            "evidence_claim_ids": ["c-f3"],
        }
    )
    plan.pop("zero_actor_justification", None)
    plan["resolution"]["expected_participants"] = 1
    return plan


# ---------------------------------------------------------------------------
# FD-28 / CW-A — one alias hop between the draw and the terminal
# ---------------------------------------------------------------------------


def test_one_rename_between_the_draw_and_the_terminal_does_not_disarm_the_review() -> None:
    """The reported evasion, exactly: the terminal's own writer never mentions the draw.

    ``recorded := projection`` and ``projection := base x factor`` computes precisely what
    ``recorded := base x factor`` computed, and the review used to derive the right fact
    and throw it away — ``_flipping_uncertainties`` said the draw flips the terminal and
    ``single_uncertain_multiplier_decides`` reported PASS beside it, because the draw is
    read into the alias rather than into a terminal term.
    """

    plan = _cite_one_leg(_relabelled(T.ferry_plan(factors=(1.1, 1.4)), ("projected crossings",)))
    findings = _findings(plan)

    assert findings["terminal_set_in_one_step"].severity == "CRITICAL"
    assert "projected_crossings" in findings["terminal_set_in_one_step"].finding
    # The rename is not a production stage: nothing does anything with it but copy it.
    assert findings["no_intermediate_production_state"].severity == "HIGH"
    # And the draw is seen, through the alias, as deciding the answer.
    assert findings["single_uncertain_multiplier_decides"].severity != "PASS"
    assert "seasonal_crossing_factor" in findings["single_uncertain_multiplier_decides"].finding


def test_a_longer_chain_of_renames_buys_no_more_than_a_short_one() -> None:
    """Three hops instead of one. The closure is transitive; the evasion is not deeper."""

    plan = _relabelled(
        T.ferry_plan(factors=(1.1, 1.4)),
        ("projected crossings", "reviewed crossings", "published crossings"),
    )
    findings = _findings(plan)

    assert findings["terminal_set_in_one_step"].severity == "CRITICAL"
    for hop in ("projected_crossings", "reviewed_crossings", "published_crossings"):
        assert hop in findings["terminal_set_in_one_step"].finding
    assert findings["no_intermediate_production_state"].severity == "HIGH"


def test_multiplying_the_relabelling_by_one_is_still_a_relabelling() -> None:
    """``recorded := projection x 1`` is arithmetic that does nothing, spelled as a stage.

    A test on the literal syntax of an assignment would let this through. The chain asks
    what the world PRODUCES, so a factor of one produces nothing and the answer is the
    same: the terminal is announced.
    """

    plan = _relabelled(T.ferry_plan(factors=(1.1, 1.4)), ("projected crossings",))
    plan["processes"][0]["occurrences"][-1]["changes"][0]["value"] = {
        "kind": "product",
        "parts": [{"kind": "literal", "value": 1}, _read("projected crossings")],
    }
    findings = _findings(plan)

    assert findings["terminal_set_in_one_step"].severity == "CRITICAL"
    assert findings["no_intermediate_production_state"].severity == "HIGH"


def test_an_intermediate_nothing_reads_is_not_a_production_state() -> None:
    """A field the mechanisms write and the answer never depends on proves nothing.

    ``no_intermediate_production_state`` used to pass on any mechanism-written field that
    was not a terminal term, with no requirement that anything read it — so a world could
    buy the check by writing a number nobody consumes.
    """

    plan = copy.deepcopy(T.ferry_plan(factors=(1.1, 1.4)))
    plan["states"].append(_state("harbour weather index", "index", "a logged sea-state index"))
    plan["processes"][0]["occurrences"].append(
        {
            "description": "the harbour logs its weather index",
            "at": "2026-02-10T06:00:00+00:00",
            "changes": [
                {
                    "op": "set",
                    "target": "harbour weather index",
                    "value": {"kind": "literal", "value": 3},
                }
            ],
        }
    )
    findings = _findings(plan)

    assert findings["no_intermediate_production_state"].severity == "HIGH"
    assert "harbour_weather_index" not in findings["no_intermediate_production_state"].finding


def test_the_shipped_recharge_world_is_told_that_one_draw_decides_it() -> None:
    """The reclassification this fix was expected to force, on a fixture believed good.

    The recharge world's answer is decided entirely by ``winter_runoff_fraction``, the
    review's own ``_flipping_uncertainties`` has always said so, and the check beside it
    reported PASS because the draw is read into the intermediate rather than into the
    terminal. It is cited, so this is a concern and not a blocker — the observation
    changes, the verdict does not, and nothing else about the world does either.
    """

    findings = _findings(T.recharge_plan())

    assert findings["single_uncertain_multiplier_decides"].severity == "MEDIUM"
    assert "winter_runoff_fraction" in findings["single_uncertain_multiplier_decides"].finding
    assert findings["multiplier_lacks_evidence"].severity == "PASS"
    assert [k for k, f in findings.items() if f.is_blocking] == []


def test_an_uncited_driver_behind_an_intermediate_is_a_blocker() -> None:
    """Same world, same shape, no citation on the driver — and now it blocks.

    Proof that the reclassification above is the citation doing the work and not the
    lineage: with nothing supporting the deciding factor, an invented number is doing the
    job the world is supposed to do, and the review says so through the intermediate.
    """

    findings = _findings(T.recharge_plan(cited_driver=False))

    assert findings["single_uncertain_multiplier_decides"].severity == "HIGH"
    assert findings["multiplier_lacks_evidence"].severity == "CRITICAL"
    assert "winter_runoff_fraction" in findings["multiplier_lacks_evidence"].finding


# ---------------------------------------------------------------------------
# FD-33 / CW-G — one actor affordance on the terminal quantity
# ---------------------------------------------------------------------------


def test_an_actor_on_the_terminal_quantity_no_longer_buys_silence() -> None:
    """The skeleton key: one affordance writing the terminal used to end the review.

    Five checks became one unconditional PASS, and the four surrendered were the ones
    that catch what is underneath — here, the environment computing the whole answer and
    an actor whose only affordance copies it into the tally.
    """

    findings = _findings(_actor_publishes_the_tally())

    assert len(findings) > 1
    assert findings["terminal_set_in_one_step"].severity == "CRITICAL"
    assert findings["no_intermediate_production_state"].severity == "HIGH"
    # The probe follows a write only one actor can make, so the draw is visible again.
    assert findings["single_uncertain_multiplier_decides"].severity == "HIGH"
    assert findings["multiplier_lacks_evidence"].severity == "CRITICAL"


def test_an_actor_who_really_decides_still_owes_no_operational_depth() -> None:
    """The other half of FD-33: scrutiny returns, refusals do not follow it.

    The recharge district's engineer can release an emergency allocation or hold it. That
    is a real decision on the terminal quantity, so the review now asks all its questions
    of this world instead of one — and the world answers every one of them.
    """

    findings = _findings(T.recharge_plan(with_actor=True))

    assert len(findings) > 1
    assert [k for k, f in findings.items() if f.is_blocking] == []
    assert findings["terminal_set_in_one_step"].severity == "PASS"
    assert findings["no_intermediate_production_state"].severity == "PASS"


# ---------------------------------------------------------------------------
# FD-25 / FD-31 / CW-E — a world that does not span the period it claims
# ---------------------------------------------------------------------------


def test_every_occurrence_of_a_weekly_process_at_one_instant_is_refused() -> None:
    """Ten 'weekly' cycles all dated 2026-02-20, and a setup step supplying moment two.

    ``len(moments) > 1`` was the whole test, so one correctly-dated setup occurrence was
    enough to pass a process that claims a cadence it never spans.
    """

    findings = _findings(_cadence(10, at="2026-02-20T06:00:00+00:00"))

    assert findings["world_skips_the_causal_period"].severity == "HIGH"
    assert "10 times" in findings["world_skips_the_causal_period"].finding


def test_repetition_spread_over_two_instants_is_the_same_defect() -> None:
    """Different syntax, same claim: ten cycles, two dates. Repeating is not spreading."""

    plan = _cadence(10)
    for index, occurrence in enumerate(plan["processes"][0]["occurrences"][1:]):
        occurrence["at"] = "2026-02-20T06:00:00+00:00" if index < 5 else "2026-02-21T06:00:00+00:00"
    findings = _findings(plan)

    assert findings["world_skips_the_causal_period"].severity == "HIGH"
    assert "2 distinct moment" in findings["world_skips_the_causal_period"].finding


def test_a_well_scheduled_process_cannot_launder_an_under_scheduled_one() -> None:
    """FD-25. The moments counted are the moments the ANSWER is made at, not any moments.

    A live review reported "the world acts at 10 distinct moments" beside a compiled world
    whose occurrences sat at two dates per process, because the count was world-wide. Here
    the whole tally is computed at a single instant while a second process, which the
    terminal never reads, is dutifully spread across the window.
    """

    plan = copy.deepcopy(T.ferry_plan(factors=(1.1, 1.4)))
    plan["states"].append(_state("harbour weather index", "index", "a logged sea-state index"))
    plan["processes"].append(
        {
            "name": "harbour weather log",
            "meaning": "the harbour logs its sea state through the quarter",
            "kind": "operational",
            "inputs": [],
            "occurrences": [
                {
                    "description": f"sea state logged in week {week}",
                    "at": f"2026-02-{10 + week:02d}T06:00:00+00:00",
                    "changes": [
                        {
                            "op": "set",
                            "target": "harbour weather index",
                            "value": {"kind": "literal", "value": week},
                        }
                    ],
                }
                for week in range(1, 6)
            ],
            "evidence_claim_ids": ["c-f1"],
        }
    )
    findings = _findings(plan)

    assert findings["world_skips_the_causal_period"].severity == "HIGH"
    assert "single moment" in findings["world_skips_the_causal_period"].finding


def test_an_undeclared_cadence_reports_that_it_could_not_be_checked() -> None:
    """The recurrence count decides the answer and nothing verifies the count.

    Eight typed cycles total 840 acre-feet and ten total 990, either side of a 900
    threshold, with no uncertainty anywhere in the world — so how often the process really
    happens is asserted by the number of occurrences somebody typed. Until a recurrence
    declaration reaches the compiled process there is nothing to check that against, and
    this must say so: a gate that cannot run must never read as a gate that ran and
    approved.
    """

    for cycles in (8, 10):
        finding = _findings(_cadence(cycles))["recurrence_is_declared"]
        assert finding.severity != "PASS"
        assert "could NOT run" in finding.finding
        assert f"x{cycles}" in finding.finding


def test_a_world_whose_processes_do_not_repeat_has_no_cadence_to_declare() -> None:
    """The recharge world's two occurrences are two stages, not two turns of a cycle.

    They write different fields, so nothing here claims a repetition, and the cadence
    check reports a clean pass rather than a concern about every world in the system.
    """

    findings = _findings(T.recharge_plan())

    assert findings["recurrence_is_declared"].severity == "PASS"
    assert findings["world_skips_the_causal_period"].severity == "PASS"


class _Occurrence:
    """A dated occurrence, enough of one for the cadence check to read."""

    def __init__(self, at: str) -> None:
        self.at = at


class _DeclaredProcess:
    """An external process carrying the recurrence declaration the semantics layer emits.

    Built by hand because :class:`sworldmodel.worldspec.ExternalProcess` does not carry
    the four ``recurrence_*`` fields yet, so ``parse_world_spec`` drops them and no
    compiled world can exercise this path. These two tests are what stops the bridge from
    being wishful: they prove the check does real work the moment the fields arrive,
    rather than sitting inert and green forever.
    """

    def __init__(self, period: str, start: str, end: str, firings: int) -> None:
        self.process_id = "weekly infiltration"
        self.recurrence_period = period
        self.recurrence_start = start
        self.recurrence_end = end
        self.recurrence_firings = firings


def test_a_declared_cadence_its_occurrences_obey_is_a_pass() -> None:
    """Four weekly firings declared over three weeks, and four weekly firings enumerated."""

    from sworldmodel.world_review import _cadence_finding, _declared_recurrence

    process = _DeclaredProcess("P1W", "2026-01-05T06:00:00+00:00", "2026-01-26T06:00:00+00:00", 4)
    occurrences = [_Occurrence(f"2026-01-{5 + 7 * i:02d}T06:00:00+00:00") for i in range(4)]
    finding = _cadence_finding(
        [(process.process_id, "adjust_field volume", occurrences, _declared_recurrence(process))]
    )

    assert finding.severity == "PASS"


def test_the_declared_count_is_re_derived_and_not_taken_from_the_process() -> None:
    """A process that reports its own firing count is grading its own homework.

    The window holds four weekly firings. The world enumerates ten and reports ten, so a
    check that compared the enumeration against the reported count would agree with
    itself and pass. Stepping the declared period across the declared window is the only
    reading that is not the world's own claim about itself.
    """

    from sworldmodel.world_review import _cadence_finding, _declared_recurrence

    process = _DeclaredProcess("P1W", "2026-01-05T06:00:00+00:00", "2026-01-26T06:00:00+00:00", 10)
    occurrences = [_Occurrence(f"2026-01-{5 + i:02d}T06:00:00+00:00") for i in range(10)]
    finding = _cadence_finding(
        [(process.process_id, "adjust_field volume", occurrences, _declared_recurrence(process))]
    )

    assert finding.severity == "HIGH"
    assert "yields 4" in finding.finding


# ---------------------------------------------------------------------------
# FD-45 — a mechanical pass that cannot complete
# ---------------------------------------------------------------------------


def test_a_faulting_mechanical_pass_blocks_instead_of_vanishing(
    monkeypatch: Any,
) -> None:
    """Zero findings and ``blocking = []`` is what an approved world looks like.

    The mechanical findings are facts computed from the compiled world with no provider
    involved, so a fault computing them is a fact about our own code — and it was
    swallowed, which published worlds that owed five blocking findings and produced none.
    """

    import sworldmodel.world_review as world_review

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the check pass fell over")

    monkeypatch.setattr(world_review, "mechanical_world_checks", boom)
    plan = T.ferry_plan()
    bundle = T._bundle(plan)
    review = review_world(
        T._compile(plan),
        bundle.evidence_store.view(T.AS_OF),
        ProgrammableGateway(fail_tasks=frozenset({"world_review"})),
        question=str(plan["resolution"]["question"]),
        evidence_render="",
    )

    assert review.failed_blocking == ("mechanical_checks_could_not_run",)
    assert review.should_repair
    assert "could NOT run" in review.findings[0].finding
    assert "RuntimeError" in review.findings[0].finding
