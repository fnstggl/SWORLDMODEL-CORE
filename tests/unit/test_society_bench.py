"""The society bench measures the compiler; these tests measure the bench.

`scripts/society_bench.py` exists because eleven live compiles were read one at a time
and two prompt changes were shipped and reverted inside the noise. An instrument built
to end that has to be trustworthy itself, and the three things it can get wrong are:

* **metric extraction** — reading `parties_holding_acts` off the wrong field would make
  every number downstream a fiction;
* **aggregation** — pooling a refusal with a compiled world, or dropping a world that
  was built before a provider outage, changes the denominator the headline rate is
  computed over;
* **the noise statement** — the one output that says whether a difference is real. It
  must be conservative, and it must say "not enough runs" in those words when N cannot
  separate anything.

Everything here runs against a scripted gateway or against literal records. No test in
this file makes a provider call, and every statistical claim is checked against a value
computed by hand in the assertion itself.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from _fakes import ProgrammableGateway
from sworldmodel.evidence import EvidenceClaim, EvidenceStore
from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_bench() -> Any:
    spec = importlib.util.spec_from_file_location("society_bench", _SCRIPTS / "society_bench.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered in sys.modules before execution: @dataclass resolves a class's
    # annotations through sys.modules[cls.__module__], and an unregistered module makes
    # every dataclass in the file fail to build.
    sys.modules["society_bench"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


AS_OF = datetime.fromisoformat("2026-03-01T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-06-30T23:59:59+00:00")
MEETING = "2026-05-20T10:00:00+00:00"

QUESTION = (
    "Will the Kelvinbrae Common Ferry Board adopt a reduced winter sailing timetable "
    "before the end of June?"
)

# The record the scripted world is compiled from. Invented, and deliberately nobody's
# acceptance question: what is under test is the harness's reading of a compilation,
# not any subject matter.
CLAIMS: dict[str, tuple[str, list[str]]] = {
    "c-1": (
        "The Kelvinbrae Common Ferry Board sets the winter sailing timetable each season",
        ["Kelvinbrae Common Ferry Board"],
    ),
    "c-2": (
        "Ardnoe and Bruichlee state their positions on a reduced winter timetable at "
        "the spring board meeting",
        ["Ardnoe", "Bruichlee", "Kelvinbrae Common Ferry Board"],
    ),
    "c-3": (
        "Ardnoe has pressed for a reduced winter timetable at successive board meetings",
        ["Ardnoe"],
    ),
}


def _store() -> EvidenceStore:
    store = EvidenceStore()
    for cid, (proposition, entities) in CLAIMS.items():
        store.add(
            EvidenceClaim(
                id=cid,
                proposition=proposition,
                normalized_value=proposition,
                entities=tuple(entities),
                valid_from=AS_OF,
                valid_until=None,
                published_at=AS_OF,
                available_at=AS_OF,
                source_id=f"src_{cid}",
                source_url="https://example.test/doc",
                source_title="fixture source",
                source_type=SourceType.OFFICIAL_INSTITUTIONAL,
                authority_level=AuthorityLevel.AUTHORITATIVE,
                supporting_excerpt=proposition,
                lineage_event_id=f"ev_{cid}",
                epistemic_type=EpistemicType.OBSERVATION,
                confidence=0.9,
                retrieved_at=AS_OF,
            )
        )
    return store


def _village(name: str, claims: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "structural_type": "organization",
        "role": "village whose sailings the winter timetable governs",
        "representation_scale": "organization",
        "represents_count": None,
        "decides": True,
        "authority": "states its position on the winter timetable at the board meeting",
        "why_material": f"{name} is one of the two villages whose agreement the board "
        "needs before it reduces the timetable",
        "terminal_state_it_can_change": "moves its own stated position, which the board "
        "reads before it adopts a reduced timetable",
        "information_received": "the position the other village states at the meeting",
        "if_removed": "one of the two positions the board weighs disappears and the "
        "decision could go the other way",
        "evidence_claim_ids": claims,
    }


def ferry_plan(*, equipped: bool) -> dict[str, Any]:
    """A three-party world, or the same world with the two villages holding nothing.

    ``equipped=False`` is `phase2/geopolitical2`'s exact shape at small scale: every
    party is present, one of them holds every affordance, and the terminal turns on
    that one party's single write. ``equipped=True`` gives the villages grounded acts
    and a channel to each other. The harness must read 1 from the first and 3 from the
    second — if it cannot tell those two worlds apart, no distribution it prints means
    anything.
    """

    villages = ["Ardnoe", "Bruichlee"]
    board: dict[str, Any] = {
        "name": "Kelvinbrae Common Ferry Board",
        "structural_type": "coalition",
        "role": "adopts the winter sailing timetable for the season",
        "representation_scale": "organization",
        "represents_count": None if equipped else 3,
        "decides": True,
        "authority": "adopts the winter timetable once the villages have been heard",
        "why_material": "its adoption is the act the question asks about",
        "terminal_state_it_can_change": "sets whether a reduced winter timetable has been adopted",
        "information_received": "the positions the villages state at the meeting",
        "if_removed": "nothing adopts a timetable and the question cannot resolve YES",
        "evidence_claim_ids": ["c-1"],
    }
    states: list[dict[str, Any]] = [
        {
            "name": "reduced winter timetable adopted",
            "owner": "Kelvinbrae Common Ferry Board",
            "state_type": "boolean",
            "initial": "UNKNOWN",
            "why_material": "the terminal reads it",
            "evidence_claim_ids": ["c-1"],
        }
    ]
    affordances: list[dict[str, Any]] = [
        {
            "name": "adopt the reduced timetable",
            "meaning": "the board adopts the reduced winter sailing timetable",
            "actor": "Kelvinbrae Common Ferry Board",
            "target": "",
            "authority_required": "the board's authority to set the timetable",
            "preconditions": "the villages have stated their positions at the spring "
            "board meeting and neither has withheld agreement",
            "visibility": "public",
            "duration_seconds": 0,
            "changes": [{"op": "set", "target": "reduced winter timetable adopted", "value": True}],
            "evidence_claim_ids": ["c-1"],
        }
    ]
    if equipped:
        for name in villages:
            states.append(
                {
                    "name": f"{name} stated position on reducing the timetable",
                    "owner": name,
                    "state_type": "category",
                    "initial": "UNKNOWN",
                    "why_material": f"what {name} has said is one of the two positions "
                    "the board weighs",
                    "evidence_claim_ids": ["c-2"],
                }
            )
            for stance, verb, claim in (
                ("supports reducing", "presses for a reduced timetable", "c-3"),
                ("withholds agreement", "objects that the crossing cannot bear it", "c-2"),
            ):
                affordances.append(
                    {
                        "name": f"{name} {verb}",
                        "meaning": f"{name} {verb} at the spring board meeting, and the "
                        "other village hears it",
                        "actor": name,
                        "target": "",
                        "authority_required": "its standing as a served village",
                        "preconditions": "",
                        "visibility": "public",
                        "duration_seconds": 0,
                        "changes": [
                            {
                                "op": "set",
                                "target": f"{name} stated position on reducing the timetable",
                                "value": stance,
                            },
                            {
                                "op": "send",
                                "target": f"{name}'s position on the winter timetable",
                                "recipients": [v for v in villages if v != name]
                                + ["Kelvinbrae Common Ferry Board"],
                                "detail": f"{name} {stance}",
                            },
                        ],
                        "evidence_claim_ids": [claim],
                    }
                )

    entities = [board] + [
        _village(n, ["c-2", "c-3"] if n == "Ardnoe" else ["c-2"]) for n in villages
    ]
    if not equipped:
        for entity in entities[1:]:
            entity["decides"] = False
    return {
        "resolution": {
            "question": QUESTION,
            "yes_condition": "The board adopts a reduced winter sailing timetable.",
            "subject_entity": "Kelvinbrae Common Ferry Board",
            "resolution_units": "an adopted winter timetable",
            "target_outcome": "a reduced winter timetable is adopted",
            "expected_participants": 3,
            "evidence_claim_ids": ["c-1"],
        },
        "entities": entities,
        "excluded_candidates": [],
        "states": states,
        "events": [],
        "affordances": affordances,
        "processes": [
            {
                "name": "the spring board meeting",
                "meaning": "the dated meeting at which the villages are heard and the "
                "board settles the winter timetable",
                "kind": "actor_moment",
                "participants": ["Kelvinbrae Common Ferry Board"] + (villages if equipped else []),
                "inputs": [],
                "at": MEETING,
                "deadline": "2026-06-30T23:59:59+00:00",
                "occurrences": [],
                "evidence_claim_ids": ["c-2"],
            }
        ],
        "uncertainties": [],
        "terminal": {
            "form": "state_equals",
            "state": "reduced winter timetable adopted",
            "value": True,
        },
        "terminal_producer_note": "Nothing establishes the adoption at the cutoff; only "
        "the board's own affordance sets it, at the spring meeting, after the villages "
        "have stated their positions.",
        "world_facts": [],
    }


def _spec(**overrides: Any) -> Any:
    base = {
        "label": "scripted",
        "question": QUESTION,
        "store_path": "/nonexistent/store.json",
        "as_of": AS_OF,
        "horizon": HORIZON,
        "gates": False,
        "max_calls": None,
    }
    base.update(overrides)
    return bench.BenchSpec(**base)


def _gateway(plan: dict[str, Any], **extra: Any) -> ProgrammableGateway:
    responses: dict[str, Any] = {
        "semantic_plan": plan,
        "semantic_review": {"verdict": "APPROVE", "reasons": []},
    }
    responses.update(extra)
    return ProgrammableGateway(responses)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def test_the_bench_reads_three_acting_parties_out_of_a_three_party_world() -> None:
    """The positive control. A scripted plan whose two villages hold grounded acts and
    tell each other must be read as three acting parties with real sends — because a
    bench that cannot register a society could never report one appearing."""

    record = bench.run_once(
        _gateway(ferry_plan(equipped=True)), _spec(), _store(), run_index=0, seed_salt=1
    )

    assert record.outcome == "compiled", record.message
    assert record.metrics["parties_holding_acts"] == 3
    assert record.metrics["entities"] == 3
    assert record.metrics["affordances"] == 5
    assert record.metrics["send_effects"] == 4
    assert record.metrics["plan_send_changes"] == 4
    # No provider was reachable from this test: only the two scripted stages ran.
    assert record.calls_by_kind == {"semantic_plan": 1, "semantic_review": 1}


def test_the_scenery_world_is_recorded_as_a_refusal_with_no_world_to_measure() -> None:
    """The defect's shape at small scale — three parties, one of them holding every act
    — no longer reaches the bench as a world at all: W4's INERT_PARTICIPANT rule refuses
    it during plan validation. So the honest record is a refusal with EMPTY metrics, and
    the bench must not manufacture a reading from a plan that was rejected. This is also
    why the report separates 'refused before a world existed' from 'a world was built
    and a gate rejected it': only the second is a society sample."""

    record = bench.run_once(
        _gateway(ferry_plan(equipped=False)), _spec(), _store(), run_index=0, seed_salt=1
    )

    assert record.outcome == "refused_plan"
    assert record.failure == "semantic_plan_invalid"
    assert "INERT_PARTICIPANT" in record.message
    assert record.metrics == {}


def test_the_bench_reads_one_acting_party_out_of_a_single_decider_world() -> None:
    """The world the compiler actually produces on the OPEC+ store, in miniature: one
    party, one affordance, no channel. The headline must read 1 — the value it has held
    across every recorded run — and the send count must read 0. If the harness could not
    return 1 here it could never show 1 moving."""

    plan = ferry_plan(equipped=True)
    plan["entities"] = [plan["entities"][0]]
    plan["resolution"]["expected_participants"] = 1
    plan["states"] = [s for s in plan["states"] if s["owner"] == "Kelvinbrae Common Ferry Board"]
    plan["affordances"] = [
        a for a in plan["affordances"] if a["actor"] == "Kelvinbrae Common Ferry Board"
    ]
    plan["processes"][0]["participants"] = ["Kelvinbrae Common Ferry Board"]
    plan["excluded_candidates"] = [
        {
            "name": name,
            "why_immaterial": "the record shows the board sets the timetable and gives "
            "this village no act bearing on it",
            "evidence_claim_ids": ["c-1"],
        }
        for name in ("Ardnoe", "Bruichlee")
    ]

    record = bench.run_once(_gateway(plan), _spec(), _store(), run_index=0, seed_salt=1)

    assert record.outcome == "compiled", record.message
    assert record.metrics["parties_holding_acts"] == 1
    assert record.metrics["affordances"] == 1
    assert record.metrics["send_effects"] == 0


def test_parties_holding_acts_counts_who_may_act_not_who_was_labelled_an_actor() -> None:
    """`phase2/geopolitical2` declared nine entities, one actor and one action. Counting
    entities marked ``is_actor``, or ``len(actors)``, both answer a different question
    than "who was handed something to do" — and on a world where an action names an
    eligible actor that no `actors` entry mentions, they disagree. The headline follows
    the affordances."""

    compilation = {
        "world_spec": {
            "entities": [{"entity_id": f"e{i}", "is_actor": i == 0} for i in range(9)],
            "actors": [{"entity_id": "e0"}],
            "actions": [
                {"action_id": "a1", "eligible_actors": ["e0", "e3"], "effects": []},
                {"action_id": "a2", "eligible_actors": ["e3"], "effects": []},
                # An environment-driven action hands nobody anything.
                {"action_id": "a3", "eligible_actors": [], "effects": []},
            ],
        }
    }

    metrics = bench.society_metrics(compilation)

    assert metrics["entities"] == 9
    assert metrics["declared_actors"] == 1
    assert metrics["parties_holding_acts"] == 2  # e0 and e3, counted once each
    assert metrics["affordances"] == 3


def test_a_send_is_counted_from_the_lowered_effect_not_from_the_planners_word() -> None:
    """The planner has never once emitted a `send` (`LAUNCH_GAP_AUDIT` M5). Whether it
    starts to must be read off the effect the lowerer actually produced, so a plan that
    says "send" and lowers to nothing cannot be scored as communication."""

    compilation = {
        "world_spec": {
            "entities": [],
            "actions": [
                {
                    "eligible_actors": ["a"],
                    "effects": [
                        {"op": "deliver_information", "to": ["b"]},
                        {"op": "set_field", "field": "x"},
                    ],
                }
            ],
            "channels": [{"channel_id": "c1"}],
        },
        "_semantic": {
            "plan": {
                "entities": [{"decides": True}, {"decides": False}, {"decides": True}],
                "affordances": [
                    {"changes": [{"op": "send"}, {"op": "set"}]},
                    {"changes": [{"op": "set"}]},
                ],
            }
        },
    }

    metrics = bench.society_metrics(compilation)

    assert metrics["send_effects"] == 1
    assert metrics["channels"] == 1
    assert metrics["plan_send_changes"] == 1
    assert metrics["plan_deciders"] == 2
    assert metrics["plan_entities"] == 3


def test_an_unreadable_plan_is_a_refusal_and_carries_its_code() -> None:
    """A refusal is a measurement, not an error: the run is recorded with the pipeline's
    own failure code so the report can show WHICH gate or stage stopped it."""

    record = bench.run_once(
        ProgrammableGateway({"semantic_plan": {"resolution": {}, "terminal": {}}}),
        _spec(),
        _store(),
        run_index=0,
        seed_salt=1,
    )

    assert record.outcome == "refused_plan"
    assert record.failure == "semantic_plan_invalid"
    assert record.metrics == {}


def test_a_provider_outage_is_never_filed_as_a_compiler_refusal() -> None:
    """`GatewayError` subclasses `SWorldModelError`, so the obvious `except` order files
    an unreachable endpoint in the same column as `coverage_incomplete` and quietly
    shrinks the denominator the headline rate is computed over. An outage is ours."""

    record = bench.run_once(
        ProgrammableGateway({}, fail_tasks=frozenset({"semantic_plan"})),
        _spec(),
        _store(),
        run_index=0,
        seed_salt=1,
    )

    assert record.outcome == "harness_error"
    assert record.failure == "gateway_error"


def test_an_exhausted_call_ceiling_is_named_apart_from_an_outage() -> None:
    """They need opposite responses — raise the ceiling versus wait for the endpoint —
    so they must not both read as one anonymous GatewayError."""

    record = bench.run_once(
        _gateway(ferry_plan(equipped=True)),
        _spec(max_calls=0),
        _store(),
        run_index=0,
        seed_salt=1,
    )

    assert record.outcome == "harness_error"
    assert record.failure == "gateway_budget_exhausted"


# ---------------------------------------------------------------------------
# The seed wrapper
# ---------------------------------------------------------------------------


def test_the_seed_wrapper_changes_the_seed_per_run_and_only_the_seed() -> None:
    """The compiler pins its planner seed to the question, so N production runs ask with
    one seed. The bench substitutes a per-run seed at the gateway boundary — and must
    change nothing else about the request, or the arms stop being comparable."""

    inner_a = ProgrammableGateway({"semantic_plan": {"resolution": {}, "terminal": {}}})
    inner_b = ProgrammableGateway({"semantic_plan": {"resolution": {}, "terminal": {}}})
    a = bench.SeedVaryingGateway(inner_a, salt=11)
    b = bench.SeedVaryingGateway(inner_b, salt=22)

    request = bench.GatewayRequest(
        task_kind="semantic_plan", prompt="p", context={}, seed=7, expected_keys=()
    )
    a.generate(request)
    b.generate(request)
    a.generate(request)

    seen_a = [r.seed for r in inner_a.seen]
    seen_b = [r.seed for r in inner_b.seen]
    assert seen_a[0] != 7, "the compiler's question-derived seed must not survive"
    assert seen_a[0] != seen_b[0], "two runs of one configuration must draw differently"
    assert seen_a[0] == seen_a[1], "one salt must reproduce one seed — the bench is replayable"
    assert inner_a.seen[0].prompt == "p"
    assert inner_a.seen[0].task_kind == "semantic_plan"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _record(outcome: str, failure: str = "", **metrics: int) -> Any:
    full = dict.fromkeys(bench.ALL_METRICS, 0)
    full.update(metrics)
    return bench.RunRecord(
        run_index=0,
        seed_salt=0,
        outcome=outcome,
        failure=failure,
        metrics=full if metrics else {},
        llm_calls=9,
        tokens_in=70_000,
        tokens_out=25_000,
        seconds=200.0,
        compiler_sha256="a" * 64,
    )


def test_the_distribution_keeps_every_mode_and_never_hides_behind_a_mean() -> None:
    """A mean of a bimodal split describes neither mode. The aggregate must carry every
    observed value with its count, so a reader sees the two peaks."""

    summary = bench.aggregate(
        _spec(),
        [
            _record("compiled", entities=1),
            _record("compiled", entities=1),
            _record("refused_gates", "coverage_incomplete", entities=11),
            _record("refused_gates", "coverage_incomplete", entities=11),
        ],
    )

    entities = summary["metrics"]["entities"]
    assert entities["counts"] == {"1": 2, "11": 2}
    assert entities["min"] == 1 and entities["max"] == 11
    assert entities["mean"] == 6.0  # the number that describes neither mode
    assert entities["distinct"] == 2


def test_a_gate_refusal_still_counts_as_a_world_that_was_built() -> None:
    """`refused_gates` means the compiler produced a world and a gate then rejected it.
    Dropping those from the distribution would measure only the worlds that passed —
    and on this store almost nothing passes, so the sample would be the empty set."""

    summary = bench.aggregate(
        _spec(),
        [
            _record("refused_gates", "actors_ungrounded", parties_holding_acts=1),
            _record("refused_plan", "semantic_plan_invalid"),
            _record("compiled", parties_holding_acts=1),
        ],
    )

    assert summary["worlds_produced"] == 2
    assert summary["outcomes"] == {
        "compiled": 1,
        "refused_gates": 1,
        "refused_plan": 1,
        "harness_error": 0,
    }
    assert summary["failures"] == {"actors_ungrounded": 1, "semantic_plan_invalid": 1}
    assert summary["headline"]["n"] == 2


def test_a_world_built_before_a_provider_outage_is_still_in_the_distribution() -> None:
    """The outage cost the gate verdict, not the world. Keying membership off the
    outcome LABEL instead of off "did a world come out" would throw away a real
    measurement because of our own infrastructure."""

    summary = bench.aggregate(
        _spec(),
        [
            _record("harness_error", "gateway_error", parties_holding_acts=4),
            _record("compiled", parties_holding_acts=1),
        ],
    )

    assert summary["worlds_produced"] == 2
    assert summary["metrics"]["parties_holding_acts"]["counts"] == {"1": 1, "4": 1}
    assert summary["headline"]["at_or_above_threshold"] == 1


def test_a_compiler_that_changed_mid_bench_is_reported_not_pooled() -> None:
    """Ten agents are editing this branch. Runs made either side of an edit are two
    configurations, and pooling them would hand back an average of two systems."""

    first = _record("compiled", parties_holding_acts=1)
    second = _record("compiled", parties_holding_acts=1)
    second.compiler_sha256 = "b" * 64

    summary = bench.aggregate(_spec(), [first, second])

    assert summary["compiler_sha256"] == ""
    assert summary["compiler_sha256_disagreement"] == ["a" * 64, "b" * 64]
    assert "THIS IS NOT ONE CONFIGURATION" in bench.render_report(summary)


def test_the_cost_of_the_bench_is_reported_because_an_unaffordable_bench_goes_unused() -> None:
    summary = bench.aggregate(_spec(), [_record("compiled", entities=3) for _ in range(4)])

    assert summary["cost"]["llm_calls"] == 36
    assert summary["cost"]["tokens_in"] == 280_000
    assert summary["cost"]["seconds_per_run_median"] == 200.0


# ---------------------------------------------------------------------------
# The exact statistics — every expectation computed by hand in the assertion
# ---------------------------------------------------------------------------


def test_the_zero_of_n_upper_bound_matches_its_closed_form() -> None:
    """For k=0 the Clopper-Pearson upper limit is 1 - alpha**(1/n) exactly. If the
    bisection did not land on it, every "smallest rate ruled out" claim would be wrong.
    At N=10 the number is 0.259 — a quarter."""

    for n in (1, 5, 8, 10, 20):
        assert (
            bench.binomial_upper_bound(0, n, 0.95) == round(1 - 0.05 ** (1 / n), 10)
            or abs(bench.binomial_upper_bound(0, n, 0.95) - (1 - 0.05 ** (1 / n))) < 1e-9
        )
    assert abs(bench.binomial_upper_bound(0, 10, 0.95) - 0.2589) < 1e-3


def test_fisher_matches_the_hypergeometric_sum_computed_by_hand() -> None:
    """0 of 5 against 5 of 5 is the most extreme split five runs can produce, and its
    two-sided p is the two tail tables over C(10,5): 2/252."""

    assert abs(bench.fisher_exact_two_sided(0, 5, 5, 0) - 2 / math.comb(10, 5)) < 1e-12
    # 0 of 5 against 4 of 5: both tables at a=0 and a=4, each C(5,4)/C(10,4).
    assert abs(bench.fisher_exact_two_sided(0, 5, 4, 1) - 10 / math.comb(10, 4)) < 1e-12
    # 0 of 10 against 5 of 10: 2 * C(10,5)/C(20,5).
    assert abs(bench.fisher_exact_two_sided(0, 10, 5, 5) - 504 / math.comb(20, 5)) < 1e-12
    assert bench.fisher_exact_two_sided(3, 7, 3, 7) == 1.0


def test_the_minimum_detectable_split_is_what_the_bench_promises_a_reader() -> None:
    """The sentence a reader acts on: at this N, how many of N must the other arm show?
    At N=5 the answer is 4 (p=0.048); at N=10 it is 5 (p=0.033); at N=3 no split exists
    at all, and the function says so by returning more than N."""

    assert bench.minimum_detectable_count(5, 5) == 4
    assert bench.minimum_detectable_count(10, 10) == 5
    assert bench.minimum_detectable_count(3, 3) > 3


def test_the_permutation_test_refuses_rather_than_approximating() -> None:
    """An exact enumeration or nothing. A p-value from an unnamed shortcut is worse than
    no p-value, so beyond the cap the function returns None and the report says to read
    the distributions instead."""

    assert bench.permutation_p([0, 0, 0, 0], [5, 5, 5, 5]) == 2 / math.comb(8, 4)
    assert bench.permutation_p([1, 1, 1], [1, 1, 1]) == 1.0
    assert bench.permutation_p(list(range(20)), list(range(20)), cap=100) is None


# ---------------------------------------------------------------------------
# The noise statement
# ---------------------------------------------------------------------------


def test_zero_of_n_is_never_reported_as_impossible() -> None:
    """The failure this harness exists to prevent, in its purest form. Ten straight runs
    without a two-party world do not mean the compiler cannot build one — a compiler
    that built one a quarter of the time would show this result about 5% of the time,
    and the statement must say so with the number in it."""

    summary = bench.aggregate(
        _spec(), [_record("refused_gates", "x", parties_holding_acts=1) for _ in range(10)]
    )
    text = " ".join(bench.noise_statement(summary))

    assert summary["headline"]["at_or_above_threshold"] == 0
    assert "does NOT mean impossible" in text
    assert "0.26" in text  # the exact upper bound at N=10
    assert "at least 5 of its 10" in text


def test_an_underpowered_bench_says_so_instead_of_printing_two_numbers() -> None:
    """At N=3 no split whatsoever reaches p<=0.05. A harness that printed a distribution
    here and left the reader to infer a result would be repeating the mistake it was
    built to end."""

    summary = bench.aggregate(
        _spec(), [_record("compiled", parties_holding_acts=1) for _ in range(3)]
    )
    text = " ".join(bench.noise_statement(summary))

    assert "NOT ENOUGH RUNS TO COMPARE ANYTHING" in text
    assert "Raise N before drawing a conclusion" in text


def test_a_bench_that_produced_no_world_says_it_measured_nothing() -> None:
    """Every run refused before lowering: there is no society sample at all, and the
    report must not present an empty distribution as a finding about societies."""

    summary = bench.aggregate(
        _spec(), [_record("refused_plan", "semantic_plan_invalid") for _ in range(5)]
    )
    text = " ".join(bench.noise_statement(summary))

    assert "measured nothing" in text
    assert "not about the societies it compiles" in text


def test_the_noise_statement_warns_that_only_the_headline_is_a_result() -> None:
    """Eleven metrics are printed. At p<=0.05 each, one in twenty looks different by
    chance, and nothing here corrects for it — so the report names the pre-registered
    metric and demotes the rest."""

    summary = bench.aggregate(
        _spec(), [_record("compiled", parties_holding_acts=1) for _ in range(10)]
    )
    text = " ".join(bench.noise_statement(summary))

    assert "pre-registered" in text
    assert "parties_holding_acts" in text
    assert "descriptive" in text


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _arm(label: str, values: list[int], **cfg: Any) -> dict[str, Any]:
    return bench.aggregate(
        _spec(label=label, **cfg),
        [_record("compiled", parties_holding_acts=v) for v in values],
    )


def test_a_real_looking_gap_that_the_n_cannot_support_is_called_not_distinguishable() -> None:
    """0 of 5 against 3 of 5 looks like a breakthrough and is p=0.17. This is the exact
    shape of the eleven-run episode, and the harness must refuse it in words."""

    text = bench.render_comparison(_arm("before", [1, 1, 1, 1, 1]), _arm("after", [1, 1, 3, 3, 3]))

    assert "NOT DISTINGUISHABLE at this N" in text
    assert "would have needed at least 4 of 5" in text


def test_a_gap_the_n_does_support_is_called_distinguishable() -> None:
    text = bench.render_comparison(_arm("before", [1, 1, 1, 1, 1]), _arm("after", [4, 4, 4, 4, 4]))

    assert "DISTINGUISHABLE at p<=0.05" in text


def test_two_arms_that_do_not_share_their_inputs_are_refused_as_an_ab() -> None:
    """The peer's own eleven runs changed the question halfway through. Two arms whose
    question, store, cutoff or gate setting differ cannot attribute a difference to the
    compiler, and the comparison must lead with that rather than with a p-value."""

    text = bench.render_comparison(
        _arm("before", [1, 1, 1, 1, 1]),
        _arm("after", [4, 4, 4, 4, 4], question="a different question entirely"),
    )

    assert "THESE ARMS DO NOT SHARE THEIR INPUTS" in text
    assert "cannot be attributed to the compiler" in text
    assert "question_sha256" in text


def test_identical_compilers_are_named_as_a_noise_measurement() -> None:
    """Two arms at the same compiler hash differ only in seeds. Whatever gap they show
    IS the noise floor, and saying so is how a reader calibrates every later A/B."""

    text = bench.render_comparison(_arm("a", [1, 1, 1]), _arm("b", [1, 3, 1]))

    assert "Compiler sha256 is IDENTICAL in both arms" in text
    assert "IS the noise" in text
