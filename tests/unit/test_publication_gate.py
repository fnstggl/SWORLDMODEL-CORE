"""The pre-rollout world review is a GATE, not a note in the margin (FD-27, CWF-6).

The live Tesla recompile is the reproduction. Its ``world_review.json`` records seven
blocking failures — six from the adversarial reviewer, plus the mechanical
``no_intermediate_production_state``, which fired correctly — beside the disposition
"recompiled; the world simulated is not the world reviewed here". The run then simulated
that unreviewed world and published ``simulation_probability: 0.0`` with status
``resolved`` and source ``weighted_simulated_trajectories``.

Two separate defects produced that artifact, and both are pinned here:

1. The review ran ONCE, against the world as it stood before repair. The recompiled
   world — the one that was actually simulated and published — was never reviewed, so
   the record described a world nobody ran. Its own disposition string said so.
2. Nothing downstream ever consulted the findings. ``should_repair`` was read exactly
   once, to decide whether to *attempt* a repair; no blocking finding could stop
   anything after that.

Everything below runs the real pipeline through ``run_forecast`` on a scripted gateway.
No network, no live provider, no case-specific code: the same rule applies to any
question whose world an adversarial review calls materially wrong.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import sworldmodel.api as api
from _fakes import FixtureResearchBackend, ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import AS_OF, HORIZON, scheduled_multiparty_world
from sworldmodel.api import ReviewRound, WorldReviewRecord, run_forecast
from sworldmodel.config import ForecastConfig
from sworldmodel.diagnosis import ForecastRefused, RunDiagnosis
from sworldmodel.world_review import _QUESTIONS, AuditFinding, WorldReview

AS_OF_DT = datetime.fromisoformat(AS_OF)
HORIZON_DT = datetime.fromisoformat(HORIZON)
QUESTION = "will both members record hold?"

# The disposition FD-27 published. It is an admission that the artifact describes a
# world nobody simulated, and it must never appear beside a standing blocking finding.
FD27_STRING = "the world simulated is not the world reviewed here"


# --------------------------------------------------------------------------- #
# Scripting
# --------------------------------------------------------------------------- #


def _blocking(*keys: str) -> dict[str, Any]:
    """A review answer in which every named question comes back blocking.

    Each carries an evidence basis, because ``_parse_findings`` demotes an unsupported
    CRITICAL/HIGH to LOW by rule — an attack that cites nothing must not block a world
    the mechanical gates passed, and a test that forgot the basis would be testing the
    demotion instead of the gate.
    """

    return {
        "findings": [
            {
                "key": key,
                "severity": "HIGH",
                "finding": f"the world fails {key}",
                "evidence_basis": "the compiled world as summarized above",
            }
            for key in keys
        ]
    }


def _clean(*keys: str) -> dict[str, Any]:
    return {
        "findings": [
            {
                "key": key,
                "severity": "PASS",
                "finding": f"the world survives {key}",
                "evidence_basis": "the compiled world as summarized above",
            }
            for key in keys
        ]
    }


def _decide(ctx: dict[str, Any]) -> dict[str, Any]:
    if ctx.get("stage") == "session":
        return act("record_position", {"position": "hold"})
    return wait_decision("waiting for the session")


def _gateway(
    reviews: list[dict[str, Any]] | None = None, *, fail: frozenset[str] = frozenset()
) -> ProgrammableGateway:
    """A gateway whose world_review answers a scripted sequence, one per round.

    The last scripted answer repeats, so a test states only the rounds it cares about.
    """

    scripted = list(reviews or [])

    def review(_ctx: dict[str, Any]) -> dict[str, Any]:
        if not scripted:
            return {"findings": []}
        return scripted.pop(0) if len(scripted) > 1 else scripted[0]

    return ProgrammableGateway(
        {
            "actor_decision": _decide,
            "reflect": {"beliefs_update": [], "new_memories": []},
            "world_review": review,
        },
        fail_tasks=fail,
    )


def _config(gw: ProgrammableGateway, world: dict[str, Any] | None = None) -> ForecastConfig:
    bundle = build_bundle(world or scheduled_multiparty_world(members=2, threshold=2))
    return ForecastConfig(gateway=gw, research_backend=FixtureResearchBackend(bundle))


def _repaired_world() -> dict[str, Any]:
    """A materially different world compiled from the same evidence store.

    Different enough that its fingerprint differs, so a test can tell which world a
    recorded review is a review OF.
    """

    world = scheduled_multiparty_world(members=2, threshold=2)
    world["world_spec"]["title"] = "the world repair came back with"
    return world


def _script_repair(monkeypatch: pytest.MonkeyPatch, gw: ProgrammableGateway) -> None:
    """Make ``_recompile`` produce ``_repaired_world`` instead of calling a provider."""

    gw.is_live = True  # type: ignore[attr-defined]
    monkeypatch.setattr(
        api,
        "compile_for_mode",
        lambda *a, **k: dict(_repaired_world()),
    )


# --------------------------------------------------------------------------- #
# 1. A world whose review keeps blocking findings does not publish a forecast.
# --------------------------------------------------------------------------- #


def test_surviving_blocking_findings_refuse_instead_of_publishing() -> None:
    """FD-27's headline. Seven blocking failures ended in a published 0.0; a blocking
    review that ends in publication is not a gate."""

    gw = _gateway(
        [
            _blocking(
                "what_process_produces_outcome",
                "process_is_represented",
                "material_components_missing",
                "decorative_actors",
                "material_actor_compressed_away",
                "expert_would_call_incomplete",
            )
        ]
    )
    with pytest.raises(ForecastRefused) as caught:
        run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    refusal = caught.value
    assert refusal.stage == "world review"
    cause = refusal.__cause__
    assert cause is not None
    details = cause.details  # type: ignore[attr-defined]
    assert details["failure"] == "world_review_blocking_findings_survived"
    # The diagnosis names the surviving findings, not merely their number.
    assert details["surviving_blocking_findings"] == [
        "decorative_actors",
        "expert_would_call_incomplete",
        "material_actor_compressed_away",
        "material_components_missing",
        "process_is_represented",
        "what_process_produces_outcome",
    ]
    for key in details["surviving_blocking_findings"]:
        assert key in str(cause), f"{key} is missing from the refusal a reader will read"
    # No rollout was paid for: the actors were never invoked.
    assert not [r for r in gw.seen if r.task_kind == "actor_decision"]


def test_the_refusal_is_recompilable_only_while_repair_still_has_a_chance() -> None:
    """A refusal earned before repair ever ran can still be fixed by a live rerun; one
    earned after repair ran and failed is a reroll, and says so."""

    gw = _gateway([_blocking("decorative_actors")])
    with pytest.raises(ForecastRefused) as caught:
        run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))
    # Non-live gateway: the recompile the review asked for could not be produced at all,
    # so repair never got its attempt.
    assert caught.value.__cause__.details["recompilable"] is True  # type: ignore[union-attr]
    assert caught.value.__cause__.details["review_rounds"] == 1  # type: ignore[union-attr]


def test_repair_that_discharges_nothing_stops_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounded honestly: a recompile that leaves the same findings standing is not tried
    again, and 'recompiled' does not discharge anything on its own."""

    gw = _gateway([_blocking("decorative_actors")])
    _script_repair(monkeypatch, gw)
    with pytest.raises(ForecastRefused) as caught:
        run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    details = caught.value.__cause__.details  # type: ignore[union-attr]
    assert details["review_rounds"] == 2, "one repair attempt, then the no-progress stop"
    assert details["recompilable"] is False, "repair ran and the finding survived it"
    record = caught.value.world_review  # type: ignore[attr-defined]
    assert "discharged none of what was raised" in record.disposition


# --------------------------------------------------------------------------- #
# 2. The recorded review describes the world that was (or would have been) simulated.
# --------------------------------------------------------------------------- #


def test_the_recorded_review_describes_the_world_carried_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The review persisted in the artifact must judge the world the run carried
    forward — never one thrown away by a recompile."""

    gw = _gateway([_blocking("decorative_actors"), _blocking("material_components_missing")])
    _script_repair(monkeypatch, gw)
    with pytest.raises(ForecastRefused) as caught:
        run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    record = caught.value.world_review  # type: ignore[attr-defined]
    payload = record.as_dict()
    assert payload["describes_simulated_world"] is True
    assert len(payload["rounds"]) == 2

    # The two rounds judged two DIFFERENT worlds, and the final one is the world the run
    # carried forward — the repaired one, whose fingerprint the record states.
    first, final = payload["rounds"]
    assert first["world_signature"] != final["world_signature"]
    assert final["world_signature"] == payload["simulated_world_signature"]
    repaired_bundle = build_bundle(_repaired_world())
    assert final["world_signature"] == api._world_signature(_compile(repaired_bundle, gw)), (
        "the final round must fingerprint the world repair actually produced"
    )

    # And the surviving findings are the FINAL world's, not the discarded one's.
    assert payload["blocking_failures"] == ["material_components_missing"]
    assert [f["key"] for f in payload["surviving_blocking_findings"]] == [
        "material_components_missing"
    ]
    assert payload["causal_simulation_valid"] is False


def test_no_disposition_claims_the_simulated_world_was_never_reviewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FD-27's own admission string must not survive anywhere in the record, and the
    repaired round must not claim the recompile discharged what it raised."""

    gw = _gateway([_blocking("decorative_actors"), _blocking("material_components_missing")])
    _script_repair(monkeypatch, gw)
    with pytest.raises(ForecastRefused) as caught:
        run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    payload = caught.value.world_review.as_dict()  # type: ignore[attr-defined]
    assert FD27_STRING not in str(payload)
    first, final = payload["rounds"]
    assert "sent back to be recompiled" in first["disposition"]
    assert "Nothing here is discharged by the recompile itself" in first["disposition"]
    assert first["blocking_failures"] == ["decorative_actors"], (
        "the repaired round keeps its own findings; repair does not erase the record"
    )
    assert "REFUSED" in final["disposition"]
    assert "material_components_missing" in final["disposition"]


# --------------------------------------------------------------------------- #
# 3. Findings the repair really does resolve publish normally, with both rounds kept.
# --------------------------------------------------------------------------- #


def test_findings_resolved_by_repair_publish_and_keep_both_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate refuses worlds an adversarial review still calls wrong — not worlds the
    repair fixed. A repaired world publishes a normal forecast, and the record shows
    what was raised and what the repair answered."""

    gw = _gateway(
        [
            _blocking("decorative_actors", "material_components_missing"),
            _clean("decorative_actors", "material_components_missing"),
        ]
    )
    _script_repair(monkeypatch, gw)
    result, ctx = run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    assert result.simulation_probability == 1.0
    assert result.status.value == "resolved"
    assert result.probability_source == "weighted_simulated_trajectories"

    record = ctx.world_review
    payload = record.as_dict()
    assert len(payload["rounds"]) == 2
    assert payload["rounds"][0]["blocking_failures"] == [
        "decorative_actors",
        "material_components_missing",
    ]
    assert payload["blocking_failures"] == []
    assert payload["surviving_blocking_findings"] == []
    assert payload["causal_simulation_valid"] is True
    assert payload["simulated_world_signature"] == api._world_signature(ctx.compiled)
    assert FD27_STRING not in str(payload)
    # The published result carries no review caveat, because the review completed.
    assert not [line for line in result.limitations if "world review" in line]


def test_a_world_that_passes_its_first_review_is_reviewed_once() -> None:
    """The gate costs nothing extra on a world nobody objects to."""

    gw = _gateway([_clean("decorative_actors")])
    result, ctx = run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    assert result.simulation_probability == 1.0
    assert len(ctx.world_review.rounds) == 1
    assert len([r for r in gw.seen if r.task_kind == "world_review"]) == 1
    assert ctx.world_review.causal_simulation_valid is True
    assert ctx.world_review.world_review_blocking == ()


# --------------------------------------------------------------------------- #
# 4. The two halves of a review are priced differently (FD-34).
#
# An OPINION that could not be obtained decides nothing — a provider outage is not a
# verdict about a world, and that behavior is unchanged. A MECHANICAL finding is a fact
# computed from the compiled world with no provider involved, so a provider outage must
# not launder it into an advisory note.
# --------------------------------------------------------------------------- #


def test_a_missing_opinion_over_a_mechanically_clean_world_never_blocks() -> None:
    """The reviewer was unreachable and the world's own checks found nothing. The run
    publishes exactly as it did before — and is never described as reviewed."""

    gw = _gateway(fail=frozenset({"world_review"}))
    result, ctx = run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    assert result.simulation_probability == 1.0
    assert result.status.value == "resolved"
    record = ctx.world_review
    assert record.error.startswith("the review could not run")
    assert record.model_opinion_obtained is False
    assert record.surviving_mechanical_blocking == ()
    assert record.blocks_publication is False
    # Not "valid" either: nobody attacked it. The publication gate is handed None, which
    # it must price as unassessed rather than as a review that passed — the fourth state
    # ("no opinion, mechanically clean") must never read as the second ("ran and passed").
    assert record.world_review_blocking is None
    assert record.causal_simulation_valid is False
    assert "no adversarial review" in " ".join(result.limitations)
    assert "NOT validated" in " ".join(result.limitations)
    assert "could not run" in record.disposition


def test_a_failed_review_with_nothing_blocking_asks_for_no_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair answers findings. When the model half failed and the mechanical half found
    nothing, there is nothing to answer — no recompile, no budget spent on an outage."""

    gw = _gateway(fail=frozenset({"world_review"}))
    recompiles: list[str] = []
    _script_repair(monkeypatch, gw)
    monkeypatch.setattr(
        api,
        "compile_for_mode",
        lambda *a, **k: (recompiles.append("called"), dict(_repaired_world()))[1],
    )
    _, ctx = run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))

    assert len(ctx.world_review.rounds) == 1
    assert not recompiles, "nothing was raised, so there was nothing to repair"
    assert "no opinion about this world was obtained" in ctx.world_review.disposition


def _errored_review_with(finding: AuditFinding) -> WorldReviewRecord:
    """A record whose model call failed while its mechanical half found something."""

    review = WorldReview(
        answers=((finding.key, False, finding.finding),),
        failed_blocking=(finding.key,),
        error="the review could not run: GatewayError: provider down",
        findings=(finding,),
    )
    return WorldReviewRecord((ReviewRound(0, "abc123", review),))


def test_an_unreachable_reviewer_cannot_launder_a_computed_blocker() -> None:
    """FD-34, the bypass this closes: with the review call failing, a CRITICAL the world
    computed about ITSELF used to become an advisory note and the run published.

    ``world_review.py``'s own comment already said the mechanical findings survive a
    failed model call. They now survive into the GATE, not merely into the record: a
    provider outage is not evidence about a world and must not launder a fact about one.
    """

    record = _errored_review_with(
        AuditFinding(
            key="terminal_set_in_one_step",
            severity="CRITICAL",
            finding="the terminal quantity is written once and never built",
            evidence_basis="computed from the compiled world",
        )
    )

    assert record.model_opinion_obtained is False
    assert len(record.surviving_mechanical_blocking) == 1
    assert record.blocks_publication is True, "a computed CRITICAL blocks whoever is down"
    assert record.causal_simulation_valid is False
    # Nothing publishes, so the seam is never consulted; it still reports honestly.
    assert record.world_review_blocking is None
    assert api._review_limitations(record) == (), "nothing is published, so nothing is caveated"
    refusal = api._review_refusal(record)
    assert "terminal_set_in_one_step" in str(refusal)
    assert "No adversarial opinion was obtained" in str(refusal)
    assert refusal.details["surviving_mechanical_blocking_findings"] == ["terminal_set_in_one_step"]
    assert refusal.details["model_opinion_obtained"] is False


def test_an_unreachable_reviewer_still_cannot_block_on_an_opinion() -> None:
    """The other half of the ruling, and the part that does NOT change: a model-authored
    finding on a record whose model call failed decides nothing.

    (Unreachable in production — a failed call returns only mechanical findings — and
    asserted directly so the rule is pinned by the rule rather than by that accident.)
    """

    record = _errored_review_with(
        AuditFinding(
            key="decorative_actors",
            severity="CRITICAL",
            finding="every actor is decoration",
            evidence_basis="the compiled world",
        )
    )

    assert record.surviving_mechanical_blocking == ()
    assert record.blocks_publication is False
    assert record.causal_simulation_valid is False
    assert record.world_review_blocking is None


# --------------------------------------------------------------------------- #
# 5. The refusal reaches the artifacts a reader actually opens.
# --------------------------------------------------------------------------- #


def test_the_refusal_reaches_the_diagnosis_with_its_review_and_a_named_mechanism(
    tmp_path: Any,
) -> None:
    """A gate code with nothing behind it is not a diagnosis. The refusal carries the
    review that earned it, the diagnosis names a mechanism from the declared vocabulary,
    and the run directory holds the same world_review.json a completed run writes."""

    import json

    from sworldmodel.cli import _write_diagnosis

    gw = _gateway([_blocking("decorative_actors", "expert_would_call_incomplete")])
    with pytest.raises(ForecastRefused) as caught:
        run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw))
    refusal = caught.value

    diagnosis = RunDiagnosis(
        question=QUESTION,
        as_of=AS_OF_DT,
        horizon=HORIZON_DT,
        bundle=refusal.bundle,
        repair_log=refusal.repair_log,
        world_review=refusal.world_review,  # type: ignore[attr-defined]
        failure=refusal.__cause__ or refusal,
        failure_stage=refusal.stage,
    )
    _write_diagnosis(tmp_path, diagnosis, refusal)

    written = json.loads((tmp_path / "diagnosis.json").read_text())
    section = written["integrity_and_grounding"]
    assert section["stopped_at_gate"] == "world_review_blocking_findings_survived"
    assert section["gate_details"]["surviving_blocking_findings"] == [
        "decorative_actors",
        "expert_would_call_incomplete",
    ]
    review = section["pre_rollout_world_review"]
    assert review["describes_simulated_world"] is True
    assert review["blocking_failures"] == [
        "decorative_actors",
        "expert_would_call_incomplete",
    ]
    causes = [c["cause"] for c in written["root_cause"]]
    # Its own root cause, not `compiler_omission` borrowed: nothing was left out by
    # accident. A world was built, examined, found unfit, sent back, and still found
    # unfit — the fix is a different world, not a missing piece added to this one.
    assert "world_refused_by_its_own_review" in causes, causes
    assert "unclassified" not in causes
    assert "compiler_omission" not in causes

    # The same artifact name a completed run writes, so one file answers "what did the
    # review say?" whether the run published or refused.
    standalone = json.loads((tmp_path / "world_review.json").read_text())
    assert standalone["surviving_blocking_findings"]
    assert standalone["causal_simulation_valid"] is False


# --------------------------------------------------------------------------- #
# Helpers used by the signature assertion above.
# --------------------------------------------------------------------------- #


def _compile(bundle: Any, gw: ProgrammableGateway) -> Any:
    from sworldmodel.world_compiler import compile_world

    contract = api._build_contract(QUESTION, AS_OF_DT, HORIZON_DT, bundle)
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF_DT),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gw,
        seed=0,
        max_branches=6,
    )


# --------------------------------------------------------------------------- #
# 6. FD-41 — a world that answers from the record is examined, not waved through.
#
# `mechanical_world_checks` used to short-circuit to a single PASS the moment
# `_cited_factual_resolution` held, so the whole settled-record class of run (the
# OPEC+/EU-Mercosur shape) received no mechanical scrutiny at all: any claim id
# attached to any initial value established any outcome, and the only remaining
# attack was one LLM opinion which — before the FD-34 ruling — went advisory the
# moment the provider hiccupped. The attack moves to the citation itself.
# --------------------------------------------------------------------------- #


def _settled_world(*, cite: list[str] | None = None) -> dict[str, Any]:
    """A world whose terminal is already YES at t0 on a cited pre-cutoff record."""

    world = scheduled_multiparty_world(members=2, threshold=2)
    cited = world["claims"][0]["id"]
    world["world_spec"]["documents"] = [
        {
            "document_id": "agreement",
            "fields": {"signed": True},
            "evidence_claim_ids": [cited] if cite is None else cite,
        }
    ]
    world["world_spec"]["terminal"]["yes_when"] = {
        "op": "equals",
        "args": [{"op": "document_field", "args": ["agreement", "signed"]}, True],
    }
    world["world_spec"]["terminal"]["unresolved_when"] = {"op": "const", "args": [False]}
    return world


def _mechanical(world: dict[str, Any]) -> dict[str, Any]:
    from sworldmodel.world_review import mechanical_world_checks

    bundle = build_bundle(world)
    gw = _gateway()
    compiled = _compile(bundle, gw)
    view = bundle.evidence_store.view(AS_OF_DT)
    return {f.key: f for f in mechanical_world_checks(compiled, view)}


def test_a_settled_record_world_is_mechanically_examined_not_waved_through() -> None:
    """The legitimate shape still passes — and now it passes something."""

    from sworldmodel.world_compiler import _cited_factual_resolution

    world = _settled_world()
    bundle = build_bundle(world)
    compiled = _compile(bundle, _gateway())
    assert _cited_factual_resolution(compiled.spec, compiled.base_world), "fixture is settled"

    findings = _mechanical(world)
    # The operational-depth attacks stay off: a question the record already answered is
    # not re-produced inside the window, and demanding a production process for it is
    # what manufactured an absolute NO on a live Bank of England run.
    assert findings["terminal_set_in_one_step"].severity == "PASS"
    # But the citation is now examined, and the checks are real checks.
    assert findings["cited_resolution_claims_exist"].severity == "PASS"
    assert findings["cited_resolution_rests_on_the_record"].severity == "PASS"
    assert findings["cited_resolution_subject_matches"].severity == "PASS"
    assert [k for k, f in findings.items() if f.is_blocking] == []
    for finding in findings.values():
        assert finding.evidence_basis.startswith("computed from the compiled world")


def test_a_settled_record_citing_a_claim_that_does_not_exist_is_refused() -> None:
    """FD-30's shape at the point it decides an answer: the citation is an existence
    check, so any id grounds anything. An id that names nothing grounds nothing."""

    findings = _mechanical(_settled_world(cite=["c_session", "c_no_such_claim"]))

    exists = findings["cited_resolution_claims_exist"]
    assert exists.severity == "CRITICAL"
    assert exists.is_blocking
    assert "c_no_such_claim" in exists.finding


def test_a_settled_record_about_something_else_is_refused() -> None:
    """Subject and measurement scope must match: 'the record already answered this' is
    not established by a record about a different subject."""

    world = _settled_world()
    # Same world, same citation, a subject the cited claim says nothing about.
    world["world_spec"]["subject_entity"] = "the Kerguelen desalination tariff"
    findings = _mechanical(world)

    match = findings["cited_resolution_subject_matches"]
    assert match.severity == "HIGH"
    assert match.is_blocking
    assert "Kerguelen" in match.finding or "kerguelen" in match.finding.lower()


def test_a_settled_record_resting_on_inference_rather_than_record_is_refused() -> None:
    """A conclusion somebody drew is not a record of what happened."""

    world = _settled_world()
    for claim in world["claims"]:
        claim["epistemic_type"] = "inference"
    findings = _mechanical(world)

    rests = findings["cited_resolution_rests_on_the_record"]
    assert rests.severity == "CRITICAL"
    assert rests.is_blocking


def test_the_settled_record_checks_reach_the_publication_gate_end_to_end() -> None:
    """Not merely computed: a settled-record world with a broken citation must not
    publish, and must not become publishable by the reviewer being unreachable."""

    world = _settled_world(cite=["c_session", "c_no_such_claim"])
    for reviews, why in (
        ([_clean(*[k for k, _ in _QUESTIONS])], "with the reviewer answering"),
        (None, "with the reviewer unreachable"),
    ):
        gw = _gateway(reviews, fail=frozenset() if reviews else frozenset({"world_review"}))
        with pytest.raises(ForecastRefused) as caught:
            run_forecast(QUESTION, AS_OF_DT, HORIZON_DT, _config(gw, world))
        details = caught.value.__cause__.details  # type: ignore[union-attr]
        assert "cited_resolution_claims_exist" in details["surviving_blocking_findings"], why
        assert "cited_resolution_claims_exist" in details["surviving_mechanical_blocking_findings"]
