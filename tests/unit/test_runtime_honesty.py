"""Runtime honesty: an unknown must never silently become a confident answer.

Covers audit findings:

* H-5 — an ``adjust_field`` whose delta expression is undetermined used to stay
  feasible (the ``None`` exemption in ``_unusable_quantity``), be coerced to ``+0.0``,
  and DETERMINE a previously-unset field — defeating any ``equals(field, None)``
  unresolved guard and producing a confident NO from an unknown.
* H-5 corollary — a direct-mode terminal over a field the world never initializes,
  with the default ``unresolved_when = const(False)``, resolved a confident NO because
  comparison coerces an absent quantity to zero. The compile gate now demands an
  is-unset guard for every such field.
* M-4 — an actor woken at a dated moment with no feasible action and ``allow_novel``
  false was skipped with no record at all, so the world looked inert for no stated
  reason.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from _fakes import ProgrammableGateway, build_bundle, wait_decision
from sworldmodel.effects import EffectExecutor
from sworldmodel.engine import run
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.evidence import EvidenceStore, EvidenceView
from sworldmodel.expressions import evaluate
from sworldmodel.models import BranchWeight, ResolutionContract, WeightProvenance
from sworldmodel.world import WorldState
from sworldmodel.world_compiler import compile_world
from sworldmodel.worldspec import Effect, TerminalExpression, parse_expr

AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")


def _terminal() -> TerminalExpression:
    return TerminalExpression(
        description="YES when total reaches 100",
        yes_when=parse_expr(
            {"op": "greater_or_equal", "args": [{"op": "field", "args": ["total"]}, 100]}
        ),
        unresolved_when=parse_expr(
            {"op": "equals", "args": [{"op": "field", "args": ["total"]}, None]}
        ),
    )


def _world() -> WorldState:
    contract = ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity="s",
        resolution_units="binary",
        terminal=_terminal(),
    )
    return WorldState(
        branch_id="b",
        parent_branch_id=None,
        weight=BranchWeight(1.0, WeightProvenance.DIRECT_EMPIRICAL, "test"),
        time=AS_OF,
        contract=contract,
        evidence=EvidenceView(store=EvidenceStore(), as_of=AS_OF),
    )


# The delta is base * mult with mult never set anywhere: the expression is
# undetermined, and no value for the field exists in the world.
_UNDETERMINED_DELTA = {
    "op": "multiply",
    "args": [{"op": "field", "args": ["base"]}, {"op": "field", "args": ["mult"]}],
}


# ---------------------------------------------------------------------------
# H-5: an undetermined adjust_field delta refuses instead of determining a field
# ---------------------------------------------------------------------------


def test_undetermined_adjust_field_delta_makes_the_action_infeasible() -> None:
    world = _world()
    eff = Effect(op="adjust_field", params=(("field", "total"), ("delta", _UNDETERMINED_DELTA)))
    ok, reason = EffectExecutor().can_apply(world, (eff,), {"actor": "a"})
    assert not ok
    assert "could not determine" in reason


def test_undetermined_adjust_field_never_determines_the_field() -> None:
    """Even when the event is built and applied (the deferred-effect path re-resolves
    parameters at fire time, past the feasibility check), the field stays unset and the
    unresolved guard over it keeps holding."""

    world = _world()
    eff = Effect(op="adjust_field", params=(("field", "total"), ("delta", _UNDETERMINED_DELTA)))
    events, deferred = EffectExecutor().build_events(world, (eff,), {"actor": "a"})
    assert not deferred
    assert events[0].payload_dict["delta"] is None  # never coerced to 0.0
    after = world.apply(events)
    assert after.get_field("total") is None, "an undetermined delta determined the field"
    assert evaluate(world.contract.terminal.unresolved_when, after) is True


def test_undetermined_transfer_and_consume_amounts_are_refused_and_move_nothing() -> None:
    world = _world()
    for op, params in (
        (
            "transfer_resource",
            (("resource", "funds"), ("from", "a"), ("to", "b"), ("amount", _UNDETERMINED_DELTA)),
        ),
        (
            "consume_resource",
            (("resource", "funds"), ("holder", "a"), ("amount", _UNDETERMINED_DELTA)),
        ),
    ):
        eff = Effect(op=op, params=params)
        ok, reason = EffectExecutor().can_apply(world, (eff,), {"actor": "a"})
        assert not ok, f"{op} with an undetermined amount stayed feasible"
        assert "could not determine" in reason
        events, _ = EffectExecutor().build_events(world, (eff,), {"actor": "a"})
        after = world.apply(events)
        assert dict(after.resources) == dict(world.resources), (
            f"{op} with an undetermined amount moved resources"
        )


def test_missing_actor_parameter_quantity_is_refused_not_zeroed() -> None:
    """``$param.amount`` with no such parameter resolves to None — the same hole, one
    binding step earlier. The action must refuse, not transfer nothing as done."""

    world = _world()
    eff = Effect(op="adjust_field", params=(("field", "total"), ("delta", "$param.amount")))
    ok, reason = EffectExecutor().can_apply(world, (eff,), {"actor": "a", "params": {}})
    assert not ok
    assert "could not determine" in reason


# ---------------------------------------------------------------------------
# H-5 corollary: a terminal over a never-initialized field must carry an
# is-unset guard, or the compile gate refuses
# ---------------------------------------------------------------------------


def _unset_field_world(unresolved_when: dict[str, Any]) -> dict[str, Any]:
    """A world whose terminal reads ``total``, a field with NO compiled initial value,
    produced only if the operator's action runs."""

    return {
        "reality": {
            "as_of": AS_OF.isoformat(),
            "horizon": HORIZON.isoformat(),
            "subject_entity": "the total",
            "resolution_units": "units",
            "target_outcome": "the total reaches 100",
            "expected_participants": 1,
        },
        "claims": [
            {
                "id": "c_actor",
                "proposition": "the operator and their mandate are documented",
                "value": True,
                "supporting_excerpt": "the operator and their mandate are documented",
            },
            {
                "id": "c_cycle",
                "proposition": "the production cycle is scheduled",
                "value": True,
                "supporting_excerpt": "the production cycle is scheduled",
            },
        ],
        "world_spec": {
            "title": "unset terminal field",
            "subject_entity": "the total",
            "resolution_units": "units",
            "entities": [
                {
                    "entity_id": "operator",
                    "name": "The Operator",
                    "kind": "person",
                    "is_actor": True,
                    "role": "operator",
                    "authority": ["produce"],
                    "representation_scale": "individual",
                    "evidence_claim_ids": ["c_actor"],
                }
            ],
            "actors": [
                {
                    "entity_id": "operator",
                    "reasoning": "produces when scheduled",
                    "memory_seeds": [
                        {
                            "content": "I run the cycle.",
                            "kind": "episodic",
                            "importance": 0.9,
                            "evidence_claim_ids": ["c_actor"],
                        }
                    ],
                }
            ],
            "fields": [{"field_id": "total", "value_type": "number"}],  # no initial
            "actions": [
                {
                    "action_id": "produce",
                    "meaning": "run a production step",
                    "eligible_actors": ["role:operator"],
                    "required_authority": ["produce"],
                    "parameters": [],
                    "visibility": "public",
                    "effects": [{"op": "adjust_field", "field": "total", "delta": 60}],
                    "evidence_claim_ids": ["c_cycle"],
                }
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "cycle",
                        "stage": "cycle",
                        "at": "2026-06-01T09:00:00+00:00",
                        "description": "the cycle",
                        "participants": ["operator"],
                        "action_ids": ["produce"],
                        "allow_novel": False,
                    }
                ]
            },
            "external_processes": [],
            "wake_rules": [],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [{"op": "field", "args": ["total"]}, 100],
                },
                "unresolved_when": unresolved_when,
                "description": "YES when total reaches 100",
            },
        },
    }


def _compile(data: dict[str, Any], gw: ProgrammableGateway) -> Any:
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="will the total reach 100?",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        required_reality_facts=bundle.required_reality_facts,
        expected_participants=bundle.expected_participants,
    )
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gw,
        seed=0,
        max_branches=2,
    )


def _wait_gateway() -> ProgrammableGateway:
    return ProgrammableGateway(
        {
            "actor_decision": lambda ctx: wait_decision("observing"),
            "reflect": {"beliefs_update": [], "new_memories": []},
        }
    )


def test_unguarded_terminal_over_unset_field_is_refused_at_compile() -> None:
    """The direct compiler's default ``unresolved_when = const(False)`` plus total
    comparison (absent -> 0.0) used to resolve a confident NO over a quantity nobody
    produced. The gate refuses with a recompilable failure naming the fields."""

    data = _unset_field_world({"op": "const", "args": [False]})
    with pytest.raises(WorldIntegrityError) as exc_info:
        _compile(data, _wait_gateway())
    details = exc_info.value.details
    assert details["failure"] == "terminal_unset_fields_unguarded"
    assert details["recompilable"] is True
    assert details["unguarded fields"] == ["total"]


def test_guarded_terminal_over_unset_field_compiles_and_stays_unresolved() -> None:
    """The exact guard shape the semantic lowerer derives for UNKNOWN states —
    ``equals(field(x), None)`` — passes the gate, and a branch in which nothing ever
    produces the field reports unresolved rather than a manufactured NO."""

    data = _unset_field_world({"op": "equals", "args": [{"op": "field", "args": ["total"]}, None]})
    gw = _wait_gateway()
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    (branch,) = result.branch_outcomes
    assert not branch.resolved
    assert branch.outcome is None


# ---------------------------------------------------------------------------
# M-4: a wake with no feasible action leaves a decision record, not silence
# ---------------------------------------------------------------------------


def _authority_mismatch_world() -> dict[str, Any]:
    """One actor, one dated node offering one action whose authority token the actor
    does not hold, and novel actions disallowed at the node."""

    return {
        "reality": {
            "as_of": AS_OF.isoformat(),
            "horizon": HORIZON.isoformat(),
            "subject_entity": "the record",
            "resolution_units": "recorded entries",
            "target_outcome": "an entry is recorded",
            "expected_participants": 1,
        },
        "claims": [
            {
                "id": "c_actor",
                "proposition": "the officer and their mandate are documented",
                "value": True,
                "supporting_excerpt": "the officer and their mandate are documented",
            },
            {
                "id": "c_session",
                "proposition": "the session is scheduled for 2026-06-01",
                "value": True,
                "supporting_excerpt": "the session is scheduled for 2026-06-01",
            },
        ],
        "world_spec": {
            "title": "authority mismatch",
            "subject_entity": "the record",
            "resolution_units": "recorded entries",
            "entities": [
                {
                    "entity_id": "officer",
                    "name": "The Officer",
                    "kind": "person",
                    "is_actor": True,
                    "role": "officer",
                    "authority": ["observe"],
                    "representation_scale": "individual",
                    "evidence_claim_ids": ["c_actor"],
                }
            ],
            "actors": [
                {
                    "entity_id": "officer",
                    "reasoning": "acts within the documented mandate",
                    "memory_seeds": [
                        {
                            "content": "I hold observer status only.",
                            "kind": "episodic",
                            "importance": 0.9,
                            "evidence_claim_ids": ["c_actor"],
                        }
                    ],
                }
            ],
            "fields": [],
            "actions": [
                {
                    "action_id": "record_entry",
                    "meaning": "record an entry",
                    "eligible_actors": ["role:officer"],
                    "required_authority": ["record"],
                    "parameters": [],
                    "visibility": "public",
                    "effects": [
                        {
                            "op": "append_record",
                            "collection": "entries",
                            "key": "$actor",
                            "value": "recorded",
                        }
                    ],
                    "evidence_claim_ids": ["c_session"],
                }
            ],
            "process": {
                "nodes": [
                    {
                        "node_id": "session",
                        "stage": "session",
                        "at": "2026-06-01T09:00:00+00:00",
                        "description": "the session",
                        "participants": ["officer"],
                        "action_ids": ["record_entry"],
                        "allow_novel": False,
                    }
                ]
            },
            "external_processes": [],
            "wake_rules": [],
            "terminal": {
                "yes_when": {
                    "op": "greater_or_equal",
                    "args": [{"op": "count", "args": ["entries"]}, 1],
                },
                "unresolved_when": {
                    "op": "less_than",
                    "args": [{"op": "count", "args": ["entries"]}, 1],
                },
                "description": "YES when an entry is recorded",
            },
        },
    }


def test_no_feasible_action_wake_is_recorded_with_reasons() -> None:
    gw = _wait_gateway()
    compiled = _compile(_authority_mismatch_world(), gw)
    result = run(compiled, gw, seed=0)

    skips = [d for d in result.actor_decisions if d.validation_status == "no_feasible_action"]
    assert skips, "the infeasible wake left no decision record — the world looks inert"
    rec = skips[0]
    assert rec.actor_id == "officer"
    assert "record_entry" in rec.validation_reason
    assert "authority" in rec.validation_reason
    # The skip records a wake, not an action: nothing was applied.
    assert rec.event_ids == []


# ---------------------------------------------------------------------------
# H-6 (export half): the evidence export carries the COMPLETE claim record
# ---------------------------------------------------------------------------


def test_evidence_export_writes_every_claim_field(tmp_path: Any) -> None:
    """The export used to write 8 of the claim's fields, dropping authority_level,
    source_type, published_at, validity, source_id, confidence, retrieved_at and
    lineage_event_id — so replayed stores misranked authority and changed which claims
    the compiler saw. Every dataclass field must reach disk, stably encoded."""

    import json

    from sworldmodel.api import _write_research_files
    from sworldmodel.evidence import EvidenceClaim

    bundle = build_bundle(_authority_mismatch_world())

    class _Cfg:
        trace_dir = tmp_path

    _write_research_files(_Cfg, {}, bundle.evidence_store)
    records = json.loads((tmp_path / "evidence_store.json").read_text())
    assert records, "no claims exported"

    expected = set(EvidenceClaim.__dataclass_fields__)
    for record in records:
        missing = expected - set(record)
        assert not missing, f"export drops claim fields: {sorted(missing)}"

    # Stable encodings: enum VALUES and ISO datetimes, never reprs.
    claims = {c.id: c for c in bundle.evidence_store.all()}
    for record in records:
        claim = claims[record["id"]]
        assert record["epistemic_type"] == claim.epistemic_type.value
        assert record["source_type"] == claim.source_type.value
        assert record["authority_level"] == claim.authority_level.value
        assert record["confidence"] == claim.confidence
        assert record["lineage_event_id"] == claim.lineage_event_id
        assert record["source_id"] == claim.source_id
        assert datetime.fromisoformat(record["published_at"]) == claim.published_at
        assert datetime.fromisoformat(record["retrieved_at"]) == claim.retrieved_at
        assert datetime.fromisoformat(record["available_at"]) == claim.available_at
        for key in ("valid_from", "valid_until", "archived_at"):
            value = getattr(claim, key)
            assert record[key] == (value.isoformat() if value is not None else None)


# ---------------------------------------------------------------------------
# M-5: per-run call/token ceilings at the gateway
# ---------------------------------------------------------------------------


def _request(kind: str = "actor_decision") -> Any:
    from sworldmodel.gateway import GatewayRequest

    return GatewayRequest(task_kind=kind, prompt="p", context={}, seed=0)


def test_call_budget_exhaustion_raises_a_gateway_error() -> None:
    from sworldmodel.errors import GatewayError

    gw = ProgrammableGateway({"actor_decision": {"mode": "wait", "reason": "r"}})
    gw.set_budget(max_calls=2)
    gw.generate(_request())
    gw.generate(_request())
    with pytest.raises(GatewayError, match="call budget exhausted"):
        gw.generate(_request())
    assert gw.call_count == 2, "the refused call must not be counted as made"


def test_token_budget_exhaustion_raises_a_gateway_error() -> None:
    from sworldmodel.errors import GatewayError

    gw = ProgrammableGateway({"actor_decision": {"mode": "wait", "reason": "r"}})
    gw.set_budget(max_tokens_total=1)
    gw.generate(_request())  # crosses the ceiling
    with pytest.raises(GatewayError, match="token budget exhausted"):
        gw.generate(_request())


def test_directly_constructed_gateways_stay_unbounded() -> None:
    gw = ProgrammableGateway({"actor_decision": {"mode": "wait", "reason": "r"}})
    for _ in range(10):
        gw.generate(_request())
    assert gw.call_count == 10


def test_run_forecast_threads_the_configured_caps_to_the_gateway() -> None:
    """A config cap of zero calls must stop the pipeline at its first model call with
    the budget's own GatewayError — proof the config reached the gateway."""

    from _fakes import FixtureResearchBackend
    from sworldmodel.api import run_forecast
    from sworldmodel.config import ForecastConfig
    from sworldmodel.errors import GatewayError

    assert ForecastConfig.__dataclass_fields__["max_calls"].default == 400
    assert ForecastConfig.__dataclass_fields__["max_tokens_total"].default == 2_000_000

    gw = _wait_gateway()
    bundle = build_bundle(_authority_mismatch_world())
    config = ForecastConfig(
        gateway=gw,
        research_backend=FixtureResearchBackend(bundle),
        max_calls=0,
    )
    with pytest.raises(GatewayError, match="call budget exhausted"):
        run_forecast("will an entry be recorded?", AS_OF, HORIZON, config)
    assert gw.call_count == 0


# --------------------------------------------------------------------------- #
# An initial-compile refusal gets its registered repair before it becomes final.
# --------------------------------------------------------------------------- #


def _initial_refusal(store) -> WorldIntegrityError:
    from sworldmodel.errors import WorldIntegrityError

    exc = WorldIntegrityError(
        "the semantic plan is invalid after validator rounds: affordance 'hold' changes nothing",
        details={"failure": "semantic_plan_invalid", "recompilable": True},
    )
    exc.partial_evidence_store = store  # type: ignore[attr-defined]
    exc.partial_live_trace = {"queries": []}  # type: ignore[attr-defined]
    return exc


def test_an_initial_compile_refusal_is_replanned_not_refused_outright(monkeypatch) -> None:
    """A geopolitical slice refused semantic_plan_invalid with ``repair_attempts: []``:
    the repair registry held a plan for exactly that failure, but the repair loop only
    wrapped the world-level gates, so a refusal from the initial compile inside
    ``research()`` never consulted it. The replan must re-read the refusal's own store
    with the plan's instruction and hand back a bundle for the normal gate path."""

    import sworldmodel.api as api
    from _fakes import FixtureResearchBackend
    from sworldmodel.config import ForecastConfig
    from sworldmodel.repair import RepairLog

    bundle = build_bundle(_authority_mismatch_world())
    exc = _initial_refusal(bundle.evidence_store)

    seen: dict[str, object] = {}

    def fake_compile(
        config,
        question,
        as_of,
        horizon,
        view,
        *,
        extra_instruction="",
        structure_id="primary",
        prior_plan=None,
    ):
        seen["instruction"] = extra_instruction
        seen["prior_plan"] = prior_plan
        return {"compiled": True}

    def fake_assemble(store, data):
        seen["store"] = store
        return bundle

    monkeypatch.setattr(api, "compile_for_mode", fake_compile)
    monkeypatch.setattr(api, "assemble_bundle", fake_assemble)

    gw = _wait_gateway()
    gw.is_live = True  # type: ignore[attr-defined]
    config = ForecastConfig(gateway=gw, research_backend=FixtureResearchBackend(bundle))
    log = RepairLog()
    out = api._replan_initial_compile("q?", AS_OF, HORIZON, config, exc, log)

    assert out is not None
    assert out.live_trace == {"queries": []}
    assert seen["store"] is bundle.evidence_store
    assert "changes nothing" in str(seen["instruction"]) or seen["instruction"], (
        "the plan's instruction must reach the recompile"
    )
    assert len(log.attempts) == 1
    assert log.attempts[0]["failure"] == "semantic_plan_invalid"
    assert "replanned" in str(log.attempts[0]["outcome"])


def test_a_replan_that_repeats_the_same_refusal_stops_as_a_reroll(monkeypatch) -> None:
    """Temperature-zero replans can reproduce the exact failure. Repeating the same
    diagnosis with nothing new to read is a reroll, and an honest run stops there
    instead of burning the ceiling."""

    import sworldmodel.api as api
    from _fakes import FixtureResearchBackend
    from sworldmodel.config import ForecastConfig
    from sworldmodel.errors import WorldIntegrityError
    from sworldmodel.repair import RepairLog

    bundle = build_bundle(_authority_mismatch_world())
    exc = _initial_refusal(bundle.evidence_store)

    def same_refusal(*a, **k):
        raise WorldIntegrityError(
            str(exc), details={"failure": "semantic_plan_invalid", "recompilable": True}
        )

    monkeypatch.setattr(api, "compile_for_mode", same_refusal)

    gw = _wait_gateway()
    gw.is_live = True  # type: ignore[attr-defined]
    config = ForecastConfig(gateway=gw, research_backend=FixtureResearchBackend(bundle))
    log = RepairLog()
    out = api._replan_initial_compile("q?", AS_OF, HORIZON, config, exc, log)

    assert out is None
    assert len(log.attempts) == 2, "one refused attempt, then the reroll stop"
    assert "reroll" in str(log.attempts[-1]["outcome"])


def test_a_final_refusal_and_a_dead_gateway_are_not_replanned() -> None:
    import sworldmodel.api as api
    from _fakes import FixtureResearchBackend
    from sworldmodel.config import ForecastConfig
    from sworldmodel.errors import WorldIntegrityError
    from sworldmodel.repair import RepairLog

    bundle = build_bundle(_authority_mismatch_world())
    config = ForecastConfig(
        gateway=_wait_gateway(), research_backend=FixtureResearchBackend(bundle)
    )

    final = WorldIntegrityError(
        "reviewed and abstained",
        details={"failure": "semantic_plan_invalid", "recompilable": False},
    )
    final.partial_evidence_store = bundle.evidence_store  # type: ignore[attr-defined]
    assert api._replan_initial_compile("q?", AS_OF, HORIZON, config, final, RepairLog()) is None

    # Recompilable, but the gateway is not live: no replan, no crash.
    exc = _initial_refusal(bundle.evidence_store)
    assert api._replan_initial_compile("q?", AS_OF, HORIZON, config, exc, RepairLog()) is None


def test_completion_exports_the_bundle_that_was_actually_simulated(monkeypatch, tmp_path) -> None:
    """The research checkpoint writes the INITIAL bundle's trace and store; when the
    pre-rollout review forces a recompile, the simulated plan rides the final bundle's
    live_trace (semantic_repair_rounds) and the checkpointed artifact describes a plan
    nobody executed. Completion must rewrite research_trace.json and
    evidence_store.json from the bundle that was simulated."""

    from dataclasses import replace as dc_replace

    import sworldmodel.api as api
    from _fakes import FixtureResearchBackend
    from sworldmodel.config import ForecastConfig

    bundle = build_bundle(_authority_mismatch_world())
    final_bundle = dc_replace(
        bundle, live_trace={"semantic_repair_rounds": [{"plan": "the simulated one"}]}
    )

    class _Ctx:
        bundle = final_bundle

        def write(self, out_dir, *, gateway_calls=None):
            return "hash"

    monkeypatch.setattr(api, "run_forecast", lambda *a, **k: (object(), _Ctx()))
    config = ForecastConfig(
        gateway=_wait_gateway(),
        research_backend=FixtureResearchBackend(bundle),
        trace_dir=tmp_path,
    )
    api.forecast("q?", AS_OF, HORIZON, config)

    import json as _json

    trace = _json.loads((tmp_path / "research_trace.json").read_text())
    assert trace == {"semantic_repair_rounds": [{"plan": "the simulated one"}]}
    store = _json.loads((tmp_path / "evidence_store.json").read_text())
    assert {c["id"] for c in store} == {c.id for c in final_bundle.evidence_store.all()}
