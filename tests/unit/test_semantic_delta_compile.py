"""Delta repair through the real semantic compile path.

The compile cycle's contract: the initial plan is the only full generation; every
revision — validator fix, reviewer correction, repair carried across cycles — is a
delta call whose reply is merged deterministically into the accepted prior plan. These
tests drive ``semantic_compile_live`` with a scripted provider and prove the contract
end to end: the corrected object changes, everything else survives byte-for-byte, no
full plan is regenerated for a local defect, and a full-plan re-emission is caught and
re-asked as a delta.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from _fakes import ProgrammableGateway
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.evidence import EvidenceClaim, EvidenceStore
from sworldmodel.ids import canonical_json
from sworldmodel.models import AuthorityLevel, EpistemicType, SourceType
from sworldmodel.semantic_compile import semantic_compile_live

AS_OF = datetime.fromisoformat("2026-01-10T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-03-01T23:59:59+00:00")


def registry_plan(*, initial: Any = "UNKNOWN") -> dict[str, Any]:
    """An invented-domain plan (a vessel registry) that is valid when the resolving
    quantity starts UNKNOWN, and carries a precise-but-uncited initial otherwise —
    exactly one validator defect, fixable by one delta."""

    return {
        "resolution": {
            "question": "Will at least one vessel be registered before March 1?",
            "yes_condition": "the registry records at least one vessel",
            "subject_entity": "Registrar of Vessels",
            "resolution_units": "registered vessels",
            "target_outcome": "a vessel is registered",
            "expected_participants": None,
            "evidence_claim_ids": ["c-1"],
        },
        "entities": [
            {
                "name": "Registrar of Vessels",
                "structural_type": "person",
                "role": "registrar with sole authority over the registry",
                "representation_scale": "individual",
                "represents_count": None,
                "decides": True,
                "authority": "may register vessels",
                "why_material": "the outcome is their own act",
                "evidence_claim_ids": ["c-1"],
            }
        ],
        "states": [
            {
                "name": "vessels registered",
                "owner": "world",
                "state_type": "quantity",
                "unit": "vessels",
                "initial": initial,
                "why_material": "the resolving count",
                "evidence_claim_ids": [],
            }
        ],
        "events": [],
        "affordances": [
            {
                "name": "register a vessel",
                "meaning": "record one vessel in the registry",
                "actor": "Registrar of Vessels",
                "authority_required": "registrar authority",
                "visibility": "public",
                "duration_seconds": 0,
                "changes": [{"op": "increase", "target": "vessels registered", "amount": 1}],
                "evidence_claim_ids": ["c-1"],
            }
        ],
        "processes": [
            {
                "name": "registry session",
                "meaning": "the dated session at which registrations are processed",
                "kind": "actor_moment",
                "participants": ["Registrar of Vessels"],
                "allowed_affordances": ["register a vessel"],
                "at": "2026-02-10T10:00:00+00:00",
                "deadline": "2026-02-28T23:59:59+00:00",
                "occurrences": [],
                "evidence_claim_ids": ["c-1"],
            }
        ],
        "uncertainties": [],
        "terminal": {
            "form": "quantity_comparison",
            "state": "vessels registered",
            "comparison": "greater_or_equal",
            "threshold": 1,
        },
        "terminal_producer_note": "only the registrar's affordance increases the count; "
        "nothing initializes it and no uncertainty writes it",
        "world_facts": [],
    }


def _view() -> Any:
    store = EvidenceStore()
    store.add(
        EvidenceClaim(
            id="c-1",
            proposition="the registrar holds sole authority over the registry",
            normalized_value="sole authority",
            entities=("Registrar of Vessels",),
            valid_from=AS_OF,
            valid_until=None,
            published_at=AS_OF,
            available_at=AS_OF,
            source_id="src",
            source_url="https://example.test/registry",
            source_title="registry charter",
            source_type=SourceType.OFFICIAL_INSTITUTIONAL,
            authority_level=AuthorityLevel.AUTHORITATIVE,
            supporting_excerpt="the registrar holds sole authority",
            lineage_event_id="ev_c1",
            epistemic_type=EpistemicType.OBSERVATION,
            confidence=0.95,
            retrieved_at=AS_OF,
        )
    )
    return store.view(AS_OF)


APPROVE = {"verdict": "APPROVE", "reasons": [], "corrections": []}

FIX_STATE_DELTA = {
    "revised": {
        "states": [
            {
                "name": "vessels registered",
                "owner": "world",
                "state_type": "quantity",
                "unit": "vessels",
                "initial": "UNKNOWN",
                "why_material": "the resolving count",
                "evidence_claim_ids": [],
            }
        ]
    },
    "removed": {},
}


def test_a_validator_defect_is_fixed_by_a_delta_never_a_regeneration() -> None:
    """One uncited precise initial → one delta round replacing that one state. The
    full plan is generated exactly once, and every object the delta did not name
    survives byte-for-byte into the compiled record."""

    defect = registry_plan(initial=7)  # precise number, no citations: validator error
    gw = ProgrammableGateway(
        {
            "semantic_plan": defect,
            "semantic_plan_delta": FIX_STATE_DELTA,
            "semantic_review": APPROVE,
        }
    )
    compilation, _resp = semantic_compile_live(
        gw, defect["resolution"]["question"], AS_OF, HORIZON, _view()
    )

    plan_calls = [r for r in gw.seen if r.task_kind == "semantic_plan"]
    delta_calls = [r for r in gw.seen if r.task_kind == "semantic_plan_delta"]
    assert len(plan_calls) == 1, "the initial plan is the only full generation"
    assert len(delta_calls) == 1
    assert "validator:" in delta_calls[0].prompt
    assert "PREVIOUS PLAN" in delta_calls[0].prompt

    final = compilation["_semantic"]["plan"]
    assert final["states"][0]["initial"] == "UNKNOWN"
    # Byte-equivalence of everything the correction did not name: entities,
    # affordances, processes, terminal, citations and all.
    for section in ("entities", "affordances", "processes", "terminal", "resolution"):
        assert canonical_json(final[section]) == canonical_json(defect[section]), section
    assert compilation["_semantic"]["delta_rounds"] == 1


def test_the_delta_prompt_shares_the_full_plan_prompt_as_an_exact_prefix() -> None:
    """Provider-side context caching prices shared prefixes; the revision prompt must
    therefore START with the initial plan prompt, byte for byte, with all
    round-specific content appended after it."""

    defect = registry_plan(initial=7)
    gw = ProgrammableGateway(
        {
            "semantic_plan": defect,
            "semantic_plan_delta": FIX_STATE_DELTA,
            "semantic_review": APPROVE,
        }
    )
    semantic_compile_live(gw, defect["resolution"]["question"], AS_OF, HORIZON, _view())
    plan_prompt = next(r for r in gw.seen if r.task_kind == "semantic_plan").prompt
    delta_prompt = next(r for r in gw.seen if r.task_kind == "semantic_plan_delta").prompt
    assert delta_prompt.startswith(plan_prompt)


def test_a_reviewer_correction_is_applied_as_a_delta() -> None:
    valid = registry_plan()
    revise_then_approve_calls: list[str] = []

    def review(_ctx: dict[str, Any]) -> dict[str, Any]:
        revise_then_approve_calls.append("review")
        return {
            "verdict": "REVISE",
            "reasons": ["the registrar needs a refusal affordance"],
            "corrections": ["add affordance 'decline a registration' for the Registrar of Vessels"],
        }

    add_affordance_delta = {
        "revised": {
            "affordances": [
                {
                    "name": "decline a registration",
                    "meaning": "refuse to register a vessel",
                    "actor": "Registrar of Vessels",
                    "authority_required": "registrar authority",
                    "visibility": "public",
                    "duration_seconds": 0,
                    "changes": [
                        {
                            "op": "send",
                            "target": "registration decision notice",
                            "recipients": ["Registrar of Vessels"],
                            "detail": "declined",
                        }
                    ],
                    "evidence_claim_ids": ["c-1"],
                }
            ]
        }
    }
    gw = ProgrammableGateway(
        {
            "semantic_plan": valid,
            "semantic_plan_delta": add_affordance_delta,
            "semantic_review": review,
        }
    )
    compilation, _resp = semantic_compile_live(
        gw, valid["resolution"]["question"], AS_OF, HORIZON, _view()
    )
    delta_calls = [r for r in gw.seen if r.task_kind == "semantic_plan_delta"]
    assert len(delta_calls) == 1
    assert "reviewer:" in delta_calls[0].prompt
    final = compilation["_semantic"]["plan"]
    assert [a["name"] for a in final["affordances"]] == [
        "register a vessel",
        "decline a registration",
    ]
    # The untouched affordance is byte-equivalent; the review record says a revision
    # was applied.
    assert canonical_json(final["affordances"][0]) == canonical_json(valid["affordances"][0])
    assert compilation["_semantic"]["review"]["revision_applied"] is True


def test_a_full_plan_reemission_gets_one_reshape_round_then_merges() -> None:
    """A model that answers the delta call with the whole plan is re-asked ONCE with
    the exact shape error; the reshaped delta then merges normally."""

    defect = registry_plan(initial=7)

    def delta(ctx: dict[str, Any]) -> dict[str, Any]:
        # First delta attempt re-emits a full plan (the old failure mode); the
        # reshape round (attempt >= 200) answers with a real delta.
        if int(ctx.get("attempt", 0)) >= 200:
            return FIX_STATE_DELTA
        return registry_plan()  # full plan, not a delta

    gw = ProgrammableGateway(
        {
            "semantic_plan": defect,
            "semantic_plan_delta": delta,
            "semantic_review": APPROVE,
        }
    )
    compilation, _resp = semantic_compile_live(
        gw, defect["resolution"]["question"], AS_OF, HORIZON, _view()
    )
    delta_calls = [r for r in gw.seen if r.task_kind == "semantic_plan_delta"]
    assert len(delta_calls) == 2
    assert "not a readable delta" in delta_calls[1].prompt
    assert "re-emitted a complete plan" in delta_calls[1].prompt
    assert compilation["_semantic"]["plan"]["states"][0]["initial"] == "UNKNOWN"


def test_an_unusable_delta_refuses_as_the_pipeline_with_the_plan_errors() -> None:
    """A revision that cannot repair the plan ends in the pipeline's own refusal type
    with the validator's precise errors — never a silent fallback, never a direct
    compile."""

    defect = registry_plan(initial=7)
    gw = ProgrammableGateway(
        {
            "semantic_plan": defect,
            "semantic_plan_delta": {"revised": {}, "removed": {}},  # changes nothing
            "semantic_review": APPROVE,
        }
    )
    with pytest.raises(WorldIntegrityError) as exc:
        semantic_compile_live(gw, defect["resolution"]["question"], AS_OF, HORIZON, _view())
    assert exc.value.details["failure"] == "semantic_plan_invalid"
    assert exc.value.details["recompilable"] is True
    assert any("initial quantity" in e for e in exc.value.details["semantic_errors"])
    # No fallback: the direct compiler was never consulted.
    assert not [r for r in gw.seen if r.task_kind == "compile_world_spec"]
