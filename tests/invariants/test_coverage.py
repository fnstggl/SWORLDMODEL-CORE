"""Evidence-to-world coverage integrity: the acceptance cases.

These lock the universal rule: every materially relevant verified evidence candidate
must be represented AND causally wired into the compiled WorldSpec that is actually
simulated, or explicitly excluded with a recorded reason. Nothing material may silently
disappear between research and simulation — for people, organizations, rules, events,
documents, or relationships — in any kind of world.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from _helpers import base_corpus, compile_dict, synthetic_corpus
from _worlds import AS_OF, HORIZON, named_body_world
from sworldmodel.coverage import (
    CandidateKind,
    Disposition,
    EvidenceCandidate,
    Materiality,
    WorldObject,
    WorldSpecView,
    assess_coverage,
    build_candidate_inventory,
    enforce_coverage,
)
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.research import build_bundle_from_dict

WINDOW_DATE = "2024-02-01T00:00:00+00:00"  # strictly between AS_OF and HORIZON

PEOPLE5 = ["Vera Nolan", "Jon Alder", "Gala Reyes", "Omar Castel", "Gabriel Cuadra"]
PEOPLE3 = ["Ada North", "Ben East", "Cara West"]


def _claim(cid: str, prop: str, value: str, entities: list[str], **kw: Any) -> dict[str, Any]:
    out = {
        "id": cid,
        "proposition": prop,
        "normalized_value": value,
        "entities": entities,
        "epistemic_type": kw.get("epistemic_type", "observation"),
        "confidence": kw.get("confidence", 0.95),
    }
    out.update({k: v for k, v in kw.items() if k in ("valid_from", "claim_key")})
    return out


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _inventory(corpus: dict[str, Any]) -> tuple[EvidenceCandidate, ...]:
    from sworldmodel.api import _build_contract

    bundle = build_bundle_from_dict(corpus)
    as_of = bundle.as_of or _dt(AS_OF)
    contract = _build_contract("q", as_of, bundle.horizon, bundle)
    view = bundle.evidence_store.view(as_of)
    return build_candidate_inventory(
        view, contract, focal_identities=tuple(e.name for e in bundle.spec.entities)
    )


# --------------------------------------------------------------------------- #
# 1. Five members in evidence, two compiled — the coverage gate rejects.
# --------------------------------------------------------------------------- #


def test_case1_five_members_two_compiled_is_rejected() -> None:
    corpus = named_body_world(PEOPLE5, represented=PEOPLE5[:2])
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    missing = " ".join(exc.value.details.get("missing_material_candidates", []))  # type: ignore[union-attr]
    assert "Gala Reyes" in missing and "Omar Castel" in missing and "Gabriel Cuadra" in missing


def test_case1_all_five_compiled_passes() -> None:
    compiled = compile_dict(named_body_world(PEOPLE5, represented=PEOPLE5))
    report = compiled.coverage_report
    assert report.is_complete
    persons = [c for c in report.candidates if c.kind is CandidateKind.PERSON]
    assert len(persons) == 5
    for c in persons:
        disp = report.disposition_for(c.candidate_id)
        assert disp is not None and disp.disposition is Disposition.INCLUDED


# --------------------------------------------------------------------------- #
# 2. A relevant organization (no named individual) is still represented.
# --------------------------------------------------------------------------- #


def test_case2_organization_without_individuals_is_covered() -> None:
    # The organization itself is the acting entity; the evidence names no individuals.
    compiled = compile_dict(named_body_world([], represented=[], org_is_actor=True))
    report = compiled.coverage_report
    orgs = [c for c in report.candidates if c.kind is CandidateKind.ORGANIZATION]
    assert orgs, "the deciding organization must appear as a candidate"
    org = orgs[0]
    assert org.is_material
    disp = report.disposition_for(org.candidate_id)
    assert disp is not None and disp.disposition is Disposition.INCLUDED
    assert not [c for c in report.candidates if c.kind is CandidateKind.PERSON]
    assert report.is_complete


# --------------------------------------------------------------------------- #
# 3. A binding institutional rule that the world omits blocks simulation.
# --------------------------------------------------------------------------- #

_CHARTER = _claim(
    "charter_rule",
    "The board charter requires unanimous consent of all members to adopt any measure.",
    "unanimous_consent",
    ["the charter"],
)


def test_case3_omitted_binding_rule_blocks() -> None:
    # Evidence: the charter requires UNANIMOUS consent; the compiled world decides by a
    # plain majority and never represents the unanimity requirement.
    corpus = named_body_world(
        PEOPLE3, represented=PEOPLE3, rule_kind="majority", extra_claims=[_CHARTER]
    )
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    missing = " ".join(exc.value.details.get("missing_material_candidates", []))  # type: ignore[union-attr]
    assert "unanimous" in missing.lower() or "charter" in missing.lower()


def test_case3_represented_rule_passes() -> None:
    corpus = named_body_world(
        PEOPLE3, represented=PEOPLE3, rule_kind="unanimous", extra_claims=[_CHARTER]
    )
    assert compile_dict(corpus).coverage_report.is_complete


# --------------------------------------------------------------------------- #
# 4. An upcoming, outcome-relevant scheduled event that the world omits blocks.
# --------------------------------------------------------------------------- #


def test_case4_omitted_inwindow_event_blocks() -> None:
    event_claim = _claim(
        "release_1",
        "A decisive inflation data print is scheduled for February 1.",
        "scheduled",
        ["the February data print"],
        valid_from=WINDOW_DATE,
    )
    corpus = named_body_world(PEOPLE3, represented=PEOPLE3, extra_claims=[event_claim])
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    missing = " ".join(exc.value.details.get("missing_material_candidates", []))  # type: ignore[union-attr]
    assert "scheduled_event" in missing


# --------------------------------------------------------------------------- #
# 5. A relevant document must be accessible to the actors (causal-use).
# --------------------------------------------------------------------------- #

_BRIEF = _claim(
    "brief_1",
    "The confidential staff briefing report sets out the decisive guidance for the vote.",
    "guidance_brief",
    ["the staff briefing"],
)
_BRIEF_REQUIRED = [
    {
        "key": "guidance_document",
        "description": "the decisive staff briefing",
        "evidence_claim_ids": ["brief_1"],
    }
]


def _document_corpus(*, accessible: bool) -> dict[str, Any]:
    return named_body_world(
        PEOPLE3,
        represented=PEOPLE3,
        extra_claims=[_BRIEF],
        extra_required=_BRIEF_REQUIRED,
        # When accessible, the chair actually holds the briefing (it enters their memory).
        doc_holder_claims=("brief_1",) if accessible else (),
    )


def test_case5_inaccessible_document_blocks() -> None:
    with pytest.raises(WorldIntegrityError):
        compile_dict(_document_corpus(accessible=False))


def test_case5_accessible_document_is_covered() -> None:
    report = compile_dict(_document_corpus(accessible=True)).coverage_report
    docs = [c for c in report.candidates if c.kind is CandidateKind.DOCUMENT and c.is_material]
    assert docs
    disp = report.disposition_for(docs[0].candidate_id)
    assert disp is not None and disp.disposition is Disposition.INCLUDED
    assert "actor_view" in disp.causal_uses
    assert report.is_complete


# --------------------------------------------------------------------------- #
# 6. An incidental person may be excluded with a recorded reason (no block).
# --------------------------------------------------------------------------- #


def test_case6_incidental_person_is_excluded_with_reason() -> None:
    gossip = _claim(
        "column_1",
        "Jane Quill published a newspaper column speculating about the sector.",
        "commentary",
        ["Jane Quill"],
    )
    report = compile_dict(
        named_body_world(PEOPLE3, represented=PEOPLE3, extra_claims=[gossip])
    ).coverage_report
    jane = next(c for c in report.candidates if c.canonical_identity == "Jane Quill")
    assert not jane.is_material
    disp = report.disposition_for(jane.candidate_id)
    assert disp is not None and disp.disposition is Disposition.EXCLUDED_IRRELEVANT
    assert disp.reason  # a concrete, recorded reason
    assert report.is_complete


# --------------------------------------------------------------------------- #
# 7. Multiple claims about one entity merge into one canonical object.
# --------------------------------------------------------------------------- #


def test_case7_same_entity_claims_merge_into_one_candidate() -> None:
    extra = [
        _claim("ada_2", "Ada North chairs the board.", "chair", ["Ada North"]),
        _claim("ada_3", "Ada North favored a hold previously.", "hold", ["Ada North"]),
    ]
    candidates = _inventory(
        named_body_world(
            ["Ada North", "Ben East"], represented=["Ada North", "Ben East"], extra_claims=extra
        )
    )
    ada = [c for c in candidates if c.canonical_identity == "Ada North"]
    assert len(ada) == 1  # one canonical person, not three
    assert {"r_ada_north", "ada_2", "ada_3"} <= set(ada[0].claim_ids)


def test_case7_alias_candidates_merge_to_one_object() -> None:
    inst = WorldObject(
        object_id="inst",
        kind="actor",
        name="Central Bank",
        claim_ids=("k1",),
        wired=True,
        uses=("actor_view", "action"),
    )
    spec = WorldSpecView(objects=(inst,), subject_entity="Central Bank")
    a = EvidenceCandidate(
        candidate_id="c_a",
        kind=CandidateKind.ORGANIZATION,
        canonical_identity="Central Bank",
        description="org",
        claim_ids=("k1",),
        lineage_ids=("e1",),
        materiality=Materiality.MATERIAL,
    )
    b = EvidenceCandidate(
        candidate_id="c_b",
        kind=CandidateKind.ORGANIZATION,
        canonical_identity="the Bank",
        description="org alias",
        claim_ids=("k1",),  # same underlying claim -> same canonical object
        lineage_ids=("e1",),
        materiality=Materiality.MATERIAL,
    )
    report = assess_coverage((a, b), spec)
    dispositions = {d.candidate_id: d.disposition for d in report.dispositions}
    assert Disposition.INCLUDED in dispositions.values()
    assert dispositions["c_b"] is Disposition.MERGED
    assert report.merged_candidates == 1
    assert report.is_complete


# --------------------------------------------------------------------------- #
# 8. Conflicting evidence is never silently resolved — and it blocks.
# --------------------------------------------------------------------------- #


def test_case8_conflicting_evidence_is_not_silently_resolved() -> None:
    conflict = [
        _claim("hawk", "Kim Vale favors a rate hike.", "hike", ["Kim Vale"], claim_key="vale"),
        _claim("dove", "Kim Vale favors a rate cut.", "cut", ["Kim Vale"], claim_key="vale"),
    ]
    corpus = named_body_world(
        ["Ada North", "Ben East"], represented=["Ada North", "Ben East"], extra_claims=conflict
    )
    corpus["contradictions"] = [["hawk", "dove"]]
    kim = next(c for c in _inventory(corpus) if c.canonical_identity == "Kim Vale")
    assert kim.materiality is Materiality.CONFLICTED
    # With no representation, a conflicted candidate is REQUIRED_BUT_UNRESOLVED and blocks.
    report = assess_coverage((kim,), WorldSpecView(objects=()))
    disp = report.disposition_for(kim.candidate_id)
    assert disp is not None
    assert disp.disposition in (Disposition.UNCERTAIN, Disposition.REQUIRED_BUT_UNRESOLVED)
    assert not report.is_complete
    with pytest.raises(WorldIntegrityError):
        enforce_coverage(report)


# --------------------------------------------------------------------------- #
# 9. The same coverage system works across materially different worlds.
# --------------------------------------------------------------------------- #


def test_case9_coverage_runs_for_every_kind_of_world() -> None:
    corpora = {
        "base_committee": base_corpus(),
        "committee_decision": synthetic_corpus("committee_decision"),
        "individual_response": synthetic_corpus("individual_response"),
        "negotiation": synthetic_corpus("negotiation"),
        "population_behavior": synthetic_corpus("population_behavior"),
        "geopolitical_process": synthetic_corpus("geopolitical_process"),
    }
    for name, corpus in corpora.items():
        report = compile_dict(corpus).coverage_report
        assert report.is_complete, f"{name} coverage incomplete"
        # Every candidate carries a disposition regardless of world shape.
        assert len(report.dispositions) == report.total_candidates


# --------------------------------------------------------------------------- #
# 10. Exclusion challenge, targeted repair, and one disposition per candidate.
# --------------------------------------------------------------------------- #


def test_exclusion_challenge_blocks_a_wrongful_exclusion() -> None:
    cand = EvidenceCandidate(
        candidate_id="x",
        kind=CandidateKind.RULE,
        canonical_identity="a five-day notice requirement",
        description="members must be given five days notice before any binding vote",
        claim_ids=("k1",),
        lineage_ids=("e1",),
        materiality=Materiality.IMMATERIAL,
    )
    spec = WorldSpecView(objects=())
    # No reviewer -> excluded with a recorded reason, run proceeds.
    r0 = assess_coverage((cand,), spec)
    d0 = r0.disposition_for(cand.candidate_id)
    assert d0 is not None and d0.disposition is Disposition.EXCLUDED_IRRELEVANT
    assert r0.is_complete
    # An independent review that disagrees invalidates the exclusion and blocks.
    r1 = assess_coverage((cand,), spec, exclusion_reviewer=lambda c: True)
    d1 = r1.disposition_for(cand.candidate_id)
    assert d1 is not None and d1.disposition is Disposition.UNCERTAIN
    assert d1.reviewer_stage == "exclusion_challenge"
    assert not r1.is_complete
    with pytest.raises(WorldIntegrityError):
        enforce_coverage(r1)


class _RepairBackend:
    """A backend whose first research drops members, but whose targeted follow-up
    research (augment_for_coverage) returns the complete roster."""

    is_live = False

    def __init__(self, incomplete: dict[str, Any], full: dict[str, Any]) -> None:
        self._incomplete = incomplete
        self._full = full
        self.augmented_with: list[str] | None = None

    def research(self, question: str, as_of: Any, horizon: Any) -> Any:
        return build_bundle_from_dict(self._incomplete)

    def augment_for_coverage(
        self, question: str, as_of: Any, horizon: Any, missing: list[str], prior: Any
    ) -> Any:
        self.augmented_with = missing
        return build_bundle_from_dict(self._full)


def test_repair_loop_recovers_via_targeted_research() -> None:
    from sworldmodel import DeterministicGateway, ForecastConfig, run_forecast

    backend = _RepairBackend(
        named_body_world(PEOPLE5, represented=PEOPLE5[:2]),
        named_body_world(PEOPLE5, represented=PEOPLE5),
    )
    config = ForecastConfig(
        gateway=DeterministicGateway(), research_backend=backend, max_branches=4
    )
    result, ctx = run_forecast("q", _dt(AS_OF), _dt(HORIZON), config)
    # The gate forced a targeted follow-up, and the recompiled world is complete.
    assert backend.augmented_with is not None
    assert any("Gala Reyes" in m for m in backend.augmented_with)
    assert ctx.compiled.coverage_report.is_complete
    assert result.integrity_manifest.represented_participants == 5


def test_repair_loop_blocks_when_augmentation_cannot_help() -> None:
    from sworldmodel import DeterministicGateway, ForecastConfig, run_forecast

    incomplete = named_body_world(PEOPLE3, represented=PEOPLE3[:1])
    backend = _RepairBackend(incomplete, incomplete)  # augment returns the same gap
    config = ForecastConfig(
        gateway=DeterministicGateway(), research_backend=backend, max_branches=4
    )
    with pytest.raises(WorldIntegrityError):
        run_forecast("q", _dt(AS_OF), _dt(HORIZON), config)


def test_case10_every_candidate_gets_exactly_one_disposition() -> None:
    report = compile_dict(named_body_world(PEOPLE3, represented=PEOPLE3)).coverage_report
    covered_ids = [d.candidate_id for d in report.dispositions]
    assert len(covered_ids) == len(set(covered_ids))  # no double-disposition
    assert set(covered_ids) == {c.candidate_id for c in report.candidates}  # none dropped
    assert report.total_candidates == len(report.candidates)
