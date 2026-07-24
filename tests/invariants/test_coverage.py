"""Evidence-to-world coverage integrity: the ten acceptance cases.

These lock the universal rule: every materially relevant verified evidence candidate
must be represented AND causally wired into the compiled world, or explicitly excluded
with a recorded reason. Nothing material may silently disappear between research and
simulation — for people, organizations, rules, events, documents, or relationships.
"""

from __future__ import annotations

from typing import Any

import pytest
from _helpers import AS_OF, HORIZON, ROSTER_PUB, base_corpus, compile_dict, synthetic_corpus
from sworldmodel.api import _build_contract
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

# --------------------------------------------------------------------------- #
# Corpus builders
# --------------------------------------------------------------------------- #

WINDOW_DATE = "2024-02-01T00:00:00+00:00"  # strictly between AS_OF and HORIZON


def _claim(cid: str, prop: str, value: str, entities: list[str], **kw: Any) -> dict[str, Any]:
    return {
        "id": cid,
        "proposition": prop,
        "normalized_value": value,
        "entities": entities,
        "epistemic_type": kw.get("epistemic_type", "observation"),
        "confidence": kw.get("confidence", 0.95),
        **{k: v for k, v in kw.items() if k in ("valid_from", "claim_key")},
    }


def _source(claims: list[dict[str, Any]], *, sid: str = "src", lineage: str = "ev") -> dict[str, Any]:
    return {
        "source_id": sid,
        "url": f"https://example.org/{sid}",
        "title": sid,
        "source_type": "official_institutional",
        "authority_level": 4,
        "published_at": ROSTER_PUB,
        "available_at": ROSTER_PUB,
        "lineage_event_id": lineage,
        "claims": claims,
    }


def _member(name: str, cid: str, *, chair: bool = False, prior: str = "hold") -> dict[str, Any]:
    slug = name.lower().replace(" ", "_")
    return {
        "actor_id": slug,
        "name": name,
        "role": "Chair" if chair else "Member",
        "is_voting_seat": True,
        "vote_power": 1,
        "prior_action": prior,
        "authority": ["vote", "introduce_proposal", "chair"] if chair else ["vote"],
        "evidence_claim_ids": [cid],
        "memory_seeds": [
            {
                "content": f"My prior position was {prior}.",
                "kind": "episodic",
                "importance": 0.7,
                "valid_time": ROSTER_PUB,
                "evidence_claim_ids": [cid],
            }
        ],
    }


def _minimal_frame() -> dict[str, Any]:
    return {
        "options": ["cut", "hold"],
        "signals": [],
        "reaction_rules": [],
        "guidance_option": "hold",
        "guidance_text": "The common position is hold.",
        "acceptance_tolerance": 0.5,
        "uncertainty": [],
    }


def named_committee(
    people: list[str],
    represented: list[str],
    *,
    rule_kind: str = "majority",
    extra_claims: list[dict[str, Any]] | None = None,
    extra_required: list[dict[str, Any]] | None = None,
    world_facts: list[dict[str, Any]] | None = None,
    decision_body: str = "the Governing Board",
) -> dict[str, Any]:
    """A committee whose *evidence* names ``people`` but whose compiled roster only
    ``represented`` — the exact shape of the motivating failure. Seat totals are made
    internally consistent so the classic seat check passes and only the coverage gate
    can catch the lost members."""

    claims = [
        _claim(
            f"r_{p.lower().replace(' ', '_')}",
            f"{p} is a voting member of {decision_body} and voted to hold at the last meeting.",
            "member",
            [p],
        )
        for p in people
    ]
    if extra_claims:
        claims += extra_claims
    n = len(represented)
    threshold = n if rule_kind == "unanimous" else n // 2 + 1
    members = [
        _member(p, f"r_{p.lower().replace(' ', '_')}", chair=(i == 0)) for i, p in enumerate(represented)
    ]
    required = [
        {
            "key": "roster",
            "description": "verified roster",
            "evidence_claim_ids": [f"r_{p.lower().replace(' ', '_')}" for p in represented],
        }
    ]
    if extra_required:
        required += extra_required
    return {
        "question_key": "coverage_case",
        "reality": {
            "as_of": AS_OF,
            "horizon": HORIZON,
            "decision_body": decision_body,
            "subject_entity": "the measure",
            "resolution_units": f"{rule_kind} of {n} seats",
            "institution_id": "board",
            "institution_name": decision_body,
            "decision_rule": {
                "kind": rule_kind,
                "total_seats": n,
                "threshold": threshold,
                "evidence_claim_ids": [],
            },
            "expected_voting_seats": n,
            "target_option": "hold",
            "terminal": {
                "mechanism": "committee_vote",
                "yes_condition": "unanimous_for_option" if rule_kind == "unanimous" else "majority_for_option",
                "target_option": "hold",
            },
            "authoritative_sources": ["roster"],
            "members": members,
        },
        "frame": _minimal_frame(),
        "world_facts": world_facts or [],
        "required_reality_facts": required,
        "sources": [_source(claims)],
        "outcome": None,
    }


def _inventory(corpus: dict[str, Any]) -> tuple[tuple[EvidenceCandidate, ...], Any]:
    bundle = build_bundle_from_dict(corpus)
    as_of = bundle.as_of or _dt(AS_OF)
    contract = _build_contract("q", as_of, bundle.horizon, bundle)
    view = bundle.evidence_store.view(as_of)
    return build_candidate_inventory(view, contract), contract


def _dt(s: str):
    from datetime import datetime

    return datetime.fromisoformat(s)


# --------------------------------------------------------------------------- #
# 1. Five members in evidence, two compiled — the coverage gate rejects.
# --------------------------------------------------------------------------- #


def test_case1_five_members_two_compiled_is_rejected() -> None:
    people = ["Vera Nolan", "Jon Alder", "Gala Reyes", "Omar Castel", "Gabriel Cuadra"]
    corpus = named_committee(people, represented=people[:2])  # seat total is consistent at 2
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    missing = " ".join(exc.value.details.get("missing_material_candidates", []))  # type: ignore[union-attr]
    assert "Gala Reyes" in missing and "Omar Castel" in missing and "Gabriel Cuadra" in missing


def test_case1_all_five_compiled_passes() -> None:
    people = ["Vera Nolan", "Jon Alder", "Gala Reyes", "Omar Castel", "Gabriel Cuadra"]
    compiled = compile_dict(named_committee(people, represented=people))
    report = compiled.coverage_report
    assert report.is_complete
    persons = [c for c in report.candidates if c.kind is CandidateKind.PERSON]
    assert len(persons) == 5
    for c in persons:
        assert report.disposition_for(c.candidate_id).disposition is Disposition.INCLUDED  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# 2. A relevant organization (no named individual) is still represented.
# --------------------------------------------------------------------------- #


def test_case2_organization_without_individuals_is_covered() -> None:
    corpus = base_corpus()  # entities are "a"/"b"/"c"/"committee": an org, no persons
    compiled = compile_dict(corpus)
    report = compiled.coverage_report
    orgs = [c for c in report.candidates if c.kind is CandidateKind.ORGANIZATION]
    assert orgs, "the decision-body organization must appear as a candidate"
    org = orgs[0]
    assert org.is_material
    disp = report.disposition_for(org.candidate_id)
    assert disp is not None and disp.disposition is Disposition.INCLUDED
    assert not [c for c in report.candidates if c.kind is CandidateKind.PERSON]
    assert report.is_complete


# --------------------------------------------------------------------------- #
# 3. A binding institutional rule that the world omits blocks simulation.
# --------------------------------------------------------------------------- #


def test_case3_omitted_binding_rule_blocks() -> None:
    # Evidence: the charter requires UNANIMOUS consent; the compiled world uses a plain
    # majority rule and never represents the unanimity requirement.
    rule_claim = _claim(
        "charter_rule",
        "The board charter requires unanimous consent of all members to adopt any measure.",
        "unanimous_consent",
        ["the charter"],
    )
    corpus = named_committee(
        ["Ada North", "Ben East", "Cara West"],
        represented=["Ada North", "Ben East", "Cara West"],
        rule_kind="majority",  # WRONG: evidence says unanimous
        extra_claims=[rule_claim],
    )
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    missing = " ".join(exc.value.details.get("missing_material_candidates", []))  # type: ignore[union-attr]
    assert "unanimous" in missing.lower() or "charter" in missing.lower()


def test_case3_represented_rule_passes() -> None:
    # Same unanimity rule, but the world now models it as a unanimous decision rule.
    rule_claim = _claim(
        "charter_rule",
        "The board charter requires unanimous consent of all members to adopt any measure.",
        "unanimous_consent",
        ["the charter"],
    )
    corpus = named_committee(
        ["Ada North", "Ben East", "Cara West"],
        represented=["Ada North", "Ben East", "Cara West"],
        rule_kind="unanimous",
        extra_claims=[rule_claim],
    )
    compiled = compile_dict(corpus)
    assert compiled.coverage_report.is_complete


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
    corpus = named_committee(
        ["Ada North", "Ben East", "Cara West"],
        represented=["Ada North", "Ben East", "Cara West"],
        extra_claims=[event_claim],  # frame ignores it entirely
    )
    with pytest.raises(WorldIntegrityError) as exc:
        compile_dict(corpus)
    missing = " ".join(exc.value.details.get("missing_material_candidates", []))  # type: ignore[union-attr]
    assert "scheduled_event" in missing


# --------------------------------------------------------------------------- #
# 5. A relevant document must be accessible to the actors (causal-use).
# --------------------------------------------------------------------------- #


def _document_corpus(*, accessible: bool) -> dict[str, Any]:
    doc_claim = _claim(
        "brief_1",
        "The confidential staff briefing report sets out the decisive guidance for the vote.",
        "guidance_brief",
        ["the staff briefing"],
    )
    people = ["Ada North", "Ben East", "Cara West"]
    corpus = named_committee(
        people,
        represented=people,
        extra_claims=[doc_claim],
        extra_required=[
            {
                "key": "guidance_document",
                "description": "the decisive staff briefing",
                "evidence_claim_ids": ["brief_1"],
            }
        ],
    )
    if accessible:
        # The chair actually holds the briefing: it enters their memory (an actor view).
        corpus["reality"]["members"][0]["memory_seeds"].append(
            {
                "content": "I have read the staff briefing.",
                "kind": "episodic",
                "importance": 0.8,
                "valid_time": ROSTER_PUB,
                "evidence_claim_ids": ["brief_1"],
            }
        )
    return corpus


def test_case5_inaccessible_document_blocks() -> None:
    with pytest.raises(WorldIntegrityError):
        compile_dict(_document_corpus(accessible=False))


def test_case5_accessible_document_is_covered() -> None:
    compiled = compile_dict(_document_corpus(accessible=True))
    report = compiled.coverage_report
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
    people = ["Ada North", "Ben East", "Cara West"]
    compiled = compile_dict(named_committee(people, represented=people, extra_claims=[gossip]))
    report = compiled.coverage_report
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
    corpus = named_committee(["Ada North", "Ben East"], represented=["Ada North", "Ben East"])
    # Two more independent claims about Ada from different events.
    corpus["sources"].append(
        _source(
            [
                _claim("ada_2", "Ada North chairs the board.", "chair", ["Ada North"]),
                _claim("ada_3", "Ada North favored a hold previously.", "hold", ["Ada North"]),
            ],
            sid="src2",
            lineage="ev2",
        )
    )
    candidates, _ = _inventory(corpus)
    ada = [c for c in candidates if c.canonical_identity == "Ada North"]
    assert len(ada) == 1  # one canonical person, not three
    assert {"r_ada_north", "ada_2", "ada_3"} <= set(ada[0].claim_ids)


def test_case7_alias_candidates_merge_to_one_object() -> None:
    inst = WorldObject(
        object_id="inst",
        kind="institution",
        name="Central Bank",
        claim_ids=("k1",),
        wired=True,
        uses=("membership", "terminal"),
    )
    spec = WorldSpecView(objects=(inst,), decision_body="Central Bank")
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
# 8. Conflicting evidence yields UNCERTAIN / REQUIRED_BUT_UNRESOLVED, not a
#    silent arbitrary choice — and it blocks.
# --------------------------------------------------------------------------- #


def test_case8_conflicting_evidence_is_not_silently_resolved() -> None:
    corpus = named_committee(["Ada North", "Ben East"], represented=["Ada North", "Ben East"])
    corpus["sources"].append(
        _source(
            [
                _claim("hawk", "Kim Vale favors a rate hike.", "hike", ["Kim Vale"], claim_key="vale_stance"),
                _claim("dove", "Kim Vale favors a rate cut.", "cut", ["Kim Vale"], claim_key="vale_stance"),
            ],
            sid="conflict",
            lineage="ev_conf",
        )
    )
    corpus["contradictions"] = [["hawk", "dove"]]
    candidates, _ = _inventory(corpus)
    kim = next(c for c in candidates if c.canonical_identity == "Kim Vale")
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
# 9. The same coverage system works across process types.
# --------------------------------------------------------------------------- #


def test_case9_coverage_runs_for_every_process_type() -> None:
    corpora = {
        "committee": base_corpus(),
        "individual_response": synthetic_corpus("individual_response"),
        "population_strata": synthetic_corpus("population_strata"),
        "seven_member_council": synthetic_corpus("seven_member_council"),
        "data_shock_board": synthetic_corpus("data_shock_board"),
    }
    for name, corpus in corpora.items():
        compiled = compile_dict(corpus)
        report = compiled.coverage_report
        assert report.is_complete, f"{name} coverage incomplete"
        # Every candidate carries a disposition regardless of process shape.
        assert len(report.dispositions) == report.total_candidates


# --------------------------------------------------------------------------- #
# 10. No production candidate is ever silently dropped: one disposition each.
# --------------------------------------------------------------------------- #


def test_case10_every_candidate_gets_exactly_one_disposition() -> None:
    people = ["Vera Nolan", "Jon Alder", "Gala Reyes"]
    compiled = compile_dict(named_committee(people, represented=people))
    report = compiled.coverage_report
    covered_ids = [d.candidate_id for d in report.dispositions]
    assert len(covered_ids) == len(set(covered_ids))  # no double-disposition
    assert set(covered_ids) == {c.candidate_id for c in report.candidates}  # none dropped
    assert report.total_candidates == len(report.candidates)
