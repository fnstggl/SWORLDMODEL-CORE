"""Evidence research and world grounding.

More detailed simulation is harmful when the underlying information is false, so this
is one of the most important stages. A research backend produces the *complete*
structured evidence store (never a truncated string) plus an evidence-grounded
:class:`ScenarioFrame`, and the load-bearing reality inputs (roster, roles, prior
actions, decision rule, deadline).

Two backends ship here:
* :class:`CorpusResearchBackend` — reads a document corpus of real, dated sources and
  materializes cited claims, runs event-level lineage de-duplication and contradiction
  detection, and builds the research plan by working *backward* from the outcome.
* :class:`MockResearchBackend` — an in-memory bundle for unit tests.

A live web/retrieval backend would implement the same ``research`` interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .evidence import EvidenceClaim, EvidenceStore, mark_contradiction
from .models import (
    AuthorityLevel,
    BranchWeight,
    DecisionRule,
    EpistemicType,
    MemorySeed,
    ReactionRule,
    RequiredRealityFact,
    ScenarioFrame,
    SignalDef,
    SourceType,
    TerminalSpec,
    UncertaintyOutcome,
    UncertaintySpec,
    WeightProvenance,
)
from .world import WorldFact


@dataclass(frozen=True)
class MemberSpec:
    actor_id: str
    name: str
    role: str
    is_voting_seat: bool
    vote_power: int
    prior_action: str | None
    authority: tuple[str, ...]
    memory_seeds: tuple[MemorySeed, ...]
    evidence_claim_ids: tuple[str, ...]
    stable_identity: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResearchBundle:
    evidence_store: EvidenceStore
    frame: ScenarioFrame
    members: tuple[MemberSpec, ...]
    institution_id: str
    institution_name: str
    decision_rule: DecisionRule
    expected_voting_seats: int
    world_facts: tuple[WorldFact, ...]
    target_option: str
    terminal_spec: TerminalSpec
    decision_body: str
    subject_entity: str
    resolution_units: str
    authoritative_sources: tuple[str, ...]
    required_reality_facts: tuple[RequiredRealityFact, ...]
    horizon: datetime
    research_plan: tuple[str, ...]
    as_of: datetime | None = None  # cutoff declared by the corpus, if any
    reference_class: dict[str, str] | None = None  # labeled diagnostic only
    outcome: dict[str, Any] | None = None  # post-cutoff; never used by the forecast


class ResearchBackend(Protocol):
    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle: ...


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


BACKWARD_PLAN = (
    "terminal decision",
    "<- final votes of every seat",
    "<- final proposal on the table",
    "<- deliberation and delivered statements",
    "<- initial member positions",
    "<- prior votes and guidance",
    "<- staff analysis and incoming external data",
)


def build_bundle_from_dict(data: dict[str, Any]) -> ResearchBundle:
    """Materialize a :class:`ResearchBundle` from a corpus dict.

    Runs real lineage grouping and contradiction detection on the claims; both are
    behavior the forecast depends on, not decoration.
    """

    store = EvidenceStore()
    claim_keys: dict[str, str] = {}
    for src in data["sources"]:
        s_type = SourceType(src["source_type"])
        authority = AuthorityLevel(int(src["authority_level"]))
        published = _dt(src["published_at"])
        available = _dt(src.get("available_at") or src["published_at"])
        retrieved = _dt(src.get("retrieved_at") or src["published_at"])
        assert published is not None and available is not None and retrieved is not None
        lineage = src["lineage_event_id"]
        for claim in src["claims"]:
            if "claim_key" in claim:
                claim_keys[claim["id"]] = str(claim["claim_key"])
            store.add(
                EvidenceClaim(
                    id=claim["id"],
                    proposition=claim["proposition"],
                    normalized_value=str(claim["normalized_value"]),
                    entities=tuple(claim.get("entities", [])),
                    valid_from=_dt(claim.get("valid_from")),
                    valid_until=_dt(claim.get("valid_until")),
                    published_at=published,
                    available_at=available,
                    source_id=src["source_id"],
                    source_url=src.get("url", ""),
                    source_title=src.get("title", ""),
                    source_type=s_type,
                    authority_level=authority,
                    supporting_excerpt=claim.get("supporting_excerpt", ""),
                    lineage_event_id=lineage,
                    epistemic_type=EpistemicType(claim.get("epistemic_type", "observation")),
                    confidence=float(claim.get("confidence", 0.9)),
                    retrieved_at=retrieved,
                    contradiction_ids=tuple(claim.get("contradiction_ids", [])),
                )
            )

    _apply_contradictions(store, data.get("contradictions", []), claim_keys)

    frame = _build_frame(data["frame"])
    reality = data["reality"]
    members = tuple(_member(m) for m in reality["members"])
    rule = reality["decision_rule"]
    decision_rule = DecisionRule(
        kind=rule["kind"],
        total_seats=int(rule["total_seats"]),
        threshold=int(rule["threshold"]),
        evidence_claim_ids=tuple(rule.get("evidence_claim_ids", [])),
    )
    terminal = reality["terminal"]
    terminal_spec = TerminalSpec(
        mechanism=terminal["mechanism"],
        yes_condition=terminal["yes_condition"],
        target_option=terminal["target_option"],
        k=terminal.get("k"),
        evidence_claim_ids=tuple(terminal.get("evidence_claim_ids", [])),
    )
    world_facts = tuple(
        WorldFact(
            fact_id=f"fact_{i}",
            text=wf["text"],
            evidence_claim_ids=tuple(wf.get("evidence_claim_ids", [])),
            available_at=_dt(wf["available_at"]),  # type: ignore[arg-type]
            epistemic_type=EpistemicType(wf.get("epistemic_type", "observation")),
        )
        for i, wf in enumerate(data.get("world_facts", []))
    )
    required = tuple(
        RequiredRealityFact(
            key=rf["key"],
            description=rf["description"],
            evidence_claim_ids=tuple(rf.get("evidence_claim_ids", [])),
        )
        for rf in data.get("required_reality_facts", [])
    )
    horizon = _dt(reality["horizon"])
    assert horizon is not None
    return ResearchBundle(
        evidence_store=store,
        frame=frame,
        members=members,
        institution_id=reality["institution_id"],
        institution_name=reality["institution_name"],
        decision_rule=decision_rule,
        expected_voting_seats=int(reality["expected_voting_seats"]),
        world_facts=world_facts,
        target_option=reality["target_option"],
        terminal_spec=terminal_spec,
        decision_body=reality["decision_body"],
        subject_entity=reality["subject_entity"],
        resolution_units=reality["resolution_units"],
        authoritative_sources=tuple(reality.get("authoritative_sources", [])),
        required_reality_facts=required,
        horizon=horizon,
        research_plan=BACKWARD_PLAN,
        as_of=_dt(reality.get("as_of")),
        reference_class=data.get("reference_class"),
        outcome=data.get("outcome"),
    )


def _member(m: dict[str, Any]) -> MemberSpec:
    seeds = tuple(
        MemorySeed(
            content=s["content"],
            kind=s.get("kind", "episodic"),
            importance=float(s.get("importance", 0.6)),
            valid_time=_dt(s.get("valid_time")),
            evidence_claim_ids=tuple(s.get("evidence_claim_ids", [])),
            tags=tuple(s.get("tags", [])),
        )
        for s in m.get("memory_seeds", [])
    )
    return MemberSpec(
        actor_id=m["actor_id"],
        name=m["name"],
        role=m["role"],
        is_voting_seat=bool(m.get("is_voting_seat", True)),
        vote_power=int(m.get("vote_power", 1)),
        prior_action=m.get("prior_action"),
        authority=tuple(m.get("authority", ["vote"])),
        memory_seeds=seeds,
        evidence_claim_ids=tuple(m.get("evidence_claim_ids", [])),
        stable_identity=tuple((k, str(v)) for k, v in m.get("stable_identity", {}).items()),
    )


def _build_frame(f: dict[str, Any]) -> ScenarioFrame:
    signals = tuple(
        SignalDef(
            name=s["name"],
            baseline=float(s.get("baseline", 0.0)),
            description=s.get("description", ""),
            evidence_claim_ids=tuple(s.get("evidence_claim_ids", [])),
        )
        for s in f.get("signals", [])
    )
    rules = tuple(
        ReactionRule(
            trigger_signal=r["trigger_signal"],
            direction=r["direction"],
            threshold=float(r["threshold"]),
            moves_to_option=r["moves_to_option"],
            rationale=r.get("rationale", ""),
            evidence_claim_ids=tuple(r.get("evidence_claim_ids", [])),
        )
        for r in f.get("reaction_rules", [])
    )
    uncertainty = tuple(_uncertainty(u) for u in f.get("uncertainty", []))
    return ScenarioFrame(
        options=tuple(f["options"]),
        signals=signals,
        reaction_rules=rules,
        guidance_option=f.get("guidance_option"),
        guidance_text=f.get("guidance_text", ""),
        acceptance_tolerance=float(f.get("acceptance_tolerance", 0.5)),
        uncertainty=uncertainty,
        guidance_evidence_ids=tuple(f.get("guidance_evidence_ids", [])),
    )


def _uncertainty(u: dict[str, Any]) -> UncertaintySpec:
    outcomes = tuple(
        UncertaintyOutcome(
            value=o["value"],
            weight=BranchWeight(
                value=float(o["weight"]),
                provenance=WeightProvenance(o["provenance"]),
                source_detail=o.get("source_detail", ""),
            ),
            signal_effects=tuple((k, float(v)) for k, v in o.get("signal_effects", [])),
            description=o.get("description", ""),
        )
        for o in u["outcomes"]
    )
    return UncertaintySpec(
        signal=u["signal"],
        why_unknown=u.get("why_unknown", ""),
        reversal_capable=bool(u.get("reversal_capable", True)),
        outcomes=outcomes,
        constraining_evidence_ids=tuple(u.get("constraining_evidence_ids", [])),
    )


def _apply_contradictions(
    store: EvidenceStore, explicit: list[list[str]], claim_keys: dict[str, str]
) -> None:
    """Record decisive contradictions.

    Detection is deliberately *precise*, not fuzzy: (1) corpus-declared pairs, and
    (2) claims that share an explicit ``claim_key`` (i.e. assert the same specific
    fact) but disagree on ``normalized_value``. Prefix/topic heuristics are avoided
    because they cause false refusals (e.g. two officers "office:" of the same body).
    """

    pairs: set[tuple[str, str]] = set()
    for pair in explicit:
        pairs.add((pair[0], pair[1]))

    by_key: dict[str, list[str]] = {}
    for cid, key in claim_keys.items():
        by_key.setdefault(key, []).append(cid)
    for ids in by_key.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = store.get(ids[i]), store.get(ids[j])
                if a.normalized_value != b.normalized_value:
                    pairs.add((a.id, b.id))

    for a_id, b_id in pairs:
        na, nb = mark_contradiction(store.get(a_id), store.get(b_id))
        store.claims[na.id] = na
        store.claims[nb.id] = nb


class CorpusResearchBackend:
    """Reads a corpus directory containing ``corpus.json``."""

    def __init__(self, corpus_dir: str | Path) -> None:
        self.corpus_dir = Path(corpus_dir)

    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle:
        path = self.corpus_dir / "corpus.json"
        data = json.loads(path.read_text())
        return build_bundle_from_dict(data)


class MockResearchBackend:
    """Wraps a pre-built bundle (or a corpus dict) for deterministic unit tests."""

    def __init__(
        self, bundle: ResearchBundle | None = None, *, data: dict[str, Any] | None = None
    ) -> None:
        if bundle is None and data is None:
            raise ValueError("MockResearchBackend needs a bundle or data")
        self._bundle = bundle if bundle is not None else build_bundle_from_dict(data)  # type: ignore[arg-type]

    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle:
        return self._bundle
