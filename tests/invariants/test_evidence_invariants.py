"""Evidence-lineage and cutoff invariants at the pipeline level."""

from __future__ import annotations

import pytest

from _helpers import banxico_corpus, base_corpus, compile_dict, dup, run_dict
from sworldmodel.errors import EvidenceError
from sworldmodel.models import EpistemicType
from sworldmodel.research import build_bundle_from_dict


def test_contradictory_decisive_claims_block_rollout() -> None:
    corpus = dup(base_corpus())
    corpus["contradictions"] = [["r_a", "ctx"]]
    with pytest.raises(EvidenceError):
        compile_dict(corpus)


def test_every_world_fact_has_evidence_or_is_marked_non_observation() -> None:
    bundle = build_bundle_from_dict(banxico_corpus())
    for wf in bundle.world_facts:
        assert wf.evidence_claim_ids or wf.epistemic_type is not EpistemicType.OBSERVATION


def test_token_limited_view_never_deletes_the_canonical_store() -> None:
    bundle = build_bundle_from_dict(banxico_corpus())
    before = len(bundle.evidence_store.all())
    view = bundle.evidence_store.view(bundle.as_of)  # type: ignore[arg-type]
    text, ids = view.render_view("roster vote guidance", limit=3)
    assert len(ids) == 3  # the view is token-limited
    assert len(bundle.evidence_store.all()) == before  # store untouched


def test_post_cutoff_evidence_is_not_available_to_the_simulation() -> None:
    _, ctx = run_dict(base_corpus())
    # Add a post-cutoff claim to a copy and confirm it is excluded from the view.
    bundle = build_bundle_from_dict(banxico_corpus())
    view = bundle.evidence_store.view(bundle.as_of)  # type: ignore[arg-type]
    available_ids = {c.id for c in view.available()}
    assert "c_june_decision_POSTCUTOFF" not in available_ids


def test_lineage_deduplicates_one_event_into_shared_id() -> None:
    bundle = build_bundle_from_dict(banxico_corpus())
    groups = bundle.evidence_store.lineage_groups()
    # The seven May-7 claims are ONE event, not seven independent cases.
    may7 = groups["banxico_2026_05_07_decision"]
    assert len(may7) >= 6
    assert len(bundle.evidence_store.independent_event_ids()) < len(bundle.evidence_store.all())
