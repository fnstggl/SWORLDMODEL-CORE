"""Every refusal, and every hollow completion, names its own mechanism.

The rule these pin is that ``root_cause`` never falls through to "unclassified" for a
failure the system itself raises, and never reports "none" for a run that finished
without resolving anything. Both readings turn a defect into a clean result: a live
OPEC+ run refused at a participant gate and reported "unclassified", and a live Tesla
run exited zero with unresolved mass 1.0 and reported "none".
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sworldmodel.diagnosis import ROOT_CAUSES, RunDiagnosis
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.ids import canonical_json

AS_OF = datetime.fromisoformat("2026-05-14T00:00:00+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T00:00:00+00:00")
SRC = Path(__file__).resolve().parents[2] / "src" / "sworldmodel"


class _Diagnosis(RunDiagnosis):
    """A diagnosis whose stage sections are supplied rather than derived, so the
    classification rules can be exercised one at a time. The research stages are given
    the shape of a run that found and read its sources, so only the rule under test
    fires."""

    runtime_section: dict[str, Any] = {"ran": False, "reason": "the world never compiled"}

    def runtime(self) -> dict[str, Any]:
        return self.runtime_section

    def discovery(self) -> dict[str, Any]:
        return {"urls_considered_count": 12, "official_domain_urls_found": ["https://x.test/a"]}

    def fetching(self) -> dict[str, Any]:
        return {"fetched_count": 8, "rejected_count": 2, "rejection_reasons": {}}

    def extraction(self) -> dict[str, Any]:
        return {
            "claims_stored": 14,
            "claim_candidates": 20,
            "claims_admissible_at_cutoff": 14,
            "extraction_calls": 8,
            "calls_returning_nothing": 1,
            "verification_rejection_reasons": {},
        }


def _for(failure: BaseException | None, **runtime: Any) -> list[str]:
    d = _Diagnosis(question="q", as_of=AS_OF, horizon=HORIZON, failure=failure)
    if runtime:
        d.runtime_section = runtime
    return [c["cause"] for c in d.root_cause()]


def _gate(code: str) -> WorldIntegrityError:
    return WorldIntegrityError(f"refused: {code}", details={"failure": code, "recompilable": True})


def _raised_failure_codes() -> set[str]:
    """Every ``"failure": "<code>"`` the package can raise."""

    pattern = re.compile(r'"failure":\s*"([a-z_]+)"')
    return {m for path in SRC.glob("*.py") for m in pattern.findall(path.read_text())}


def test_every_failure_the_system_can_raise_is_classified() -> None:
    unclassified = sorted(c for c in _raised_failure_codes() if _for(_gate(c)) == ["unclassified"])
    assert not unclassified, (
        "these gates refuse without naming a mechanism, so their runs report "
        f"'unclassified': {unclassified}"
    )


def test_every_classification_uses_the_declared_vocabulary() -> None:
    for code in sorted(_raised_failure_codes()):
        for cause in _for(_gate(code)):
            assert cause in ROOT_CAUSES, (
                f"{code} classified as {cause!r}, which is not a root cause"
            )


def test_a_completed_run_that_resolved_nothing_is_not_reported_as_clean() -> None:
    causes = _for(
        None,
        ran=True,
        resolved_branches=0,
        branches=2,
        actor_invocations=0,
        event_count=6,
        terminal_producer_lineage={
            "b1": [{"terminal_term": "deliveries_Q3", "unproduced": True}],
        },
    )
    assert causes == ["terminal_never_determined"]

    resolved = _for(None, ran=True, resolved_branches=2, branches=2, actor_invocations=4)
    assert resolved == ["none"]


def test_the_same_failure_about_something_else_is_progress() -> None:
    """A live Bank of England run was stopped after two repair rounds for "no new
    diagnosis" while the compiler was in fact changing the world each time: the orphan
    term moved from bailey_public_stance to bailey_vote. Same code, different world, and
    the second attempt was never made."""

    from sworldmodel.api import _failure_signature

    def orphan(term: str) -> WorldIntegrityError:
        return WorldIntegrityError(
            "the outcome is an input",
            details={
                "failure": "terminal_has_no_producer",
                "recompilable": True,
                "terminal terms with no producer": [term],
                "fields any action can write": [],
            },
        )

    assert _failure_signature(orphan("bailey_vote")) != _failure_signature(
        orphan("bailey_public_stance")
    )
    assert _failure_signature(orphan("bailey_vote")) == _failure_signature(orphan("bailey_vote"))

    # Counts and free text move for reasons that are not a different world.
    noisy = WorldIntegrityError(
        "x", details={"failure": "coverage_incomplete", "material_candidates": 14, "note": "a"}
    )
    quieter = WorldIntegrityError(
        "x", details={"failure": "coverage_incomplete", "material_candidates": 9, "note": "b"}
    )
    assert _failure_signature(noisy) == _failure_signature(quieter)


def test_a_blocked_search_channel_is_named_rather_than_blamed_on_the_compiler() -> None:
    """A pass where half the queries came back "empty result set, block, or challenge"
    compiled empty worlds and was reported as an actor-discovery failure — which points
    at the compiler for something it never saw. A search channel that returns nothing is
    a fact about the channel, and belongs in the record as one.

    The two halves below are the two cases that are genuinely different, and each is
    observable: discovery was WATCHED failing, versus the compiled world simply held no
    producer with nothing observed about discovery either way. The healthy half asserted
    ``actor_discovery_failure`` — discovery language for the case where discovery was
    never observed at all, which is the very confusion the blocked half exists to stop.
    It now asserts what the gate actually found, and additionally that no discovery
    language appears, so a future rename cannot slip past this by swapping the token."""

    blocked = {
        "urls_considered_count": 4,
        "official_domain_urls_found": ["https://x.test/a"],
        "queries_used": 5,
        "search_failures": [
            {"channel": "authoritative", "error": "search returned no result links"},
            {"channel": "authoritative", "error": "search returned no result links"},
            {"channel": "general", "error": "search returned no result links"},
        ],
    }

    class _Blocked(_Diagnosis):
        def discovery(self) -> dict[str, Any]:
            return blocked

    d = _Blocked(question="q", as_of=AS_OF, horizon=HORIZON, failure=_gate("no_causal_producer"))
    causes = [c["cause"] for c in d.root_cause()]
    assert "discovery_failure" in causes
    why = next(c["why"] for c in d.root_cause() if c["cause"] == "discovery_failure")
    assert "3 of 5 searches" in why

    # A run whose searches worked is unaffected, and still reports the gate it stopped at.
    healthy = {**blocked, "search_failures": []}

    class _Healthy(_Diagnosis):
        def discovery(self) -> dict[str, Any]:
            return healthy

    d2 = _Healthy(question="q", as_of=AS_OF, horizon=HORIZON, failure=_gate("no_causal_producer"))
    assert [c["cause"] for c in d2.root_cause()] == ["compiler_omission"]
    assert "discovery" not in canonical_json(d2.root_cause()), (
        "a run that watched its searches succeed must not be filed under any discovery "
        "cause — that is this test's whole point, applied to its own healthy case"
    )


def test_nothing_can_act_with_idle_actors_gets_a_no_research_action_repair() -> None:
    """A live Bank of England run compiled Andrew Bailey as an actor and gave him no
    action — no way to make the very statement the question is about — and repair kept
    re-emptying the world instead of adding his one action. An actor with nothing to do
    is a different, more fixable defect than an empty world, and the repair for it needs
    no research: the action is the thing the actor's role already lets it do."""

    from sworldmodel.repair import plan_repair

    idle = WorldIntegrityError(
        "nothing can act",
        details={
            "failure": "nothing_can_act",
            "recompilable": True,
            "actors_without_actions": ["andrew_bailey"],
        },
    )
    plan = plan_repair(idle, "Will Andrew Bailey signal support for a cut?")
    assert plan is not None
    assert plan.queries == ()  # no research: the action is known from the role
    assert "andrew_bailey" in plan.instruction
    assert "gave them no actions" in plan.instruction

    # The genuinely empty world still gets the research-backed repair.
    empty = WorldIntegrityError(
        "nothing can act",
        details={"failure": "nothing_can_act", "recompilable": True, "actors_without_actions": []},
    )
    empty_plan = plan_repair(empty, "Will OPEC+ raise quotas?")
    assert empty_plan is not None
    assert empty_plan.queries  # research for the missing producer


def test_a_store_with_nothing_admissible_at_the_cutoff_names_the_archive_gap() -> None:
    """A holdout run whose cutoff sat seconds in the past stored claims that all
    postdated as_of, compiled from an empty admissible view, and was filed as
    compiler_omission — pointing at the compiler for a record it never saw.
    archive_coverage_failure was in the vocabulary and never emitted."""

    class _Starved(_Diagnosis):
        def extraction(self) -> dict[str, Any]:
            return {
                "claims_stored": 9,
                "claim_candidates": 12,
                "claims_admissible_at_cutoff": 0,
                "extraction_calls": 8,
                "calls_returning_nothing": 1,
                "verification_rejection_reasons": {},
            }

    d = _Starved(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        failure=_gate("semantic_plan_invalid"),
    )
    causes = [c["cause"] for c in d.root_cause()]
    assert "archive_coverage_failure" in causes
    assert causes.index("archive_coverage_failure") == 0, (
        "the evidence-stage cause must be named before any compiler-stage cause"
    )


def test_research_that_happened_under_an_early_refusal_reaches_the_diagnosis() -> None:
    """A refusal firing before the bundle exists still did its research, and the record
    of it must reach the diagnosis.

    Ten sealed pastcasts refused because ``web.archive.org`` was unreachable — 232 URLs
    discovered, 0 fetched, every source rejected for "no archived capture". All ten
    reported root cause ``compiler_omission`` and stated "this run's research record
    contains no discovery pass at all", beside a ``research_trace.json`` in the same
    directory listing every query and every rejection. The compile-stage raise passed no
    research to the diagnosis, so its counters read zero and it named the one component
    that was working.

    There is deliberately no synthetic ``ResearchBundle`` here: that type requires a
    ``WorldSpec``, and a run refused at compile never produced one. Wrapping a trace in a
    bundle would assert a world that does not exist, which is the shape of defect this
    whole vocabulary exists to prevent.
    """

    trace = {
        "queries": ["fomc september 2024 decision"],
        "attempted_urls": [f"https://news.test/{i}" for i in range(232)],
        "sources_rejected": [
            {"url": f"https://news.test/{i}", "reason": "no archived capture"} for i in range(232)
        ],
        "sources_fetched": [],
    }
    d = RunDiagnosis(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        partial_live_trace=trace,
        failure=_gate("no_causal_producer"),
        failure_stage="compilation",
    )

    # The run's own record, not zeros standing in for it.
    assert d.discovery()["urls_considered_count"] == 232
    assert d.fetching()["rejected_count"] == 232
    assert d.fetching()["fetched_count"] == 0
    # And it can no longer state that no discovery pass ran, because one did.
    rendered = canonical_json(d.root_cause())
    assert "no discovery pass at all" not in rendered
