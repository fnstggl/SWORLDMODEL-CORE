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
