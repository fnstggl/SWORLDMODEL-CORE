"""A roster shortfall must reach the targeted-research repair loop.

Exposed after the merge checkpoint by a live run:

    question : "Will the Federal Reserve announce a reduction in the federal funds
               target range at its September 2026 FOMC meeting?"
    expected : the reality gate's roster shortfall triggers targeted follow-up research
               and a recompile, exactly as a coverage miss does
    actual   : ``WorldIntegrityError: participant roster does not match verified
               reality`` propagated immediately; no follow-up research was attempted
    cause    : ``api._compile_with_repair`` keys repair on the coverage gate's
               ``missing_material_candidates`` detail, and ``verify_reality`` raised
               with only counts, so the loop re-raised instead of repairing
    fix      : ``verify_reality`` now names the shortfall in the same label shape the
               repair loop already parses. No new subsystem, no fabricated participants:
               if research still cannot find them, the gate refuses again.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from _worlds import AS_OF, HORIZON, named_body_world
from sworldmodel import DeterministicGateway, ForecastConfig, run_forecast
from sworldmodel.errors import WorldIntegrityError
from sworldmodel.research import build_bundle_from_dict

PEOPLE = ["Vera Nolan", "Jon Alder", "Gala Reyes", "Omar Castel", "Gabriel Cuadra"]


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _under_represented() -> dict[str, Any]:
    """Evidence names five participants; the compiled world represents two but still
    declares the true size of five — the shape the reality gate must catch."""

    corpus = named_body_world(PEOPLE, represented=PEOPLE[:2])
    corpus["reality"]["expected_participants"] = 5
    return corpus


class _RepairBackend:
    is_live = False

    def __init__(self, first: dict[str, Any], repaired: dict[str, Any]) -> None:
        self._first = first
        self._repaired = repaired
        self.augmented_with: list[str] | None = None

    def research(self, question: str, as_of: Any, horizon: Any) -> Any:
        return build_bundle_from_dict(self._first)

    def augment_for_coverage(
        self, question: str, as_of: Any, horizon: Any, missing: list[str], prior: Any
    ) -> Any:
        self.augmented_with = missing
        return build_bundle_from_dict(self._repaired)


def test_roster_shortfall_is_reported_as_a_repairable_missing_candidate() -> None:
    from sworldmodel import MockResearchBackend
    from sworldmodel.api import _build_contract
    from sworldmodel.world_compiler import compile_world

    backend = MockResearchBackend(data=_under_represented())
    bundle = backend.research("q", _dt(AS_OF), _dt(HORIZON))
    as_of = bundle.as_of or _dt(AS_OF)
    contract = _build_contract("q", as_of, bundle.horizon, bundle)
    with pytest.raises(WorldIntegrityError) as exc:
        compile_world(
            contract,
            bundle.evidence_store.view(as_of),
            bundle.spec,
            bundle.uncertainties,
            bundle.world_facts,
        )
    missing = exc.value.details.get("missing_material_candidates")
    assert isinstance(missing, list) and missing, "the shortfall must be repairable"
    assert "remaining 3 of 5" in missing[0]
    # The label parses with the same rule the coverage repair path already uses.
    from sworldmodel.live_research import _missing_query

    assert _missing_query(missing[0])


def test_roster_shortfall_triggers_targeted_research_and_recompiles() -> None:
    backend = _RepairBackend(_under_represented(), named_body_world(PEOPLE, represented=PEOPLE))
    config = ForecastConfig(
        gateway=DeterministicGateway(), research_backend=backend, max_branches=4
    )
    result, ctx = run_forecast("q", _dt(AS_OF), _dt(HORIZON), config)

    assert backend.augmented_with is not None, "the shortfall never reached the repair loop"
    assert "participants" in " ".join(backend.augmented_with)
    assert result.integrity_manifest.represented_participants == 5
    assert ctx.compiled.coverage_report.is_complete
    assert result.probability_source == "weighted_simulated_trajectories"


def test_unrepairable_shortfall_still_refuses() -> None:
    # A backend whose follow-up research finds nothing new must still be refused: the
    # gate is never satisfied by inventing the missing participants.
    incomplete = _under_represented()
    backend = _RepairBackend(incomplete, incomplete)
    config = ForecastConfig(
        gateway=DeterministicGateway(), research_backend=backend, max_branches=4
    )
    with pytest.raises(WorldIntegrityError):
        run_forecast("q", _dt(AS_OF), _dt(HORIZON), config)


def test_surplus_roster_is_not_treated_as_repairable() -> None:
    # Representing MORE participants than verified reality is not a research gap; it is
    # a false world and must fail without pretending research can fix it.
    from sworldmodel import MockResearchBackend
    from sworldmodel.api import _build_contract
    from sworldmodel.world_compiler import compile_world

    corpus = named_body_world(PEOPLE[:3], represented=PEOPLE[:3])
    corpus["reality"]["expected_participants"] = 2  # world represents 3
    backend = MockResearchBackend(data=corpus)
    bundle = backend.research("q", _dt(AS_OF), _dt(HORIZON))
    as_of = bundle.as_of or _dt(AS_OF)
    contract = _build_contract("q", as_of, bundle.horizon, bundle)
    with pytest.raises(WorldIntegrityError) as exc:
        compile_world(
            contract,
            bundle.evidence_store.view(as_of),
            bundle.spec,
            bundle.uncertainties,
            bundle.world_facts,
        )
    assert "missing_material_candidates" not in exc.value.details
