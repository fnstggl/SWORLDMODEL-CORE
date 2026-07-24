"""Shared test builders for the universal engine.

``base_corpus`` is a small committee-shaped WorldSpec — but the committee-ness lives
entirely in the *data*; the runtime that executes it is domain-free. Individual tests
mutate the corpus to craft edge cases (missing participant, duplicated name, unverified
roster, ...). Everything here is synthetic; there are no Banxico facts.
"""

from __future__ import annotations

import copy
import json
import pathlib
from datetime import datetime
from typing import Any

from _worlds import AS_OF, HORIZON, committee_world, run_corpus  # noqa: F401 (re-export)
from sworldmodel import MockResearchBackend
from sworldmodel.api import _build_contract
from sworldmodel.compiled import CompiledWorld
from sworldmodel.world_compiler import compile_world

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BANXICO_CORPUS = REPO_ROOT / "evaluation" / "banxico" / "corpus" / "corpus.json"
SYNTHETIC_ROOT = REPO_ROOT / "evaluation" / "synthetic"


def base_corpus() -> dict[str, Any]:
    """A minimal 3-seat committee: all three lean hold; target = unanimous hold.

    Calm branch -> unanimous hold (YES); a decisive shock moves every seat to cut (NO).
    """

    return committee_world({"a": "hold", "b": "hold", "c": "hold"}, target="hold")


def dup(corpus: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(corpus)


def run_dict(corpus: dict[str, Any], *, seed: int = 0, max_branches: int = 24) -> Any:
    """Run the full pipeline on a corpus dict. Returns (result, ctx)."""

    return run_corpus(corpus, seed=seed, max_branches=max_branches)


def compile_dict(corpus: dict[str, Any], *, gateway: Any = None) -> CompiledWorld:
    """Compile a corpus into a CompiledWorld (runs the reality-integrity gate)."""

    backend = MockResearchBackend(data=corpus)
    bundle = backend.research("q", datetime.fromisoformat(AS_OF), datetime.fromisoformat(HORIZON))
    as_of = bundle.as_of or datetime.fromisoformat(AS_OF)
    contract = _build_contract("q", as_of, bundle.horizon, bundle)
    evidence_view = bundle.evidence_store.view(as_of)
    return compile_world(
        contract, evidence_view, bundle.spec, bundle.uncertainties, bundle.world_facts, seed=0
    )


def banxico_corpus() -> dict[str, Any]:
    return json.loads(BANXICO_CORPUS.read_text())


def synthetic_corpus(name: str) -> dict[str, Any]:
    return json.loads((SYNTHETIC_ROOT / name / "corpus.json").read_text())
