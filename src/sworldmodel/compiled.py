"""The compiled world container: a verified base world plus the compiled program.

Produced by a world compiler (LLM live, or an authored corpus offline) and consumed by
the one universal :mod:`engine`. Nothing here is question-specific."""

from __future__ import annotations

from dataclasses import dataclass

from .coverage import CompilationCoverageReport
from .gateway import GatewayResponse
from .models import RealityManifest, UncertaintyVariable
from .uncertainty import ScenarioSet
from .world import WorldState
from .worldspec import WorldSpec


@dataclass(frozen=True)
class CompiledWorld:
    """A verified base world, the compiled program that will be executed, and the two
    integrity records that permitted the run: the reality manifest and the
    evidence-to-world coverage report for *this exact* :class:`WorldSpec`."""

    base_world: WorldState
    spec: WorldSpec
    scenario_set: ScenarioSet
    manifest: RealityManifest
    coverage_report: CompilationCoverageReport
    uncertainty_variables: tuple[UncertaintyVariable, ...] = ()
    compile_responses: tuple[GatewayResponse, ...] = ()
