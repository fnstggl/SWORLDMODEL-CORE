"""The compiled world container: a verified base world plus the compiled program.

Produced by a world compiler (LLM live, or an authored corpus offline) and consumed by
the one universal :mod:`engine`. Nothing here is question-specific."""

from __future__ import annotations

from dataclasses import dataclass

from .gateway import GatewayResponse
from .models import RealityManifest, UncertaintyVariable
from .uncertainty import ScenarioSet
from .world import WorldState
from .worldspec import WorldSpec


@dataclass(frozen=True)
class CompiledWorld:
    base_world: WorldState
    spec: WorldSpec
    scenario_set: ScenarioSet
    manifest: RealityManifest
    uncertainty_variables: tuple[UncertaintyVariable, ...] = ()
    compile_responses: tuple[GatewayResponse, ...] = ()
