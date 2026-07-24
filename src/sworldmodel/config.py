"""Run configuration for the single public entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .gateway import DeterministicGateway, ModelGateway
from .research import ResearchBackend


@dataclass
class ForecastConfig:
    gateway: ModelGateway
    research_backend: ResearchBackend
    seed: int = 0
    trace_dir: Path | None = None
    max_branches: int = 24
    include_reference_class_diagnostic: bool = True
    run_label: str = "run"

    @classmethod
    def offline(cls, research_backend: ResearchBackend, **kw: object) -> ForecastConfig:
        """Deterministic, network-free configuration (offline reasoner)."""

        return cls(gateway=DeterministicGateway(), research_backend=research_backend, **kw)  # type: ignore[arg-type]
