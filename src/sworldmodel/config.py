"""Run configuration for the single public entry point."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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

    @classmethod
    def live(
        cls,
        *,
        seed: int = 0,
        trace_dir: Path | None = None,
        max_branches: int = 8,
        research_budget: object | None = None,
        transport: object | None = None,
        now: datetime | None = None,
        model: str | None = None,
    ) -> ForecastConfig:
        """Production configuration: live DeepSeek gateway + live research backend.

        Gateway and research share ONE HTTP transport so all network calls are counted
        together. This is the only config whose ``is_live`` is True.
        """

        from .deepseek_gateway import DeepSeekGateway
        from .http import UrllibTransport
        from .live_research import LiveResearchBackend, ResearchBudget

        shared_transport = transport or UrllibTransport()
        gateway = DeepSeekGateway(shared_transport, model=model)  # type: ignore[arg-type]
        backend = LiveResearchBackend(
            gateway,
            shared_transport,  # type: ignore[arg-type]
            budget=research_budget or ResearchBudget(),  # type: ignore[arg-type]
            now=now,
        )
        return cls(
            gateway=gateway,
            research_backend=backend,
            seed=seed,
            trace_dir=trace_dir,
            max_branches=max_branches,
        )

    @property
    def is_live(self) -> bool:
        return getattr(self.gateway, "is_live", False) and getattr(
            self.research_backend, "is_live", False
        )
