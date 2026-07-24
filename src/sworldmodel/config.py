"""Run configuration for the single public entry point.

There is one production configuration and it is live: a real DeepSeek gateway and the
live research backend, sharing one HTTP transport so every network call is counted
together. There is deliberately no ``offline()`` constructor and no deterministic
gateway to reach — a config that can quietly produce a forecast without touching the
real world is exactly how a simulator starts reporting its own assumptions back to you.

Tests construct :class:`ForecastConfig` directly with fixtures from ``tests/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .engine import RunBudget
from .gateway import ModelGateway
from .research import ResearchBackend


@dataclass
class ForecastConfig:
    gateway: ModelGateway
    research_backend: ResearchBackend
    seed: int = 0
    trace_dir: Path | None = None
    max_branches: int = 8
    run_label: str = "run"
    budget: RunBudget = field(default_factory=RunBudget)

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
        budget: RunBudget | None = None,
    ) -> ForecastConfig:
        """The production configuration: live DeepSeek + live research, one transport."""

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
            budget=budget or RunBudget(),
        )

    @property
    def is_live(self) -> bool:
        return getattr(self.gateway, "is_live", False) and getattr(
            self.research_backend, "is_live", False
        )
