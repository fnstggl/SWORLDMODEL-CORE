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
    # How many *causal structures* may be simulated when the evidence leaves the
    # structure genuinely open. 1 means the compiled structure is taken as given.
    max_structures: int = 3
    run_label: str = "run"
    budget: RunBudget = field(default_factory=RunBudget)
    # A ceiling on the whole compile-and-repair phase. Repair stops when it stops making
    # progress, which is the right rule and has no clock in it: a question whose compiler
    # keeps producing genuinely different worlds can repair for as long as the research
    # backend will feed it. A live OPEC+ run spent forty minutes there and was killed
    # from outside with nothing written. Reaching this stops repairing and refuses with
    # the last gate's own diagnosis, which is a result; being killed is not.
    max_compile_seconds: float = 900.0

    @classmethod
    def live(
        cls,
        *,
        seed: int = 0,
        trace_dir: Path | None = None,
        max_branches: int = 8,
        max_structures: int = 3,
        max_compile_seconds: float = 900.0,
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
            max_structures=max_structures,
            max_compile_seconds=max_compile_seconds,
            budget=budget or RunBudget(),
        )

    @property
    def is_live(self) -> bool:
        return getattr(self.gateway, "is_live", False) and getattr(
            self.research_backend, "is_live", False
        )
