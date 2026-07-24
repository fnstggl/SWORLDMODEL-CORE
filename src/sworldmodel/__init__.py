"""SWORLDMODEL-CORE: a reality-first social world model.

A forecast comes only from simulated actor trajectories inside a verified, persistent
world that was researched and compiled from the question alone. The public surface is
intentionally tiny: build a live :class:`ForecastConfig` and call :func:`forecast`.

Nothing exported here can produce a forecast without the live path.
"""

from __future__ import annotations

from .api import forecast, run_forecast
from .config import ForecastConfig
from .deepseek_gateway import DeepSeekGateway
from .engine import RunBudget
from .errors import (
    ContractMutationError,
    CutoffViolationError,
    EvidenceError,
    GatewayError,
    IntentValidationError,
    MassConservationError,
    SWorldModelError,
    WorldIntegrityError,
)
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .http import UrllibTransport
from .live_research import LiveResearchBackend, ResearchBudget
from .models import ForecastResult, ForecastStatus, ResolutionContract
from .research import ResearchBundle

__all__ = [
    "forecast",
    "run_forecast",
    "ForecastConfig",
    "ForecastResult",
    "ForecastStatus",
    "ResolutionContract",
    "RunBudget",
    "ModelGateway",
    "GatewayRequest",
    "GatewayResponse",
    "DeepSeekGateway",
    "LiveResearchBackend",
    "ResearchBudget",
    "UrllibTransport",
    "ResearchBundle",
    "SWorldModelError",
    "WorldIntegrityError",
    "CutoffViolationError",
    "ContractMutationError",
    "IntentValidationError",
    "MassConservationError",
    "EvidenceError",
    "GatewayError",
]

__version__ = "0.2.0"
