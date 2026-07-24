"""SWORLDMODEL-CORE: a reality-first social world model.

Forecasts come only from simulated actor trajectories inside a verified, persistent
world. The public surface is intentionally tiny: build a :class:`ForecastConfig` and
call :func:`forecast`.
"""

from __future__ import annotations

from .api import forecast, run_forecast
from .config import ForecastConfig
from .deepseek_gateway import DeepSeekGateway
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
from .gateway import DeterministicGateway, ModelGateway, ScriptedGateway
from .http import FakeTransport, UrllibTransport
from .live_research import LiveResearchBackend, ResearchBudget
from .models import ForecastResult, ForecastStatus, ResolutionContract
from .research import (
    CorpusResearchBackend,
    MockResearchBackend,
    ResearchBundle,
    build_bundle_from_dict,
)

__all__ = [
    "forecast",
    "run_forecast",
    "ForecastConfig",
    "ForecastResult",
    "ForecastStatus",
    "ResolutionContract",
    "DeterministicGateway",
    "ScriptedGateway",
    "ModelGateway",
    "DeepSeekGateway",
    "LiveResearchBackend",
    "ResearchBudget",
    "UrllibTransport",
    "FakeTransport",
    "CorpusResearchBackend",
    "MockResearchBackend",
    "ResearchBundle",
    "build_bundle_from_dict",
    "SWorldModelError",
    "WorldIntegrityError",
    "CutoffViolationError",
    "ContractMutationError",
    "IntentValidationError",
    "MassConservationError",
    "EvidenceError",
    "GatewayError",
]

__version__ = "0.1.0"
