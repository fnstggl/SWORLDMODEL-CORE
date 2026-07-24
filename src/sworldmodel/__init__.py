"""SWORLDMODEL-CORE: a reality-first social world model.

Forecasts come only from simulated actor trajectories inside a verified, persistent
world. The public surface is intentionally tiny: build a :class:`ForecastConfig` and
call :func:`forecast`.
"""

from __future__ import annotations

from .api import forecast, run_forecast
from .config import ForecastConfig
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
