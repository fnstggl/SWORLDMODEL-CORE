"""The model gateway: the single, swappable boundary to a reasoning model.

The kernel only talks to a model through :class:`ModelGateway`, which owns nothing but
the boundary and its accounting (calls, tokens, latency, retries, failures, per-stage
counts).

**No implementation lives here.** The production path uses
:class:`~sworldmodel.deepseek_gateway.DeepSeekGateway`; deterministic and scripted
stand-ins live under ``tests/`` where they cannot be reached from a live run. A
deterministic reasoner sitting in the production package is how a simulation ends up
being driven by hand-written rules while reporting that actors decided, so it is not
allowed to sit here — not even as a fallback, and not even labeled as one.

A gateway failure is a real failure: it raises, and the affected branch's mass stays
unresolved. It is never converted into a default action or a prior.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from typing import Any

from .errors import GatewayError


@dataclass(frozen=True)
class GatewayRequest:
    task_kind: str
    prompt: str
    context: dict[str, Any]
    seed: int
    temperature: float = 0.0
    expected_keys: tuple[str, ...] = ()


@dataclass
class GatewayResponse:
    task_kind: str
    data: dict[str, Any]
    raw_text: str
    model: str
    params: dict[str, Any]
    seed: int
    prompt_hash: str
    tokens_in: int
    tokens_out: int
    retries: int = 0
    validation_failures: tuple[str, ...] = ()
    latency_ms: int = 0


class ModelGateway(abc.ABC):
    """Base gateway with thread-safe call/token/latency accounting."""

    is_live: bool = False

    def __init__(self) -> None:
        self.call_count: int = 0
        self.total_tokens: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.retries: int = 0
        self.failed_calls: int = 0
        self.calls: list[GatewayResponse] = []
        self.stage_calls: dict[str, int] = {}
        self.latencies_ms: list[int] = []
        self._lock = threading.Lock()
        # Per-run ceilings. None means unbounded, which is the default so directly
        # constructed gateways (tests, tools) are never throttled; a run threads its
        # configured caps in through set_budget.
        self._max_calls: int | None = None
        self._max_tokens_total: int | None = None

    def set_budget(
        self, *, max_calls: int | None = None, max_tokens_total: int | None = None
    ) -> None:
        """Install per-run call/token ceilings. ``None`` leaves a ceiling unbounded.

        Wall clocks alone cannot stop a run that is spending fast: a pathological
        loop of cheap calls stays under every deadline while burning the provider
        budget. Exhaustion raises :class:`GatewayError` from :meth:`generate`, which
        every existing handler already converts into an honest refusal or an
        unresolved branch — never into a default answer.
        """

        with self._lock:
            self._max_calls = max_calls
            self._max_tokens_total = max_tokens_total
            # The ceilings are PER RUN: a caller reusing one gateway across runs must
            # not have run two refused at its first call because run one spent the
            # lifetime counters. Snapshot here and compare deltas.
            self._budget_base_calls = self.call_count
            self._budget_base_tokens = self.total_tokens

    def _check_budget(self, request: GatewayRequest) -> None:
        with self._lock:
            spent_calls = self.call_count - getattr(self, "_budget_base_calls", 0)
            if self._max_calls is not None and spent_calls >= self._max_calls:
                raise GatewayError(
                    f"call budget exhausted: {spent_calls} model calls made, cap "
                    f"{self._max_calls}; refusing {request.task_kind!r} rather than "
                    "spending past the configured ceiling"
                )
            spent_tokens = self.total_tokens - getattr(self, "_budget_base_tokens", 0)
            if self._max_tokens_total is not None and spent_tokens >= self._max_tokens_total:
                raise GatewayError(
                    f"token budget exhausted: {spent_tokens} tokens used, cap "
                    f"{self._max_tokens_total}; refusing {request.task_kind!r} rather "
                    "than spending past the configured ceiling"
                )

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @abc.abstractmethod
    def _generate(self, request: GatewayRequest) -> GatewayResponse: ...

    def generate(self, request: GatewayRequest) -> GatewayResponse:
        self._check_budget(request)
        response = self._generate(request)
        with self._lock:
            self.call_count += 1
            self.total_tokens += response.tokens_in + response.tokens_out
            self.total_tokens_in += response.tokens_in
            self.total_tokens_out += response.tokens_out
            self.retries += response.retries
            self.calls.append(response)
            self.stage_calls[response.task_kind] = self.stage_calls.get(response.task_kind, 0) + 1
            if response.latency_ms:
                self.latencies_ms.append(response.latency_ms)
        return response

    def note_failure(self) -> None:
        with self._lock:
            self.failed_calls += 1

    def stage_call_counts(self) -> dict[str, int]:
        return dict(self.stage_calls)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
