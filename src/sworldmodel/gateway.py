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

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @abc.abstractmethod
    def _generate(self, request: GatewayRequest) -> GatewayResponse: ...

    def generate(self, request: GatewayRequest) -> GatewayResponse:
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
