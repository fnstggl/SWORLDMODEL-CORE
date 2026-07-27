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
import copy
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
    # Prompt tokens the PROVIDER served from its own prefix/context cache (e.g.
    # DeepSeek's automatic context caching). A subset of tokens_in; 0 when the
    # provider reports nothing. Recorded so a run's cache hit rate is measured, not
    # asserted.
    tokens_cached: int = 0


class ModelGateway(abc.ABC):
    """Base gateway with thread-safe call/token/latency accounting."""

    is_live: bool = False

    # Whether byte-identical requests are answered from this gateway's own memo table
    # instead of re-asked. Off in the base class: a scripted test gateway may
    # legitimately script different answers for repeated prompts, and memoizing under
    # it would rewrite the fixture. The live gateway turns it on — an identical
    # (model, task, prompt, seed, temperature) request is the same decision context,
    # and the run's own audit already classifies re-deciding it as spinning.
    memoize_identical_requests: bool = False

    def __init__(self) -> None:
        self.call_count: int = 0
        self.total_tokens: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.total_tokens_cached: int = 0
        self.retries: int = 0
        self.failed_calls: int = 0
        self.memo_hits: int = 0
        self.calls: list[GatewayResponse] = []
        self.stage_calls: dict[str, int] = {}
        self.latencies_ms: list[int] = []
        self._lock = threading.Lock()
        self._memo: dict[tuple[str, str, str, int, float, tuple[str, ...]], GatewayResponse] = {}
        self._inflight: dict[
            tuple[str, str, str, int, float, tuple[str, ...]], threading.Event
        ] = {}
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

    def _memo_key(
        self, request: GatewayRequest
    ) -> tuple[str, str, str, int, float, tuple[str, ...]]:
        return (
            self.model_id,
            request.task_kind,
            request.prompt,
            request.seed,
            request.temperature,
            tuple(request.expected_keys),
        )

    def _memo_reuse(self, cached: GatewayResponse) -> GatewayResponse:
        """Serve a byte-identical request from the memo — the identical decision
        context was already answered in this run, and re-asking it re-spends the
        provider without any new information reaching the model. The reuse is
        recorded (params carries memo_hit) so the call log shows it, and none of the
        spend counters move — nothing was spent."""

        reused = GatewayResponse(
            task_kind=cached.task_kind,
            data=copy.deepcopy(cached.data),
            raw_text=cached.raw_text,
            model=cached.model,
            params={**cached.params, "memo_hit": True},
            seed=cached.seed,
            prompt_hash=cached.prompt_hash,
            tokens_in=cached.tokens_in,
            tokens_out=cached.tokens_out,
            retries=0,
            validation_failures=cached.validation_failures,
            latency_ms=0,
            tokens_cached=cached.tokens_cached,
        )
        with self._lock:
            self.memo_hits += 1
            self.calls.append(reused)
        return reused

    def generate(self, request: GatewayRequest) -> GatewayResponse:
        if not self.memoize_identical_requests:
            return self._generate_accounted(request)
        # Memoization with in-flight deduplication. Without the in-flight wait, two
        # identical requests racing from parallel branches would BOTH reach the
        # provider and could return different completions — so a run's answers would
        # depend on thread timing, which is exactly what serial-versus-parallel
        # equivalence forbids. The second caller waits for the first and reuses its
        # response, the same behavior a serial run gets from the memo.
        key = self._memo_key(request)
        while True:
            with self._lock:
                cached = self._memo.get(key)
                if cached is None and key not in self._inflight:
                    self._inflight[key] = threading.Event()
                    break  # this caller performs the real request
                event = self._inflight.get(key)
            if cached is not None:
                return self._memo_reuse(cached)
            assert event is not None
            event.wait()
            # The in-flight call finished (or failed). Re-check: a success is in the
            # memo; a failure leaves it empty, and this caller takes over the request.
        try:
            response = self._generate_accounted(request)
            with self._lock:
                self._memo[key] = response
            return response
        finally:
            with self._lock:
                pending = self._inflight.pop(key, None)
            if pending is not None:
                pending.set()

    def _generate_accounted(self, request: GatewayRequest) -> GatewayResponse:
        self._check_budget(request)
        response = self._generate(request)
        with self._lock:
            self.call_count += 1
            self.total_tokens += response.tokens_in + response.tokens_out
            self.total_tokens_in += response.tokens_in
            self.total_tokens_out += response.tokens_out
            self.total_tokens_cached += response.tokens_cached
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
