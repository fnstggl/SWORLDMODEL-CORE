"""Live DeepSeek gateway (OpenAI-compatible chat completions).

Makes real network calls through the injected :class:`HttpTransport`. Configuration
comes from the environment (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`,
`DEEPSEEK_MODEL`) — no secrets in the repository. It supports structured JSON output,
schema validation with bounded deterministic repair, timeouts, exponential backoff on
transient failures, real token/latency accounting, prompt hashes, and raw-response
preservation. A repeated provider failure raises :class:`GatewayError`, which leaves
the affected world mass explicitly unresolved — never a default action or a prior.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .errors import GatewayError
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .http import HttpError, HttpTransport, UrllibTransport
from .ids import prompt_hash

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# Temperature per task: low for extraction/compilation, higher for actor cognition.
DEFAULT_TEMPERATURES: dict[str, float] = {
    "resolution": 0.1,
    "research_plan": 0.3,
    "followup_queries": 0.4,
    "extract_claims": 0.1,
    "contradiction": 0.1,
    "compile_world_spec": 0.3,
    "interpret_novel": 0.2,
    "actor_decision": 0.7,
    "reflect": 0.5,
}
DEFAULT_MAX_TOKENS: dict[str, int] = {
    "extract_claims": 3000,
    "compile_world_spec": 8000,
    "interpret_novel": 1500,
    "actor_decision": 1500,
}


class DeepSeekGateway(ModelGateway):
    is_live = True

    def __init__(
        self,
        transport: HttpTransport | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 4,
        timeout: float = 90.0,
        temperatures: dict[str, float] | None = None,
        default_max_tokens: int = 2500,
        backoff_base: float = 0.5,
    ) -> None:
        super().__init__()
        self.transport = transport or UrllibTransport()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperatures = {**DEFAULT_TEMPERATURES, **(temperatures or {})}
        self.default_max_tokens = default_max_tokens
        self.backoff_base = backoff_base
        if not self.api_key:
            raise GatewayError("DEEPSEEK_API_KEY is not set; cannot run a live forecast")

    @property
    def model_id(self) -> str:
        return self._model

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _temperature(self, request: GatewayRequest) -> float:
        return self.temperatures.get(request.task_kind, request.temperature or 0.2)

    def _max_tokens(self, request: GatewayRequest) -> int:
        return DEFAULT_MAX_TOKENS.get(request.task_kind, self.default_max_tokens)

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        system = (
            "You are a careful analyst. Reply with a SINGLE valid JSON object and "
            "nothing else. Do not invent facts, sources, or citations. If a value is "
            "unknown, use null. Never restate the instructions."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request.prompt},
        ]
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature(request),
            "max_tokens": self._max_tokens(request),
            "response_format": {"type": "json_object"},
            "seed": request.seed,
        }

        last_error = ""
        retries = 0
        validation_failures: list[str] = []
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = self.transport.post_json(
                    self._endpoint(),
                    body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
            except HttpError as exc:  # transport-level failure: retry transient
                last_error = str(exc)
                retries += 1
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                break
            latency = int((time.monotonic() - start) * 1000)

            if resp.status in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status}: {resp.text[:200]}"
                retries += 1
                if attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                break
            if not resp.ok:
                # 4xx (bad request / auth): non-transient, surface immediately.
                raise GatewayError(
                    f"DeepSeek {resp.status} for {request.task_kind}: {resp.text[:300]}"
                )

            data, tokens_in, tokens_out, content = self._parse(resp.text)
            if data is None or not self._schema_ok(data, request.expected_keys):
                failure = f"malformed/missing-keys on attempt {attempt}"
                validation_failures.append(failure)
                last_error = failure
                retries += 1
                if attempt < self.max_retries:
                    # Deterministic repair: same decision context, stricter instruction.
                    body["messages"] = messages + [
                        {"role": "assistant", "content": content[:2000]},
                        {
                            "role": "user",
                            "content": (
                                "That was not a valid JSON object with the required keys "
                                f"{list(request.expected_keys)}. Return ONLY the corrected JSON object."
                            ),
                        },
                    ]
                    continue
                break

            return GatewayResponse(
                task_kind=request.task_kind,
                data=data,
                raw_text=content,
                model=self._model,
                params={"temperature": body["temperature"], "max_tokens": body["max_tokens"]},
                seed=request.seed,
                prompt_hash=prompt_hash(request.prompt),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                retries=retries,
                validation_failures=tuple(validation_failures),
                latency_ms=latency,
            )

        self.note_failure()
        raise GatewayError(
            f"DeepSeek call for {request.task_kind!r} failed after {retries} retries: {last_error}"
        )

    def _sleep(self, attempt: int) -> None:
        time.sleep(min(self.backoff_base * (2**attempt), 8.0))

    @staticmethod
    def _parse(text: str) -> tuple[dict[str, Any] | None, int, int, str]:
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError:
            return None, 0, 0, text
        usage = envelope.get("usage", {}) or {}
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None, tokens_in, tokens_out, text
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None, tokens_in, tokens_out, content
        if not isinstance(data, dict):
            return None, tokens_in, tokens_out, content
        return data, tokens_in, tokens_out, content

    @staticmethod
    def _schema_ok(data: dict[str, Any], expected_keys: tuple[str, ...]) -> bool:
        return all(k in data for k in expected_keys)
