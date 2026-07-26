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
import random
import time
from typing import Any

from .errors import GatewayError
from .gateway import GatewayRequest, GatewayResponse, ModelGateway
from .http import HttpError, HttpTransport, UrllibTransport
from .ids import prompt_hash
from .jsonsalvage import salvage_json

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
    "semantic_plan": 0.3,
    "semantic_review": 0.2,
    "interpret_novel": 0.2,
    "exclusion_challenge": 0.1,
    "world_review": 0.2,
    "trajectory_audit": 0.2,
    "actor_decision": 0.7,
    "reflect": 0.5,
}
# Output budgets per stage. These must cover the model's *reasoning* tokens as well
# as its content: this provider counts both against `max_tokens`, so a budget sized
# only for the expected JSON silently returns an empty or truncated body.
DEFAULT_MAX_TOKENS: dict[str, int] = {
    "extract_claims": 4000,
    # The world-compile stage emits large nested JSON: entities, actors with grounded
    # plans, actions with timing and effects, a dated process graph, external
    # processes, wake rules, uncertainties and a terminal expression.
    "compile_world_spec": 16000,
    # The semantic plan is meaning without syntax — smaller than a WorldSpec, but it
    # still enumerates entities, states, affordances, processes and uncertainties, and
    # the provider's reasoning tokens count against the same budget.
    "semantic_plan": 12000,
    # One verdict plus per-finding reasons and exact corrections.
    "semantic_review": 4000,
    "interpret_novel": 2000,
    "exclusion_challenge": 800,
    # Fourteen adversarial findings with a sentence of attack and a sentence of
    # evidence basis each, plus the model's own thinking.
    "world_review": 6000,
    # Six trajectory judgements over a per-branch event digest, plus thinking.
    "trajectory_audit": 4000,
    # An actor returns its plan disposition, plan update, intention, information needs,
    # commitments and revisit conditions — considerably more than a bare action.
    "actor_decision": 3000,
    "reflect": 2000,
}


# Failures that mean the ENDPOINT is unreachable rather than one request failing.
_OUTAGE_MARKERS = (
    "connection refused",
    "connection reset",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "no route to host",
)


def _is_outage(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _OUTAGE_MARKERS)


def _failure_kind(exc: Exception) -> str:
    """Name the failure class so the record distinguishes what actually went wrong."""

    text = str(exc).lower()
    if _is_outage(exc):
        return "endpoint unreachable"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    return "transport failure"


def _retry_after(headers: dict[str, str]) -> float:
    """A server that says when to come back is answered on its schedule, capped."""

    raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), "")
    try:
        return min(float(raw), 60.0)
    except (TypeError, ValueError):
        return 0.0


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
        default_max_tokens: int = 3000,
        max_output_tokens: int = 32000,
        backoff_base: float = 0.5,
        outage_patience_seconds: float = 180.0,
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
        self.max_output_tokens = max_output_tokens
        self.backoff_base = backoff_base
        # How long to keep retrying when the *endpoint* is down (connection refused,
        # reset, DNS failure) rather than a single request failing. A live Banxico run
        # met a refused connection, spent 7.5 seconds of backoff, and reported failed
        # research — for an outage that had passed by the time anyone read the log.
        # Outages are minutes-shaped; per-request failures are seconds-shaped.
        self.outage_patience_seconds = outage_patience_seconds
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
        outage_deadline: float | None = None
        attempt = -1
        while True:
            attempt += 1
            start = time.monotonic()
            try:
                resp = self.transport.post_json(
                    self._endpoint(),
                    body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
            except HttpError as exc:  # transport-level failure
                last_error = f"{_failure_kind(exc)}: {exc}"
                retries += 1
                if _is_outage(exc):
                    # The endpoint is down, not this request failing. Outages last
                    # minutes; keep retrying with capped backoff until the patience
                    # budget is spent, so a two-minute blip does not end a forty-minute
                    # run.
                    if outage_deadline is None:
                        outage_deadline = time.monotonic() + self.outage_patience_seconds
                    if time.monotonic() < outage_deadline:
                        self._sleep(attempt, ceiling=30.0)
                        continue
                elif attempt < self.max_retries:
                    self._sleep(attempt)
                    continue
                break
            latency = int((time.monotonic() - start) * 1000)

            if resp.status in (429, 500, 502, 503, 504):
                kind = "rate limited" if resp.status == 429 else f"server error {resp.status}"
                last_error = f"{kind}: {resp.text[:200]}"
                retries += 1
                if attempt < self.max_retries:
                    self._sleep(attempt, floor=_retry_after(resp.headers))
                    continue
                break
            if not resp.ok:
                # 4xx (bad request / auth): non-transient, surface immediately, named.
                kind = (
                    "authentication failure"
                    if resp.status in (401, 403)
                    else f"client error {resp.status}"
                )
                raise GatewayError(f"DeepSeek {kind} for {request.task_kind}: {resp.text[:300]}")

            data, tokens_in, tokens_out, content = self._parse(resp.text)
            if data is None or not self._schema_ok(data, request.expected_keys):
                truncated = self._looks_truncated(resp.text, content)
                if truncated:
                    # Salvage is the LAST resort, not the first.
                    #
                    # A recovered prefix is a real world with pieces missing — nine
                    # entities become five, the `actors` key disappears entirely — and it
                    # satisfies the schema check just as well as a complete one. Using it
                    # while a larger budget is still available would silently simulate a
                    # truncated roster. So retry with more room first, and fall back to
                    # the prefix only when there is no room left; that still beats
                    # discarding everything, which is how a provider limit came to be
                    # reported as "no actors were compiled".
                    room_left = body["max_tokens"] < self.max_output_tokens
                    salvaged = salvage_json(content)
                    if (
                        not room_left
                        and salvaged is not None
                        and self._schema_ok(salvaged, request.expected_keys)
                    ):
                        validation_failures.append(
                            f"truncated on attempt {attempt} with no output budget "
                            "left; recovered the parsable prefix, which may be incomplete"
                        )
                        return GatewayResponse(
                            task_kind=request.task_kind,
                            data=salvaged,
                            raw_text=content,
                            model=self._model,
                            params={
                                "temperature": body["temperature"],
                                "max_tokens": body["max_tokens"],
                                "recovered_from_truncation": True,
                            },
                            seed=request.seed,
                            prompt_hash=prompt_hash(request.prompt),
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            retries=retries,
                            validation_failures=tuple(validation_failures),
                            latency_ms=latency,
                        )
                failure = (
                    f"{'truncated' if truncated else 'malformed/missing-keys'} on attempt {attempt}"
                )
                validation_failures.append(failure)
                last_error = failure
                retries += 1
                if attempt < self.max_retries:
                    if truncated:
                        # Retrying a truncated response at the same cap re-truncates at
                        # exactly the same point, forever. Give it more room instead.
                        body["max_tokens"] = min(
                            int(body["max_tokens"] * 2), self.max_output_tokens
                        )
                        continue
                    # Otherwise: deterministic repair. Same decision context, stricter
                    # instruction. This corrects malformed JSON; it never supplies
                    # content the model did not produce.
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

    @staticmethod
    def _looks_truncated(envelope_text: str, content: str) -> bool:
        """Whether the provider stopped mid-object rather than producing bad JSON.

        The two failures need opposite responses — more room versus a correction — so
        they are distinguished rather than both retried the same way.
        """

        try:
            envelope = json.loads(envelope_text)
            if envelope["choices"][0].get("finish_reason") == "length":
                return True
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass
        stripped = content.strip()
        return bool(stripped) and not stripped.endswith(("}", "]"))

    def _sleep(self, attempt: int, *, ceiling: float = 8.0, floor: float = 0.0) -> None:
        """Exponential backoff with full jitter.

        Jitter is not decoration: four branches retrying in lockstep re-arrive
        together, and a rate limit met by a synchronized retry is met again.
        """

        base = min(self.backoff_base * (2**attempt), ceiling)
        time.sleep(max(floor, base * (0.5 + random.random() / 2)))

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
