"""DeepSeekGateway unit tests — mocked HTTP, no network."""

from __future__ import annotations

import json

import pytest

from _fakes import FakeTransport
from sworldmodel.deepseek_gateway import DeepSeekGateway
from sworldmodel.errors import GatewayError
from sworldmodel.gateway import GatewayRequest
from sworldmodel.http import HttpResponse


def _envelope(content: dict | str, *, tokens_in: int = 10, tokens_out: int = 5) -> str:
    text = content if isinstance(content, str) else json.dumps(content)
    return json.dumps(
        {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        }
    )


def _resp(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse("api", "api", status, {}, body, 5)


def _req(expected=("answer",)) -> GatewayRequest:
    return GatewayRequest(
        task_kind="resolution", prompt="p", context={}, seed=1, expected_keys=expected
    )


def _gateway(transport: FakeTransport) -> DeepSeekGateway:
    return DeepSeekGateway(transport, api_key="test-key", max_retries=3, backoff_base=0.0)


def test_successful_structured_call_records_real_usage() -> None:
    t = FakeTransport().add(
        lambda u: True, _resp(_envelope({"answer": 4}, tokens_in=42, tokens_out=9))
    )
    gw = _gateway(t)
    assert gw.is_live
    r = gw.generate(_req())
    assert r.data == {"answer": 4}
    assert r.tokens_in == 42 and r.tokens_out == 9
    assert gw.total_tokens == 51
    assert gw.stage_call_counts() == {"resolution": 1}


def test_transient_5xx_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def route(method, url, body):
        calls["n"] += 1
        return _resp("upstream", 503) if calls["n"] == 1 else _resp(_envelope({"answer": 1}))

    gw = _gateway(FakeTransport().add(lambda u: True, route))
    r = gw.generate(_req())
    assert r.data == {"answer": 1}
    assert r.retries >= 1


def test_malformed_json_is_repaired_and_retried() -> None:
    calls = {"n": 0}

    def route(method, url, body):
        calls["n"] += 1
        return (
            _resp(_envelope("not-json-at-all"))
            if calls["n"] == 1
            else _resp(_envelope({"answer": 2}))
        )

    gw = _gateway(FakeTransport().add(lambda u: True, route))
    r = gw.generate(_req())
    assert r.data == {"answer": 2}
    assert r.validation_failures  # a repair happened


def test_repeated_provider_failure_raises_gateway_error() -> None:
    gw = _gateway(FakeTransport().add(lambda u: True, _resp("upstream", 500)))
    with pytest.raises(GatewayError):
        gw.generate(_req())
    assert gw.failed_calls == 1


def test_auth_error_is_not_retried() -> None:
    calls = {"n": 0}

    def route(method, url, body):
        calls["n"] += 1
        return _resp('{"error":"bad key"}', 401)

    gw = _gateway(FakeTransport().add(lambda u: True, route))
    with pytest.raises(GatewayError):
        gw.generate(_req())
    assert calls["n"] == 1  # surfaced immediately, no retries


def test_missing_key_triggers_repair_then_failure() -> None:
    gw = _gateway(FakeTransport().add(lambda u: True, _resp(_envelope({"other": 1}))))
    with pytest.raises(GatewayError):
        gw.generate(_req(expected=("answer",)))
    assert gw.failed_calls == 1
