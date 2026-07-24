"""HTTP transport abstraction for live research and model calls.

`UrllibTransport` is the real production transport: it honours the environment's
HTTPS proxy and CA bundle, follows redirects, and records every request for the
audit trail. `FakeTransport` replays canned responses for deterministic tests, so
the production parsing/retry/extraction code is exercised with only the socket
mocked.
"""

from __future__ import annotations

import gzip
import os
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

DEFAULT_UA = "Mozilla/5.0 (compatible; SWorldModel/0.1; +research)"


class HttpError(Exception):
    """A network-level failure (DNS, TLS, timeout, connection reset)."""


@dataclass
class HttpResponse:
    url: str
    final_url: str
    status: int
    headers: dict[str, str]
    text: str
    elapsed_ms: int

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


@dataclass
class HttpCall:
    method: str
    url: str
    final_url: str
    status: int
    elapsed_ms: int
    bytes: int
    error: str | None = None


class HttpTransport(Protocol):
    calls: list[HttpCall]

    def get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
    ) -> HttpResponse: ...

    def post_json(
        self,
        url: str,
        body: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse: ...


def _ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca and os.path.exists(ca):
        return ssl.create_default_context(cafile=ca)
    return ssl.create_default_context()


class UrllibTransport:
    """Real transport. Uses urllib, which picks up HTTPS_PROXY automatically and the
    CA bundle from SSL_CERT_FILE."""

    def __init__(self, *, user_agent: str = DEFAULT_UA) -> None:
        self.user_agent = user_agent
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_ssl_context())
        )
        self.calls: list[HttpCall] = []

    def _record(self, call: HttpCall) -> None:
        self.calls.append(call)

    def get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
    ) -> HttpResponse:
        return self._request("GET", url, None, headers, timeout)

    def post_json(
        self,
        url: str,
        body: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        import json

        data = json.dumps(body).encode("utf-8")
        h = {"Content-Type": "application/json", **(headers or {})}
        return self._request("POST", url, data, h, timeout)

    def _request(
        self,
        method: str,
        url: str,
        data: bytes | None,
        headers: dict[str, str] | None,
        timeout: float,
    ) -> HttpResponse:
        req_headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip", **(headers or {})}
        req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
        start = time.monotonic()
        try:
            resp = self._opener.open(req, timeout=timeout)
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            text = raw.decode(resp.headers.get_content_charset() or "utf-8", errors="replace")
            elapsed = int((time.monotonic() - start) * 1000)
            hr = HttpResponse(
                url=url,
                final_url=resp.geturl(),
                status=resp.status,
                headers={k.lower(): v for k, v in resp.headers.items()},
                text=text,
                elapsed_ms=elapsed,
            )
            self._record(HttpCall(method, url, hr.final_url, hr.status, elapsed, len(raw)))
            return hr
        except urllib.error.HTTPError as exc:  # 4xx/5xx: keep the body for inspection
            elapsed = int((time.monotonic() - start) * 1000)
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            self._record(
                HttpCall(method, url, url, exc.code, elapsed, len(body), error=str(exc.code))
            )
            return HttpResponse(url, url, exc.code, dict(exc.headers or {}), body, elapsed)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            self._record(HttpCall(method, url, url, 0, elapsed, 0, error=str(exc)))
            raise HttpError(f"{method} {url} failed: {exc}") from exc


Route = tuple[Callable[[str], bool], object]


class FakeTransport:
    """Deterministic transport for tests. Routes are matched in order; each route maps
    a URL predicate to an ``HttpResponse`` or a callable ``(method, url, body)`` ->
    ``HttpResponse``. Unmatched requests raise ``HttpError`` (a network failure)."""

    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.calls: list[HttpCall] = []

    def add(self, predicate: Callable[[str], bool], response: object) -> FakeTransport:
        self.routes.append((predicate, response))
        return self

    def add_url(self, url: str, response: HttpResponse) -> FakeTransport:
        return self.add(lambda u: u == url, response)

    def _dispatch(self, method: str, url: str, body: dict[str, object] | None) -> HttpResponse:
        for predicate, response in self.routes:
            if predicate(url):
                resp = response(method, url, body) if callable(response) else response
                assert isinstance(resp, HttpResponse)
                self.calls.append(
                    HttpCall(
                        method, url, resp.final_url, resp.status, resp.elapsed_ms, len(resp.text)
                    )
                )
                return resp
        self.calls.append(HttpCall(method, url, url, 0, 0, 0, error="no route"))
        raise HttpError(f"FakeTransport: no route for {method} {url}")

    def get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
    ) -> HttpResponse:
        return self._dispatch("GET", url, None)

    def post_json(
        self,
        url: str,
        body: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        return self._dispatch("POST", url, body)


def html_response(
    url: str, text: str, *, status: int = 200, final_url: str | None = None
) -> HttpResponse:
    return HttpResponse(url, final_url or url, status, {"content-type": "text/html"}, text, 1)
