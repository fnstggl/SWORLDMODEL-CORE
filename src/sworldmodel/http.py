"""HTTP transport abstraction for live research and model calls.

:class:`UrllibTransport` is the real production transport. Research URLs are scraped
off third-party HTML, so the transport treats every URL as hostile input and enforces
:class:`FetchPolicy` on *every* hop of a request:

* a scheme allowlist (``http``/``https`` only — no ``file:``, ``gopher:``, ``data:``);
* every address the host resolves to must be globally routable, so a hostname that
  resolves to loopback, private, link-local, or otherwise reserved space is refused;
* redirects are followed manually and re-validated at each hop, because the address
  checked before a redirect says nothing about where the redirect points;
* the response body size cap is enforced *while* reading (and while decompressing),
  never after the bytes are already in memory;
* the declared content type must be on an allowlist.

It honours the environment's HTTPS proxy and CA bundle: the default ``ProxyHandler``
is left in place and TLS verification is never disabled. When a proxy is in use the
proxy performs its own connection, so the local address check is a first line of
defence rather than the only one — which is why the scheme, size, redirect and
content-type limits are enforced independently of it.

:class:`FakeTransport` replays canned responses for deterministic tests, so the
production parsing/retry/extraction code is exercised with only the socket mocked. It
deliberately does not resolve hostnames; policy enforcement belongs at the socket.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

DEFAULT_UA = "Mozilla/5.0 (compatible; SWorldModel/0.1; +research)"

# Only these two schemes describe a document that can be fetched over the network and
# attributed to a publisher. Everything else (file:, data:, gopher:, ftp:, ...) either
# reads the local machine or cannot be attributed, so it is not a research source.
ALLOWED_SCHEMES = frozenset({"http", "https"})

# Content types a research source can plausibly be. A body that is not one of these
# cannot yield attributable text, and fetching it only spends budget and risk.
ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "text/",
    "application/pdf",
    "application/x-pdf",
    "application/json",
    "application/ld+json",
    "application/xml",
    "application/xhtml+xml",
    "application/rss+xml",
    "application/atom+xml",
)

# Redirect status codes that carry a Location we may follow.
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})

# Transfer-loop sizing. These are resource bounds, not evidence judgments: they cap the
# work a single hostile response can cause. Any finite value preserves the invariant.
_READ_CHUNK_BYTES = 64 * 1024
_GZIP_WBITS = 16 + zlib.MAX_WBITS


class HttpError(Exception):
    """A network-level failure (DNS, TLS, timeout, connection reset)."""


class UrlRejected(HttpError):
    """The request was refused by :class:`FetchPolicy` before or during transfer.

    A subclass of :class:`HttpError` so existing callers that treat a fetch failure as
    "this source is unavailable" keep working, while callers that want to record *why*
    a URL was refused can catch it specifically.
    """


@dataclass(frozen=True)
class FetchPolicy:
    """The limits every real request is held to. See the module docstring."""

    allowed_schemes: frozenset[str] = ALLOWED_SCHEMES
    allowed_content_types: tuple[str, ...] = ALLOWED_CONTENT_TYPES
    max_redirects: int = 5
    max_bytes: int = 8_000_000
    # Loopback/private targets are only ever legitimate for a local test server, so
    # this must be turned on explicitly and is never on in production.
    allow_private_addresses: bool = False


DEFAULT_POLICY = FetchPolicy()


@dataclass
class HttpResponse:
    url: str
    final_url: str
    status: int
    headers: dict[str, str]
    text: str
    elapsed_ms: int
    content: bytes = b""  # raw (decompressed) body, needed to read binary formats (PDF)

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

    def post_form(
        self,
        url: str,
        form: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse: ...


# ---------------------------------------------------------------------------
# Fetch policy enforcement
# ---------------------------------------------------------------------------


def check_url_shape(url: str, policy: FetchPolicy = DEFAULT_POLICY) -> None:
    """Validate what can be known from the URL alone, without touching the network.

    Raises :class:`UrlRejected` for a disallowed scheme, a missing host, or a host that
    is written as a literal address in non-global space. This is the cheap pre-filter a
    discovery queue can apply to scraped URLs; it is *not* sufficient on its own,
    because a hostname's addresses are only known after resolution.
    """

    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise UrlRejected(f"unparseable URL {url!r}: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in policy.allowed_schemes:
        raise UrlRejected(f"scheme {scheme or '(none)'!r} is not fetchable: {url}")
    host = parts.hostname
    if not host:
        raise UrlRejected(f"URL has no host: {url}")
    literal = _as_ip(host)
    if literal is not None:
        _check_address(literal, host, policy)


def resolve_and_check(url: str, policy: FetchPolicy = DEFAULT_POLICY) -> None:
    """Full validation: URL shape plus every address the host actually resolves to.

    Every resolved address must be globally routable. Resolution failure is a refusal,
    not a pass: a host we cannot evaluate is a host we do not fetch.
    """

    check_url_shape(url, policy)
    host = urllib.parse.urlsplit(url).hostname
    assert host is not None  # check_url_shape rejects a missing host
    if _as_ip(host) is not None:
        return  # already validated as a literal
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlRejected(f"host {host!r} does not resolve: {exc}") from exc
    if not infos:
        raise UrlRejected(f"host {host!r} resolves to no addresses")
    for info in infos:
        addr = _as_ip(str(info[4][0]))
        if addr is None:
            raise UrlRejected(f"host {host!r} resolved to an unparseable address")
        _check_address(addr, host, policy)


def _as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host.strip("[]").split("%", 1)[0])
    except ValueError:
        return None


def _check_address(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address, host: str, policy: FetchPolicy
) -> None:
    if policy.allow_private_addresses:
        return
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:127.0.0.1 must be judged as 127.0.0.1
        addr = mapped
    unroutable = (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
        or not addr.is_global
    )
    if unroutable:
        raise UrlRejected(f"host {host!r} resolves to non-public address {addr} — refused")


def _check_content_type(content_type: str, url: str, policy: FetchPolicy) -> None:
    kind = content_type.split(";", 1)[0].strip().lower()
    if not kind:
        raise UrlRejected(f"{url}: response declares no content type")
    if not any(kind.startswith(allowed) for allowed in policy.allowed_content_types):
        raise UrlRejected(f"{url}: content type {kind!r} is not an admissible source type")


def _ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca and os.path.exists(ca):
        return ssl.create_default_context(cafile=ca)
    return ssl.create_default_context()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Suppress urllib's automatic redirect following.

    Returning ``None`` makes urllib surface the 3xx as an ``HTTPError`` so the transport
    can validate the redirect target before deciding to follow it.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


class UrllibTransport:
    """Real transport. Uses urllib, which picks up HTTPS_PROXY automatically and the
    CA bundle from SSL_CERT_FILE. Every hop is validated against :class:`FetchPolicy`."""

    def __init__(
        self, *, user_agent: str = DEFAULT_UA, policy: FetchPolicy = DEFAULT_POLICY
    ) -> None:
        self.user_agent = user_agent
        self.policy = policy
        # ProxyHandler is part of the default handler set and is deliberately kept, so
        # HTTPS_PROXY continues to be honoured.
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=_ssl_context()), _NoRedirect()
        )
        self.calls: list[HttpCall] = []
        # Branches are simulated concurrently, so the call log is written from several
        # threads. It is an audit record: losing an entry would understate what the run
        # actually did on the wire.
        self._lock = threading.Lock()

    def _record(self, call: HttpCall) -> None:
        with self._lock:
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

    def post_form(
        self,
        url: str,
        form: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        """An urlencoded POST — the shape Google's batchexecute endpoint requires."""

        h = {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8", **(headers or {})}
        return self._request("POST", url, form.encode("utf-8"), h, timeout)

    def _request(
        self,
        method: str,
        url: str,
        data: bytes | None,
        headers: dict[str, str] | None,
        timeout: float,
    ) -> HttpResponse:
        start = time.monotonic()
        current = url
        try:
            for _hop in range(self.policy.max_redirects + 1):
                # Re-validated on every hop: the address checked before a redirect says
                # nothing about where that redirect points.
                resolve_and_check(current, self.policy)
                outcome = self._single_hop(method, url, current, data, headers, timeout, start)
                if isinstance(outcome, str):
                    current = outcome
                    continue
                return outcome
            raise UrlRejected(f"{url}: more than {self.policy.max_redirects} redirects")
        except UrlRejected as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            self._record(HttpCall(method, url, current, 0, elapsed, 0, error=str(exc)))
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            self._record(HttpCall(method, url, current, 0, elapsed, 0, error=str(exc)))
            raise HttpError(f"{method} {url} failed: {exc}") from exc
        except (http.client.HTTPException, zlib.error) as exc:
            # A truncated chunked body (IncompleteRead), a malformed status line, a
            # corrupt gzip stream: protocol-level failures of ONE document. None of
            # these is an OSError, so none was caught here — a live EU-Mercosur run
            # died twenty minutes in when one page's chunked response ended 15 bytes
            # short, and the raw http.client.IncompleteRead destroyed the entire run
            # with no artifacts. One bad page is a rejected page, never a dead run.
            elapsed = int((time.monotonic() - start) * 1000)
            partial = len(getattr(exc, "partial", b"") or b"")
            detail = f"{type(exc).__name__}: {exc}" + (
                f" ({partial} bytes received before truncation)" if partial else ""
            )
            self._record(HttpCall(method, url, current, 0, elapsed, partial, error=detail))
            raise HttpError(f"{method} {url} failed mid-body: {detail}") from exc

    def _single_hop(
        self,
        method: str,
        origin_url: str,
        current: str,
        data: bytes | None,
        headers: dict[str, str] | None,
        timeout: float,
        start: float,
    ) -> HttpResponse | str:
        """Perform one hop. Returns the next URL (a redirect) or the final response."""

        req_headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip", **(headers or {})}
        req = urllib.request.Request(current, data=data, headers=req_headers, method=method)
        try:
            resp = self._opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            location = str(exc.headers.get("Location", "")) if exc.headers else ""
            if exc.code in _REDIRECT_CODES and location:
                return urllib.parse.urljoin(current, location)
            # 4xx/5xx: keep a bounded body for inspection (a caller may need the reason).
            elapsed = int((time.monotonic() - start) * 1000)
            encoding = str(exc.headers.get("Content-Encoding", "")) if exc.headers else ""
            body = self._read_capped(exc, encoding, note=f"{origin_url} error") if exc.fp else b""
            text = body.decode("utf-8", errors="replace")
            self._record(
                HttpCall(method, origin_url, current, exc.code, elapsed, len(body), str(exc.code))
            )
            return HttpResponse(
                origin_url, current, exc.code, dict(exc.headers or {}), text, elapsed
            )
        with resp:
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            _check_content_type(hdrs.get("content-type", ""), current, self.policy)
            raw = self._read_capped(resp, hdrs.get("content-encoding", ""), note=current)
            charset = resp.headers.get_content_charset() or "utf-8"
        elapsed = int((time.monotonic() - start) * 1000)
        hr = HttpResponse(
            url=origin_url,
            final_url=resp.geturl(),
            status=resp.status,
            headers=hdrs,
            text=raw.decode(charset, errors="replace"),
            elapsed_ms=elapsed,
            content=raw,
        )
        self._record(HttpCall(method, origin_url, hr.final_url, hr.status, elapsed, len(raw)))
        return hr

    def _read_capped(self, fp: object, content_encoding: str, *, note: str) -> bytes:
        """Read (and gzip-decompress) a body, refusing as soon as the cap is passed.

        The cap is checked against bytes transferred *and* bytes produced, so neither a
        long body nor a small highly-compressed one can exceed it. Enforcement happens
        during the read: an oversized body is never fully materialized.
        """

        limit = self.policy.max_bytes
        read = getattr(fp, "read", None)
        if not callable(read):  # pragma: no cover - urllib always provides read()
            return b""
        decompressor = zlib.decompressobj(_GZIP_WBITS) if content_encoding == "gzip" else None
        out = bytearray()
        transferred = 0
        while True:
            chunk = read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            transferred += len(chunk)
            if transferred > limit:
                raise UrlRejected(f"{note}: response exceeds {limit} bytes")
            if decompressor is None:
                out += chunk
            else:
                out += decompressor.decompress(chunk, limit - len(out) + 1)
            if len(out) > limit:
                raise UrlRejected(f"{note}: decompressed body exceeds {limit} bytes")
        if decompressor is not None:
            out += decompressor.flush(max(1, limit - len(out) + 1))
            if len(out) > limit:
                raise UrlRejected(f"{note}: decompressed body exceeds {limit} bytes")
        return bytes(out)


Route = tuple[Callable[[str], bool], object]


class FakeTransport:
    """Deterministic transport for tests. Routes are matched in order; each route maps
    a URL predicate to an ``HttpResponse`` or a callable ``(method, url, body)`` ->
    ``HttpResponse``. Unmatched requests raise ``HttpError`` (a network failure).

    It performs no DNS and enforces no fetch policy: policy belongs at the socket, and a
    test double that resolved hostnames could not serve fixture domains.
    """

    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.calls: list[HttpCall] = []
        # Branches are simulated concurrently, so the call log is written from several
        # threads. It is an audit record: losing an entry would understate what the run
        # actually did on the wire.
        self._lock = threading.Lock()

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

    def post_form(
        self,
        url: str,
        form: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        return self._dispatch("POST", url, {"form": form})


def html_response(
    url: str, text: str, *, status: int = 200, final_url: str | None = None
) -> HttpResponse:
    return HttpResponse(
        url,
        final_url or url,
        status,
        {"content-type": "text/html"},
        text,
        1,
        content=text.encode("utf-8"),
    )


def json_response(url: str, text: str, *, status: int = 200) -> HttpResponse:
    return HttpResponse(
        url, url, status, {"content-type": "application/json"}, text, 1, content=text.encode()
    )


def pdf_response(url: str, content: bytes, *, status: int = 200) -> HttpResponse:
    # `text` is the lossy decode a real server body would produce for binary bytes;
    # the PDF path reads `content` instead.
    return HttpResponse(
        url,
        url,
        status,
        {"content-type": "application/pdf"},
        content.decode("latin-1"),
        1,
        content=content,
    )
