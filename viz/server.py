"""A tiny read-only replay server for simulation traces — stdlib only, localhost.

It serves one self-contained page and two JSON endpoints backed entirely by
``replay.py``. It runs no simulation, imports no ``sworldmodel``, and writes nothing:
it reads the artifacts a run already produced and hands them to the browser.

    python3 -m viz.server            # from the repo root
    python3 viz/server.py --port 8765 --root artifacts

Endpoints
    GET /                     the replay page (viz/index.html)
    GET /api/traces           every run trace under --root, newest first
    GET /api/replay?trace=... one run, normalized for replay (build_replay)

Every trace path is resolved and confined to --root, so the browser cannot walk it
outside the artifacts tree.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from replay import build_replay, discover_traces

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class ReplayHandler(BaseHTTPRequestHandler):
    root: Path = REPO / "artifacts"

    # -- helpers ---------------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: object, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _confined(self, raw: str) -> Path | None:
        """Resolve a requested trace path and refuse anything outside --root."""

        try:
            p = Path(raw)
            p = p if p.is_absolute() else (self.root / p)
            p = p.resolve()
        except (OSError, ValueError):
            return None
        root = self.root.resolve()
        if p != root and root not in p.parents:
            return None
        return p

    # -- routing ---------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/traces":
                self._json({"root": str(self.root), "traces": discover_traces(self.root)})
            elif route == "/api/replay":
                raw = (parse_qs(parsed.query).get("trace") or [""])[0]
                target = self._confined(raw) if raw else None
                if not raw:
                    self._json({"error": "missing ?trace="}, 400)
                elif target is None or not target.exists():
                    self._json({"error": f"trace not found or out of bounds: {raw}"}, 404)
                else:
                    self._json(build_replay(target))
            else:
                self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass  # the browser navigated away mid-response; nothing to do
        except Exception as exc:  # noqa: BLE001 — a viewer must never 500 silently
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    do_HEAD = do_GET

    def log_message(self, *_args: object) -> None:
        pass  # quiet by default; the terminal is for the user, not the request log


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only replay viewer for simulation traces.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--root",
        default=str(REPO / "artifacts"),
        help="directory to scan for run traces (default: <repo>/artifacts)",
    )
    args = ap.parse_args()

    ReplayHandler.root = Path(args.root).resolve()
    httpd = ThreadingHTTPServer((args.host, args.port), ReplayHandler)
    n = len(discover_traces(ReplayHandler.root))
    print(f"replay viewer → http://{args.host}:{args.port}")
    print(f"scanning {ReplayHandler.root} — {n} trace(s) found")
    print("Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        httpd.server_close()


if __name__ == "__main__":
    main()
