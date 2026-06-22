"""Localhost-only dashboard server (stdlib http.server)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bot.dashboard.page import PAGE_HTML
from bot.dashboard.state import DashboardState


def route(path: str, state: DashboardState) -> tuple[int, str, str]:
    """Pure router: path -> (status, content_type, body)."""
    if path == "/" or path.startswith("/index"):
        return 200, "text/html; charset=utf-8", PAGE_HTML
    if path == "/api/status":
        return 200, "application/json", json.dumps(state.get(), default=str)
    return 404, "text/plain", "not found"


def make_handler(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            status, ctype, body = route(self.path, state)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # silence default stderr logging
            pass
    return Handler


def serve(state: DashboardState, port: int = 8787) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    httpd.serve_forever()
