"""Local HTTP bridge between LM Studio agents and the Chrome search extension.

Runs on localhost:3456. The Chrome extension polls /pending for search
queries and posts results to /results. Our LM Studio backend calls
/search?q=... and /fetch?url=... synchronously.

Usage:
    python -m polybillionaire.search_bridge
    # or
    pb search-bridge
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Pending requests: id -> {type, query/url, event, result}
_pending: dict[str, dict] = {}
_lock = threading.Lock()

PORT = 3456


class BridgeHandler(BaseHTTPRequestHandler):
    """Handles requests from both the Chrome extension and the LM Studio backend."""

    def log_message(self, fmt, *args):
        # Quiet logging — only errors
        pass

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _json_response(self, code: int, data: dict | list | str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # ── Called by our Python backend ──────────────────────────

    def _handle_search(self, query: str, max_results: int = 5):
        """Synchronous search — blocks until Chrome extension responds."""
        req_id = str(uuid.uuid4())[:8]
        event = threading.Event()
        with _lock:
            _pending[req_id] = {
                "type": "search",
                "query": query,
                "max_results": max_results,
                "event": event,
                "result": None,
            }

        # Wait for Chrome extension to post results (max 30s)
        if event.wait(timeout=30):
            with _lock:
                result = _pending.pop(req_id, {}).get("result", [])
            self._json_response(200, result)
        else:
            with _lock:
                _pending.pop(req_id, None)
            self._json_response(504, {"error": "Chrome extension timeout"})

    def _handle_fetch(self, url: str):
        """Synchronous fetch — blocks until Chrome extension responds."""
        req_id = str(uuid.uuid4())[:8]
        event = threading.Event()
        with _lock:
            _pending[req_id] = {
                "type": "fetch",
                "url": url,
                "event": event,
                "result": None,
            }

        if event.wait(timeout=30):
            with _lock:
                result = _pending.pop(req_id, {}).get("result", "")
            self._json_response(200, {"text": result})
        else:
            with _lock:
                _pending.pop(req_id, None)
            self._json_response(504, {"error": "Chrome extension timeout"})

    # ── Called by Chrome extension ────────────────────────────

    def _handle_pending(self):
        """Return the next pending request for the extension to handle."""
        with _lock:
            for req_id, req in _pending.items():
                if req.get("result") is None and not req["event"].is_set():
                    data = {"id": req_id, "type": req["type"]}
                    if req["type"] == "search":
                        data["query"] = req["query"]
                        data["max_results"] = req.get("max_results", 5)
                    elif req["type"] == "fetch":
                        data["url"] = req["url"]
                    self._json_response(200, data)
                    return
        # No pending requests
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _handle_results(self):
        """Receive results from the Chrome extension."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return

        req_id = data.get("id", "")
        result = data.get("result")

        with _lock:
            if req_id in _pending:
                _pending[req_id]["result"] = result
                _pending[req_id]["event"].set()

        self._json_response(200, {"ok": True})

    # ── Routing ───────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/search":
            query = params.get("q", [""])[0]
            max_results = int(params.get("n", ["5"])[0])
            if not query:
                self._json_response(400, {"error": "Missing q parameter"})
                return
            self._handle_search(query, max_results)

        elif parsed.path == "/fetch":
            url = params.get("url", [""])[0]
            if not url:
                self._json_response(400, {"error": "Missing url parameter"})
                return
            self._handle_fetch(url)

        elif parsed.path == "/pending":
            self._handle_pending()

        elif parsed.path == "/health":
            with _lock:
                n = len(_pending)
            self._json_response(200, {"status": "ok", "pending": n})

        else:
            self._json_response(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/results":
            self._handle_results()
        else:
            self._json_response(404, {"error": "Not found"})


class ThreadedHTTPServer(HTTPServer):
    """Handle each request in a new thread so /search can block."""
    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread,
                             args=(request, client_address))
        t.daemon = True
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def run(port: int = PORT):
    server = ThreadedHTTPServer(("127.0.0.1", port), BridgeHandler)
    print(f"Search bridge running on http://localhost:{port}")
    print("Waiting for Chrome extension to connect...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down search bridge")
        server.shutdown()


if __name__ == "__main__":
    run()
