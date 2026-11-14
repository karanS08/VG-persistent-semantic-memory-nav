"""Tests for DaaamBridgeClient's HTTP transport -- run against a real local test
server (not daaam_bridge_server / not the live daaam_dev container) so these are
safe to run standalone without a GPU or the DAAAM bridge up.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from daaam_adapter.bridge_client import DaaamBridgeClient


class _CountingHandler(BaseHTTPRequestHandler):
    """Counts distinct TCP connections (via a class-level set of client ports)
    separately from requests handled, so a test can tell "one connection, many
    requests" (keep-alive working) apart from "one connection per request".

    protocol_version MUST be "HTTP/1.1" -- BaseHTTPRequestHandler defaults to
    "HTTP/1.0", which closes the connection after every response regardless of
    what the client asks for. Without this override, the test would show 1
    connection per request even with a correctly keep-alive client, since the
    SERVER would refuse to keep the socket open -- caught via a real failing run
    (6 requests, 6 distinct connections) before landing on this fix.
    """

    protocol_version = "HTTP/1.1"
    connections_seen: set = set()
    requests_handled: int = 0

    def log_message(self, format, *args):  # silence test output
        pass

    def _respond(self, payload: dict):
        type(self).connections_seen.add(self.client_address[1])
        type(self).requests_handled += 1
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._respond({"status": "ok", "episode_running": False})
        elif self.path == "/positions":
            self._respond({"positions": {}})
        elif self.path == "/updates":
            self._respond({"updates": []})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # MUST consume the request body before responding: on a kept-alive
        # connection, any unread bytes left in the socket get misread as the start
        # of the NEXT request line, corrupting the stream. Caught via a real
        # `400 Bad request syntax ('{"config_path": ...}GET /positions ...')`
        # failure before adding this.
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        if self.path == "/frame":
            self._respond({"status": "ok", "frame_id": 0, "num_tracks": 0})
        elif self.path == "/episode/start":
            self._respond({"status": "started", "output_dir": "/tmp/x"})
        elif self.path == "/episode/stop":
            self._respond({"status": "stopped", "corrections": []})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def test_server():
    _CountingHandler.connections_seen = set()
    _CountingHandler.requests_handled = 0
    # ThreadingHTTPServer, not plain HTTPServer: with HTTP/1.1 keep-alive, a
    # single-threaded HTTPServer blocks its one accept loop inside the current
    # connection's handler waiting for a possible next request on that same
    # socket -- which deadlocks shutdown() once a client holds a connection open
    # (exactly what a keep-alive-reusing client is supposed to do). Hit this for
    # real (a hung test run, killed manually) before switching to the threading
    # variant, which handles each connection on its own thread so the accept loop
    # stays responsive.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=5)


def test_client_round_trips_all_endpoints(test_server):
    client = DaaamBridgeClient(base_url=test_server, timeout=5.0)
    assert client.health()["status"] == "ok"
    assert client.start_episode()["status"] == "started"
    resp = client.push_frame(
        frame_id=0, timestamp=0.0,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8), depth=np.zeros((4, 4), dtype=np.float32),
        transform=[0, 0, 0, 0, 0, 0, 1], camera_intrinsics={"fx": 1, "fy": 1, "cx": 2, "cy": 2},
    )
    assert resp["status"] == "ok"
    assert client.get_positions() == {}
    assert client.pull_updates() == []
    assert client.stop_episode()["status"] == "stopped"


def test_client_reuses_one_tcp_connection_across_calls(test_server):
    """The actual real-time-optimization behavior under test: multiple calls
    through the same DaaamBridgeClient should share one persistent TCP connection
    (HTTP keep-alive via requests.Session) rather than opening a fresh one per
    call. 6 requests, 1 connection."""
    client = DaaamBridgeClient(base_url=test_server, timeout=5.0)
    client.health()
    client.start_episode()
    client.get_positions()
    client.pull_updates()
    client.health()
    client.stop_episode()

    assert _CountingHandler.requests_handled == 6
    assert len(_CountingHandler.connections_seen) == 1, (
        f"expected all 6 requests to share one TCP connection (keep-alive), "
        f"saw {len(_CountingHandler.connections_seen)} distinct connections"
    )


def test_separate_clients_do_not_share_a_connection(test_server):
    """Sanity check on the test methodology itself: two independently-constructed
    clients should NOT share a connection, so the single-connection result above
    is really about session reuse, not e.g. the test server coalescing everything."""
    a = DaaamBridgeClient(base_url=test_server, timeout=5.0)
    b = DaaamBridgeClient(base_url=test_server, timeout=5.0)
    a.health()
    b.health()
    assert len(_CountingHandler.connections_seen) == 2
