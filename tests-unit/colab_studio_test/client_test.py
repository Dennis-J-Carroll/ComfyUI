"""Tests ComfyClient against a real stub HTTP server on a loopback socket."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from colab_studio.client import ComfyClient, ComfyError

STATE = {"ready": True, "history_hits": 0, "history_ready_after": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/system_stats"):
            if not STATE["ready"]:
                self._json(503, {"error": "booting"})
            else:
                self._json(200, {"system": {"comfyui_version": "0.10.0"}})
        elif self.path.startswith("/history/"):
            STATE["history_hits"] += 1
            if STATE["history_hits"] <= STATE["history_ready_after"]:
                self._json(200, {})
            else:
                pid = self.path.rsplit("/", 1)[-1]
                self._json(200, {pid: {"outputs": {"7": {"images": [
                    {"filename": "out_001.png", "subfolder": "colab", "type": "output"}
                ]}}}})
        elif self.path.startswith("/view"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"PNG!")
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if self.path == "/prompt":
            body = json.loads(raw)
            if "prompt" not in body:
                self._json(400, {"error": "no_prompt", "node_errors": {}})
            elif body["prompt"].get("bad"):
                self._json(400, {"error": {"message": "bad graph"},
                                 "node_errors": {"5": "nope"}})
            else:
                self._json(200, {"prompt_id": "pid-123", "number": 1,
                                 "node_errors": {}})
        elif self.path == "/upload/image":
            self._json(200, {"name": "uploaded.png", "subfolder": "",
                             "type": "input"})
        else:
            self._json(404, {})


@pytest.fixture
def server():
    STATE.update({"ready": True, "history_hits": 0, "history_ready_after": 0})
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", STATE
    httpd.shutdown()


def test_wait_ready_true_when_server_answers(server):
    url, _ = server
    assert ComfyClient(url).wait_ready(timeout=5) is True


def test_wait_ready_false_when_never_ready(server):
    url, state = server
    state["ready"] = False
    assert ComfyClient(url).wait_ready(timeout=1.5) is False


def test_submit_returns_prompt_id(server):
    url, _ = server
    assert ComfyClient(url).submit({"1": {"class_type": "X", "inputs": {}}}) == "pid-123"


def test_submit_raises_with_node_errors_on_400(server):
    url, _ = server
    with pytest.raises(ComfyError) as exc:
        ComfyClient(url).submit({"bad": True})
    assert "5" in str(exc.value)


def test_wait_result_polls_until_outputs_appear(server):
    url, state = server
    state["history_ready_after"] = 2
    refs = ComfyClient(url).wait_result("pid-123", timeout=10)
    assert refs[0]["filename"] == "out_001.png"
    assert state["history_hits"] > 2


def test_wait_result_times_out(server):
    url, state = server
    state["history_ready_after"] = 10_000
    with pytest.raises(TimeoutError):
        ComfyClient(url).wait_result("pid-123", timeout=1.0)


def test_fetch_image_returns_bytes(server):
    url, _ = server
    data = ComfyClient(url).fetch_image(
        {"filename": "out_001.png", "subfolder": "colab", "type": "output"})
    assert data == b"PNG!"


def test_upload_image_returns_name_for_loadimage(server, tmp_path):
    url, _ = server
    p = tmp_path / "in.png"
    p.write_bytes(b"PNG!")
    assert ComfyClient(url).upload_image(str(p)) == "uploaded.png"


def test_generate_end_to_end_returns_image_bytes(server):
    url, _ = server
    out = ComfyClient(url).generate({"1": {"class_type": "X", "inputs": {}}}, timeout=10)
    assert out == [b"PNG!"]
