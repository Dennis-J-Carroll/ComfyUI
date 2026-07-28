"""Makes this package robust to test-collection order (Phase 0 item 0.h).

`comfy/cli_args.py` parses `sys.argv` into a module-level `args` Namespace
exactly once, at cli_args' own import time, and only if
`comfy.options.args_parsing` has already been enabled -- otherwise it
silently parses `[]` and gets the argparse defaults (`args.cpu == False`).
`comfy/model_management.py` then reads that same frozen `args.cpu` at ITS
OWN import time to decide `cpu_state`, and immediately derives `vram_state`
and `total_vram` from that decision in module-level code -- so once either
module has been imported with the wrong `args.cpu`, nothing done afterwards
via `enable_args_parsing()` can retroactively fix it.

workflows_test.py needs CPU mode -- this machine, and most CI runners, are
CPU-only. Its fixture used to set `sys.argv` and call
`comfy.options.enable_args_parsing()` itself, right before `import nodes`.
That works when this package happens to be collected first. It silently
breaks when an earlier-collected package already imported `comfy.cli_args`
with args_parsing still disabled -- confirmed via instrumentation: running
`pytest tests-unit/assets_test tests-unit/colab_studio_test/workflows_test.py`
imports `comfy.cli_args` during collection of
`tests-unit/assets_test/queries/test_asset.py` (args_parsing is False at
that point, so `args.cpu` freezes False), while `comfy.model_management`
itself is NOT yet imported. `import nodes` inside workflows_test.py's
fixture then imports `comfy.model_management` for the first time, which
reads the frozen `args.cpu = False` and crashes immediately, mid-import, at
`total_vram = get_total_memory(get_torch_device())` with "Torch not
compiled with CUDA enabled" on any CPU-only torch build. Because the crash
happens DURING that import, nothing the fixture could do to
`comfy.model_management` *after* `import nodes` would help -- the fix has
to land before `comfy.model_management` is ever imported.

pytest imports a package's conftest.py before collecting/importing its test
modules, so doing the following here, at *module* scope, guarantees it runs
before workflows_test.py's fixture ever calls `import nodes`, regardless of
what an earlier-collected package already touched this session.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import comfy.options
import pytest

sys.argv = ["main.py", "--cpu"]
comfy.options.enable_args_parsing()

# --- Defend against modules an earlier-collected package already cached. ---
#
# (a) comfy.cli_args imported while args_parsing was still disabled: its
#     `args` Namespace was already built from `parser.parse_args([])` and
#     will never be re-parsed. Patch the one field comfy.model_management
#     actually reads at its own import time. This is the branch that fires
#     in the reproduced hostile order described above -- verified: without
#     it, `import nodes` inside the workflows_test fixture crashes at
#     comfy/model_management.py's module level before the fixture body
#     resumes.
_cli_args = sys.modules.get("comfy.cli_args")
if _cli_args is not None:
    _cli_args.args.cpu = True

# (b) comfy.model_management itself already imported (and therefore already
#     ran its own CPUState/VRAMState decision at import time). A no-op in
#     the hostile order above, since (a) prevents model_management from
#     ever being imported with the wrong state in the first place -- but a
#     defensive guard for any other hostile order where model_management is
#     the module that ends up cached. Best effort: `total_vram`, computed
#     from the GPU path at that module's own import time, is not
#     recomputed here. `vram_state` is forced to DISABLED alongside
#     `cpu_state`, mirroring model_management.py's own
#     `if cpu_state != CPUState.GPU: vram_state = VRAMState.DISABLED`, so
#     downstream code never observes a state combination the module itself
#     would not have produced.
_mm = sys.modules.get("comfy.model_management")
if _mm is not None:
    _mm.cpu_state = _mm.CPUState.CPU
    _mm.vram_state = _mm.VRAMState.DISABLED


# --- Shared stub ComfyUI HTTP server -------------------------------------
#
# One real HTTP server on a loopback socket, shared by client_test.py and
# telemetry_test.py (E3: VramProbe is deliberately tested against the same
# stub client.py already exercises, not a second fake). Living in conftest.py
# rather than being imported between test modules avoids each importer's
# `server` parameter looking like a pyflakes F811 redefinition of an
# imported name -- pytest fixtures defined here are visible to every test
# module in this package with no import at all.

def _default_devices():
    """One plausible A100 entry, matching server.py's system_stats schema
    (system, devices[]; devices[0] is the primary device)."""
    return [{"name": "NVIDIA A100-SXM4-40GB", "type": "cuda", "index": 0,
             "vram_total": 42_949_672_960, "vram_free": 21_474_836_480,
             "torch_vram_total": 20_000_000_000,
             "torch_vram_free": 10_000_000_000}]


STATE = {"ready": True, "history_hits": 0, "history_ready_after": 0,
         "history_error": False, "devices": _default_devices()}

# What ComfyUI actually puts in /history when a node blows up at runtime:
# outputs stays empty forever, so only `status` reveals the failure.
ERROR_HISTORY = {
    "outputs": {},
    "status": {
        "status_str": "error",
        "completed": False,
        "messages": [
            ["execution_start", {"prompt_id": "pid-123"}],
            ["execution_error", {
                "node_id": "5",
                "node_type": "KSampler",
                "exception_type": "torch.cuda.OutOfMemoryError",
                "exception_message": "CUDA out of memory. Tried to allocate 2 GiB",
            }],
        ],
    },
}

# In-flight: completed is False but status_str is not "error". Treating this
# as a failure would abort every generate on the first poll.
RUNNING_HISTORY = {
    "outputs": {},
    "status": {"status_str": "success", "completed": False, "messages": []},
}


class _StubHandler(BaseHTTPRequestHandler):
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
                self._json(200, {"system": {"comfyui_version": "0.10.0"},
                                 "devices": STATE["devices"]})
        elif self.path.startswith("/history/"):
            STATE["history_hits"] += 1
            pid = self.path.rsplit("/", 1)[-1]
            if STATE["history_error"]:
                self._json(200, {pid: ERROR_HISTORY})
            elif STATE["history_hits"] <= STATE["history_ready_after"]:
                self._json(200, {pid: RUNNING_HISTORY})
            else:
                self._json(200, {pid: {"outputs": {"7": {"images": [
                    {"filename": "out_001.png", "subfolder": "colab", "type": "output"}
                ]}}, "status": {"status_str": "success", "completed": True,
                                "messages": []}}})
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
    STATE.update({"ready": True, "history_hits": 0, "history_ready_after": 0,
                  "history_error": False, "devices": _default_devices()})
    httpd = HTTPServer(("127.0.0.1", 0), _StubHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", STATE
    httpd.shutdown()
