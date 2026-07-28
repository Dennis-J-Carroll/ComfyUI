"""HTTP client for a running ComfyUI server.

Deliberately talks HTTP rather than importing comfy: comfy/cli_args.py:236
parses sys.argv when args_parsing is enabled, and comfy/model_management.py:238
probes the GPU at import time. Both are hostile inside a notebook kernel.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Callable

import requests


class ComfyError(RuntimeError):
    """Server rejected a request. Carries node_errors when present."""


def _execution_error(hist: dict) -> str | None:
    """Failure detail from a /history entry, or None if it has not failed.

    Only `status_str == "error"` counts. `completed` is legitimately False
    while a job is still running, so keying on it would abort every generate
    on the first poll.
    """
    status = hist.get("status") or {}
    if status.get("status_str") != "error":
        return None
    for message in status.get("messages") or []:
        if not (isinstance(message, (list, tuple)) and len(message) == 2):
            continue
        name, payload = message
        if name not in ("execution_error", "execution_interrupted"):
            continue
        if not isinstance(payload, dict):
            continue
        return (
            f"{payload.get('exception_type') or name}: "
            f"{payload.get('exception_message') or '(no message)'} "
            f"[node {payload.get('node_id')} "
            f"{payload.get('node_type')}]"
        )
    return "server reported status_str=error with no execution_error message"


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def wait_ready(self, timeout: float = 180.0, interval: float = 1.0) -> bool:
        """Poll /system_stats until the server answers. Start the tunnel only
        after this returns True, or the public URL 502s."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/system_stats", timeout=5)
                if r.status_code == 200:
                    return True
            except (requests.exceptions.MissingSchema,
                    requests.exceptions.InvalidSchema,
                    requests.exceptions.InvalidURL) as err:
                raise ComfyError(f"invalid base_url {self.base_url!r}: {err}") from err
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False

    def system_stats(self) -> dict:
        """GET /system_stats -- the server's own device inventory and memory
        counters (server.py's system_stats handler; devices[0] is the
        primary device). This is the only window into VRAM used by the
        diffusion model, which runs in this server's process, not the
        notebook kernel -- see colab_studio.telemetry for how it is turned
        into an (honestly qualified) observed-peak reading."""
        r = requests.get(f"{self.base_url}/system_stats", timeout=10)
        if r.status_code != 200:
            raise ComfyError(f"system_stats failed ({r.status_code}): {r.text[:300]}")
        return r.json()

    def upload_image(self, path: str) -> str:
        """Upload to the server's input/ dir. Returns the name to put in
        LoadImage.inputs.image."""
        with open(path, "rb") as fh:
            r = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (os.path.basename(path), fh)},
                data={"overwrite": "true"},
                timeout=120,
            )
        if r.status_code != 200:
            raise ComfyError(f"upload failed ({r.status_code}): {r.text[:300]}")
        return r.json()["name"]

    def submit(self, graph: dict) -> str:
        r = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": graph, "client_id": self.client_id},
            timeout=60,
        )
        if r.status_code != 200:
            try:
                payload = r.json()
            except ValueError:
                raise ComfyError(f"submit failed ({r.status_code}): {r.text[:300]}")
            raise ComfyError(
                f"submit rejected: {payload.get('error')} "
                f"node_errors={payload.get('node_errors')}"
            )
        return r.json()["prompt_id"]

    def wait_result(self, prompt_id: str, timeout: float = 600.0,
                    interval: float = 1.0,
                    on_poll: Callable[[], None] | None = None) -> list[dict]:
        """Poll /history until outputs appear. Returns image refs.

        Raises ComfyError as soon as the server reports an execution failure.
        Structural rejections come back from submit()'s 400 path, but runtime
        failures (OOM, tensor mismatch) only ever show up here -- and they
        leave `outputs` empty forever, so ignoring `status` means a silent
        full-timeout hang ending in a diagnostic-free TimeoutError.

        `on_poll`, when given, is called once per pass through this loop --
        the natural point to sample telemetry (see
        colab_studio.telemetry.VramProbe) at the same cadence as `interval`.
        Its exceptions are swallowed: a telemetry failure must never be able
        to abort a render that may otherwise run for many minutes.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if on_poll is not None:
                try:
                    on_poll()
                except Exception:
                    pass
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
            if r.status_code == 200:
                hist = r.json().get(prompt_id)
                if hist:
                    detail = _execution_error(hist)
                    if detail:
                        raise ComfyError(
                            f"execution failed for {prompt_id}: {detail}")
                    if hist.get("outputs"):
                        refs: list[dict] = []
                        for node_out in hist["outputs"].values():
                            refs.extend(node_out.get("images", []))
                        if refs:
                            return refs
            time.sleep(interval)
        raise TimeoutError(f"no outputs for {prompt_id} within {timeout}s")

    def fetch_image(self, ref: dict) -> bytes:
        r = requests.get(
            f"{self.base_url}/view",
            params={"filename": ref["filename"],
                    "subfolder": ref.get("subfolder", ""),
                    "type": ref.get("type", "output")},
            timeout=120,
        )
        if r.status_code != 200:
            raise ComfyError(f"view failed ({r.status_code})")
        return r.content

    def generate(self, graph: dict, timeout: float = 600.0,
                on_poll: Callable[[], None] | None = None) -> list[bytes]:
        """submit -> wait -> fetch. The whole inline-cell path in one call.

        `on_poll` is threaded straight through to wait_result() -- pass
        VramProbe.sample for VRAM telemetry sampled once per poll.
        """
        pid = self.submit(graph)
        return [self.fetch_image(ref)
                for ref in self.wait_result(pid, timeout, on_poll=on_poll)]
