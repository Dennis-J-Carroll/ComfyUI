"""HTTP client for a running ComfyUI server.

Deliberately talks HTTP rather than importing comfy: comfy/cli_args.py:236
parses sys.argv when args_parsing is enabled, and comfy/model_management.py:238
probes the GPU at import time. Both are hostile inside a notebook kernel.
"""
from __future__ import annotations

import os
import time
import uuid

import requests


class ComfyError(RuntimeError):
    """Server rejected a request. Carries node_errors when present."""


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
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False

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
                    interval: float = 1.0) -> list[dict]:
        """Poll /history until outputs appear. Returns image refs."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
            if r.status_code == 200:
                hist = r.json().get(prompt_id)
                if hist and hist.get("outputs"):
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

    def generate(self, graph: dict, timeout: float = 600.0) -> list[bytes]:
        """submit -> wait -> fetch. The whole inline-cell path in one call."""
        pid = self.submit(graph)
        return [self.fetch_image(ref) for ref in self.wait_result(pid, timeout)]
