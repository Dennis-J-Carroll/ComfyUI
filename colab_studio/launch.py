"""Background the ComfyUI server so the notebook kernel stays usable.

The original notebook ended with `!python3 main.py`, which never returns --
every cell after it was unreachable. Backgrounding is what makes the
generate/log/ops cells exist at all.

Ordering matters: server -> wait_ready() -> tunnel. Starting the tunnel
first prints a URL that 502s until the server finishes booting.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

CLOUDFLARED = "/usr/local/bin/cloudflared"
TUNNEL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def start_server(comfy_dir: str, flags: list[str], log_path: str,
                 port: int = 8188, python: str | None = None) -> subprocess.Popen:
    """Launch main.py detached, stdout+stderr to log_path. Returns at once."""
    exe = python or sys.executable
    cmd = [exe, "main.py", "--listen", "127.0.0.1", "--port", str(port), *flags]
    log = open(log_path, "wb")
    return subprocess.Popen(
        cmd, cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_tunnel(port: int, log_path: str, timeout: float = 40.0,
                 interval: float = 1.0) -> str | None:
    """Start cloudflared and scrape the public URL out of its log.

    Call only after ComfyClient.wait_ready() returns True.
    """
    if os.path.exists(log_path):
        os.remove(log_path)
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() not in (None,):
            return None
        try:
            with open(log_path, "r", errors="ignore") as fh:
                m = TUNNEL_RE.search(fh.read())
            if m:
                return m.group(0)
        except FileNotFoundError:
            pass
        time.sleep(interval)
    return None


def tail(log_path: str, n: int = 40) -> str:
    """Last n lines of a logfile. Empty string if it does not exist yet."""
    try:
        with open(log_path, "r", errors="ignore") as fh:
            return "\n".join(fh.read().splitlines()[-n:])
    except FileNotFoundError:
        return ""
