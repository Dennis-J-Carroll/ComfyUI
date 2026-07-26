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

# Handle on the tunnel started by the last successful start_tunnel().
# start_tunnel() returns only a URL, so without this the notebook holds no
# reference: re-tunnelling would delete the log the running cloudflared still
# has an fd on, spawn a second process on the same port, and leave the first
# alive and unkillable from the kernel.
_TUNNEL: subprocess.Popen | None = None


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


def current_tunnel() -> subprocess.Popen | None:
    """The cloudflared process from the last start_tunnel(), if still alive."""
    if _TUNNEL is not None and _TUNNEL.poll() is None:
        return _TUNNEL
    return None


def stop_tunnel(timeout: float = 10.0) -> bool:
    """Terminate the tunnel from the last start_tunnel().

    Returns True if a live process was stopped. Safe to call when no tunnel
    is running.
    """
    global _TUNNEL
    proc, _TUNNEL = _TUNNEL, None
    if proc is None or proc.poll() is not None:
        return False
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    return True


def start_tunnel(port: int, log_path: str, timeout: float = 40.0,
                 interval: float = 1.0) -> str | None:
    """Start cloudflared and scrape the public URL out of its log.

    Call only after ComfyClient.wait_ready() returns True.

    Any tunnel from a previous call is stopped first: the log is truncated
    here, so leaving the old process holding an fd on it would orphan a second
    cloudflared on the same port with no way to reach it from the notebook.
    """
    global _TUNNEL
    stop_tunnel()
    if os.path.exists(log_path):
        os.remove(log_path)
    log = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    except FileNotFoundError:
        return None
    # Recorded before the scrape loop so a caller can always stop what was
    # started, including on the timeout path below.
    _TUNNEL = proc
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            _TUNNEL = None
            return None
        try:
            with open(log_path, "r", errors="ignore") as fh:
                m = TUNNEL_RE.search(fh.read())
            if m:
                # start_new_session=True: the tunnel must outlive this cell.
                return m.group(0)
        except FileNotFoundError:
            pass
        time.sleep(interval)
    proc.terminate()
    _TUNNEL = None
    return None


def tail(log_path: str, n: int = 40) -> str:
    """Last n lines of a logfile. Empty string if it does not exist yet."""
    try:
        with open(log_path, "r", errors="ignore") as fh:
            return "\n".join(fh.read().splitlines()[-n:])
    except FileNotFoundError:
        return ""
