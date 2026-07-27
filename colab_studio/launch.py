"""Background the ComfyUI server so the notebook kernel stays usable.

The original notebook ended with `!python3 main.py`, which never returns --
every cell after it was unreachable. Backgrounding is what makes the
generate/log/ops cells exist at all.

Ordering matters: server -> wait_ready() -> tunnel. Starting the tunnel
first prints a URL that 502s until the server finishes booting.

RuntimeSupervisor owns that lifecycle end to end (start/wait/stop/restart
the server, start/stop the tunnel, close() both) so it doesn't have to live
in the notebook. The module-level functions below (start_server,
start_tunnel, stop_tunnel, current_tunnel, tail) are thin wrappers over a
shared, module-level default RuntimeSupervisor instance, kept for backward
compatibility with the generated notebook -- their signatures and behaviour
are unchanged.
"""
from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
import time

from colab_studio.client import ComfyClient

CLOUDFLARED = "/usr/local/bin/cloudflared"
TUNNEL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

# Handle on the tunnel started by the last successful start_tunnel().
# start_tunnel() returns only a URL, so without this the notebook holds no
# reference: re-tunnelling would delete the log the running cloudflared still
# has an fd on, spawn a second process on the same port, and leave the first
# alive and unkillable from the kernel.
#
# This is module state rather than per-supervisor state deliberately: a
# cloudflared tunnel is a singleton concept for one kernel (one log file,
# one port scraped via start_new_session=True), so every RuntimeSupervisor
# shares it rather than each tracking its own. Kept as a module global,
# rather than folded into RuntimeSupervisor, so this file's existing tests
# -- which poke launch._TUNNEL directly -- keep working unchanged.
_TUNNEL: subprocess.Popen | None = None


class RuntimeSupervisor:
    """Owns the lifecycle of one ComfyUI server process (plus the shared
    cloudflared tunnel, see _TUNNEL above) for this notebook kernel.

    start_server() refuses to launch a second server while one this
    instance started is still alive: it returns the existing Popen
    unchanged rather than raising. That matches the common case -- a
    notebook cell re-run to "make sure the server is up" -- where the
    caller wants a live process back, not an exception. Call stop_server()
    or restart_server() first to force a fresh one.
    """

    def __init__(self) -> None:
        self._server: subprocess.Popen | None = None
        self._port: int = 8188
        # So a dying kernel doesn't orphan the server or cloudflared: this
        # runs at interpreter exit even if the notebook never calls close().
        atexit.register(self.close)

    # ---------------------------------------------------------- server --

    def start_server(self, comfy_dir: str, flags: list[str], log_path: str,
                     port: int = 8188, python: str | None = None) -> subprocess.Popen:
        """Launch main.py detached, stdout+stderr to log_path. Returns at once.

        Returns the already-running process unchanged if one started by
        this supervisor is still alive -- see class docstring.
        """
        if self._server is not None and self._server.poll() is None:
            return self._server
        exe = python or sys.executable
        cmd = [exe, "main.py", "--listen", "127.0.0.1", "--port", str(port), *flags]
        # The parent's copy of the log fd must be closed once Popen has
        # duplicated it for the child, on every path -- including if Popen
        # itself raises. A `with` block does that regardless of outcome;
        # the fd survives in the child, which has its own duplicate.
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(
                cmd, cwd=comfy_dir, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._server = proc
        self._port = port
        return proc

    def wait_ready(self, timeout: float = 300, client: ComfyClient | None = None) -> bool:
        """Poll until the server answers. Delegates to a ComfyClient aimed
        at this supervisor's port by default; pass one in to override."""
        if client is None:
            client = ComfyClient(f"http://127.0.0.1:{self._port}")
        return client.wait_ready(timeout=timeout)

    def stop_server(self, timeout: float = 30) -> bool:
        """Terminate the server this supervisor started.

        Idempotent: returns False without raising if nothing was ever
        started, or if it had already exited.
        """
        proc, self._server = self._server, None
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
        return True

    def restart_server(self, comfy_dir: str, flags: list[str], log_path: str,
                       port: int = 8188, python: str | None = None,
                       stop_timeout: float = 30) -> subprocess.Popen:
        """stop_server() the current process, then start_server() a new one."""
        self.stop_server(timeout=stop_timeout)
        return self.start_server(comfy_dir, flags, log_path, port=port, python=python)

    # ---------------------------------------------------------- tunnel --

    def current_tunnel(self) -> subprocess.Popen | None:
        """The cloudflared process from the last start_tunnel(), if still alive."""
        if _TUNNEL is not None and _TUNNEL.poll() is None:
            return _TUNNEL
        return None

    def stop_tunnel(self, timeout: float = 10.0) -> bool:
        """Terminate the tunnel from the last start_tunnel().

        Idempotent: returns False without raising if no tunnel is running.
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

    def start_tunnel(self, port: int, log_path: str, timeout: float = 40.0,
                     interval: float = 1.0) -> str | None:
        """Start cloudflared and scrape the public URL out of its log.

        Call only after wait_ready() returns True.

        Any tunnel from a previous call is stopped first: the log is
        truncated here, so leaving the old process holding an fd on it
        would orphan a second cloudflared on the same port with no way to
        reach it from the notebook.
        """
        global _TUNNEL
        self.stop_tunnel()
        if os.path.exists(log_path):
            os.remove(log_path)
        try:
            # Closed via `with` in every path, including if Popen raises --
            # the old code left this fd open on the FileNotFoundError path.
            with open(log_path, "wb") as log:
                proc = subprocess.Popen(
                    [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
        except FileNotFoundError:
            return None
        # Recorded before the scrape loop so a caller can always stop what
        # was started, including on the timeout path below.
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

    # ------------------------------------------------------------ close --

    def close(self) -> None:
        """Stop both the server and the tunnel. Idempotent -- safe to call
        more than once, and registered via atexit so a dying kernel doesn't
        orphan either process."""
        self.stop_server()
        self.stop_tunnel()


# Backs the module-level functions below. Shared across the process so the
# generated notebook, which only ever calls the free functions, gets the
# hardened lifecycle (single-server guard, closed log handles, atexit
# cleanup) without having to know RuntimeSupervisor exists.
_SUPERVISOR = RuntimeSupervisor()


def start_server(comfy_dir: str, flags: list[str], log_path: str,
                 port: int = 8188, python: str | None = None) -> subprocess.Popen:
    """Launch main.py detached, stdout+stderr to log_path. Returns at once."""
    return _SUPERVISOR.start_server(comfy_dir, flags, log_path, port=port, python=python)


def current_tunnel() -> subprocess.Popen | None:
    """The cloudflared process from the last start_tunnel(), if still alive."""
    return _SUPERVISOR.current_tunnel()


def stop_tunnel(timeout: float = 10.0) -> bool:
    """Terminate the tunnel from the last start_tunnel().

    Returns True if a live process was stopped. Safe to call when no tunnel
    is running.
    """
    return _SUPERVISOR.stop_tunnel(timeout=timeout)


def start_tunnel(port: int, log_path: str, timeout: float = 40.0,
                 interval: float = 1.0) -> str | None:
    """Start cloudflared and scrape the public URL out of its log.

    Call only after ComfyClient.wait_ready() returns True.

    Any tunnel from a previous call is stopped first: the log is truncated
    here, so leaving the old process holding an fd on it would orphan a second
    cloudflared on the same port with no way to reach it from the notebook.
    """
    return _SUPERVISOR.start_tunnel(port, log_path, timeout=timeout, interval=interval)


def tail(log_path: str, n: int = 40) -> str:
    """Last n lines of a logfile. Empty string if it does not exist yet."""
    try:
        with open(log_path, "r", errors="ignore") as fh:
            return "\n".join(fh.read().splitlines()[-n:])
    except FileNotFoundError:
        return ""
