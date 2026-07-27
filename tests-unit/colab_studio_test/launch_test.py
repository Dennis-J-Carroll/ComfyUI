import os
import subprocess
import sys
import time

import pytest

from colab_studio import launch


@pytest.fixture(autouse=True)
def _no_leaked_tunnel_handle():
    """launch._TUNNEL is module state; leaking it between tests would let one
    test's fake process be terminated by the next."""
    launch._TUNNEL = None
    yield
    launch._TUNNEL = None


class FakeTunnel:
    """Stands in for the cloudflared Popen."""

    def __init__(self, alive=True):
        self.returncode = None if alive else 0
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


def test_start_server_returns_immediately_and_process_lives(tmp_path):
    """The regression guard for the original bug: launching must not block."""
    fake = tmp_path / "main.py"
    fake.write_text("import time\nprint('booting', flush=True)\ntime.sleep(30)\n")
    log = tmp_path / "server.log"

    t0 = time.time()
    proc = launch.start_server(str(tmp_path), [], str(log), python=sys.executable)
    elapsed = time.time() - t0
    try:
        assert elapsed < 5.0, "start_server blocked; it must background the process"
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_start_server_writes_stdout_to_logfile(tmp_path):
    fake = tmp_path / "main.py"
    fake.write_text("print('hello from server', flush=True)\n")
    log = tmp_path / "server.log"
    proc = launch.start_server(str(tmp_path), [], str(log), python=sys.executable)
    proc.wait(timeout=15)
    assert "hello from server" in log.read_text()


def test_start_server_passes_flags_through(tmp_path):
    fake = tmp_path / "main.py"
    fake.write_text("import sys\nprint(' '.join(sys.argv[1:]), flush=True)\n")
    log = tmp_path / "server.log"
    proc = launch.start_server(str(tmp_path), ["--highvram"], str(log),
                               port=9999, python=sys.executable)
    proc.wait(timeout=15)
    out = log.read_text()
    assert "--highvram" in out
    assert "--port 9999" in out


def test_tail_returns_last_n_lines(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(f"line{i}" for i in range(100)))
    out = launch.tail(str(log), n=5)
    assert "line99" in out
    assert "line50" not in out


def test_tail_handles_missing_file(tmp_path):
    assert launch.tail(str(tmp_path / "nope.log")) == ""


def test_start_tunnel_extracts_url_from_log(tmp_path, monkeypatch):
    log = tmp_path / "cf.log"

    def fake_popen(cmd, **kwargs):
        log.write_text(
            "INF Requesting new quick tunnel...\n"
            "INF +----------------------------------+\n"
            "INF |  https://fuzzy-panda-42.trycloudflare.com  |\n"
        )
        class P:
            def poll(self):
                return None
        return P()

    monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
    url = launch.start_tunnel(8188, str(log), timeout=5)
    assert url == "https://fuzzy-panda-42.trycloudflare.com"


def test_start_tunnel_returns_none_on_timeout(tmp_path, monkeypatch):
    log = tmp_path / "cf.log"
    log.write_text("INF starting...\n")

    class P:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    procs = []

    def fake_popen(cmd, **kwargs):
        p = P()
        procs.append(p)
        return p

    monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
    assert launch.start_tunnel(8188, str(log), timeout=2) is None
    assert procs[0].terminated is True


def test_start_tunnel_returns_none_when_cloudflared_missing(tmp_path, monkeypatch):
    log = tmp_path / "cf.log"

    def fake_popen(cmd, **kwargs):
        raise FileNotFoundError("no such file: cloudflared")

    monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
    assert launch.start_tunnel(8188, str(log), timeout=2) is None
    assert launch.current_tunnel() is None


def _tunnel_popen(log, procs, write_url=True):
    def fake_popen(cmd, **kwargs):
        if write_url:
            log.write_text("INF |  https://fuzzy-panda-42.trycloudflare.com  |\n")
        p = FakeTunnel()
        procs.append(p)
        return p
    return fake_popen


def test_start_tunnel_exposes_the_process_for_shutdown(tmp_path, monkeypatch):
    """start_tunnel returned only a URL, so the notebook held no handle on
    cloudflared and could never stop it."""
    log, procs = tmp_path / "cf.log", []
    monkeypatch.setattr(launch.subprocess, "Popen", _tunnel_popen(log, procs))

    assert launch.start_tunnel(8188, str(log), timeout=5)
    assert launch.current_tunnel() is procs[0]


def test_stop_tunnel_terminates_the_running_tunnel(tmp_path, monkeypatch):
    log, procs = tmp_path / "cf.log", []
    monkeypatch.setattr(launch.subprocess, "Popen", _tunnel_popen(log, procs))
    launch.start_tunnel(8188, str(log), timeout=5)

    assert launch.stop_tunnel() is True
    assert procs[0].terminated is True
    assert launch.current_tunnel() is None


def test_stop_tunnel_is_a_noop_when_nothing_is_running():
    assert launch.stop_tunnel() is False


def test_stop_tunnel_kills_a_tunnel_that_ignores_terminate(monkeypatch):
    class Stubborn(FakeTunnel):
        def terminate(self):
            self.terminated = True      # stays alive: returncode untouched

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("cloudflared", timeout)

    launch._TUNNEL = proc = Stubborn()
    assert launch.stop_tunnel(timeout=0.1) is True
    assert proc.killed is True


def test_retunnelling_stops_the_previous_cloudflared(tmp_path, monkeypatch):
    """Re-tunnel used to delete the log the first cloudflared still held an fd
    on and spawn a second process on the same port, leaving both alive."""
    log, procs = tmp_path / "cf.log", []
    monkeypatch.setattr(launch.subprocess, "Popen", _tunnel_popen(log, procs))

    launch.start_tunnel(8188, str(log), timeout=5)
    launch.start_tunnel(8188, str(log), timeout=5)

    assert len(procs) == 2
    assert procs[0].terminated is True, "first tunnel orphaned"
    assert launch.current_tunnel() is procs[1]


def test_start_tunnel_clears_the_handle_when_it_times_out(tmp_path, monkeypatch):
    log, procs = tmp_path / "cf.log", []
    log.write_text("INF starting...\n")
    monkeypatch.setattr(launch.subprocess, "Popen",
                        _tunnel_popen(log, procs, write_url=False))

    assert launch.start_tunnel(8188, str(log), timeout=2) is None
    assert procs[0].terminated is True
    assert launch.current_tunnel() is None


# --------------------------------------------------------------------------
# RuntimeSupervisor (item 0.e)
# --------------------------------------------------------------------------

SLEEPY_MAIN = "import time\ntime.sleep(30)\n"


def _sleepy(tmp_path):
    """A fake main.py that stays alive until stopped."""
    (tmp_path / "main.py").write_text(SLEEPY_MAIN)
    return str(tmp_path / "server.log")


def test_supervisor_closes_parent_log_handle_after_start_server(tmp_path, monkeypatch):
    """The regression guard for the fd-leak bug: start_server's own `open()`
    handle must be closed once Popen has duplicated it for the child,
    regardless of how the caller happens to check."""
    log_path = _sleepy(tmp_path)
    opened = []
    real_open = open

    def spy_open(path, *a, **kw):
        fh = real_open(path, *a, **kw)
        if os.fspath(path) == log_path:
            opened.append(fh)
        return fh

    monkeypatch.setattr(launch, "open", spy_open, raising=False)
    sup = launch.RuntimeSupervisor()
    proc = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
    try:
        assert len(opened) == 1
        assert opened[0].closed is True
    finally:
        sup.stop_server(timeout=10)
        proc.wait(timeout=10)


def test_second_start_server_returns_existing_process_without_spawning_new_one(tmp_path, monkeypatch):
    log_path = _sleepy(tmp_path)
    spawned = []
    real_popen = subprocess.Popen

    def counting_popen(*a, **kw):
        p = real_popen(*a, **kw)
        spawned.append(p)
        return p

    monkeypatch.setattr(launch.subprocess, "Popen", counting_popen)
    sup = launch.RuntimeSupervisor()
    try:
        p1 = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
        p2 = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
        assert p2 is p1
        assert len(spawned) == 1
    finally:
        sup.stop_server(timeout=10)


def test_stop_server_on_never_started_supervisor_returns_false_and_does_not_raise():
    sup = launch.RuntimeSupervisor()
    assert sup.stop_server() is False


def test_start_stop_start_leaves_no_leaked_process(tmp_path):
    log_path = _sleepy(tmp_path)
    sup = launch.RuntimeSupervisor()

    p1 = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
    assert sup.stop_server(timeout=10) is True
    p1.wait(timeout=10)
    assert p1.poll() is not None, "first server was not actually reaped"

    p2 = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
    try:
        assert p2 is not p1
        assert p2.poll() is None, "second server did not start"
    finally:
        assert sup.stop_server(timeout=10) is True
        p2.wait(timeout=10)
        assert p2.poll() is not None, "second server was not actually reaped"


def test_restart_server_stops_the_old_process_and_starts_a_new_one(tmp_path):
    log_path = _sleepy(tmp_path)
    sup = launch.RuntimeSupervisor()

    p1 = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
    p2 = sup.restart_server(str(tmp_path), [], log_path, python=sys.executable)
    try:
        assert p2 is not p1
        p1.wait(timeout=10)
        assert p1.poll() is not None, "old process was not stopped"
        assert p2.poll() is None, "new process did not start"
    finally:
        sup.stop_server(timeout=10)


def test_close_is_idempotent(tmp_path):
    log_path = _sleepy(tmp_path)
    sup = launch.RuntimeSupervisor()
    proc = sup.start_server(str(tmp_path), [], log_path, python=sys.executable)
    sup.close()
    sup.close()  # must not raise the second time
    proc.wait(timeout=10)
    assert proc.poll() is not None


def test_atexit_hook_is_registered(monkeypatch):
    registered = []
    monkeypatch.setattr(launch.atexit, "register", lambda fn: registered.append(fn))
    sup = launch.RuntimeSupervisor()
    assert sup.close in registered


def test_wait_ready_delegates_to_the_given_client():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def wait_ready(self, timeout=180.0, interval=1.0):
            self.calls.append(timeout)
            return True

    sup = launch.RuntimeSupervisor()
    client = FakeClient()
    assert sup.wait_ready(timeout=5, client=client) is True
    assert client.calls == [5]


def test_wait_ready_builds_a_client_for_its_own_port_by_default(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, base_url):
            seen["base_url"] = base_url

        def wait_ready(self, timeout=180.0, interval=1.0):
            seen["timeout"] = timeout
            return False

    monkeypatch.setattr(launch, "ComfyClient", FakeClient)
    sup = launch.RuntimeSupervisor()
    sup._port = 9999
    assert sup.wait_ready(timeout=7) is False
    assert seen == {"base_url": "http://127.0.0.1:9999", "timeout": 7}


def test_module_level_start_server_still_works_and_returns_a_live_popen(tmp_path):
    """Backward compatibility: the module-level wrapper keeps its original
    signature and behaviour, even though it now delegates to a supervisor."""
    log_path = _sleepy(tmp_path)
    proc = launch.start_server(str(tmp_path), [], log_path, python=sys.executable)
    try:
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        launch._SUPERVISOR.stop_server(timeout=10)
