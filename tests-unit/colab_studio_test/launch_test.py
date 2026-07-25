import sys
import time

from colab_studio import launch


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
