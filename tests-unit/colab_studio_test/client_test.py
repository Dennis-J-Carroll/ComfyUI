"""Tests ComfyClient against a real stub HTTP server on a loopback socket.

The stub server itself (Handler, STATE, and the `server` fixture) lives in
conftest.py, shared with telemetry_test.py -- see the comment there.
"""
import contextlib
import copy
import io
import json
import sys
import time

import pytest

from colab_studio.client import ComfyClient, ComfyError, _execution_error


@pytest.fixture(scope="module")
def comfy_execution():
    """ComfyUI's execution module, for asserting against its real types.

    comfy.options.enable_args_parsing() MUST precede the import, and
    model_management probes the device at import time -- hence --cpu.
    """
    sys.argv = ["main.py", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import execution
    return execution


def test_wait_ready_true_when_server_answers(server):
    url, _ = server
    assert ComfyClient(url).wait_ready(timeout=5) is True


def test_wait_ready_false_when_never_ready(server):
    url, state = server
    state["ready"] = False
    assert ComfyClient(url).wait_ready(timeout=1.5) is False


def test_wait_ready_raises_promptly_on_schemeless_base_url():
    with pytest.raises(ComfyError):
        ComfyClient("127.0.0.1:9").wait_ready(timeout=2.0)


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


def test_wait_result_raises_promptly_on_execution_error(server):
    """A runtime failure (OOM, tensor mismatch) leaves outputs empty forever.
    Without reading status, the user waits the full 10 minutes and then gets a
    diagnostic-free TimeoutError."""
    url, state = server
    state["history_error"] = True
    t0 = time.time()
    with pytest.raises(ComfyError) as exc:
        ComfyClient(url).wait_result("pid-123", timeout=30.0)
    assert time.time() - t0 < 10.0, "must fail fast, not wait out the timeout"
    msg = str(exc.value)
    assert "OutOfMemoryError" in msg          # the server's exception_type
    assert "CUDA out of memory" in msg        # the server's exception_message
    assert "KSampler" in msg                  # which node died


def test_wait_result_does_not_mistake_an_in_flight_job_for_a_failure(server):
    """`completed: False` is normal while running; only status_str=="error"
    is a failure."""
    url, state = server
    state["history_ready_after"] = 2
    assert ComfyClient(url).wait_result("pid-123", timeout=10)


def test_generate_surfaces_execution_errors(server):
    url, state = server
    state["history_error"] = True
    with pytest.raises(ComfyError):
        ComfyClient(url).generate({"1": {"class_type": "X", "inputs": {}}},
                                  timeout=30.0)


def test_error_parser_matches_comfyui_own_execution_status(comfy_execution):
    """Binds _execution_error to ComfyUI's real type, not to our stub.

    conftest.py's ERROR_HISTORY is hand-written; if it drifted from what the server
    actually sends, every error test would still pass while the fix silently
    never fired. This builds a genuine PromptQueue.ExecutionStatus, mirrors
    what execution.py's task_done() stores (_asdict into the history entry),
    round-trips it through JSON exactly as the HTTP layer does, and asserts we
    extract the detail.

    Contract (execution.py:1140, main.py:237):
      status_str='success'|'error'; completed=e.success (so False on error --
      which is why only status_str may be used as the error signal);
      messages=[(event, data), ...]; execution_error data carries
      node_id/node_type/exception_type/exception_message.
    """
    status = comfy_execution.PromptQueue.ExecutionStatus(
        status_str="error",
        completed=False,
        messages=[("execution_start", {"prompt_id": "pid-123"}),
                  ("execution_error", {
                      "prompt_id": "pid-123", "node_id": "5",
                      "node_type": "KSampler",
                      "exception_type": "torch.cuda.OutOfMemoryError",
                      "exception_message": "CUDA out of memory",
                      "traceback": [], "executed": []})],
    )
    hist = json.loads(json.dumps(
        {"prompt": [], "outputs": {}, "status": copy.deepcopy(status._asdict())}))

    detail = _execution_error(hist)
    assert detail is not None, "real ExecutionStatus not recognised as an error"
    assert "torch.cuda.OutOfMemoryError" in detail
    assert "CUDA out of memory" in detail
    assert "KSampler" in detail

    # And the success shape must NOT look like an error.
    ok = comfy_execution.PromptQueue.ExecutionStatus(
        status_str="success", completed=True, messages=[])
    assert _execution_error(
        json.loads(json.dumps({"status": copy.deepcopy(ok._asdict())}))) is None


def test_error_parser_tolerates_a_missing_or_null_status():
    """task_done() stores status=None when it is given none (execution.py:1152),
    and an in-flight entry has no status yet. Neither may raise."""
    assert _execution_error({}) is None
    assert _execution_error({"status": None}) is None
    assert _execution_error({"status": {}}) is None


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


# --- E3: system_stats() and the on_poll telemetry hook -------------------

def test_system_stats_returns_parsed_body(server):
    url, _ = server
    stats = ComfyClient(url).system_stats()
    assert stats["system"]["comfyui_version"] == "0.10.0"
    assert stats["devices"][0]["name"] == "NVIDIA A100-SXM4-40GB"


def test_system_stats_raises_comfyerror_on_non_200(server):
    url, state = server
    state["ready"] = False
    with pytest.raises(ComfyError):
        ComfyClient(url).system_stats()


def test_on_poll_is_invoked_once_per_poll_iteration(server):
    """wait_result's poll loop is the natural sampling point -- on_poll must
    fire exactly once per pass through it, not once total or once per
    history hit of some other multiple."""
    url, state = server
    state["history_ready_after"] = 3
    calls = []
    ComfyClient(url).wait_result("pid-123", timeout=10, on_poll=lambda: calls.append(1))
    assert len(calls) == state["history_hits"]
    assert len(calls) >= 3


def test_on_poll_raising_does_not_abort_generation(server):
    """A telemetry failure must never be able to kill a render. wait_result
    must swallow an on_poll exception and keep polling until real outputs
    appear."""
    url, state = server
    state["history_ready_after"] = 2

    def boom():
        raise RuntimeError("telemetry backend exploded")

    refs = ComfyClient(url).wait_result("pid-123", timeout=10, on_poll=boom)
    assert refs[0]["filename"] == "out_001.png"


def test_generate_threads_on_poll_through_to_wait_result(server):
    url, _ = server
    calls = []
    out = ComfyClient(url).generate(
        {"1": {"class_type": "X", "inputs": {}}}, timeout=10,
        on_poll=lambda: calls.append(1))
    assert out == [b"PNG!"]
    assert calls, "generate(on_poll=...) never reached wait_result's loop"
