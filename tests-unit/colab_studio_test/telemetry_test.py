"""Tests VramProbe against the same stub HTTP server client_test.py uses,
per the brief: no second stub. The `server` fixture and its STATE dict
(with a configurable `devices` list) live in conftest.py, shared by both
test modules with no import needed -- that's what makes it one stub.
"""
from __future__ import annotations

import pytest

from colab_studio.client import ComfyClient
from colab_studio.telemetry import VramProbe, describe


def test_probe_records_max_across_several_differing_samples_not_last_or_first(server):
    url, state = server
    client = ComfyClient(url)
    probe = VramProbe(client)

    # 10 GB used, then 30 GB used (the max), then 5 GB used (the last).
    total = state["devices"][0]["vram_total"]
    for used_gb in (10, 30, 5):
        state["devices"][0]["vram_free"] = total - used_gb * 2**30
        probe.sample()

    summary = probe.summary()
    assert summary.sample_count == 3
    assert summary.peak_used_gb == pytest.approx(30.0, abs=0.01)


def test_probe_with_cpu_type_device_records_no_samples(server):
    """Isolates the `type == "cpu"` guard itself: this device otherwise has
    perfectly well-formed vram_total/vram_free fields, so a missing-keys
    fallback alone would not explain skipping it."""
    url, state = server
    state["devices"] = [{"name": "CPU", "type": "cpu",
                         "vram_total": 16_000_000_000,
                         "vram_free": 8_000_000_000}]
    probe = VramProbe(ComfyClient(url))
    probe.sample()
    summary = probe.summary()
    assert summary.sample_count == 0
    assert summary.peak_used_gb == 0.0


def test_probe_with_no_devices_records_no_samples(server):
    url, state = server
    state["devices"] = []
    probe = VramProbe(ComfyClient(url))
    probe.sample()
    assert probe.summary().sample_count == 0


def test_probe_with_missing_vram_keys_records_no_samples(server):
    url, state = server
    state["devices"] = [{"name": "Weird GPU", "type": "cuda"}]  # no vram_total/free
    probe = VramProbe(ComfyClient(url))
    probe.sample()
    assert probe.summary().sample_count == 0


def test_probe_with_failing_client_records_no_samples_and_does_not_raise():
    """No server listening at all -- the request itself fails."""
    probe = VramProbe(ComfyClient("http://127.0.0.1:1"))
    probe.sample()  # must not raise
    summary = probe.summary()
    assert summary.sample_count == 0
    assert summary.peak_used_gb == 0.0


def test_probe_with_server_that_is_never_ready_records_no_samples(server):
    url, state = server
    state["ready"] = False
    probe = VramProbe(ComfyClient(url))
    probe.sample()
    assert probe.summary().sample_count == 0


def test_summary_reports_device_name_total_gb_and_percent(server):
    url, state = server
    probe = VramProbe(ComfyClient(url))
    probe.sample()
    summary = probe.summary()
    assert summary.device_name == "NVIDIA A100-SXM4-40GB"
    assert summary.total_gb == pytest.approx(40.0, abs=0.01)
    assert summary.peak_used_gb == pytest.approx(20.0, abs=0.01)
    assert summary.percent_used == pytest.approx(50.0, abs=0.5)
    assert summary.sample_count == 1


def test_describe_formats_a_readable_line_with_the_sampling_qualifier(server):
    url, _ = server
    probe = VramProbe(ComfyClient(url))
    probe.sample()
    line = describe(probe.summary(), interval_s=1.0)
    assert "NVIDIA A100-SXM4-40GB" in line
    assert "20.0" in line and "40.0" in line
    assert "observed" in line.lower()
    assert "sampl" in line.lower()  # "sampled"/"sampling" + the count


def test_describe_is_honest_when_no_samples_were_taken():
    probe = VramProbe(ComfyClient("http://127.0.0.1:1"))
    line = describe(probe.summary())
    assert "0" not in line or "no" in line.lower()
    assert "unmeasured" in line.lower() or "no vram sample" in line.lower()
