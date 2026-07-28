"""Observed-peak VRAM sampling for a running ComfyUI server subprocess.

The diffusion model runs in ComfyUI's server subprocess, not the notebook
kernel: `torch.cuda.max_memory_allocated()` called from a notebook cell
would see only the kernel's own (empty) CUDA allocator and report ~0. The
only window into the server's memory is the server's own GET
/system_stats endpoint (server.py; `devices[0]` is the primary device;
used bytes = `vram_total - vram_free`).

VramProbe is meant to be driven from ComfyClient.wait_result's `on_poll`
hook -- i.e. sampled once per poll interval (1s by default). That makes
every number this module produces an OBSERVED PEAK AT N-SECOND SAMPLING,
NOT A TRUE MAXIMUM: a spike between two polls is invisible to it, and
model load can peak higher than steady-state sampling ever catches. Never
present or label this number as an exact ceiling -- `describe()` below
always states the sampling interval and the word "observed" for exactly
this reason, and any other surface printing this value must do the same.
"""
from __future__ import annotations

from dataclasses import dataclass

from colab_studio.client import ComfyClient

_BYTES_PER_GB = 2**30


@dataclass(frozen=True)
class VramSummary:
    """A snapshot of what VramProbe has observed so far. See the module
    docstring: `peak_used_gb` is a polled sample, not a true maximum."""
    device_name: str | None
    peak_used_gb: float
    total_gb: float
    percent_used: float
    sample_count: int


class VramProbe:
    """Polled VRAM sampler, driven externally (e.g. by `on_poll`).

    Degrades silently in every case that isn't a genuine reading: no
    devices reported, a non-GPU device (`type == "cpu"`), a response
    missing the expected keys, or a request that fails outright all just
    mean "no sample was recorded" -- none of them raise. A telemetry
    failure must never be able to interrupt a render.

    All I/O goes through the injected ComfyClient, so this is testable
    against a plain stub HTTP server with no GPU involved.
    """

    def __init__(self, client: ComfyClient) -> None:
        self._client = client
        self._device_name: str | None = None
        self._total_bytes = 0
        self._peak_used_bytes = 0
        self._sample_count = 0

    def sample(self) -> None:
        """Record one observation. Never raises."""
        try:
            stats = self._client.system_stats()
            devices = stats.get("devices") or []
            if not devices:
                return
            device = devices[0]
            if not isinstance(device, dict) or device.get("type") == "cpu":
                return
            total = device["vram_total"]
            free = device["vram_free"]
            used = total - free
            name = device.get("name")
        except Exception:
            return

        self._device_name = name or self._device_name
        self._total_bytes = total
        self._peak_used_bytes = max(self._peak_used_bytes, used)
        self._sample_count += 1

    def summary(self) -> VramSummary:
        """The peak observed so far. `sample_count == 0` means no usable
        reading was ever taken -- report that honestly, don't fake a zero."""
        if self._sample_count == 0:
            return VramSummary(device_name=None, peak_used_gb=0.0,
                               total_gb=0.0, percent_used=0.0, sample_count=0)
        percent = (self._peak_used_bytes / self._total_bytes * 100.0
                  if self._total_bytes else 0.0)
        return VramSummary(
            device_name=self._device_name,
            peak_used_gb=self._peak_used_bytes / _BYTES_PER_GB,
            total_gb=self._total_bytes / _BYTES_PER_GB,
            percent_used=percent,
            sample_count=self._sample_count,
        )


def describe(summary: VramSummary, interval_s: float = 1.0) -> str:
    """One readable line for a summary. Always names this an *observed*
    peak and states the sampling interval -- see the module docstring for
    why that qualifier may never be dropped."""
    if summary.sample_count == 0:
        return ("no VRAM samples were taken (no GPU reported by the server, "
                "or /system_stats was unavailable) -- peak VRAM is "
                "unmeasured for this run.")
    return (
        f"observed peak VRAM (sampled every {interval_s:g}s): "
        f"{summary.peak_used_gb:.1f} / {summary.total_gb:.1f} GB "
        f"({summary.percent_used:.0f}%) on {summary.device_name}, "
        f"{summary.sample_count} samples"
    )
