"""Map detected GPU VRAM to a model profile, resolution ceiling and
ComfyUI launch flags.

Colab reassigns GPU tiers without warning, so nothing here may be
hardcoded to a subscription level. Thresholds are calibration targets
(spec E4) -- adjust the constants, not the structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FLUX_DISK_GB = 16.06     # flux1-dev-fp8.safetensors, HEAD-verified
DISK_HEADROOM_GB = 8.0   # room for outputs, pip wheels, HF temp files

LOW_MAX_VRAM = 12.0
MID_MAX_VRAM = 20.0


@dataclass(frozen=True)
class Advice:
    tier: str
    profile: str
    max_side: int
    launch_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def recommend(vram_gb: float, disk_free_gb: float = 100.0) -> Advice:
    notes: list[str] = []

    if vram_gb < LOW_MAX_VRAM:
        tier, profile, max_side = "low", "sdxl", 768
        flags = ["--normalvram"]
        notes.append(
            f"{vram_gb:.1f} GB VRAM is tight for SDXL. Capped at 768px; "
            "expect OOM at 1024 with a large batch."
        )
    elif vram_gb < MID_MAX_VRAM:
        tier, profile, max_side = "mid", "sdxl", 1024
        flags = ["--normalvram"]
        notes.append(
            "SDXL at 1024px is comfortable. Flux fp8 (16 GB) will not fit "
            "alongside activations -- not offered at this tier."
        )
    else:
        tier, profile, max_side = "high", "flux-dev", 1024
        flags = ["--highvram"]
        notes.append("Enough VRAM for Flux dev fp8 and SDXL at 1024px.")

    if profile.startswith("flux") and disk_free_gb < FLUX_DISK_GB + DISK_HEADROOM_GB:
        notes.append(
            f"Only {disk_free_gb:.0f} GB disk free; Flux needs "
            f"{FLUX_DISK_GB:.0f} GB plus headroom. Falling back to SDXL."
        )
        profile = "sdxl"

    return Advice(tier=tier, profile=profile, max_side=max_side,
                  launch_flags=flags, notes=notes)
