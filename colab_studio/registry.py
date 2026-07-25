"""Model registry. Pure data -- no network, no filesystem.

Every entry below was HEAD-verified against huggingface.co on 2026-07-25.
Sizes are real, not estimates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    filename: str
    dest_subdir: str          # relative to ComfyUI models/
    size_gb: float
    dest_filename: str | None = None

    @property
    def target_name(self) -> str:
        """Filename as written to disk. Diffusers-layout repos all use the
        same generic filename, so those entries must rename."""
        return self.dest_filename or os.path.basename(self.filename)


UPSCALER = ModelSpec(
    repo="Kim2091/UltraSharp",
    filename="4x-UltraSharp.pth",
    dest_subdir="upscale_models",
    size_gb=0.06,
)

CONTROLNET_CANNY = ModelSpec(
    repo="diffusers/controlnet-canny-sdxl-1.0",
    filename="diffusion_pytorch_model.fp16.safetensors",
    dest_subdir="controlnet",
    size_gb=2.33,
    dest_filename="controlnet-canny-sdxl.safetensors",
)

PROFILES: dict[str, list[ModelSpec]] = {
    "sdxl": [
        ModelSpec(
            repo="stabilityai/stable-diffusion-xl-base-1.0",
            filename="sd_xl_base_1.0.safetensors",
            dest_subdir="checkpoints",
            size_gb=6.46,
        ),
        ModelSpec(
            repo="stabilityai/sdxl-vae",
            filename="sdxl_vae.safetensors",
            dest_subdir="vae",
            size_gb=0.31,
        ),
    ],
    # fp8 all-in-one: UNet + T5 + CLIP-L + VAE in one file, so it loads
    # via CheckpointLoaderSimple rather than a 3-loader split.
    "flux-dev": [
        ModelSpec(
            repo="Comfy-Org/flux1-dev",
            filename="flux1-dev-fp8.safetensors",
            dest_subdir="checkpoints",
            size_gb=16.06,
        ),
    ],
    "flux-schnell": [
        ModelSpec(
            repo="Comfy-Org/flux1-schnell",
            filename="flux1-schnell-fp8.safetensors",
            dest_subdir="checkpoints",
            size_gb=16.05,
        ),
    ],
}

CHECKPOINT_NAME: dict[str, str] = {
    "sdxl": "sd_xl_base_1.0.safetensors",
    "flux-dev": "flux1-dev-fp8.safetensors",
    "flux-schnell": "flux1-schnell-fp8.safetensors",
}


def resolve(profile: str, controlnet: bool = False,
            upscale: bool = True) -> list[ModelSpec]:
    """Full download list for a profile. Raises KeyError on unknown profile."""
    specs = list(PROFILES[profile])
    if upscale:
        specs.append(UPSCALER)
    if controlnet:
        specs.append(CONTROLNET_CANNY)
    return specs


def total_gb(specs: list[ModelSpec]) -> float:
    return round(sum(s.size_gb for s in specs), 2)
