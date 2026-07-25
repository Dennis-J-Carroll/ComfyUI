"""Download ModelSpecs into a ComfyUI models/ tree.

Uses local_dir= so huggingface_hub writes straight to the destination.
The previous implementation downloaded to the HF cache and then copied,
doubling disk use -- fatal for a 16 GB checkpoint on a Colab VM.
"""
from __future__ import annotations

import os
import shutil
from typing import Callable

from huggingface_hub import hf_hub_download

from colab_studio.registry import ModelSpec

Emit = Callable[[str], None]


def _noop(_: str) -> None:
    return None


def download(spec: ModelSpec, models_dir: str, emit: Emit | None = None) -> str:
    """Fetch one spec. Returns the final on-disk path. Idempotent."""
    log = emit or _noop
    dest_dir = os.path.join(models_dir, spec.dest_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    final = os.path.join(dest_dir, spec.target_name)

    if os.path.exists(final):
        log(f"[=] {spec.target_name} already present, skipping")
        return final

    log(f"[+] {spec.target_name} ({spec.size_gb:.2f} GB) from {spec.repo}")
    got = hf_hub_download(
        repo_id=spec.repo,
        filename=spec.filename,
        local_dir=dest_dir,
    )

    # Nested filenames land in a subtree, and diffusers-layout repos all use
    # the same generic name; flatten and rename to the target.
    if os.path.abspath(got) != os.path.abspath(final):
        os.replace(got, final)
        stray = os.path.dirname(got)
        while os.path.abspath(stray) != os.path.abspath(dest_dir):
            try:
                os.rmdir(stray)
            except OSError:
                break
            stray = os.path.dirname(stray)

    log(f"[v] {spec.target_name} ready")
    return final


def download_all(specs: list[ModelSpec], models_dir: str,
                 emit: Emit | None = None) -> list[str]:
    return [download(s, models_dir, emit) for s in specs]
