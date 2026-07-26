# ComfyUI Colab Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Colab notebook that runs SDXL/Flux image workflows (and optionally Wan 2.2 video) end to end, with in-notebook guidance so the user never leaves Colab.

**Architecture:** All logic lives in a tested `colab_studio/` Python package in this repo. A build script (`build_notebook.py`) inlines those tested modules into `%%writefile` cells to generate `ComfyUI_Colab_Studio.ipynb`. This gives one source of truth: the code that ships is the code the tests exercise. The notebook talks to ComfyUI over HTTP only — it never imports `comfy`.

**Tech Stack:** Python 3.10+, pytest, `requests`, `huggingface_hub`, ComfyUI 0.10.0, cloudflared.

**Spec:** `docs/superpowers/specs/2026-07-25-comfyui-colab-studio-design.md`

## Global Constraints

- **No `print()` anywhere in `colab_studio/**`.** `pyproject.toml:19` selects ruff rule `T` (print-usage) repo-wide. Library functions return data or accept an `emit` callable; only notebook cells print. `pyproject.toml:28` excludes `*.ipynb`, so generated notebook cells are exempt.
- **Python 3.10 compatible.** `pyproject.toml:31` pins pylint `py-version = "3.10"`. Every module starts with `from __future__ import annotations` so `X | None` annotations are legal.
- **The notebook must never reference MCP.** `grep -ri "mcp" ComfyUI_Colab_Studio.ipynb` must return nothing (spec criterion 8).
- **The notebook kernel must never `import comfy`.** `comfy/cli_args.py:236` only parses `sys.argv` when `comfy.options.args_parsing` is set, and `comfy/model_management.py:238` probes the GPU at import time. All notebook→server communication is HTTP.
- **Tests live in `tests-unit/colab_studio_test/`**, matching the existing `tests-unit/folder_paths_test/` pattern.
- **Test runner:** `.venv/bin/python -m pytest`.
- Verified model facts (do not re-guess): SDXL is `stabilityai/stable-diffusion-xl-base-1.0` (6.46 GB) — `Comfy-Org/stable-diffusion-xl-base-1.0` returns 401 and does not exist. Flux dev fp8 is `Comfy-Org/flux1-dev` / `flux1-dev-fp8.safetensors`, **16.06 GB, all-in-one**, loads via `CheckpointLoaderSimple`. ControlNet is `diffusers/controlnet-canny-sdxl-1.0` / `diffusion_pytorch_model.fp16.safetensors` (2.33 GB), must be renamed on download.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `colab_studio/__init__.py` | Package marker. Empty. |
| `colab_studio/registry.py` | `ModelSpec` dataclass + model table + `resolve()`. Pure data, no I/O. |
| `colab_studio/advice.py` | GPU VRAM → recommended profile / resolution / launch flags. Pure function. (Spec E4) |
| `colab_studio/workflows.py` | API-format graph builders. Pure dict construction, no I/O. |
| `colab_studio/fetch.py` | Downloads `ModelSpec`s via `huggingface_hub`. |
| `colab_studio/client.py` | HTTP client for a running ComfyUI server. |
| `colab_studio/launch.py` | Backgrounded server launch, readiness poll, cloudflared tunnel. |
| `build_notebook.py` | Assembles `ComfyUI_Colab_Studio.ipynb` from the modules above. |
| `tests-unit/colab_studio_test/*` | pytest suite. |

Split rationale: `registry`/`advice`/`workflows` are pure and fast to test; `fetch`/`client`/`launch` touch the outside world and are tested with stubs. Keeping them apart means the pure logic stays trivially testable.

---

### Task 1: Package scaffolding and model registry

**Files:**
- Create: `colab_studio/__init__.py`, `colab_studio/registry.py`
- Create: `tests-unit/colab_studio_test/__init__.py`, `tests-unit/colab_studio_test/registry_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ModelSpec(repo: str, filename: str, dest_subdir: str, size_gb: float, dest_filename: str | None)`, with property `target_name -> str`. Function `resolve(profile: str, controlnet: bool = False, upscale: bool = True) -> list[ModelSpec]`. Constant `PROFILES: dict[str, list[ModelSpec]]`. Used by Tasks 4, 7.

- [ ] **Step 1: Install pytest into the venv**

```bash
.venv/bin/python -m pip install -r tests-unit/requirements.txt
```

Expected: `pytest` installs successfully.

- [ ] **Step 2: Write the failing test**

Create `tests-unit/colab_studio_test/__init__.py` (empty), then `tests-unit/colab_studio_test/registry_test.py`:

```python
import pytest
from colab_studio.registry import ModelSpec, PROFILES, resolve


def test_modelspec_target_name_defaults_to_basename():
    spec = ModelSpec(repo="a/b", filename="split/x.safetensors",
                     dest_subdir="checkpoints", size_gb=1.0)
    assert spec.target_name == "x.safetensors"


def test_modelspec_target_name_honours_rename():
    spec = ModelSpec(repo="diffusers/controlnet-canny-sdxl-1.0",
                     filename="diffusion_pytorch_model.fp16.safetensors",
                     dest_subdir="controlnet", size_gb=2.33,
                     dest_filename="controlnet-canny-sdxl.safetensors")
    assert spec.target_name == "controlnet-canny-sdxl.safetensors"


def test_sdxl_profile_uses_stabilityai_not_comfy_org():
    # Comfy-Org/stable-diffusion-xl-base-1.0 returns 401 -- it does not exist.
    specs = PROFILES["sdxl"]
    repos = {s.repo for s in specs}
    assert "stabilityai/stable-diffusion-xl-base-1.0" in repos
    assert not any(r.startswith("Comfy-Org/stable-diffusion-xl") for r in repos)


def test_flux_dev_is_single_all_in_one_checkpoint():
    specs = PROFILES["flux-dev"]
    assert len(specs) == 1
    assert specs[0].dest_subdir == "checkpoints"
    assert specs[0].size_gb > 15


def test_resolve_includes_upscaler_by_default():
    names = [s.target_name for s in resolve("sdxl")]
    assert "4x-UltraSharp.pth" in names


def test_resolve_excludes_controlnet_unless_requested():
    without = [s.dest_subdir for s in resolve("sdxl", controlnet=False)]
    assert "controlnet" not in without
    with_cn = [s.dest_subdir for s in resolve("sdxl", controlnet=True)]
    assert "controlnet" in with_cn


def test_resolve_rejects_unknown_profile():
    with pytest.raises(KeyError):
        resolve("not-a-profile")
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/registry_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'colab_studio'`

- [ ] **Step 4: Write minimal implementation**

Create `colab_studio/__init__.py` (empty file), then `colab_studio/registry.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/registry_test.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Verify no print() violations**

```bash
.venv/bin/python -m ruff check colab_studio/ 2>/dev/null || pipx run ruff check colab_studio/
```

Expected: no `T201` violations. If ruff is unavailable, run `grep -rn "print(" colab_studio/` and confirm zero hits.

- [ ] **Step 7: Commit**

```bash
git add colab_studio/ tests-unit/colab_studio_test/
git commit -m "feat(colab): add model registry with verified HF repos and sizes"
```

---

### Task 2: GPU advice heuristic (spec E4)

**Files:**
- Create: `colab_studio/advice.py`
- Create: `tests-unit/colab_studio_test/advice_test.py`

**Interfaces:**
- Consumes: `colab_studio.registry.PROFILES` (validates recommended profile exists).
- Produces: `Advice(tier: str, profile: str, max_side: int, launch_flags: list[str], notes: list[str])` and `recommend(vram_gb: float, disk_free_gb: float = 100.0) -> Advice`. Used by Task 7.

**Context:** This is the one genuinely opinionated piece of the design. Thresholds below are the starting point; spec E4 says to calibrate them against observed hardware. The tests pin *behaviour at the boundaries*, not taste, so recalibration only touches the constants.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/advice_test.py`:

```python
from colab_studio.advice import recommend, Advice
from colab_studio.registry import PROFILES


def test_t4_is_mid_tier():
    # A Colab T4 reports ~14.7 GB, not a clean 16.
    a = recommend(14.7)
    assert a.tier == "mid"
    assert a.profile == "sdxl"


def test_t4_never_gets_highvram():
    # --highvram on a T4 is actively wrong; this is the regression guard.
    assert "--highvram" not in recommend(14.7).launch_flags


def test_l4_is_high_tier_and_gets_flux():
    a = recommend(22.5)
    assert a.tier == "high"
    assert a.profile.startswith("flux")
    assert "--highvram" in a.launch_flags


def test_a100_is_high_tier():
    assert recommend(40.0).tier == "high"


def test_low_vram_drops_resolution_and_avoids_flux():
    a = recommend(8.0)
    assert a.tier == "low"
    assert a.max_side <= 768
    assert not a.profile.startswith("flux")


def test_recommended_profile_always_exists_in_registry():
    for vram in (6.0, 8.0, 14.7, 16.0, 22.5, 40.0, 80.0):
        assert recommend(vram).profile in PROFILES


def test_low_disk_forces_small_profile_even_on_big_gpu():
    # Flux is 16 GB; refusing it when disk is tight prevents a mid-download death.
    a = recommend(40.0, disk_free_gb=12.0)
    assert a.profile == "sdxl"
    assert any("disk" in n.lower() for n in a.notes)


def test_advice_is_frozen():
    a = recommend(22.5)
    assert isinstance(a, Advice)
    try:
        a.tier = "low"          # type: ignore[misc]
        raise AssertionError("Advice should be immutable")
    except AttributeError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/advice_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'colab_studio.advice'`

- [ ] **Step 3: Write minimal implementation**

Create `colab_studio/advice.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/advice_test.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add colab_studio/advice.py tests-unit/colab_studio_test/advice_test.py
git commit -m "feat(colab): add VRAM-based model and launch-flag advice"
```

---

### Task 3: Workflow builders, validated against ComfyUI's real validator

**Files:**
- Create: `colab_studio/workflows.py`
- Create: `tests-unit/colab_studio_test/workflows_test.py`

**Interfaces:**
- Consumes: `colab_studio.registry.CHECKPOINT_NAME`.
- Produces: `sdxl_txt2img()`, `flux_txt2img()`, `img2img()`, `upscale()`, `controlnet_canny()`, each returning `dict[str, dict]` in ComfyUI API format. Shared signature: `(ckpt: str, prompt: str, negative: str = "", seed: int = 0, steps: int = 25, cfg: float = 7.0, width: int = 1024, height: int = 1024, batch: int = 1)`. Extra params noted per function. Used by Tasks 5, 7.

**Context:** The test harness calls `execution.validate_prompt` — the exact code path `POST /prompt` uses. It checks class names, input key names, enum membership and link types. Dummy model files are registered via `folder_paths.add_model_folder_path` so ENUM checks pass without downloading anything. `comfy.options.enable_args_parsing()` must be called *before* importing `nodes`, or `comfy/model_management.py:238` will try to probe a GPU and crash on a CPU box.

Node input names below were dumped from `INPUT_TYPES()` on ComfyUI 0.10.0 — do not guess them.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/workflows_test.py`:

```python
"""Validates generated graphs against ComfyUI's own prompt validator.

This is structural validation: class names, input keys, enum membership,
link types. It does not execute nodes.
"""
import asyncio
import contextlib
import io
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def comfy(tmp_path_factory):
    """Boot ComfyUI's node registry in CPU mode with dummy models present."""
    sys.argv = ["main.py", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()   # MUST precede `import nodes`

    root = tmp_path_factory.mktemp("models")
    for sub, name in [
        ("checkpoints", "sd_xl_base_1.0.safetensors"),
        ("checkpoints", "flux1-dev-fp8.safetensors"),
        ("vae", "sdxl_vae.safetensors"),
        ("controlnet", "controlnet-canny-sdxl.safetensors"),
        ("upscale_models", "4x-UltraSharp.pth"),
    ]:
        d = root / sub
        d.mkdir(exist_ok=True)
        (d / name).touch()

    probe = os.path.join(REPO, "input", "_wf_probe.png")
    os.makedirs(os.path.join(REPO, "input"), exist_ok=True)
    open(probe, "a").close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import folder_paths
        for sub in ("checkpoints", "vae", "controlnet", "upscale_models"):
            folder_paths.add_model_folder_path(sub, str(root / sub))
        import nodes
        import execution
        r = nodes.init_extra_nodes(init_api_nodes=False)
        if asyncio.iscoroutine(r):
            asyncio.run(r)
    yield execution
    os.remove(probe)


def assert_valid(execution, name, graph):
    res = asyncio.run(execution.validate_prompt(name, graph, None))
    assert res[0], f"{name} invalid: {res[1]} / {res[3]}"


def test_sdxl_txt2img_validates(comfy):
    from colab_studio.workflows import sdxl_txt2img
    assert_valid(comfy, "sdxl", sdxl_txt2img("sd_xl_base_1.0.safetensors", "a cat"))


def test_flux_txt2img_validates(comfy):
    from colab_studio.workflows import flux_txt2img
    assert_valid(comfy, "flux", flux_txt2img("flux1-dev-fp8.safetensors", "a cat"))


def test_flux_forces_cfg_one(comfy):
    # Flux ignores CFG; real guidance rides on FluxGuidance. cfg != 1.0 scorches output.
    from colab_studio.workflows import flux_txt2img
    g = flux_txt2img("flux1-dev-fp8.safetensors", "a cat", cfg=7.0)
    ks = [n for n in g.values() if n["class_type"] == "KSampler"][0]
    assert ks["inputs"]["cfg"] == 1.0
    fg = [n for n in g.values() if n["class_type"] == "FluxGuidance"]
    assert len(fg) == 1


def test_img2img_validates_and_uses_vaeencode(comfy):
    from colab_studio.workflows import img2img
    g = img2img("sd_xl_base_1.0.safetensors", "a cat", image="_wf_probe.png", denoise=0.6)
    assert_valid(comfy, "img2img", g)
    assert any(n["class_type"] == "VAEEncode" for n in g.values())
    assert not any(n["class_type"] == "EmptyLatentImage" for n in g.values())


def test_upscale_validates_and_saves_upscaled_image(comfy):
    from colab_studio.workflows import upscale
    g = upscale("sd_xl_base_1.0.safetensors", "a cat")
    assert_valid(comfy, "upscale", g)
    save = [n for n in g.values() if n["class_type"] == "SaveImage"][0]
    src = save["inputs"]["images"][0]
    assert g[src]["class_type"] == "ImageUpscaleWithModel"


def test_controlnet_validates_and_rewires_both_conditionings(comfy):
    from colab_studio.workflows import controlnet_canny
    g = controlnet_canny("sd_xl_base_1.0.safetensors", "a cat", image="_wf_probe.png")
    assert_valid(comfy, "controlnet", g)
    ks = [n for n in g.values() if n["class_type"] == "KSampler"][0]
    pos, neg = ks["inputs"]["positive"], ks["inputs"]["negative"]
    assert g[pos[0]]["class_type"] == "ControlNetApplyAdvanced"
    assert g[neg[0]]["class_type"] == "ControlNetApplyAdvanced"
    assert pos[1] == 0 and neg[1] == 1


def test_seed_and_size_are_threaded_through(comfy):
    from colab_studio.workflows import sdxl_txt2img
    g = sdxl_txt2img("sd_xl_base_1.0.safetensors", "a cat",
                     seed=1234, width=768, height=512, steps=12)
    ks = [n for n in g.values() if n["class_type"] == "KSampler"][0]
    lat = [n for n in g.values() if n["class_type"] == "EmptyLatentImage"][0]
    assert ks["inputs"]["seed"] == 1234
    assert ks["inputs"]["steps"] == 12
    assert (lat["inputs"]["width"], lat["inputs"]["height"]) == (768, 512)


def test_negative_control_bad_sampler_is_rejected(comfy):
    # Proves the validator is engaged rather than vacuously passing.
    from colab_studio.workflows import sdxl_txt2img
    g = sdxl_txt2img("sd_xl_base_1.0.safetensors", "a cat")
    ks_id = [k for k, n in g.items() if n["class_type"] == "KSampler"][0]
    g[ks_id]["inputs"]["sampler_name"] = "not_a_real_sampler"
    res = asyncio.run(comfy.validate_prompt("bad", g, None))
    assert not res[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/workflows_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'colab_studio.workflows'`

- [ ] **Step 3: Write minimal implementation**

Create `colab_studio/workflows.py`:

```python
"""API-format graph builders for ComfyUI.

API format is a flat dict: {node_id: {"class_type": str, "inputs": {...}}}.
Links are ["source_node_id", output_index] pairs.

Input key names come from INPUT_TYPES() on ComfyUI 0.10.0. They are not
guessable -- change them only against a fresh dump.
"""
from __future__ import annotations

Graph = dict


def _base(ckpt: str, prompt: str, negative: str, seed: int, steps: int,
          cfg: float, sampler: str, scheduler: str, denoise: float,
          prefix: str) -> Graph:
    """Shared spine: checkpoint -> two text encodes -> sampler -> decode -> save.

    Node "4" (the latent source) is deliberately left out; each builder
    supplies either EmptyLatentImage or VAEEncode.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": sampler,
                         "scheduler": scheduler, "positive": ["2", 0],
                         "negative": ["3", 0], "latent_image": ["4", 0],
                         "denoise": denoise}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def _empty_latent(width: int, height: int, batch: int) -> Graph:
    return {"class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch}}


def sdxl_txt2img(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
                 steps: int = 25, cfg: float = 7.0, width: int = 1024,
                 height: int = 1024, batch: int = 1,
                 sampler: str = "dpmpp_2m", scheduler: str = "karras") -> Graph:
    g = _base(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
              1.0, "colab/sdxl")
    g["4"] = _empty_latent(width, height, batch)
    return g


def flux_txt2img(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
                 steps: int = 20, cfg: float = 1.0, width: int = 1024,
                 height: int = 1024, batch: int = 1,
                 guidance: float = 3.5) -> Graph:
    """Flux ignores CFG entirely -- it must be 1.0, with real guidance
    supplied by FluxGuidance. Any other cfg produces scorched output, so the
    parameter is overridden rather than trusted."""
    g = _base(ckpt, prompt, negative, seed, steps, 1.0, "euler", "simple",
              1.0, "colab/flux")
    g["4"] = _empty_latent(width, height, batch)
    g["8"] = {"class_type": "FluxGuidance",
              "inputs": {"conditioning": ["2", 0], "guidance": guidance}}
    g["5"]["inputs"]["positive"] = ["8", 0]
    return g


def img2img(ckpt: str, prompt: str, image: str, negative: str = "",
            seed: int = 0, steps: int = 25, cfg: float = 7.0,
            denoise: float = 0.6, sampler: str = "dpmpp_2m",
            scheduler: str = "karras") -> Graph:
    """`image` is a filename already present in the server's input/ dir --
    upload it first via ComfyClient.upload_image()."""
    g = _base(ckpt, prompt, negative, seed, steps, cfg, sampler, scheduler,
              denoise, "colab/img2img")
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["4"] = {"class_type": "VAEEncode",
              "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}}
    return g


def upscale(ckpt: str, prompt: str, negative: str = "", seed: int = 0,
            steps: int = 25, cfg: float = 7.0, width: int = 1024,
            height: int = 1024, batch: int = 1,
            model_name: str = "4x-UltraSharp.pth",
            sampler: str = "dpmpp_2m", scheduler: str = "karras") -> Graph:
    """txt2img then a pure image-space upscale. No image input, so this is
    the one optional feature needing no upload path."""
    g = sdxl_txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                     batch, sampler, scheduler)
    g["11"] = {"class_type": "UpscaleModelLoader",
               "inputs": {"model_name": model_name}}
    g["12"] = {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["11", 0], "image": ["6", 0]}}
    g["7"]["inputs"]["images"] = ["12", 0]
    g["7"]["inputs"]["filename_prefix"] = "colab/upscale"
    return g


def controlnet_canny(ckpt: str, prompt: str, image: str, negative: str = "",
                     seed: int = 0, steps: int = 25, cfg: float = 7.0,
                     width: int = 1024, height: int = 1024, batch: int = 1,
                     strength: float = 0.8, low_threshold: float = 0.4,
                     high_threshold: float = 0.8,
                     control_net: str = "controlnet-canny-sdxl.safetensors",
                     sampler: str = "dpmpp_2m",
                     scheduler: str = "karras") -> Graph:
    """SDXL only. Canny is a core node -- no comfyui_controlnet_aux needed.

    ControlNetApplyAdvanced emits BOTH conditionings, so the sampler's
    positive and negative must be rewired to outputs 0 and 1 of the same
    node. Rewiring only positive is a silent correctness bug.
    """
    g = sdxl_txt2img(ckpt, prompt, negative, seed, steps, cfg, width, height,
                     batch, sampler, scheduler)
    g["10"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["13"] = {"class_type": "Canny",
               "inputs": {"image": ["10", 0], "low_threshold": low_threshold,
                          "high_threshold": high_threshold}}
    g["14"] = {"class_type": "ControlNetLoader",
               "inputs": {"control_net_name": control_net}}
    g["15"] = {"class_type": "ControlNetApplyAdvanced",
               "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                          "control_net": ["14", 0], "image": ["13", 0],
                          "strength": strength, "start_percent": 0.0,
                          "end_percent": 1.0}}
    g["5"]["inputs"]["positive"] = ["15", 0]
    g["5"]["inputs"]["negative"] = ["15", 1]
    g["7"]["inputs"]["filename_prefix"] = "colab/controlnet"
    return g
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/workflows_test.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add colab_studio/workflows.py tests-unit/colab_studio_test/workflows_test.py
git commit -m "feat(colab): add API-format workflow builders validated against ComfyUI"
```

---

### Task 4: Model fetcher

**Files:**
- Create: `colab_studio/fetch.py`
- Create: `tests-unit/colab_studio_test/fetch_test.py`

**Interfaces:**
- Consumes: `colab_studio.registry.ModelSpec`.
- Produces: `download(spec: ModelSpec, models_dir: str, emit: Callable[[str], None] | None = None) -> str` returning the final path, and `download_all(specs, models_dir, emit=None) -> list[str]`. Used by Task 7.

**Context:** The old `download_models.py` called `hf_hub_download()` then `shutil.copy2()`, writing every model to disk twice — fatal for a 16 GB Flux pull on a Colab VM. Passing `local_dir=` writes straight to the destination. It also passed `resume_download`, deprecated in current `huggingface_hub`.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/fetch_test.py`:

```python
import os
import pytest
from colab_studio.registry import ModelSpec
from colab_studio import fetch


@pytest.fixture
def spec():
    return ModelSpec(repo="a/b", filename="sub/model.safetensors",
                     dest_subdir="checkpoints", size_gb=1.0)


@pytest.fixture
def renamed_spec():
    return ModelSpec(repo="diffusers/cn", filename="diffusion_pytorch_model.fp16.safetensors",
                     dest_subdir="controlnet", size_gb=2.33,
                     dest_filename="controlnet-canny-sdxl.safetensors")


def test_download_uses_local_dir_and_never_copies(monkeypatch, tmp_path, spec):
    calls = {}

    def fake_hub_download(**kwargs):
        calls.update(kwargs)
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    monkeypatch.setattr(fetch.shutil, "copy2",
                        lambda *a, **k: pytest.fail("must not copy: doubles disk use"))

    out = fetch.download(spec, str(tmp_path))
    assert "local_dir" in calls
    assert "resume_download" not in calls   # deprecated in huggingface_hub
    assert os.path.exists(out)


def test_download_renames_diffusers_layout_file(monkeypatch, tmp_path, renamed_spec):
    def fake_hub_download(**kwargs):
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    out = fetch.download(renamed_spec, str(tmp_path))
    assert os.path.basename(out) == "controlnet-canny-sdxl.safetensors"
    assert os.path.exists(out)


def test_download_skips_when_file_already_present(monkeypatch, tmp_path, spec):
    dest = tmp_path / "checkpoints" / "model.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_text("already here")

    monkeypatch.setattr(fetch, "hf_hub_download",
                        lambda **k: pytest.fail("should not re-download"))
    out = fetch.download(spec, str(tmp_path))
    assert out == str(dest)
    assert dest.read_text() == "already here"


def test_emit_receives_progress_messages(monkeypatch, tmp_path, spec):
    msgs = []

    def fake_hub_download(**kwargs):
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    fetch.download(spec, str(tmp_path), emit=msgs.append)
    assert any("model.safetensors" in m for m in msgs)


def test_download_all_returns_one_path_per_spec(monkeypatch, tmp_path, spec, renamed_spec):
    def fake_hub_download(**kwargs):
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    out = fetch.download_all([spec, renamed_spec], str(tmp_path))
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/fetch_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'colab_studio.fetch'`

- [ ] **Step 3: Write minimal implementation**

Create `colab_studio/fetch.py`:

```python
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
```

Note: `shutil` is imported because the test asserts `shutil.copy2` is never called — importing it is what makes that guard meaningful.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/fetch_test.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add colab_studio/fetch.py tests-unit/colab_studio_test/fetch_test.py
git commit -m "feat(colab): add single-write model fetcher"
```

---

### Task 5: ComfyUI HTTP client

**Files:**
- Create: `colab_studio/client.py`
- Create: `tests-unit/colab_studio_test/client_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ComfyClient(base_url: str = "http://127.0.0.1:8188")` with methods `wait_ready(timeout: float = 180.0) -> bool`, `upload_image(path: str) -> str`, `submit(graph: dict) -> str`, `wait_result(prompt_id: str, timeout: float = 600.0) -> list[dict]`, `fetch_image(ref: dict) -> bytes`, and `generate(graph, timeout=600.0) -> list[bytes]`. Used by Task 7.

**Context:** Endpoint shapes verified against `server.py` on ComfyUI 0.10.0:
- `GET /system_stats` (`server.py:603`) — readiness probe.
- `POST /prompt` (`server.py:863`) — body `{"prompt": graph, "client_id": str}`, returns `{"prompt_id": ...}`. Returns HTTP 400 with `{"error":…, "node_errors":…}` on invalid graphs.
- `GET /history/{prompt_id}` (`server.py:850`) — `{prompt_id: {"outputs": {node_id: {"images": [{filename, subfolder, type}]}}}}`.
- `GET /view?filename=&subfolder=&type=` (`server.py:476`) — raw image bytes.
- `POST /upload/image` (`server.py:424`) — multipart field `image`, returns `{"name","subfolder","type"}`. `name` feeds straight into `LoadImage.inputs.image`.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/client_test.py`:

```python
"""Tests ComfyClient against a real stub HTTP server on a loopback socket."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from colab_studio.client import ComfyClient, ComfyError

STATE = {"ready": True, "history_hits": 0, "history_ready_after": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/system_stats"):
            if not STATE["ready"]:
                self._json(503, {"error": "booting"})
            else:
                self._json(200, {"system": {"comfyui_version": "0.10.0"}})
        elif self.path.startswith("/history/"):
            STATE["history_hits"] += 1
            if STATE["history_hits"] <= STATE["history_ready_after"]:
                self._json(200, {})
            else:
                pid = self.path.rsplit("/", 1)[-1]
                self._json(200, {pid: {"outputs": {"7": {"images": [
                    {"filename": "out_001.png", "subfolder": "colab", "type": "output"}
                ]}}}})
        elif self.path.startswith("/view"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"PNG!")
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if self.path == "/prompt":
            body = json.loads(raw)
            if "prompt" not in body:
                self._json(400, {"error": "no_prompt", "node_errors": {}})
            elif body["prompt"].get("bad"):
                self._json(400, {"error": {"message": "bad graph"},
                                 "node_errors": {"5": "nope"}})
            else:
                self._json(200, {"prompt_id": "pid-123", "number": 1,
                                 "node_errors": {}})
        elif self.path == "/upload/image":
            self._json(200, {"name": "uploaded.png", "subfolder": "",
                             "type": "input"})
        else:
            self._json(404, {})


@pytest.fixture
def server():
    STATE.update({"ready": True, "history_hits": 0, "history_ready_after": 0})
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", STATE
    httpd.shutdown()


def test_wait_ready_true_when_server_answers(server):
    url, _ = server
    assert ComfyClient(url).wait_ready(timeout=5) is True


def test_wait_ready_false_when_never_ready(server):
    url, state = server
    state["ready"] = False
    assert ComfyClient(url).wait_ready(timeout=1.5) is False


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/client_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'colab_studio.client'`

- [ ] **Step 3: Write minimal implementation**

Create `colab_studio/client.py`:

```python
"""HTTP client for a running ComfyUI server.

Deliberately talks HTTP rather than importing comfy: comfy/cli_args.py:236
parses sys.argv when args_parsing is enabled, and comfy/model_management.py:238
probes the GPU at import time. Both are hostile inside a notebook kernel.
"""
from __future__ import annotations

import os
import time
import uuid

import requests


class ComfyError(RuntimeError):
    """Server rejected a request. Carries node_errors when present."""


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def wait_ready(self, timeout: float = 180.0, interval: float = 1.0) -> bool:
        """Poll /system_stats until the server answers. Start the tunnel only
        after this returns True, or the public URL 502s."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/system_stats", timeout=5)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False

    def upload_image(self, path: str) -> str:
        """Upload to the server's input/ dir. Returns the name to put in
        LoadImage.inputs.image."""
        with open(path, "rb") as fh:
            r = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (os.path.basename(path), fh)},
                data={"overwrite": "true"},
                timeout=120,
            )
        if r.status_code != 200:
            raise ComfyError(f"upload failed ({r.status_code}): {r.text[:300]}")
        return r.json()["name"]

    def submit(self, graph: dict) -> str:
        r = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": graph, "client_id": self.client_id},
            timeout=60,
        )
        if r.status_code != 200:
            try:
                payload = r.json()
            except ValueError:
                raise ComfyError(f"submit failed ({r.status_code}): {r.text[:300]}")
            raise ComfyError(
                f"submit rejected: {payload.get('error')} "
                f"node_errors={payload.get('node_errors')}"
            )
        return r.json()["prompt_id"]

    def wait_result(self, prompt_id: str, timeout: float = 600.0,
                    interval: float = 1.0) -> list[dict]:
        """Poll /history until outputs appear. Returns image refs."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
            if r.status_code == 200:
                hist = r.json().get(prompt_id)
                if hist and hist.get("outputs"):
                    refs: list[dict] = []
                    for node_out in hist["outputs"].values():
                        refs.extend(node_out.get("images", []))
                    if refs:
                        return refs
            time.sleep(interval)
        raise TimeoutError(f"no outputs for {prompt_id} within {timeout}s")

    def fetch_image(self, ref: dict) -> bytes:
        r = requests.get(
            f"{self.base_url}/view",
            params={"filename": ref["filename"],
                    "subfolder": ref.get("subfolder", ""),
                    "type": ref.get("type", "output")},
            timeout=120,
        )
        if r.status_code != 200:
            raise ComfyError(f"view failed ({r.status_code})")
        return r.content

    def generate(self, graph: dict, timeout: float = 600.0) -> list[bytes]:
        """submit -> wait -> fetch. The whole inline-cell path in one call."""
        pid = self.submit(graph)
        return [self.fetch_image(ref) for ref in self.wait_result(pid, timeout)]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/client_test.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add colab_studio/client.py tests-unit/colab_studio_test/client_test.py
git commit -m "feat(colab): add ComfyUI HTTP client with upload and generate"
```

---

### Task 6: Backgrounded launch, readiness poll and tunnel

**Files:**
- Create: `colab_studio/launch.py`
- Create: `tests-unit/colab_studio_test/launch_test.py`

**Interfaces:**
- Consumes: `colab_studio.client.ComfyClient`.
- Produces: `start_server(comfy_dir: str, flags: list[str], log_path: str, port: int = 8188, python: str | None = None) -> subprocess.Popen`, `start_tunnel(port: int, log_path: str, timeout: float = 40.0) -> str | None`, `tail(log_path: str, n: int = 40) -> str`. Used by Task 7.

**Context:** This is the load-bearing fix. The old notebook's final cell was `!python3 main.py`, which never returns, making every cell after it unreachable. `Popen` with output to a logfile keeps the kernel free. The tunnel must start *after* `wait_ready()` succeeds, or the printed URL 502s.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/launch_test.py`:

```python
import os
import sys
import time

import pytest

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
        def poll(self):
            return None

    monkeypatch.setattr(launch.subprocess, "Popen", lambda cmd, **kw: P())
    assert launch.start_tunnel(8188, str(log), timeout=2) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/launch_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'colab_studio.launch'`

- [ ] **Step 3: Write minimal implementation**

Create `colab_studio/launch.py`:

```python
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


def start_tunnel(port: int, log_path: str, timeout: float = 40.0,
                 interval: float = 1.0) -> str | None:
    """Start cloudflared and scrape the public URL out of its log.

    Call only after ComfyClient.wait_ready() returns True.
    """
    if os.path.exists(log_path):
        os.remove(log_path)
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() not in (None,):
            return None
        try:
            with open(log_path, "r", errors="ignore") as fh:
                m = TUNNEL_RE.search(fh.read())
            if m:
                return m.group(0)
        except FileNotFoundError:
            pass
        time.sleep(interval)
    return None


def tail(log_path: str, n: int = 40) -> str:
    """Last n lines of a logfile. Empty string if it does not exist yet."""
    try:
        with open(log_path, "r", errors="ignore") as fh:
            return "\n".join(fh.read().splitlines()[-n:])
    except FileNotFoundError:
        return ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/launch_test.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run the whole suite**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/ -v
```

Expected: 37 passed.

- [ ] **Step 6: Commit**

```bash
git add colab_studio/launch.py tests-unit/colab_studio_test/launch_test.py
git commit -m "feat(colab): background server launch with readiness-gated tunnel"
```

---

### Task 7: Notebook builder

**Files:**
- Create: `build_notebook.py`
- Create: `tests-unit/colab_studio_test/build_notebook_test.py`
- Generate: `ComfyUI_Colab_Studio.ipynb`

**Interfaces:**
- Consumes: every `colab_studio/*.py` module (read as text and inlined).
- Produces: `build(out_path: str) -> dict` returning the notebook JSON, and a `__main__` entry point.

**Context:** The notebook must run from a fresh Colab with zero uploads, so helper modules are inlined via `%%writefile` rather than cloned. `build_notebook.py` reads the *tested* module files and embeds their text, which keeps one source of truth — the shipped code is the tested code.

Cell order is fixed by the spec: config, preflight, install, persistence, download, workflows, launch, generate, logs, ops, handbook.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/build_notebook_test.py`:

```python
import json
import re
import pytest

import build_notebook


@pytest.fixture(scope="module")
def nb(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "ComfyUI_Colab_Studio.ipynb"
    build_notebook.build(str(out))
    return json.loads(out.read_text())


def sources(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def test_notebook_is_valid_json_with_cells(nb):
    assert nb["cells"]
    assert nb["nbformat"] == 4


def test_no_mcp_reference_anywhere(nb):
    """Spec criterion 8: the notebook must not depend on colab-mcp."""
    blob = json.dumps(nb).lower()
    assert "mcp" not in blob


def test_notebook_never_imports_comfy_in_kernel(nb):
    """comfy/model_management.py:238 probes the GPU at import time."""
    for src in sources(nb):
        if src.startswith("%%writefile"):
            continue          # inlined library files are written, not executed
        assert not re.search(r"^\s*import comfy\b", src, re.M)
        assert not re.search(r"^\s*from comfy\b", src, re.M)


def test_launch_cell_does_not_block(nb):
    """The original bug: `!python3 main.py` never returns."""
    for src in sources(nb):
        assert "!python3 main.py" not in src
        assert "!python main.py" not in src


def test_library_modules_are_inlined(nb):
    blob = "\n".join(sources(nb))
    for mod in ("registry.py", "advice.py", "workflows.py",
                "fetch.py", "client.py", "launch.py"):
        assert f"%%writefile colab_studio/{mod}" in blob


def test_inlined_source_matches_repo_source(nb):
    """Single source of truth: shipped code == tested code."""
    with open("colab_studio/advice.py") as fh:
        real = fh.read()
    blob = "\n".join(sources(nb))
    assert "LOW_MAX_VRAM = 12.0" in real
    assert "LOW_MAX_VRAM = 12.0" in blob


def test_tunnel_starts_after_readiness_check(nb):
    """Ordering guard: tunnel before wait_ready prints a URL that 502s."""
    blob = "\n".join(sources(nb))
    assert blob.index("wait_ready") < blob.index("start_tunnel")


def test_config_cell_exposes_expected_form_fields(nb):
    blob = "\n".join(sources(nb))
    for field in ("MODE", "IMAGE_MODEL", "PERSIST", "CONTROLNET"):
        assert f"{field} =" in blob


def test_handbook_cell_present_with_troubleshooting(nb):
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    blob = "\n".join("".join(c["source"]) for c in md).lower()
    assert "out of memory" in blob or "oom" in blob
    assert "disk" in blob
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/build_notebook_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'build_notebook'`

- [ ] **Step 3: Write minimal implementation**

Create `build_notebook.py`. Because this file is long, build it in the order below — the structure is: helpers, then one function per cell, then `build()`.

```python
"""Generate ComfyUI_Colab_Studio.ipynb.

Library modules are read from colab_studio/ and inlined as %%writefile
cells, so the notebook needs no uploads and no git clone of this repo,
and the shipped code is exactly the tested code.

This file is excluded from ruff's print rule only because it emits no
prints -- the generated notebook cells do, and *.ipynb is excluded
(pyproject.toml:28).
"""
from __future__ import annotations

import json
import os

MODULES = ["registry.py", "advice.py", "workflows.py", "fetch.py",
           "client.py", "launch.py"]

HERE = os.path.dirname(os.path.abspath(__file__))


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip().splitlines(keepends=True)}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip().splitlines(keepends=True)}


def _module_cells() -> list[dict]:
    cells = []
    for mod in MODULES:
        with open(os.path.join(HERE, "colab_studio", mod)) as fh:
            body = fh.read()
        cells.append(_code(f"%%writefile colab_studio/{mod}\n{body}"))
    return cells


def build(out_path: str) -> dict:
    cells: list[dict] = [
        _md("""
# ComfyUI Colab Studio

Self-contained. Nothing to upload. **Runtime > Run all**, then use the
generate cell at the bottom or open the tunnel URL.

| Cell | What it does |
|---|---|
| 1 | Pick model + persistence options |
| 2 | Detect GPU, print what fits, choose launch flags |
| 3-6 | Install ComfyUI, link Drive, download models, write workflows |
| 7 | Start the server **in the background** and open a public URL |
| 8 | Generate an image without leaving this notebook |
| 9-10 | Logs, restart, free VRAM, disk usage |
| 11 | Handbook: model sizes, settings, error fixes |
"""),

        _code("""
#@title 1. Config
MODE = "image"  #@param ["image", "video"]
IMAGE_MODEL = "auto"  #@param ["auto", "sdxl", "flux-dev", "flux-schnell"]
PERSIST = "outputs-only"  #@param ["outputs-only", "everything", "off"]
CONTROLNET = False  #@param {type:"boolean"}
UPSCALER = True  #@param {type:"boolean"}
PORT = 8188
COMFY_DIR = "/content/ComfyUI"
print(f"mode={MODE} model={IMAGE_MODEL} persist={PERSIST} "
      f"controlnet={CONTROLNET} upscaler={UPSCALER}")
"""),

        _code("""
#@title 2. Preflight - GPU, disk, and what actually fits
import shutil, subprocess, torch

if not torch.cuda.is_available():
    print("!! No GPU. Runtime > Change runtime type > GPU, then rerun.")
    VRAM_GB, GPU_NAME = 0.0, "cpu"
else:
    props = torch.cuda.get_device_properties(0)
    GPU_NAME, VRAM_GB = props.name, props.total_memory / 2**30

DISK_FREE_GB = shutil.disk_usage("/content").free / 2**30
print(f"GPU:  {GPU_NAME}  ({VRAM_GB:.1f} GB VRAM)")
print(f"Disk: {DISK_FREE_GB:.0f} GB free")
"""),

        _code("""
#@title 3. Install ComfyUI
import os
%cd /content
if not os.path.isdir(COMFY_DIR):
    !git clone https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}
%cd {COMFY_DIR}
!pip install -q -r requirements.txt
!pip install -q huggingface_hub torchsde requests
if MODE == "video":
    os.makedirs("custom_nodes", exist_ok=True)
    for url, name in [
        ("https://github.com/kijai/ComfyUI-WanVideoWrapper.git", "ComfyUI-WanVideoWrapper"),
        ("https://github.com/city96/ComfyUI-GGUF.git", "ComfyUI-GGUF"),
        ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", "ComfyUI-VideoHelperSuite"),
    ]:
        d = os.path.join("custom_nodes", name)
        if not os.path.isdir(d):
            os.system(f"git clone {url} {d}")
        req = os.path.join(d, "requirements.txt")
        if os.path.isfile(req):
            os.system(f"pip install -q -r {req}")
os.makedirs("colab_studio", exist_ok=True)
open("colab_studio/__init__.py", "a").close()
print("ComfyUI installed at", COMFY_DIR)
"""),

        _code("""
#@title 4. Persistence (Drive)
# Models stay on VM disk by default: one Flux checkpoint is 16 GB and the
# free Drive tier is 15 GB, so symlinking models/ fills the quota instantly
# and streams every read over FUSE.
import os

def _link(src, dest):
    os.makedirs(src, exist_ok=True)
    if os.path.isdir(dest) and not os.path.islink(dest):
        os.system(f'cp -rn "{dest}"/* "{src}"/ 2>/dev/null')
        os.system(f'rm -rf "{dest}"')
    if not os.path.islink(dest):
        os.symlink(src, dest)

if PERSIST != "off":
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = "/content/drive/MyDrive/ComfyUI"
    subs = ["output", "user"]
    if PERSIST == "everything":
        print("!! Models on Drive: slow (FUSE) and Flux alone is 16 GB.")
        subs += ["models"]
    for sub in subs:
        _link(os.path.join(DRIVE, sub), os.path.join(COMFY_DIR, sub))
    print("Persisting:", ", ".join(subs))
else:
    print("No persistence - everything is lost when the runtime recycles.")
"""),
    ]

    cells += _module_cells()

    cells += [
        _code("""
#@title 5. Choose profile and download models
import sys
sys.path.insert(0, COMFY_DIR)
from colab_studio.advice import recommend
from colab_studio.registry import resolve, total_gb, CHECKPOINT_NAME
from colab_studio.fetch import download_all

ADVICE = recommend(VRAM_GB, DISK_FREE_GB)
PROFILE = ADVICE.profile if IMAGE_MODEL == "auto" else IMAGE_MODEL
LAUNCH_FLAGS = ADVICE.launch_flags
MAX_SIDE = ADVICE.max_side

print(f"tier={ADVICE.tier}  profile={PROFILE}  max_side={MAX_SIDE}")
print(f"launch flags: {' '.join(LAUNCH_FLAGS) or '(none)'}")
for n in ADVICE.notes:
    print(" -", n)

SPECS = resolve(PROFILE, controlnet=CONTROLNET, upscale=UPSCALER)
print(f"\\nDownloading {len(SPECS)} files, {total_gb(SPECS)} GB total")
download_all(SPECS, os.path.join(COMFY_DIR, "models"), emit=print)
CKPT = CHECKPOINT_NAME[PROFILE]
"""),

        _code("""
#@title 6. Write workflows into the ComfyUI sidebar
import json, os
from colab_studio import workflows

WF_DIR = os.path.join(COMFY_DIR, "user", "default", "workflows")
os.makedirs(WF_DIR, exist_ok=True)
API_DIR = "/content/wf_api"
os.makedirs(API_DIR, exist_ok=True)

built = {
    "sdxl_txt2img": workflows.sdxl_txt2img(CKPT, "a prompt"),
    "upscale": workflows.upscale(CKPT, "a prompt"),
}
if PROFILE.startswith("flux"):
    built["flux_txt2img"] = workflows.flux_txt2img(CKPT, "a prompt")
if CONTROLNET:
    built["controlnet_canny"] = workflows.controlnet_canny(CKPT, "a prompt", image="input.png")

for name, graph in built.items():
    with open(os.path.join(API_DIR, f"{name}.json"), "w") as fh:
        json.dump(graph, fh, indent=1)
print("workflows written:", ", ".join(built))
"""),

        _code("""
#@title 7. Launch server (backgrounded) then open the tunnel
import os
from colab_studio.client import ComfyClient
from colab_studio.launch import start_server, start_tunnel

if not os.path.isfile("/usr/local/bin/cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    !chmod +x /usr/local/bin/cloudflared

SERVER_LOG = "/content/comfyui.log"
TUNNEL_LOG = "/content/cloudflared.log"

SERVER = start_server(COMFY_DIR, LAUNCH_FLAGS, SERVER_LOG, port=PORT)
CLIENT = ComfyClient(f"http://127.0.0.1:{PORT}")

print("waiting for server...")
if CLIENT.wait_ready(timeout=300):
    # Tunnel only AFTER readiness, or the URL 502s.
    URL = start_tunnel(PORT, TUNNEL_LOG)
    print("ComfyUI ready.  Public URL:", URL or "(tunnel failed, see cell 10)")
else:
    print("server did not come up - run the log cell below")
"""),

        _code("""
#@title 8. Generate an image here (no tunnel needed)
from IPython.display import Image, display
from colab_studio import workflows

prompt = "a lighthouse in a storm, dramatic light"  #@param {type:"string"}
negative = "blurry, watermark, text"  #@param {type:"string"}
steps = 25  #@param {type:"slider", min:4, max:60, step:1}
cfg = 7.0  #@param {type:"number"}
seed = 0  #@param {type:"integer"}
width = 1024  #@param {type:"integer"}
height = 1024  #@param {type:"integer"}
use_upscaler = False  #@param {type:"boolean"}

width, height = min(width, MAX_SIDE), min(height, MAX_SIDE)
kw = dict(prompt=prompt, negative=negative, seed=seed, steps=steps,
          width=width, height=height)
if use_upscaler:
    graph = workflows.upscale(CKPT, cfg=cfg, **kw)
elif PROFILE.startswith("flux"):
    graph = workflows.flux_txt2img(CKPT, **kw)
else:
    graph = workflows.sdxl_txt2img(CKPT, cfg=cfg, **kw)

for i, data in enumerate(CLIENT.generate(graph)):
    path = f"/content/gen_{i}.png"
    with open(path, "wb") as fh:
        fh.write(data)
    display(Image(filename=path))
"""),

        _code("""
#@title 8b. Image-to-image / ControlNet (upload or URL)
import os, requests
from IPython.display import Image, display
from colab_studio import workflows

source = "upload"  #@param ["upload", "url"]
image_url = ""  #@param {type:"string"}
mode = "img2img"  #@param ["img2img", "controlnet"]
prompt2 = "an oil painting of the same scene"  #@param {type:"string"}
denoise = 0.6  #@param {type:"slider", min:0.1, max:1.0, step:0.05}

local = "/content/source_image.png"
if source == "upload":
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    with open(local, "wb") as fh:
        fh.write(up[name])
else:
    with open(local, "wb") as fh:
        fh.write(requests.get(image_url, timeout=60).content)

server_name = CLIENT.upload_image(local)
if mode == "controlnet":
    graph = workflows.controlnet_canny(CKPT, prompt2, image=server_name)
else:
    graph = workflows.img2img(CKPT, prompt2, image=server_name, denoise=denoise)

for i, data in enumerate(CLIENT.generate(graph)):
    path = f"/content/edit_{i}.png"
    with open(path, "wb") as fh:
        fh.write(data)
    display(Image(filename=path))
"""),

        _code("""
#@title 9. Server log
from colab_studio.launch import tail
lines = 60  #@param {type:"integer"}
print(tail(SERVER_LOG, n=lines) or "(log empty)")
"""),

        _code("""
#@title 10. Ops - restart, free VRAM, disk, re-tunnel
import os, shutil, requests
action = "disk usage"  #@param ["disk usage", "free VRAM", "restart server", "re-tunnel", "list models"]

if action == "disk usage":
    u = shutil.disk_usage("/content")
    print(f"{u.free/2**30:.1f} GB free of {u.total/2**30:.1f} GB")
elif action == "free VRAM":
    requests.post(f"http://127.0.0.1:{PORT}/free",
                  json={"unload_models": True, "free_memory": True}, timeout=30)
    print("asked ComfyUI to unload models")
elif action == "restart server":
    from colab_studio.launch import start_server
    SERVER.terminate(); SERVER.wait(timeout=30)
    SERVER = start_server(COMFY_DIR, LAUNCH_FLAGS, SERVER_LOG, port=PORT)
    print("restarted:", CLIENT.wait_ready(timeout=300))
elif action == "re-tunnel":
    from colab_studio.launch import start_tunnel
    print("URL:", start_tunnel(PORT, TUNNEL_LOG))
else:
    for root, _, fs in os.walk(os.path.join(COMFY_DIR, "models")):
        for f in fs:
            p = os.path.join(root, f)
            print(f"{os.path.getsize(p)/2**30:6.2f} GB  {os.path.relpath(p, COMFY_DIR)}")
"""),

        _md("""
## Handbook

### Model sizes and what fits

| Profile | Download | Needs | Notes |
|---|---|---|---|
| `sdxl` | 6.8 GB | ~12 GB VRAM at 1024px | Best all-rounder on a T4 |
| `flux-dev` | 16.1 GB | ~20 GB VRAM | All-in-one fp8: UNet + T5 + CLIP-L + VAE |
| `flux-schnell` | 16.1 GB | ~20 GB VRAM | 4-step; much faster, slightly lower fidelity |
| upscaler | 0.06 GB | negligible | 4x-UltraSharp, image-space |
| ControlNet canny | 2.33 GB | +2 GB VRAM | SDXL only |

### Settings that matter

| Model | steps | cfg | sampler / scheduler |
|---|---|---|---|
| SDXL | 25-30 | 6-8 | `dpmpp_2m` / `karras` |
| Flux dev | 20-25 | **1.0** | `euler` / `simple`, guidance 3.5 |
| Flux schnell | **4** | **1.0** | `euler` / `simple` |

**Flux cfg must be 1.0.** Flux does not use classifier-free guidance; real
guidance rides on the `FluxGuidance` node. Any other cfg scorches the image.

### When it breaks

| Symptom | Cause | Fix |
|---|---|---|
| Tunnel URL 502s | Server still booting | Rerun cell 7; it waits for readiness first |
| `CUDA out of memory` | Resolution or batch too high | Drop to 768px, batch 1, run "free VRAM" in cell 10 |
| `No such file or directory: ...safetensors` | Download interrupted | Rerun cell 5 - it skips completed files |
| Disk full mid-download | Flux is 16 GB | Use `sdxl`, or set `PERSIST="off"` to reclaim Drive space |
| Generate cell hangs | Server died | Check cell 9 log, then "restart server" in cell 10 |
| Session dropped | Colab idle timeout | Rerun all; with `PERSIST` on, models and outputs survive |

### Adding a LoRA without leaving Colab

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="OWNER/REPO", filename="lora.safetensors",
                local_dir=f"{COMFY_DIR}/models/loras")
```

Then use the `LoraLoader` node in the tunnel UI (it is a core node, already
available).
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    with open(out_path, "w") as fh:
        json.dump(nb, fh, indent=1)
    return nb


if __name__ == "__main__":
    build(os.path.join(HERE, "ComfyUI_Colab_Studio.ipynb"))
```

- [ ] **Step 4: Generate the notebook**

```bash
.venv/bin/python build_notebook.py && .venv/bin/python -c "import json;print(len(json.load(open('ComfyUI_Colab_Studio.ipynb'))['cells']),'cells')"
```

Expected: the notebook is written and reports its cell count.

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/build_notebook_test.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add build_notebook.py ComfyUI_Colab_Studio.ipynb tests-unit/colab_studio_test/build_notebook_test.py
git commit -m "feat(colab): generate self-contained ComfyUI Colab Studio notebook"
```

---

### Task 8: Independence verification and delivery

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-comfyui-colab-studio-design.md` (record execution results)
- Create: `tests-unit/colab_studio_test/independence_test.py`

**Interfaces:**
- Consumes: the generated `ComfyUI_Colab_Studio.ipynb`.
- Produces: nothing consumed downstream. This is the acceptance gate.

**Context:** Spec criterion 8 requires the notebook to work with no local agent, no MCP server, no `uv`. A prose constraint drifts the moment a cell gets convenient; the grep either passes or it doesn't.

- [ ] **Step 1: Write the failing test**

Create `tests-unit/colab_studio_test/independence_test.py`:

```python
"""Spec criterion 8 -- the notebook must not depend on local tooling."""
import json
import os
import re

import pytest

NB = "ComfyUI_Colab_Studio.ipynb"


@pytest.fixture(scope="module")
def blob():
    if not os.path.exists(NB):
        pytest.skip("run build_notebook.py first")
    return json.dumps(json.load(open(NB)))


def test_no_mcp_reference(blob):
    assert not re.search(r"\bmcp\b", blob, re.I)


def test_no_uv_or_uvx_dependency(blob):
    assert not re.search(r"\buvx?\s+(run|pip|git\+)", blob)


def test_no_localhost_websocket_proxy(blob):
    assert "mcpProxyToken" not in blob
    assert "mcpProxyPort" not in blob


def test_no_absolute_developer_paths(blob):
    """The original colab_setup.sh cp'd from /home/<user>/Desktop -- a path
    that does not exist on a Colab VM, which killed the script under set -e."""
    assert "/home/dennisjcarroll" not in blob
    assert not re.search(r"/Users/[a-z]+/", blob)


def test_no_clone_of_this_repo_for_helpers(blob):
    assert "git clone" in blob                     # ComfyUI itself, fine
    assert "Desktop/ComfyUI" not in blob
```

- [ ] **Step 2: Run test to verify it fails or passes honestly**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/independence_test.py -v
```

Expected: PASS if Task 7 was done correctly. If any test fails, the notebook has picked up a local dependency — fix `build_notebook.py`, regenerate, rerun. Do not weaken the test.

- [ ] **Step 3: Run the full suite and the linter**

```bash
.venv/bin/python -m pytest tests-unit/colab_studio_test/ -v
grep -rn "print(" colab_studio/ && echo "VIOLATION: no print() allowed in colab_studio/" || echo "clean"
```

Expected: 51 passed; `clean` for the print check.

- [ ] **Step 4: Record execution results in the spec**

Open `docs/superpowers/specs/2026-07-25-comfyui-colab-studio-design.md` and replace the "Currently" column of the E1–E4 table with observed results. Run **E4 first** — it needs only `torch.cuda.get_device_properties()` and `nvidia-smi`, no downloads. E1–E3 require a GPU runtime; if `colab-mcp`'s proxied tools cannot change runtime type, set it by hand via Runtime → Change runtime type → GPU and note that in the spec.

For each of E1–E4, write either the measured value or `still unverified — <reason>`. Do not mark anything verified that was not executed.

- [ ] **Step 5: Deliver the notebook**

Upload `ComfyUI_Colab_Studio.ipynb` to the user's Google Drive via the Google Drive MCP and return the Colab link. Ask before uploading — this puts a file in their account.

- [ ] **Step 6: Commit**

```bash
git add tests-unit/colab_studio_test/independence_test.py docs/superpowers/specs/2026-07-25-comfyui-colab-studio-design.md
git commit -m "test(colab): enforce notebook independence from local tooling"
```

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| Cell map (11 cells) | 7 |
| Backgrounded launch | 6 |
| Tunnel after readiness | 6 (impl), 7 (ordering test) |
| No `import comfy` in kernel | 7 (test) |
| Drive = outputs + user only | 7 (cell 4) |
| Registry-driven fetcher, no double-write | 4 |
| ~~Two workflow formats~~ → API format only | 7 (cell 6) — **UI format NOT delivered; claim dropped by decision** |
| GPU auto-detect | 2, 7 (cell 2) |
| Model registry, verified repos/sizes | 1 |
| SDXL + Flux txt2img | 3 |
| img2img, upscale (included) | 3, 7 (cell 8b) |
| ControlNet (opt-in) | 1, 3, 7 |
| Upload path: picker + URL | 5, 7 (cell 8b) |
| Handbook | 7 |
| colab-mcp non-dependency | 8 |
| E1–E4 execution testing | 8 |
| Delivery via Drive | 8 |
| Acceptance criteria 1–4, 7, 8 | 6, 7, 8 |
| Acceptance criteria 5, 6 | **NOT delivered — withdrawn, see below** |

### Corrected 2026-07-25 (final review)

**The "No gaps" claim above was wrong when written.** It asserted full spec
coverage while mapping two items that were not in fact delivered. Recorded
here rather than quietly amended:

**Dropped by decision (not defects — scope calls made after review):**

| Claimed | Reality | Disposition |
|---|---|---|
| "Two workflow formats" → Task 7 | Cell 6 created `user/default/workflows/` and never wrote to it. Only API format was ever produced, while the cell title and its `print()` both claimed sidebar workflows. | UI-format writer **dropped**. Acceptance criterion 5 **withdrawn**; cell 6 retitled and its output corrected to name `/content/wf_api`. |
| "Acceptance criteria 1–8" | Criterion 5 (sidebar workflows) and criterion 6 (`MODE=video` reproduces Wan 2.2) were both unmet. `MODE=video` cloned three custom-node repos and then generated SDXL images, since nothing downstream was video-aware. | `MODE` **removed entirely**. Criterion 6 **withdrawn**; video stays in `Wan2.2_Colab_Pipeline.ipynb`. |

**Defects found in final review and fixed** (none of which this Self-Review
caught): cell 8b's `files.upload()` blocked *Run all* and made cells 9–10
unreachable — the same class of bug as the `!python3 main.py` this plan
existed to fix; `upscale`/`img2img`/`controlnet_canny` built SDXL-shaped
graphs for Flux checkpoints; the disk guard measured `/content` rather than
the models destination; `wait_result` ignored `hist["status"]`; re-tunnel
orphaned the first cloudflared; an explicit `IMAGE_MODEL` bypassed both
guards silently.

**Process lesson.** The coverage table mapped spec items to *task numbers*,
not to *evidence*. Every row was satisfied by a task having been written,
not by an assertion that the behaviour existed — which is exactly how two
unmet criteria passed a self-review. The suite now carries a drift guard and
scoped, non-vacuous ordering and gating tests (see Task 7/8 test files) so
the equivalent claim is machine-checked next time.

**Placeholder scan:** No TBD/TODO. Every code step contains complete runnable code. Every test step contains actual assertions.

**Type consistency:** `ModelSpec.target_name` (Task 1) is used by `fetch.download` (Task 4). `Advice.profile`/`.launch_flags`/`.max_side` (Task 2) are consumed by notebook cells 5, 7, 8 (Task 7). `CHECKPOINT_NAME` (Task 1) feeds `CKPT` (Task 7). Workflow builder signatures (Task 3) match every call site in cells 6, 8, 8b. `ComfyClient.generate`/`.upload_image`/`.wait_ready` (Task 5) match cells 7, 8, 8b. `start_server`/`start_tunnel`/`tail` (Task 6) match cells 7, 9, 10. Consistent.

**Known risk carried forward:** Tasks 1–7 are structurally validated only. E1 (Flux all-in-one loads), E2 (diffusers ControlNet converts), E3 (VRAM headroom) remain unexecuted until Task 8 Step 4 on real GPU hardware.
