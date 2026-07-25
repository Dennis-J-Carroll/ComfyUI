# ComfyUI Colab Studio — Design

**Date:** 2026-07-25
**Status:** Approved for planning
**Goal:** A single Colab notebook that runs image (and optionally video) workflows end to end, with enough in-notebook guidance that the user never has to leave Colab to look something up.

---

## Problem

The existing `Wan2.2_Colab_Pipeline.ipynb` has three structural faults:

1. **It builds a video pipeline.** The stated need is image generation.
2. **Cell 11 is `!python3 main.py` — it never returns.** No cell after launch can ever execute. This is the root reason the notebook cannot be self-servicing: troubleshooting cells, log tails, re-download cells, and generate cells placed after launch are unreachable.
3. **Tunnel starts before the server.** `cloudflared` runs in cell 10, the server in cell 11, so the printed URL 502s until boot completes.

Supporting faults: `download_models.py` uses `hf_hub_download` then `shutil.copy2`, writing every model to disk twice; it passes the deprecated `resume_download`; `--highvram` is hardcoded regardless of GPU tier; and symlinking `models/` to Drive streams multi-GB downloads over FUSE.

---

## Deliverable

`ComfyUI_Colab_Studio.ipynb` — new. The existing Wan notebook is left untouched.

Helper sources (`comfy_fetch.py`, `workflows/*.json`) are kept in the repo for editing and diffing, but the notebook writes them to the VM via `%%writefile`. **The notebook must work from a fresh Colab with zero uploads** — that requirement is what rules out `git clone`-ing this repo for helpers.

### Delivery

**Primary: Google Drive.** On completion the `.ipynb` is written to the user's Drive via the connected Google Drive MCP, and the Colab link handed back. No repo push, no manual upload, no filesystem step — which is the original requirement ("never leave the browser") applied to getting the file itself.

**Secondary: GitHub raw.** Once this repo is pushed, `colab.research.google.com/github/<user>/<repo>/blob/master/ComfyUI_Colab_Studio.ipynb` opens it directly. Better for re-sharing and versioning; requires the push.

Both paths are independent of `colab-mcp`.

---

## Architecture

### Cell map

| # | Cell | Purpose |
|---|------|---------|
| 0 | MD | TL;DR — which cells to run, what each does |
| 1 | **Config** | `MODE` (image/video), `IMAGE_MODEL`, `VIDEO_MODEL`, `PERSIST`, `CONTROLNET` — Colab form dropdowns |
| 2 | **Preflight + advice** | Detect GPU name/VRAM/disk; print what fits and expected sec/image; **set** `LAUNCH_FLAGS` |
| 3 | **Install** | Clone ComfyUI, pip deps; Wan/GGUF/VHS custom nodes only when `MODE=video` |
| 4 | **Persistence** | Drive: symlink `output/` + `user/` only. Models stay on VM disk (see below) |
| 5 | **Download** | Registry-driven fetcher |
| 6 | **Write workflows** | UI-format → `user/default/workflows/`; API-format → `/content/wf_api/` |
| 7 | **Launch (backgrounded)** | `Popen` → poll `/system_stats` → *then* tunnel → print URL |
| 8 | **Generate inline** | Form fields → patch API graph → `POST /prompt` → poll → display image in notebook |
| 9 | **Log tail** | Last N lines of server log; rerunnable |
| 10 | **Ops** | Restart server, `POST /free`, re-tunnel, list models, disk usage |
| 11 | MD | **Colab handbook** — model/VRAM/speed table, sampler+steps+cfg per model, error→fix table, session/disk/quota quirks, adding LoRAs |

### Load-bearing decisions

**Backgrounded launch.** `subprocess.Popen(main.py, stdout=logfile)`, then poll `GET /system_stats` (`server.py:603`) until 200, *then* start cloudflared. This is not a bug fix — it is what makes cells 8–10 exist at all.

**The notebook kernel must never `import comfy`.** `comfy/cli_args.py:236` only parses `sys.argv` when `comfy.options.args_parsing` is set, and `comfy/model_management.py:238` probes the GPU at import time. Importing ComfyUI into the Colab kernel risks argv poisoning and import-time device detection. All notebook→server communication is HTTP. Cell 2's GPU advice uses plain `torch.cuda`, not ComfyUI internals.

**Drive persists `output/` and `user/` only; models go to VM disk.** Empirically justified: one Flux checkpoint is 16.06 GB, exceeding the 15 GB free Drive quota on its own. Persisting models is opt-in behind a printed size warning.

**Registry-driven fetcher.** `comfy_fetch.py` replaces `download_models.py`: a dict of `model_id → [(repo, filename, dest_folder, dest_filename)]`. Uses `hf_hub_download(local_dir=...)` (no double-write), drops deprecated `resume_download`, skips files already present, prints sizes before downloading.

**Two workflow formats, both hand-authored.** UI-format for the ComfyUI sidebar (`user/default/workflows/`, served by `/userdata/{file}`), API-format for cell 8. A UI→API converter is fragile for 7–10 node graphs; duplicating small JSON is dumber and more reliable.

**GPU auto-detect despite a Pro subscription.** Colab reassigns tiers. Cell 2 detects and adapts rather than hardcoding `--highvram`, which is actively wrong on a T4.

**The notebook has zero dependency on `colab-mcp`.** See Tooling below. This is a hard constraint, not a preference: the notebook must run for anyone who opens it in a browser, with no local agent, no MCP server, and no `uv` install. No cell may import, detect, probe for, or branch on the presence of MCP tooling. `colab-mcp` is a build- and test-time convenience for the author only.

---

## Model registry

All entries HEAD-verified on 2026-07-25.

| ID | Repo | File | Size | Dest |
|----|------|------|------|------|
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | `sd_xl_base_1.0.safetensors` | 6.46 GB | `models/checkpoints` |
| `sdxl-vae` | `stabilityai/sdxl-vae` | `sdxl_vae.safetensors` | 0.31 GB | `models/vae` |
| `flux-dev` | `Comfy-Org/flux1-dev` | `flux1-dev-fp8.safetensors` | 16.06 GB | `models/checkpoints` |
| `flux-schnell` | `Comfy-Org/flux1-schnell` | `flux1-schnell-fp8.safetensors` | 16.05 GB | `models/checkpoints` |
| `upscale` | `Kim2091/UltraSharp` | `4x-UltraSharp.pth` | 0.06 GB | `models/upscale_models` |
| `cn-canny` | `diffusers/controlnet-canny-sdxl-1.0` | `diffusion_pytorch_model.fp16.safetensors` | 2.33 GB | `models/controlnet` (rename) |

**Corrections found during verification:**
- `Comfy-Org/stable-diffusion-xl-base-1.0` returns **401 — the repo does not exist.** The original guess would have failed at runtime. Correct namespace is `stabilityai/`.
- Flux fp8 is **16 GB, not ~11 GB** as commonly assumed.
- `flux1-dev-fp8.safetensors` is an **all-in-one** (UNet + T5 + CLIP-L + VAE). It loads via `CheckpointLoaderSimple`, giving a 7-node graph instead of the 10-node `UNETLoader`+`DualCLIPLoader`+`VAELoader` split.
- The ControlNet file is diffusers-layout and must be renamed on download. `comfy/controlnet.py:623` has an explicit diffusers branch (`if "controlnet_cond_embedding.conv_in.weight" in controlnet_data`), so it will load.
- Flux ControlNet (`flux1-canny-dev`, **22.17 GB**) is ruled out as prohibitive.

---

## Workflows

Five graphs, all validated (see Testing).

**Core (always):**
- `sdxl_txt2img` — 7 nodes. `dpmpp_2m` / `karras`, 25 steps, cfg 7.0, 1024×1024.
- `flux_txt2img` — 8 nodes. `euler` / `simple`, 20 steps. **cfg must be 1.0**; real guidance goes in `FluxGuidance.guidance` (3.5).

**Optional, included (zero/near-zero cost):**
- `img2img` — swaps `EmptyLatentImage` for `LoadImage`+`VAEEncode`, `denoise` 0.6. **0 MB extra, 0 custom nodes.**
- `upscale` — appends `UpscaleModelLoader`+`ImageUpscaleWithModel`. **64 MB, 0 custom nodes.** The only optional needing no upload path — it post-processes the decoded image.

**Optional, opt-in (`CONTROLNET=False` default):**
- `controlnet_canny` — adds core `Canny` + `ControlNetLoader` + `ControlNetApplyAdvanced`. **2.33 GB, 0 custom nodes.** SDXL only.

The original YAGNI cut assumed ControlNet required the `comfyui_controlnet_aux` custom node. **That assumption was falsified** — `Canny` is core. Every node for all three optionals ships with ComfyUI 0.10.0.

`LoraLoader` is also core and is deliberately **not** in scope.

### Image upload path

`LoadImage` enumerates the server-side `input/` directory, which the Colab user cannot write to. So `img2img` and `controlnet_canny` need an upload step **for the inline cell only** — through the tunnel UI the browser handles it natively.

`POST /upload/image` (`server.py:424`) returns `{"name", "subfolder", "type"}`; `name` feeds directly into `LoadImage.inputs.image`. Cell 8 provides both `google.colab.files.upload()` for local files and a URL field that fetches then POSTs.

---

## Testing

Validation runs against `execution.validate_prompt` — the exact code path `POST /prompt` invokes — with dummy model files registered via `folder_paths.add_model_folder_path` so ENUM membership checks pass.

| Workflow | Result |
|---|---|
| `sdxl_txt2img` | PASS |
| `flux_txt2img` | PASS |
| `img2img` | PASS |
| `upscale` | PASS |
| `controlnet_canny` | PASS |
| bad sampler name (negative control) | FAIL as expected |
| missing checkpoint (negative control) | FAIL as expected |

Scripts: `scratchpad/validate_wf.py`, `scratchpad/validate_optional.py`. To be moved into `tests/` during implementation.

**Scope of this validation — stated honestly:** it is *structural*. It checks node class names, input key names, enum membership, and link types. Negative controls confirm the validator is genuinely engaged rather than vacuously passing. It does **not** execute any node.

### Execution testing (via `colab-mcp`, during implementation — not performed, see below)

These four require a GPU. `colab-mcp` was expected to make them executable during implementation; it did not — the Task 8 build environment was CPU-only Linux with no GPU and no Colab session attached, so `colab-mcp`'s browser-proxied tools were never exercised. They remain open assumptions, not resolved implementation tasks — see the status column below.

| # | Claim | Currently |
|---|-------|-----------|
| E1 | Flux fp8 all-in-one loads via `CheckpointLoaderSimple` | still unverified — requires a live Colab GPU runtime; no GPU in the build environment |
| E2 | diffusers ControlNet state dict converts at runtime (`comfy/controlnet.py:623`) | still unverified — requires a live Colab GPU runtime; no GPU in the build environment |
| E3 | VRAM headroom per model on the actually-assigned GPU | still unverified — requires a live Colab GPU runtime; no GPU in the build environment |
| E4 | Cell 2's VRAM→model→resolution→flags thresholds | thresholds implemented and unit-tested at boundaries (12.0, 20.0); NOT calibrated against real hardware — requires a live Colab GPU runtime |

**Run E4 first.** It needs only `torch.cuda.get_device_properties().total_memory` and `nvidia-smi` — seconds, no downloads. E1–E3 require 16 GB (Flux) and 2.33 GB (ControlNet) pulls that may stall. Sequencing E4 first unblocks the one genuinely opinionated piece of the design regardless of how the heavy downloads go. It should be calibrated against observed numbers, not guessed.

**Prerequisite — GPU runtime.** `colab-mcp` opens a scratch notebook, and Colab's default runtime is **CPU**. E1 and E3 require a GPU runtime; E2 requires one to instantiate the ControlNet. Whether the proxied tool set can change runtime type is **unverified** — the tool list is empty until the browser connects, so it is only answerable live. Until confirmed, assume the human must set Runtime → Change runtime type → GPU before E1–E3, and treat that as a documented step rather than a surprise.

**Execution-time note (Task 8).** The build environment used to implement this task is CPU-only Linux with no GPU and no Colab session attached, so `colab-mcp`'s CPU-default scratch notebook was never reached in practice either — confirming, rather than resolving, the note above: a human must set Runtime → Change runtime type → GPU before E1–E3 can be attempted.

---

## Tooling: `colab-mcp` (development only, non-load-bearing)

Registered in user-scope MCP config on 2026-07-25, verified `✔ Connected`:

```json
"colab-mcp": {
  "command": "uvx",
  "args": ["git+https://github.com/googlecolab/colab-mcp"],
  "timeout": 30000
}
```

**What it is.** [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp) — a local stdio MCP server that starts a WebSocket on an ephemeral localhost port with a `secrets`-generated token, origin-locked to `colab.research.google.com` / `colab.google.com`. Its one static tool, `open_colab_browser_connection`, opens a browser tab to `/notebooks/empty.ipynb#mcpProxyToken=…&mcpProxyPort=…`; the Colab **frontend** then connects back and the real notebook tools are proxied from that session via `FastMCPProxy`. Hence the `notifications/tools/list_changed` requirement — the tool list is empty until the browser connects. 60s connect timeout. Requires Python >=3.13, which `uvx` provisions itself.

**Consequences worth recording:**
- The browser tab must be open and stay open. There is no headless runtime access.
- It opens a *scratch* notebook, not an arbitrary file. Shipping `ComfyUI_Colab_Studio.ipynb` still needs Drive or a GitHub raw URL.
- Google describes it as early-stage.

**Why it is in this spec at all.** It converts four previously unverifiable acceptance criteria into executable ones (see below). It does **not** appear anywhere in the deliverable. Re-read the hard constraint in Architecture: no notebook cell may reference it.

---

## Acceptance criteria

1. Fresh Colab, `MODE=image`, Run All → tunnel URL prints and resolves without a 502.
2. Cells 8–10 are reachable and rerunnable after launch.
3. Cell 8 produces an image displayed inline, without opening the tunnel.
4. Cell 2 prints GPU-appropriate advice and sets launch flags that differ between T4 and L4/A100.
5. All five workflows appear in the ComfyUI sidebar.
6. `MODE=video` reproduces current Wan 2.2 behavior.
7. Second session with `PERSIST` on reuses Drive outputs without re-downloading.
8. **Independence check.** With `colab-mcp` stopped and no local agent attached, a cold Colab session opened straight from Drive or a GitHub raw URL still satisfies criteria 1–7. `grep -ri "mcp\|colab-mcp" ComfyUI_Colab_Studio.ipynb` returns nothing.

---

## Out of scope

LoRA loading UI, IP-Adapter, inpainting, Flux ControlNet, video upscaling, multi-GPU, custom node management beyond the fixed set.
