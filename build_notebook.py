"""Generate ComfyUI_Colab_Studio.ipynb.

Library modules are read from colab_studio/ and inlined as %%writefile
cells, so the notebook needs no uploads and no git clone of this repo,
and the shipped code is exactly the tested code.

This file is subject to ruff's print rule like the rest of the repo and
satisfies it by having no print calls of its own. The `print(` occurrences
below are inside string literals destined for notebook cells, and *.ipynb
is excluded from linting (pyproject.toml:28).
"""
from __future__ import annotations

import json
import os

from colab_studio import compat

MODULES = ["compat.py", "registry.py", "advice.py", "workflows.py",
           "fetch.py", "client.py", "telemetry.py", "launch.py"]

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
generate cell at the bottom. No public URL is opened unless you turn one
on in cell 1.

This notebook does **images** only. Video generation is not yet shipped in
Colab Studio.

| Cell | What it does |
|---|---|
| 1 | Pick model, persistence, ComfyUI pin, and public-tunnel options |
| 2 | Detect GPU and disk |
| 3 | Install ComfyUI, pinned to the tested revision by default |
| 4 | Link Drive (optional persistence) |
| *(6 unnumbered)* | `%%writefile` the helper library into `colab_studio/` |
| 5 | Choose profile, re-check disk, download models |
| 6 | Write API workflows for the generate cells |
| 7 | Start the server **in the background**; open a public tunnel only if `OPEN_PUBLIC_UI` is on |
| 8 | Generate an image without leaving this notebook |
| 8b | Image-to-image / ControlNet - **tick `run_this` first**; Run all skips it |
| 9-10 | Logs, restart, free VRAM, disk usage, re-tunnel |
| *(last)* | Handbook: model sizes, settings, error fixes |
"""),

        _code("""
#@title 1. Config
IMAGE_MODEL = "auto"  #@param ["auto", "sdxl", "flux-dev", "flux-schnell"]
PERSIST = "outputs-only"  #@param ["outputs-only", "everything", "off"]
CONTROLNET = False  #@param {type:"boolean"}
USE_UPSCALER = True  #@param {type:"boolean"}
COMFY_REF = "pinned"  #@param ["pinned", "latest"]
OPEN_PUBLIC_UI = False  #@param {type:"boolean"}
PORT = 8188
COMFY_DIR = "/content/ComfyUI"
print(f"model={IMAGE_MODEL} persist={PERSIST} "
      f"controlnet={CONTROLNET} upscaler={USE_UPSCALER}")
print(f"comfy_ref={COMFY_REF} open_public_ui={OPEN_PUBLIC_UI}")
if OPEN_PUBLIC_UI:
    print("!! OPEN_PUBLIC_UI is on: cell 7 will start a public Cloudflare "
          "tunnel. The URL is not authentication -- see cell 7's warning.")
"""),

        _code("""
#@title 2. Preflight - GPU, disk, and what actually fits
import shutil, torch

if not torch.cuda.is_available():
    print("!! No GPU. Runtime > Change runtime type > GPU, then rerun.")
    VRAM_GB, GPU_NAME = 0.0, "cpu"
else:
    props = torch.cuda.get_device_properties(0)
    GPU_NAME, VRAM_GB = props.name, props.total_memory / 2**30

# Indicative only. Cell 5 re-measures at the real models/ destination, which
# may be a Drive symlink by then.
DISK_FREE_GB = shutil.disk_usage("/content").free / 2**30
print(f"GPU:  {GPU_NAME}  ({VRAM_GB:.1f} GB VRAM)")
print(f"Disk: {DISK_FREE_GB:.0f} GB free on /content")
"""),

        _code(f"""
#@title 3. Install ComfyUI
import os
%cd /content
if not os.path.isdir(COMFY_DIR):
    !git clone {compat.REPO_URL}.git {{COMFY_DIR}}
%cd {{COMFY_DIR}}

# Kept in sync with colab_studio/compat.py -- that module is the source of
# truth; this literal is generated from it, not hand-typed twice.
TESTED_REF = "{compat.TESTED_REF}"
if COMFY_REF == "pinned":
    !git fetch --quiet origin
    !git checkout --quiet {{TESTED_REF}}
else:
    print("!" * 70)
    print("!! UNSUPPORTED: COMFY_REF='latest'. Colab Studio's workflow graphs")
    print("!! were structurally validated only against {compat.short_ref()} "
          "({compat.TESTED_DATE}).")
    print("!! Upstream node contracts, input names, or defaults may have")
    print("!! changed since then -- generation can fail in ways nobody here")
    print("!! has tested. Set COMFY_REF='pinned' in cell 1 unless you")
    print("!! specifically need something newer than that.")
    print("!" * 70)
    !git fetch --quiet origin master
    !git checkout --quiet FETCH_HEAD

resolved = !git rev-parse HEAD
print("ComfyUI commit:", resolved[0])

!pip install -q -r requirements.txt
!pip install -q huggingface_hub torchsde requests
# Core ComfyUI only: every node these workflows use ships with it, so there
# are no custom_nodes clones to wait on.
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
import os, shutil, sys
sys.path.insert(0, COMFY_DIR)
from colab_studio.advice import recommend
from colab_studio.registry import resolve, total_gb, CHECKPOINT_NAME
from colab_studio.fetch import download_all

MODELS_DIR = os.path.join(COMFY_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
# Measure the filesystem the checkpoints actually land on. With
# PERSIST="everything" cell 4 symlinked models/ to Drive, whose free tier is
# 15 GB -- less than one Flux checkpoint. Cell 2's /content figure would
# happily approve a download that cannot fit.
MODELS_FREE_GB = shutil.disk_usage(MODELS_DIR).free / 2**30
print(f"models/ lands on a filesystem with {MODELS_FREE_GB:.0f} GB free")

ADVICE = recommend(VRAM_GB, MODELS_FREE_GB)
PROFILE = ADVICE.profile if IMAGE_MODEL == "auto" else IMAGE_MODEL
LAUNCH_FLAGS = ADVICE.launch_flags
MAX_SIDE = ADVICE.max_side

if IMAGE_MODEL != "auto" and IMAGE_MODEL != ADVICE.profile:
    print("!" * 70)
    print(f"!! OVERRIDE: you picked '{IMAGE_MODEL}', but this runtime was "
          f"sized for '{ADVICE.profile}'.")
    print("!! Launch flags and the resolution cap still follow the recommended")
    print("!! profile, so expect a failed download or an OOM at model load.")
    for n in ADVICE.notes:
        print("!!  -", n)
    print(f"!! Set IMAGE_MODEL='auto' in cell 1 for '{ADVICE.profile}'.")
    print("!" * 70)

print(f"tier={ADVICE.tier}  profile={PROFILE}  max_side={MAX_SIDE}")
print(f"launch flags: {' '.join(LAUNCH_FLAGS) or '(none)'}")
for n in ADVICE.notes:
    print(" -", n)

# The canny weights are SDXL. On a Flux profile they are 2.33 GB that no
# workflow can use, so drop them rather than download them. Every cell below
# reads USE_CONTROLNET, not the raw form field.
USE_CONTROLNET = CONTROLNET
if USE_CONTROLNET and PROFILE.startswith("flux"):
    print(f"!! ControlNet is SDXL-only; disabled for profile '{PROFILE}'. "
          "Skipping a 2.33 GB unusable download.")
    USE_CONTROLNET = False

SPECS = resolve(PROFILE, controlnet=USE_CONTROLNET, upscale=USE_UPSCALER)
print(f"\\nDownloading {len(SPECS)} files, {total_gb(SPECS)} GB total")
download_all(SPECS, MODELS_DIR, emit=print)
CKPT = CHECKPOINT_NAME[PROFILE]
"""),

        _code("""
#@title 6. Write API workflows for the generate cells
# API format only -- that is what cells 8/8b POST to /prompt. These are not
# ComfyUI sidebar workflows: the sidebar wants UI format, which is a different
# schema and is out of scope. In the tunnel UI, build graphs by hand.
import json, os
from colab_studio import workflows

API_DIR = "/content/wf_api"
os.makedirs(API_DIR, exist_ok=True)

# profile=PROFILE is what keeps a Flux checkpoint out of an SDXL-shaped graph.
built = {
    "txt2img": (workflows.flux_txt2img(CKPT, "a prompt")
                if PROFILE.startswith("flux")
                else workflows.sdxl_txt2img(CKPT, "a prompt")),
    "img2img": workflows.img2img(CKPT, "a prompt", image="input.png",
                                 profile=PROFILE),
}
if USE_UPSCALER:
    built["upscale"] = workflows.upscale(CKPT, "a prompt", profile=PROFILE)
if USE_CONTROLNET:
    built["controlnet_canny"] = workflows.controlnet_canny(
        CKPT, "a prompt", image="input.png", profile=PROFILE)

for name, graph in built.items():
    with open(os.path.join(API_DIR, f"{name}.json"), "w") as fh:
        json.dump(graph, fh, indent=1)
print(f"API workflows written to {API_DIR}: " + ", ".join(built))
"""),

        _code("""
#@title 7. Launch server (backgrounded); tunnel only if OPEN_PUBLIC_UI
import os
from colab_studio.client import ComfyClient
from colab_studio.launch import RuntimeSupervisor

# RuntimeSupervisor owns the server + tunnel lifecycle (single-server guard,
# closed log handles, atexit cleanup on kernel death) so this cell doesn't
# have to reimplement any of that. The single-server guard is per-instance
# (see launch.py), so re-running this cell must reuse the same instance --
# a fresh RuntimeSupervisor() on every run would have no memory of the
# server the previous run started, spawn a second one that fails to bind
# the port, and then silently report "ready" off the first, orphaned one.
try:
    SUPERVISOR
except NameError:
    SUPERVISOR = RuntimeSupervisor()

SERVER_LOG = "/content/comfyui.log"
TUNNEL_LOG = "/content/cloudflared.log"

SERVER = SUPERVISOR.start_server(COMFY_DIR, LAUNCH_FLAGS, SERVER_LOG, port=PORT)
CLIENT = ComfyClient(f"http://127.0.0.1:{PORT}")

print("waiting for server...")
if SUPERVISOR.wait_ready(timeout=300, client=CLIENT):
    print("ComfyUI ready.")
    if OPEN_PUBLIC_UI:
        # A Quick Tunnel URL is obscurity, not authentication: anyone who
        # has it (or finds it) can generate, upload, browse output history,
        # and reach ComfyUI's management-adjacent routes. Nothing below
        # secures it -- treat the URL as fully public.
        print("!" * 70)
        print("!! WARNING: OPEN_PUBLIC_UI is on. Starting a public tunnel.")
        print("!! The URL is NOT authentication. Anyone who has it can use")
        print("!! this ComfyUI server: generate, upload, and read history.")
        print("!" * 70)
        if not os.path.isfile("/usr/local/bin/cloudflared"):
            !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
            !chmod +x /usr/local/bin/cloudflared
        # Tunnel only AFTER readiness, or the URL 502s.
        URL = SUPERVISOR.start_tunnel(PORT, TUNNEL_LOG)
        print("Public URL:", URL or "(tunnel failed, see cell 10)")
    else:
        print("OPEN_PUBLIC_UI is off -- no public tunnel started. Use cells "
              "8/8b to generate here, or set OPEN_PUBLIC_UI=True in cell 1 "
              "and rerun this cell for a tunnel.")
else:
    print("server did not come up - run the log cell below")
"""),

        _code("""
#@title 8. Generate an image here (no tunnel needed)
from IPython.display import Image, display
from colab_studio import workflows
from colab_studio.telemetry import VramProbe, describe

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
# Profile dispatch lives inside the builders (workflows._spine), so passing
# profile=PROFILE is enough: on a Flux profile the cfg above is overridden to
# 1.0 and a FluxGuidance node is wired in, whichever branch is taken.
if use_upscaler:
    graph = workflows.upscale(CKPT, cfg=cfg, profile=PROFILE, **kw)
elif PROFILE.startswith("flux"):
    graph = workflows.flux_txt2img(CKPT, **kw)
else:
    graph = workflows.sdxl_txt2img(CKPT, cfg=cfg, **kw)
if PROFILE.startswith("flux") and cfg != 1.0:
    print(f"note: Flux ignores cfg; using 1.0 with FluxGuidance, not {cfg}.")

# VRAM telemetry: sampled from GET /system_stats once per wait_result() poll
# (1s by default) -- an OBSERVED PEAK, not a true maximum. See
# colab_studio/telemetry.py for why a spike between polls is invisible to it.
vram_probe = VramProbe(CLIENT)
for i, data in enumerate(CLIENT.generate(graph, on_poll=vram_probe.sample)):
    path = f"/content/gen_{i}.png"
    with open(path, "wb") as fh:
        fh.write(data)
    display(Image(filename=path))
print(describe(vram_probe.summary()))
"""),

        _code("""
#@title 8b. Image-to-image / ControlNet (upload or URL)
run_this = False  #@param {type:"boolean"}
source = "upload"  #@param ["upload", "url"]
image_url = ""  #@param {type:"string"}
mode = "img2img"  #@param ["img2img", "controlnet"]
prompt2 = "an oil painting of the same scene"  #@param {type:"string"}
denoise = 0.6  #@param {type:"slider", min:0.1, max:1.0, step:0.05}

# Gated on purpose: source="upload" opens a file picker that blocks the
# kernel, so an ungated body would stall "Runtime > Run all" here and leave
# every cell below unreachable -- exactly the bug this notebook exists to fix.
if not run_this:
    print("Skipped so Run all can continue. Tick run_this, then run this "
          "cell on its own.")
elif mode == "controlnet" and not USE_CONTROLNET:
    print("ControlNet is not available: tick CONTROLNET in cell 1 (SDXL "
          "profiles only) and rerun cell 5 to fetch the weights.")
else:
    import requests
    from IPython.display import Image, display
    from colab_studio import workflows
    from colab_studio.telemetry import VramProbe, describe

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
        graph = workflows.controlnet_canny(CKPT, prompt2, image=server_name,
                                           profile=PROFILE)
    else:
        graph = workflows.img2img(CKPT, prompt2, image=server_name,
                                  denoise=denoise, profile=PROFILE)

    # See cell 8: VRAM telemetry is an OBSERVED PEAK sampled once per poll,
    # not a true maximum.
    vram_probe = VramProbe(CLIENT)
    for i, data in enumerate(CLIENT.generate(graph, on_poll=vram_probe.sample)):
        path = f"/content/edit_{i}.png"
        with open(path, "wb") as fh:
            fh.write(data)
        display(Image(filename=path))
    print(describe(vram_probe.summary()))
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
    SERVER = SUPERVISOR.restart_server(COMFY_DIR, LAUNCH_FLAGS, SERVER_LOG, port=PORT)
    print("restarted:", CLIENT.wait_ready(timeout=300))
elif action == "re-tunnel":
    # Same warning as cell 7: the URL is not authentication.
    print("!! Public tunnel: anyone with the URL can use this server.")
    # Kill the old cloudflared first: it holds an fd on TUNNEL_LOG, and two
    # tunnels on one port leaves the first one unreachable from here.
    print("stopped previous tunnel:", SUPERVISOR.stop_tunnel())
    print("URL:", SUPERVISOR.start_tunnel(PORT, TUNNEL_LOG))
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
| `sdxl` | 6.46 GB | ~12 GB VRAM at 1024px | Best all-rounder on a T4 |
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
| No public URL printed | `OPEN_PUBLIC_UI` is off by default | Set `OPEN_PUBLIC_UI=True` in cell 1 and rerun cell 7, or run cell 10's `re-tunnel` action. **The URL is not authentication** -- anyone who has it can use this server. |
| Tunnel URL 502s | Server still booting | Rerun cell 7; it waits for readiness first |
| `CUDA out of memory` | Resolution or batch too high | Drop to 768px, batch 1, run "free VRAM" in cell 10 |
| `No such file or directory: ...safetensors` | Download interrupted | Rerun cell 5 - it skips completed files |
| Disk full mid-download | Flux is 16 GB | Use `sdxl`, or set `PERSIST="off"` to reclaim Drive space |
| Generate cell hangs | Server died | Check cell 9 log, then "restart server" in cell 10 |
| `ComfyError: execution failed` | A node raised at runtime | The message names the node and exception; usually OOM - drop resolution |
| Cell 8b did nothing | `run_this` is unticked | Tick it; it defaults off so **Run all** does not stall on the file picker |
| ControlNet disabled on Flux | Canny weights are SDXL | Set `IMAGE_MODEL="sdxl"` in cell 1, or use img2img instead |
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
