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
