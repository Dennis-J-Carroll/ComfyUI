# E2 — ControlNet run: step-by-step

**Goal:** prove that the diffusers-layout SDXL ControlNet state dict actually converts and runs. It has never been executed. I read `comfy/controlnet.py` and found an explicit diffusers branch, so it *should* work — but a 2.33 GiB download that fails to load is the worst failure mode in this project, which is why this run exists.

**Time:** ~15 min, mostly download. **Download:** 8.85 GiB.

**Cell references use the `#@title` numbers** (stable), with the Colab index in brackets. They don't match — eight `%%writefile` cells sit between "4." and "5.".

| `#@title` | Colab index |
|---|---|
| 1. Config | 1 |
| 2. Preflight | 2 |
| 3. Install | 3 |
| 4. Persistence | 4 |
| *(library cells)* | 5–12 |
| 5. Choose profile / download | **13** |
| 6. Write API workflows | 14 |
| 7. Launch server | 15 |
| 8. Generate | **16** |
| 8b. img2img / ControlNet | **17** |
| 9. Server log | 18 |
| 10. Ops | 19 |

---

## Step 0 — Prerequisites

1. **Merge PR #3** first, or you won't get the VRAM telemetry line this run is also meant to capture.
2. Open the notebook from master:
   ```
   https://colab.research.google.com/github/Dennis-J-Carroll/ComfyUI/blob/master/ComfyUI_Colab_Studio.ipynb
   ```
3. **Runtime → Change runtime type → GPU.** Confirm bottom-right shows a GPU.

If you still have a live A100 session with Flux already downloaded, you can reuse it — skip to Step 2 and just rerun from `#@title 5`. Nothing needs a restart; ComfyUI loads checkpoints on demand.

---

## Step 1 — Configure (`#@title 1`, index 1)

Change **two** fields, leave the rest:

| Field | Set to | Why |
|---|---|---|
| `IMAGE_MODEL` | **`sdxl`** | `auto` picks `flux-dev` on an A100, and ControlNet is SDXL-only |
| `CONTROLNET` | **`True`** | fetches the canny weights |
| `PERSIST` | `outputs-only` | unchanged |
| `USE_UPSCALER` | `True` | unchanged |
| `OPEN_PUBLIC_UI` | **`False`** | leave off; nothing here needs the tunnel |
| `COMFY_REF` | `pinned` | leave pinned |

Run it. Expect a one-line echo of the settings.

---

## Step 2 — Run `#@title 2` → `#@title 5` (indices 2–13)

Run each in order. The eight library cells (5–12) write files and finish instantly.

### At `#@title 5` (index 13) — expect an OVERRIDE banner

**This is not an error.** You're asking for `sdxl` on hardware sized for `flux-dev`, so the Important-8 warning fires:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! OVERRIDE: you picked 'sdxl', but this runtime was sized for 'flux-dev'.
!! Launch flags and the resolution cap still follow the recommended
!! profile, so expect a failed download or an OOM at model load.
...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

In this specific case the warning over-worries: SDXL on an A100 with `--highvram` is comfortable. The warning is generic and can't know that. Proceed.

Then expect:

```
tier=high  profile=sdxl  max_side=1024
launch flags: --highvram

Downloading 3 files, 8.85 GB total
[+] sd_xl_base_1.0.safetensors (6.46 GB) from stabilityai/stable-diffusion-xl-base-1.0
[v] sd_xl_base_1.0.safetensors ready
[=] 4x-UltraSharp.pth already present, skipping        <- if reusing the session
[+] controlnet-canny-sdxl.safetensors (2.33 GB) from diffusers/controlnet-canny-sdxl-1.0
[v] controlnet-canny-sdxl.safetensors ready
```

### 📋 Checkpoint A — the rename

The HF file is named `diffusion_pytorch_model.fp16.safetensors`. The line above must say **`controlnet-canny-sdxl.safetensors`**. That rename is what stops every diffusers-layout model colliding on one generic filename.

❌ If it lands as `diffusion_pytorch_model.fp16.safetensors`, `ModelSpec.target_name` isn't being honoured — stop and tell me.

### 📋 Checkpoint B — no skip message

You must **not** see `ControlNet is SDXL-only; disabled for profile ...`. That guard only fires on Flux. Seeing it means `IMAGE_MODEL` didn't take — recheck cell 1.

---

## Step 3 — `#@title 6` (index 14): confirm the graph was built

Expect `controlnet_canny` in the list:

```
API workflows written to /content/wf_api: txt2img, img2img, upscale, controlnet_canny
```

❌ Missing `controlnet_canny` → `USE_CONTROLNET` came out False. Stop.

---

## Step 4 — `#@title 7` (index 15): launch

```
waiting for server...
ComfyUI ready.  Public URL: (tunnel disabled - set OPEN_PUBLIC_UI=True in cell 1 to open one)
```

Must print **no** `trycloudflare.com` URL.

---

## Step 5 — `#@title 8` (index 16): SDXL baseline + VRAM

Defaults are fine. Run it.

Purpose is twofold: confirm SDXL still works on this profile, and capture its peak VRAM for comparison against Flux.

### 📋 Record

```
SDXL, 1024x1024, 25 steps
  observed peak VRAM line: ______________________________________
  Prompt executed in (from #@title 9): ______ seconds
```

Note `cfg` is respected here — no `Flux ignores cfg` line, since SDXL genuinely uses CFG.

---

## Step 6 — `#@title 8b` (index 17): **the E2 test**

| Field | Set to |
|---|---|
| `run_this` | **✅ tick it** (defaults off so *Run all* isn't blocked by the file picker) |
| `source` | `upload` |
| `mode` | **`controlnet`** |
| `prompt2` | anything, e.g. `a stained glass window, vivid colours` |
| `denoise` | ignored in controlnet mode |

Run it, pick any image with clear edges — architecture, a line drawing, or a high-contrast photo works best for canny.

### ✅ PASS

An image generates that **follows your input's edge structure** while taking style from `prompt2`. Silhouette recognisable, content reimagined.

### ❌ FAIL — and what each means

| Symptom | Meaning |
|---|---|
| `ComfyError: execution failed … [node 14 ControlNetLoader]` | **This is E2 failing.** The diffusers state dict didn't convert. **Send me the whole traceback** — this is the exact thing E2 was flagged to catch. |
| `ControlNet is not available: tick CONTROLNET in cell 1` | `USE_CONTROLNET` False — go back to Step 1 |
| `ValueError: controlnet_canny is SDXL-only` | Profile is still Flux — `IMAGE_MODEL` didn't take |
| Output ignores input edges entirely | Loaded but not conditioning; possibly a `strength` or wiring issue. Send both images. |
| `CUDA out of memory` | Unlikely on an A100. Drop to 768. |

### 📋 Record

```
E2 result: ☐ pass  ☐ fail
  observed peak VRAM line: ______________________________________
  if failed, full traceback + #@title 9 log
```

---

## Step 7 — Optional: img2img sanity (2 min)

Same cell, `mode="img2img"`, `denoise=0.6`. Should return a restyled version of your source, recognisably the same composition. Confirms the SDXL img2img path, which so far has only run on Flux.

---

## What to send back

```
STEP 2  rename correct (controlnet-canny-sdxl.safetensors)?  yes / no
STEP 3  controlnet_canny present in workflow list?            yes / no
STEP 4  no public URL printed?                                yes / no
STEP 5  SDXL peak VRAM line: ...................................
        SDXL seconds: ......
STEP 6  E2: pass / fail
        ControlNet peak VRAM line: .............................
        if fail: traceback + server log
STEP 7  img2img: pass / fail / skipped
```

Those close **E2** and **E3** (Flux and SDXL peaks give the real headroom margin, which is what `advice.py`'s thresholds should be based on rather than my estimates).

---

## What this run cannot tell us

Low (<12 GB) and mid (12–20 GB) tier thresholds. Running SDXL on an A100 exercises the *code path* but not the *memory envelope* — a T4 has 14.7 GB and the question there is whether SDXL at 1024px actually fits. Only a T4 session answers that, and it's the tier where being wrong causes an OOM rather than wasted headroom.
