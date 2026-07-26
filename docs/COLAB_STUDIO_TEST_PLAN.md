# ComfyUI Colab Studio — Manual Test Plan

**For:** you, running on real Colab hardware
**Artifact under test:** `ComfyUI_Colab_Studio.ipynb` (19 cells)
**Branch:** `feat/colab-studio` (21 commits, not merged)

---

## Read this first: what is and isn't proven

**Verified on this machine — 95 automated tests, ruff clean:**

| Area | How it was proven |
|---|---|
| All 5 workflow graphs are structurally valid | Run through `execution.validate_prompt` — the exact code path `POST /prompt` uses. Negative controls (bad sampler, missing checkpoint) correctly fail, so the check isn't vacuous. |
| Every node input key name | Dumped from `INPUT_TYPES()` on ComfyUI 0.10.0, then re-derived independently during review. |
| Every model repo, filename and size | HEAD-requested against huggingface.co. This caught a repo that doesn't exist (`Comfy-Org/stable-diffusion-xl-base-1.0` → 401). |
| `start_server` does not block | A test fails if it takes >5s to return. |
| HTTP client behaviour | Real stub HTTP server on a loopback socket. |
| Notebook independence | No `mcp`, no `uv`/`uvx`, no developer paths, no clone of this repo. |

**NOT verified — nobody has run a single node. This is what you're testing.**

| ID | Claim | Status |
|---|---|---|
| **E1** | Flux fp8 all-in-one loads via `CheckpointLoaderSimple` | Inferred from file layout + 16.06 GB size. Never executed. |
| **E2** | Diffusers ControlNet converts at runtime (`comfy/controlnet.py:623`) | Read from source. Never executed. |
| **E3** | VRAM headroom per model on real hardware | Unmeasured. |
| **E4** | The VRAM→profile→flags thresholds (12.0 / 20.0 GB) | Implemented and boundary-unit-tested. **Never calibrated against a real GPU.** |

If something below fails, that's the test doing its job. Record it rather than working around it.

---

## Phase 0 — Get the notebook into Colab (5 min)

**Option A — open straight from GitHub (fastest, no upload).** The branch is pushed to your fork, so Colab can open it directly:

```
https://colab.research.google.com/github/Dennis-J-Carroll/ComfyUI/blob/feat/colab-studio/ComfyUI_Colab_Studio.ipynb
```

Colab opens it read-only from the branch; use *File → Save a copy in Drive* if you want to keep your edits. PR for review: https://github.com/Dennis-J-Carroll/ComfyUI/pull/1

**Option B — I push it to your Drive.** Say the word and I'll upload via the Google Drive MCP and hand you the link. Nothing has gone into your Drive so far.

**Option C — upload manually.** File is at `/home/dennisjcarroll/Desktop/ComfyUI/ComfyUI_Colab_Studio.ipynb`. In Colab: File → Upload notebook.

**Then, before running anything:** Runtime → Change runtime type → **GPU**. Colab defaults to CPU. Cell 2 will tell you if you forgot, but you'll have wasted the install.

---

## Phase 1 — Preflight and the E4 calibration (10 min, no downloads)

**Run cells 1 and 2 only.** Stop there. This is the cheapest, highest-value step — it needs no model downloads, so do it first even if you abandon the rest.

Cell 1 is config (leave defaults: `IMAGE_MODEL=auto`, `PERSIST=outputs-only`, `CONTROLNET=False`, `USE_UPSCALER=True`).

Cell 2 prints your GPU, VRAM and free disk.

### ✅ Pass criteria
- Prints a real GPU name and VRAM figure, not `!! No GPU`.
- Prints free disk in GB.

### 📋 Record this — it's the E4 calibration data

| What cell 2 printed | Your value |
|---|---|
| GPU name | |
| VRAM (GB) | |
| Free disk (GB) | |

**Then run cell 11 (`5. Choose profile and download models`) — but read its first six printed lines and be ready to interrupt (⏹) before the download starts if the profile looks wrong.**

| What it printed | Your value |
|---|---|
| `tier=` | |
| `profile=` | |
| `max_side=` | |
| `launch flags:` | |

### 🎯 The E4 judgement — this is the part only you can make

Compare against the thresholds I implemented (`<12 GB` = low, `12–20` = mid, `≥20` = high):

- **T4 (~14.7 GB)** should give `tier=mid`, `profile=sdxl`, `max_side=1024`, flags `--normalvram`. **It must NOT say `--highvram`** — that flag OOMs a T4, and it's the single most important line to check.
- **L4 (~22.5 GB)** should give `tier=high`, `profile=flux-dev`, `--highvram`.
- **A100 (~40 GB)** should give `tier=high`.

**Does the recommendation match what you know actually works on this GPU?** If it recommends Flux on something that will thrash, or caps resolution lower than you'd accept, the thresholds are wrong — tell me the numbers and I'll recalibrate. The constants live in `colab_studio/advice.py:15-16`.

---

## Phase 2 — Install and first image (20–30 min, mostly download wait)

Run cells 3 → 14 in order. Cells 5–10 write library files and complete instantly.

Cell 11 downloads models — SDXL is ~6.8 GB, Flux ~16.1 GB. This is the slow part.

### Checkpoints along the way

| Cell | Expect | ❌ If not |
|---|---|---|
| 3. Install | Clones ComfyUI, pip completes | Note the error; usually a transient pip failure — rerun |
| 4. Persistence | Drive auth prompt, then `Persisting: output, user` | |
| 11. Download | Per-file `[+] name (X GB)` then `[v] name ready` | If it dies mid-download, rerun — it skips completed files |
| 12. Workflows | `API workflows written to /content/wf_api: …` | |
| 13. Launch | `waiting for server...` then `ComfyUI ready. Public URL: https://….trycloudflare.com` | See Phase 5 |

### 🔴 The critical structural test — cell 13

**The whole point of this rewrite is that cell 13 finishes and returns control.** The old notebook's launch cell ran forever, making everything below it unreachable.

- ✅ **PASS:** cell 13 shows a completed execution counter and prints a URL. Cells 14+ are runnable.
- ❌ **FAIL:** cell 13 spins indefinitely. That's the original bug back — stop and tell me.

**Open the tunnel URL.** It should load the ComfyUI graph UI, not a 502. A 502 means the tunnel started before the server was ready — the ordering fix failed.

### Cell 14 — generate an image without leaving Colab

Defaults are fine. Run it.

- ✅ **PASS:** an image renders **inline in the notebook**. You never touched the tunnel.
- ❌ **FAIL:** record the exact error.

**This validates E1 if `profile=flux-dev`** — the 16 GB all-in-one actually loading through `CheckpointLoaderSimple` is the single biggest untested assumption in the build. If you're on SDXL, E1 stays unverified; to test it, set `IMAGE_MODEL="flux-dev"` in cell 1 and rerun from cell 11.

### 📋 Record

| Item | Result |
|---|---|
| Cell 13 returned control (not blocking) | ☐ yes ☐ no |
| Tunnel URL loaded (no 502) | ☐ yes ☐ no |
| Cell 14 produced an inline image | ☐ yes ☐ no |
| Seconds per image | |
| **E1** — Flux loaded via CheckpointLoaderSimple | ☐ pass ☐ fail ☐ not tested (SDXL) |
| Peak VRAM (cell 17 → `free VRAM`, or `!nvidia-smi`) | **E3:** |

---

## Phase 3 — img2img and the upload path (10 min)

Run **cell 15** (`8b. Image-to-image / ControlNet`). **Tick `run_this` first** — it defaults off so that *Run all* is not stalled by the file picker, and the cell just prints a skip message until you tick it. Then leave `source="upload"`, `mode="img2img"`. Pick any image.

This exercises the one piece the tunnel UI would otherwise handle for you: `POST /upload/image` → the returned `name` feeding a `LoadImage` node.

- ✅ **PASS:** file picker appears, upload completes, a transformed image renders inline.
- ❌ **FAIL:** a `ComfyError` naming the upload, or a `LoadImage` complaint that the file isn't found.

Also try `source="url"` with any direct image URL.

---

## Phase 4 — ControlNet (optional, +2.33 GB) — this is E2

Only if you want ControlNet, and only on an **SDXL** profile — on Flux, cell 11 disables it by design. **Set `CONTROLNET=True` in cell 1, rerun cell 11** (downloads 2.33 GB), then cell 15 with `run_this` ticked and `mode="controlnet"`.

**This is the E2 test, and the riskiest download in the project.** The file is diffusers-layout (`diffusion_pytorch_model.fp16.safetensors`, renamed on the way down). I read `comfy/controlnet.py:623` and found an explicit diffusers branch, so it *should* convert — but nobody has run it. A 2.33 GB download that fails to load is the worst failure mode here.

- ✅ **PASS:** image generates, following your input's edges.
- ❌ **FAIL:** likely a state-dict/key error on load. **Send me the full traceback** — that's exactly what E2 was flagged to catch.

| **E2** — diffusers ControlNet converted and ran | ☐ pass ☐ fail ☐ not tested |
|---|---|

---

## Phase 5 — Ops cells and failure recovery (5 min)

These exist because cell 13 no longer blocks. Verify each is reachable:

| Cell | Action | Expect |
|---|---|---|
| 16 | Run as-is | Last 60 lines of the ComfyUI server log |
| 17 | `disk usage` | Free/total GB |
| 17 | `list models` | Each downloaded file with its size |
| 17 | `free VRAM` | `asked ComfyUI to unload models` |
| 17 | `restart server` | `restarted: True` |
| 17 | `re-tunnel` | A fresh `trycloudflare.com` URL |

**Deliberately break it — two failure paths worth exercising, because both were bugs the review caught:**

1. **Restart race.** Run `restart server`, then immediately run cell 14 before boot finishes. It should wait, or raise a clear `ComfyError` — not hang, not mislead.
2. **Execution error surfaces fast.** Force an OOM: set `width=height=2048` in cell 14 on a small GPU. You should get a **`ComfyError` naming the failing node within seconds**. If instead it sits silent for ten minutes and then raises a bare `TimeoutError`, that's the pre-fix behaviour and a regression — `wait_result` is supposed to read the history `status` field now.
3. **Re-tunnel doesn't leak.** Run `re-tunnel` twice. Each run should print a fresh URL and the previous tunnel should be stopped, not left running. If old `trycloudflare.com` URLs keep working after a re-tunnel, the previous process was orphaned.

---

## Phase 6 — Persistence across sessions (do this later)

Runtime → Disconnect and delete runtime. Reopen the notebook. Run all.

- ✅ **PASS:** cell 4 remounts Drive, your `output/` images from last session are still there.
- ⚠️ **EXPECTED:** models re-download. That's deliberate — models live on VM disk, not Drive, because one Flux checkpoint (16.06 GB) exceeds the free Drive quota (15 GB) by itself. Set `PERSIST="everything"` to change it, and accept slow FUSE reads.

---

## Known limits — don't report these as bugs

| Behaviour | Why |
|---|---|
| Flux ignores your `cfg` slider | Flux has no classifier-free guidance. cfg is forced to 1.0; real guidance rides on `FluxGuidance`. Any other cfg scorches output. |
| `img2img` ignores width/height | Its latent comes from your source image. |
| There is no video option | `MODE` was removed: it installed three custom-node repos and then generated SDXL images anyway. Video is `Wan2.2_Colab_Pipeline.ipynb`'s job. |
| ControlNet is SDXL-only | Flux ControlNet is 22.17 GB — ruled out as prohibitive. On a Flux profile, cell 5 disables ControlNet and says so rather than downloading 2.33 GB of unusable weights. |
| Cell 8b does nothing until you tick `run_this` | Its file picker would block **Run all** and make the cells below unreachable. Tick `run_this`, then run 8b on its own. |
| No workflows in the ComfyUI sidebar | Only API-format graphs are produced, written to `/content/wf_api/` for cells 8 and 8b. The sidebar needs UI format, which is out of scope. |

---

## Scorecard — copy this block, fill it, paste it back

```
HARDWARE
  GPU name .................
  VRAM (GB) ................
  Free disk (GB) ...........

E4 CALIBRATION  (cell 5 output)
  tier= ....................
  profile= .................
  max_side= ................
  launch flags= ............
  Does this match what you know works on this GPU?  yes / no / notes:

STRUCTURAL  (the things this rewrite exists to fix)
  Cell 13 returned control, did not block ........ pass / fail
  Tunnel URL loaded, no 502 ...................... pass / fail
  Cell 14 rendered an image inline ............... pass / fail
  Run all reached cell 17 without stalling ....... pass / fail
  Cells 16/17 reachable and rerunnable ........... pass / fail

EXECUTION CRITERIA
  E1  Flux loaded via CheckpointLoaderSimple ..... pass / fail / not tested
  E2  Diffusers ControlNet converted and ran ..... pass / fail / not tested
  E3  Peak VRAM observed (GB) ....................
  E4  Thresholds correct for this GPU ............ yes / no

FAILURE-PATH CHECKS
  OOM raised ComfyError in seconds, not a 10-min hang ... pass / fail
  re-tunnel stopped the previous tunnel ................. pass / fail

TIMINGS
  Install + download (min) ........
  Seconds per image ...............
```

## Reporting back

Paste the scorecard plus, for anything that failed:
1. The cell number
2. The full traceback
3. Cell 16's log output

The three things I most want, in order: **the E4 calibration numbers from Phase 1** (cheapest, and the thresholds are currently guesses), **whether cell 13 returned control** (the whole architecture rests on it), and **the E2 ControlNet traceback if it fails** (the riskiest untested path).

---

## Appendix — running the automated suite yourself

```bash
cd /home/dennisjcarroll/Desktop/ComfyUI
.venv/bin/python -m pytest tests-unit/colab_studio_test/ -v   # 62 tests
~/.local/bin/ruff check colab_studio/ tests-unit/colab_studio_test/ build_notebook.py
```

To regenerate the notebook after editing any `colab_studio/` module:

```bash
.venv/bin/python build_notebook.py
```

The modules are inlined into the notebook at build time, so **editing the `.ipynb` directly gets overwritten** — edit the module and rebuild.
