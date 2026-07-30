# ComfyUI Colab Studio — Manual Test Plan

**For:** you, running on real Colab hardware
**Artifact under test:** `ComfyUI_Colab_Studio.ipynb`
**Branch:** PR #1 (`feat/colab-studio`) is **MERGED** into `master`
(merge commit `6f572b1c`). Phase 0 hardening (ComfyUI revision pin,
opt-in tunnel, RuntimeSupervisor lifecycle) is **also merged** via PR #2.
VRAM telemetry is open in PR #3 — merge it before running, or `#@title 8`
will not print the peak-VRAM line.

---

## Read this first: what is and isn't proven

**Verified on this machine — 132 automated tests
(`tests-unit/colab_studio_test/`, computed by running the suite, not
copied from an earlier count), ruff clean:**

| Area | How it was proven |
|---|---|
| All 5 workflow graphs are structurally valid | Run through `execution.validate_prompt` — the exact code path `POST /prompt` uses. Negative controls (bad sampler, missing checkpoint) correctly fail, so the check isn't vacuous. |
| Every node input key name | Dumped from `INPUT_TYPES()` on ComfyUI 0.10.0, then re-derived independently during review. |
| Every model repo, filename and size | HEAD-requested against huggingface.co. This caught a repo that doesn't exist (`Comfy-Org/stable-diffusion-xl-base-1.0` → 401). |
| `start_server` does not block | A test fails if it takes >5s to return. |
| HTTP client behaviour | Real stub HTTP server on a loopback socket. |
| Notebook independence | No `mcp`, no `uv`/`uvx`, no developer paths, no clone of this repo. |
| **E5** — fetch → launch → readiness → generate, end to end | Executed against a real ComfyUI server on CPU (SD1.5, 384×384, 8 steps): 1 image in 145 s. `start_server` returned in 0.00 s against a real server, not a stub. Does **not** touch E1–E4 below — those remain GPU- and model-specific. Full record: `docs/superpowers/specs/2026-07-25-comfyui-colab-studio-design.md` §E5. |

**Still open. E2 is the last untested code path in the image pipeline** — see `docs/E2_CONTROLNET_TEST_RUN.md` for a step-by-step.

| ID | Claim | Status |
|---|---|---|
| ~~**E1**~~ | Flux fp8 all-in-one loads via `CheckpointLoaderSimple` | ✅ **CLOSED** — A100, 2026-07-27. 1024×1024 in 13.85 s, ~1.91 it/s. |
| **E2** | Diffusers ControlNet converts at runtime (`comfy/controlnet.py:623`) | Read from source. Never executed. |
| **E3** | VRAM headroom per model on real hardware | Partial — Flux ran without OOM on a 40 GB A100, but the *margin* was never measured. PR #3 adds the measurement; this run captures it. |
| **E4** | The VRAM→profile→flags thresholds (12.0 / 20.0 GB) | High tier ✅ confirmed on A100. **Low (<12 GB) and mid (12–20 GB) still uncalibrated** — no T4 or L4 run. The T4 case matters most: `--highvram` there causes OOM. |

If something below fails, that's the test doing its job. Record it rather than working around it.

---

## Tunnel behaviour (`OPEN_PUBLIC_UI`)

The launch cell no longer starts a public Cloudflare Quick Tunnel by
default. This changed after the merge above — if you're testing an older
checkout, this section won't apply yet.

- **Default (`OPEN_PUBLIC_UI = False` in cell 1):** the server starts, cell
  7 waits for readiness, and **no tunnel is opened**. `Runtime > Run all`
  reaches every cell with no public URL ever printed. Use cell 8 / 8b to
  generate inline — that path needs no tunnel at all.
- **To enable:** set `OPEN_PUBLIC_UI = True` in cell 1, then run (or
  rerun) cell 7. It prints a warning *before* the tunnel starts, then
  starts it — only after `wait_ready()` has already succeeded, same
  ordering guarantee as before.
- **What to check:**
  - With the default off, confirm no `trycloudflare.com` URL appears
    anywhere in cell 7's output on a plain `Run all`.
  - With it on, confirm the warning prints *before* the URL, and that the
    warning says plainly that the URL is not authentication.
  - Cell 10's `re-tunnel` action still works for restarting a tunnel
    manually (independent of `OPEN_PUBLIC_UI`, since picking that action
    is itself an explicit request) and still stops the previous tunnel
    first — check no stale `trycloudflare.com` URL keeps answering after
    a re-tunnel.
- **Nothing about the tunnel is secured.** It's a public URL with no
  authentication in front of it, whether opened from cell 7 or cell 10.
  Treat it as fully public for as long as it's up.

---

## Phase 0 — Get the notebook into Colab (5 min)

**Option A — open straight from GitHub (fastest, no upload).** PR #1 is
merged into `master`, so Colab can open the merged notebook directly:

```
https://colab.research.google.com/github/Dennis-J-Carroll/ComfyUI/blob/master/ComfyUI_Colab_Studio.ipynb
```

For a reproducible link that won't move under you as `master` gets more
commits, pin to the merge commit instead:

```
https://colab.research.google.com/github/Dennis-J-Carroll/ComfyUI/blob/6f572b1c/ComfyUI_Colab_Studio.ipynb
```

Colab opens either read-only from that ref; use *File → Save a copy in
Drive* if you want to keep your edits. Merged PR: https://github.com/Dennis-J-Carroll/ComfyUI/pull/1

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

> **Cell numbering.** Colab's cell indices and the `#@title` numbers do **not** match — eight `%%writefile` library cells sit between "4." and "5.". Always go by the `#@title` label. Current mapping: `1`→idx 1, `2`→2, `3`→3, `4`→4, library→5–12, `5`→**13**, `6`→14, `7`→**15**, `8`→**16**, `8b`→**17**, `9`→18, `10`→**19**. This shifts whenever a library module is added, which is why the `#@title` numbers are the stable reference.

**Then run `#@title 5` (`Choose profile and download models`, index 13) — but read its first six printed lines and be ready to interrupt (⏹) before the download starts if the profile looks wrong.**

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

Run `#@title 3` → `#@title 8` in order (indices 3–16). The eight library cells (indices 5–12) write files and finish instantly.

`#@title 5` downloads models — SDXL is ~6.8 GB, Flux ~16.1 GB. This is the slow part.

### Checkpoints along the way

| Cell | Expect | ❌ If not |
|---|---|---|
| 3. Install | Clones ComfyUI, pip completes | Note the error; usually a transient pip failure — rerun |
| 4. Persistence | Drive auth prompt, then `Persisting: output, user` | |
| 11. Download | Per-file `[+] name (X GB)` then `[v] name ready` | If it dies mid-download, rerun — it skips completed files |
| 12. Workflows | `API workflows written to /content/wf_api: …` | |
| 13. Launch | `waiting for server...` then `ComfyUI ready. Public URL: https://….trycloudflare.com` | See Phase 5 |

### 🔴 The critical structural test — `#@title 7` (index 15)

**The whole point of this rewrite is that `#@title 7` finishes and returns control.** The old notebook's launch cell ran forever, making everything below it unreachable.

- ✅ **PASS:** `#@title 7` shows a completed execution counter and returns. Every cell below it is runnable.
- ❌ **FAIL:** `#@title 7` spins indefinitely. That's the original bug back — stop and tell me.

**Open the tunnel URL.** It should load the ComfyUI graph UI, not a 502. A 502 means the tunnel started before the server was ready — the ordering fix failed.

### Cell 14 — generate an image without leaving Colab

Defaults are fine. Run it.

- ✅ **PASS:** an image renders **inline in the notebook**. You never touched the tunnel.
- ❌ **FAIL:** record the exact error.

**This validates E1 if `profile=flux-dev`** — the 16 GB all-in-one actually loading through `CheckpointLoaderSimple` is the single biggest untested assumption in the build. If you're on SDXL, E1 stays unverified; to test it, set `IMAGE_MODEL="flux-dev"` in cell 1 and rerun from `#@title 5`.

### 📋 Record

| Item | Result |
|---|---|
| Cell 13 returned control (not blocking) | ☐ yes ☐ no |
| Tunnel URL loaded (no 502) | ☐ yes ☐ no |
| Cell 14 produced an inline image | ☐ yes ☐ no |
| Seconds per image | |
| **E1** — Flux loaded via CheckpointLoaderSimple | ☐ pass ☐ fail ☐ not tested (SDXL) |
| Observed peak VRAM (printed by `#@title 8` itself) | **E3:** |

---

## Phase 3 — img2img and the upload path (10 min)

Run **`#@title 8b`** (index 17). **Tick `run_this` first** — it defaults off so that *Run all* is not stalled by the file picker, and the cell just prints a skip message until you tick it. Then leave `source="upload"`, `mode="img2img"`. Pick any image.

This exercises the one piece the tunnel UI would otherwise handle for you: `POST /upload/image` → the returned `name` feeding a `LoadImage` node.

- ✅ **PASS:** file picker appears, upload completes, a transformed image renders inline.
- ❌ **FAIL:** a `ComfyError` naming the upload, or a `LoadImage` complaint that the file isn't found.

Also try `source="url"` with any direct image URL.

---

## Phase 4 — ControlNet (optional, +2.33 GB) — this is E2

Only if you want ControlNet, and only on an **SDXL** profile — on Flux, `#@title 5` disables it by design. **Set `CONTROLNET=True` in cell 1, rerun `#@title 5`** (downloads 2.33 GB), then `#@title 8b` with `run_this` ticked and `mode="controlnet"`.

**This is the E2 test, and the riskiest download in the project.** The file is diffusers-layout (`diffusion_pytorch_model.fp16.safetensors`, renamed on the way down). I read `comfy/controlnet.py:623` and found an explicit diffusers branch, so it *should* convert — but nobody has run it. A 2.33 GB download that fails to load is the worst failure mode here.

- ✅ **PASS:** image generates, following your input's edges.
- ❌ **FAIL:** likely a state-dict/key error on load. **Send me the full traceback** — that's exactly what E2 was flagged to catch.

| **E2** — diffusers ControlNet converted and ran | ☐ pass ☐ fail ☐ not tested |
|---|---|

---

## Phase 5 — Ops cells and failure recovery (5 min)

These exist because `#@title 7` no longer blocks. Verify each is reachable:

| Cell | Action | Expect |
|---|---|---|
| 16 | Run as-is | Last 60 lines of the ComfyUI server log |
| 17 | `disk usage` | Free/total GB |
| 17 | `list models` | Each downloaded file with its size |
| 17 | `free VRAM` | `asked ComfyUI to unload models` |
| 17 | `restart server` | `restarted: True` |
| 17 | `re-tunnel` | A fresh `trycloudflare.com` URL |

**Deliberately break it — two failure paths worth exercising, because both were bugs the review caught:**

1. **Restart race.** Run `restart server`, then immediately run `#@title 8` before boot finishes. It should wait, or raise a clear `ComfyError` — not hang, not mislead.
2. **Execution error surfaces fast.** Force an OOM: set `width=height=2048` in `#@title 8` on a small GPU. You should get a **`ComfyError` naming the failing node within seconds**. If instead it sits silent for ten minutes and then raises a bare `TimeoutError`, that's the pre-fix behaviour and a regression — `wait_result` is supposed to read the history `status` field now.
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
| There is no video option | `MODE` was removed: it installed three custom-node repos and then generated SDXL images anyway. Video generation is not yet shipped in Colab Studio. |
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
  Run all reached the ops cell without stalling ..... pass / fail
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

The three things I most want, in order: **the E4 calibration numbers from Phase 1** (cheapest, and the thresholds are currently guesses), **whether `#@title 7` returned control** (the whole architecture rests on it), and **the E2 ControlNet traceback if it fails** (the riskiest untested path).

---

## Appendix — running the automated suite yourself

```bash
cd /home/dennisjcarroll/Desktop/ComfyUI
.venv/bin/python -m pytest tests-unit/colab_studio_test/ -v   # count drifts -- see the collected total the run prints
~/.local/bin/ruff check colab_studio/ tests-unit/colab_studio_test/ build_notebook.py
```

To regenerate the notebook after editing any `colab_studio/` module:

```bash
.venv/bin/python build_notebook.py
```

The modules are inlined into the notebook at build time, so **editing the `.ipynb` directly gets overwritten** — edit the module and rebuild.
