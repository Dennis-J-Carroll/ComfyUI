"""Makes this package robust to test-collection order (Phase 0 item 0.h).

`comfy/cli_args.py` parses `sys.argv` into a module-level `args` Namespace
exactly once, at cli_args' own import time, and only if
`comfy.options.args_parsing` has already been enabled -- otherwise it
silently parses `[]` and gets the argparse defaults (`args.cpu == False`).
`comfy/model_management.py` then reads that same frozen `args.cpu` at ITS
OWN import time to decide `cpu_state`, and immediately derives `vram_state`
and `total_vram` from that decision in module-level code -- so once either
module has been imported with the wrong `args.cpu`, nothing done afterwards
via `enable_args_parsing()` can retroactively fix it.

workflows_test.py needs CPU mode -- this machine, and most CI runners, are
CPU-only. Its fixture used to set `sys.argv` and call
`comfy.options.enable_args_parsing()` itself, right before `import nodes`.
That works when this package happens to be collected first. It silently
breaks when an earlier-collected package already imported `comfy.cli_args`
with args_parsing still disabled -- confirmed via instrumentation: running
`pytest tests-unit/assets_test tests-unit/colab_studio_test/workflows_test.py`
imports `comfy.cli_args` during collection of
`tests-unit/assets_test/queries/test_asset.py` (args_parsing is False at
that point, so `args.cpu` freezes False), while `comfy.model_management`
itself is NOT yet imported. `import nodes` inside workflows_test.py's
fixture then imports `comfy.model_management` for the first time, which
reads the frozen `args.cpu = False` and crashes immediately, mid-import, at
`total_vram = get_total_memory(get_torch_device())` with "Torch not
compiled with CUDA enabled" on any CPU-only torch build. Because the crash
happens DURING that import, nothing the fixture could do to
`comfy.model_management` *after* `import nodes` would help -- the fix has
to land before `comfy.model_management` is ever imported.

pytest imports a package's conftest.py before collecting/importing its test
modules, so doing the following here, at *module* scope, guarantees it runs
before workflows_test.py's fixture ever calls `import nodes`, regardless of
what an earlier-collected package already touched this session.
"""
from __future__ import annotations

import sys

import comfy.options

sys.argv = ["main.py", "--cpu"]
comfy.options.enable_args_parsing()

# --- Defend against modules an earlier-collected package already cached. ---
#
# (a) comfy.cli_args imported while args_parsing was still disabled: its
#     `args` Namespace was already built from `parser.parse_args([])` and
#     will never be re-parsed. Patch the one field comfy.model_management
#     actually reads at its own import time. This is the branch that fires
#     in the reproduced hostile order described above -- verified: without
#     it, `import nodes` inside the workflows_test fixture crashes at
#     comfy/model_management.py's module level before the fixture body
#     resumes.
_cli_args = sys.modules.get("comfy.cli_args")
if _cli_args is not None:
    _cli_args.args.cpu = True

# (b) comfy.model_management itself already imported (and therefore already
#     ran its own CPUState/VRAMState decision at import time). A no-op in
#     the hostile order above, since (a) prevents model_management from
#     ever being imported with the wrong state in the first place -- but a
#     defensive guard for any other hostile order where model_management is
#     the module that ends up cached. Best effort: `total_vram`, computed
#     from the GPU path at that module's own import time, is not
#     recomputed here. `vram_state` is forced to DISABLED alongside
#     `cpu_state`, mirroring model_management.py's own
#     `if cpu_state != CPUState.GPU: vram_state = VRAMState.DISABLED`, so
#     downstream code never observes a state combination the module itself
#     would not have produced.
_mm = sys.modules.get("comfy.model_management")
if _mm is not None:
    _mm.cpu_state = _mm.CPUState.CPU
    _mm.vram_state = _mm.VRAMState.DISABLED
