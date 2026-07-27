"""Regression guard for Phase 0 item 0.h.

workflows_test.py's fixture needs CPU mode, and used to arrange for it
itself (sys.argv + comfy.options.enable_args_parsing()) right before
`import nodes`. That silently broke whenever an earlier-collected test
package already imported comfy.cli_args with args_parsing still disabled:
`comfy.cli_args.args.cpu` freezes False at cli_args' own import time and is
never re-parsed, so by the time `import nodes` (inside the fixture) pulls
in comfy.model_management for the first time, it reads the frozen False and
crashes mid-import with "Torch not compiled with CUDA enabled" on any
CPU-only torch build -- see conftest.py in this package for the fix and
full mechanism.

The damage is process-global (module import caching), so the only honest
way to prove order-independence is to run pytest in a child process with
the hostile order and inspect what actually happened.

tests-unit/assets_test/services/test_asset_management.py is the minimal
reproducer. It was found by instrumenting pytest's `pytest_collectreport`
hook to check `sys.modules` after each module in tests-unit/assets_test was
collected: this is the specific module whose *collection* (importing
app.assets.services.asset_management, which transitively imports
comfy.cli_args) first freezes `comfy.cli_args.args.cpu = False` for the
process, before comfy.options.args_parsing is ever enabled. (A first
attempt at this reproducer used queries/test_asset.py, based on which test
item first observed comfy.cli_args already cached -- that turned out to be
a false lead: running just that one file alongside workflows_test.py does
NOT reproduce the bug, because it does not itself import comfy.cli_args;
it was merely the first test *executed* after some other, earlier-collected
module had already done so during the collection phase. Collection for the
whole session happens before any test runs, so "first test to observe X in
sys.modules" and "module that imports X" are different things and must not
be conflated -- the collectreport hook was needed to tell them apart.)

Scoping to just this one module instead of all of tests-unit/assets_test
keeps this test fast (~6s vs ~30s) and, importantly, sidesteps a known,
unrelated confound: 4 tests elsewhere under tests-unit/assets_test fail on
this developer's checkout because of 339 real files already present under
models/ (proven unrelated to collection order: they pass on a clean
checkout with this same fix present). test_asset_management.py is not one
of those 4, so unlike the full directory, a plain `returncode == 0`
assertion on this scoped child run is NOT polluted by that confound --
verified empirically before writing this assertion (both by confirming the
4 known failures live in other files, and by confirming this scoped
combination reproduces 14 ERRORs, returncode != 0, when run against the
pre-fix code). The CUDA-probe string check is kept as a second, more
targeted signature of the specific bug, independent of whatever else might
someday change in assets_test.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUDA_PROBE_ERROR = "Torch not compiled with CUDA enabled"


def test_workflows_test_survives_hostile_collection_order():
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests-unit/assets_test/services/test_asset_management.py",
            "tests-unit/colab_studio_test/workflows_test.py",
            "-q",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    assert CUDA_PROBE_ERROR not in output, output
    assert result.returncode == 0, output
