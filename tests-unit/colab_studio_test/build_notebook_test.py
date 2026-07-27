import json
import os
import re
import pytest

import build_notebook
from colab_studio import compat

COMMITTED_NB = os.path.join(build_notebook.HERE, "ComfyUI_Colab_Studio.ipynb")


@pytest.fixture(scope="module")
def nb(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb") / "ComfyUI_Colab_Studio.ipynb"
    build_notebook.build(str(out))
    return json.loads(out.read_text())


def sources(nb):
    return ["".join(c["source"]) for c in nb["cells"]]


def runnable_sources(nb):
    """Cells the kernel actually executes -- %%writefile cells are data."""
    return [s for s in sources(nb) if not s.startswith("%%writefile")]


def cell_titled(nb, prefix):
    """The one runnable cell whose #@title starts with `prefix`."""
    hits = [s for s in runnable_sources(nb) if f"#@title {prefix}" in s]
    assert len(hits) == 1, f"expected exactly one {prefix!r} cell, got {len(hits)}"
    return hits[0]


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
    for src in runnable_sources(nb):
        assert "!python3 main.py" not in src
        assert "!python main.py" not in src


def test_no_cell_opens_a_file_picker_during_run_all(nb):
    """files.upload() blocks the kernel on a picker. Ungated, it stalls
    'Run all' and makes every cell below unreachable -- the same class of bug
    as `!python3 main.py`, just further down the notebook."""
    gated = [s for s in runnable_sources(nb) if "files.upload()" in s]
    for src in gated:
        assert "run_this" in src, (
            "files.upload() must sit behind a run_this gate:\n" + src[:400])
        # The gate has to be a form field, or the user cannot flip it.
        assert re.search(r'run_this = False\s+#@param \{type:"boolean"\}', src)


def test_run_all_reaches_the_cells_after_the_upload_cell(nb):
    """Regression guard for cell ordering: the log and ops cells must come
    after 8b and must not themselves be gated."""
    titles = [s for s in runnable_sources(nb)]
    idx = {t: i for i, s in enumerate(titles)
           for t in re.findall(r"#@title (\S+)", s)}
    assert idx["8b."] < idx["9."] < idx["10."]


def test_library_modules_are_inlined(nb):
    blob = "\n".join(sources(nb))
    for mod in ("registry.py", "advice.py", "workflows.py",
                "fetch.py", "client.py", "launch.py"):
        assert f"%%writefile colab_studio/{mod}" in blob


def test_inlined_source_matches_repo_source(nb):
    """Single source of truth: shipped code == tested code, in full.

    A substring check would pass while the rest of a module had drifted.
    """
    bodies = {}
    for src in sources(nb):
        if not src.startswith("%%writefile"):
            continue
        header, _, body = src.partition("\n")
        bodies[os.path.basename(header.strip())] = body

    assert set(bodies) == set(build_notebook.MODULES)
    for mod in build_notebook.MODULES:
        path = os.path.join(build_notebook.HERE, "colab_studio", mod)
        with open(path) as fh:
            real = fh.read()
        # _code() strips the cell, so compare ignoring trailing newlines only.
        assert bodies[mod].rstrip("\n") == real.rstrip("\n"), (
            f"inlined {mod} differs from {path}")


def test_committed_notebook_matches_the_generator(nb):
    """Drift guard: the notebook is GENERATED and must never be hand-edited.

    Without this, editing build_notebook.py or colab_studio/*.py and
    forgetting to regenerate ships a stale notebook with a green suite.
    """
    assert os.path.exists(COMMITTED_NB), f"{COMMITTED_NB} is not committed"
    with open(COMMITTED_NB) as fh:
        committed = json.load(fh)
    assert committed == nb, (
        "ComfyUI_Colab_Studio.ipynb is stale -- run: python build_notebook.py")


def test_tunnel_starts_after_readiness_check(nb):
    """Ordering guard: tunnel before wait_ready prints a URL that 502s.

    Scoped to the launch cell. Against the whole notebook this passed
    vacuously -- `wait_ready` matched inlined client.py and `start_tunnel`
    matched inlined launch.py, so MODULES order alone satisfied it.
    """
    src = cell_titled(nb, "7.")
    assert src.index("wait_ready(") < src.index("start_tunnel("), \
        "tunnel is started before the readiness check"
    # Stronger: the call must sit inside the `if CLIENT.wait_ready(...)` block.
    calls = [ln for ln in src.splitlines()
             if "start_tunnel(" in ln and "import" not in ln]
    assert calls, "launch cell never calls start_tunnel"
    for line in calls:
        assert line.startswith(" "), \
            f"start_tunnel is not guarded by the readiness branch: {line!r}"


def test_config_cell_exposes_expected_form_fields(nb):
    src = cell_titled(nb, "1.")
    for field in ("IMAGE_MODEL", "PERSIST", "CONTROLNET", "USE_UPSCALER"):
        assert f"{field} =" in src


def test_video_mode_is_gone(nb):
    """MODE=video cloned three custom-node repos then generated SDXL images
    anyway. Video is out of scope; Wan2.2_Colab_Pipeline.ipynb keeps it."""
    blob = "\n".join(runnable_sources(nb))
    assert not re.search(r"\bMODE\b", blob)
    assert not re.search(r"\bVIDEO_MODEL\b", blob)
    for repo in ("WanVideoWrapper", "ComfyUI-GGUF", "VideoHelperSuite"):
        assert repo not in blob


def test_no_cell_claims_sidebar_workflows(nb):
    """Cell 6 writes API format only; the UI-format writer was dropped."""
    blob = "\n".join(runnable_sources(nb))
    assert "user/default/workflows" not in blob
    assert "WF_DIR" not in blob


def _call_args(src, start):
    """Text between the paren at `start` and its match."""
    depth, i = 0, start
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return src[start + 1:i]
        i += 1
    raise AssertionError("unbalanced parentheses in cell source")


def test_generate_cells_pass_the_profile_to_the_builders(nb):
    """C3: upscale/img2img/controlnet_canny are profile-sensitive; a call site
    that omits profile= silently builds an SDXL graph for a Flux checkpoint."""
    seen = 0
    for title in ("6.", "8.", "8b."):
        src = cell_titled(nb, title)
        for builder in ("upscale", "img2img", "controlnet_canny"):
            for m in re.finditer(rf"workflows\.{builder}\(", src):
                args = _call_args(src, m.end() - 1)
                assert "profile=" in args, (
                    f"cell {title} calls workflows.{builder}() without "
                    f"profile=: {args!r}")
                seen += 1
    assert seen >= 4, f"expected several profile-sensitive calls, found {seen}"


def test_disk_guard_measures_the_models_destination(nb):
    """PERSIST=everything symlinks models/ to Drive, so /content's free space
    is the wrong filesystem to hand recommend()."""
    src = cell_titled(nb, "5.")
    assert "disk_usage(MODELS_DIR)" in src
    assert "recommend(VRAM_GB, MODELS_FREE_GB)" in src


def test_explicit_model_override_warns(nb):
    src = cell_titled(nb, "5.")
    assert 'IMAGE_MODEL != "auto" and IMAGE_MODEL != ADVICE.profile' in src
    assert "OVERRIDE" in src


def test_retunnel_stops_the_previous_tunnel(nb):
    src = cell_titled(nb, "10.")
    assert src.index("stop_tunnel()") < src.index("start_tunnel(PORT")


def test_handbook_cell_present_with_troubleshooting(nb):
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    blob = "\n".join("".join(c["source"]) for c in md).lower()
    assert "out of memory" in blob or "oom" in blob
    assert "disk" in blob


# --- Phase 0 items 0.b / 0.c / 0.d --------------------------------------

def test_no_wan_reference_anywhere(nb):
    """0.c: Wan2.2_Colab_Pipeline.ipynb has never been committed to any
    branch -- it's a dead end for every user but this one machine."""
    blob = json.dumps(nb)
    assert "Wan2.2_Colab_Pipeline" not in blob


def test_pinned_tested_ref_present(nb):
    """0.b: the install cell must actually use the exact revision the
    graphs were validated against. Sourced from colab_studio/compat.py, not
    hardcoded here, so this can't drift from the single source of truth.

    Scoped to cell 3's own source, not the whole notebook -- compat.py's
    module body is separately inlined via %%writefile and would otherwise
    satisfy a whole-notebook substring check whether or not cell 3's
    install logic wires the ref in at all.
    """
    src = cell_titled(nb, "3.")
    assert compat.TESTED_REF in src


def test_latest_ref_prints_unsupported_warning(nb):
    """0.b: choosing COMFY_REF='latest' must warn, clearly, that the graphs
    were only validated against the pinned ref."""
    src = cell_titled(nb, "3.")
    idx_branch = src.index('COMFY_REF == "pinned"')
    idx_warning = src.index("UNSUPPORTED")
    assert idx_branch < idx_warning, \
        "UNSUPPORTED warning must live in the COMFY_REF branch, not before it"


def test_open_public_ui_defaults_false(nb):
    """0.d: the public tunnel must be opt-in, off by default."""
    src = cell_titled(nb, "1.")
    assert re.search(r'OPEN_PUBLIC_UI = False\s+#@param \{type:"boolean"\}', src)


def test_start_tunnel_guarded_by_open_public_ui(nb):
    """0.d: Run All with defaults must never reach start_tunnel. Every
    non-import call to start_tunnel in the launch cell must be indented
    strictly deeper than the `if OPEN_PUBLIC_UI` line that guards it, and
    must appear after it."""
    src = cell_titled(nb, "7.")
    lines = src.splitlines()

    guard_indent = None
    guard_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s*if OPEN_PUBLIC_UI\b", line):
            guard_idx = i
            guard_indent = len(line) - len(line.lstrip(" "))
            break
    assert guard_idx is not None, "launch cell has no `if OPEN_PUBLIC_UI` guard"

    calls = [(i, line) for i, line in enumerate(lines)
             if "start_tunnel(" in line and "import" not in line]
    assert calls, "launch cell never calls start_tunnel"
    for i, line in calls:
        assert i > guard_idx, f"start_tunnel called before the OPEN_PUBLIC_UI guard: {line!r}"
        call_indent = len(line) - len(line.lstrip(" "))
        assert call_indent > guard_indent, \
            f"start_tunnel is not nested inside the OPEN_PUBLIC_UI guard: {line!r}"


def test_launch_cell_reuses_supervisor_across_reruns(nb):
    """0.d: RuntimeSupervisor's single-server guard (self._server) is
    per-instance, not module state (unlike _TUNNEL). An unconditional
    `SUPERVISOR = RuntimeSupervisor()` at the top of the launch cell would
    forget the server the previous run started on every re-run, spawn a
    second process that fails to bind the port, and report readiness off
    the first, now-orphaned one -- silently defeating the guard this cell
    exists to apply. The cell must only construct a fresh instance when
    none already exists in the kernel's namespace."""
    src = cell_titled(nb, "7.")
    assert not re.search(r"^SUPERVISOR = RuntimeSupervisor\(\)", src, re.M), (
        "launch cell unconditionally rebinds SUPERVISOR on every run -- "
        "a re-run loses the single-server guard")
    assert "SUPERVISOR = RuntimeSupervisor()" in src, \
        "launch cell never constructs a SUPERVISOR at all"


def test_wait_ready_precedes_start_tunnel_within_launch_cell(nb):
    """0.d: tunnel must only start after wait_ready() succeeds. Scoped to
    the launch cell's own source so a coincidental match against inlined
    library code (wait_ready lives in client.py, start_tunnel in
    launch.py) can't satisfy this vacuously."""
    src = cell_titled(nb, "7.")
    assert src.index("wait_ready(") < src.index("start_tunnel(")
