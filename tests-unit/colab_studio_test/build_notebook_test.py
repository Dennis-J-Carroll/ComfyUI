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
        if src.startswith("%%writefile"):
            continue          # inlined library files are written, not executed
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
