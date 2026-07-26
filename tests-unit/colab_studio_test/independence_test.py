"""Spec criterion 8 -- the notebook must not depend on local tooling."""
import json
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(REPO, "ComfyUI_Colab_Studio.ipynb")


@pytest.fixture(scope="module")
def blob():
    """Resolved from __file__, so a different cwd cannot silently skip the
    whole acceptance gate. Missing notebook is a failure, not a skip."""
    if not os.path.exists(NB):
        pytest.fail(f"{NB} missing -- run: python build_notebook.py")
    with open(NB) as fh:
        return json.dumps(json.load(fh))


def test_no_mcp_reference(blob):
    assert not re.search(r"\bmcp\b", blob, re.I)


def test_no_uv_or_uvx_dependency(blob):
    assert not re.search(r"\buvx?\s+(run|pip|git\+)", blob)


def test_no_localhost_websocket_proxy(blob):
    assert "mcpProxyToken" not in blob
    assert "mcpProxyPort" not in blob


def test_no_absolute_developer_paths(blob):
    """The original colab_setup.sh cp'd from /home/<user>/Desktop -- a path
    that does not exist on a Colab VM, which killed the script under set -e."""
    assert "/home/dennisjcarroll" not in blob
    assert not re.search(r"/Users/[a-z]+/", blob)


def test_no_clone_of_this_repo_for_helpers(blob):
    assert "git clone" in blob                     # ComfyUI itself, fine
    assert "Desktop/ComfyUI" not in blob
