"""Compatibility record: the exact upstream ComfyUI revision this fork's
workflow graphs, node input names, and launch flags were validated against.

ComfyUI's node registry, `INPUT_TYPES()` keys, and default values are not a
stable API -- upstream renames inputs, changes defaults, and removes nodes
without a deprecation cycle. `colab_studio/workflows.py`'s graphs were
structurally validated (`execution.validate_prompt`, the exact code path
`POST /prompt` uses) against exactly `TESTED_REF`. Running against a
different revision means running against node contracts nobody here has
checked -- pinning is what makes "the structural-validation tests pass"
mean something more than "passed on whatever upstream commit happened to
be current the day this notebook was built."

Pure data plus one formatting helper. No I/O, no imports beyond stdlib.
"""
from __future__ import annotations

REPO_URL = "https://github.com/comfyanonymous/ComfyUI"
TESTED_REF = "806e092ed42772e4ce7abf44c97c50021cc4bd10"
TESTED_DATE = "2026-07-26"


def short_ref(ref: str = TESTED_REF, length: int = 12) -> str:
    """First `length` characters of a commit SHA, for compact printing."""
    return ref[:length]
