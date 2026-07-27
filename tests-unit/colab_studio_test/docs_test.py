"""Phase 0 item 0.f -- docs/COLAB_STUDIO_TEST_PLAN.md must reflect reality:
PR #1 merged into master (merge commit 6f572b1c), not the stale
feat/colab-studio-unmerged claim it shipped with."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC = os.path.join(REPO, "docs", "COLAB_STUDIO_TEST_PLAN.md")


def _text():
    with open(DOC) as fh:
        return fh.read()


def test_doc_exists():
    assert os.path.exists(DOC)


def test_doc_no_longer_claims_branch_unmerged():
    text = _text()
    assert "not merged" not in text.lower()


def test_doc_records_merged_state():
    text = _text()
    assert "6f572b1c" in text
    assert "merged" in text.lower()
