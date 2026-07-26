import os
import pytest
from colab_studio.registry import ModelSpec
from colab_studio import fetch


@pytest.fixture
def spec():
    return ModelSpec(repo="a/b", filename="sub/model.safetensors",
                     dest_subdir="checkpoints", size_gb=1.0)


@pytest.fixture
def renamed_spec():
    return ModelSpec(repo="diffusers/cn", filename="diffusion_pytorch_model.fp16.safetensors",
                     dest_subdir="controlnet", size_gb=2.33,
                     dest_filename="controlnet-canny-sdxl.safetensors")


def test_download_uses_local_dir_and_never_copies(monkeypatch, tmp_path, spec):
    calls = {}

    def fake_hub_download(**kwargs):
        calls.update(kwargs)
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    monkeypatch.setattr(fetch.shutil, "copy2",
                        lambda *a, **k: pytest.fail("must not copy: doubles disk use"))

    out = fetch.download(spec, str(tmp_path))
    assert "local_dir" in calls
    assert "resume_download" not in calls   # deprecated in huggingface_hub
    assert os.path.exists(out)


def test_download_renames_diffusers_layout_file(monkeypatch, tmp_path, renamed_spec):
    def fake_hub_download(**kwargs):
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    out = fetch.download(renamed_spec, str(tmp_path))
    assert os.path.basename(out) == "controlnet-canny-sdxl.safetensors"
    assert os.path.exists(out)


def test_download_skips_when_file_already_present(monkeypatch, tmp_path, spec):
    dest = tmp_path / "checkpoints" / "model.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_text("already here")

    monkeypatch.setattr(fetch, "hf_hub_download",
                        lambda **k: pytest.fail("should not re-download"))
    out = fetch.download(spec, str(tmp_path))
    assert out == str(dest)
    assert dest.read_text() == "already here"


def test_emit_receives_progress_messages(monkeypatch, tmp_path, spec):
    msgs = []

    def fake_hub_download(**kwargs):
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    fetch.download(spec, str(tmp_path), emit=msgs.append)
    assert any("model.safetensors" in m for m in msgs)


def test_download_all_returns_one_path_per_spec(monkeypatch, tmp_path, spec, renamed_spec):
    def fake_hub_download(**kwargs):
        target = os.path.join(kwargs["local_dir"], kwargs["filename"])
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        open(target, "a").close()
        return target

    monkeypatch.setattr(fetch, "hf_hub_download", fake_hub_download)
    out = fetch.download_all([spec, renamed_spec], str(tmp_path))
    assert len(out) == 2
