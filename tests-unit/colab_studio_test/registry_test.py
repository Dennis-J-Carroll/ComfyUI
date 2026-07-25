import pytest
from colab_studio.registry import ModelSpec, PROFILES, resolve


def test_modelspec_target_name_defaults_to_basename():
    spec = ModelSpec(repo="a/b", filename="split/x.safetensors",
                     dest_subdir="checkpoints", size_gb=1.0)
    assert spec.target_name == "x.safetensors"


def test_modelspec_target_name_honours_rename():
    spec = ModelSpec(repo="diffusers/controlnet-canny-sdxl-1.0",
                     filename="diffusion_pytorch_model.fp16.safetensors",
                     dest_subdir="controlnet", size_gb=2.33,
                     dest_filename="controlnet-canny-sdxl.safetensors")
    assert spec.target_name == "controlnet-canny-sdxl.safetensors"


def test_sdxl_profile_uses_stabilityai_not_comfy_org():
    # Comfy-Org/stable-diffusion-xl-base-1.0 returns 401 -- it does not exist.
    specs = PROFILES["sdxl"]
    repos = {s.repo for s in specs}
    assert "stabilityai/stable-diffusion-xl-base-1.0" in repos
    assert not any(r.startswith("Comfy-Org/stable-diffusion-xl") for r in repos)


def test_flux_dev_is_single_all_in_one_checkpoint():
    specs = PROFILES["flux-dev"]
    assert len(specs) == 1
    assert specs[0].dest_subdir == "checkpoints"
    assert specs[0].size_gb > 15


def test_resolve_includes_upscaler_by_default():
    names = [s.target_name for s in resolve("sdxl")]
    assert "4x-UltraSharp.pth" in names


def test_resolve_excludes_controlnet_unless_requested():
    without = [s.dest_subdir for s in resolve("sdxl", controlnet=False)]
    assert "controlnet" not in without
    with_cn = [s.dest_subdir for s in resolve("sdxl", controlnet=True)]
    assert "controlnet" in with_cn


def test_resolve_rejects_unknown_profile():
    with pytest.raises(KeyError):
        resolve("not-a-profile")
